# ============================================================
# Evaluate Behavioral Activity Recognition (54-55 classes)
# BLHO + BBH combined, with Diffusion V2 imputation
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion_v2 import create_sensor_diffusion_v2
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt"
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion_v2_pretrain")
CLASSIFIER_CHECKPOINT = "checkpoints/activity_classifier_robust/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

# BLHO + BBH only (no state)
DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
}

DDIM_STEPS = 50
BATCH_SIZE = 32


# ============================================================
# COSINE SCHEDULE
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    return betas.float()


def make_schedule(T, schedule_type):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"alpha_bar": alpha_bar}


# ============================================================
# DDIM SAMPLING
# ============================================================
@torch.no_grad()
def ddim_sample_v2(model, stacked_latents, observed_mask, alpha_bar, T, ddim_steps=50):
    B, K, D, T_len = stacked_latents.shape
    device = stacked_latents.device
    missing_mask = 1.0 - observed_mask

    z = stacked_latents.clone()
    noise_init = torch.randn_like(stacked_latents)
    z = observed_mask[:, :, None, None] * z + missing_mask[:, :, None, None] * noise_init

    alpha_bar = alpha_bar.to(device)
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy_input = observed_mask[:, :, None, None] * stacked_latents + \
                      missing_mask[:, :, None, None] * z

        noise_pred = model(noisy_input, t_batch, observed_mask)

        ab_now = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        pred_x0 = (z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

        dir_zt = torch.sqrt(1 - ab_next) * noise_pred
        z_new = torch.sqrt(ab_next) * pred_x0 + dir_zt

        z = observed_mask[:, :, None, None] * stacked_latents + \
            missing_mask[:, :, None, None] * z_new

    return z


def ddim_sample_v2_guided(model, classifier, stacked_latents, observed_mask,
                           alpha_bar, T, sensor_names, norm_stats,
                           ddim_steps=50, guidance_scale=1.0):
    """DDIM sampling with classifier guidance via entropy minimization."""
    B, K, D, T_len = stacked_latents.shape
    device = stacked_latents.device
    missing_mask = 1.0 - observed_mask

    z = stacked_latents.clone()
    noise_init = torch.randn_like(stacked_latents)
    z = observed_mask[:, :, None, None] * z + missing_mask[:, :, None, None] * noise_init

    alpha_bar = alpha_bar.to(device)
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy_input = observed_mask[:, :, None, None] * stacked_latents + \
                      missing_mask[:, :, None, None] * z

        with torch.no_grad():
            noise_pred = model(noisy_input, t_batch, observed_mask)

        ab_now = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        pred_x0 = (z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

        # Classifier guidance: only in later steps when pred_x0 is reliable
        if guidance_scale > 0 and i >= ddim_steps // 3:
            pred_x0_g = pred_x0.detach().requires_grad_(True)
            latents_dict = {}
            for ki, name in enumerate(sensor_names):
                mean = norm_stats[name]["mean"].to(device)
                std = norm_stats[name]["std"].to(device)
                latents_dict[name] = pred_x0_g[:, ki] * std + mean
            logits = classifier(latents_dict, sensor_names)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            entropy.backward()
            grad = pred_x0_g.grad.clone()
            grad_norm = grad.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            grad = grad / grad_norm
            grad = grad * missing_mask[:, :, None, None]
            pred_x0 = (pred_x0 - guidance_scale * grad).clamp(-5.0, 5.0)

        dir_zt = torch.sqrt(1 - ab_next) * noise_pred
        z_new = torch.sqrt(ab_next) * pred_x0 + dir_zt

        z = observed_mask[:, :, None, None] * stacked_latents + \
            missing_mask[:, :, None, None] * z_new

    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*70}")
    print("BEHAVIORAL ACTIVITY RECOGNITION EVALUATION")
    print("BLHO + BBH (54-55 classes)")
    print(f"{'='*70}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load test dataset (BLHO + BBH)
    print("Loading Behavioral test dataset (BLHO + BBH)...")
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    n_classes = test_dataset.n_classes
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}, Classes: {n_classes}")

    # Load VAE
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load Diffusion V2
    print("Loading Diffusion V2...")
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T_diff = diff_ckpt["T"]
    cfg = diff_ckpt["config"]
    diffusion = create_sensor_diffusion_v2(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()
    print(f"Diffusion: epoch {diff_ckpt.get('epoch', '?')}, loss={diff_ckpt.get('loss', '?'):.4f}")

    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T_diff, diff_ckpt["schedule"])

    # Load classifier
    print("\nLoading activity classifier...")
    if not Path(CLASSIFIER_CHECKPOINT).exists():
        print(f"Classifier not found at {CLASSIFIER_CHECKPOINT}")
        print("Please train it first!")
        return

    classifier_ckpt = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE)
    model_type = classifier_ckpt.get("model_type", "mlp")
    classifier = create_activity_classifier(
        model_type=model_type,
        n_classes=n_classes,
    ).to(DEVICE)
    classifier.load_state_dict(classifier_ckpt["model_state"])
    classifier.eval()
    print(f"Classifier: {n_classes} classes")

    # Compute global mean latents for mean-fill baseline (from test set)
    print("\nComputing mean latents for mean-fill baseline...")
    mean_latents_global = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in test_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                mean_latents_global[k].append(outputs[k]["mu"].cpu())
    mean_latents_global = {k: torch.cat(v, dim=0).mean(dim=0, keepdim=True).to(DEVICE)
                           for k, v in mean_latents_global.items()}

    GUIDANCE_SCALE = 2.0

    # Evaluation scenarios: (missing_sensors, mode)
    # mode: "real" | "diff" | "guided" | "mean"
    scenarios = {
        "all_real":               ([], "real"),
        "phone_acc+diff":         (["phone_acc"], "diff"),
        "phone_acc+guided":       (["phone_acc"], "guided"),
        "phone_acc+mean":         (["phone_acc"], "mean"),
        "watch_acc+diff":         (["watch_acc"], "diff"),
        "watch_acc+guided":       (["watch_acc"], "guided"),
        "watch_acc+mean":         (["watch_acc"], "mean"),
        "glasses_acc+diff":       (["glasses_acc"], "diff"),
        "glasses_acc+guided":     (["glasses_acc"], "guided"),
        "glasses_acc+mean":       (["glasses_acc"], "mean"),
        "phone_all+diff":         (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "diff"),
        "phone_all+guided":       (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "guided"),
        "phone_all+mean":         (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "mean"),
        "watch_all+diff":         (["watch_acc", "watch_gyro"], "diff"),
        "watch_all+guided":       (["watch_acc", "watch_gyro"], "guided"),
        "watch_all+mean":         (["watch_acc", "watch_gyro"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name} | Mode: {mode}")
        print(f"Missing: {missing_sensors if missing_sensors else 'None'}")
        print(f"{'='*70}")

        all_preds = []
        all_labels = []

        for batch in tqdm(test_loader, desc=f"Eval {scenario_name}"):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            # VAE encode
            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            if mode == "real":
                final_latents = latents
            elif mode in ("diff", "guided"):
                latents_norm = {}
                for name in SENSOR_NAMES:
                    mean = norm_stats[name]["mean"].to(DEVICE)
                    std = norm_stats[name]["std"].to(DEVICE)
                    latents_norm[name] = (latents[name] - mean) / std

                stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)

                observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for sensor in missing_sensors:
                    observed_mask[:, SENSOR_NAMES.index(sensor)] = 0.0

                if mode == "diff":
                    imputed = ddim_sample_v2(diffusion, stacked, observed_mask,
                                             sched["alpha_bar"], T_diff, DDIM_STEPS)
                else:
                    imputed = ddim_sample_v2_guided(diffusion, classifier, stacked,
                                                    observed_mask, sched["alpha_bar"],
                                                    T_diff, SENSOR_NAMES, norm_stats,
                                                    DDIM_STEPS, GUIDANCE_SCALE)

                final_latents = {}
                for i, name in enumerate(SENSOR_NAMES):
                    mean = norm_stats[name]["mean"].to(DEVICE)
                    std = norm_stats[name]["std"].to(DEVICE)
                    final_latents[name] = imputed[:, i] * std + mean
            else:
                # Mean-fill baseline
                final_latents = dict(latents)
                for name in missing_sensors:
                    final_latents[name] = mean_latents_global[name].expand(B, -1, -1)

            # Classify
            with torch.no_grad():
                logits = classifier(final_latents, SENSOR_NAMES)
                preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        results[scenario_name] = {"accuracy": acc, "f1_macro": f1_macro}

        print(f"\nAccuracy: {acc:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"BEHAVIORAL ACTIVITIES - SUMMARY (guidance_scale={GUIDANCE_SCALE})")
    print(f"{'='*70}")
    print(f"\n{'Scenario':<25} {'Accuracy':<12} {'Macro F1':<12}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
