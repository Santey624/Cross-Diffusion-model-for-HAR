# ============================================================
# Evaluate Activity Recognition with Imputed Latents
# Compares: real latents vs diffusion-imputed latents
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion import create_sensor_diffusion_model
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_epoch_047.pt"
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion")
# Switch between regular and robust classifier
# CLASSIFIER_CHECKPOINT = "checkpoints/activity_classifier/best_model.pt"
CLASSIFIER_CHECKPOINT = "checkpoints/activity_classifier_robust/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

DDIM_STEPS = 50  # Fewer steps for faster eval
BATCH_SIZE = 32

OUTPUT_DIR = Path("outputs/activity_imputation_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Missing scenarios to test
SCENARIOS = {
    "all_real": {"missing": []},  # Baseline: all sensors available
    "missing_phone_acc": {"missing": ["phone_acc"]},
    "missing_watch_acc": {"missing": ["watch_acc"]},
    "missing_glasses_acc": {"missing": ["glasses_acc"]},
    "missing_phone_all": {"missing": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"]},
    "missing_watch_all": {"missing": ["watch_acc", "watch_gyro"]},
}


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
# FAST DDIM SAMPLER
# ============================================================
@torch.no_grad()
def ddim_sample_batch(model, target_modality, shape, conditions, alpha_bar, T, ddim_steps=50):
    B = shape[0]
    device = next(model.parameters()).device
    z = torch.randn(shape, device=device)
    alpha_bar = alpha_bar.to(device)

    # Quadratic spacing
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noise_pred = model(
            target_modality=target_modality,
            z_t=z,
            t=t_batch,
            conditions=conditions,
        )

        ab_now = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        pred_x0 = (z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

        dir_zt = torch.sqrt(1 - ab_next) * noise_pred
        z = torch.sqrt(ab_next) * pred_x0 + dir_zt

    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Evaluating Activity Recognition with Imputed Latents")
    print(f"DDIM Steps: {DDIM_STEPS}")
    print(f"{'='*60}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.n_classes}")

    # Load VAE
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load diffusion model
    print("Loading diffusion model...")
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T = diff_ckpt["T"]
    schedule_type = diff_ckpt["schedule"]
    cfg = diff_ckpt["config"]

    diffusion = create_sensor_diffusion_model(
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_conv_blocks=cfg["num_conv_blocks"],
        num_attn_blocks=cfg["num_attn_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()

    # Load latent normalization stats
    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)

    sched = make_schedule(T, schedule_type)

    # Load classifier
    print("Loading activity classifier...")
    cls_ckpt = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE)
    classifier = create_activity_classifier(
        model_type=cls_ckpt["model_type"],
        n_classes=cls_ckpt["n_classes"],
    ).to(DEVICE)
    classifier.load_state_dict(cls_ckpt["model_state"])
    classifier.eval()

    print(f"Classifier trained accuracy: {cls_ckpt['acc']:.4f}")

    # Evaluate each scenario
    all_results = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        missing = scenario_cfg["missing"]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"Missing: {missing if missing else 'None (baseline)'}")
        print(f"{'='*60}")

        correct = 0
        total = 0

        for batch in tqdm(test_loader, desc=scenario_name, leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            # Encode all sensors with VAE
            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            # Normalize latents for diffusion
            latents_norm = {}
            for name in SENSOR_NAMES:
                mean = norm_stats[name]["mean"].to(DEVICE)
                std = norm_stats[name]["std"].to(DEVICE)
                latents_norm[name] = (latents[name] - mean) / std

            # Impute missing sensors
            final_latents = {}
            for name in SENSOR_NAMES:
                if name in missing:
                    # Build conditions (all non-missing sensors)
                    conditions = {
                        k: (latents_norm[k] if k not in missing else None)
                        for k in SENSOR_NAMES
                    }

                    # Sample with diffusion
                    imputed_norm = ddim_sample_batch(
                        model=diffusion,
                        target_modality=name,
                        shape=latents_norm[name].shape,
                        conditions=conditions,
                        alpha_bar=sched["alpha_bar"],
                        T=T,
                        ddim_steps=DDIM_STEPS,
                    )

                    # Denormalize
                    mean = norm_stats[name]["mean"].to(DEVICE)
                    std = norm_stats[name]["std"].to(DEVICE)
                    final_latents[name] = imputed_norm * std + mean
                else:
                    # Use real latent
                    final_latents[name] = latents[name]

            # Classify
            with torch.no_grad():
                logits = classifier(final_latents, SENSOR_NAMES)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += B

        accuracy = correct / total
        all_results[scenario_name] = accuracy
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Activity Recognition Accuracy")
    print(f"{'='*60}")

    baseline_acc = all_results.get("all_real", 0)
    for scenario, acc in all_results.items():
        drop = baseline_acc - acc if scenario != "all_real" else 0
        print(f"  {scenario:25s}: {acc:.4f}  (drop: {drop:+.4f})")

    # Save
    with open(OUTPUT_DIR / "results.txt", "w") as f:
        f.write("Activity Recognition with Imputed Latents\n")
        f.write(f"DDIM Steps: {DDIM_STEPS}\n\n")
        for scenario, acc in all_results.items():
            f.write(f"{scenario}: {acc:.4f}\n")

    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
