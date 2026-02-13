# ============================================================
# Evaluate Activity Recognition with Imputed Latents
# Compares: real vs diffusion-imputed vs zero-filled (no imputation)
# Includes: Accuracy, Macro F1, Per-Class Analysis
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import Counter

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion import create_sensor_diffusion_model
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_best.pt"
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion")
CLASSIFIER_CHECKPOINT = "checkpoints/activity_classifier_robust/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

DDIM_STEPS = 50
BATCH_SIZE = 32

OUTPUT_DIR = Path("outputs/activity_imputation_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Missing scenarios
SCENARIOS = {
    "all_real": {"missing": []},
    "missing_phone_acc": {"missing": ["phone_acc"]},
    "missing_watch_acc": {"missing": ["watch_acc"]},
    "missing_glasses_acc": {"missing": ["glasses_acc"]},
    "missing_phone_all": {"missing": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"]},
    "missing_watch_all": {"missing": ["watch_acc", "watch_gyro"]},
}

# Imputation methods to compare
IMPUTATION_METHODS = ["diffusion", "zeros", "mean"]


# ============================================================
# METRICS
# ============================================================
def compute_metrics(all_preds, all_labels, n_classes):
    """Compute accuracy, macro F1, per-class metrics."""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    accuracy = (all_preds == all_labels).mean()

    # Per-class precision, recall, F1
    per_class = {}
    f1_scores = []

    for c in range(n_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        support = (all_labels == c).sum()

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        if support > 0:  # Only include classes that exist in test set
            f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
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
# DDIM SAMPLER
# ============================================================
@torch.no_grad()
def ddim_sample_batch(model, target_modality, shape, conditions, alpha_bar, T, ddim_steps=50):
    B = shape[0]
    device = next(model.parameters()).device
    z = torch.randn(shape, device=device)
    alpha_bar = alpha_bar.to(device)

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
    print(f"\n{'='*70}")
    print("Evaluating Activity Recognition: Diffusion vs Zero-Fill vs Mean-Fill")
    print(f"DDIM Steps: {DDIM_STEPS}")
    print(f"{'='*70}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    n_classes = test_dataset.n_classes
    idx_to_label = test_dataset.idx_to_label
    print(f"Test samples: {len(test_dataset)}, Classes: {n_classes}")

    # Class distribution
    print("\nClass distribution in test set:")
    all_test_labels = []
    for batch in test_loader:
        all_test_labels.extend(batch["label"].tolist())
    label_counts = Counter(all_test_labels)
    top_10 = label_counts.most_common(10)
    print(f"  Top 10 classes: {[(idx_to_label[c], cnt) for c, cnt in top_10]}")
    print(f"  Total classes with samples: {len(label_counts)}")

    # Load VAE
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load diffusion
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

    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T, schedule_type)

    # Compute mean latents for mean-fill baseline
    print("Computing mean latents for baseline...")
    mean_latents = {}
    for name in SENSOR_NAMES:
        mean_latents[name] = norm_stats[name]["mean"].to(DEVICE) * 0  # Zero in normalized space

    # Load classifier
    print("Loading classifier...")
    cls_ckpt = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE)
    classifier = create_activity_classifier(
        model_type=cls_ckpt["model_type"],
        n_classes=cls_ckpt["n_classes"],
    ).to(DEVICE)
    classifier.load_state_dict(cls_ckpt["model_state"])
    classifier.eval()

    # Results storage
    all_results = {}

    # Evaluate each scenario x method
    for scenario_name, scenario_cfg in SCENARIOS.items():
        missing = scenario_cfg["missing"]

        if not missing:
            # Baseline: all real
            print(f"\n{'='*70}")
            print(f"Scenario: {scenario_name} (baseline)")
            print(f"{'='*70}")

            all_preds = []
            all_labels = []

            for batch in tqdm(test_loader, desc="all_real", leave=False):
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels = batch["label"].to(DEVICE)

                with torch.no_grad():
                    outputs = vae(sensor_data)
                    latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}
                    logits = classifier(latents, SENSOR_NAMES)
                    preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            metrics = compute_metrics(all_preds, all_labels, n_classes)
            all_results["all_real"] = {"all": metrics}
            print(f"  Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")

        else:
            print(f"\n{'='*70}")
            print(f"Scenario: {scenario_name}")
            print(f"Missing: {missing}")
            print(f"{'='*70}")

            all_results[scenario_name] = {}

            for method in IMPUTATION_METHODS:
                all_preds = []
                all_labels = []

                for batch in tqdm(test_loader, desc=f"{method}", leave=False):
                    sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                    labels = batch["label"].to(DEVICE)
                    B = labels.size(0)

                    with torch.no_grad():
                        outputs = vae(sensor_data)
                        latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

                    # Normalize for diffusion
                    latents_norm = {}
                    for name in SENSOR_NAMES:
                        mean = norm_stats[name]["mean"].to(DEVICE)
                        std = norm_stats[name]["std"].to(DEVICE)
                        latents_norm[name] = (latents[name] - mean) / std

                    # Impute missing
                    final_latents = {}
                    for name in SENSOR_NAMES:
                        if name in missing:
                            if method == "diffusion":
                                conditions = {
                                    k: (latents_norm[k] if k not in missing else None)
                                    for k in SENSOR_NAMES
                                }
                                imputed_norm = ddim_sample_batch(
                                    model=diffusion,
                                    target_modality=name,
                                    shape=latents_norm[name].shape,
                                    conditions=conditions,
                                    alpha_bar=sched["alpha_bar"],
                                    T=T,
                                    ddim_steps=DDIM_STEPS,
                                )
                                mean = norm_stats[name]["mean"].to(DEVICE)
                                std = norm_stats[name]["std"].to(DEVICE)
                                final_latents[name] = imputed_norm * std + mean

                            elif method == "zeros":
                                # Fill with zeros (same shape as real latent)
                                final_latents[name] = torch.zeros_like(latents[name])

                            elif method == "mean":
                                # Fill with mean (expand to full shape)
                                mean = norm_stats[name]["mean"].to(DEVICE)  # (1, 8, 1)
                                T_shared = latents[name].shape[2]
                                final_latents[name] = mean.expand(B, -1, T_shared)
                        else:
                            final_latents[name] = latents[name]

                    with torch.no_grad():
                        logits = classifier(final_latents, SENSOR_NAMES)
                        preds = logits.argmax(dim=1)

                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

                metrics = compute_metrics(all_preds, all_labels, n_classes)
                all_results[scenario_name][method] = metrics
                print(f"  {method:10s}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY - All Scenarios")
    print(f"{'='*70}")

    baseline = all_results["all_real"]["all"]
    print(f"\n{'Scenario':<25} {'Method':<12} {'Accuracy':<10} {'Drop':<10} {'Macro F1':<10}")
    print("-" * 70)
    print(f"{'all_real':<25} {'-':<12} {baseline['accuracy']:.4f}     {'-':<10} {baseline['macro_f1']:.4f}")

    for scenario_name, methods in all_results.items():
        if scenario_name == "all_real":
            continue
        for method, metrics in methods.items():
            drop = baseline['accuracy'] - metrics['accuracy']
            print(f"{scenario_name:<25} {method:<12} {metrics['accuracy']:.4f}     {drop:+.4f}     {metrics['macro_f1']:.4f}")

    # ============================================================
    # DIFFUSION vs BASELINES
    # ============================================================
    print(f"\n{'='*70}")
    print("DIFFUSION IMPROVEMENT over baselines")
    print(f"{'='*70}")

    for scenario_name, methods in all_results.items():
        if scenario_name == "all_real" or "diffusion" not in methods:
            continue

        diff_acc = methods["diffusion"]["accuracy"]
        zero_acc = methods["zeros"]["accuracy"]
        mean_acc = methods["mean"]["accuracy"]

        diff_vs_zero = diff_acc - zero_acc
        diff_vs_mean = diff_acc - mean_acc

        print(f"{scenario_name}:")
        print(f"  Diffusion vs Zeros: {diff_vs_zero:+.4f}")
        print(f"  Diffusion vs Mean:  {diff_vs_mean:+.4f}")

    # ============================================================
    # PER-CLASS ANALYSIS (for worst scenario)
    # ============================================================
    print(f"\n{'='*70}")
    print("PER-CLASS ANALYSIS (missing_watch_all with diffusion)")
    print(f"{'='*70}")

    if "missing_watch_all" in all_results and "diffusion" in all_results["missing_watch_all"]:
        pc = all_results["missing_watch_all"]["diffusion"]["per_class"]
        # Sort by support
        sorted_classes = sorted(pc.items(), key=lambda x: x[1]["support"], reverse=True)

        print(f"\n{'Class':<8} {'Support':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
        print("-" * 55)
        for c, m in sorted_classes[:15]:  # Top 15 by support
            if m["support"] > 0:
                print(f"{idx_to_label[c]:<8} {m['support']:<10} {m['precision']:.4f}       {m['recall']:.4f}     {m['f1']:.4f}")

    # Save results
    with open(OUTPUT_DIR / "detailed_results.txt", "w") as f:
        f.write("Activity Recognition Evaluation\n")
        f.write(f"DDIM Steps: {DDIM_STEPS}\n\n")

        f.write("="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"{'Scenario':<25} {'Method':<12} {'Accuracy':<10} {'Macro F1':<10}\n")
        f.write("-"*60 + "\n")

        for scenario_name, methods in all_results.items():
            if scenario_name == "all_real":
                f.write(f"{scenario_name:<25} {'-':<12} {methods['all']['accuracy']:.4f}     {methods['all']['macro_f1']:.4f}\n")
            else:
                for method, metrics in methods.items():
                    f.write(f"{scenario_name:<25} {method:<12} {metrics['accuracy']:.4f}     {metrics['macro_f1']:.4f}\n")

    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
