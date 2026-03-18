# ============================================================
# Evaluate Imputation Quality: Diffusion vs Mean vs Zero
# Measures MSE and Cosine Similarity in latent space
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion_v2 import create_sensor_diffusion_v2
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt"
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion_v2_pretrain")
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh":  "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

DDIM_STEPS = 50
BATCH_SIZE = 32


def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-6, max=0.999).float()


def make_schedule(T, schedule_type):
    betas = cosine_beta_schedule(T) if schedule_type == "cosine" else torch.linspace(1e-4, 0.02, T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return {"alpha_bar": alpha_bar}


@torch.no_grad()
def ddim_sample_v2(model, stacked_latents, observed_mask, alpha_bar, T, ddim_steps=50):
    B, K, D, T_len = stacked_latents.shape
    device = stacked_latents.device
    missing_mask = 1.0 - observed_mask

    z = stacked_latents.clone()
    z = observed_mask[:, :, None, None] * z + missing_mask[:, :, None, None] * torch.randn_like(z)

    alpha_bar = alpha_bar.to(device)
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy_input = observed_mask[:, :, None, None] * stacked_latents + \
                      missing_mask[:, :, None, None] * z
        noise_pred = model(noisy_input, t_batch, observed_mask)

        ab_now, ab_next = alpha_bar[t_now], alpha_bar[t_next]
        pred_x0 = torch.clamp((z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now), -5, 5)
        z_new = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred

        z = observed_mask[:, :, None, None] * stacked_latents + \
            missing_mask[:, :, None, None] * z_new

    return z


def main():
    print(f"\n{'='*70}")
    print("IMPUTATION QUALITY EVALUATION")
    print("Metrics: MSE and Cosine Similarity (real vs imputed latents)")
    print(f"{'='*70}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Load VAE
    vae = SensorMultiModalVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE)["model_state"])
    vae.eval()

    # Load Diffusion
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T_diff = diff_ckpt["T"]
    cfg = diff_ckpt["config"]
    diffusion = create_sensor_diffusion_v2(
        d_model=cfg["d_model"], num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"], dropout=0.0,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()

    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T_diff, diff_ckpt["schedule"])

    # Compute global mean latents
    mean_latents = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in test_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                mean_latents[k].append(outputs[k]["mu"].cpu())
    mean_latents = {k: torch.cat(v).mean(0, keepdim=True).to(DEVICE) for k, v in mean_latents.items()}

    # Missing patterns to evaluate
    patterns = {
        "phone_acc":  ["phone_acc"],
        "watch_acc":  ["watch_acc"],
        "glasses_acc": ["glasses_acc"],
        "phone_all":  ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
        "watch_all":  ["watch_acc", "watch_gyro"],
        "only_watch": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"],
        "only_phone": ["watch_acc", "watch_gyro", "glasses_acc"],
    }

    results = {}

    for pattern_name, missing_sensors in patterns.items():
        print(f"\nPattern: {pattern_name} (missing: {missing_sensors})")

        mse_diff = mse_mean = mse_zero = 0.0
        cos_diff = cos_mean = cos_zero = 0.0
        n_total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=pattern_name, leave=False):
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}
                B = next(iter(latents.values())).shape[0]

                # Normalize
                latents_norm = {}
                for name in SENSOR_NAMES:
                    mean = norm_stats[name]["mean"].to(DEVICE)
                    std = norm_stats[name]["std"].to(DEVICE)
                    latents_norm[name] = (latents[name] - mean) / std

                stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)
                observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for name in missing_sensors:
                    observed_mask[:, SENSOR_NAMES.index(name)] = 0.0

                # Diffusion imputation
                imputed = ddim_sample_v2(diffusion, stacked, observed_mask,
                                         sched["alpha_bar"], T_diff, DDIM_STEPS)

                # Evaluate only on missing sensors
                for name in missing_sensors:
                    idx = SENSOR_NAMES.index(name)
                    real = latents_norm[name]  # (B, D, T)
                    mean_n = norm_stats[name]["mean"].to(DEVICE)
                    std_n = norm_stats[name]["std"].to(DEVICE)

                    # Diffusion imputed (normalized)
                    imp_diff = imputed[:, idx]
                    # Mean imputed (normalized)
                    imp_mean = (mean_latents[name].expand(B, -1, -1) - mean_n) / std_n
                    # Zero imputed (normalized: (0 - mean) / std)
                    imp_zero = (-mean_n / std_n).expand(B, -1, real.shape[-1])

                    mse_diff += F.mse_loss(imp_diff, real).item()
                    mse_mean += F.mse_loss(imp_mean, real).item()
                    mse_zero += F.mse_loss(imp_zero, real).item()

                    # Cosine similarity (flatten D*T)
                    r_flat = real.flatten(1)
                    cos_diff += F.cosine_similarity(imp_diff.flatten(1), r_flat).mean().item()
                    cos_mean += F.cosine_similarity(imp_mean.flatten(1), r_flat).mean().item()
                    cos_zero += F.cosine_similarity(imp_zero.flatten(1), r_flat).mean().item()

                    n_total += 1

        n = n_total / len(missing_sensors) * len(test_loader)  # normalize per sensor per batch
        n_s = len(missing_sensors) * len(test_loader)
        results[pattern_name] = {
            "mse_diff": mse_diff / n_s,
            "mse_mean": mse_mean / n_s,
            "mse_zero": mse_zero / n_s,
            "cos_diff": cos_diff / n_s,
            "cos_mean": cos_mean / n_s,
            "cos_zero": cos_zero / n_s,
        }
        r = results[pattern_name]
        print(f"  MSE  — Diff: {r['mse_diff']:.4f} | Mean: {r['mse_mean']:.4f} | Zero: {r['mse_zero']:.4f}")
        print(f"  CoSim— Diff: {r['cos_diff']:.4f} | Mean: {r['cos_mean']:.4f} | Zero: {r['cos_zero']:.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY — MSE (lower=better)")
    print(f"{'='*70}")
    print(f"{'Pattern':<15} {'MSE Diff':>10} {'MSE Mean':>10} {'MSE Zero':>10} {'Winner':>10}")
    print("-" * 60)
    for name, r in results.items():
        best = min(r['mse_diff'], r['mse_mean'], r['mse_zero'])
        winner = "Diff" if best == r['mse_diff'] else ("Mean" if best == r['mse_mean'] else "Zero")
        print(f"{name:<15} {r['mse_diff']:>10.4f} {r['mse_mean']:>10.4f} {r['mse_zero']:>10.4f} {winner:>10}")

    print(f"\n{'='*70}")
    print("SUMMARY — Cosine Similarity (higher=better)")
    print(f"{'='*70}")
    print(f"{'Pattern':<15} {'CoS Diff':>10} {'CoS Mean':>10} {'CoS Zero':>10} {'Winner':>10}")
    print("-" * 60)
    for name, r in results.items():
        best = max(r['cos_diff'], r['cos_mean'], r['cos_zero'])
        winner = "Diff" if best == r['cos_diff'] else ("Mean" if best == r['cos_mean'] else "Zero")
        print(f"{name:<15} {r['cos_diff']:>10.4f} {r['cos_mean']:>10.4f} {r['cos_zero']:>10.4f} {winner:>10}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
