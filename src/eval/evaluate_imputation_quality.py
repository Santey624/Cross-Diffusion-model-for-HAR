# ============================================================
# Evaluate Imputation Quality: Diffusion vs Mean vs Zero
# Measures MSE, Cosine Similarity, and FFT-MSE in latent space
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
from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
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


def fft_mse(pred, target):
    """MSE in frequency domain (magnitude spectrum along time axis)."""
    pred_fft = torch.fft.rfft(pred, dim=-1).abs()
    target_fft = torch.fft.rfft(target, dim=-1).abs()
    return F.mse_loss(pred_fft, target_fft).item()


def main():
    print(f"\n{'='*70}")
    print("IMPUTATION QUALITY EVALUATION")
    print("Metrics: MSE, Cosine Similarity, FFT-MSE (real vs imputed latents)")
    print(f"{'='*70}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Load VAE
    vae = SensorMultiModalVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE)["model_state"])
    vae.eval()

    # Load Diffusion (V2 or V3)
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T_diff = diff_ckpt["T"]
    cfg = diff_ckpt["config"]
    if diff_ckpt.get("version", "v2") == "v3_conditional":
        diffusion = create_sensor_diffusion_v3(
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"], dropout=0.0,
        ).to(DEVICE)
        print("Using Conditional Diffusion V3")
    else:
        diffusion = create_sensor_diffusion_v2(
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"], dropout=0.0,
        ).to(DEVICE)
        print("Using Diffusion V2")
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

        # Track metrics per individual sensor
        sensor_acc = {name: {"mse_diff": 0, "mse_mean": 0, "mse_zero": 0,
                              "fft_diff": 0, "fft_mean": 0, "fft_zero": 0,
                              "cos_diff": 0, "cos_mean": 0, "cos_zero": 0,
                              "n": 0}
                      for name in missing_sensors}

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

                # Evaluate per missing sensor individually
                for name in missing_sensors:
                    idx = SENSOR_NAMES.index(name)
                    real = latents_norm[name]  # (B, D, T)
                    mean_n = norm_stats[name]["mean"].to(DEVICE)
                    std_n = norm_stats[name]["std"].to(DEVICE)

                    imp_diff = imputed[:, idx]
                    imp_mean = (mean_latents[name].expand(B, -1, -1) - mean_n) / std_n
                    imp_zero = (-mean_n / std_n).expand(B, -1, real.shape[-1])

                    sensor_acc[name]["mse_diff"] += F.mse_loss(imp_diff, real).item()
                    sensor_acc[name]["mse_mean"] += F.mse_loss(imp_mean, real).item()
                    sensor_acc[name]["mse_zero"] += F.mse_loss(imp_zero, real).item()

                    r_flat = real.flatten(1)
                    sensor_acc[name]["cos_diff"] += F.cosine_similarity(imp_diff.flatten(1), r_flat).mean().item()
                    sensor_acc[name]["cos_mean"] += F.cosine_similarity(imp_mean.flatten(1), r_flat).mean().item()
                    sensor_acc[name]["cos_zero"] += F.cosine_similarity(imp_zero.flatten(1), r_flat).mean().item()

                    sensor_acc[name]["fft_diff"] += fft_mse(imp_diff, real)
                    sensor_acc[name]["fft_mean"] += fft_mse(imp_mean, real)
                    sensor_acc[name]["fft_zero"] += fft_mse(imp_zero, real)

                    sensor_acc[name]["n"] += 1

        # Normalize by number of batches per sensor
        results[pattern_name] = {}
        for name in missing_sensors:
            n = max(sensor_acc[name]["n"], 1)
            results[pattern_name][name] = {k: v / n for k, v in sensor_acc[name].items() if k != "n"}

    # ============================================================
    # SUMMARY — Per-Sensor: Euclidean vs Fourier comparison
    # ============================================================
    print(f"\n\n{'='*110}")
    print("IMPUTATION QUALITY — Per-Sensor: Euclidean MSE vs FFT-MSE (Fourier)")
    print(f"{'='*110}")

    for pattern_name, sensor_data in results.items():
        missing = patterns[pattern_name]
        print(f"\nPattern: {pattern_name}  |  Missing sensors: {missing}")
        hdr = (f"  {'Sensor':<14}"
               f" {'EucDiff':>9} {'EucMean':>9} {'EucZero':>9} {'Best(Euc)':>10}"
               f" | {'FFTDiff':>9} {'FFTMean':>9} {'FFTZero':>9} {'Best(FFT)':>10}"
               f" | {'CosDiff':>8} {'CosMean':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for sensor, m in sensor_data.items():
            euc_best_val = min(m["mse_diff"], m["mse_mean"], m["mse_zero"])
            fft_best_val = min(m["fft_diff"], m["fft_mean"], m["fft_zero"])
            euc_best = ("Diff" if euc_best_val == m["mse_diff"]
                        else "Mean" if euc_best_val == m["mse_mean"] else "Zero")
            fft_best = ("Diff" if fft_best_val == m["fft_diff"]
                        else "Mean" if fft_best_val == m["fft_mean"] else "Zero")
            print(f"  {sensor:<14}"
                  f" {m['mse_diff']:>9.4f} {m['mse_mean']:>9.4f} {m['mse_zero']:>9.4f} {euc_best:>10}"
                  f" | {m['fft_diff']:>9.4f} {m['fft_mean']:>9.4f} {m['fft_zero']:>9.4f} {fft_best:>10}"
                  f" | {m['cos_diff']:>8.4f} {m['cos_mean']:>8.4f}")

    # ============================================================
    # AGGREGATE SUMMARY — Count wins per method
    # ============================================================
    print(f"\n\n{'='*60}")
    print("AGGREGATE WINS (across all sensors and patterns)")
    print(f"{'='*60}")
    euc_wins = {"Diff": 0, "Mean": 0, "Zero": 0}
    fft_wins = {"Diff": 0, "Mean": 0, "Zero": 0}
    for sensor_data in results.values():
        for m in sensor_data.values():
            euc_best_val = min(m["mse_diff"], m["mse_mean"], m["mse_zero"])
            fft_best_val = min(m["fft_diff"], m["fft_mean"], m["fft_zero"])
            euc_best = ("Diff" if euc_best_val == m["mse_diff"]
                        else "Mean" if euc_best_val == m["mse_mean"] else "Zero")
            fft_best = ("Diff" if fft_best_val == m["fft_diff"]
                        else "Mean" if fft_best_val == m["fft_mean"] else "Zero")
            euc_wins[euc_best] += 1
            fft_wins[fft_best] += 1
    total = sum(euc_wins.values())
    print(f"  Euclidean MSE wins: Diff={euc_wins['Diff']}/{total}  Mean={euc_wins['Mean']}/{total}  Zero={euc_wins['Zero']}/{total}")
    print(f"  FFT-MSE wins:       Diff={fft_wins['Diff']}/{total}  Mean={fft_wins['Mean']}/{total}  Zero={fft_wins['Zero']}/{total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    if "--v3" in sys.argv:
        DIFFUSION_DIR = Path("checkpoints/sensor_diffusion_v3")
    main()
