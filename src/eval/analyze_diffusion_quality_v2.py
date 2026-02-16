# ============================================================
# Analyze Diffusion V2 Reconstruction Quality
# Adapted for multi-sensor masking architecture
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
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from torch.utils.data import ConcatDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion-dir", type=str, default="checkpoints/sensor_diffusion_v2")
args, _ = parser.parse_known_args()


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_best.pt"
DIFFUSION_DIR = Path(args.diffusion_dir)
NORMALIZER_PATH = "data/sensor_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

DDIM_STEPS = 50
BATCH_SIZE = 32
MAX_BATCHES = 50

K = len(SENSOR_NAMES)

DEVICE_GROUPS = {
    "phone": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
    "watch": ["watch_acc", "watch_gyro"],
    "glasses": ["glasses_acc"],
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
# DDIM SAMPLER for V2
# ============================================================
@torch.no_grad()
def ddim_sample_v2(model, stacked_latents, observed_mask, alpha_bar, T, ddim_steps=50):
    """
    DDIM sampling for V2 model.

    Args:
        model: SensorJointDiffusionV2
        stacked_latents: (B, K, D, T) — observed sensors have real values
        observed_mask: (B, K) — 1.0 if observed
        alpha_bar: diffusion schedule
        T: total timesteps
        ddim_steps: number of DDIM steps

    Returns:
        imputed: (B, K, D, T) — with missing sensors filled in
    """
    B, K_s, D, T_len = stacked_latents.shape
    device = stacked_latents.device
    missing_mask = 1.0 - observed_mask  # (B, K)

    # Initialize: observed = real, missing = random noise
    z = stacked_latents.clone()
    noise_init = torch.randn_like(stacked_latents)
    z = observed_mask[:, :, None, None] * z + missing_mask[:, :, None, None] * noise_init

    alpha_bar = alpha_bar.to(device)

    # DDIM timesteps (quadratic spacing)
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        # Always keep observed sensors clean
        noisy_input = observed_mask[:, :, None, None] * stacked_latents + \
                      missing_mask[:, :, None, None] * z

        noise_pred = model(noisy_input, t_batch, observed_mask)

        ab_now = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        # Only update missing sensors
        pred_x0 = (z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

        dir_zt = torch.sqrt(1 - ab_next) * noise_pred
        z_new = torch.sqrt(ab_next) * pred_x0 + dir_zt

        # Keep observed, update missing
        z = observed_mask[:, :, None, None] * stacked_latents + \
            missing_mask[:, :, None, None] * z_new

    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*70}")
    print("Diffusion V2 Reconstruction Quality Analysis")
    print(f"{'='*70}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"], "testing", normalizer),
        CogAgeSensorDataset(DATA_ROOTS["bbh"], "testing", normalizer),
        CogAgeSensorDataset(DATA_ROOTS["state"], "testing", normalizer),
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    # Load VAE
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load diffusion V2
    print("Loading diffusion V2 model...")
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T_diff = diff_ckpt["T"]
    schedule_type = diff_ckpt["schedule"]
    cfg = diff_ckpt["config"]

    diffusion = create_sensor_diffusion_v2(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()
    print(f"Diffusion V2: T={T_diff}, d_model={cfg['d_model']}, blocks={cfg['num_blocks']}")

    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T_diff, schedule_type)

    # ============================================================
    # SINGLE SENSOR MISSING
    # ============================================================
    print(f"\n{'='*70}")
    print("SINGLE SENSOR MISSING")
    print(f"{'='*70}")

    single_results = {name: {"l_mse": [], "corr": []} for name in SENSOR_NAMES}

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Single sensor", total=min(MAX_BATCHES, len(test_loader)))):
        if batch_idx >= MAX_BATCHES:
            break

        sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
        B = sensor_data[SENSOR_NAMES[0]].size(0)

        with torch.no_grad():
            outputs = vae(sensor_data)
            latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

        # Normalize and stack
        latents_norm = {}
        for name in SENSOR_NAMES:
            mean = norm_stats[name]["mean"].to(DEVICE)
            std = norm_stats[name]["std"].to(DEVICE)
            latents_norm[name] = (latents[name] - mean) / std

        stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)  # (B, K, D, T)

        for target_idx, target_name in enumerate(SENSOR_NAMES):
            # Mask: all observed except target
            observed_mask = torch.ones(B, K, device=DEVICE)
            observed_mask[:, target_idx] = 0.0

            imputed = ddim_sample_v2(diffusion, stacked, observed_mask, sched["alpha_bar"], T_diff, DDIM_STEPS)

            # Denormalize imputed target
            mean = norm_stats[target_name]["mean"].to(DEVICE)
            std = norm_stats[target_name]["std"].to(DEVICE)
            imputed_target = imputed[:, target_idx] * std + mean
            real = latents[target_name]

            l_mse = F.mse_loss(imputed_target, real).item()
            x_flat = imputed_target.flatten().cpu().numpy()
            y_flat = real.flatten().cpu().numpy()
            corr = np.corrcoef(x_flat, y_flat)[0, 1] if x_flat.std() > 1e-8 else 0.0

            single_results[target_name]["l_mse"].append(l_mse)
            single_results[target_name]["corr"].append(corr)

    print(f"\n{'Sensor':<15} {'L-MSE':<12} {'Correlation':<12}")
    print("-" * 40)
    for name in SENSOR_NAMES:
        l_mse = np.mean(single_results[name]["l_mse"])
        corr = np.mean(single_results[name]["corr"])
        print(f"{name:<15} {l_mse:.6f}     {corr:.4f}")

    # ============================================================
    # DEVICE-LEVEL MISSING
    # ============================================================
    print(f"\n{'='*70}")
    print("DEVICE-LEVEL MISSING")
    print(f"{'='*70}")

    for device_name, device_sensors in DEVICE_GROUPS.items():
        device_indices = [SENSOR_NAMES.index(s) for s in device_sensors]
        device_results = {name: {"l_mse": [], "corr": []} for name in device_sensors}

        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{device_name} missing", total=min(MAX_BATCHES, len(test_loader)))):
            if batch_idx >= MAX_BATCHES:
                break

            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            B = sensor_data[SENSOR_NAMES[0]].size(0)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            latents_norm = {}
            for name in SENSOR_NAMES:
                mean = norm_stats[name]["mean"].to(DEVICE)
                std = norm_stats[name]["std"].to(DEVICE)
                latents_norm[name] = (latents[name] - mean) / std

            stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)

            # Mask: device sensors missing
            observed_mask = torch.ones(B, K, device=DEVICE)
            for idx in device_indices:
                observed_mask[:, idx] = 0.0

            imputed = ddim_sample_v2(diffusion, stacked, observed_mask, sched["alpha_bar"], T_diff, DDIM_STEPS)

            for target_name in device_sensors:
                target_idx = SENSOR_NAMES.index(target_name)
                mean = norm_stats[target_name]["mean"].to(DEVICE)
                std = norm_stats[target_name]["std"].to(DEVICE)
                imputed_target = imputed[:, target_idx] * std + mean
                real = latents[target_name]

                l_mse = F.mse_loss(imputed_target, real).item()
                x_flat = imputed_target.flatten().cpu().numpy()
                y_flat = real.flatten().cpu().numpy()
                corr = np.corrcoef(x_flat, y_flat)[0, 1] if x_flat.std() > 1e-8 else 0.0

                device_results[target_name]["l_mse"].append(l_mse)
                device_results[target_name]["corr"].append(corr)

        print(f"\n{device_name.upper()} missing ({len(device_sensors)} sensors):")
        print(f"  {'Sensor':<15} {'L-MSE':<12} {'Correlation':<12}")
        print("  " + "-" * 40)
        for name in device_sensors:
            l_mse = np.mean(device_results[name]["l_mse"])
            corr = np.mean(device_results[name]["corr"])
            print(f"  {name:<15} {l_mse:.6f}     {corr:.4f}")

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON: V2 Diffusion vs Mean-Fill vs Zero-Fill")
    print(f"{'='*70}")

    baseline_results = {name: {"diff": [], "mean": [], "zero": []} for name in SENSOR_NAMES}

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Baselines", total=min(MAX_BATCHES, len(test_loader)))):
        if batch_idx >= MAX_BATCHES:
            break

        sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
        B = sensor_data[SENSOR_NAMES[0]].size(0)

        with torch.no_grad():
            outputs = vae(sensor_data)
            latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

        latents_norm = {}
        for name in SENSOR_NAMES:
            mean = norm_stats[name]["mean"].to(DEVICE)
            std = norm_stats[name]["std"].to(DEVICE)
            latents_norm[name] = (latents[name] - mean) / std

        stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)

        for target_idx, target_name in enumerate(SENSOR_NAMES):
            real = latents[target_name]
            T_len = real.shape[2]

            # Diffusion
            observed_mask = torch.ones(B, K, device=DEVICE)
            observed_mask[:, target_idx] = 0.0
            imputed = ddim_sample_v2(diffusion, stacked, observed_mask, sched["alpha_bar"], T_diff, DDIM_STEPS)
            mean_s = norm_stats[target_name]["mean"].to(DEVICE)
            std_s = norm_stats[target_name]["std"].to(DEVICE)
            imputed_target = imputed[:, target_idx] * std_s + mean_s

            # Mean-fill
            mean_fill = norm_stats[target_name]["mean"].to(DEVICE).expand(B, -1, T_len)

            # Zero-fill
            zero_fill = torch.zeros_like(real)

            baseline_results[target_name]["diff"].append(F.mse_loss(imputed_target, real).item())
            baseline_results[target_name]["mean"].append(F.mse_loss(mean_fill, real).item())
            baseline_results[target_name]["zero"].append(F.mse_loss(zero_fill, real).item())

    print(f"\n{'Sensor':<15} {'Diffusion':<12} {'Mean-Fill':<12} {'Zero-Fill':<12} {'Diff vs Mean':<12}")
    print("-" * 65)
    for name in SENSOR_NAMES:
        diff_mse = np.mean(baseline_results[name]["diff"])
        mean_mse = np.mean(baseline_results[name]["mean"])
        zero_mse = np.mean(baseline_results[name]["zero"])
        improvement = mean_mse - diff_mse
        marker = "  ✓" if improvement > 0 else ""
        print(f"{name:<15} {diff_mse:.6f}     {mean_mse:.6f}     {zero_mse:.6f}     {improvement:+.6f}{marker}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
