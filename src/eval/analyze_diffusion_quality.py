# ============================================================
# Analyze Diffusion Reconstruction Quality per Sensor
# Computes: L-MSE, Correlation, Signal-MSE for imputed vs real
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion import create_sensor_diffusion_model
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from torch.utils.data import ConcatDataset


# ============================================================
# CONFIG
# ============================================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion-dir", type=str, default="checkpoints/sensor_diffusion",
                    help="Directory containing diffusion checkpoint")
args, _ = parser.parse_known_args()

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
MAX_BATCHES = 50  # Limit for faster analysis

# Device groups
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
# METRICS
# ============================================================
def compute_correlation(x, y):
    """Compute Pearson correlation between flattened tensors."""
    x_flat = x.flatten().cpu().numpy()
    y_flat = y.flatten().cpu().numpy()

    if x_flat.std() < 1e-8 or y_flat.std() < 1e-8:
        return 0.0

    return np.corrcoef(x_flat, y_flat)[0, 1]


def compute_cosine_sim(x, y):
    """Compute cosine similarity between tensors."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    return F.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).item()


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*70}")
    print("Diffusion Reconstruction Quality Analysis")
    print(f"DDIM Steps: {DDIM_STEPS}, Max Batches: {MAX_BATCHES}")
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
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load VAE
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    print(f"VAE from epoch {vae_ckpt.get('epoch', '?')}, recon={vae_ckpt.get('test_recon', '?')}")

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
    print(f"Diffusion: T={T}, schedule={schedule_type}")
    print(f"Config: hidden={cfg['hidden_dim']}, heads={cfg['num_heads']}, "
          f"conv_blocks={cfg['num_conv_blocks']}, attn_blocks={cfg['num_attn_blocks']}")

    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T, schedule_type)

    # ============================================================
    # SINGLE SENSOR MISSING ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("SINGLE SENSOR MISSING - Imputation Quality")
    print(f"{'='*70}")

    single_results = {name: {"l_mse": [], "corr": [], "cos_sim": []} for name in SENSOR_NAMES}

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Single sensor", total=min(MAX_BATCHES, len(test_loader)))):
        if batch_idx >= MAX_BATCHES:
            break

        sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}

        with torch.no_grad():
            outputs = vae(sensor_data)
            latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

        # Normalize
        latents_norm = {}
        for name in SENSOR_NAMES:
            mean = norm_stats[name]["mean"].to(DEVICE)
            std = norm_stats[name]["std"].to(DEVICE)
            latents_norm[name] = (latents[name] - mean) / std

        # Impute each sensor one at a time
        for target in SENSOR_NAMES:
            conditions = {
                k: (latents_norm[k] if k != target else None)
                for k in SENSOR_NAMES
            }

            imputed_norm = ddim_sample_batch(
                model=diffusion,
                target_modality=target,
                shape=latents_norm[target].shape,
                conditions=conditions,
                alpha_bar=sched["alpha_bar"],
                T=T,
                ddim_steps=DDIM_STEPS,
            )

            # Denormalize
            mean = norm_stats[target]["mean"].to(DEVICE)
            std = norm_stats[target]["std"].to(DEVICE)
            imputed = imputed_norm * std + mean

            # Metrics
            real = latents[target]
            l_mse = F.mse_loss(imputed, real).item()
            corr = compute_correlation(imputed, real)
            cos_sim = compute_cosine_sim(imputed, real)

            single_results[target]["l_mse"].append(l_mse)
            single_results[target]["corr"].append(corr)
            single_results[target]["cos_sim"].append(cos_sim)

    # Print single sensor results
    print(f"\n{'Sensor':<15} {'L-MSE':<10} {'Correlation':<12} {'Cosine Sim':<12}")
    print("-" * 50)
    for name in SENSOR_NAMES:
        l_mse = np.mean(single_results[name]["l_mse"])
        corr = np.mean(single_results[name]["corr"])
        cos_sim = np.mean(single_results[name]["cos_sim"])
        print(f"{name:<15} {l_mse:.6f}   {corr:.4f}       {cos_sim:.4f}")

    # ============================================================
    # DEVICE-LEVEL MISSING ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print("DEVICE-LEVEL MISSING - Imputation Quality")
    print(f"{'='*70}")

    device_results = {}

    for device_name, device_sensors in DEVICE_GROUPS.items():
        device_results[device_name] = {name: {"l_mse": [], "corr": []} for name in device_sensors}

        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{device_name} missing", total=min(MAX_BATCHES, len(test_loader)))):
            if batch_idx >= MAX_BATCHES:
                break

            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            # Normalize
            latents_norm = {}
            for name in SENSOR_NAMES:
                mean = norm_stats[name]["mean"].to(DEVICE)
                std = norm_stats[name]["std"].to(DEVICE)
                latents_norm[name] = (latents[name] - mean) / std

            # Impute all sensors from this device
            for target in device_sensors:
                conditions = {
                    k: (latents_norm[k] if k not in device_sensors else None)
                    for k in SENSOR_NAMES
                }

                imputed_norm = ddim_sample_batch(
                    model=diffusion,
                    target_modality=target,
                    shape=latents_norm[target].shape,
                    conditions=conditions,
                    alpha_bar=sched["alpha_bar"],
                    T=T,
                    ddim_steps=DDIM_STEPS,
                )

                mean = norm_stats[target]["mean"].to(DEVICE)
                std = norm_stats[target]["std"].to(DEVICE)
                imputed = imputed_norm * std + mean

                real = latents[target]
                l_mse = F.mse_loss(imputed, real).item()
                corr = compute_correlation(imputed, real)

                device_results[device_name][target]["l_mse"].append(l_mse)
                device_results[device_name][target]["corr"].append(corr)

    # Print device-level results
    for device_name, sensors in device_results.items():
        print(f"\n{device_name.upper()} missing (all {len(DEVICE_GROUPS[device_name])} sensors):")
        print(f"  {'Sensor':<15} {'L-MSE':<12} {'Correlation':<12}")
        print("  " + "-" * 40)
        for name, metrics in sensors.items():
            l_mse = np.mean(metrics["l_mse"])
            corr = np.mean(metrics["corr"])
            print(f"  {name:<15} {l_mse:.6f}     {corr:.4f}")

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON: Diffusion vs Mean-Fill vs Zero-Fill")
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

        # Normalize
        latents_norm = {}
        for name in SENSOR_NAMES:
            mean = norm_stats[name]["mean"].to(DEVICE)
            std = norm_stats[name]["std"].to(DEVICE)
            latents_norm[name] = (latents[name] - mean) / std

        for target in SENSOR_NAMES:
            real = latents[target]
            T_len = real.shape[2]

            # Diffusion imputation
            conditions = {
                k: (latents_norm[k] if k != target else None)
                for k in SENSOR_NAMES
            }
            imputed_norm = ddim_sample_batch(
                model=diffusion,
                target_modality=target,
                shape=latents_norm[target].shape,
                conditions=conditions,
                alpha_bar=sched["alpha_bar"],
                T=T,
                ddim_steps=DDIM_STEPS,
            )
            mean = norm_stats[target]["mean"].to(DEVICE)
            std = norm_stats[target]["std"].to(DEVICE)
            imputed_diff = imputed_norm * std + mean

            # Mean-fill
            mean_fill = norm_stats[target]["mean"].to(DEVICE).expand(B, -1, T_len)

            # Zero-fill
            zero_fill = torch.zeros_like(real)

            # MSEs
            baseline_results[target]["diff"].append(F.mse_loss(imputed_diff, real).item())
            baseline_results[target]["mean"].append(F.mse_loss(mean_fill, real).item())
            baseline_results[target]["zero"].append(F.mse_loss(zero_fill, real).item())

    print(f"\n{'Sensor':<15} {'Diffusion':<12} {'Mean-Fill':<12} {'Zero-Fill':<12} {'Diff vs Mean':<12}")
    print("-" * 65)
    for name in SENSOR_NAMES:
        diff_mse = np.mean(baseline_results[name]["diff"])
        mean_mse = np.mean(baseline_results[name]["mean"])
        zero_mse = np.mean(baseline_results[name]["zero"])
        improvement = mean_mse - diff_mse
        print(f"{name:<15} {diff_mse:.6f}     {mean_mse:.6f}     {zero_mse:.6f}     {improvement:+.6f}")

    # ============================================================
    # DATA STATISTICS
    # ============================================================
    print(f"\n{'='*70}")
    print("DATA STATISTICS")
    print(f"{'='*70}")

    print(f"\nDataset sizes:")
    for name, root in DATA_ROOTS.items():
        try:
            train_ds = CogAgeSensorDataset(root, "training", normalizer)
            test_ds = CogAgeSensorDataset(root, "testing", normalizer)
            print(f"  {name}: train={len(train_ds)}, test={len(test_ds)}")
        except Exception as e:
            print(f"  {name}: error - {e}")

    print(f"\nLatent statistics (from norm_stats):")
    for name in SENSOR_NAMES:
        mean = norm_stats[name]["mean"].cpu().numpy().flatten()
        std = norm_stats[name]["std"].cpu().numpy().flatten()
        print(f"  {name:<15}: mean={mean.mean():.4f}, std={std.mean():.4f}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
