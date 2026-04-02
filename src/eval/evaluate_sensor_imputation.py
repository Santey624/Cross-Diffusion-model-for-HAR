# ============================================================
# Evaluate Sensor-Level Joint Diffusion
# Supports single-sensor and device-level missing scenarios
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion import create_sensor_diffusion_model


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_best.pt"
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion")
LATENTS_DIR = Path("data/sensor_latents")

NUM_EVAL_SAMPLES = 100
DDIM_STEPS = 200

OUTPUT_DIR = Path("outputs/sensor_imputation_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation scenarios
SCENARIOS = {
    # All 7 single-sensor missing
    "single_phone_acc":   {"missing": ["phone_acc"]},
    "single_phone_gyro":  {"missing": ["phone_gyro"]},
    "single_phone_grav":  {"missing": ["phone_grav"]},
    "single_phone_lacc":  {"missing": ["phone_lacc"]},
    "single_watch_acc":   {"missing": ["watch_acc"]},
    "single_watch_gyro":  {"missing": ["watch_gyro"]},
    "single_glasses_acc": {"missing": ["glasses_acc"]},
    # Device-level missing
    "device_phone":   {"missing": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"]},
    "device_watch":   {"missing": ["watch_acc", "watch_gyro"]},
    "device_glasses": {"missing": ["glasses_acc"]},
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
# DDIM SAMPLER (quadratic spacing)
# ============================================================
@torch.no_grad()
def ddim_sample(model, target_modality, shape, conditions, alpha_bar, T, ddim_steps=200):
    B = shape[0]
    device = next(model.parameters()).device
    z = torch.randn(shape, device=device)
    alpha_bar = alpha_bar.to(device)

    # Quadratic spacing
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    for i in tqdm(range(len(tau) - 1), desc=f"DDIM {target_modality}", leave=False):
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
    print("Evaluating Sensor-Level Joint Diffusion")
    print(f"DDIM Steps: {DDIM_STEPS}")
    print(f"{'='*60}\n")

    # Load diffusion checkpoint
    ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T = ckpt["T"]
    schedule_type = ckpt["schedule"]
    cfg = ckpt["config"]
    print(f"Schedule: {schedule_type}, T: {T}, Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.6f}")

    # Load VAE
    print("Loading Sensor VAE (shared latent space)...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load latents (all same shape: (N, D, T_SHARED))
    print("Loading sensor latents...")
    latents = {}
    for name in SENSOR_NAMES:
        latents[name] = torch.load(LATENTS_DIR / f"train_latents_{name}_mu.pt")[:NUM_EVAL_SAMPLES]
        print(f"  {name:15s}: {latents[name].shape}")

    # Load normalization stats
    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)

    # Normalize
    latents_norm = {}
    for name in SENSOR_NAMES:
        mean = norm_stats[name]["mean"].to(DEVICE)
        std = norm_stats[name]["std"].to(DEVICE)
        latents_norm[name] = (latents[name].to(DEVICE) - mean) / std

    # Load diffusion model
    model = create_sensor_diffusion_model(
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_conv_blocks=cfg["num_conv_blocks"],
        num_attn_blocks=cfg["num_attn_blocks"],
        dropout=0.0,
    ).to(DEVICE)

    model_dict = model.state_dict()
    pretrained = {k: v for k, v in ckpt["model_state"].items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model.eval()

    # Build schedule
    sched = make_schedule(T, schedule_type)

    # Run each scenario
    all_results = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        missing = scenario_cfg["missing"]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"Missing sensors: {missing}")
        print(f"{'='*60}")

        scenario_results = {}

        for target_name in missing:
            # Build conditions: all except missing sensors
            conditions = {}
            for name in SENSOR_NAMES:
                if name in missing:
                    conditions[name] = None
                else:
                    conditions[name] = latents_norm[name]

            # Sample
            target_shape = latents_norm[target_name].shape
            imputed_norm = ddim_sample(
                model=model,
                target_modality=target_name,
                shape=target_shape,
                conditions=conditions,
                alpha_bar=sched["alpha_bar"],
                T=T,
                ddim_steps=DDIM_STEPS,
            )

            # Denormalize
            mean = norm_stats[target_name]["mean"].to(DEVICE)
            std = norm_stats[target_name]["std"].to(DEVICE)
            imputed = imputed_norm * std + mean

            # Latent MSE
            gt = latents[target_name].to(DEVICE)
            latent_mse = F.mse_loss(imputed, gt).item()

            # Mean-fill baseline (training mean latent)
            mean_lat = latents[target_name].mean(dim=0, keepdim=True).to(DEVICE)
            mean_fill = mean_lat.expand_as(gt)
            latent_mse_mean = F.mse_loss(mean_fill, gt).item()

            # Decode with VAE (using shared decoder)
            with torch.no_grad():
                imputed_signals    = vae.decode_sensor(target_name, imputed).cpu()
                gt_signals         = vae.decode_sensor(target_name, gt).cpu()
                mean_signals       = vae.decode_sensor(target_name, mean_fill).cpu()

            signal_mse      = F.mse_loss(imputed_signals, gt_signals).item()
            signal_mse_mean = F.mse_loss(mean_signals,   gt_signals).item()

            best_l = "Diff" if latent_mse < latent_mse_mean else "Mean"
            best_s = "Diff" if signal_mse  < signal_mse_mean  else "Mean"

            scenario_results[target_name] = {
                "latent_mse": latent_mse, "latent_mse_mean": latent_mse_mean,
                "signal_mse": signal_mse, "signal_mse_mean": signal_mse_mean,
            }

            print(f"  {target_name:15s}: "
                  f"L-MSE Diff={latent_mse:.4f}  Mean={latent_mse_mean:.4f}  [{best_l}] | "
                  f"S-MSE Diff={signal_mse:.4f}  Mean={signal_mse_mean:.4f}  [{best_s}]")

            # Visualize first 3 samples
            n_ch = imputed_signals.shape[2]
            for idx in range(min(3, imputed_signals.shape[0])):
                fig, axs = plt.subplots(n_ch, 1, figsize=(14, 3 * n_ch))
                if n_ch == 1:
                    axs = [axs]
                for ch in range(n_ch):
                    axs[ch].plot(gt_signals[idx, :, ch].numpy(), label="Ground Truth", alpha=0.8)
                    axs[ch].plot(imputed_signals[idx, :, ch].numpy(), label="Diffusion",
                                 alpha=0.8, linestyle='--')
                    axs[ch].set_title(f"{target_name} Ch{ch}")
                    axs[ch].legend()
                    axs[ch].grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"{scenario_name}_{target_name}_sample_{idx}.png", dpi=150)
                plt.close()

        all_results[scenario_name] = scenario_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - All Scenarios")
    print(f"{'='*60}")

    for scenario_name, results in all_results.items():
        print(f"\n  {scenario_name}:")
        avg_latent = sum(r["latent_mse"] for r in results.values()) / len(results)
        avg_signal = sum(r["signal_mse"] for r in results.values()) / len(results)
        for sensor, r in results.items():
            print(f"    {sensor:15s}: L-MSE={r['latent_mse']:.6f}  S-MSE={r['signal_mse']:.6f}")
        if len(results) > 1:
            print(f"    {'AVG':15s}: L-MSE={avg_latent:.6f}  S-MSE={avg_signal:.6f}")

    # Save metrics
    with open(OUTPUT_DIR / "all_metrics.txt", "w") as f:
        for scenario_name, results in all_results.items():
            f.write(f"\n{scenario_name}:\n")
            for sensor, r in results.items():
                f.write(f"  {sensor}: latent_mse={r['latent_mse']:.6f}, "
                        f"signal_mse={r['signal_mse']:.6f}\n")

    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
