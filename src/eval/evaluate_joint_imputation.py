# ============================================================
# Evaluate Joint Model Imputation Quality
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.temporal_vae import TemporalMultiModalVAE
from src.models.joint_diffusion import create_joint_diffusion_model


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAE checkpoint
VAE_CHECKPOINT = "checkpoints/vae_gpu_epoch_050.pt"

# Joint diffusion checkpoint
DIFFUSION_DIR = Path("checkpoints/joint_diffusion")
DIFFUSION_EPOCH = 300

# Latents
LATENTS_DIR = Path("data/latents")

# Which modality to impute
MISSING_MODALITY = "phone"  # "phone", "watch", or "glasses"

# How many samples to evaluate
NUM_EVAL_SAMPLES = 10

# Output
OUTPUT_DIR = Path("outputs/joint_imputation_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DDIM sampling
DDIM_STEPS = 50


# ============================================================
# DIFFUSION UTILS
# ============================================================
def make_ddpm_schedule(T: int, beta_start: float, beta_end: float, device: str):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - torch.cat([alpha_bar.new_ones(1), alpha_bar[:-1]])) / (1.0 - alpha_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


@torch.no_grad()
def ddpm_sample(model, target_modality, shape, phone_cond, watch_cond, glasses_cond, sched, T, ddim_steps=None):
    """DDIM sampling with joint model"""
    B, D, seq_len = shape
    device = next(model.parameters()).device
    z = torch.randn(shape, device=device)

    if ddim_steps is not None:
        timesteps = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, dtype=torch.long, device=device)

    for i, t_val in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        noise_pred = model(
            target_modality=target_modality,
            z_t=z,
            t=t,
            phone_latent=phone_cond,
            watch_latent=watch_cond,
            glasses_latent=glasses_cond,
        )

        alpha_bar_t = sched["alpha_bar"][t_val].view(-1, 1, 1)

        if i < len(timesteps) - 1:
            alpha_bar_prev = sched["alpha_bar"][timesteps[i + 1]].view(-1, 1, 1)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device).view(-1, 1, 1)

        pred_x0 = (z - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -3, 3)
        dir_zt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
        z = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_zt

    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"Evaluating JOINT model imputation for: {MISSING_MODALITY.upper()}")
    print(f"{'='*60}\n")

    # Load VAE
    print("Loading VAE...")
    vae = TemporalMultiModalVAE(z_phone=32, z_watch=32, z_glasses=16).to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load latents
    print("Loading latents...")
    phone_latents = torch.load(LATENTS_DIR / "train_latents_phone_mu.pt")[:NUM_EVAL_SAMPLES]
    watch_latents = torch.load(LATENTS_DIR / "train_latents_watch_mu.pt")[:NUM_EVAL_SAMPLES]
    glasses_latents = torch.load(LATENTS_DIR / "train_latents_glasses_mu.pt")[:NUM_EVAL_SAMPLES]

    # Load joint diffusion model
    ckpt_path = DIFFUSION_DIR / f"joint_ddpm_epoch_{DIFFUSION_EPOCH:03d}.pt"
    print(f"Loading joint diffusion checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = create_joint_diffusion_model(
        hidden_dim=ckpt['config']['hidden_dim'],
        num_heads=ckpt['config']['num_heads'],
        num_conv_blocks=ckpt['config']['num_conv_blocks'],
        num_attn_blocks=ckpt['config']['num_attn_blocks'],
        dropout=ckpt['config']['dropout'],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load normalization stats
    stats_path = DIFFUSION_DIR / "normalization_stats.pt"
    print(f"Loading normalization stats: {stats_path}")
    stats = torch.load(stats_path, map_location=DEVICE)

    # Normalize all modalities
    phone_mean = stats['phone']['mean'].to(DEVICE)
    phone_std = stats['phone']['std'].to(DEVICE)
    phone_norm = (phone_latents.to(DEVICE) - phone_mean) / phone_std

    watch_mean = stats['watch']['mean'].to(DEVICE)
    watch_std = stats['watch']['std'].to(DEVICE)
    watch_norm = (watch_latents.to(DEVICE) - watch_mean) / watch_std

    glasses_mean = stats['glasses']['mean'].to(DEVICE)
    glasses_std = stats['glasses']['std'].to(DEVICE)
    glasses_norm = (glasses_latents.to(DEVICE) - glasses_mean) / glasses_std

    # Determine target and conditions
    if MISSING_MODALITY == "phone":
        target_shape = phone_norm.shape
        phone_cond = None
        watch_cond = watch_norm
        glasses_cond = glasses_norm
        ground_truth_latents = phone_latents
        target_mean = phone_mean
        target_std = phone_std

    elif MISSING_MODALITY == "watch":
        target_shape = watch_norm.shape
        phone_cond = phone_norm
        watch_cond = None
        glasses_cond = glasses_norm
        ground_truth_latents = watch_latents
        target_mean = watch_mean
        target_std = watch_std

    elif MISSING_MODALITY == "glasses":
        target_shape = glasses_norm.shape
        phone_cond = phone_norm
        watch_cond = watch_norm
        glasses_cond = None
        ground_truth_latents = glasses_latents
        target_mean = glasses_mean
        target_std = glasses_std

    # Sample imputed latents
    print(f"\nImputing {MISSING_MODALITY} latents with joint model...")
    T = ckpt["T"]
    beta_start = ckpt["beta_start"]
    beta_end = ckpt["beta_end"]
    sched = make_ddpm_schedule(T, beta_start, beta_end, DEVICE)

    imputed_norm = ddpm_sample(
        model=model,
        target_modality=MISSING_MODALITY,
        shape=target_shape,
        phone_cond=phone_cond,
        watch_cond=watch_cond,
        glasses_cond=glasses_cond,
        sched=sched,
        T=T,
        ddim_steps=DDIM_STEPS,
    )

    # Denormalize
    imputed_latents = imputed_norm * target_std + target_mean

    # Decode with VAE
    print("\nDecoding latents with VAE...")
    with torch.no_grad():
        if MISSING_MODALITY == "phone":
            imputed_signals = vae.phone.decoder(imputed_latents)
            ground_truth_signals = vae.phone.decoder(ground_truth_latents.to(DEVICE))
        elif MISSING_MODALITY == "watch":
            imputed_signals = vae.watch.decoder(imputed_latents)
            ground_truth_signals = vae.watch.decoder(ground_truth_latents.to(DEVICE))
        elif MISSING_MODALITY == "glasses":
            imputed_signals = vae.glasses.decoder(imputed_latents)
            ground_truth_signals = vae.glasses.decoder(ground_truth_latents.to(DEVICE))

    imputed_signals = imputed_signals.cpu()
    ground_truth_signals = ground_truth_signals.cpu()

    # Compute metrics
    print("\nComputing metrics...")
    latent_mse = F.mse_loss(imputed_latents.cpu(), ground_truth_latents.cpu()).item()
    signal_mse = F.mse_loss(imputed_signals, ground_truth_signals).item()

    print(f"\n{'='*60}")
    print(f"RESULTS (JOINT MODEL):")
    print(f"  Latent MSE:  {latent_mse:.6f}")
    print(f"  Signal MSE:  {signal_mse:.6f}")
    print(f"{'='*60}\n")

    # Visualize
    print("Creating visualizations...")
    num_plot_samples = min(5, NUM_EVAL_SAMPLES)

    for sample_idx in range(num_plot_samples):
        fig, axs = plt.subplots(imputed_signals.shape[2], 1, figsize=(12, 3 * imputed_signals.shape[2]))

        if imputed_signals.shape[2] == 1:
            axs = [axs]

        for ch in range(imputed_signals.shape[2]):
            real = ground_truth_signals[sample_idx, :, ch].numpy()
            imputed = imputed_signals[sample_idx, :, ch].numpy()

            axs[ch].plot(real, label="Real", alpha=0.8, linewidth=1.5)
            axs[ch].plot(imputed, label="Imputed (Joint)", alpha=0.8, linewidth=1.5, linestyle='--')
            axs[ch].set_title(f"{MISSING_MODALITY.upper()} - Channel {ch} (Joint Model)")
            axs[ch].legend()
            axs[ch].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{MISSING_MODALITY}_joint_sample_{sample_idx}.png", dpi=150)
        plt.close()

    print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")

    # Save metrics
    metrics_path = OUTPUT_DIR / f"{MISSING_MODALITY}_joint_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Missing Modality: {MISSING_MODALITY}\n")
        f.write(f"Model: Joint Diffusion with Cross-Modal Attention\n")
        f.write(f"Num Samples: {NUM_EVAL_SAMPLES}\n")
        f.write(f"Latent MSE: {latent_mse:.6f}\n")
        f.write(f"Signal MSE: {signal_mse:.6f}\n")

    print(f"✅ Metrics saved to: {metrics_path}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
