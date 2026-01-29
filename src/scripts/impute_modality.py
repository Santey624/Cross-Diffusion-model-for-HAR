# ============================================================
# Modality Imputation via Conditional Temporal Diffusion
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.temporal_diffusion import (
    create_phone_denoiser,
    create_watch_denoiser,
    create_glasses_denoiser,
)


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
CHECKPOINT_DIR = Path("checkpoints/temporal_diffusion")
LATENTS_DIR = Path("data/latents")
OUTPUT_DIR = Path("data/imputed_latents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Which modality to impute
MISSING_MODALITY = "glasses"  # "phone", "watch", or "glasses"

# Checkpoint epoch to use
EPOCH = 200

# Sampling parameters
NUM_SAMPLES = 100  # How many samples to impute
DDIM_STEPS = 50  # Use DDIM for faster sampling (set to None for full DDPM)


# ============================================================
# DDPM SCHEDULE
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


# ============================================================
# DDPM SAMPLING
# ============================================================
@torch.no_grad()
def ddpm_sample(
    model: torch.nn.Module,
    shape: tuple,
    conditions: list[torch.Tensor],
    sched: dict,
    T: int,
    ddim_steps: int = None,
) -> torch.Tensor:
    """
    Sample from diffusion model using DDPM or DDIM

    Args:
        model: trained ConditionalTemporalDenoiser
        shape: (B, D, T) shape of target modality
        conditions: list of condition latents
        sched: diffusion schedule
        T: total diffusion steps
        ddim_steps: if not None, use DDIM with this many steps

    Returns:
        sampled latent (B, D, T)
    """
    B, D, seq_len = shape
    device = next(model.parameters()).device

    # Start from pure noise
    z = torch.randn(shape, device=device)

    # DDIM: use subset of timesteps
    if ddim_steps is not None:
        timesteps = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, dtype=torch.long, device=device)

    # Reverse diffusion process
    for i, t_val in enumerate(tqdm(timesteps, desc="Sampling")):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = model(z, t, conditions)

        # DDPM update
        if ddim_steps is None:
            # Standard DDPM
            alpha = sched["alphas"][t_val].view(-1, 1, 1)
            alpha_bar = sched["alpha_bar"][t_val].view(-1, 1, 1)
            beta = sched["betas"][t_val].view(-1, 1, 1)

            if t_val > 0:
                noise = torch.randn_like(z)
            else:
                noise = 0.0

            # Predict x0
            sqrt_recip_alpha_bar = 1.0 / torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
            pred_x0 = sqrt_recip_alpha_bar * (z - sqrt_one_minus_alpha_bar * noise_pred)

            # Clip (optional, helps stability)
            pred_x0 = torch.clamp(pred_x0, -3, 3)

            # Posterior mean
            alpha_bar_prev = sched["alpha_bar"][t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
            alpha_bar_prev = alpha_bar_prev.view(-1, 1, 1)

            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar) * pred_x0 +
                torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * z
            )

            posterior_variance = sched["posterior_variance"][t_val].view(-1, 1, 1)
            z = posterior_mean + torch.sqrt(posterior_variance) * noise

        else:
            # DDIM update (deterministic, faster)
            alpha_bar_t = sched["alpha_bar"][t_val].view(-1, 1, 1)

            if i < len(timesteps) - 1:
                alpha_bar_prev = sched["alpha_bar"][timesteps[i + 1]].view(-1, 1, 1)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device).view(-1, 1, 1)

            # Predict x0
            pred_x0 = (z - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)

            # Direction pointing to z_t
            dir_zt = torch.sqrt(1 - alpha_bar_prev) * noise_pred

            # Next sample
            z = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_zt

    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"Imputing missing modality: {MISSING_MODALITY.upper()}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading latents...")
    phone_latents = torch.load(LATENTS_DIR / "train_latents_phone_mu.pt")
    watch_latents = torch.load(LATENTS_DIR / "train_latents_watch_mu.pt")
    glasses_latents = torch.load(LATENTS_DIR / "train_latents_glasses_mu.pt")

    # Select first NUM_SAMPLES
    phone_latents = phone_latents[:NUM_SAMPLES]
    watch_latents = watch_latents[:NUM_SAMPLES]
    glasses_latents = glasses_latents[:NUM_SAMPLES]

    # Determine target and conditions based on missing modality
    if MISSING_MODALITY == "phone":
        model = create_phone_denoiser().to(DEVICE)
        ckpt_path = CHECKPOINT_DIR / f"phone_ddpm_epoch_{EPOCH:03d}.pt"
        stats_path = CHECKPOINT_DIR / "phone_latent_stats.pt"
        target_shape = phone_latents.shape
        conditions_raw = [watch_latents, glasses_latents]
        ground_truth = phone_latents

    elif MISSING_MODALITY == "watch":
        model = create_watch_denoiser().to(DEVICE)
        ckpt_path = CHECKPOINT_DIR / f"watch_ddpm_epoch_{EPOCH:03d}.pt"
        stats_path = CHECKPOINT_DIR / "watch_latent_stats.pt"
        target_shape = watch_latents.shape
        conditions_raw = [phone_latents, glasses_latents]
        ground_truth = watch_latents

    elif MISSING_MODALITY == "glasses":
        model = create_glasses_denoiser().to(DEVICE)
        ckpt_path = CHECKPOINT_DIR / f"glasses_ddpm_epoch_{EPOCH:03d}.pt"
        stats_path = CHECKPOINT_DIR / "glasses_latent_stats.pt"
        target_shape = glasses_latents.shape
        conditions_raw = [phone_latents, watch_latents]
        ground_truth = glasses_latents

    else:
        raise ValueError(f"Unknown modality: {MISSING_MODALITY}")

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load normalization stats
    print(f"Loading normalization stats: {stats_path}")
    stats = torch.load(stats_path, map_location=DEVICE)

    # Normalize conditions
    conditions = []
    for i, cond in enumerate(conditions_raw):
        cond_mean = stats["condition_stats"][i]["mean"]
        cond_std = stats["condition_stats"][i]["std"]
        cond_norm = (cond - cond_mean) / cond_std
        conditions.append(cond_norm.to(DEVICE))

    # Create diffusion schedule
    T = ckpt["T"]
    beta_start = ckpt["beta_start"]
    beta_end = ckpt["beta_end"]
    sched = make_ddpm_schedule(T, beta_start, beta_end, DEVICE)

    # Sample imputed latents
    print(f"\nSampling imputed {MISSING_MODALITY} latents...")
    print(f"Shape: {target_shape}")
    print(f"Using {'DDIM' if DDIM_STEPS else 'DDPM'} sampling")

    imputed_norm = ddpm_sample(
        model=model,
        shape=target_shape,
        conditions=conditions,
        sched=sched,
        T=T,
        ddim_steps=DDIM_STEPS,
    )

    # Denormalize
    target_mean = stats["target_mean"]
    target_std = stats["target_std"]
    imputed = imputed_norm * target_std + target_mean

    # Save imputed latents
    output_path = OUTPUT_DIR / f"imputed_{MISSING_MODALITY}_latents.pt"
    torch.save({
        "imputed": imputed.cpu(),
        "ground_truth": ground_truth.cpu(),
    }, output_path)

    print(f"\n✅ Saved imputed latents to: {output_path}")

    # Compute reconstruction error
    mse = F.mse_loss(imputed.cpu(), ground_truth).item()
    print(f"\nReconstruction MSE: {mse:.6f}")

    print(f"\n{'='*60}")
    print("Imputation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
