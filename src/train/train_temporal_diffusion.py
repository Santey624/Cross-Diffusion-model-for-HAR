# ============================================================
# Train Conditional Temporal Diffusion Models
# For modality imputation in VAE latent space
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

# Paths to temporal latents
LATENTS_DIR = Path("data/latents")
PHONE_PATH = LATENTS_DIR / "train_latents_phone_mu.pt"
WATCH_PATH = LATENTS_DIR / "train_latents_watch_mu.pt"
GLASSES_PATH = LATENTS_DIR / "train_latents_glasses_mu.pt"

# Output directory
OUT_DIR = Path("checkpoints/temporal_diffusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion hyperparameters
T = 1000
BETA_START = 1e-4
BETA_END = 2e-2

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 200
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 10

# Model hyperparameters
HIDDEN_DIM = 128
NUM_BLOCKS = 4
DROPOUT = 0.1

# AMP
USE_AMP = True

# Which modality to train (can be "phone", "watch", "glasses", or "all")
TRAIN_MODALITY = "all"  # Change to "phone", "watch", or "glasses" for individual training


# ============================================================
# DIFFUSION SCHEDULE (DDPM)
# ============================================================
def make_ddpm_schedule(T: int, beta_start: float, beta_end: float, device: str):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # For sampling
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


# q_sample: add noise
def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sched: dict) -> torch.Tensor:
    """
    Add noise to clean latent z0 at timestep t
    z0: (B, D, T)
    t: (B,)
    noise: (B, D, T)
    returns: (B, D, T)
    """
    # Gather per-sample coefficients
    sqrt_ab = sched["sqrt_alpha_bar"][t].view(-1, 1, 1)  # (B, 1, 1)
    sqrt_omb = sched["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1)  # (B, 1, 1)
    return sqrt_ab * z0 + sqrt_omb * noise


# ============================================================
# TRAIN ONE MODALITY
# ============================================================
def train_modality(
    modality_name: str,
    model: torch.nn.Module,
    target_latents: torch.Tensor,
    condition_latents: list[torch.Tensor],
    sched: dict,
    epochs: int,
):
    """
    Train one conditional diffusion model

    Args:
        modality_name: "phone", "watch", or "glasses"
        model: ConditionalTemporalDenoiser
        target_latents: (N, D_target, T_target)
        condition_latents: list of [(N, D1, T1), (N, D2, T2)]
        sched: diffusion schedule dict
        epochs: number of training epochs
    """
    print(f"\n{'='*60}")
    print(f"Training diffusion model for: {modality_name.upper()}")
    print(f"Target shape: {target_latents.shape}")
    print(f"Condition shapes: {[c.shape for c in condition_latents]}")
    print(f"{'='*60}\n")

    # Normalize latents (improves diffusion stability)
    target_mean = target_latents.mean(dim=(0, 2), keepdim=True)
    target_std = target_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    target_norm = (target_latents - target_mean) / target_std

    condition_norms = []
    condition_stats = []
    for cond in condition_latents:
        cond_mean = cond.mean(dim=(0, 2), keepdim=True)
        cond_std = cond.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        cond_norm = (cond - cond_mean) / cond_std
        condition_norms.append(cond_norm)
        condition_stats.append({"mean": cond_mean, "std": cond_std})

    # Save normalization stats
    stats_path = OUT_DIR / f"{modality_name}_latent_stats.pt"
    torch.save({
        "target_mean": target_mean,
        "target_std": target_std,
        "condition_stats": condition_stats,
    }, stats_path)
    print(f"Saved normalization stats to: {stats_path}")

    # Create dataset
    ds = TensorDataset(target_norm, *condition_norms)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dl, desc=f"[{modality_name.upper()}] Epoch {epoch}/{epochs}")
        for batch in pbar:
            z0 = batch[0].to(DEVICE, non_blocking=True)  # target
            conds = [c.to(DEVICE, non_blocking=True) for c in batch[1:]]  # conditions

            # Sample random timesteps
            t = torch.randint(0, T, (z0.size(0),), device=DEVICE, dtype=torch.long)

            # Sample noise
            noise = torch.randn_like(z0)

            # Add noise to target
            z_t = q_sample(z0, t, noise, sched)

            # Predict noise
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                noise_pred = model(z_t, t, conds)
                loss = F.mse_loss(noise_pred, noise)

            # Backward
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = running_loss / len(dl)
        print(f"\n[{modality_name.upper()}] Epoch {epoch:03d} | loss={avg_loss:.6f}\n")

        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == epochs:
            ckpt_path = OUT_DIR / f"{modality_name}_ddpm_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "T": T,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
                "modality": modality_name,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print(f"\n✅ Finished training {modality_name.upper()} diffusion model\n")


# ============================================================
# MAIN
# ============================================================
def main():
    # Load latents
    print("Loading temporal latents...")
    phone_latents = torch.load(PHONE_PATH)    # (N, 32, 100)
    watch_latents = torch.load(WATCH_PATH)    # (N, 32, 34)
    glasses_latents = torch.load(GLASSES_PATH)  # (N, 16, 10)

    print(f"Phone:   {phone_latents.shape}")
    print(f"Watch:   {watch_latents.shape}")
    print(f"Glasses: {glasses_latents.shape}")

    # Create diffusion schedule
    sched = make_ddpm_schedule(T, BETA_START, BETA_END, device=DEVICE)

    # Train phone model (conditioned on watch + glasses)
    if TRAIN_MODALITY in ["phone", "all"]:
        phone_model = create_phone_denoiser(
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout=DROPOUT
        ).to(DEVICE)

        train_modality(
            modality_name="phone",
            model=phone_model,
            target_latents=phone_latents,
            condition_latents=[watch_latents, glasses_latents],
            sched=sched,
            epochs=EPOCHS,
        )

    # Train watch model (conditioned on phone + glasses)
    if TRAIN_MODALITY in ["watch", "all"]:
        watch_model = create_watch_denoiser(
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout=DROPOUT
        ).to(DEVICE)

        train_modality(
            modality_name="watch",
            model=watch_model,
            target_latents=watch_latents,
            condition_latents=[phone_latents, glasses_latents],
            sched=sched,
            epochs=EPOCHS,
        )

    # Train glasses model (conditioned on phone + watch)
    if TRAIN_MODALITY in ["glasses", "all"]:
        glasses_model = create_glasses_denoiser(
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout=DROPOUT
        ).to(DEVICE)

        train_modality(
            modality_name="glasses",
            model=glasses_model,
            target_latents=glasses_latents,
            condition_latents=[phone_latents, watch_latents],
            sched=sched,
            epochs=EPOCHS,
        )

    print("\n" + "="*60)
    print("✅ All temporal diffusion models trained successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
