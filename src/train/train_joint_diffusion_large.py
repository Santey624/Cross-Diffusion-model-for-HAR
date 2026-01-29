# ============================================================
# Train LARGE Joint Temporal Diffusion Model
# Optimized hyperparameters for better reconstruction quality
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

from src.models.joint_diffusion import create_joint_diffusion_model


# ============================================================
# OPTIMIZED CONFIG FOR LARGE MODEL
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
LATENTS_DIR = Path("data/latents")
PHONE_PATH = LATENTS_DIR / "train_latents_phone_mu.pt"
WATCH_PATH = LATENTS_DIR / "train_latents_watch_mu.pt"
GLASSES_PATH = LATENTS_DIR / "train_latents_glasses_mu.pt"

# Output (separate directory for large model)
OUT_DIR = Path("checkpoints/joint_diffusion_large")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion hyperparameters (MORE STEPS = better quality)
T = 2000          # 2x more diffusion steps
BETA_START = 1e-4
BETA_END = 2e-2

# Training hyperparameters
BATCH_SIZE = 256  # Larger batches (better gradients)
EPOCHS = 500      # Train longer
LR = 1e-4         # Learning rate
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 25   # Save every 25 epochs

# LARGE MODEL hyperparameters
HIDDEN_DIM = 512         # 2x larger hidden dim
NUM_HEADS = 8            # 2x more attention heads
NUM_CONV_BLOCKS = 5      # More temporal processing
NUM_ATTN_BLOCKS = 4      # Deeper cross-modal attention
DROPOUT = 0.1

# AMP
USE_AMP = True

# Training strategy
MODALITY_MASK_PROB = 0.3


# ============================================================
# DIFFUSION SCHEDULE
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


def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sched: dict) -> torch.Tensor:
    """Add noise to clean latent z0 at timestep t"""
    sqrt_ab = sched["sqrt_alpha_bar"][t].view(-1, 1, 1)
    sqrt_omb = sched["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1)
    return sqrt_ab * z0 + sqrt_omb * noise


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Training LARGE Joint Temporal Diffusion Model")
    print(f"{'='*60}")
    print(f"\nModel Config:")
    print(f"  Hidden Dim:      {HIDDEN_DIM}")
    print(f"  Num Heads:       {NUM_HEADS}")
    print(f"  Conv Blocks:     {NUM_CONV_BLOCKS}")
    print(f"  Attn Blocks:     {NUM_ATTN_BLOCKS}")
    print(f"\nTraining Config:")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Batch Size:      {BATCH_SIZE}")
    print(f"  Learning Rate:   {LR}")
    print(f"  Diffusion Steps: {T}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading latents...")
    phone_latents = torch.load(PHONE_PATH)
    watch_latents = torch.load(WATCH_PATH)
    glasses_latents = torch.load(GLASSES_PATH)

    N = phone_latents.shape[0]
    print(f"Phone:   {phone_latents.shape}")
    print(f"Watch:   {watch_latents.shape}")
    print(f"Glasses: {glasses_latents.shape}")
    print(f"Total samples: {N}")

    # Normalize latents
    print("\nNormalizing latents...")
    phone_mean = phone_latents.mean(dim=(0, 2), keepdim=True)
    phone_std = phone_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    phone_norm = (phone_latents - phone_mean) / phone_std

    watch_mean = watch_latents.mean(dim=(0, 2), keepdim=True)
    watch_std = watch_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    watch_norm = (watch_latents - watch_mean) / watch_std

    glasses_mean = glasses_latents.mean(dim=(0, 2), keepdim=True)
    glasses_std = glasses_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    glasses_norm = (glasses_latents - glasses_mean) / glasses_std

    # Save normalization stats
    stats_path = OUT_DIR / "normalization_stats.pt"
    torch.save({
        'phone': {'mean': phone_mean, 'std': phone_std},
        'watch': {'mean': watch_mean, 'std': watch_std},
        'glasses': {'mean': glasses_mean, 'std': glasses_std},
    }, stats_path)
    print(f"Saved normalization stats to: {stats_path}")

    # Create dataset
    ds = TensorDataset(phone_norm, watch_norm, glasses_norm)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    # Create LARGE model
    print("\nCreating LARGE joint diffusion model...")
    model = create_joint_diffusion_model(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_conv_blocks=NUM_CONV_BLOCKS,
        num_attn_blocks=NUM_ATTN_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    print(f"Estimated memory: ~{total_params * 4 / 1e9:.2f} GB")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Diffusion schedule
    sched = make_ddpm_schedule(T, BETA_START, BETA_END, DEVICE)

    # Training loop
    print("\nStarting training...")
    print(f"{'='*60}\n")

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_loss_per_modality = {'phone': 0.0, 'watch': 0.0, 'glasses': 0.0}
        count_per_modality = {'phone': 0, 'watch': 0, 'glasses': 0}

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}")
        for phone_batch, watch_batch, glasses_batch in pbar:
            phone_batch = phone_batch.to(DEVICE)
            watch_batch = watch_batch.to(DEVICE)
            glasses_batch = glasses_batch.to(DEVICE)

            B = phone_batch.shape[0]

            # Randomly select target modality
            target_modality = random.choice(['phone', 'watch', 'glasses'])

            # Get target latent
            if target_modality == 'phone':
                z0 = phone_batch
            elif target_modality == 'watch':
                z0 = watch_batch
            else:
                z0 = glasses_batch

            # Sample timesteps
            t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.long)

            # Sample noise
            noise = torch.randn_like(z0)

            # Add noise
            z_t = q_sample(z0, t, noise, sched)

            # Random masking (same logic as before)
            use_phone = (target_modality != 'phone') and (random.random() > MODALITY_MASK_PROB or (target_modality == 'watch' and random.random() > 0.5) or (target_modality == 'glasses' and random.random() > 0.5))
            use_watch = (target_modality != 'watch') and (random.random() > MODALITY_MASK_PROB or (target_modality == 'phone' and random.random() > 0.5) or (target_modality == 'glasses' and random.random() > 0.5))
            use_glasses = (target_modality != 'glasses') and (random.random() > MODALITY_MASK_PROB or (target_modality == 'phone' and random.random() > 0.5) or (target_modality == 'watch' and random.random() > 0.5))

            # Ensure at least one condition
            if not (use_phone or use_watch or use_glasses):
                available = [m for m in ['phone', 'watch', 'glasses'] if m != target_modality]
                forced = random.choice(available)
                if forced == 'phone':
                    use_phone = True
                elif forced == 'watch':
                    use_watch = True
                else:
                    use_glasses = True

            # Prepare conditions
            phone_cond = phone_batch if use_phone else None
            watch_cond = watch_batch if use_watch else None
            glasses_cond = glasses_batch if use_glasses else None

            # Forward pass
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                noise_pred = model(
                    target_modality=target_modality,
                    z_t=z_t,
                    t=t,
                    phone_latent=phone_cond,
                    watch_latent=watch_cond,
                    glasses_latent=glasses_cond,
                )
                loss = F.mse_loss(noise_pred, noise)

            # Backward
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            # Track loss
            running_loss += loss.item()
            running_loss_per_modality[target_modality] += loss.item()
            count_per_modality[target_modality] += 1

            pbar.set_postfix({'loss': f"{loss.item():.6f}", 'target': target_modality})

        # Epoch summary
        avg_loss = running_loss / len(dl)
        phone_loss = running_loss_per_modality['phone'] / max(count_per_modality['phone'], 1)
        watch_loss = running_loss_per_modality['watch'] / max(count_per_modality['watch'], 1)
        glasses_loss = running_loss_per_modality['glasses'] / max(count_per_modality['glasses'], 1)

        print(f"\nEpoch {epoch:03d} | loss={avg_loss:.6f}")
        print(f"  Phone:   {phone_loss:.6f}")
        print(f"  Watch:   {watch_loss:.6f}")
        print(f"  Glasses: {glasses_loss:.6f}")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = OUT_DIR / "joint_ddpm_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'loss': avg_loss,
                'T': T,
                'beta_start': BETA_START,
                'beta_end': BETA_END,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_heads': NUM_HEADS,
                    'num_conv_blocks': NUM_CONV_BLOCKS,
                    'num_attn_blocks': NUM_ATTN_BLOCKS,
                    'dropout': DROPOUT,
                }
            }, best_path)
            print(f"  ✅ New best model! Saved to: {best_path}")

        print()

        # Regular checkpoints
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt_path = OUT_DIR / f"joint_ddpm_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'T': T,
                'beta_start': BETA_START,
                'beta_end': BETA_END,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_heads': NUM_HEADS,
                    'num_conv_blocks': NUM_CONV_BLOCKS,
                    'num_attn_blocks': NUM_ATTN_BLOCKS,
                    'dropout': DROPOUT,
                }
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}\n")

    print(f"\n{'='*60}")
    print(f"✅ LARGE joint diffusion training finished!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
