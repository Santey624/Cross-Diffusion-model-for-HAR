# ============================================================
# Train Sensor Diffusion V2
# 2D Attention + Random Multi-Sensor Masking
# Trains on pre-extracted latents (fast)
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import math
import numpy as np

from src.models.sensor_joint_diffusion_v2 import create_sensor_diffusion_v2
from src.models.sensor_vae import SENSOR_NAMES


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENTS_DIR = Path("data/sensor_latents")
OUT_DIR = Path("checkpoints/sensor_diffusion_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion
T = 1000
SCHEDULE = "cosine"

# Model (smaller than V1 — less overfitting with small data)
D_MODEL = 128
NUM_HEADS = 4
NUM_BLOCKS = 4
DROPOUT = 0.1

# Training
BATCH_SIZE = 128
EPOCHS = 500
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

# Min-SNR loss weighting
MIN_SNR_GAMMA = 5.0

# Masking strategy: how many sensors to mask per sample
# During training, randomly mask 1-4 sensors (out of 7)
MASK_MIN = 1
MASK_MAX = 4


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


def make_schedule(T, schedule_type, device):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T).to(device)
    else:
        betas = torch.linspace(1e-4, 0.02, T, device=device)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "snr": alpha_bar / (1.0 - alpha_bar),
    }


# ============================================================
# RANDOM MASKING
# ============================================================
def generate_random_masks(B, K, mask_min, mask_max, device):
    """
    Generate random observed masks.

    Returns:
        observed_mask: (B, K) — 1.0 if observed, 0.0 if missing
    """
    masks = torch.ones(B, K, device=device)
    for i in range(B):
        n_missing = random.randint(mask_min, mask_max)
        missing_idx = random.sample(range(K), n_missing)
        masks[i, missing_idx] = 0.0
    return masks


# ============================================================
# MAIN
# ============================================================
def main():
    K = len(SENSOR_NAMES)

    print(f"\n{'='*60}")
    print("Training Sensor Diffusion V2")
    print(f"{'='*60}")
    print(f"2D Attention (Temporal + Feature/Cross-Sensor)")
    print(f"Random multi-sensor masking: {MASK_MIN}-{MASK_MAX} of {K}")
    print(f"Schedule: {SCHEDULE}, T: {T}")
    print(f"d_model: {D_MODEL}, blocks: {NUM_BLOCKS}, heads: {NUM_HEADS}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading sensor latents...")
    latents = {}
    for name in SENSOR_NAMES:
        path = LATENTS_DIR / f"train_latents_{name}_mu.pt"
        latents[name] = torch.load(path)
        print(f"  {name:15s}: {latents[name].shape}")

    N = latents[SENSOR_NAMES[0]].shape[0]
    D = latents[SENSOR_NAMES[0]].shape[1]
    T_shared = latents[SENSOR_NAMES[0]].shape[2]
    print(f"Samples: {N}, Latent dim: {D}, T_shared: {T_shared}")

    # Normalize per sensor
    print("\nNormalizing latents...")
    norm_stats = {}
    latents_norm = {}
    for name in SENSOR_NAMES:
        z = latents[name]
        mean = z.mean(dim=(0, 2), keepdim=True)
        std = z.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        latents_norm[name] = (z - mean) / std
        norm_stats[name] = {"mean": mean, "std": std}

    torch.save(norm_stats, OUT_DIR / "normalization_stats.pt")

    # Stack all latents: (N, K, D, T)
    stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)
    print(f"Stacked tensor: {stacked.shape}")

    ds = TensorDataset(stacked)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    pin_memory=True, num_workers=4)

    # Model
    print("\nCreating V2 model...")
    model = create_sensor_diffusion_v2(
        n_sensors=K,
        latent_dim=D,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Diffusion schedule
    sched = make_schedule(T, SCHEDULE, DEVICE)
    sqrt_ab = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]
    snr = sched["snr"]

    best_loss = float('inf')

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for (batch_stacked,) in pbar:
            batch_stacked = batch_stacked.to(DEVICE)  # (B, K, D, T)
            B = batch_stacked.shape[0]

            # Generate random masks
            observed_mask = generate_random_masks(B, K, MASK_MIN, MASK_MAX, DEVICE)
            missing_mask = 1.0 - observed_mask  # (B, K) — 1 where missing

            # Count missing per sample for logging
            n_missing_per_batch = missing_mask.sum(dim=1)  # (B,)

            # Sample timestep and noise
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(batch_stacked)  # (B, K, D, T)

            # Create noisy input:
            # - Observed sensors: keep clean latents (no noise)
            # - Missing sensors: add diffusion noise
            z0 = batch_stacked
            z_t = sqrt_ab[t].view(-1, 1, 1, 1) * z0 + sqrt_1_ab[t].view(-1, 1, 1, 1) * noise

            # Mix: observed stays clean, missing gets noised
            noisy_input = (
                observed_mask[:, :, None, None] * z0 +
                missing_mask[:, :, None, None] * z_t
            )

            # Forward
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                noise_pred = model(noisy_input, t, observed_mask)  # (B, K, D, T)

                # Loss: only on MISSING sensors
                # (B, K, D, T) * (B, K, 1, 1) -> masked
                per_sample_loss = ((noise_pred - noise) ** 2 * missing_mask[:, :, None, None])

                # Normalize by number of missing sensors per sample
                n_missing_per_sample = missing_mask.sum(dim=1, keepdim=True).clamp_min(1)  # (B, 1)
                per_sample_loss = per_sample_loss.sum(dim=(1, 2, 3)) / (n_missing_per_sample.squeeze() * D * T_shared)

                # Min-SNR weighting
                snr_t = snr[t]
                weight = torch.clamp(snr_t, max=MIN_SNR_GAMMA) / snr_t
                loss = (weight * per_sample_loss).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        lr_sched.step()
        n_batches = len(dl)
        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            lr = lr_sched.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'T': T,
                'schedule': SCHEDULE,
                'version': 'v2',
                'config': {
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'num_blocks': NUM_BLOCKS,
                    'dropout': DROPOUT,
                    'mask_min': MASK_MIN,
                    'mask_max': MASK_MAX,
                },
            }, OUT_DIR / "best_model.pt")

        # Periodic save
        if epoch % 100 == 0 or epoch == EPOCHS:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'T': T,
                'schedule': SCHEDULE,
                'version': 'v2',
                'config': {
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'num_blocks': NUM_BLOCKS,
                    'dropout': DROPOUT,
                    'mask_min': MASK_MIN,
                    'mask_max': MASK_MAX,
                },
            }, OUT_DIR / f"epoch_{epoch:03d}.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
