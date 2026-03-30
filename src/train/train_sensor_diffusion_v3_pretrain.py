# ============================================================
# Train Sensor Conditional Diffusion V3 with WISDM Pre-Training
# V3: Proper conditional diffusion — p(z_missing | z_observed)
#     Missing sensors cross-attend to observed sensors (keys/values)
# Phase 1: Pre-train on WISDM + CogAge (more data)
# Phase 2: Fine-tune on CogAge only
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import random
import math
import numpy as np

from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.models.sensor_vae import SENSOR_NAMES


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COGAGE_LATENTS_DIR = Path("data/sensor_latents")
WISDM_LATENTS_DIR = Path("data/wisdm_latents")
OUT_DIR = Path("checkpoints/sensor_diffusion_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensors that have real WISDM data
WISDM_REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}

# Diffusion
T = 1000
SCHEDULE = "cosine"

# Model
D_MODEL = 128
NUM_HEADS = 4
NUM_BLOCKS = 4
DROPOUT = 0.1

# Training
BATCH_SIZE = 128
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True
MIN_SNR_GAMMA = 5.0
RECON_LOSS_WEIGHT = 0.1  # Weight for auxiliary reconstruction loss on pred_x0
FFT_LOSS_WEIGHT = 0.05   # Weight for frequency-domain loss on pred_x0

# Phase 1: Pre-train on WISDM + CogAge
PRETRAIN_EPOCHS = 300

# Phase 2: Fine-tune on CogAge only
FINETUNE_EPOCHS = 200
FINETUNE_LR = 5e-5

# Masking
MASK_MIN = 1
MASK_MAX = 6


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
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "snr": alpha_bar / (1.0 - alpha_bar),
    }


def generate_random_masks(B, K, mask_min, mask_max, device):
    masks = torch.ones(B, K, device=device)
    for i in range(B):
        n_missing = random.randint(mask_min, mask_max)
        missing_idx = random.sample(range(K), n_missing)
        masks[i, missing_idx] = 0.0
    return masks


def generate_wisdm_aware_masks(B, K, sensor_names, wisdm_real, device):
    """
    For WISDM samples: mark zero-filled sensors as 'observed' (mask=1)
    so V3 cross-attention can use all 7 sensors as keys.

    Zero-filled sensors (phone_grav, phone_lacc, glasses_acc) have zero
    latent values — they act as neutral conditioning keys. Loss is NOT
    computed on them (observed sensors are not generation targets).

    Only real WISDM sensors are randomly masked as generation targets,
    so the model learns to impute from observed context including zero-keys.
    This ensures V3 cross-attention sees all sensor slots as potential keys
    during pretraining, matching the CogAge fine-tuning distribution.
    """
    masks = torch.ones(B, K, device=device)

    real_indices = [i for i, name in enumerate(sensor_names) if name in wisdm_real]

    for i in range(B):
        # Only mask 1-2 of the real WISDM sensors as generation targets
        # Zero-filled sensors remain observed (mask=1, zero latent = neutral key)
        n_mask_real = random.randint(1, min(2, len(real_indices)))
        masked_real = random.sample(real_indices, n_mask_real)
        for idx in masked_real:
            masks[i, idx] = 0.0

    return masks


# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, loader, opt, scaler, sched, epoch, total_epochs,
                mask_fn, device):
    model.train()
    epoch_loss = 0.0

    sqrt_ab = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]
    snr = sched["snr"]
    K = len(SENSOR_NAMES)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for (batch_stacked,) in pbar:
        batch_stacked = batch_stacked.to(device)
        B = batch_stacked.shape[0]

        observed_mask = mask_fn(B, K, device)
        missing_mask = 1.0 - observed_mask

        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(batch_stacked)

        z0 = batch_stacked
        z_t = sqrt_ab[t].view(-1, 1, 1, 1) * z0 + sqrt_1_ab[t].view(-1, 1, 1, 1) * noise

        noisy_input = (
            observed_mask[:, :, None, None] * z0 +
            missing_mask[:, :, None, None] * z_t
        )

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            noise_pred = model(noisy_input, t, observed_mask)

            per_sample_loss = ((noise_pred - noise) ** 2 * missing_mask[:, :, None, None])
            n_missing_per_sample = missing_mask.sum(dim=1, keepdim=True).clamp_min(1)
            D = batch_stacked.shape[2]
            T_shared = batch_stacked.shape[3]
            per_sample_loss = per_sample_loss.sum(dim=(1, 2, 3)) / (n_missing_per_sample.squeeze() * D * T_shared)

            snr_t = snr[t]
            weight = torch.clamp(snr_t, max=MIN_SNR_GAMMA) / snr_t
            noise_loss = (weight * per_sample_loss).mean()

            # Auxiliary reconstruction loss: supervise pred_x0 directly on missing sensors
            ab_t = sched["alpha_bar"][t].view(-1, 1, 1, 1).to(device)
            pred_x0 = (z_t - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            recon_err = ((pred_x0 - z0) ** 2 * missing_mask[:, :, None, None])
            recon_loss = recon_err.sum(dim=(1, 2, 3)) / (n_missing_per_sample.squeeze() * D * T_shared)
            recon_loss = recon_loss.mean()

            # FFT loss: match frequency spectrum of pred_x0 to real x0 on missing sensors
            pred_fft = torch.fft.rfft(pred_x0, dim=-1).abs()
            real_fft = torch.fft.rfft(z0, dim=-1).abs()
            fft_err = ((pred_fft - real_fft) ** 2 * missing_mask[:, :, None, None])
            fft_loss = fft_err.sum(dim=(1, 2, 3)) / (n_missing_per_sample.squeeze() * D * (T_shared // 2 + 1))
            fft_loss = fft_loss.mean()

            loss = noise_loss + RECON_LOSS_WEIGHT * recon_loss + FFT_LOSS_WEIGHT * fft_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(loader)


# ============================================================
# MAIN
# ============================================================
def main():
    K = len(SENSOR_NAMES)

    print(f"\n{'='*70}")
    print("Training Sensor Diffusion V3 with WISDM Pre-Training")
    print(f"{'='*70}")
    print(f"Phase 1: Pre-train {PRETRAIN_EPOCHS} epochs (WISDM + CogAge)")
    print(f"Phase 2: Fine-tune {FINETUNE_EPOCHS} epochs (CogAge only)")
    print(f"{'='*70}\n")

    # ========== LOAD COGAGE LATENTS ==========
    print("Loading CogAge latents...")
    cogage_latents = {}
    for name in SENSOR_NAMES:
        path = COGAGE_LATENTS_DIR / f"train_latents_{name}_mu.pt"
        cogage_latents[name] = torch.load(path)
        print(f"  {name:15s}: {cogage_latents[name].shape}")

    N_cogage = cogage_latents[SENSOR_NAMES[0]].shape[0]

    # ========== LOAD WISDM LATENTS ==========
    print("\nLoading WISDM latents...")
    wisdm_latents = {}
    for name in SENSOR_NAMES:
        path = WISDM_LATENTS_DIR / f"train_latents_{name}_mu.pt"
        if path.exists():
            wisdm_latents[name] = torch.load(path)
            marker = "REAL" if name in WISDM_REAL_SENSORS else "ZERO"
            print(f"  {name:15s}: {wisdm_latents[name].shape} [{marker}]")
        else:
            print(f"  {name:15s}: NOT FOUND - skipping WISDM")
            return

    N_wisdm = wisdm_latents[SENSOR_NAMES[0]].shape[0]
    print(f"\nCogAge: {N_cogage}, WISDM: {N_wisdm}, Combined: {N_cogage + N_wisdm}")

    # ========== NORMALIZE ==========
    print("\nNormalizing (using CogAge stats)...")
    # Use CogAge stats for normalization (target domain)
    norm_stats = {}
    for name in SENSOR_NAMES:
        z = cogage_latents[name]
        mean = z.mean(dim=(0, 2), keepdim=True)
        std = z.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        norm_stats[name] = {"mean": mean, "std": std}

    torch.save(norm_stats, OUT_DIR / "normalization_stats.pt")

    # Normalize both datasets
    cogage_norm = {}
    wisdm_norm = {}
    for name in SENSOR_NAMES:
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        cogage_norm[name] = (cogage_latents[name] - mean) / std
        wisdm_norm[name] = (wisdm_latents[name] - mean) / std

    # Stack
    cogage_stacked = torch.stack([cogage_norm[name] for name in SENSOR_NAMES], dim=1)
    wisdm_stacked = torch.stack([wisdm_norm[name] for name in SENSOR_NAMES], dim=1)

    print(f"CogAge stacked: {cogage_stacked.shape}")
    print(f"WISDM stacked: {wisdm_stacked.shape}")

    # ========== CREATE DATASETS ==========
    combined_ds = TensorDataset(torch.cat([cogage_stacked, wisdm_stacked], dim=0))
    cogage_ds = TensorDataset(cogage_stacked)

    # Track which samples are WISDM (for masking)
    is_wisdm = torch.cat([
        torch.zeros(N_cogage, dtype=torch.bool),
        torch.ones(N_wisdm, dtype=torch.bool),
    ])

    combined_dl = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, pin_memory=True, num_workers=4)
    cogage_dl = DataLoader(cogage_ds, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=True, pin_memory=True, num_workers=4)

    # ========== MODEL ==========
    print("\nCreating V3 Conditional Diffusion model...")
    model = create_sensor_diffusion_v3(
        n_sensors=K,
        latent_dim=cogage_stacked.shape[2],
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    sched = make_schedule(T, SCHEDULE, DEVICE)

    # ========== PHASE 1: PRE-TRAIN ==========
    print(f"\n{'='*70}")
    print(f"PHASE 1: Pre-training on WISDM + CogAge ({PRETRAIN_EPOCHS} epochs)")
    print(f"Combined samples: {len(combined_ds)}")
    print(f"{'='*70}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    best_loss = float('inf')

    # For pre-training, use standard random masking
    # (WISDM zero-filled sensors will just be noise anyway)
    def pretrain_mask_fn(B, K, device):
        return generate_random_masks(B, K, MASK_MIN, MASK_MAX, device)

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        avg_loss = train_epoch(
            model, combined_dl, opt, scaler, sched,
            epoch, PRETRAIN_EPOCHS, pretrain_mask_fn, DEVICE,
        )
        lr_sched_pt.step()

        if epoch % 20 == 0 or epoch == 1:
            lr = lr_sched_pt.get_last_lr()[0]
            print(f"[PT] Epoch {epoch:3d}/{PRETRAIN_EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'phase': 'pretrain',
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'T': T,
                'schedule': SCHEDULE,
                'version': 'v3_conditional',
                'config': {
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'num_blocks': NUM_BLOCKS,
                    'dropout': DROPOUT,
                },
            }, OUT_DIR / "pretrain_best.pt")

    print(f"\nPre-training complete. Best loss: {best_loss:.6f}")

    # ========== PHASE 2: FINE-TUNE ==========
    print(f"\n{'='*70}")
    print(f"PHASE 2: Fine-tuning on CogAge only ({FINETUNE_EPOCHS} epochs)")
    print(f"CogAge samples: {len(cogage_ds)}")
    print(f"{'='*70}\n")

    opt_ft = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    lr_sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=FINETUNE_EPOCHS)
    scaler_ft = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    best_ft_loss = float('inf')

    def finetune_mask_fn(B, K, device):
        return generate_random_masks(B, K, MASK_MIN, MASK_MAX, device)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        avg_loss = train_epoch(
            model, cogage_dl, opt_ft, scaler_ft, sched,
            epoch, FINETUNE_EPOCHS, finetune_mask_fn, DEVICE,
        )
        lr_sched_ft.step()

        if epoch % 20 == 0 or epoch == 1:
            lr = lr_sched_ft.get_last_lr()[0]
            print(f"[FT] Epoch {epoch:3d}/{FINETUNE_EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if avg_loss < best_ft_loss:
            best_ft_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'phase': 'finetune',
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'T': T,
                'schedule': SCHEDULE,
                'version': 'v3_conditional',
                'config': {
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'num_blocks': NUM_BLOCKS,
                    'dropout': DROPOUT,
                },
            }, OUT_DIR / "best_model.pt")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Pre-train best: {best_loss:.6f}")
    print(f"Fine-tune best: {best_ft_loss:.6f}")
    print(f"Checkpoints: {OUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
