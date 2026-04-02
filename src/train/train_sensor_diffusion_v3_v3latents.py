# ============================================================
# Train Sensor Conditional Diffusion V3 on VAE V3 Latents
# Input latent shape: (B, 7, 16, 64)
# VAE V3: strong alignment (align_weight=1.0, full sequence + cosine)
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import math

from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.models.sensor_vae import SENSOR_NAMES


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COGAGE_LATENTS_DIR = Path("data/sensor_latents_v3")
WISDM_LATENTS_DIR  = Path("data/wisdm_latents_v3")
OUT_DIR            = Path("checkpoints/sensor_diffusion_v3_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WISDM_REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}

# Diffusion
T        = 1000
SCHEDULE = "cosine"

# Model
D_MODEL    = 192
NUM_HEADS  = 4
NUM_BLOCKS = 4
DROPOUT    = 0.1

# Training
BATCH_SIZE   = 128
LR           = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 0.5
USE_AMP      = True
MIN_SNR_GAMMA = 5.0
RECON_LOSS_WEIGHT = 0.05
FFT_LOSS_WEIGHT   = 0.05

# Phase 1: Pre-train on WISDM + CogAge
PRETRAIN_EPOCHS = 300

# Phase 2: Fine-tune on CogAge only
FINETUNE_EPOCHS = 200
FINETUNE_LR     = 5e-5

MASK_MIN = 1
MASK_MAX = 6


# ============================================================
# SCHEDULE
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-6, max=0.999).float()


def make_schedule(T, schedule_type, device):
    betas = cosine_beta_schedule(T).to(device) if schedule_type == "cosine" \
            else torch.linspace(1e-4, 0.02, T, device=device)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "alpha_bar":                alpha_bar,
        "sqrt_alpha_bar":           torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "snr":                      alpha_bar / (1.0 - alpha_bar),
    }


def generate_random_masks(B, K, mask_min, mask_max, device):
    masks = torch.ones(B, K, device=device)
    for i in range(B):
        n_missing = random.randint(mask_min, mask_max)
        for idx in random.sample(range(K), n_missing):
            masks[i, idx] = 0.0
    return masks


# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, loader, opt, scaler, sched, epoch, total_epochs, device):
    model.train()
    epoch_loss = 0.0
    K = len(SENSOR_NAMES)

    sqrt_ab   = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]
    snr       = sched["snr"]

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for (batch_stacked,) in pbar:
        batch_stacked = batch_stacked.to(device)
        B     = batch_stacked.shape[0]
        D_lat = batch_stacked.shape[2]
        T_lat = batch_stacked.shape[3]

        observed_mask = generate_random_masks(B, K, MASK_MIN, MASK_MAX, device)
        missing_mask  = 1.0 - observed_mask

        t     = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(batch_stacked)

        z0  = batch_stacked
        z_t = (sqrt_ab[t].view(-1, 1, 1, 1) * z0
               + sqrt_1_ab[t].view(-1, 1, 1, 1) * noise)

        noisy_input = (observed_mask[:, :, None, None] * z0
                       + missing_mask[:, :, None, None] * z_t)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            noise_pred = model(noisy_input, t, observed_mask)

            n_miss = missing_mask.sum(dim=1, keepdim=True).clamp_min(1)
            per_s  = ((noise_pred - noise) ** 2 * missing_mask[:, :, None, None])
            per_s  = per_s.sum(dim=(1, 2, 3)) / (n_miss.squeeze() * D_lat * T_lat)

            weight     = torch.clamp(snr[t], max=MIN_SNR_GAMMA) / snr[t]
            noise_loss = (weight * per_s).mean()

            ab_t      = sched["alpha_bar"][t].view(-1, 1, 1, 1).to(device)
            low_noise = (ab_t.squeeze() > 0.1)
            if low_noise.any():
                ab_t_ln   = ab_t[low_noise]
                z_t_ln    = z_t[low_noise]
                z0_ln     = z0[low_noise]
                np_ln     = noise_pred[low_noise]
                mm_ln     = missing_mask[low_noise]
                nm_ln     = mm_ln.sum(dim=1, keepdim=True).clamp_min(1)
                pred_x0   = (z_t_ln - torch.sqrt(1 - ab_t_ln) * np_ln) / torch.sqrt(ab_t_ln)
                pred_x0   = torch.clamp(pred_x0, -10.0, 10.0)
                recon_err  = ((pred_x0 - z0_ln) ** 2 * mm_ln[:, :, None, None])
                recon_loss = (recon_err.sum(dim=(1, 2, 3))
                              / (nm_ln.squeeze() * D_lat * T_lat)).mean()

                pred_fft  = torch.fft.rfft(pred_x0, dim=-1).abs()
                real_fft  = torch.fft.rfft(z0_ln,   dim=-1).abs()
                fft_err   = ((pred_fft - real_fft) ** 2 * mm_ln[:, :, None, None])
                fft_loss  = (fft_err.sum(dim=(1, 2, 3))
                             / (nm_ln.squeeze() * D_lat * (T_lat // 2 + 1))).mean()

                loss = noise_loss + RECON_LOSS_WEIGHT * recon_loss + FFT_LOSS_WEIGHT * fft_loss
            else:
                loss = noise_loss

        if torch.isnan(loss):
            opt.zero_grad(set_to_none=True)
            continue

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
    print("Training Diffusion V3 on VAE V3 Latents (D=16, T=64)")
    print(f"Phase 1: Pre-train {PRETRAIN_EPOCHS} epochs (WISDM + CogAge)")
    print(f"Phase 2: Fine-tune {FINETUNE_EPOCHS} epochs (CogAge only)")
    print(f"{'='*70}\n")

    # Load CogAge V3 latents
    print("Loading CogAge V3 latents...")
    cogage = {}
    for name in SENSOR_NAMES:
        cogage[name] = torch.load(COGAGE_LATENTS_DIR / f"training_latents_{name}_mu.pt")
        print(f"  {name:15s}: {cogage[name].shape}")

    N_cogage = cogage[SENSOR_NAMES[0]].shape[0]

    # Load WISDM V3 latents
    print("\nLoading WISDM V3 latents...")
    wisdm = {}
    for name in SENSOR_NAMES:
        p = WISDM_LATENTS_DIR / f"train_latents_{name}_mu.pt"
        if not p.exists():
            print(f"  {name}: NOT FOUND — skipping WISDM")
            return
        wisdm[name] = torch.load(p)
        marker = "REAL" if name in WISDM_REAL_SENSORS else "ZERO"
        print(f"  {name:15s}: {wisdm[name].shape} [{marker}]")

    N_wisdm = wisdm[SENSOR_NAMES[0]].shape[0]
    print(f"\nCogAge: {N_cogage}, WISDM: {N_wisdm}, Total: {N_cogage + N_wisdm}")

    # Normalize using CogAge stats
    print("\nNormalizing (CogAge stats)...")
    norm_stats = {}
    for name in SENSOR_NAMES:
        z    = cogage[name]
        mean = z.mean(dim=(0, 2), keepdim=True)
        std  = z.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        norm_stats[name] = {"mean": mean, "std": std}
    torch.save(norm_stats, OUT_DIR / "normalization_stats.pt")

    cogage_norm = {n: (cogage[n] - norm_stats[n]["mean"]) / norm_stats[n]["std"]
                   for n in SENSOR_NAMES}
    wisdm_norm  = {n: (wisdm[n]  - norm_stats[n]["mean"]) / norm_stats[n]["std"]
                   for n in SENSOR_NAMES}

    cogage_stacked = torch.stack([cogage_norm[n] for n in SENSOR_NAMES], dim=1)
    wisdm_stacked  = torch.stack([wisdm_norm[n]  for n in SENSOR_NAMES], dim=1)
    combined       = torch.cat([cogage_stacked, wisdm_stacked], dim=0)

    print(f"CogAge stacked : {cogage_stacked.shape}")
    print(f"WISDM stacked  : {wisdm_stacked.shape}")

    # Datasets
    combined_dl = DataLoader(TensorDataset(combined), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    cogage_dl   = DataLoader(TensorDataset(cogage_stacked), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    latent_dim = cogage_stacked.shape[2]  # 16
    print(f"\nCreating Diffusion V3 (latent_dim={latent_dim}, d_model={D_MODEL})...")
    model = create_sensor_diffusion_v3(
        n_sensors=K, latent_dim=latent_dim,
        d_model=D_MODEL, num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS, dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    sched = make_schedule(T, SCHEDULE, DEVICE)

    def save_ckpt(path, phase, loss):
        torch.save({
            "phase": phase, "model_state": model.state_dict(),
            "loss": loss, "T": T, "schedule": SCHEDULE,
            "version": "v3_conditional_v3latents",
            "config": {"d_model": D_MODEL, "num_heads": NUM_HEADS,
                       "num_blocks": NUM_BLOCKS, "dropout": DROPOUT,
                       "latent_dim": latent_dim},
        }, path)

    # -------- Phase 1: Pre-train --------
    print(f"\n{'='*70}")
    print(f"PHASE 1: Pre-training ({PRETRAIN_EPOCHS} epochs, {len(combined)} samples)")
    print(f"{'='*70}\n")

    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))
    best_pt = float('inf')

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        loss = train_epoch(model, combined_dl, opt, scaler, sched, epoch, PRETRAIN_EPOCHS, DEVICE)
        lr_sch.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"[PT] Epoch {epoch:3d}/{PRETRAIN_EPOCHS} | Loss: {loss:.4f} | LR: {lr_sch.get_last_lr()[0]:.2e}")
        if loss < best_pt:
            best_pt = loss
            save_ckpt(OUT_DIR / "pretrain_best.pt", "pretrain", loss)

    print(f"\nPre-training done. Best: {best_pt:.6f}")

    # -------- Phase 2: Fine-tune --------
    print(f"\n{'='*70}")
    print(f"PHASE 2: Fine-tuning ({FINETUNE_EPOCHS} epochs, {N_cogage} CogAge samples)")
    print(f"{'='*70}\n")

    opt_ft    = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    lr_sch_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=FINETUNE_EPOCHS)
    scaler_ft = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))
    best_ft   = float('inf')

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        loss = train_epoch(model, cogage_dl, opt_ft, scaler_ft, sched, epoch, FINETUNE_EPOCHS, DEVICE)
        lr_sch_ft.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"[FT] Epoch {epoch:3d}/{FINETUNE_EPOCHS} | Loss: {loss:.4f} | LR: {lr_sch_ft.get_last_lr()[0]:.2e}")
        if loss < best_ft:
            best_ft = loss
            save_ckpt(OUT_DIR / "best_model.pt", "finetune", loss)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Pre-train best: {best_pt:.6f}")
    print(f"Fine-tune best: {best_ft:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
