# ============================================================
# Train Cross-Sensor Signal Diffusion on the Opportunity dataset
#
# Same model & objective as the CogAge cross-sensor diffusion
# (src/models/signal_cross_diffusion.py), but on the 14 triaxial
# body-IMU sensors extracted from Opportunity.
#
# Objective (per masked/missing sensor):
#   x0_pred    = (x_t - sqrt(1-ab)*eps_pred) / sqrt(ab)
#   noise_loss = MSE(eps_pred, eps)
#   recon_loss = MSE(x0_pred, x0)          [normalized signal space]
#   fft_loss   = MSE(|FFT(x0_pred)|, |FFT(x0)|)
#   total      = noise_loss + L_RECON*recon_loss + L_FFT*fft_loss
#
# Prereq: run the preprocessing first to create the .npy arrays:
#   python -m src.data.opportunity.preprocess_opportunity
#
# Usage (from repo root):
#   python -m src.train.trainingonopportunitydataset.train_signal_cross_diffusion_opportunity
# ============================================================

import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.signal_cross_diffusion import (
    create_signal_cross_diffusion, T_COMMON,
)
from src.data.opportunity.opportunity_sensor_dataset import OpportunitySensorDataset
from src.data.opportunity.opportunity_constants import OPP_SENSOR_NAMES, OPP_DEVICE_GROUPS

# Local aliases so the training body reads like the CogAge script
SENSOR_NAMES  = OPP_SENSOR_NAMES
DEVICE_GROUPS = OPP_DEVICE_GROUPS


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPP_ROOT = "data/opportunity/arrays"

OUT_DIR = Path("checkpoints/opportunity_signal_cross_diffusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T          = 1000
SCHEDULE   = "cosine"
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 3e-4

D_MODEL    = 128
NUM_HEADS  = 4
NUM_BLOCKS = 6
DROPOUT    = 0.1

NUM_WORKERS = 4

# Missing-pattern sampling probabilities:
#   P_DEVICE -> drop a whole body location (e.g. both shoes)
#   P_SINGLE -> drop a single sensor
#   rest     -> drop a random 2-3 sensors
P_DEVICE = 0.35
P_SINGLE = 0.40

LAMBDA_RECON = 1.0
LAMBDA_FFT   = 0.001


# ============================================================
# Noise schedule
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float()


# ============================================================
# Realistic missing pattern sampling
# ============================================================
def sample_missing_idx():
    """Returns list of sensor indices to mask."""
    p = random.random()
    if p < P_DEVICE:
        device  = random.choice(list(DEVICE_GROUPS.keys()))
        missing = DEVICE_GROUPS[device]
    elif p < P_DEVICE + P_SINGLE:
        missing = [random.choice(SENSOR_NAMES)]
    else:
        missing = random.sample(SENSOR_NAMES, random.randint(2, 3))
    return [SENSOR_NAMES.index(s) for s in missing]


# ============================================================
# FFT loss
# ============================================================
def fft_loss(pred, target):
    """pred, target: (B, C, T) — compares magnitude spectra along T."""
    pred_mag   = torch.fft.rfft(pred,   dim=-1).abs()
    target_mag = torch.fft.rfft(target, dim=-1).abs()
    return F.mse_loss(pred_mag, target_mag)


# ============================================================
# Signal preparation: stack all sensors, interpolate to T_COMMON
# ============================================================
def stack_signals(batch, device):
    """Returns: (B, K, C, T_COMMON)."""
    parts = []
    for name in SENSOR_NAMES:
        x = batch[name].to(device).permute(0, 2, 1).float()   # (B, C, T)
        x = F.interpolate(x, size=T_COMMON, mode="linear", align_corners=False)
        parts.append(x)
    return torch.stack(parts, dim=1)


# ============================================================
# Normalization stats (per-sensor, per-channel z-score)
# ============================================================
def compute_norm_stats(train_loader, device):
    print("Computing normalization stats...")
    sums  = {k: 0.0 for k in SENSOR_NAMES}
    sqs   = {k: 0.0 for k in SENSOR_NAMES}
    count = 0
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="norm stats", leave=False):
            for name in SENSOR_NAMES:
                x = batch[name].to(device).float()
                sums[name] = sums[name] + x.mean(dim=(0, 1))
                sqs[name]  = sqs[name]  + (x ** 2).mean(dim=(0, 1))
            count += 1
    stats = {}
    for name in SENSOR_NAMES:
        mean = sums[name] / count
        std  = ((sqs[name] / count) - mean ** 2).clamp(min=1e-6).sqrt()
        stats[name] = {"mean": mean.cpu(), "std": std.cpu()}
    return stats


# ============================================================
# One forward pass of the diffusion objective (shared train/eval)
# ============================================================
def diffusion_step(model, stacked, norm_mean, norm_std, alpha_bar):
    stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                 / norm_std[None, :, :, None]
    B = stacked.shape[0]

    missing_idx   = sample_missing_idx()
    observed_mask = torch.ones(B, len(SENSOR_NAMES), device=stacked.device)
    for i in missing_idx:
        observed_mask[:, i] = 0.0

    t_step = torch.randint(0, T, (B,), device=stacked.device)
    noise  = torch.randn_like(stacked_norm)
    ab3    = alpha_bar[t_step][:, None, None]

    noisy = stacked_norm.clone()
    for i in missing_idx:
        noisy[:, i] = (torch.sqrt(ab3) * stacked_norm[:, i]
                       + torch.sqrt(1 - ab3) * noise[:, i])

    noise_pred = model(noisy, t_step, observed_mask)

    noise_loss = sum(
        F.mse_loss(noise_pred[:, i], noise[:, i]) for i in missing_idx
    ) / len(missing_idx)

    recon_loss = torch.tensor(0.0, device=stacked.device)
    freq_loss  = torch.tensor(0.0, device=stacked.device)
    for i in missing_idx:
        x0_pred = ((noisy[:, i] - torch.sqrt(1 - ab3) * noise_pred[:, i])
                   / torch.sqrt(ab3)).clamp(-5, 5)
        recon_loss = recon_loss + F.mse_loss(x0_pred, stacked_norm[:, i])
        freq_loss  = freq_loss  + fft_loss(x0_pred, stacked_norm[:, i])
    recon_loss = recon_loss / len(missing_idx)
    freq_loss  = freq_loss  / len(missing_idx)

    return noise_loss, recon_loss, freq_loss


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training Opportunity Cross-Sensor Signal Diffusion")
    print(f"K_sensors={len(SENSOR_NAMES)}, T={T}, T_COMMON={T_COMMON}, Epochs={EPOCHS}")
    print(f"lambda_recon={LAMBDA_RECON}, lambda_fft={LAMBDA_FFT}")
    print(f"D_model={D_MODEL}, Blocks={NUM_BLOCKS}, Heads={NUM_HEADS}")
    print(f"{'='*65}\n")

    train_ds = OpportunitySensorDataset(OPP_ROOT, "training")
    test_ds  = OpportunitySensorDataset(OPP_ROOT, "testing")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Normalization stats
    norm_stats = compute_norm_stats(train_loader, DEVICE)
    norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
    norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
    torch.save(
        {k: {"mean": norm_stats[k]["mean"], "std": norm_stats[k]["std"]}
         for k in SENSOR_NAMES},
        OUT_DIR / "normalization_stats.pt",
    )

    # Noise schedule
    betas     = cosine_beta_schedule(T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)

    # Model
    print(f"\nCreating Signal Cross-Sensor Diffusion...")
    model = create_signal_cross_diffusion(
        n_sensors=len(SENSOR_NAMES),
        in_channels=3,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_noise = train_recon = train_fft = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            stacked = stack_signals(batch, DEVICE)
            noise_loss, recon_loss, freq_loss = diffusion_step(
                model, stacked, norm_mean, norm_std, alpha_bar)

            loss = noise_loss + LAMBDA_RECON * recon_loss + LAMBDA_FFT * freq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_noise += noise_loss.item()
            train_recon += recon_loss.item()
            train_fft   += freq_loss.item()
            n_batches   += 1

        train_noise /= n_batches
        train_recon /= n_batches
        train_fft   /= n_batches
        scheduler.step()

        # Eval
        model.eval()
        eval_noise = eval_recon = eval_fft = 0.0
        n_eval = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                stacked = stack_signals(batch, DEVICE)
                noise_loss, recon_loss, freq_loss = diffusion_step(
                    model, stacked, norm_mean, norm_std, alpha_bar)
                eval_noise += noise_loss.item()
                eval_recon += recon_loss.item()
                eval_fft   += freq_loss.item()
                n_eval     += 1

        eval_noise /= n_eval
        eval_recon /= n_eval
        eval_fft   /= n_eval
        eval_total  = eval_noise + LAMBDA_RECON * eval_recon + LAMBDA_FFT * eval_fft

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train noise={train_noise:.4f} recon={train_recon:.4f} fft={train_fft:.4f} | "
                  f"eval noise={eval_noise:.4f} recon={eval_recon:.4f} fft={eval_fft:.4f} | "
                  f"LR={lr_now:.2e}")

        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "loss":        eval_total,
            "noise_loss":  eval_noise,
            "recon_loss":  eval_recon,
            "fft_loss":    eval_fft,
            "T":           T,
            "schedule":    SCHEDULE,
            "sensor_names": SENSOR_NAMES,
            "config": {
                "n_sensors":    len(SENSOR_NAMES),
                "in_channels":  3,
                "d_model":      D_MODEL,
                "num_heads":    NUM_HEADS,
                "num_blocks":   NUM_BLOCKS,
                "dropout":      DROPOUT,
                "t_common":     T_COMMON,
                "lambda_recon": LAMBDA_RECON,
                "lambda_fft":   LAMBDA_FFT,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if eval_total < best_loss:
            best_loss = eval_total
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best: noise={eval_noise:.4f} recon={eval_recon:.4f} fft={eval_fft:.4f}")

    print(f"\n{'='*65}")
    print(f"Done. Best eval loss: {best_loss:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
