# ============================================================
# Train Cross-Sensor Signal Diffusion + Reconstruction Loss
#
# Same as train_signal_cross_diffusion but with additional losses:
#   x0_pred = (x_t - sqrt(1-ab)*eps_pred) / sqrt(ab)  [differentiable]
#   recon_loss = MSE(x0_pred, x_original)              [in signal space]
#   fft_loss   = MSE(|FFT(x0_pred)|, |FFT(x_original)|)
#
# Total: noise_loss + LAMBDA_RECON * recon_loss + LAMBDA_FFT * fft_loss
#
# No VAE needed — x0_pred is directly the signal.
# New checkpoint: checkpoints/signal_cross_diffusion_recon/
#
# Usage:
#   python -m src.train.train_signal_cross_diffusion_recon
# ============================================================

import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.models.signal_cross_diffusion import (
    create_signal_cross_diffusion, T_COMMON,
)
from src.models.sensor_vae import SENSOR_NAMES
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
COGAGE_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUT_DIR = Path("checkpoints/signal_cross_diffusion_recon_v3")
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

# Device groups for realistic missing patterns
DEVICE_GROUPS = {
    "phone":   ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
    "watch":   ["watch_acc", "watch_gyro"],
    "glasses": ["glasses_acc"],
}
# Sampling probabilities:
# 0.35 → full device missing
# 0.25 → single sensor missing
# 0.40 → random 2-3 sensors missing
P_DEVICE  = 0.35
P_SINGLE  = 0.40
# rest = random 2-3

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
        # Full device missing
        device = random.choice(list(DEVICE_GROUPS.keys()))
        missing = DEVICE_GROUPS[device]
    elif p < P_DEVICE + P_SINGLE:
        # Single random sensor
        missing = [random.choice(SENSOR_NAMES)]
    else:
        # Random 2-3 sensors
        missing = random.sample(SENSOR_NAMES, random.randint(2, 3))
    return [SENSOR_NAMES.index(s) for s in missing]


# ============================================================
# FFT loss
# ============================================================
def fft_loss(pred, target):
    """pred, target: (B, C, T) — compares magnitude spectra along T"""
    pred_mag   = torch.fft.rfft(pred,   dim=-1).abs()
    target_mag = torch.fft.rfft(target, dim=-1).abs()
    return F.mse_loss(pred_mag, target_mag)


# ============================================================
# Signal preparation
# ============================================================
def stack_signals(batch, device):
    """Returns: (B, K, C, T_COMMON)"""
    parts = []
    for name in SENSOR_NAMES:
        x = batch[name].to(device)
        x = x.permute(0, 2, 1).float()
        x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
        parts.append(x)
    return torch.stack(parts, dim=1)

    # This function is used to make the sensor data have same common time length (T_common) for the diffusion model.


# ============================================================
# Normalization stats
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
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training Signal Cross-Sensor Diffusion + Recon Loss")
    print(f"T={T}, T_COMMON={T_COMMON}, Epochs={EPOCHS}")
    print(f"lambda_recon={LAMBDA_RECON}, lambda_fft={LAMBDA_FFT}")
    print(f"D_model={D_MODEL}, Blocks={NUM_BLOCKS}, Heads={NUM_HEADS}")
    print(f"{'='*65}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    behavioral_train = get_combined_labeled_dataset(
        {"blho": COGAGE_ROOTS["blho"], "bbh": COGAGE_ROOTS["bbh"]}, "training", normalizer,
    )
    behavioral_test = get_combined_labeled_dataset(
        {"blho": COGAGE_ROOTS["blho"], "bbh": COGAGE_ROOTS["bbh"]}, "testing", normalizer,
    )
    state_train = CogAgeLabeledDataset(COGAGE_ROOTS["state"], "training", normalizer)
    state_test  = CogAgeLabeledDataset(COGAGE_ROOTS["state"], "testing",  normalizer)

    train_ds = ConcatDataset([behavioral_train, state_train])
    test_ds  = ConcatDataset([behavioral_test,  state_test])
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
    betas     = cosine_beta_schedule(T)             # If this is a noise amount
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)    # How much the original signal is kept

    # Model
    print(f"\nCreating Signal Cross-Sensor Diffusion...")
    model = create_signal_cross_diffusion(
        n_sensors=len(SENSOR_NAMES),
        in_channels=3,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE) # Step from 209-215 creates a neural network model
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
            stacked = stack_signals(batch, DEVICE)   # (B, K, C, T_COMMON)
            B, K, C, T_len = stacked.shape

            stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                         / norm_std[None, :, :, None]

            missing_idx   = sample_missing_idx()
            observed_mask = torch.ones(B, K, device=DEVICE)
            for i in missing_idx:
                observed_mask[:, i] = 0.0

            t_step = torch.randint(0, T, (B,), device=DEVICE)
            noise  = torch.randn_like(stacked_norm)
            ab3    = alpha_bar[t_step][:, None, None]   # (B, 1, 1)

            noisy = stacked_norm.clone()
            for i in missing_idx:
                noisy[:, i] = (torch.sqrt(ab3) * stacked_norm[:, i]
                               + torch.sqrt(1 - ab3) * noise[:, i])

            noise_pred = model(noisy, t_step, observed_mask)  # (B, K, C, T)

            # 1) Noise prediction loss
            noise_loss = sum(
                F.mse_loss(noise_pred[:, i], noise[:, i])
                for i in missing_idx
            ) / len(missing_idx)

            # 2) Reconstruction loss: recover x0 in normalized signal space
            recon_loss = torch.tensor(0.0, device=DEVICE)
            freq_loss  = torch.tensor(0.0, device=DEVICE)
            for i in missing_idx:
                x0_pred = ((noisy[:, i] - torch.sqrt(1 - ab3) * noise_pred[:, i])
                           / torch.sqrt(ab3)).clamp(-5, 5)   # (B, C, T_COMMON)
                recon_loss = recon_loss + F.mse_loss(x0_pred, stacked_norm[:, i])
                freq_loss  = freq_loss  + fft_loss(x0_pred, stacked_norm[:, i])
            recon_loss = recon_loss / len(missing_idx)
            freq_loss  = freq_loss  / len(missing_idx)

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
                stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                             / norm_std[None, :, :, None]
                B = stacked.shape[0]

                missing_idx   = sample_missing_idx()
                observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for i in missing_idx:
                    observed_mask[:, i] = 0.0

                t_step = torch.randint(0, T, (B,), device=DEVICE)
                noise  = torch.randn_like(stacked_norm)
                ab3    = alpha_bar[t_step][:, None, None]

                noisy = stacked_norm.clone()
                for i in missing_idx:
                    noisy[:, i] = (torch.sqrt(ab3) * stacked_norm[:, i]
                                   + torch.sqrt(1 - ab3) * noise[:, i])

                noise_pred = model(noisy, t_step, observed_mask)

                noise_loss = sum(
                    F.mse_loss(noise_pred[:, i], noise[:, i])
                    for i in missing_idx
                ) / len(missing_idx)

                recon_loss = torch.tensor(0.0, device=DEVICE)
                freq_loss  = torch.tensor(0.0, device=DEVICE)
                for i in missing_idx:
                    x0_pred = ((noisy[:, i] - torch.sqrt(1 - ab3) * noise_pred[:, i])
                               / torch.sqrt(ab3)).clamp(-5, 5)
                    recon_loss = recon_loss + F.mse_loss(x0_pred, stacked_norm[:, i])
                    freq_loss  = freq_loss  + fft_loss(x0_pred, stacked_norm[:, i])
                recon_loss = recon_loss / len(missing_idx)
                freq_loss  = freq_loss  / len(missing_idx)

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
