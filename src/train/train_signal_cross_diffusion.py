# ============================================================
# Train Cross-Sensor Signal Diffusion (no VAE)
#
# p(x_missing | x_observed) directly at signal level.
# All sensors interpolated to T_COMMON=256 for cross-attention.
#
# Uses only labeled CogAge data (blho+bbh+state) — no WISDM needed.
#
# Usage:
#   python -m src.train.train_signal_cross_diffusion
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

OUT_DIR = Path("checkpoints/signal_cross_diffusion")
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

# Random masking: how many sensors to mask per batch
MIN_MISSING = 1
MAX_MISSING = 3

NUM_WORKERS = 4


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
# Signal preparation
# ============================================================
def stack_signals(batch, device):
    """
    Returns: (B, K, C, T_COMMON) — all sensors interpolated to T_COMMON
    """
    parts = []
    for name in SENSOR_NAMES:
        x = batch[name].to(device)              # (B, T_native, C)
        x = x.permute(0, 2, 1).float()          # (B, C, T_native)
        x = F.interpolate(x, size=T_COMMON,
                          mode='linear', align_corners=False)  # (B, C, T_COMMON)
        parts.append(x)
    return torch.stack(parts, dim=1)            # (B, K, C, T_COMMON)


# ============================================================
# Normalization stats (per sensor, per channel, mean/std over training)
# ============================================================
def compute_norm_stats(train_loader, device):
    print("Computing normalization stats...")
    sums = {k: 0.0 for k in SENSOR_NAMES}
    sqs  = {k: 0.0 for k in SENSOR_NAMES}
    count = 0
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="norm stats", leave=False):
            for name in SENSOR_NAMES:
                x = batch[name].to(device).float()   # (B, T, C)
                sums[name] = sums[name] + x.mean(dim=(0, 1))
                sqs[name]  = sqs[name]  + (x ** 2).mean(dim=(0, 1))
            count += 1
    stats = {}
    for name in SENSOR_NAMES:
        mean = sums[name] / count                          # (C,)
        std  = ((sqs[name] / count) - mean ** 2).clamp(min=1e-6).sqrt()
        stats[name] = {"mean": mean.cpu(), "std": std.cpu()}
    return stats


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training Signal Cross-Sensor Diffusion (no VAE)")
    print(f"T={T}, T_COMMON={T_COMMON}, Epochs={EPOCHS}")
    print(f"D_model={D_MODEL}, Blocks={NUM_BLOCKS}, Heads={NUM_HEADS}")
    print(f"{'='*65}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # All CogAge data combined (labeled + unlabeled both fine, no labels needed)
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

    # Normalization: compute per-sensor mean/std of raw signals
    norm_stats = compute_norm_stats(train_loader, DEVICE)
    # Fix typo in compute_norm_stats — recompute cleanly
    norm_mean = torch.stack([
        norm_stats[k]["mean"] for k in SENSOR_NAMES
    ]).to(DEVICE)   # (K, C)
    norm_std = torch.stack([
        norm_stats[k]["std"] for k in SENSOR_NAMES
    ]).to(DEVICE)   # (K, C)
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
        train_loss = 0.0
        n_batches  = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            stacked = stack_signals(batch, DEVICE)   # (B, K, C, T_COMMON)
            B, K, C, T_len = stacked.shape

            # Normalize: (B, K, C, T) - mean/std over (K, C)
            stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                         / norm_std[None, :, :, None]

            # Random missing mask
            n_missing = random.randint(MIN_MISSING, MAX_MISSING)
            missing_idx = random.sample(range(K), n_missing)
            observed_mask = torch.ones(B, K, device=DEVICE)
            for i in missing_idx:
                observed_mask[:, i] = 0.0

            # Add noise to missing sensors only
            t_step = torch.randint(0, T, (B,), device=DEVICE)
            noise  = torch.randn_like(stacked_norm)
            ab3    = alpha_bar[t_step][:, None, None]   # (B,1,1) for (B,C,T) slices

            ab3 = alpha_bar[t_step][:, None, None]   # (B, 1, 1) for per-sensor slice
            noisy = stacked_norm.clone()
            for i in missing_idx:
                noisy[:, i] = (torch.sqrt(ab3) * stacked_norm[:, i]
                               + torch.sqrt(1 - ab3) * noise[:, i])

            noise_pred = model(noisy, t_step, observed_mask)  # (B, K, C, T)

            # Loss only on missing sensors
            loss = sum(
                F.mse_loss(noise_pred[:, i], noise[:, i])
                for i in missing_idx
            ) / len(missing_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        train_loss /= n_batches
        scheduler.step()

        # Eval
        model.eval()
        eval_loss = 0.0
        n_eval    = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                stacked = stack_signals(batch, DEVICE)
                stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                             / norm_std[None, :, :, None]
                B = stacked.shape[0]

                n_missing   = random.randint(MIN_MISSING, MAX_MISSING)
                missing_idx = random.sample(range(len(SENSOR_NAMES)), n_missing)
                observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for i in missing_idx:
                    observed_mask[:, i] = 0.0

                t_step = torch.randint(0, T, (B,), device=DEVICE)
                noise  = torch.randn_like(stacked_norm)
                ab3    = alpha_bar[t_step][:, None, None]   # (B,1,1) for (B,C,T) slices

                ab3 = alpha_bar[t_step][:, None, None]
                noisy = stacked_norm.clone()
                for i in missing_idx:
                    noisy[:, i] = (torch.sqrt(ab3) * stacked_norm[:, i]
                                   + torch.sqrt(1 - ab3) * noise[:, i])

                noise_pred = model(noisy, t_step, observed_mask)
                loss = sum(
                    F.mse_loss(noise_pred[:, i], noise[:, i])
                    for i in missing_idx
                ) / len(missing_idx)

                eval_loss += loss.item()
                n_eval    += 1

        eval_loss /= n_eval

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} "
                  f"eval={eval_loss:.4f} | LR={lr_now:.2e}")

        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "loss":        eval_loss,
            "T":           T,
            "schedule":    SCHEDULE,
            "config": {
                "n_sensors":   len(SENSOR_NAMES),
                "in_channels": 3,
                "d_model":     D_MODEL,
                "num_heads":   NUM_HEADS,
                "num_blocks":  NUM_BLOCKS,
                "dropout":     DROPOUT,
                "t_common":    T_COMMON,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best: {best_loss:.6f}")

    print(f"\n{'='*65}")
    print(f"Done. Best eval loss: {best_loss:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
