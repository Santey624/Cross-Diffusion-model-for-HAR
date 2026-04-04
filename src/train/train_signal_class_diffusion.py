# ============================================================
# Train Class-Conditional Diffusion on Raw Sensor Signals
#
# No VAE — works directly on normalized sensor signals.
# All sensors interpolated to T_MODEL=256.
# Conditioning: timestep + activity class + sensor_id
#
# Usage:
#   python -m src.train.train_signal_class_diffusion
#   python -m src.train.train_signal_class_diffusion --state
# ============================================================

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.signal_class_diffusion import (
    create_signal_class_diffusion, SENSOR_T, T_MODEL,
)
from src.models.sensor_vae import SENSOR_NAMES
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE = "--state" in sys.argv

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

tag = "state" if USE_STATE else "behavioral"
OUT_DIR = Path(f"checkpoints/signal_class_diffusion_{tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

T          = 1000
SCHEDULE   = "cosine"
BATCH_SIZE = 64
EPOCHS     = 150
LR         = 3e-4
BASE_CH    = 64
EMB_DIM    = 128
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


def prepare_signal(x, name):
    """
    x: (B, T, C) raw sensor signal
    Returns: (B, C, SENSOR_T[name]) resampled to sensor-native length (mult of 8)
    """
    x = x.permute(0, 2, 1).float()    # (B, C, T)
    t = SENSOR_T.get(name, T_MODEL)
    x = F.interpolate(x, size=t, mode='linear', align_corners=False)
    return x


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training Signal Class-Conditional Diffusion — {tag}")
    print(f"T={T}, T_model={T_MODEL}, Epochs={EPOCHS}")
    print(f"Base_ch={BASE_CH}, Emb_dim={EMB_DIM}")
    print(f"{'='*65}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    if USE_STATE:
        train_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
        test_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing",  normalizer)
    else:
        train_ds = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
        test_ds  = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)

    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Noise schedule
    betas     = cosine_beta_schedule(T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)

    # Model
    print(f"\nCreating Signal Class-Conditional Diffusion...")
    model = create_signal_class_diffusion(
        n_sensors=len(SENSOR_NAMES),
        n_classes=n_classes,
        in_channels=3,
        base_ch=BASE_CH,
        emb_dim=EMB_DIM,
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
            labels = batch["label"].to(DEVICE)

            # Pick a random sensor for this batch step
            sensor_idx  = torch.randint(len(SENSOR_NAMES), (1,)).item()
            sensor_name = SENSOR_NAMES[sensor_idx]
            sensor_ids  = torch.full((labels.shape[0],), sensor_idx,
                                     dtype=torch.long, device=DEVICE)

            x0 = prepare_signal(batch[sensor_name], sensor_name).to(DEVICE)
            t_sensor = x0.shape[-1]   # actual length for this sensor

            B      = x0.shape[0]
            t_step = torch.randint(0, T, (B,), device=DEVICE)
            noise  = torch.randn_like(x0)
            ab     = alpha_bar[t_step][:, None, None]
            x_t    = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

            noise_pred = model(x_t, t_step, labels, sensor_ids)
            loss = F.mse_loss(noise_pred, noise)

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
                labels = batch["label"].to(DEVICE)
                loss_sum = 0.0
                for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
                    sensor_ids = torch.full((labels.shape[0],), sensor_idx,
                                            dtype=torch.long, device=DEVICE)
                    x0     = prepare_signal(batch[sensor_name], sensor_name).to(DEVICE)
                    B      = x0.shape[0]
                    t_step = torch.randint(0, T, (B,), device=DEVICE)
                    noise  = torch.randn_like(x0)
                    ab     = alpha_bar[t_step][:, None, None]
                    x_t    = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
                    noise_pred = model(x_t, t_step, labels, sensor_ids)
                    loss_sum += F.mse_loss(noise_pred, noise).item()
                eval_loss += loss_sum / len(SENSOR_NAMES)
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
                "n_classes":   n_classes,
                "in_channels": 3,
                "base_ch":     BASE_CH,
                "emb_dim":     EMB_DIM,
                "t_model":     T_MODEL,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best: {best_loss:.6f}")

    print(f"\n{'='*65}")
    print(f"Training done. Best eval loss: {best_loss:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
