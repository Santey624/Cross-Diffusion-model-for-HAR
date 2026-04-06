# ============================================================
# Train Noisy C-LSTM-A Classifier for Classifier-Guided Diffusion
#
# Key difference to normal classifier:
#   During training, signals are corrupted with random noise levels
#   matching the diffusion forward process. This allows the classifier
#   to predict classes from noisy/partially-denoised signals.
#
# Used at inference time to guide DDIM sampling toward
# activity-consistent signals.
#
# Usage:
#   python -m src.train.train_noisy_classifier
#   python -m src.train.train_noisy_classifier --state
# ============================================================

import sys
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.clstm_classifier import create_clstm_classifier
from src.models.sensor_vae import SENSOR_NAMES, SENSOR_SPECS
from src.models.signal_cross_diffusion import T_COMMON
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

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

OUT_DIR = Path(f"checkpoints/clstm_noisy_{tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion noise schedule (must match signal_cross_diffusion)
T_DIFF     = 1000
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
NUM_WORKERS = 4

# Noise augmentation config
# p_clean: probability of using clean signal (no noise)
# p_noisy: probability of adding diffusion noise at random t
P_CLEAN = 0.3
P_NOISY = 0.7

NATIVE_LENS = {name: SENSOR_SPECS[name]["seq_len"] for name in SENSOR_NAMES}


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
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training Noisy C-LSTM-A — {tag}")
    print(f"P_clean={P_CLEAN}, P_noisy={P_NOISY} | Device: {DEVICE}")
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
                              drop_last=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Noise schedule
    betas     = cosine_beta_schedule(T_DIFF)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)

    # Classifier — works on T_COMMON length signals
    # We interpolate all sensors to T_COMMON for consistent noising
    classifier = create_clstm_classifier(
        n_sensors=len(SENSOR_NAMES),
        n_classes=n_classes,
        cnn_channels=64,
        lstm_hidden=64,
        d_attn=128,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"C-LSTM-A params: {n_params/1e6:.2f}M")

    opt       = torch.optim.Adam(classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            labels = batch["label"].to(DEVICE)
            B      = labels.size(0)

            signals = {}
            for name in SENSOR_NAMES:
                x = batch[name].to(DEVICE).float()   # (B, T, C)
                x = x.permute(0, 2, 1)               # (B, C, T)
                x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)

                # Add noise with probability P_NOISY
                if random.random() < P_NOISY:
                    t_step = torch.randint(0, T_DIFF, (B,), device=DEVICE)
                    ab     = alpha_bar[t_step][:, None, None]   # (B, 1, 1)
                    noise  = torch.randn_like(x)
                    x = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise

                signals[name] = x

            logits = classifier(signals, SENSOR_NAMES)
            loss   = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        scheduler.step()

        # Eval on clean signals
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                labels  = batch["label"].to(DEVICE)
                signals = {}
                for name in SENSOR_NAMES:
                    x = batch[name].to(DEVICE).float().permute(0, 2, 1)
                    x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
                    signals[name] = x
                preds = classifier(signals, SENSOR_NAMES).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch":       epoch,
                "model_state": classifier.state_dict(),
                "accuracy":    acc,
                "t_diff":      T_DIFF,
                "config": {
                    "n_sensors":    len(SENSOR_NAMES),
                    "n_classes":    n_classes,
                    "cnn_channels": 64,
                    "lstm_hidden":  64,
                    "d_attn":       128,
                    "n_heads":      4,
                    "n_layers":     2,
                    "dropout":      0.3,
                    "t_common":     T_COMMON,
                },
            }, OUT_DIR / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} "
                  f"| Acc: {acc:.4f} | LR: {lr_now:.2e}")

    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
