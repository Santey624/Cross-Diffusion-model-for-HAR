# ============================================================
# Train C-LSTM-A Classifier on V4 Decoded Signals — Behavioral
#
# Pipeline:
#   sensor signal → VAE V4 encode → V4 decode → C-LSTM-A
#
# Flags:
#   --robust  50% of batches randomly mask 1-3 sensors via V4.impute()
#   --state   use state dataset (6 classes) instead of behavioral (55)
# ============================================================

import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae_v4 import SensorSharedPrivateVAE, SENSOR_NAMES
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROBUST    = "--robust" in sys.argv
USE_STATE = "--state"  in sys.argv

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v4/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
    tag = "state"
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }
    tag = "behavioral"

OUT_DIR = Path(f"checkpoints/clstm_v4_{tag}{'_robust' if ROBUST else ''}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
MASK_PROB  = 0.5


def decode_outputs(outputs):
    """V4 outputs dict → {name: (B, C, T)} for classifier."""
    return {
        name: out["recon"].permute(0, 2, 1)   # (B, T, C) → (B, C, T)
        for name, out in outputs.items()
    }


def main():
    print(f"\n{'='*60}")
    print(f"Training C-LSTM-A on V4 Signals — {tag}")
    print(f"Robust: {ROBUST} | Device: {DEVICE}")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    if USE_STATE:
        train_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
        test_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing",  normalizer)
    else:
        train_ds = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
        test_ds  = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)

    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Load V4 VAE (frozen)
    print(f"\nLoading VAE V4 from {VAE_CHECKPOINT}...")
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    cfg      = vae_ckpt["config"]
    vae = SensorSharedPrivateVAE(
        d_shared=cfg["d_shared"],
        d_private=cfg["d_private"],
        t_lat=cfg["t_lat"],
    ).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  Loaded (d_shared={cfg['d_shared']}, d_private={cfg['d_private']}, "
          f"t_lat={cfg['t_lat']}, epoch={vae_ckpt['epoch']})")

    # Create classifier
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
        # ---- Train ----
        classifier.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)

            with torch.no_grad():
                if ROBUST and random.random() < MASK_PROB:
                    # Pick 1–3 sensors to impute via V4.impute()
                    n_mask   = random.randint(1, 3)
                    missing  = random.sample(SENSOR_NAMES, n_mask)
                    avail    = {k: sensor_data[k] for k in SENSOR_NAMES if k not in missing}

                    # Encode available sensors via full forward pass
                    outputs, _, _ = vae(avail, mask_ratio=0.0)
                    decoded = decode_outputs(outputs)

                    # Impute missing sensors
                    for name in missing:
                        imputed = vae.impute(avail, name)          # (B, T, C)
                        decoded[name] = imputed.permute(0, 2, 1)   # (B, C, T)
                else:
                    outputs, _, _ = vae(sensor_data, mask_ratio=0.0)
                    decoded = decode_outputs(outputs)

            logits = classifier(decoded, SENSOR_NAMES)
            loss   = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        scheduler.step()

        # ---- Eval ----
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels      = batch["label"].to(DEVICE)
                outputs, _, _ = vae(sensor_data, mask_ratio=0.0)
                decoded       = decode_outputs(outputs)
                preds = classifier(decoded, SENSOR_NAMES).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch":       epoch,
                "model_state": classifier.state_dict(),
                "accuracy":    acc,
                "config": {
                    "n_sensors":   len(SENSOR_NAMES),
                    "n_classes":   n_classes,
                    "cnn_channels": 64,
                    "lstm_hidden":  64,
                    "d_attn":       128,
                    "n_heads":      4,
                    "n_layers":     2,
                    "dropout":      0.3,
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
