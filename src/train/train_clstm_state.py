# ============================================================
# Train C-LSTM-A Classifier on Decoded Signals — State (6 classes)
#
# Option 2 pipeline:
#   sensor signal -> VAE encode -> latent z -> VAE decode -> C-LSTM-A
#
# Flags:
#   --robust  50% of batches randomly mask 1-3 sensors (mean-fill latent -> decode)
# ============================================================

import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import CogAgeLabeledDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
STATE_ROOT = "data/cogage/python/arrays/state"

ROBUST = "--robust" in sys.argv
OUT_DIR = Path("checkpoints/clstm_state_robust" if ROBUST else "checkpoints/clstm_state")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
MASK_PROB = 0.5


def decode_latents(vae, latents):
    decoded = {}
    for name in SENSOR_NAMES:
        z = latents[name]
        sig = vae.decode_sensor(name, z)      # (B, T, C)
        decoded[name] = sig.permute(0, 2, 1)  # (B, C, T)
    return decoded


def main():
    print(f"\n{'='*60}")
    print(f"Training C-LSTM-A — State Activities (6 classes)")
    print(f"Robust: {ROBUST} | Device: {DEVICE}")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    train_ds = CogAgeLabeledDataset(STATE_ROOT, "training", normalizer)
    test_ds  = CogAgeLabeledDataset(STATE_ROOT, "testing",  normalizer)
    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Load VAE (frozen)
    vae = SensorMultiModalVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE)["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Pre-compute mean latents for robust masking
    mean_latents = None
    if ROBUST:
        print("Computing mean latents for robust masking...")
        sums = {k: 0.0 for k in SENSOR_NAMES}
        count = 0
        with torch.no_grad():
            for batch in train_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                outputs = vae(sensor_data)
                for k in SENSOR_NAMES:
                    sums[k] = sums[k] + outputs[k]["mu"].mean(dim=0, keepdim=True)
                count += 1
        mean_latents = {k: (sums[k] / count).to(DEVICE) for k in SENSOR_NAMES}

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

    opt = torch.optim.Adam(classifier.parameters(), lr=LR)
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
            B = labels.size(0)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

                if ROBUST and random.random() < MASK_PROB:
                    n_mask = random.randint(1, 3)
                    for name in random.sample(SENSOR_NAMES, n_mask):
                        latents[name] = mean_latents[name].expand(B, -1, -1)

                decoded = decode_latents(vae, latents)

            logits = classifier(decoded, SENSOR_NAMES)
            loss = criterion(logits, labels)

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
                labels = batch["label"].to(DEVICE)
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}
                decoded = decode_latents(vae, latents)
                preds = classifier(decoded, SENSOR_NAMES).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state": classifier.state_dict(),
                "accuracy": acc,
                "config": {
                    "n_sensors": len(SENSOR_NAMES),
                    "n_classes": n_classes,
                    "cnn_channels": 64,
                    "lstm_hidden": 64,
                    "d_attn": 128,
                    "n_heads": 4,
                    "n_layers": 2,
                    "dropout": 0.3,
                },
            }, OUT_DIR / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} "
                  f"| Acc: {acc:.4f} | LR: {lr:.2e}")

    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
