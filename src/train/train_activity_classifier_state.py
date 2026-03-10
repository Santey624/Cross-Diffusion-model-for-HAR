# ============================================================
# Train Activity Classifier for STATE Activities (6 classes)
# ============================================================

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import CogAgeLabeledDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATE_ROOT = "data/cogage/python/arrays/state"
VAE_CHECKPOINT = "checkpoints/sensor_vae_best.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"
OUT_DIR = Path("checkpoints/activity_classifier_state")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 512


def main():
    print(f"\n{'='*60}")
    print("Training State Activity Classifier (6 classes)")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load datasets
    train_ds = CogAgeLabeledDataset(STATE_ROOT, "training", normalizer)
    test_ds = CogAgeLabeledDataset(STATE_ROOT, "testing", normalizer)
    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load VAE (frozen)
    vae = SensorMultiModalVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE)["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Create classifier
    classifier = create_activity_classifier(n_classes=n_classes, hidden_dims=[HIDDEN_DIM, 256, 128]).to(DEVICE)
    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier params: {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        classifier.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            logits = classifier(latents, SENSOR_NAMES)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        scheduler.step()

        # Eval
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels = batch["label"].to(DEVICE)

                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}
                preds = classifier(latents, SENSOR_NAMES).argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state": classifier.state_dict(),
                "accuracy": acc,
                "config": {"hidden_dims": [HIDDEN_DIM, 256, 128], "n_classes": n_classes},
            }, OUT_DIR / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | LR: {lr:.2e}")

    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
