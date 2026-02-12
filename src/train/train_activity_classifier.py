# ============================================================
# Train Activity Classifier on Sensor Latents
# Uses VAE to encode sensors → MLP/Transformer → activity class
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_best.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUT_DIR = Path("checkpoints/activity_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_TYPE = "mlp"  # "mlp" or "transformer"
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Training Activity Classifier on Sensor Latents")
    print(f"Model: {MODEL_TYPE}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load datasets with labels
    print("Loading datasets...")
    train_dataset = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)

    n_classes = train_dataset.n_classes
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {n_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Load VAE (frozen)
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Create classifier
    print("Creating classifier...")
    classifier = create_activity_classifier(
        model_type=MODEL_TYPE,
        n_classes=n_classes,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)

            # Encode with VAE (frozen)
            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            # Classify
            optimizer.zero_grad()
            logits = classifier(latents, SENSOR_NAMES)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Eval
        classifier.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test {epoch}/{EPOCHS}", leave=False):
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels = batch["label"].to(DEVICE)

                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

                logits = classifier(latents, SENSOR_NAMES)
                loss = F.cross_entropy(logits, labels)

                test_loss += loss.item()
                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state': classifier.state_dict(),
                'acc': test_acc,
                'n_classes': n_classes,
                'model_type': MODEL_TYPE,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label,
            }, OUT_DIR / "best_model.pt")

    # Save final
    torch.save({
        'epoch': EPOCHS,
        'model_state': classifier.state_dict(),
        'acc': test_acc,
        'n_classes': n_classes,
        'model_type': MODEL_TYPE,
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
    }, OUT_DIR / "final_model.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
