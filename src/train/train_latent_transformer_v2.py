# ============================================================
# Train Latent Transformer/MLP on VAE V2 Latents — Robust
#
# Flags:
#   --model transformer|mlp  (default: transformer)
#   --state                  State activities (6 classes)
#   --robust                 50% batches with mean-fill masking
# ============================================================

import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

MODEL_TYPE = "transformer" if "--model" not in sys.argv else \
             sys.argv[sys.argv.index("--model") + 1]
ROBUST     = "--robust" in sys.argv
USE_STATE  = "--state"  in sys.argv

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
    EPOCHS = 150
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }
    EPOCHS = 100

tag    = f"{MODEL_TYPE}_{'state' if USE_STATE else 'behavioral'}{'_robust' if ROBUST else ''}"
OUT_DIR = Path(f"checkpoints/latent_{tag}_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LR         = 1e-3
MASK_PROB  = 0.5

# Transformer — SMALL params (fix for behavioral not converging)
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
DROPOUT  = 0.3


def main():
    print(f"\n{'='*60}")
    print(f"Training Latent {MODEL_TYPE.upper()} — {tag}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    train_ds  = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    test_ds   = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)
    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Load VAE V2 (frozen)
    ckpt_vae   = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    cfg        = ckpt_vae["config"]
    latent_dim = cfg["latent_dim"]   # 16
    t_shared   = cfg["t_shared"]     # 64
    print(f"VAE V2: latent_dim={latent_dim}, t_shared={t_shared}")

    vae = SensorMultiModalVAE(latent_dim=latent_dim, t_shared=t_shared).to(DEVICE)
    vae.load_state_dict(ckpt_vae["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Mean latents for robust masking
    mean_latents = None
    if ROBUST:
        print("Computing mean latents...")
        sums  = {k: 0.0 for k in SENSOR_NAMES}
        count = 0
        with torch.no_grad():
            for batch in train_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                outputs = vae(sensor_data)
                for k in SENSOR_NAMES:
                    sums[k] = sums[k] + outputs[k]["mu"].mean(dim=0, keepdim=True)
                count += 1
        mean_latents = {k: (sums[k] / count).to(DEVICE) for k in SENSOR_NAMES}
        print("  Done.")

    # Classifier
    clf_kwargs = {"d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                  "dropout": DROPOUT} if MODEL_TYPE == "transformer" \
                 else {"hidden_dims": [512, 256, 128], "dropout": DROPOUT}

    classifier = create_activity_classifier(
        model_type=MODEL_TYPE,
        n_sensors=len(SENSOR_NAMES),
        latent_dim=latent_dim,
        t_shared=t_shared,
        n_classes=n_classes,
        **clf_kwargs,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"{MODEL_TYPE} params: {n_params/1e6:.2f}M\n")

    opt       = torch.optim.Adam(classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        classifier.train()
        train_loss = train_correct = train_total = 0

        for batch in train_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels      = batch["label"].to(DEVICE)
            B           = labels.size(0)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            if ROBUST and random.random() < MASK_PROB:
                n_miss  = random.randint(1, 3)
                missing = random.sample(SENSOR_NAMES, n_miss)
                for name in missing:
                    latents[name] = mean_latents[name].expand(B, -1, -1)

            opt.zero_grad()
            logits = classifier(latents, SENSOR_NAMES)
            loss   = criterion(logits, labels)
            loss.backward()
            opt.step()

            train_loss    += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += B

        scheduler.step()

        # ---- Eval ----
        classifier.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels      = batch["label"].to(DEVICE)
                outputs     = vae(sensor_data)
                latents     = {k: outputs[k]["mu"] for k in SENSOR_NAMES}
                preds       = classifier(latents, SENSOR_NAMES).argmax(1)
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)

        train_acc = train_correct / train_total
        test_acc  = test_correct  / test_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state": classifier.state_dict(),
                "acc": test_acc,
                "n_classes": n_classes,
                "model_type": MODEL_TYPE,
                "latent_dim": latent_dim,
                "t_shared": t_shared,
                "label_to_idx": train_ds.label_to_idx,
                "idx_to_label": train_ds.idx_to_label,
            }, OUT_DIR / "best_model.pt")

    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
