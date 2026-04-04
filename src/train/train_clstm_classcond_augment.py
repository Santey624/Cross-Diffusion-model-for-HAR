# ============================================================
# Train C-LSTM-A with Class-Conditional Diffusion Augmentation
#
# Pipeline:
#   50% of batches: all sensors real (VAE V2 encode → decode)
#   50% of batches: 1-3 sensors missing → class-cond diffusion
#                   (uses GT label during training → stable signal)
#
# At eval time: predicted label used instead of GT
# The classifier learns to use diffusion-generated signals
#
# Usage:
#   python -m src.train.train_clstm_classcond_augment
#   python -m src.train.train_clstm_classcond_augment --state
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

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.class_conditional_diffusion import create_class_conditional_diffusion
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE = "--state" in sys.argv

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

tag = "state" if USE_STATE else "behavioral"
DIFF_DIR        = Path(f"checkpoints/class_cond_diffusion_{tag}")
OUT_DIR         = Path(f"checkpoints/clstm_classcond_{tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
AUG_PROB    = 0.5   # fraction of batches with diffusion augmentation
DDIM_STEPS  = 10    # fewer steps during training for speed (50 at eval)
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
# DDIM sampling (class-conditional)
# ============================================================
@torch.no_grad()
def ddim_sample_class(model, class_label, alpha_bar, T,
                      latent_dim, t_lat, ddim_steps):
    B      = class_label.shape[0]
    device = class_label.device
    z      = torch.randn(B, latent_dim, t_lat, device=device)
    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)
    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch    = torch.full((B,), t_now, device=device, dtype=torch.long)
        noise_pred = model(z, t_batch, class_label)
        ab_now     = alpha_bar[t_now]
        ab_next    = alpha_bar[t_next]
        pred_x0    = ((z - torch.sqrt(1 - ab_now) * noise_pred)
                      / torch.sqrt(ab_now)).clamp(-5, 5)
        z = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training C-LSTM-A + Class-Cond Diffusion Augmentation — {tag}")
    print(f"Aug prob: {AUG_PROB}, DDIM steps (train): {DDIM_STEPS}")
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

    # VAE V2 (frozen)
    print(f"\nLoading VAE V2...")
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    cfg_vae  = vae_ckpt["config"]
    vae = SensorMultiModalVAE(
        latent_dim=cfg_vae["latent_dim"],
        t_shared=cfg_vae["t_shared"],
    ).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    latent_dim = cfg_vae["latent_dim"]
    t_lat      = cfg_vae["t_shared"]

    # Class-Conditional Diffusion (frozen)
    print(f"Loading Class-Cond Diffusion from {DIFF_DIR}...")
    diff_ckpt  = torch.load(DIFF_DIR / "best_model.pt", map_location=DEVICE)
    cfg_diff   = diff_ckpt["config"]
    diff_model = create_class_conditional_diffusion(
        latent_dim=cfg_diff["latent_dim"],
        t_lat=cfg_diff["t_lat"],
        n_classes=cfg_diff["n_classes"],
        d_model=cfg_diff["d_model"],
        n_blocks=cfg_diff["n_blocks"],
        emb_dim=cfg_diff["emb_dim"],
    ).to(DEVICE)
    diff_model.load_state_dict(diff_ckpt["model_state"])
    diff_model.eval()
    for p in diff_model.parameters():
        p.requires_grad = False

    T_diff     = diff_ckpt["T"]
    betas      = cosine_beta_schedule(T_diff)
    alpha_bar  = torch.cumprod(1.0 - betas, dim=0)
    norm_stats = torch.load(DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
    print(f"  Diffusion loaded (loss={diff_ckpt['loss']:.4f})")

    # Classifier
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
    print(f"\nC-LSTM-A params: {n_params/1e6:.2f}M")

    opt       = torch.optim.Adam(classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        train_loss = 0.0
        n_aug = n_real = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels      = batch["label"].to(DEVICE)
            B           = labels.size(0)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

                if random.random() < AUG_PROB:
                    # Diffusion augmentation: replace 1-3 sensors with generated ones
                    n_missing = random.randint(1, 3)
                    missing   = random.sample(SENSOR_NAMES, n_missing)

                    for name in missing:
                        z_gen = ddim_sample_class(
                            diff_model, labels, alpha_bar, T_diff,
                            latent_dim, t_lat, DDIM_STEPS,
                        )  # (B, D, T_lat) — normalized
                        latents[name] = (
                            z_gen * norm_stats[name]["std"].to(DEVICE)[None, :, None]
                            + norm_stats[name]["mean"].to(DEVICE)[None, :, None]
                        )
                    n_aug += 1
                else:
                    n_real += 1

                # Decode all sensors
                decoded = {}
                for name in SENSOR_NAMES:
                    sig = vae.decode_sensor(name, latents[name])  # (B, T, C)
                    decoded[name] = sig.permute(0, 2, 1)           # (B, C, T)

            logits = classifier(decoded, SENSOR_NAMES)
            loss   = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        scheduler.step()

        # ---- Eval (all real sensors) ----
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels      = batch["label"].to(DEVICE)
                outputs = vae(sensor_data)
                decoded = {}
                for name in SENSOR_NAMES:
                    sig = vae.decode_sensor(name, outputs[name]["mu"])
                    decoded[name] = sig.permute(0, 2, 1)
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
                    "n_sensors":    len(SENSOR_NAMES),
                    "n_classes":    n_classes,
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
                  f"| Acc: {acc:.4f} | aug={n_aug} real={n_real} | LR: {lr_now:.2e}")

    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
