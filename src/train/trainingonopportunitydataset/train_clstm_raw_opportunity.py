# ============================================================
# Train C-LSTM-A on Raw Sensor Signals (no VAE)  —  Opportunity
#
# Baseline classifier that works directly on raw signals.
# AdaptiveAvgPool handles different sensor lengths automatically.
#
# Same model & recipe as the CogAge raw classifier
# (src/train/train_clstm_raw.py), but on the 14 triaxial body-IMU
# sensors extracted from Opportunity, and without SensorNormalizer
# (Opportunity signals are fed raw, matching how the cross-sensor
# diffusion model was trained).
#
# Flags:
#   --track NAME      label track to classify (default: locomotion)
#   --augment-cross   30% of batches: missing sensors via cross-sensor signal diffusion
#   --tcommon         resample ALL sensors to T_COMMON=256 (no native lengths)
#
# Prereq: preprocessed arrays with per-track label files:
#   python -m src.data.opportunity.preprocess_opportunity
#
# Usage (from repo root):
#   python -m src.train.trainingonopportunitydataset.train_clstm_raw_opportunity
#   python -m src.train.trainingonopportunitydataset.train_clstm_raw_opportunity --track locomotion
#   python -m src.train.trainingonopportunitydataset.train_clstm_raw_opportunity --augment-cross
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
from src.models.signal_cross_diffusion import create_signal_cross_diffusion, T_COMMON
from src.data.opportunity.opportunity_constants import (
    OPP_SENSOR_NAMES, OPP_SENSOR_FILES, DEFAULT_LABEL_TRACK,
)
from src.data.opportunity.opportunity_labeled_dataset import OpportunityLabeledDataset

SENSOR_NAMES = OPP_SENSOR_NAMES


# ============================================================
# CONFIG
# ============================================================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT_CROSS = "--augment-cross" in sys.argv
USE_TCOMMON   = "--tcommon"       in sys.argv

# Label track (analog of CogAge's --state). Default: locomotion.
TRACK = DEFAULT_LABEL_TRACK
for i, a in enumerate(sys.argv):
    if a == "--track" and i + 1 < len(sys.argv):
        TRACK = sys.argv[i + 1]
        break
tag = TRACK

OPP_ROOT = "data/opportunity/arrays"

suffix = "_augment_cross" if AUGMENT_CROSS else ""
if USE_TCOMMON:
    suffix += "_tcommon"
OUT_DIR = Path(f"checkpoints/clstm_raw_opportunity_{tag}{suffix}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROSS_DIFF_DIR = Path("checkpoints/opportunity_signal_cross_diffusion")
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 1e-3
AUG_PROB_CROSS = 0.3
DDIM_STEPS     = 10
NUM_WORKERS    = 4

# Native signal lengths per sensor (for interpolation back)
NATIVE_LENS = {name: OPP_SENSOR_FILES[name][1] for name in SENSOR_NAMES}


# ============================================================
# Noise schedule + DDIM
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float()


@torch.no_grad()
def ddim_impute_cross(model, stacked_norm, observed_mask, alpha_bar, T, ddim_steps):
    """
    Cross-sensor DDIM imputation.
    stacked_norm: (B, K, C, T_COMMON) — normalized, missing=noise
    observed_mask: (B, K)
    Returns: (B, K, C, T_COMMON) imputed
    """
    B, K, C, T_len = stacked_norm.shape
    device = stacked_norm.device
    missing_idx = (observed_mask[0] == 0).nonzero(as_tuple=True)[0].tolist()

    z = stacked_norm.clone()
    for i in missing_idx:
        z[:, i] = torch.randn(B, C, T_len, device=device)

    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)

    for step in range(len(tau) - 1):
        t_now, t_next = tau[step], tau[step + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy = stacked_norm.clone()
        for i in missing_idx:
            noisy[:, i] = z[:, i]

        noise_pred = model(noisy, t_batch, observed_mask)
        ab_now  = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        for i in missing_idx:
            pred_x0 = ((z[:, i] - torch.sqrt(1 - ab_now) * noise_pred[:, i])
                       / torch.sqrt(ab_now)).clamp(-5, 5)
            z[:, i] = (torch.sqrt(ab_next) * pred_x0
                       + torch.sqrt(1 - ab_next) * noise_pred[:, i])

    result = stacked_norm.clone()
    for i in missing_idx:
        result[:, i] = z[:, i]
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training C-LSTM-A on Raw Signals — Opportunity ({tag})")
    print(f"AugmentCross: {AUGMENT_CROSS} | Device: {DEVICE}")
    print(f"{'='*65}\n")

    train_ds = OpportunityLabeledDataset(OPP_ROOT, "training", label_track=TRACK)
    test_ds  = OpportunityLabeledDataset(OPP_ROOT, "testing",  label_track=TRACK)

    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Cross-sensor diffusion (only if augmenting)
    cross_diff_model = cross_alpha_bar = T_cross = None
    cross_norm_mean = cross_norm_std = None

    if AUGMENT_CROSS:
        print(f"Loading Signal Cross-Sensor Diffusion from {CROSS_DIFF_DIR}...")
        cross_ckpt = torch.load(CROSS_DIFF_DIR / "best_model.pt", map_location=DEVICE)
        cfg_cross  = cross_ckpt["config"]
        cross_diff_model = create_signal_cross_diffusion(
            n_sensors=cfg_cross["n_sensors"],
            in_channels=cfg_cross["in_channels"],
            d_model=cfg_cross["d_model"],
            num_heads=cfg_cross["num_heads"],
            num_blocks=cfg_cross["num_blocks"],
            dropout=0.0,
        ).to(DEVICE)
        cross_diff_model.load_state_dict(cross_ckpt["model_state"])
        cross_diff_model.eval()
        for p in cross_diff_model.parameters():
            p.requires_grad = False
        T_cross     = cross_ckpt["T"]
        cross_betas = cosine_beta_schedule(T_cross)
        cross_alpha_bar = torch.cumprod(1.0 - cross_betas, dim=0)
        norm_stats  = torch.load(CROSS_DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
        cross_norm_mean = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
        cross_norm_std  = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
        print(f"  Loaded (loss={cross_ckpt['loss']:.4f})")

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

            # Build signal dict: (B, C, T) for each sensor
            signals = {}
            for name in SENSOR_NAMES:
                x = batch[name].to(DEVICE).permute(0, 2, 1).float()  # (B, C, T_native)
                if USE_TCOMMON:
                    x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
                signals[name] = x

            if AUGMENT_CROSS and random.random() < AUG_PROB_CROSS:
                n_missing   = random.randint(1, 3)
                missing     = random.sample(SENSOR_NAMES, n_missing)
                missing_idx = [SENSOR_NAMES.index(s) for s in missing]
                with torch.no_grad():
                    # Stack all signals to (B, K, C, T_COMMON)
                    parts = []
                    for name in SENSOR_NAMES:
                        x = signals[name].float()           # (B, C, T_native)
                        x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
                        parts.append(x)
                    stacked = torch.stack(parts, dim=1)     # (B, K, C, T_COMMON)

                    # Normalize
                    stacked_norm = (stacked - cross_norm_mean[None, :, :, None]) \
                                 / cross_norm_std[None, :, :, None]

                    observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                    for i in missing_idx:
                        observed_mask[:, i] = 0.0

                    imputed_norm = ddim_impute_cross(
                        cross_diff_model, stacked_norm, observed_mask,
                        cross_alpha_bar, T_cross, DDIM_STEPS,
                    )
                    # Denormalize
                    imputed = imputed_norm * cross_norm_std[None, :, :, None] \
                            + cross_norm_mean[None, :, :, None]

                    for i in missing_idx:
                        name = SENSOR_NAMES[i]
                        gen  = imputed[:, i]   # (B, C, T_COMMON)
                        if USE_TCOMMON:
                            signals[name] = gen
                        else:
                            signals[name] = F.interpolate(
                                gen, size=NATIVE_LENS[name],
                                mode='linear', align_corners=False,
                            )

            logits = classifier(signals, SENSOR_NAMES)
            loss   = criterion(logits, labels)

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
                labels  = batch["label"].to(DEVICE)
                signals = {}
                for name in SENSOR_NAMES:
                    x = batch[name].to(DEVICE).permute(0, 2, 1).float()
                    if USE_TCOMMON:
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
                  f"| Acc: {acc:.4f} | LR: {lr_now:.2e}")

    print(f"\nBest Accuracy: {best_acc:.4f}")
    print(f"Saved to: {OUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
