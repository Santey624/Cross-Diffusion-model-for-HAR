# ============================================================
# Train C-LSTM-A on Raw Sensor Signals (no VAE)
#
# Baseline classifier that works directly on normalized signals.
# AdaptiveAvgPool handles different sensor lengths automatically.
#
# Flags:
#   --state          use state dataset (6 classes)
#   --augment        50% of batches: missing sensors via signal class-cond diffusion
#   --augment-cross  30% of batches: missing sensors via cross-sensor signal diffusion
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
from src.models.signal_class_diffusion import (
    create_signal_class_diffusion, SENSOR_T, T_MODEL,
)
from src.models.signal_cross_diffusion import create_signal_cross_diffusion, T_COMMON
from src.models.sensor_vae import SENSOR_NAMES
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE      = "--state"         in sys.argv
AUGMENT        = "--augment"       in sys.argv
AUGMENT_CROSS  = "--augment-cross" in sys.argv

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
tag = "state" if USE_STATE else "behavioral"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

if AUGMENT_CROSS:
    suffix = "_augment_cross"
elif AUGMENT:
    suffix = "_augment"
else:
    suffix = ""
OUT_DIR = Path(f"checkpoints/clstm_raw_{tag}{suffix}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIFF_DIR       = Path(f"checkpoints/signal_class_diffusion_{tag}")
CROSS_DIFF_DIR = Path("checkpoints/signal_cross_diffusion")
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
AUG_PROB    = 0.5
AUG_PROB_CROSS = 0.3
DDIM_STEPS  = 10
NUM_WORKERS = 4


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
def ddim_sample_signal(model, class_label, sensor_id, alpha_bar, T, ddim_steps,
                        t_sensor=T_MODEL):
    """Returns (B, 3, t_sensor)"""
    B      = class_label.shape[0]
    device = class_label.device
    x      = torch.randn(B, 3, t_sensor, device=device)
    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)
    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch    = torch.full((B,), t_now, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch, class_label, sensor_id)
        ab_now     = alpha_bar[t_now]
        ab_next    = alpha_bar[t_next]
        pred_x0    = ((x - torch.sqrt(1 - ab_now) * noise_pred)
                      / torch.sqrt(ab_now)).clamp(-5, 5)
        x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
    return x


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


def prepare_signal(x, target_len):
    """(B, T, C) → (B, C, target_len)"""
    x = x.permute(0, 2, 1).float()
    return F.interpolate(x, size=target_len, mode='linear', align_corners=False)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training C-LSTM-A on Raw Signals — {tag}")
    print(f"Augment: {AUGMENT} | AugmentCross: {AUGMENT_CROSS} | Device: {DEVICE}")
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

    # Signal diffusion (only if augmenting)
    diff_model = alpha_bar = T_diff = None
    cross_diff_model = cross_alpha_bar = T_cross = None
    cross_norm_mean = cross_norm_std = None

    if AUGMENT:
        print(f"Loading Signal Class-Cond Diffusion from {DIFF_DIR}...")
        diff_ckpt  = torch.load(DIFF_DIR / "best_model.pt", map_location=DEVICE)
        cfg_diff   = diff_ckpt["config"]
        diff_model = create_signal_class_diffusion(
            n_sensors=cfg_diff["n_sensors"],
            n_classes=cfg_diff["n_classes"],
            in_channels=cfg_diff["in_channels"],
            base_ch=cfg_diff["base_ch"],
            emb_dim=cfg_diff["emb_dim"],
        ).to(DEVICE)
        diff_model.load_state_dict(diff_ckpt["model_state"])
        diff_model.eval()
        for p in diff_model.parameters():
            p.requires_grad = False
        T_diff    = diff_ckpt["T"]
        betas     = cosine_beta_schedule(T_diff)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        print(f"  Loaded (loss={diff_ckpt['loss']:.4f})")

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

    # Native signal lengths per sensor (for interpolation back)
    from src.models.sensor_vae import SENSOR_SPECS
    native_lens = {name: SENSOR_SPECS[name]["seq_len"] for name in SENSOR_NAMES}

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            labels = batch["label"].to(DEVICE)
            B      = labels.size(0)

            # Build decoded dict: (B, C, T_native) for each sensor
            signals = {}
            for name in SENSOR_NAMES:
                x = batch[name].to(DEVICE)         # (B, T, C)
                signals[name] = x.permute(0, 2, 1) # (B, C, T)

            if AUGMENT and random.random() < AUG_PROB:
                n_missing  = random.randint(1, 3)
                missing    = random.sample(SENSOR_NAMES, n_missing)
                with torch.no_grad():
                    for name in missing:
                        sidx = torch.full((B,), SENSOR_NAMES.index(name),
                                          dtype=torch.long, device=DEVICE)
                        t_sensor = SENSOR_T.get(name, T_MODEL)
                        gen = ddim_sample_signal(
                            diff_model, labels, sidx,
                            alpha_bar, T_diff, DDIM_STEPS,
                            t_sensor=t_sensor,
                        )   # (B, 3, t_sensor)
                        # Interpolate to native length if needed
                        if gen.shape[-1] != native_lens[name]:
                            gen = F.interpolate(gen, size=native_lens[name],
                                                mode='linear', align_corners=False)
                        signals[name] = gen

            elif AUGMENT_CROSS and random.random() < AUG_PROB_CROSS:
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
                        signals[name] = F.interpolate(
                            gen, size=native_lens[name],
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
                signals = {
                    name: batch[name].to(DEVICE).permute(0, 2, 1)
                    for name in SENSOR_NAMES
                }
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
