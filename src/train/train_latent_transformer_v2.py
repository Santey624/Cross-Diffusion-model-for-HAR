# ============================================================
# Train Latent Transformer/MLP on VAE V2 Latents — Robust
#
# Flags:
#   --model transformer|mlp  (default: transformer)
#   --state                  State activities (6 classes)
#   --robust                 50% batches with diffusion imputation
# ============================================================

import sys
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

MODEL_TYPE = "transformer" if "--model" not in sys.argv else \
             sys.argv[sys.argv.index("--model") + 1]
ROBUST    = "--robust" in sys.argv
USE_STATE = "--state"  in sys.argv
USE_V1    = "--v1"     in sys.argv

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt" if USE_V1 \
                 else "checkpoints/sensor_vae_v2/best_model.pt"
vae_tag = "v1" if USE_V1 else "v2"

DIFFUSION_DIR = Path("checkpoints/sensor_diffusion_v3") if USE_V1 \
                else Path("checkpoints/sensor_diffusion_v3_v2")

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
    EPOCHS = 150
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }
    EPOCHS = 100

tag     = f"{MODEL_TYPE}_{'state' if USE_STATE else 'behavioral'}{'_robust' if ROBUST else ''}_{vae_tag}"
OUT_DIR = Path(f"checkpoints/latent_{tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LR         = 1e-3
MASK_PROB  = 0.5
DDIM_STEPS = 20   # fewer steps during training (faster, good enough for augmentation)

# Transformer — SMALL params (fix for behavioral not converging)
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 2
DROPOUT  = 0.3


# ============================================================
# DDIM HELPERS
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    ab    = f_t / f_t[0]
    return torch.clamp(1 - ab[1:] / ab[:-1], 1e-6, 0.999).float()


@torch.no_grad()
def ddim_impute(model, stacked, observed_mask, alpha_bar, T, steps=20):
    """Impute missing sensors (observed_mask=0) via DDIM."""
    B, K, D, Tl = stacked.shape
    device  = stacked.device
    missing = 1.0 - observed_mask
    alpha_bar = alpha_bar.to(device)

    z   = observed_mask[:, :, None, None] * stacked + \
          missing[:, :, None, None] * torch.randn_like(stacked)
    tau = (torch.linspace(0, 1, steps + 1, device=device) ** 2 * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_b    = torch.full((B,), t_now, device=device, dtype=torch.long)
        noisy  = observed_mask[:, :, None, None] * stacked + missing[:, :, None, None] * z
        noise_pred = model(noisy, t_b, observed_mask)
        ab_now, ab_next = alpha_bar[t_now], alpha_bar[t_next]
        pred_x0 = ((z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)).clamp(-5, 5)
        z_new   = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
        z = observed_mask[:, :, None, None] * stacked + missing[:, :, None, None] * z_new

    return z


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

    # Load VAE (V1 or V2)
    ckpt_vae = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    if USE_V1:
        latent_dim, t_shared = 8, 32
        print(f"VAE V1: latent_dim={latent_dim}, t_shared={t_shared}")
        vae = SensorMultiModalVAE().to(DEVICE)
    else:
        cfg = ckpt_vae["config"]
        latent_dim, t_shared = cfg["latent_dim"], cfg["t_shared"]
        print(f"VAE V2: latent_dim={latent_dim}, t_shared={t_shared}")
        vae = SensorMultiModalVAE(latent_dim=latent_dim, t_shared=t_shared).to(DEVICE)

    vae.load_state_dict(ckpt_vae["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Load diffusion model for robust imputation
    diffusion  = None
    norm_stats = None
    alpha_bar  = None
    diff_T     = None
    if ROBUST:
        print(f"Loading diffusion from {DIFFUSION_DIR}...")
        diff_ckpt  = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
        dcfg       = diff_ckpt["config"]
        diffusion  = create_sensor_diffusion_v3(
            d_model=dcfg["d_model"], num_heads=dcfg["num_heads"],
            num_blocks=dcfg["num_blocks"], dropout=0.0,
        ).to(DEVICE)
        diffusion.load_state_dict(diff_ckpt["model_state"])
        diffusion.eval()
        norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
        diff_T     = diff_ckpt["T"]
        betas      = cosine_beta_schedule(diff_T) if diff_ckpt["schedule"] == "cosine" \
                     else torch.linspace(1e-4, 0.02, diff_T)
        alpha_bar  = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)
        print(f"  Diffusion loaded (loss={diff_ckpt.get('loss', '?'):.4f})")

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

            # Robust: randomly impute missing sensors via diffusion
            if ROBUST and random.random() < MASK_PROB:
                n_miss  = random.randint(1, 3)
                missing = random.sample(SENSOR_NAMES, n_miss)

                latents_norm = {
                    n: (latents[n] - norm_stats[n]["mean"].to(DEVICE))
                       / norm_stats[n]["std"].to(DEVICE)
                    for n in SENSOR_NAMES
                }
                stacked = torch.stack([latents_norm[n] for n in SENSOR_NAMES], dim=1)
                observed = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for name in missing:
                    observed[:, SENSOR_NAMES.index(name)] = 0.0

                imputed = ddim_impute(diffusion, stacked, observed, alpha_bar, diff_T, DDIM_STEPS)

                for i, name in enumerate(SENSOR_NAMES):
                    if name in missing:
                        latents[name] = (imputed[:, i]
                                         * norm_stats[name]["std"].to(DEVICE)
                                         + norm_stats[name]["mean"].to(DEVICE))

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
