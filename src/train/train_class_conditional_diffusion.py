# ============================================================
# Train Class-Conditional Diffusion for Sensor Latent Imputation
#
# For each sensor independently:
#   VAE V2 latent z + activity class → predict noise
#
# At inference time:
#   available sensors → C-LSTM-A → predicted class
#   → diffusion generates missing sensor latent conditioned on class
#   → VAE V2 decode → imputed signal
#
# Usage:
#   python -m src.train.train_class_conditional_diffusion
#   python -m src.train.train_class_conditional_diffusion --state
# ============================================================

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.class_conditional_diffusion import create_class_conditional_diffusion
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STATE = "--state" in sys.argv

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"
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

OUT_DIR = Path(f"checkpoints/class_cond_diffusion_{tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion
T          = 1000
SCHEDULE   = "cosine"
BATCH_SIZE = 64
EPOCHS     = 150
LR         = 3e-4

# Model
D_MODEL  = 128
N_BLOCKS = 6
EMB_DIM  = 128

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
# MAIN
# ============================================================
def main():
    print(f"\n{'='*65}")
    print(f"Training Class-Conditional Diffusion — {tag}")
    print(f"T={T}, Schedule={SCHEDULE}, Epochs={EPOCHS}")
    print(f"D_model={D_MODEL}, N_blocks={N_BLOCKS}")
    print(f"{'='*65}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Datasets
    if USE_STATE:
        train_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
        test_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing",  normalizer)
    else:
        train_ds = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
        test_ds  = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)

    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # VAE V2 (frozen)
    print(f"\nLoading VAE V2 from {VAE_CHECKPOINT}...")
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
    print(f"  latent_dim={latent_dim}, t_lat={t_lat}")

    # Noise schedule
    betas     = cosine_beta_schedule(T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)

    # Normalization stats per sensor (computed from training set)
    print("\nComputing latent normalization stats...")
    sums  = {k: 0.0 for k in SENSOR_NAMES}
    sq    = {k: 0.0 for k in SENSOR_NAMES}
    count = 0
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="norm stats", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                mu = outputs[k]["mu"]                    # (B, D, T)
                sums[k] = sums[k] + mu.mean(dim=(0, 2)) # (D,)
                sq[k]   = sq[k]   + (mu ** 2).mean(dim=(0, 2))
            count += 1
    norm_mean = {k: (sums[k] / count).to(DEVICE) for k in SENSOR_NAMES}
    norm_std  = {k: ((sq[k] / count - (sums[k] / count) ** 2).clamp(min=1e-6).sqrt()).to(DEVICE)
                 for k in SENSOR_NAMES}
    torch.save(
        {k: {"mean": norm_mean[k].cpu(), "std": norm_std[k].cpu()} for k in SENSOR_NAMES},
        OUT_DIR / "normalization_stats.pt",
    )

    # Diffusion model — one shared model for ALL sensors
    # (sensors in same latent space; class conditioning provides sensor identity implicitly)
    print(f"\nCreating Class-Conditional Diffusion model...")
    model = create_class_conditional_diffusion(
        latent_dim=latent_dim,
        t_lat=t_lat,
        n_classes=n_classes,
        d_model=D_MODEL,
        n_blocks=N_BLOCKS,
        emb_dim=EMB_DIM,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels      = batch["label"].to(DEVICE)   # (B,)

            with torch.no_grad():
                outputs = vae(sensor_data)
                # Pick a random sensor per batch step to train on
                name = SENSOR_NAMES[torch.randint(len(SENSOR_NAMES), (1,)).item()]
                mu = outputs[name]["mu"]  # (B, D, T_lat)
                # Normalize
                z0 = (mu - norm_mean[name][None, :, None]) / norm_std[name][None, :, None]

            B = z0.shape[0]
            t_step = torch.randint(0, T, (B,), device=DEVICE)
            noise  = torch.randn_like(z0)
            ab     = alpha_bar[t_step][:, None, None]
            z_t    = torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * noise

            noise_pred = model(z_t, t_step, labels)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        train_loss /= n_batches
        scheduler.step()

        # ---- Eval ----
        model.eval()
        eval_loss = 0.0
        n_eval    = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels      = batch["label"].to(DEVICE)

                outputs = vae(sensor_data)
                loss_sum = 0.0
                for name in SENSOR_NAMES:
                    mu = outputs[name]["mu"]
                    z0 = (mu - norm_mean[name][None, :, None]) / norm_std[name][None, :, None]
                    B  = z0.shape[0]
                    t_step = torch.randint(0, T, (B,), device=DEVICE)
                    noise  = torch.randn_like(z0)
                    ab     = alpha_bar[t_step][:, None, None]
                    z_t    = torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * noise
                    noise_pred = model(z_t, t_step, labels)
                    loss_sum += F.mse_loss(noise_pred, noise).item()
                eval_loss += loss_sum / len(SENSOR_NAMES)
                n_eval    += 1

        eval_loss /= n_eval

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} "
                  f"eval={eval_loss:.4f} | LR={lr_now:.2e}")

        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "loss":        eval_loss,
            "T":           T,
            "schedule":    SCHEDULE,
            "n_classes":   n_classes,
            "config": {
                "latent_dim": latent_dim,
                "t_lat":      t_lat,
                "d_model":    D_MODEL,
                "n_blocks":   N_BLOCKS,
                "emb_dim":    EMB_DIM,
                "n_classes":  n_classes,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best: {best_loss:.6f}")

    print(f"\n{'='*65}")
    print(f"Training done. Best eval loss: {best_loss:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
