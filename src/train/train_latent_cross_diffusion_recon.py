# ============================================================
# Train Latent Cross-Sensor Diffusion with Reconstruction Loss
#
# Key idea: train diffusion with BOTH:
#   1. Noise-prediction loss (standard diffusion)
#   2. Signal reconstruction loss:
#      z0_pred = (z_t - sqrt(1-ab)*eps_pred) / sqrt(ab)  [differentiable]
#      recon   = VAE.decode(z0_pred)
#      loss    = MSE(recon, original_signal)
#
# VAE is frozen. Gradient flows: recon_loss → decoder → z0_pred → diffusion.
#
# Usage:
#   python -m src.train.train_latent_cross_diffusion_recon
# ============================================================

import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.models.latent_cross_diffusion_recon import create_latent_cross_diffusion_recon
from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"

COGAGE_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUT_DIR = Path("checkpoints/latent_cross_diffusion_recon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T           = 1000
BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 3e-4
D_MODEL     = 128
NUM_HEADS   = 4
NUM_BLOCKS  = 6
DROPOUT     = 0.1
MIN_MISSING = 1
MAX_MISSING = 3
NUM_WORKERS = 4

# Loss weights: total = noise_loss + LAMBDA_RECON * recon_loss
LAMBDA_RECON = 1.0


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
    print(f"Training Latent Cross-Sensor Diffusion + Recon Loss")
    print(f"T={T}, Epochs={EPOCHS}, lambda_recon={LAMBDA_RECON}")
    print(f"D_model={D_MODEL}, Blocks={NUM_BLOCKS}, Heads={NUM_HEADS}")
    print(f"{'='*65}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    behavioral_train = get_combined_labeled_dataset(
        {"blho": COGAGE_ROOTS["blho"], "bbh": COGAGE_ROOTS["bbh"]}, "training", normalizer,
    )
    behavioral_test = get_combined_labeled_dataset(
        {"blho": COGAGE_ROOTS["blho"], "bbh": COGAGE_ROOTS["bbh"]}, "testing", normalizer,
    )
    state_train = CogAgeLabeledDataset(COGAGE_ROOTS["state"], "training", normalizer)
    state_test  = CogAgeLabeledDataset(COGAGE_ROOTS["state"], "testing",  normalizer)

    train_ds = ConcatDataset([behavioral_train, state_train])
    test_ds  = ConcatDataset([behavioral_test,  state_test])
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Load VAE V2 — frozen
    print(f"Loading VAE V2 from {VAE_CHECKPOINT}...")
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
    t_shared   = cfg_vae["t_shared"]
    print(f"  VAE frozen: latent_dim={latent_dim}, t_shared={t_shared}")

    # Noise schedule
    betas     = cosine_beta_schedule(T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).to(DEVICE)

    # Diffusion model
    print(f"\nCreating Latent Cross-Sensor Diffusion...")
    model = create_latent_cross_diffusion_recon(
        n_sensors=len(SENSOR_NAMES),
        latent_dim=latent_dim,
        t_shared=t_shared,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_noise_loss = 0.0
        train_recon_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            sensor_data = {
                name: batch[name].to(DEVICE).float()
                for name in SENSOR_NAMES
            }
            B = next(iter(sensor_data.values())).shape[0]
            K = len(SENSOR_NAMES)

            # Encode all sensors → latents (no grad through VAE)
            with torch.no_grad():
                mus = []
                for name in SENSOR_NAMES:
                    mu, _ = vae.encode_sensor(name, sensor_data[name])
                    mus.append(mu)
                latents = torch.stack(mus, dim=1)   # (B, K, D, T_SHARED)

            # Random missing mask
            n_missing   = random.randint(MIN_MISSING, MAX_MISSING)
            missing_idx = random.sample(range(K), n_missing)
            observed_mask = torch.ones(B, K, device=DEVICE)
            for i in missing_idx:
                observed_mask[:, i] = 0.0

            # Add noise to missing latents only
            t_step = torch.randint(0, T, (B,), device=DEVICE)
            noise  = torch.randn_like(latents)
            ab     = alpha_bar[t_step][:, None, None, None]   # (B, 1, 1, 1)

            noisy = latents.clone()
            for i in missing_idx:
                noisy[:, i] = (torch.sqrt(ab) * latents[:, i]
                               + torch.sqrt(1 - ab) * noise[:, i])

            # Predict noise
            noise_pred = model(noisy, t_step, observed_mask)  # (B, K, D, T_SHARED)

            # 1) Noise prediction loss (missing sensors only)
            noise_loss = sum(
                F.mse_loss(noise_pred[:, i], noise[:, i])
                for i in missing_idx
            ) / len(missing_idx)

            # 2) Signal reconstruction loss:
            #    recover z0_pred from noise_pred (differentiable),
            #    decode with frozen VAE, compare to original signal
            recon_loss = torch.tensor(0.0, device=DEVICE)
            for i in missing_idx:
                # z0_pred: (B, D, T_SHARED)
                z0_pred = ((noisy[:, i] - torch.sqrt(1 - ab[:, 0]) * noise_pred[:, i])
                           / torch.sqrt(ab[:, 0])).clamp(-10, 10)
                name    = SENSOR_NAMES[i]
                recon   = vae.decode_sensor(name, z0_pred)   # (B, T, C)
                target  = sensor_data[name]                   # (B, T, C)
                recon_loss = recon_loss + F.mse_loss(recon, target)
            recon_loss = recon_loss / len(missing_idx)

            loss = noise_loss + LAMBDA_RECON * recon_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_noise_loss += noise_loss.item()
            train_recon_loss += recon_loss.item()
            n_batches += 1

        train_noise_loss /= n_batches
        train_recon_loss /= n_batches
        scheduler.step()

        # Eval
        model.eval()
        eval_noise_loss = 0.0
        eval_recon_loss = 0.0
        n_eval = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                sensor_data = {
                    name: batch[name].to(DEVICE).float()
                    for name in SENSOR_NAMES
                }
                B = next(iter(sensor_data.values())).shape[0]
                K = len(SENSOR_NAMES)

                mus = []
                for name in SENSOR_NAMES:
                    mu, _ = vae.encode_sensor(name, sensor_data[name])
                    mus.append(mu)
                latents = torch.stack(mus, dim=1)

                n_missing   = random.randint(MIN_MISSING, MAX_MISSING)
                missing_idx = random.sample(range(K), n_missing)
                observed_mask = torch.ones(B, K, device=DEVICE)
                for i in missing_idx:
                    observed_mask[:, i] = 0.0

                t_step = torch.randint(0, T, (B,), device=DEVICE)
                noise  = torch.randn_like(latents)
                ab     = alpha_bar[t_step][:, None, None, None]

                noisy = latents.clone()
                for i in missing_idx:
                    noisy[:, i] = (torch.sqrt(ab) * latents[:, i]
                                   + torch.sqrt(1 - ab) * noise[:, i])

                noise_pred = model(noisy, t_step, observed_mask)

                noise_loss = sum(
                    F.mse_loss(noise_pred[:, i], noise[:, i])
                    for i in missing_idx
                ) / len(missing_idx)

                recon_loss = torch.tensor(0.0, device=DEVICE)
                for i in missing_idx:
                    z0_pred = ((noisy[:, i] - torch.sqrt(1 - ab[:, 0]) * noise_pred[:, i])
                               / torch.sqrt(ab[:, 0])).clamp(-10, 10)
                    name   = SENSOR_NAMES[i]
                    recon  = vae.decode_sensor(name, z0_pred)
                    target = sensor_data[name]
                    recon_loss = recon_loss + F.mse_loss(recon, target)
                recon_loss = recon_loss / len(missing_idx)

                eval_noise_loss += noise_loss.item()
                eval_recon_loss += recon_loss.item()
                n_eval += 1

        eval_noise_loss /= n_eval
        eval_recon_loss /= n_eval
        eval_total = eval_noise_loss + LAMBDA_RECON * eval_recon_loss

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_noise={train_noise_loss:.4f} train_recon={train_recon_loss:.6f} | "
                  f"eval_noise={eval_noise_loss:.4f} eval_recon={eval_recon_loss:.6f} | "
                  f"LR={lr_now:.2e}")

        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "loss":        eval_total,
            "noise_loss":  eval_noise_loss,
            "recon_loss":  eval_recon_loss,
            "T":           T,
            "config": {
                "n_sensors":   len(SENSOR_NAMES),
                "latent_dim":  latent_dim,
                "t_shared":    t_shared,
                "d_model":     D_MODEL,
                "num_heads":   NUM_HEADS,
                "num_blocks":  NUM_BLOCKS,
                "dropout":     DROPOUT,
                "lambda_recon": LAMBDA_RECON,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if eval_total < best_loss:
            best_loss = eval_total
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best: noise={eval_noise_loss:.4f} recon={eval_recon_loss:.6f}")

    print(f"\n{'='*65}")
    print(f"Done. Best eval loss: {best_loss:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
