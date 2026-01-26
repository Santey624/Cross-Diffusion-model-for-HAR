# ============================================================
# train_latent_diffusion.py
# Train a DDPM-style diffusion model on VAE latents (mu)
# ============================================================

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENTS_PATH = "data/latents/train_latents_mu.pt"
OUT_DIR = Path("checkpoints/latent_diffusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Latent dimension (should be 80 for 32+32+16)
Z_DIM = 80

# Diffusion
T = 1000
BETA_START = 1e-4
BETA_END = 2e-2

# Training (Recommendet by Diffusion paper)
BATCH_SIZE = 256
EPOCHS = 200
LR = 2e-4 
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 10

# AMP
USE_AMP = True


# =========================
# UTIL: time embedding
# =========================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer timesteps
        returns: (B, dim)
        """
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =========================
# MODEL: simple MLP denoiser
# =========================
class LatentDenoiser(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 512, t_dim: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(t_dim)

        layers = []
        in_dim = z_dim + t_dim
        for i in range(depth - 1):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        layers += [nn.Linear(hidden, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t)
        x = torch.cat([z_t, t_emb], dim=1)
        return self.net(x)


# =========================
# DIFFUSION SCHEDULE (DDPM)
# =========================
def make_ddpm_schedule(T: int, beta_start: float, beta_end: float, device: str):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # for sampling
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - torch.cat([alpha_bar.new_ones(1), alpha_bar[:-1]])) / (1.0 - alpha_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


# q_sample: add noise
def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sched: dict) -> torch.Tensor:
    # gather per-sample coefficients
    sqrt_ab = sched["sqrt_alpha_bar"][t].unsqueeze(1)              # (B,1)
    sqrt_omb = sched["sqrt_one_minus_alpha_bar"][t].unsqueeze(1)   # (B,1)
    return sqrt_ab * z0 + sqrt_omb * noise


# =========================
# TRAIN
# =========================
def main():
    assert Path(LATENTS_PATH).exists(), f"Missing latents: {LATENTS_PATH}"

    print("Loading latents...")
    z = torch.load(LATENTS_PATH)  # [N, Z_DIM]
    if isinstance(z, dict) and "latents" in z:
        z = z["latents"]
    assert z.ndim == 2, f"Expected latents [N,D], got {tuple(z.shape)}"
    assert z.shape[1] == Z_DIM, f"Expected Z_DIM={Z_DIM}, got {z.shape[1]}"

    # OPTIONAL: Standardize latents to improve diffusion stability
    z_mean = z.mean(dim=0, keepdim=True)
    z_std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
    z_norm = (z - z_mean) / z_std

    # Save stats for later sampling/decoding
    stats_path = OUT_DIR / "latent_stats.pt"
    torch.save({"mean": z_mean, "std": z_std}, stats_path)
    print(f"Saved latent stats to: {stats_path}")

    ds = TensorDataset(z_norm)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    model = LatentDenoiser(z_dim=Z_DIM, hidden=512, t_dim=256, depth=4, dropout=0.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    sched = make_ddpm_schedule(T, BETA_START, BETA_END, device=DEVICE)

    print("Training latent diffusion...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for (z0,) in tqdm(dl, desc=f"[Train] Epoch {epoch}/{EPOCHS}"):
            z0 = z0.to(DEVICE, non_blocking=True)

            t = torch.randint(0, T, (z0.size(0),), device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(z0)

            z_t = q_sample(z0, t, noise, sched)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                noise_pred = model(z_t, t)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            running += loss.item()

        avg_loss = running / len(dl)
        print(f"\nEpoch {epoch:03d} | loss={avg_loss:.6f}\n")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt_path = OUT_DIR / f"latent_ddpm_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "T": T,
                    "beta_start": BETA_START,
                    "beta_end": BETA_END,
                    "Z_DIM": Z_DIM,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print("✅ Latent diffusion training finished.")


if __name__ == "__main__":
    main()
