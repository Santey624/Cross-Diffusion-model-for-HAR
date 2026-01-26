# ============================================================
# Sample latents from trained latent diffusion model
# ============================================================

import math
from pathlib import Path
import torch
import torch.nn.functional as F

from src.train.train_latent_diffusion import (
    LatentDenoiser,
    make_ddpm_schedule,
)

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "checkpoints/latent_diffusion/latent_ddpm_epoch_200.pt"
STATS_PATH = "checkpoints/latent_diffusion/latent_stats.pt"

OUT_PATH = Path("outputs/generated_latents.pt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 64
Z_DIM = 80
T = 1000


# -------------------------
# Load model
# -------------------------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

model = LatentDenoiser(
    z_dim=Z_DIM,
    hidden=512,
    t_dim=256,
    depth=4,
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------------
# Load stats
# -------------------------
stats = torch.load(STATS_PATH)
z_mean = stats["mean"].to(DEVICE)
z_std = stats["std"].to(DEVICE)

# -------------------------
# Diffusion schedule
# -------------------------
sched = make_ddpm_schedule(
    T=T,
    beta_start=ckpt["beta_start"],
    beta_end=ckpt["beta_end"],
    device=DEVICE,
)

# -------------------------
# Sampling loop
# -------------------------
z = torch.randn(N_SAMPLES, Z_DIM, device=DEVICE)

with torch.no_grad():
    for t in reversed(range(T)):
        t_batch = torch.full((N_SAMPLES,), t, device=DEVICE, dtype=torch.long)

        eps = model(z, t_batch)

        beta = sched["betas"][t]
        sqrt_one_minus_ab = sched["sqrt_one_minus_alpha_bar"][t]
        sqrt_recip_alpha = sched["sqrt_recip_alphas"][t]

        z = sqrt_recip_alpha * (z - beta / sqrt_one_minus_ab * eps)

        if t > 0:
            noise = torch.randn_like(z)
            z = z + torch.sqrt(sched["posterior_variance"][t]) * noise

# de-normalize
z = z * z_std + z_mean

torch.save(z.cpu(), OUT_PATH)
print(f"✅ Saved sampled latents: {OUT_PATH}")
print(f"Shape: {z.shape}")
