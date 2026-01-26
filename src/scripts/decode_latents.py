# ============================================================
# Decode sampled latents into sensor time series
# ============================================================

import torch
from pathlib import Path

from src.models.temporal_vae import TemporalMultiModalVAE

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENTS_PATH = "outputs/generated_latents.pt"
VAE_CKPT = "checkpoints/vae_gpu_epoch_050.pt"

OUT_DIR = Path("outputs/generated_timeseries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load VAE
# -------------------------
ckpt = torch.load(VAE_CKPT, map_location=DEVICE)

vae = TemporalMultiModalVAE(
    z_phone=32,
    z_watch=32,
    z_glasses=16,
).to(DEVICE)

vae.load_state_dict(ckpt["model_state"])
vae.eval()

# -------------------------
# Load latents
# -------------------------
z = torch.load(LATENTS_PATH).to(DEVICE)

# split latent
z_phone, z_watch, z_glasses = torch.split(z, [32, 32, 16], dim=1)

# expand temporal dims (simple broadcast)
z_phone = z_phone.unsqueeze(-1).repeat(1, 1, 100)     # T' ~ encoder output
z_watch = z_watch.unsqueeze(-1).repeat(1, 1, 34)
z_glasses = z_glasses.unsqueeze(-1).repeat(1, 1, 10)

with torch.no_grad():
    phone = vae.phone.decoder(z_phone)
    watch = vae.watch.decoder(z_watch)
    glasses = vae.glasses.decoder(z_glasses)

torch.save(
    {
        "phone": phone.cpu(),
        "watch": watch.cpu(),
        "glasses": glasses.cpu(),
    },
    OUT_DIR / "generated_timeseries.pt",
)

print("✅ Generated time series saved.")
print("phone :", phone.shape)
print("watch :", watch.shape)
print("glasses:", glasses.shape)
