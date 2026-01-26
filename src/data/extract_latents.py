# ============================================================
# Extract Latents (mu) from trained TemporalMultiModalVAE
# ============================================================

import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm

from src.models.temporal_vae import TemporalMultiModalVAE
from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64

CHECKPOINT_PATH = "checkpoints/vae_gpu_epoch_050.pt"
NORMALIZER_PATH = "data/combined_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUTPUT_DIR = Path("data/latents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "train_latents_mu.pt"


# ============================================================
# LOAD NORMALIZER
# ============================================================

print("Loading normalizer...")
normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)


# ============================================================
# DATASET
# ============================================================

print("Loading datasets...")

dataset = ConcatDataset([
    CogAgeVAEDataset(DATA_ROOTS["blho"], "training", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["bbh"], "training", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["state"], "training", normalizer),
])

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

print(f"Total samples: {len(dataset)}")


# ============================================================
# LOAD VAE
# ============================================================

print("Loading VAE checkpoint...")

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

vae = TemporalMultiModalVAE(
    z_phone=32,
    z_watch=32,
    z_glasses=16,
).to(DEVICE)

vae.load_state_dict(ckpt["model_state"])
vae.eval()

for p in vae.parameters():
    p.requires_grad = False


# ============================================================
# EXTRACT LATENTS (MU)
# ============================================================

print("Extracting latents (mu)...")

latents = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Encoding"):
        phone = batch["phone"].to(DEVICE, non_blocking=True)
        watch = batch["watch"].to(DEVICE, non_blocking=True)
        glasses = batch["glasses"].to(DEVICE, non_blocking=True)

        # IMPORTANT: use encoder mean (mu), NOT sampled z
        outputs = vae.encode(phone, watch, glasses)
        mu = outputs["mu"]              # shape: [B, z_dim]

        latents.append(mu.cpu())

latents = torch.cat(latents, dim=0)


# ============================================================
# SAVE
# ============================================================

torch.save(latents, OUTPUT_FILE)

print("========================================")
print("Latent extraction finished.")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Latent shape: {latents.shape}")
print(f"Mean |mu| : {latents.mean(dim=0).abs().mean():.4f}")
print(f"Mean std  : {latents.std(dim=0).mean():.4f}")
print("========================================")
