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

# Save temporal latents separately for each modality
OUTPUT_FILE_PHONE = OUTPUT_DIR / "train_latents_phone_mu.pt"
OUTPUT_FILE_WATCH = OUTPUT_DIR / "train_latents_watch_mu.pt"
OUTPUT_FILE_GLASSES = OUTPUT_DIR / "train_latents_glasses_mu.pt"


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
# EXTRACT TEMPORAL LATENTS (MU) - NO POOLING
# ============================================================

print("Extracting temporal latents (mu) for each modality...")

latents_phone = []
latents_watch = []
latents_glasses = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Encoding"):
        phone = batch["phone"].to(DEVICE, non_blocking=True)
        watch = batch["watch"].to(DEVICE, non_blocking=True)
        glasses = batch["glasses"].to(DEVICE, non_blocking=True)

        # Forward pass - returns nested dict
        outputs = vae(phone, watch, glasses)

        # Extract temporal mu (B, D, T') for each modality - NO temporal pooling
        mu_phone = outputs["phone"]["mu"]      # (B, 32, T_phone')
        mu_watch = outputs["watch"]["mu"]      # (B, 32, T_watch')
        mu_glasses = outputs["glasses"]["mu"]  # (B, 16, T_glasses')

        latents_phone.append(mu_phone.cpu())
        latents_watch.append(mu_watch.cpu())
        latents_glasses.append(mu_glasses.cpu())

# Concatenate batches
latents_phone = torch.cat(latents_phone, dim=0)    # (N, 32, T_phone')
latents_watch = torch.cat(latents_watch, dim=0)    # (N, 32, T_watch')
latents_glasses = torch.cat(latents_glasses, dim=0)  # (N, 16, T_glasses')


# ============================================================
# SAVE TEMPORAL LATENTS SEPARATELY
# ============================================================

torch.save(latents_phone, OUTPUT_FILE_PHONE)
torch.save(latents_watch, OUTPUT_FILE_WATCH)
torch.save(latents_glasses, OUTPUT_FILE_GLASSES)

print("========================================")
print("Temporal latent extraction finished.")
print(f"\nSaved temporal latents to:")  # noqa: F541
print(f"  Phone:   {OUTPUT_FILE_PHONE}")
print(f"  Watch:   {OUTPUT_FILE_WATCH}")
print(f"  Glasses: {OUTPUT_FILE_GLASSES}")

print(f"\nShapes:")# noqa: F541
print(f"  Phone latents:   {latents_phone.shape}   (N, 32, T_phone')")
print(f"  Watch latents:   {latents_watch.shape}   (N, 32, T_watch')")
print(f"  Glasses latents: {latents_glasses.shape} (N, 16, T_glasses')")

print(f"\nStatistics:")# noqa: F541
print(f"  Phone   - mean |mu|: {latents_phone.abs().mean():.4f}, std: {latents_phone.std():.4f}")
print(f"  Watch   - mean |mu|: {latents_watch.abs().mean():.4f}, std: {latents_watch.std():.4f}")
print(f"  Glasses - mean |mu|: {latents_glasses.abs().mean():.4f}, std: {latents_glasses.std():.4f}")
print("========================================")
