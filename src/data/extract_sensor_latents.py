# ============================================================
# Extract Sensor-Level Latents (mu) from trained SensorMultiModalVAE
# ============================================================

import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64

CHECKPOINT_PATH = "checkpoints/sensor_vae_best.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUTPUT_DIR = Path("data/sensor_latents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================
print("Loading normalizer...")
normalizer = SensorNormalizer.load(NORMALIZER_PATH)

print("Loading datasets...")
dataset = ConcatDataset([
    CogAgeSensorDataset(DATA_ROOTS["blho"], "training", normalizer),
    CogAgeSensorDataset(DATA_ROOTS["bbh"], "training", normalizer),
    CogAgeSensorDataset(DATA_ROOTS["state"], "training", normalizer),
])

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=True, num_workers=4,
)

print(f"Total samples: {len(dataset)}")

print("Loading VAE checkpoint...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
vae = SensorMultiModalVAE().to(DEVICE)
vae.load_state_dict(ckpt["model_state"])
vae.eval()

for p in vae.parameters():
    p.requires_grad = False


# ============================================================
# EXTRACT
# ============================================================
print("Extracting temporal latents (mu) for each sensor...")

latents = {k: [] for k in SENSOR_NAMES}

with torch.no_grad():
    for batch in tqdm(loader, desc="Encoding"):
        sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
        outputs = vae(sensor_data)

        for key in SENSOR_NAMES:
            mu = outputs[key]["mu"]  # (B, 8, T')
            latents[key].append(mu.cpu())

# Concatenate
for key in SENSOR_NAMES:
    latents[key] = torch.cat(latents[key], dim=0)


# ============================================================
# SAVE
# ============================================================
print("\nSaving latents...")
for key in SENSOR_NAMES:
    path = OUTPUT_DIR / f"train_latents_{key}_mu.pt"
    torch.save(latents[key], path)

print(f"\n{'='*60}")
print("Sensor latent extraction finished.")
print(f"\nShapes:")
for key in SENSOR_NAMES:
    t = latents[key]
    print(f"  {key:15s}: {t.shape}  mean|mu|={t.abs().mean():.4f}, std={t.std():.4f}")
print(f"\nSaved to: {OUTPUT_DIR}")
print(f"{'='*60}")
