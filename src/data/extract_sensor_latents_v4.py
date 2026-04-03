# ============================================================
# Extract Sensor Latents V4 (Shared+Private VAE)
#
# Saves BOTH z_shared and z_private for each sensor.
# Diffusion / correlation analysis uses z_shared.
# Classification can use z_shared (activity) or z_shared+z_private.
# ============================================================

import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm

from src.models.sensor_vae_v4 import SensorSharedPrivateVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/sensor_vae_v4/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
BATCH_SIZE      = 64

DATA_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUTPUT_DIR = Path("data/sensor_latents_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract(split):
    dataset = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"],  split, normalizer),
        CogAgeSensorDataset(DATA_ROOTS["bbh"],   split, normalizer),
        CogAgeSensorDataset(DATA_ROOTS["state"], split, normalizer),
    ])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=True, num_workers=4)

    shared_lats  = {k: [] for k in SENSOR_NAMES}
    private_lats = {k: [] for k in SENSOR_NAMES}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {split}"):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
            # Use all sensors for PoE (no masking at inference)
            outputs, mu_shared, _ = vae(sensor_data, mask_ratio=0.0)

            for key in SENSOR_NAMES:
                # z_shared is the same for all sensors per sample (from PoE)
                # mu_s per-sensor is also stored for analysis
                shared_lats[key].append(outputs[key]["mu_s"].cpu())
                private_lats[key].append(outputs[key]["mu_p"].cpu())

    print(f"\n{split} shapes:")
    for key in SENSOR_NAMES:
        zs = torch.cat(shared_lats[key],  dim=0)
        zp = torch.cat(private_lats[key], dim=0)

        torch.save(zs, OUTPUT_DIR / f"{split}_latents_{key}_mu.pt")       # shared (used by corr check)
        torch.save(zp, OUTPUT_DIR / f"{split}_latents_{key}_private.pt")  # private

        print(f"  {key:15s}: shared={zs.shape} |μ|={zs.abs().mean():.4f}  "
              f"private={zp.shape} |μ|={zp.abs().mean():.4f}")


# ============================================================
# MAIN
# ============================================================
print("Loading normalizer...")
normalizer = SensorNormalizer.load(NORMALIZER_PATH)

print("Loading VAE V4 checkpoint...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
cfg  = ckpt["config"]
print(f"  d_shared={cfg['d_shared']}, d_private={cfg['d_private']}, t_lat={cfg['t_lat']}, "
      f"epoch={ckpt['epoch']}, recon={ckpt['test_recon']:.6f}")

vae = SensorSharedPrivateVAE(
    d_shared=cfg["d_shared"], d_private=cfg["d_private"], t_lat=cfg["t_lat"]
).to(DEVICE)
vae.load_state_dict(ckpt["model_state"])
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

extract("training")
extract("testing")

print(f"\nSaved to: {OUTPUT_DIR}")
print("NOTE: *_mu.pt = per-sensor shared encoder output (mu_s)")
print("      *_private.pt = per-sensor private encoder output (mu_p)")
print("Run check_latent_correlation --v4 to verify R² of z_shared")
