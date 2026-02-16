# ============================================================
# Extract Latents from WISDM dataset using trained VAE
# Only phone_acc, phone_gyro, watch_acc, watch_gyro have real data
# phone_grav, phone_lacc, glasses_acc are zero-filled (skipped)
# ============================================================

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

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

WISDM_ROOT = "data/wisdm/arrays"
OUTPUT_DIR = Path("data/wisdm_latents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These sensors have real WISDM data
REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}
# These are zero-filled placeholders
ZERO_SENSORS = {"phone_grav", "phone_lacc", "glasses_acc"}


def main():
    print(f"\n{'='*60}")
    print("Extracting WISDM Latents")
    print(f"{'='*60}\n")

    # Load normalizer (same as CogAge)
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load WISDM training data
    print("Loading WISDM dataset...")
    dataset = CogAgeSensorDataset(WISDM_ROOT, "training", normalizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=True, num_workers=4)
    print(f"WISDM train samples: {len(dataset)}")

    # Load VAE
    print("Loading VAE...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    vae = SensorMultiModalVAE().to(DEVICE)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Extract latents
    print("Extracting latents...")
    latents = {k: [] for k in SENSOR_NAMES}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding WISDM"):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)

            for key in SENSOR_NAMES:
                mu = outputs[key]["mu"]  # (B, 8, 32)
                latents[key].append(mu.cpu())

    # Concatenate
    for key in SENSOR_NAMES:
        latents[key] = torch.cat(latents[key], dim=0)

    # Save only REAL sensors (the ones with actual WISDM data)
    print("\nSaving latents...")
    for key in SENSOR_NAMES:
        path = OUTPUT_DIR / f"train_latents_{key}_mu.pt"
        is_real = key in REAL_SENSORS

        if is_real:
            torch.save(latents[key], path)
            print(f"  {key:15s}: {latents[key].shape} | "
                  f"mean={latents[key].mean():.4f}, std={latents[key].std():.4f} [REAL]")
        else:
            # Save but mark as zero-filled (latents from zero input)
            torch.save(latents[key], path)
            print(f"  {key:15s}: {latents[key].shape} | "
                  f"mean={latents[key].mean():.4f}, std={latents[key].std():.4f} [ZERO-INPUT]")

    # Save metadata
    metadata = {
        "n_samples": len(dataset),
        "real_sensors": list(REAL_SENSORS),
        "zero_sensors": list(ZERO_SENSORS),
    }
    torch.save(metadata, OUTPUT_DIR / "metadata.pt")

    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
