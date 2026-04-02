# ============================================================
# Extract WISDM Latents V3 (D=16, T=64) using VAE V3
# ============================================================

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE      = 64
CHECKPOINT_PATH = "checkpoints/sensor_vae_v3/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
WISDM_ROOT      = "data/wisdm/arrays"

OUTPUT_DIR = Path("data/wisdm_latents_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}
ZERO_SENSORS = {"phone_grav", "phone_lacc", "glasses_acc"}


def main():
    print(f"\n{'='*60}")
    print("Extracting WISDM Latents V3 (D=16, T=64)")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print("Loading WISDM dataset...")
    dataset = CogAgeSensorDataset(WISDM_ROOT, "training", normalizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         pin_memory=True, num_workers=4)
    print(f"WISDM train samples: {len(dataset)}")

    print("Loading VAE V3...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg  = ckpt["config"]
    print(f"  latent_dim={cfg['latent_dim']}, t_shared={cfg['t_shared']}, "
          f"epoch={ckpt['epoch']}, recon={ckpt['test_recon']:.6f}, "
          f"align_weight={cfg['align_weight']}")

    vae = SensorMultiModalVAE(latent_dim=cfg["latent_dim"], t_shared=cfg["t_shared"]).to(DEVICE)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("Extracting latents...")
    latents = {k: [] for k in SENSOR_NAMES}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding WISDM"):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for key in SENSOR_NAMES:
                latents[key].append(outputs[key]["mu"].cpu())

    for key in SENSOR_NAMES:
        latents[key] = torch.cat(latents[key], dim=0)

    print("\nSaving latents...")
    for key in SENSOR_NAMES:
        torch.save(latents[key], OUTPUT_DIR / f"train_latents_{key}_mu.pt")
        marker = "REAL" if key in REAL_SENSORS else "ZERO-INPUT"
        print(f"  {key:15s}: {latents[key].shape} | "
              f"mean={latents[key].mean():.4f}, std={latents[key].std():.4f} [{marker}]")

    torch.save({
        "n_samples": len(dataset),
        "real_sensors": list(REAL_SENSORS),
        "zero_sensors": list(ZERO_SENSORS),
    }, OUTPUT_DIR / "metadata.pt")

    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
