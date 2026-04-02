# ============================================================
# Extract Sensor Latents V3 (D=16, T=64)
# from trained SensorMultiModalVAE V3 (strong alignment)
# ============================================================

import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/sensor_vae_v3/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
BATCH_SIZE      = 64

DATA_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUTPUT_DIR = Path("data/sensor_latents_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract(split):
    dataset = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"],  split, normalizer),
        CogAgeSensorDataset(DATA_ROOTS["bbh"],   split, normalizer),
        CogAgeSensorDataset(DATA_ROOTS["state"], split, normalizer),
    ])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=True, num_workers=4)

    latents = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {split}"):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for key in SENSOR_NAMES:
                latents[key].append(outputs[key]["mu"].cpu())

    for key in SENSOR_NAMES:
        latents[key] = torch.cat(latents[key], dim=0)
        path = OUTPUT_DIR / f"{split}_latents_{key}_mu.pt"
        torch.save(latents[key], path)

    print(f"\n{split} shapes:")
    for key in SENSOR_NAMES:
        t = latents[key]
        print(f"  {key:15s}: {t.shape}  mean|mu|={t.abs().mean():.4f}  std={t.std():.4f}")


# ============================================================
# MAIN
# ============================================================
print("Loading normalizer...")
normalizer = SensorNormalizer.load(NORMALIZER_PATH)

print("Loading VAE V3 checkpoint...")
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

extract("training")
extract("testing")

print(f"\nSaved to: {OUTPUT_DIR}")
