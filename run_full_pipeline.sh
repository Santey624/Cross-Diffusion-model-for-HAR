#!/bin/bash
# ============================================================
# Full Pipeline: WISDM Integration + VAE + Diffusion V2
# Run inside Docker container
# ============================================================

set -e

echo "============================================"
echo "STEP 1: Download & Preprocess WISDM"
echo "============================================"
python -m src.data.download_wisdm

echo ""
echo "============================================"
echo "STEP 2: Train VAE on CogAge + WISDM"
echo "============================================"
python -m src.train.train_sensor_vae_combined

echo ""
echo "============================================"
echo "STEP 3: Extract CogAge Latents (new VAE)"
echo "============================================"
# Override checkpoint path to use combined VAE
python -c "
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer

DEVICE = 'cuda'
normalizer = SensorNormalizer.load('data/sensor_normalizer_combined.npz')

# Try combined VAE, fallback to original
ckpt_path = 'checkpoints/sensor_vae_combined_best.pt'
if not Path(ckpt_path).exists():
    ckpt_path = 'checkpoints/sensor_vae_best.pt'
print(f'Using VAE: {ckpt_path}')

ckpt = torch.load(ckpt_path, map_location=DEVICE)
vae = SensorMultiModalVAE().to(DEVICE)
vae.load_state_dict(ckpt['model_state'])
vae.eval()

for split, prefix, out_prefix in [('training', 'train', 'train'), ('testing', 'test', 'test')]:
    dataset = ConcatDataset([
        CogAgeSensorDataset('data/cogage/python/arrays/blho', split, normalizer),
        CogAgeSensorDataset('data/cogage/python/arrays/bbh', split, normalizer),
        CogAgeSensorDataset('data/cogage/python/arrays/state', split, normalizer),
    ])
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    latents = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Encoding CogAge {split}'):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for key in SENSOR_NAMES:
                latents[key].append(outputs[key]['mu'].cpu())
    out_dir = Path('data/sensor_latents')
    out_dir.mkdir(exist_ok=True)
    for key in SENSOR_NAMES:
        t = torch.cat(latents[key], dim=0)
        torch.save(t, out_dir / f'{out_prefix}_latents_{key}_mu.pt')
        print(f'  {key}: {t.shape}')
print('CogAge latents extracted.')
"

echo ""
echo "============================================"
echo "STEP 4: Extract WISDM Latents (new VAE)"
echo "============================================"
python -m src.data.extract_wisdm_latents

echo ""
echo "============================================"
echo "STEP 5: Train Diffusion V2 (Pre-train + Fine-tune)"
echo "============================================"
python -m src.train.train_sensor_diffusion_v2_pretrain

echo ""
echo "============================================"
echo "STEP 6: Analyze Results"
echo "============================================"
python -m src.eval.analyze_diffusion_quality_v2 --diffusion-dir checkpoints/sensor_diffusion_v2_pretrain

echo ""
echo "============================================"
echo "PIPELINE COMPLETE!"
echo "============================================"
