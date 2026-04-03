# ============================================================
# Check cross-sensor latent correlations
# If sensors are uncorrelated in latent space, mean-fill is
# theoretically optimal and diffusion cannot beat it.
# ============================================================

import argparse
import torch
import numpy as np
from pathlib import Path
from src.models.sensor_vae import SENSOR_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("--v3", action="store_true", help="Use V3 latents")
parser.add_argument("--v2", action="store_true", help="Use V2 latents")
args = parser.parse_args()

V3_DIR = Path("data/sensor_latents_v3")
V2_DIR = Path("data/sensor_latents_v2")
V1_DIR = Path("data/sensor_latents")

if args.v3 and V3_DIR.exists() and (V3_DIR / f"training_latents_{SENSOR_NAMES[0]}_mu.pt").exists():
    LAT_DIR = V3_DIR
    TAG = "V3 (D=16, T=64, strong align)"
elif (args.v2 or not args.v3) and V2_DIR.exists() and (V2_DIR / f"training_latents_{SENSOR_NAMES[0]}_mu.pt").exists():
    LAT_DIR = V2_DIR
    TAG = "V2 (D=16, T=64)"
else:
    LAT_DIR = V1_DIR
    TAG = "V1 (D=8, T=32)"

print(f"\nLatent correlation analysis — {TAG}")
print(f"{'='*70}")

latents = {}
for name in SENSOR_NAMES:
    p = LAT_DIR / f"training_latents_{name}_mu.pt"
    if p.exists():
        z = torch.load(p, map_location="cpu")  # (N, D, T)
        # Flatten D and T → (N, D*T), then take mean per sample as summary
        latents[name] = z.flatten(1).numpy()   # (N, D*T)

print(f"Loaded {len(latents)} sensors, N={next(iter(latents.values())).shape[0]} samples\n")

# Per-sensor mean vector (D*T,) → used for mean-fill
# Correlation: use first principal component (mean across D*T) per sample
# Simpler: take global mean across D and T → scalar per sample
sensor_means = {k: v.mean(axis=1) for k, v in latents.items()}  # (N,) per sensor

print("Cross-sensor Pearson correlation of mean-latent (scalar per sample):")
print(f"  {'':14}", end="")
for n in SENSOR_NAMES:
    print(f"  {n[:10]:>10}", end="")
print()
print("  " + "-" * (14 + 12 * len(SENSOR_NAMES)))

for n1 in SENSOR_NAMES:
    print(f"  {n1:<14}", end="")
    for n2 in SENSOR_NAMES:
        a = sensor_means[n1]
        b = sensor_means[n2]
        corr = np.corrcoef(a, b)[0, 1]
        print(f"  {corr:>10.3f}", end="")
    print()

print(f"\n{'='*70}")
print("R² between sensors (how much variance of one can be predicted from another):")
print(f"  {'':14}", end="")
for n in SENSOR_NAMES:
    print(f"  {n[:10]:>10}", end="")
print()
print("  " + "-" * (14 + 12 * len(SENSOR_NAMES)))

for n1 in SENSOR_NAMES:
    print(f"  {n1:<14}", end="")
    for n2 in SENSOR_NAMES:
        a = sensor_means[n1]
        b = sensor_means[n2]
        corr = np.corrcoef(a, b)[0, 1]
        print(f"  {corr**2:>10.3f}", end="")
    print()

print(f"\n{'='*70}")
print("Interpretation:")
print("  R²=0.0 → sensors uncorrelated → mean-fill is optimal, diffusion can't help")
print("  R²>0.3 → moderate correlation → diffusion can potentially beat mean-fill")
print("  R²>0.6 → strong correlation → diffusion should clearly beat mean-fill")
