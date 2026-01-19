import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.models.temporal_vae import TemporalMultiModalVAE
from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
NUM_TRAJ_SAMPLES = 3          # wie viele Samples für Zeitplots
LATENT_DIMS_TO_PLOT = [0, 1, 2, 3]

CHECKPOINT_PATH = "checkpoints/vae_B_epoch_040.pt"
NORMALIZER_PATH = "data/combined_normalizer.npz"

OUT_DIR = Path("latent_analysis")
TRAJ_DIR = OUT_DIR / "trajectories"
PCA_DIR = OUT_DIR / "pca"

TRAJ_DIR.mkdir(parents=True, exist_ok=True)
PCA_DIR.mkdir(parents=True, exist_ok=True)

# modality-specific folders
for k in ["phone", "watch", "glasses"]:
    (TRAJ_DIR / k).mkdir(parents=True, exist_ok=True)

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

# ============================================================
# DATA
# ============================================================
print("Loading normalizer...")
normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)

test_dataset = ConcatDataset([
    CogAgeVAEDataset(DATA_ROOTS["blho"], "testing", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["bbh"], "testing", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["state"], "testing", normalizer),
])

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test samples: {len(test_dataset)}")

# ============================================================
# MODEL
# ============================================================
model = TemporalMultiModalVAE().to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

# ============================================================
# HELPERS
# ============================================================
def plot_latent_trajectory(z, title, out_path, dims):
    """
    z: (D, T)
    """
    plt.figure(figsize=(10, 4))
    for d in dims:
        plt.plot(z[d], label=f"dim {d}")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("latent value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_pca(z_2d, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ============================================================
# MAIN ANALYSIS
# ============================================================
latent_store = {
    "phone": [],
    "watch": [],
    "glasses": [],
}

print("Collecting latents...")
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        phone = batch["phone"].to(DEVICE)
        watch = batch["watch"].to(DEVICE)
        glasses = batch["glasses"].to(DEVICE)

        outputs = model(phone, watch, glasses)

        for k in ["phone", "watch", "glasses"]:
            z = outputs[k]["z"]          # (B, D, T')
            z_mean = z.mean(dim=-1)      # (B, D)
            latent_store[k].append(z_mean.cpu())

        # --------- Trajectory plots (nur erste Samples) ---------
        if i == 0:
            for k in ["phone", "watch", "glasses"]:
                z_full = outputs[k]["z"].cpu()   # (B, D, T')
                for s in range(min(NUM_TRAJ_SAMPLES, z_full.shape[0])):
                    z_sample = z_full[s]
                    plot_latent_trajectory(
                        z_sample,
                        title=f"{k.upper()} latent trajectory | sample {s}",
                        out_path=TRAJ_DIR / k / f"{k}_traj_sample{s}.png",
                        dims=LATENT_DIMS_TO_PLOT
                    )

# ============================================================
# PCA
# ============================================================
print("Running PCA...")
for k in ["phone", "watch", "glasses"]:
    Z = torch.cat(latent_store[k], dim=0).numpy()   # (N, D)
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)

    plot_pca(
        Z_2d,
        title=f"{k.upper()} latent PCA",
        out_path=PCA_DIR / f"{k}_pca.png"
    )

print("\n✅ Latent analysis finished.")
print(f"Results saved to: {OUT_DIR}")
