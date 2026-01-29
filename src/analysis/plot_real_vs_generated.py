import torch
import matplotlib.pyplot as plt
from pathlib import Path


from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer

# =========================
# CONFIG
# =========================
DATA_ROOT = "data/cogage/python/arrays/blho"
SPLIT = "testing"        # or "training"
GEN_PATH = "outputs/generated_timeseries/generated_timeseries.pt"

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IDX = 0
CHANNEL_PHONE = 0
CHANNEL_WATCH = 0
CHANNEL_GLASSES = 0

# =========================
# LOAD REAL DATA (Dataset)
# =========================
normalizer = MultiModalNormalizer.load("data/combined_normalizer.npz")
dataset = CogAgeVAEDataset(DATA_ROOT, SPLIT, normalizer)

real = dataset[SAMPLE_IDX]

# =========================
# LOAD GENERATED
# =========================
gen = torch.load(GEN_PATH)

real_phone = real["phone"][:, CHANNEL_PHONE]
gen_phone = gen["phone"][SAMPLE_IDX, :, CHANNEL_PHONE]

real_watch = real["watch"][:, CHANNEL_WATCH]
gen_watch = gen["watch"][SAMPLE_IDX, :, CHANNEL_WATCH]

real_glasses = real["glasses"][:, CHANNEL_GLASSES]
gen_glasses = gen["glasses"][SAMPLE_IDX, :, CHANNEL_GLASSES]

# =========================
# PLOT
# =========================
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

axs[0].plot(real_phone, label="real", alpha=0.8)
axs[0].plot(gen_phone, label="generated", alpha=0.8)
axs[0].set_title("PHONE")
axs[0].legend()

axs[1].plot(real_watch, label="real", alpha=0.8)
axs[1].plot(gen_watch, label="generated", alpha=0.8)
axs[1].set_title("WATCH")
axs[1].legend()

axs[2].plot(real_glasses, label="real", alpha=0.8)
axs[2].plot(gen_glasses, label="generated", alpha=0.8)
axs[2].set_title("GLASSES")
axs[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "real_vs_generated.png", dpi=150)
plt.show()
