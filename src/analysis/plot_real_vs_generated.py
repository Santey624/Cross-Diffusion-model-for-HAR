import torch
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
REAL_BATCH_PATH = "data/cogage/python/arrays/blho/testing/batch_000.pt"
GEN_PATH = "outputs/generated_timeseries.pt"

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IDX = 0       # welches Sample
CHANNEL_PHONE = 0    # welchen Kanal plotten
CHANNEL_WATCH = 0
CHANNEL_GLASSES = 0


# =========================
# LOAD DATA
# =========================
real = torch.load(REAL_BATCH_PATH)
gen = torch.load(GEN_PATH)

real_phone = real["phone"][SAMPLE_IDX, :, CHANNEL_PHONE]
gen_phone = gen["phone"][SAMPLE_IDX, :, CHANNEL_PHONE]

real_watch = real["watch"][SAMPLE_IDX, :, CHANNEL_WATCH]
gen_watch = gen["watch"][SAMPLE_IDX, :, CHANNEL_WATCH]

real_glasses = real["glasses"][SAMPLE_IDX, :, CHANNEL_GLASSES]
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
