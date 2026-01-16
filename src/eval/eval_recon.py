import torch
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader, ConcatDataset

from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer
from src.models.vae import MultiModalVAE


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1          # wichtig: einzelne Sequenzen
NUM_SAMPLES = 5         # wie viele Test-Samples plotten

CHECKPOINT_PATH = "checkpoints/vae_epoch_030.pt"
NORMALIZER_PATH = "data/combined_normalizer.npz"

OUT_DIR = Path("outputs/recon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

ds_blho_test = CogAgeVAEDataset(DATA_ROOTS["blho"], "testing", normalizer)
ds_bbh_test = CogAgeVAEDataset(DATA_ROOTS["bbh"], "testing", normalizer)
ds_state_test = CogAgeVAEDataset(DATA_ROOTS["state"], "testing", normalizer)

test_dataset = ConcatDataset([
    ds_blho_test,
    ds_bbh_test,
    ds_state_test
])

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# MODEL
# ============================================================
model = MultiModalVAE(
    z_device=32,
    z_fused=64
).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Model loaded.")


# ============================================================
# PLOTTING UTILS
# ============================================================
def plot_recon(original, recon, title, out_path, channels=(0, 1, 2)):
    """
    original, recon: (T, C)
    """
    t = range(original.shape[0])

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 3 * len(channels)))
    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ax.plot(t, original[:, ch], label="original", alpha=0.8)
        ax.plot(t, recon[:, ch], label="recon", alpha=0.8)
        ax.set_title(f"{title} | channel {ch}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# RUN
# ============================================================
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= NUM_SAMPLES:
            break

        phone = batch["phone"].to(DEVICE)
        watch = batch["watch"].to(DEVICE)
        glasses = batch["glasses"].to(DEVICE)

        out = model(phone, watch, glasses)

        # remove batch dim
        phone_np = phone[0].cpu().numpy()
        watch_np = watch[0].cpu().numpy()
        glasses_np = glasses[0].cpu().numpy()

        recon_phone = out["recon_phone"][0].cpu().numpy()
        recon_watch = out["recon_watch"][0].cpu().numpy()
        recon_glasses = out["recon_glasses"][0].cpu().numpy()

        # ---- PHONE (pick a few channels) ----
        plot_recon(
            phone_np,
            recon_phone,
            title="PHONE",
            out_path=OUT_DIR / f"sample_{i}_phone.png",
            channels=(0, 3, 6)   # accel x, gyro x, gravity x
        )

        # ---- WATCH ----
        plot_recon(
            watch_np,
            recon_watch,
            title="WATCH",
            out_path=OUT_DIR / f"sample_{i}_watch.png",
            channels=(0, 3)
        )

        # ---- GLASSES ----
        plot_recon(
            glasses_np,
            recon_glasses,
            title="GLASSES",
            out_path=OUT_DIR / f"sample_{i}_glasses.png",
            channels=(0,)        # meist reicht 1 channel
        )

        print(f"Saved recon plots for sample {i}")

print("✅ Reconstruction plots saved.")
