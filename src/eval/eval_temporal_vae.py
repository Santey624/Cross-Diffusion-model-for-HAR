import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.models.temporal_vae import TemporalMultiModalVAE
from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer
from src.losses.temporal_vae_loss import temporal_vae_loss


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
BETA = 1e-4
NUM_SAMPLES = 5        # wie viele Test-Samples visualisieren
MAX_LEN = 300          # Plot-Länge

CHECKPOINT_PATH = "checkpoints/vae_B_epoch_040.pt"
NORMALIZER_PATH = "data/combined_normalizer.npz"

OUTPUT_DIR = Path("eval_outputs")
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}


# ============================================================
# PLOTTING UTILS
# ============================================================
def plot_single_channel(x, recon, title, out_path, ch, max_len=300):
    x = x[:max_len, ch].cpu().numpy()
    recon = recon[:max_len, ch].cpu().numpy()

    plt.figure(figsize=(10, 3))
    plt.plot(x, label="original", linewidth=2)
    plt.plot(recon, label="reconstruction", linestyle="--")
    plt.title(f"{title} | channel {ch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_multi_channel(x, recon, title, out_path, channels, max_len=300):
    t = range(min(x.shape[0], max_len))
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 3 * len(channels)))

    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ax.plot(t, x[:max_len, ch].cpu(), label="original", alpha=0.8)
        ax.plot(t, recon[:max_len, ch].cpu(), label="recon", alpha=0.8)
        ax.set_title(f"{title} | channel {ch}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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
# EVAL LOOP
# ============================================================
total_loss = 0.0
total_recon = 0.0
total_kl = 0.0

all_inputs = {k: [] for k in ["phone", "watch", "glasses"]}
all_recons = {k: [] for k in ["phone", "watch", "glasses"]}

with torch.no_grad():
    for batch in tqdm(test_loader, desc="[Eval]"):
        phone = batch["phone"].to(DEVICE)
        watch = batch["watch"].to(DEVICE)
        glasses = batch["glasses"].to(DEVICE)

        outputs = model(phone, watch, glasses)

        loss, parts = temporal_vae_loss(
            outputs=outputs,
            batch={"phone": phone, "watch": watch, "glasses": glasses},
            beta=BETA
        )

        total_loss += loss.item()
        total_recon += parts["recon"].item()
        total_kl += parts["kl"].item()

        for k in ["phone", "watch", "glasses"]:
            all_inputs[k].append(batch[k].cpu())
            all_recons[k].append(outputs[k]["recon"].cpu())


# ============================================================
# RESULTS
# ============================================================
n = len(test_loader)
print("\n================= EVAL RESULTS =================")
print(f"Loss : {total_loss / n:.4f}")
print(f"Recon: {total_recon / n:.4f}")
print(f"KL   : {total_kl / n:.4f}")
print("================================================\n")


# ============================================================
# FLATTEN & PLOT
# ============================================================
flat_inputs = {k: torch.cat(all_inputs[k], dim=0) for k in all_inputs}
flat_recons = {k: torch.cat(all_recons[k], dim=0) for k in all_recons}

channel_counts = {
    "phone": 12,
    "watch": 6,
    "glasses": 3,
}

overview_channels = {
    "phone": (0, 3, 6),
    "watch": (0, 3),
    "glasses": (0,),
}

print("Saving reconstruction plots...")

for k in ["phone", "watch", "glasses"]:
    (PLOT_DIR / k).mkdir(parents=True, exist_ok=True)

    for i in range(NUM_SAMPLES):
        x = flat_inputs[k][i]
        recon = flat_recons[k][i]

        # --- single channel plots ---
        for ch in range(channel_counts[k]):
            plot_single_channel(
                x, recon,
                title=f"{k.upper()} sample {i}",
                out_path=PLOT_DIR / k / f"{k}_sample{i}_ch{ch}.png",
                ch=ch,
                max_len=MAX_LEN
            )

        # --- overview plot ---
        plot_multi_channel(
            x, recon,
            title=f"{k.upper()} sample {i} (overview)",
            out_path=PLOT_DIR / k / f"{k}_sample{i}_overview.png",
            channels=overview_channels[k],
            max_len=MAX_LEN
        )

print(f"✅ All plots saved to: {PLOT_DIR.resolve()}")
