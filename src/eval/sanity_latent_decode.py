import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.temporal_vae import TemporalMultiModalVAE
from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer


# ---------------------------
# CONFIG
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/vae_B_epoch_040.pt"
NORMALIZER_PATH = "data/combined_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUT_DIR = Path("eval_outputs/sanity_latent_decode")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1          # wichtig: einzelnes Sample für saubere Plots
NUM_SAMPLES = 5         # wie viele Samples testen
MAX_LEN = 300           # Plot-Länge
NOISE_STD = 0.05        # Stärke der Latent-Störung (0.02–0.1 ist sinnvoll)

# welche Kanäle als "overview" (wie vorher: 0,3,6 etc.)
CHANNELS_TO_PLOT = {
    "phone":   (0, 3, 6),
    "watch":   (0, 3),
    "glasses": (0, 1, 2),
}


# ---------------------------
# PLOTTING
# ---------------------------
def plot_triplet(original, recon, recon_pert, title, out_path, channels, max_len=300):
    """
    original, recon, recon_pert: (T, C) torch tensors on CPU or GPU
    """
    original = original[:max_len].detach().cpu()
    recon = recon[:max_len].detach().cpu()
    recon_pert = recon_pert[:max_len].detach().cpu()

    t = range(original.shape[0])
    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 3 * len(channels)))
    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ax.plot(t, original[:, ch], label="original", alpha=0.9)
        ax.plot(t, recon[:, ch], label="recon(z)", alpha=0.9)
        ax.plot(t, recon_pert[:, ch], label=f"recon(z+N(0,{NOISE_STD}))", alpha=0.9, linestyle="--")
        ax.set_title(f"{title} | channel {ch}")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# MAIN
# ---------------------------
def main():
    print("Loading normalizer...")
    normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)

    test_dataset = ConcatDataset([
        CogAgeVAEDataset(DATA_ROOTS["blho"], "testing", normalizer),
        CogAgeVAEDataset(DATA_ROOTS["bbh"], "testing", normalizer),
        CogAgeVAEDataset(DATA_ROOTS["state"], "testing", normalizer),
    ])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = TemporalMultiModalVAE().to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # je nachdem wie du gespeichert hast:
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    # Wir decoden z_perturbed über die jeweiligen Decoder.
    # Dafür brauchen wir Zugriff auf model.phone.decoder / model.watch.decoder / model.glasses.decoder.
    # Wenn deine TemporalMultiModalVAE intern anders heißt, passe die Attribute an.
    decoders = {
        "phone": model.phone.decoder,
        "watch": model.watch.decoder,
        "glasses": model.glasses.decoder,
    }

    # Loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Sanity check")):
            if i >= NUM_SAMPLES:
                break

            phone = batch["phone"].to(DEVICE)      # (1, T, C)
            watch = batch["watch"].to(DEVICE)
            glasses = batch["glasses"].to(DEVICE)

            outputs = model(phone, watch, glasses)

            for mod in ["phone", "watch", "glasses"]:
                # original und recon
                x = batch[mod][0]  # (T, C) auf CPU
                recon = outputs[mod]["recon"][0]  # (T, C) auf DEVICE

                # latents: (B, D, T')
                z = outputs[mod]["z"]  # (1, D, T')
                z_pert = z + NOISE_STD * torch.randn_like(z)

                # decode perturbed latent
                recon_pert = decoders[mod](z_pert)[0]  # (T, C)

                out_path = OUT_DIR / f"sample_{i}_{mod}.png"
                plot_triplet(
                    original=x,
                    recon=recon,
                    recon_pert=recon_pert,
                    title=f"{mod.upper()} sanity | sample {i}",
                    out_path=out_path,
                    channels=CHANNELS_TO_PLOT[mod],
                    max_len=MAX_LEN,
                )

            print(f"Saved: sample {i} plots")

    print(f"✅ Done. Plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
