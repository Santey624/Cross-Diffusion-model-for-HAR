# ============================================================
# Evaluate VAE Reconstruction Quality
# Measure baseline reconstruction error without imputation
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.models.temporal_vae import TemporalMultiModalVAE
from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAE checkpoint
VAE_CHECKPOINT = "checkpoints/vae_gpu_epoch_050.pt"

# Data
DATA_ROOT = "data/cogage/python/arrays/blho"
SPLIT = "training"  # Evaluate on training data to match latents

# How many samples to evaluate
NUM_EVAL_SAMPLES = 100

# Output
OUTPUT_DIR = Path("outputs/vae_reconstruction_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN EVALUATION
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Evaluating VAE Reconstruction Quality")
    print(f"{'='*60}\n")

    # Load VAE
    print("Loading VAE...")
    vae = TemporalMultiModalVAE(z_phone=32, z_watch=32, z_glasses=16).to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load dataset
    print(f"Loading dataset from: {DATA_ROOT}")
    normalizer = MultiModalNormalizer()
    normalizer.fit_from_dataset(DATA_ROOT, split=SPLIT)

    dataset = CogAgeVAEDataset(
        root_dir=DATA_ROOT,
        split=SPLIT,
        normalizer=normalizer,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Evaluating on {NUM_EVAL_SAMPLES} samples\n")

    # Collect reconstruction errors
    phone_mse_list = []
    watch_mse_list = []
    glasses_mse_list = []

    # For visualization
    sample_indices = np.random.choice(len(dataset), min(5, NUM_EVAL_SAMPLES), replace=False)

    with torch.no_grad():
        for i in range(NUM_EVAL_SAMPLES):
            sample = dataset[i]

            phone_real = sample["phone_sensor"].unsqueeze(0).to(DEVICE)    # (1, 600, 12)
            watch_real = sample["watch_sensor"].unsqueeze(0).to(DEVICE)    # (1, 204, 12)
            glasses_real = sample["glasses_sensor"].unsqueeze(0).to(DEVICE)  # (1, 100, 1)

            # Encode and decode
            phone_mu, phone_logvar = vae.phone.encoder(phone_real)
            watch_mu, watch_logvar = vae.watch.encoder(watch_real)
            glasses_mu, glasses_logvar = vae.glasses.encoder(glasses_real)

            phone_recon = vae.phone.decoder(phone_mu)
            watch_recon = vae.watch.decoder(watch_mu)
            glasses_recon = vae.glasses.decoder(glasses_mu)

            # Compute MSE
            phone_mse = F.mse_loss(phone_recon, phone_real).item()
            watch_mse = F.mse_loss(watch_recon, watch_real).item()
            glasses_mse = F.mse_loss(glasses_recon, glasses_real).item()

            phone_mse_list.append(phone_mse)
            watch_mse_list.append(watch_mse)
            glasses_mse_list.append(glasses_mse)

            # Visualize some samples
            if i in sample_indices:
                fig, axs = plt.subplots(3, 1, figsize=(14, 10))

                # Phone (show first 3 channels)
                for ch in range(3):
                    axs[0].plot(phone_real[0, :, ch].cpu().numpy(),
                               label=f"Real Ch{ch}", alpha=0.7, linewidth=1.5)
                    axs[0].plot(phone_recon[0, :, ch].cpu().numpy(),
                               label=f"Recon Ch{ch}", alpha=0.7, linewidth=1.5, linestyle='--')
                axs[0].set_title(f"Phone Reconstruction (MSE: {phone_mse:.4f})")
                axs[0].legend(ncol=6)
                axs[0].grid(alpha=0.3)

                # Watch (show first 3 channels)
                for ch in range(3):
                    axs[1].plot(watch_real[0, :, ch].cpu().numpy(),
                               label=f"Real Ch{ch}", alpha=0.7, linewidth=1.5)
                    axs[1].plot(watch_recon[0, :, ch].cpu().numpy(),
                               label=f"Recon Ch{ch}", alpha=0.7, linewidth=1.5, linestyle='--')
                axs[1].set_title(f"Watch Reconstruction (MSE: {watch_mse:.4f})")
                axs[1].legend(ncol=6)
                axs[1].grid(alpha=0.3)

                # Glasses (1 channel)
                axs[2].plot(glasses_real[0, :, 0].cpu().numpy(),
                           label="Real", alpha=0.7, linewidth=1.5)
                axs[2].plot(glasses_recon[0, :, 0].cpu().numpy(),
                           label="Recon", alpha=0.7, linewidth=1.5, linestyle='--')
                axs[2].set_title(f"Glasses Reconstruction (MSE: {glasses_mse:.4f})")
                axs[2].legend()
                axs[2].grid(alpha=0.3)

                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"vae_reconstruction_sample_{i}.png", dpi=150)
                plt.close()

            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{NUM_EVAL_SAMPLES} samples...")

    # Compute statistics
    phone_mse_mean = np.mean(phone_mse_list)
    phone_mse_std = np.std(phone_mse_list)
    watch_mse_mean = np.mean(watch_mse_list)
    watch_mse_std = np.std(watch_mse_list)
    glasses_mse_mean = np.mean(glasses_mse_list)
    glasses_mse_std = np.std(glasses_mse_list)

    print(f"\n{'='*60}")
    print("VAE RECONSTRUCTION RESULTS:")
    print(f"{'='*60}")
    print(f"\nPhone Reconstruction:")
    print(f"  MSE:  {phone_mse_mean:.6f} ± {phone_mse_std:.6f}")
    print(f"\nWatch Reconstruction:")
    print(f"  MSE:  {watch_mse_mean:.6f} ± {watch_mse_std:.6f}")
    print(f"\nGlasses Reconstruction:")
    print(f"  MSE:  {glasses_mse_mean:.6f} ± {glasses_mse_std:.6f}")
    print(f"\n{'='*60}\n")

    # Create box plots
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.boxplot([phone_mse_list, watch_mse_list, glasses_mse_list],
               labels=['Phone', 'Watch', 'Glasses'])
    ax.set_ylabel('MSE')
    ax.set_title('VAE Reconstruction Error Distribution')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vae_reconstruction_boxplot.png", dpi=150)
    plt.close()

    # Save metrics
    metrics_path = OUTPUT_DIR / "vae_reconstruction_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"VAE Reconstruction Evaluation\n")
        f.write(f"Checkpoint: {VAE_CHECKPOINT}\n")
        f.write(f"Num Samples: {NUM_EVAL_SAMPLES}\n\n")
        f.write(f"Phone MSE: {phone_mse_mean:.6f} ± {phone_mse_std:.6f}\n")
        f.write(f"Watch MSE: {watch_mse_mean:.6f} ± {watch_mse_std:.6f}\n")
        f.write(f"Glasses MSE: {glasses_mse_mean:.6f} ± {glasses_mse_std:.6f}\n")

    print(f"✅ Metrics saved to: {metrics_path}")
    print(f"✅ Visualizations saved to: {OUTPUT_DIR}")

    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print("\nCompare these numbers with your imputation results:")
    print("  - If imputation MSE ≈ VAE reconstruction MSE:")
    print("    → Diffusion is doing a good job, VAE is the bottleneck")
    print("  - If imputation MSE >> VAE reconstruction MSE:")
    print("    → Diffusion needs improvement")
    print("\nCurrent imputation results (joint model):")
    print("  Phone imputation Signal MSE: 1.644")
    print("  Glasses imputation Signal MSE: 1.942")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
