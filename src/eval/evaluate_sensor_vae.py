# ============================================================
# Evaluate Sensor-Level VAE Reconstruction
# Plots ground truth vs reconstruction for each sensor
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_epoch_050.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

NUM_PLOT_SAMPLES = 5
OUTPUT_DIR = Path("outputs/sensor_vae_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"\n{'='*60}")
    print("Evaluating Sensor-Level VAE Reconstruction")
    print(f"{'='*60}\n")

    # Load normalizer + dataset
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    test_dataset = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"], "testing", normalizer),
        CogAgeSensorDataset(DATA_ROOTS["bbh"], "testing", normalizer),
        CogAgeSensorDataset(DATA_ROOTS["state"], "testing", normalizer),
    ])

    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}")

    # Load VAE
    vae = SensorMultiModalVAE().to(DEVICE)
    ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()

    # Compute MSE per sensor
    sensor_mse = {k: 0.0 for k in SENSOR_NAMES}
    n_batches = 0

    # Store first batch for plotting
    first_batch_gt = None
    first_batch_recon = None

    with torch.no_grad():
        for batch in loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)

            for key in SENSOR_NAMES:
                mse = F.mse_loss(outputs[key]["recon"], sensor_data[key]).item()
                sensor_mse[key] += mse

            if first_batch_gt is None:
                first_batch_gt = {k: sensor_data[k].cpu() for k in SENSOR_NAMES}
                first_batch_recon = {k: outputs[k]["recon"].cpu() for k in SENSOR_NAMES}

            n_batches += 1

    # Print results
    print(f"\n{'='*60}")
    print("Reconstruction MSE per Sensor:")
    print(f"{'='*60}")
    for key in SENSOR_NAMES:
        avg = sensor_mse[key] / n_batches
        print(f"  {key:15s}: {avg:.6f}")
    total_avg = sum(sensor_mse.values()) / (len(SENSOR_NAMES) * n_batches)
    print(f"  {'AVERAGE':15s}: {total_avg:.6f}")
    print(f"{'='*60}\n")

    # Plot reconstructions
    print("Creating plots...")
    ch_names = ["X", "Y", "Z"]

    for idx in range(min(NUM_PLOT_SAMPLES, first_batch_gt[SENSOR_NAMES[0]].shape[0])):
        fig, axs = plt.subplots(len(SENSOR_NAMES), 3, figsize=(18, 3 * len(SENSOR_NAMES)))

        for row, key in enumerate(SENSOR_NAMES):
            gt = first_batch_gt[key][idx]      # (T, 3)
            recon = first_batch_recon[key][idx]  # (T, 3)

            for ch in range(3):
                ax = axs[row, ch]
                ax.plot(gt[:, ch].numpy(), label="GT", alpha=0.8)
                ax.plot(recon[:, ch].numpy(), label="Recon", alpha=0.8, linestyle='--')
                if row == 0:
                    ax.set_title(ch_names[ch])
                if ch == 0:
                    ax.set_ylabel(key, fontsize=9)
                ax.grid(alpha=0.3)
                if row == 0 and ch == 0:
                    ax.legend(fontsize=7)

        plt.suptitle(f"Sensor VAE Reconstruction - Sample {idx}", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"sensor_vae_recon_sample_{idx}.png", dpi=150)
        plt.close()

    print(f"Saved {min(NUM_PLOT_SAMPLES, first_batch_gt[SENSOR_NAMES[0]].shape[0])} plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
