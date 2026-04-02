# ============================================================
# VAE Reconstruction Evaluation — V1 vs V2
#
# Metrics (time domain):
#   MSE, MAE, R² (Pearson correlation²)
#
# Metrics (frequency domain):
#   FFT-MAG-MSE  : MSE on FFT magnitude spectrum
#   FFT-PHASE-MAE: MAE on FFT phase spectrum
#   Spectral-Corr: Pearson correlation of power spectra
#
# Flags:
#   --v1   evaluate only V1
#   --v2   evaluate only V2 (default: both)
#   --plot save example reconstruction + spectrum plots
# ============================================================

import sys
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ONLY_V1 = "--v1" in sys.argv
ONLY_V2 = "--v2" in sys.argv
DO_PLOT  = "--plot" in sys.argv

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
DATA_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}
OUTPUT_DIR = Path("eval_outputs/vae_reconstruction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE     = 64
NUM_PLOT_SAMPLES = 3


# ============================================================
# METRICS
# ============================================================
def fft_metrics(gt, recon):
    """
    gt, recon: (N, T, 3) — unnormalized sensor signals (on CPU as numpy)
    Returns dict with per-channel averaged frequency-domain metrics.
    """
    # FFT along time axis
    gt_fft    = np.fft.rfft(gt,    axis=1)   # (N, T//2+1, 3)
    recon_fft = np.fft.rfft(recon, axis=1)

    gt_mag    = np.abs(gt_fft)
    recon_mag = np.abs(recon_fft)
    gt_phase  = np.angle(gt_fft)
    recon_phase = np.angle(recon_fft)

    mag_mse  = float(np.mean((gt_mag - recon_mag) ** 2))
    phase_mae = float(np.mean(np.abs(gt_phase - recon_phase)))

    # Spectral correlation: average over channels, then samples
    corrs = []
    for ch in range(gt_mag.shape[-1]):
        for i in range(gt_mag.shape[0]):
            g = gt_mag[i, :, ch]
            r = recon_mag[i, :, ch]
            if g.std() > 1e-8 and r.std() > 1e-8:
                corrs.append(np.corrcoef(g, r)[0, 1])
    spec_corr = float(np.mean(corrs)) if corrs else float("nan")

    return {"fft_mag_mse": mag_mse, "fft_phase_mae": phase_mae, "spec_corr": spec_corr}


def time_metrics(gt, recon):
    """gt, recon: (N, T, 3) numpy arrays."""
    mse = float(np.mean((gt - recon) ** 2))
    mae = float(np.mean(np.abs(gt - recon)))
    # R²
    ss_res = np.sum((gt - recon) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {"mse": mse, "mae": mae, "r2": float(r2)}


# ============================================================
# EVALUATE ONE VAE
# ============================================================
def evaluate_vae(vae, loader, label):
    vae.eval()

    # Accumulate per-sensor arrays
    gt_arrays    = {k: [] for k in SENSOR_NAMES}
    recon_arrays = {k: [] for k in SENSOR_NAMES}

    with torch.no_grad():
        for batch in loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs     = vae(sensor_data)
            for k in SENSOR_NAMES:
                gt_arrays[k].append(sensor_data[k].cpu().numpy())
                recon_arrays[k].append(outputs[k]["recon"].cpu().numpy())

    # Concatenate
    gt_arrays    = {k: np.concatenate(v, axis=0) for k, v in gt_arrays.items()}
    recon_arrays = {k: np.concatenate(v, axis=0) for k, v in recon_arrays.items()}

    # Compute metrics
    results = {}
    for k in SENSOR_NAMES:
        gt    = gt_arrays[k]     # (N, T, 3)
        recon = recon_arrays[k]
        tm = time_metrics(gt, recon)
        fm = fft_metrics(gt, recon)
        results[k] = {**tm, **fm}

    # Print table
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    print(f"  {'Sensor':<16} {'MSE':>9} {'MAE':>9} {'R²':>7} | "
          f"{'FFT-MAG-MSE':>12} {'FFT-PH-MAE':>11} {'Spec-Corr':>10}")
    print("  " + "-" * 88)
    for k in SENSOR_NAMES:
        m = results[k]
        print(f"  {k:<16} {m['mse']:>9.5f} {m['mae']:>9.5f} {m['r2']:>7.4f} | "
              f"{m['fft_mag_mse']:>12.5f} {m['fft_phase_mae']:>11.5f} {m['spec_corr']:>10.4f}")

    # Averages
    avg = {}
    for metric in ["mse", "mae", "r2", "fft_mag_mse", "fft_phase_mae", "spec_corr"]:
        avg[metric] = np.mean([results[k][metric] for k in SENSOR_NAMES])
    print("  " + "-" * 88)
    print(f"  {'AVERAGE':<16} {avg['mse']:>9.5f} {avg['mae']:>9.5f} {avg['r2']:>7.4f} | "
          f"{avg['fft_mag_mse']:>12.5f} {avg['fft_phase_mae']:>11.5f} {avg['spec_corr']:>10.4f}")
    print(f"{'='*100}\n")

    return results, gt_arrays, recon_arrays


# ============================================================
# PLOTS
# ============================================================
def save_plots(gt_arrays, recon_arrays, label, tag, n_samples=NUM_PLOT_SAMPLES):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    ch_names = ["X", "Y", "Z"]

    for idx in range(n_samples):
        fig, axs = plt.subplots(len(SENSOR_NAMES), 4,
                                figsize=(20, 3 * len(SENSOR_NAMES)))

        for row, k in enumerate(SENSOR_NAMES):
            gt    = gt_arrays[k][idx]     # (T, 3)
            recon = recon_arrays[k][idx]

            # Time domain — first 3 cols (one per channel)
            for ch in range(3):
                ax = axs[row, ch]
                ax.plot(gt[:, ch], label="GT",    alpha=0.8, linewidth=0.8)
                ax.plot(recon[:, ch], label="Rec", alpha=0.8, linewidth=0.8, linestyle="--")
                if row == 0:
                    ax.set_title(ch_names[ch])
                if ch == 0:
                    ax.set_ylabel(k, fontsize=8)
                ax.grid(alpha=0.3)
                if row == 0 and ch == 0:
                    ax.legend(fontsize=7)

            # Frequency domain — 4th col (mean over channels)
            ax = axs[row, 3]
            T = gt.shape[0]
            freqs = np.fft.rfftfreq(T)
            gt_mag    = np.abs(np.fft.rfft(gt,    axis=0)).mean(axis=1)
            recon_mag = np.abs(np.fft.rfft(recon, axis=0)).mean(axis=1)
            ax.semilogy(freqs, gt_mag + 1e-8,    label="GT",  alpha=0.8, linewidth=0.8)
            ax.semilogy(freqs, recon_mag + 1e-8, label="Rec", alpha=0.8, linewidth=0.8,
                        linestyle="--")
            if row == 0:
                ax.set_title("Spectrum (avg ch)")
            ax.set_xlabel("Freq", fontsize=7)
            ax.grid(alpha=0.3, which="both")

        plt.suptitle(f"{label} — Sample {idx}", fontsize=12)
        plt.tight_layout()
        path = OUTPUT_DIR / f"{tag}_sample_{idx}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    test_ds = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"],  "testing", normalizer),
        CogAgeSensorDataset(DATA_ROOTS["bbh"],   "testing", normalizer),
        CogAgeSensorDataset(DATA_ROOTS["state"], "testing", normalizer),
    ])
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"\nTest samples: {len(test_ds)}")

    vaes_to_eval = []

    if not ONLY_V2:
        v1_path = Path("checkpoints/sensor_vae_combined_best.pt")
        if v1_path.exists():
            ckpt = torch.load(v1_path, map_location=DEVICE)
            vae1 = SensorMultiModalVAE().to(DEVICE)
            vae1.load_state_dict(ckpt["model_state"])
            vaes_to_eval.append((vae1, "VAE V1 (D=8, T=32)", "v1"))
        else:
            print(f"V1 checkpoint not found: {v1_path}")

    if not ONLY_V1:
        v2_path = Path("checkpoints/sensor_vae_v2/best_model.pt")
        if v2_path.exists():
            ckpt = torch.load(v2_path, map_location=DEVICE)
            cfg  = ckpt["config"]
            vae2 = SensorMultiModalVAE(latent_dim=cfg["latent_dim"],
                                       t_shared=cfg["t_shared"]).to(DEVICE)
            vae2.load_state_dict(ckpt["model_state"])
            vaes_to_eval.append((vae2, f"VAE V2 (D={cfg['latent_dim']}, T={cfg['t_shared']})", "v2"))
        else:
            print(f"V2 checkpoint not found: {v2_path}")

    if not vaes_to_eval:
        print("No checkpoints found.")
        return

    all_results = {}
    for vae, label, tag in vaes_to_eval:
        results, gt_arrays, recon_arrays = evaluate_vae(vae, loader, label)
        all_results[tag] = results
        if DO_PLOT:
            print(f"Saving plots for {label}...")
            save_plots(gt_arrays, recon_arrays, label, tag)

    # ---- Side-by-side comparison (if both evaluated) ----
    if "v1" in all_results and "v2" in all_results:
        r1 = all_results["v1"]
        r2 = all_results["v2"]
        metrics = ["mse", "mae", "r2", "fft_mag_mse", "fft_phase_mae", "spec_corr"]
        higher_better = {"r2", "spec_corr"}

        print(f"\n{'='*100}")
        print("  V1 vs V2 — COMPARISON (↓ better for MSE/MAE/FFT, ↑ better for R²/Corr)")
        print(f"{'='*100}")
        print(f"  {'Sensor':<16}", end="")
        for m in metrics:
            print(f"  {m:>13}", end="")
        print()
        print("  " + "-" * 96)

        for k in SENSOR_NAMES:
            print(f"  {k:<16}", end="")
            for m in metrics:
                v1_val = r1[k][m]
                v2_val = r2[k][m]
                if m in higher_better:
                    better = "↑V2" if v2_val > v1_val else "↑V1"
                    delta  = v2_val - v1_val
                else:
                    better = "↓V2" if v2_val < v1_val else "↓V1"
                    delta  = v1_val - v2_val   # positive = V2 better
                sign = "+" if delta >= 0 else ""
                print(f"  {sign}{delta:>+9.4f}{better:>4}", end="")
            print()

        # Average row
        print("  " + "-" * 96)
        print(f"  {'AVERAGE':<16}", end="")
        for m in metrics:
            v1_avg = np.mean([r1[k][m] for k in SENSOR_NAMES])
            v2_avg = np.mean([r2[k][m] for k in SENSOR_NAMES])
            if m in higher_better:
                better = "↑V2" if v2_avg > v1_avg else "↑V1"
                delta  = v2_avg - v1_avg
            else:
                better = "↓V2" if v2_avg < v1_avg else "↓V1"
                delta  = v1_avg - v2_avg
            print(f"  {delta:>+9.4f}{better:>4}", end="")
        print()
        print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
