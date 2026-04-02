# ============================================================
# Evaluate Imputation Quality — Diffusion vs Mean vs Zero
#
# Compares how well each method reconstructs missing latents.
# Metrics: MSE (Euclidean), FFT-MAG-MSE (Fourier), Cosine Sim
#
# Flags:
#   --v3        V3 diffusion on V1 latents (trained WITH FFT loss)
#   --v3v2      V3 diffusion on V2 latents (default)
#   --v2        V2 diffusion on V1 latents (no FFT loss, older)
# ============================================================

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion_v2 import create_sensor_diffusion_v2
from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
DATA_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}
DDIM_STEPS = 50
BATCH_SIZE = 32

# Select mode from flags
if "--v3" in sys.argv:
    MODE = "v3"
elif "--v2" in sys.argv:
    MODE = "v2"
else:
    MODE = "v3v2"

CONFIG = {
    "v3": {
        "label":       "Diffusion V3 on V1-Latents (FFT-Loss ✓)",
        "vae_ckpt":    "checkpoints/sensor_vae_combined_best.pt",
        "vae_v":       1,
        "diff_dir":    Path("checkpoints/sensor_diffusion_v3"),
        "arch":        "v3",
    },
    "v3v2": {
        "label":       "Diffusion V3 on V2-Latents (FFT-Loss ✓ retrain pending)",
        "vae_ckpt":    "checkpoints/sensor_vae_v2/best_model.pt",
        "vae_v":       2,
        "diff_dir":    Path("checkpoints/sensor_diffusion_v3_v2"),
        "arch":        "v3",
    },
    "v2": {
        "label":       "Diffusion V2 on V1-Latents (no FFT-Loss)",
        "vae_ckpt":    "checkpoints/sensor_vae_combined_best.pt",
        "vae_v":       1,
        "diff_dir":    Path("checkpoints/sensor_diffusion_v2"),
        "arch":        "v2",
    },
}

MISSING_PATTERNS = {
    "phone_acc":   ["phone_acc"],
    "watch_acc":   ["watch_acc"],
    "glasses_acc": ["glasses_acc"],
    "phone_all":   ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
    "watch_all":   ["watch_acc", "watch_gyro"],
    "only_watch":  ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"],
    "only_phone":  ["watch_acc", "watch_gyro", "glasses_acc"],
}


def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    ab    = f_t / f_t[0]
    return torch.clamp(1 - ab[1:] / ab[:-1], 1e-6, 0.999).float()


@torch.no_grad()
def ddim_sample(model, stacked, observed_mask, alpha_bar, T, steps=50):
    B = stacked.shape[0]
    device  = stacked.device
    missing = 1.0 - observed_mask
    alpha_bar = alpha_bar.to(device)

    z   = observed_mask[:, :, None, None] * stacked + \
          missing[:, :, None, None] * torch.randn_like(stacked)
    tau = (torch.linspace(0, 1, steps + 1, device=device) ** 2 * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_b    = torch.full((B,), t_now, device=device, dtype=torch.long)
        noisy  = observed_mask[:, :, None, None] * stacked + missing[:, :, None, None] * z
        noise_pred = model(noisy, t_b, observed_mask)
        ab_now, ab_next = alpha_bar[t_now], alpha_bar[t_next]
        pred_x0 = ((z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)).clamp(-5, 5)
        z_new   = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
        z = observed_mask[:, :, None, None] * stacked + missing[:, :, None, None] * z_new

    return z


def fft_mag_mse(pred, target):
    """MSE on FFT magnitude spectrum along time axis."""
    return F.mse_loss(
        torch.fft.rfft(pred,   dim=-1).abs(),
        torch.fft.rfft(target, dim=-1).abs(),
    ).item()


def run_eval(cfg):
    label    = cfg["label"]
    diff_dir = cfg["diff_dir"]

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    # VAE
    vae_ckpt = torch.load(cfg["vae_ckpt"], map_location=DEVICE)
    if cfg["vae_v"] == 1:
        latent_dim, t_shared = 8, 32
        vae = SensorMultiModalVAE().to(DEVICE)
    else:
        vc = vae_ckpt["config"]
        latent_dim, t_shared = vc["latent_dim"], vc["t_shared"]
        vae = SensorMultiModalVAE(latent_dim=latent_dim, t_shared=t_shared).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    print(f"  VAE: latent_dim={latent_dim}, t_shared={t_shared}")

    # Diffusion
    if not (diff_dir / "best_model.pt").exists():
        print(f"  Checkpoint not found: {diff_dir}/best_model.pt — skipping.")
        return None

    diff_ckpt = torch.load(diff_dir / "best_model.pt", map_location=DEVICE)
    dcfg      = diff_ckpt["config"]
    diff_T    = diff_ckpt["T"]

    if cfg["arch"] == "v3":
        diff_latent_dim = diff_ckpt["model_state"]["sensor_embeddings.weight"].shape[1]
        diffusion = create_sensor_diffusion_v3(
            d_model=dcfg["d_model"], num_heads=dcfg["num_heads"],
            num_blocks=dcfg["num_blocks"], dropout=0.0,
            latent_dim=diff_latent_dim,
        ).to(DEVICE)
    else:
        diffusion = create_sensor_diffusion_v2(
            d_model=dcfg["d_model"], num_heads=dcfg["num_heads"],
            num_blocks=dcfg["num_blocks"], dropout=0.0,
        ).to(DEVICE)

    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()
    norm_stats = torch.load(diff_dir / "normalization_stats.pt", map_location=DEVICE)
    betas      = cosine_beta_schedule(diff_T) if diff_ckpt["schedule"] == "cosine" \
                 else torch.linspace(1e-4, 0.02, diff_T)
    alpha_bar  = torch.cumprod(1.0 - betas, dim=0)
    print(f"  Diffusion loss={diff_ckpt.get('loss', '?'):.4f}")

    # Data
    normalizer  = SensorNormalizer.load(NORMALIZER_PATH)
    test_ds     = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Mean latents
    sums  = {k: 0.0 for k in SENSOR_NAMES}
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            sd = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            out = vae(sd)
            for k in SENSOR_NAMES:
                sums[k] = sums[k] + out[k]["mu"].mean(0, keepdim=True)
            count += 1
    mean_latents = {k: (sums[k] / count).to(DEVICE) for k in SENSOR_NAMES}

    # Evaluate each pattern
    pattern_results = {}

    for pat_name, missing_sensors in MISSING_PATTERNS.items():
        acc = {s: {"mse_d": 0, "mse_m": 0, "mse_z": 0,
                   "fft_d": 0, "fft_m": 0, "fft_z": 0,
                   "cos_d": 0, "cos_m": 0, "n": 0}
               for s in missing_sensors}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=pat_name, leave=False):
                sd  = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                out = vae(sd)
                lat = {k: out[k]["mu"] for k in SENSOR_NAMES}
                B   = next(iter(lat.values())).shape[0]

                lat_n = {k: (lat[k] - norm_stats[k]["mean"].to(DEVICE))
                              / norm_stats[k]["std"].to(DEVICE)
                         for k in SENSOR_NAMES}
                stacked  = torch.stack([lat_n[k] for k in SENSOR_NAMES], dim=1)
                obs_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for s in missing_sensors:
                    obs_mask[:, SENSOR_NAMES.index(s)] = 0.0

                imputed = ddim_sample(diffusion, stacked, obs_mask, alpha_bar, diff_T, DDIM_STEPS)

                for s in missing_sensors:
                    idx  = SENSOR_NAMES.index(s)
                    real = lat_n[s]
                    mn   = norm_stats[s]["mean"].to(DEVICE)
                    st   = norm_stats[s]["std"].to(DEVICE)

                    imp_d = imputed[:, idx]
                    imp_m = (mean_latents[s].expand(B, -1, -1) - mn) / st
                    imp_z = (-mn / st).expand(B, -1, real.shape[-1])

                    a = acc[s]
                    a["mse_d"] += F.mse_loss(imp_d, real).item()
                    a["mse_m"] += F.mse_loss(imp_m, real).item()
                    a["mse_z"] += F.mse_loss(imp_z, real).item()
                    a["fft_d"] += fft_mag_mse(imp_d, real)
                    a["fft_m"] += fft_mag_mse(imp_m, real)
                    a["fft_z"] += fft_mag_mse(imp_z, real)
                    a["cos_d"] += F.cosine_similarity(imp_d.flatten(1), real.flatten(1)).mean().item()
                    a["cos_m"] += F.cosine_similarity(imp_m.flatten(1), real.flatten(1)).mean().item()
                    a["n"] += 1

        pattern_results[pat_name] = {
            s: {k: v / max(acc[s]["n"], 1) for k, v in acc[s].items() if k != "n"}
            for s in missing_sensors
        }

    # Print results
    print(f"\n  {'Pattern':<14} {'Sensor':<14}"
          f" {'MSE-Diff':>9} {'MSE-Mean':>9} {'MSE-Zero':>9} {'BestMSE':>8}"
          f" | {'FFT-Diff':>9} {'FFT-Mean':>9} {'FFT-Zero':>9} {'BestFFT':>8}"
          f" | {'Cos-Diff':>9}")
    print("  " + "-" * 118)

    euc_wins = {"Diff": 0, "Mean": 0, "Zero": 0}
    fft_wins = {"Diff": 0, "Mean": 0, "Zero": 0}

    for pat_name, sensors in pattern_results.items():
        for i, (s, m) in enumerate(sensors.items()):
            eb = min(m["mse_d"], m["mse_m"], m["mse_z"])
            fb = min(m["fft_d"], m["fft_m"], m["fft_z"])
            ew = "Diff" if eb == m["mse_d"] else ("Mean" if eb == m["mse_m"] else "Zero")
            fw = "Diff" if fb == m["fft_d"] else ("Mean" if fb == m["fft_m"] else "Zero")
            euc_wins[ew] += 1
            fft_wins[fw] += 1
            pat_label = pat_name if i == 0 else ""
            print(f"  {pat_label:<14} {s:<14}"
                  f" {m['mse_d']:>9.4f} {m['mse_m']:>9.4f} {m['mse_z']:>9.4f} {ew:>8}"
                  f" | {m['fft_d']:>9.4f} {m['fft_m']:>9.4f} {m['fft_z']:>9.4f} {fw:>8}"
                  f" | {m['cos_d']:>9.4f}")

    total = sum(euc_wins.values())
    print(f"\n  Wins (MSE): Diff={euc_wins['Diff']}/{total}  Mean={euc_wins['Mean']}/{total}  Zero={euc_wins['Zero']}/{total}")
    print(f"  Wins (FFT): Diff={fft_wins['Diff']}/{total}  Mean={fft_wins['Mean']}/{total}  Zero={fft_wins['Zero']}/{total}")

    return pattern_results


def main():
    print(f"\n{'='*80}")
    print("IMPUTATION QUALITY — MSE (Euclidean) vs FFT-MAG-MSE (Fourier)")
    print(f"Mode: {MODE}")
    print(f"{'='*80}")

    run_eval(CONFIG[MODE])


if __name__ == "__main__":
    main()
