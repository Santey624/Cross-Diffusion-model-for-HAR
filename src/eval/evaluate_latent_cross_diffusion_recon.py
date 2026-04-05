# ============================================================
# Evaluate Latent Cross-Sensor Diffusion + Recon Loss
#
# Pipeline:
#   available sensors → VAE encode → observed latents
#   DDIM impute missing latents (cross-sensor diffusion)
#   all latents → VAE decode → signals → C-LSTM-A → classification
#
# Compares:
#   - Baseline:   all sensors real (VAE encode → decode)
#   - LatDiffR:   missing via latent cross diffusion (recon-trained)
#   - Mean-fill:  missing = mean decoded signal from training set
#
# Usage:
#   python -m src.eval.evaluate_latent_cross_diffusion_recon
#   python -m src.eval.evaluate_latent_cross_diffusion_recon --state
# ============================================================

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.models.clstm_classifier import create_clstm_classifier
from src.models.latent_cross_diffusion_recon import create_latent_cross_diffusion_recon
from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE = "--state" in sys.argv

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"
DIFF_DIR        = Path("checkpoints/latent_cross_diffusion_recon")

tag = "state" if USE_STATE else "behavioral"
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_{tag}/best_model.pt"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

DDIM_STEPS = 25
BATCH_SIZE = 128


# ============================================================
# Noise schedule
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float()


# ============================================================
# DDIM imputation in latent space
# ============================================================
@torch.no_grad()
def ddim_impute_latents(model, latents, observed_mask, alpha_bar, T, ddim_steps):
    """
    latents:       (B, K, D, T_SHARED) — observed=real, missing=noise
    observed_mask: (B, K)
    Returns:       (B, K, D, T_SHARED) — imputed
    """
    B, K, D, T_lat = latents.shape
    device = latents.device
    missing_idx = (observed_mask[0] == 0).nonzero(as_tuple=True)[0].tolist()

    z = latents.clone()
    for i in missing_idx:
        z[:, i] = torch.randn(B, D, T_lat, device=device)

    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)

    for step in range(len(tau) - 1):
        t_now, t_next = tau[step], tau[step + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy = latents.clone()
        for i in missing_idx:
            noisy[:, i] = z[:, i]

        noise_pred = model(noisy, t_batch, observed_mask)
        ab_now  = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        for i in missing_idx:
            pred_z0 = ((z[:, i] - torch.sqrt(1 - ab_now) * noise_pred[:, i])
                       / torch.sqrt(ab_now)).clamp(-10, 10)
            z[:, i] = (torch.sqrt(ab_next) * pred_z0
                       + torch.sqrt(1 - ab_next) * noise_pred[:, i])

    result = latents.clone()
    for i in missing_idx:
        result[:, i] = z[:, i]
    return result


# ============================================================
# METRICS
# ============================================================
def compute_metrics(all_labels, all_probs, n_classes):
    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds_arr  = probs_arr.argmax(axis=1)
    acc = accuracy_score(labels_arr, preds_arr)
    af1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    labels_oh = label_binarize(labels_arr, classes=list(range(n_classes)))
    try:
        map_score = average_precision_score(labels_oh, probs_arr, average="macro")
    except Exception:
        map_score = float("nan")
    try:
        auc_score = roc_auc_score(labels_oh, probs_arr, average="macro", multi_class="ovr")
    except Exception:
        auc_score = float("nan")
    return {"acc": acc, "af1": af1, "map": map_score, "auc": auc_score}


# ============================================================
# MAIN
# ============================================================
def main():
    task = "State (6 classes)" if USE_STATE else "Behavioral (55 classes)"
    print(f"\n{'='*75}")
    print(f"Latent Cross-Diffusion + Recon Loss Eval  |  C-LSTM-A {task}")
    print(f"{'='*75}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    if USE_STATE:
        test_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing",  normalizer)
        train_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
    else:
        test_ds  = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)
        train_ds = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    n_classes = test_ds.n_classes

    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # VAE V2 — frozen
    print(f"Loading VAE V2...")
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    cfg_vae  = vae_ckpt["config"]
    vae = SensorMultiModalVAE(
        latent_dim=cfg_vae["latent_dim"],
        t_shared=cfg_vae["t_shared"],
    ).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Diffusion model
    print(f"Loading Latent Cross-Diffusion (recon-trained)...")
    diff_ckpt  = torch.load(DIFF_DIR / "best_model.pt", map_location=DEVICE)
    cfg        = diff_ckpt["config"]
    diff_model = create_latent_cross_diffusion_recon(
        n_sensors=cfg["n_sensors"],
        latent_dim=cfg["latent_dim"],
        t_shared=cfg["t_shared"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diff_model.load_state_dict(diff_ckpt["model_state"])
    diff_model.eval()
    T_diff    = diff_ckpt["T"]
    betas     = cosine_beta_schedule(T_diff)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    print(f"  noise={diff_ckpt['noise_loss']:.4f}, recon={diff_ckpt['recon_loss']:.6f}, "
          f"epoch={diff_ckpt['epoch']}")

    # C-LSTM-A Classifier
    print(f"Loading C-LSTM-A from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print("  NOT FOUND. Train first:")
        print(f"  python -m src.train.train_clstm_raw{' --state' if USE_STATE else ''}")
        return
    clf_ckpt   = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
    cfg_clf    = clf_ckpt["config"]
    classifier = create_clstm_classifier(
        n_sensors=cfg_clf["n_sensors"],
        n_classes=cfg_clf["n_classes"],
        cnn_channels=cfg_clf["cnn_channels"],
        lstm_hidden=cfg_clf["lstm_hidden"],
        d_attn=cfg_clf["d_attn"],
        n_heads=cfg_clf["n_heads"],
        n_layers=cfg_clf["n_layers"],
        dropout=0.0,
    ).to(DEVICE)
    classifier.load_state_dict(clf_ckpt["model_state"])
    classifier.eval()
    print(f"  best acc: {clf_ckpt.get('accuracy', float('nan')):.4f}")

    # Mean decoded signals from training set
    print("\nComputing mean decoded signals...")
    mean_signals = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="mean precompute", leave=False):
            sensor_data = {k: batch[k].to(DEVICE).float() for k in SENSOR_NAMES}
            for name in SENSOR_NAMES:
                mu, _ = vae.encode_sensor(name, sensor_data[name])
                recon = vae.decode_sensor(name, mu)   # (B, T, C)
                mean_signals[name].append(recon.permute(0, 2, 1).cpu())
    mean_signals = {
        k: torch.cat(v).mean(0, keepdim=True).to(DEVICE)
        for k, v in mean_signals.items()
    }

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = {
        "all_real":               ([], "real"),
        "phone_acc+latdiffr":     (["phone_acc"], "latdiffr"),
        "phone_acc+mean":         (["phone_acc"], "mean"),
        "watch_acc+latdiffr":     (["watch_acc"], "latdiffr"),
        "watch_acc+mean":         (["watch_acc"], "mean"),
        "glasses_acc+latdiffr":   (["glasses_acc"], "latdiffr"),
        "glasses_acc+mean":       (["glasses_acc"], "mean"),
        "phone_all+latdiffr":     (["phone_acc","phone_gyro","phone_grav","phone_lacc"], "latdiffr"),
        "phone_all+mean":         (["phone_acc","phone_gyro","phone_grav","phone_lacc"], "mean"),
        "watch_all+latdiffr":     (["watch_acc","watch_gyro"], "latdiffr"),
        "watch_all+mean":         (["watch_acc","watch_gyro"], "mean"),
        "only_watch+latdiffr":    (["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"], "latdiffr"),
        "only_watch+mean":        (["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"], "mean"),
        "only_phone+latdiffr":    (["watch_acc","watch_gyro","glasses_acc"], "latdiffr"),
        "only_phone+mean":        (["watch_acc","watch_gyro","glasses_acc"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []
        missing_idx = [SENSOR_NAMES.index(s) for s in missing_sensors]

        for batch in tqdm(test_loader, desc=f"{scenario_name:<28}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE).float() for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            with torch.no_grad():
                # Encode all sensors
                mus = []
                for name in SENSOR_NAMES:
                    mu, _ = vae.encode_sensor(name, sensor_data[name])
                    mus.append(mu)
                latents = torch.stack(mus, dim=1)   # (B, K, D, T_SHARED)

                if mode == "real":
                    decoded = {
                        name: vae.decode_sensor(name, latents[:, i]).permute(0, 2, 1)
                        for i, name in enumerate(SENSOR_NAMES)
                    }

                elif mode == "latdiffr":
                    observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                    for i in missing_idx:
                        observed_mask[:, i] = 0.0

                    imputed = ddim_impute_latents(
                        diff_model, latents, observed_mask,
                        alpha_bar, T_diff, DDIM_STEPS,
                    )
                    decoded = {}
                    for i, name in enumerate(SENSOR_NAMES):
                        recon = vae.decode_sensor(name, imputed[:, i])  # (B, T, C)
                        decoded[name] = recon.permute(0, 2, 1)           # (B, C, T)

                else:  # mean
                    decoded = {
                        name: vae.decode_sensor(name, latents[:, i]).permute(0, 2, 1)
                        for i, name in enumerate(SENSOR_NAMES)
                    }
                    for name in missing_sensors:
                        decoded[name] = mean_signals[name].expand(B, -1, -1)

                logits = classifier(decoded, SENSOR_NAMES)
                probs  = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # Results
    # ============================================================
    print(f"\n{'='*100}")
    print(f"LATENT CROSS-DIFFUSION + RECON LOSS EVAL  |  C-LSTM-A {task}")
    print(f"{'='*100}")
    print(f"  {'Scenario':<30} {'Acc':>7} {'AF1':>7} {'MAP':>7} {'AUC':>7}")
    print("  " + "-" * 62)
    for name, m in results.items():
        print(f"  {name:<30} {m['acc']:>7.4f} {m['af1']:>7.4f} "
              f"{m['map']:>7.4f} {m['auc']:>7.4f}")

    groups = [
        ("phone_acc",  ["phone_acc"]),
        ("watch_acc",  ["watch_acc"]),
        ("glasses_acc",["glasses_acc"]),
        ("phone_all",  ["phone_acc","phone_gyro","phone_grav","phone_lacc"]),
        ("watch_all",  ["watch_acc","watch_gyro"]),
        ("only_watch", ["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"]),
        ("only_phone", ["watch_acc","watch_gyro","glasses_acc"]),
    ]

    real = results.get("all_real", {})
    fmt  = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}" if m else "    —    "

    print(f"\n{'='*100}")
    print("  COMPARISON: LatDiffRecon vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'LatDiffR':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        d = results.get(f"{pat}+latdiffr")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("LatDiffR", d), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = f"  Δ={d['acc']-m['acc']:+.3f}" if d and m else ""
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
