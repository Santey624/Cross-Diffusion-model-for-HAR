# ============================================================
# Evaluate Cross-Sensor Signal Diffusion with Classifier Guidance
#
# Classifier-Guided DDIM:
#   At each denoising step:
#     1. Predict eps from diffusion model
#     2. Recover x0_pred = (x_t - sqrt(1-ab)*eps) / sqrt(ab)
#     3. Run noisy classifier on x0_pred → grad log p(y|x0_pred)
#     4. eps_guided = eps - sqrt(1-ab) * w * grad
#     5. DDIM step with eps_guided
#
# The classifier gradient steers generation toward activity-consistent
# signals without requiring cross-device correlation.
#
# Usage:
#   python -m src.eval.evaluate_signal_cross_diffusion_guided
#   python -m src.eval.evaluate_signal_cross_diffusion_guided --state
#   python -m src.eval.evaluate_signal_cross_diffusion_guided --w 2.0
# ============================================================

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.models.clstm_classifier import create_clstm_classifier
from src.models.signal_cross_diffusion import create_signal_cross_diffusion, T_COMMON
from src.models.sensor_vae import SENSOR_NAMES, SENSOR_SPECS
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE = "--state" in sys.argv

# Guidance strength — parse --w X from argv
GUIDANCE_W = 1.0
for i, arg in enumerate(sys.argv):
    if arg == "--w" and i + 1 < len(sys.argv):
        GUIDANCE_W = float(sys.argv[i + 1])

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
tag = "state" if USE_STATE else "behavioral"

DIFF_DIR        = Path("checkpoints/signal_cross_diffusion")
NOISY_CLF_CKPT  = f"checkpoints/clstm_noisy_{tag}/best_model.pt"
RAW_CLF_CKPT    = f"checkpoints/clstm_raw_{tag}/best_model.pt"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

NATIVE_LENS = {name: SENSOR_SPECS[name]["seq_len"] for name in SENSOR_NAMES}
DDIM_STEPS  = 25
BATCH_SIZE  = 64   # smaller due to gradient computation


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
# Classifier-Guided DDIM imputation
# ============================================================
def ddim_impute_guided(diff_model, noisy_clf, stacked_norm, observed_mask,
                        labels, alpha_bar, T, ddim_steps, guidance_w):
    """
    Cross-sensor DDIM with classifier guidance.

    stacked_norm:  (B, K, C, T_COMMON) — normalized, observed sensors clean
    observed_mask: (B, K)
    labels:        (B,) — GT activity labels for guidance
    guidance_w:    guidance strength (0 = no guidance)

    Returns: (B, K, C, T_COMMON) imputed
    """
    B, K, C, T_len = stacked_norm.shape
    device = stacked_norm.device
    missing_idx = (observed_mask[0] == 0).nonzero(as_tuple=True)[0].tolist()

    z = stacked_norm.clone()
    for i in missing_idx:
        z[:, i] = torch.randn(B, C, T_len, device=device)

    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)

    for step in range(len(tau) - 1):
        t_now, t_next = tau[step], tau[step + 1]
        ab_now  = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        # ---- Diffusion noise prediction (no grad needed) ----
        with torch.no_grad():
            t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)
            noisy   = stacked_norm.clone()
            for i in missing_idx:
                noisy[:, i] = z[:, i]
            noise_pred = diff_model(noisy, t_batch, observed_mask)  # (B, K, C, T)

        # ---- Classifier guidance on x0_pred ----
        if guidance_w > 0:
            # Recover x0_pred for missing sensors
            x0_preds = {}
            for i in missing_idx:
                x0_pred = ((z[:, i] - torch.sqrt(1 - ab_now) * noise_pred[:, i])
                           / torch.sqrt(ab_now)).clamp(-5, 5)
                x0_preds[i] = x0_pred

            # Build signal dict for classifier — requires grad on missing sensors
            clf_inputs = {}
            grad_targets = {}
            for i, name in enumerate(SENSOR_NAMES):
                if i in missing_idx:
                    x = x0_preds[i].detach().requires_grad_(True)
                    grad_targets[i] = x
                    clf_inputs[name] = x
                else:
                    clf_inputs[name] = stacked_norm[:, i].detach()

            # Forward through noisy classifier
            logits = noisy_clf(clf_inputs, SENSOR_NAMES)
            log_probs = F.log_softmax(logits, dim=1)
            # Sum log prob of GT label
            selected = log_probs[torch.arange(B, device=device), labels].sum()
            selected.backward()

            # Apply guidance to noise prediction
            noise_pred = noise_pred.clone()
            for i in missing_idx:
                if grad_targets[i].grad is not None:
                    grad = grad_targets[i].grad   # (B, C, T_COMMON)
                    # eps_guided = eps - sqrt(1-ab) * w * grad(log p(y|x0))
                    noise_pred[:, i] = (noise_pred[:, i]
                                        - guidance_w * torch.sqrt(1 - ab_now) * grad)

        # ---- DDIM step ----
        with torch.no_grad():
            for i in missing_idx:
                pred_x0 = ((z[:, i] - torch.sqrt(1 - ab_now) * noise_pred[:, i])
                           / torch.sqrt(ab_now)).clamp(-5, 5)
                z[:, i] = (torch.sqrt(ab_next) * pred_x0
                           + torch.sqrt(1 - ab_next) * noise_pred[:, i])

    result = stacked_norm.clone()
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
    return {"acc": acc, "af1": af1}


# ============================================================
# MAIN
# ============================================================
def main():
    task = "State (6 classes)" if USE_STATE else "Behavioral (55 classes)"
    print(f"\n{'='*75}")
    print(f"Classifier-Guided Signal Cross-Diffusion  |  C-LSTM-A {task}")
    print(f"Guidance w={GUIDANCE_W}, DDIM steps={DDIM_STEPS}")
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

    # Diffusion model
    print(f"Loading Signal Cross-Diffusion...")
    diff_ckpt  = torch.load(DIFF_DIR / "best_model.pt", map_location=DEVICE)
    cfg        = diff_ckpt["config"]
    diff_model = create_signal_cross_diffusion(
        n_sensors=cfg["n_sensors"],
        in_channels=cfg["in_channels"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diff_model.load_state_dict(diff_ckpt["model_state"])
    diff_model.eval()
    for p in diff_model.parameters():
        p.requires_grad = False
    T_diff    = diff_ckpt["T"]
    betas     = cosine_beta_schedule(T_diff)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    norm_stats = torch.load(DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
    norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
    norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
    print(f"  loss={diff_ckpt['loss']:.6f}")

    # Noisy classifier (for guidance)
    print(f"Loading Noisy Classifier from {NOISY_CLF_CKPT}...")
    if not Path(NOISY_CLF_CKPT).exists():
        print("  NOT FOUND. Train first:")
        print(f"  python -m src.train.train_noisy_classifier{' --state' if USE_STATE else ''}")
        return
    noisy_ckpt = torch.load(NOISY_CLF_CKPT, map_location=DEVICE)
    cfg_clf    = noisy_ckpt["config"]
    noisy_clf  = create_clstm_classifier(
        n_sensors=cfg_clf["n_sensors"],
        n_classes=cfg_clf["n_classes"],
        cnn_channels=cfg_clf["cnn_channels"],
        lstm_hidden=cfg_clf["lstm_hidden"],
        d_attn=cfg_clf["d_attn"],
        n_heads=cfg_clf["n_heads"],
        n_layers=cfg_clf["n_layers"],
        dropout=0.0,
    ).to(DEVICE)
    noisy_clf.load_state_dict(noisy_ckpt["model_state"])
    noisy_clf.train()   # keep dropout off but enable grad
    print(f"  acc={noisy_ckpt['accuracy']:.4f}")

    # Raw classifier (for final classification)
    print(f"Loading Raw Classifier from {RAW_CLF_CKPT}...")
    raw_ckpt   = torch.load(RAW_CLF_CKPT, map_location=DEVICE)
    cfg_raw    = raw_ckpt["config"]
    raw_clf    = create_clstm_classifier(
        n_sensors=cfg_raw["n_sensors"],
        n_classes=cfg_raw["n_classes"],
        cnn_channels=cfg_raw["cnn_channels"],
        lstm_hidden=cfg_raw["lstm_hidden"],
        d_attn=cfg_raw["d_attn"],
        n_heads=cfg_raw["n_heads"],
        n_layers=cfg_raw["n_layers"],
        dropout=0.0,
    ).to(DEVICE)
    raw_clf.load_state_dict(raw_ckpt["model_state"])
    raw_clf.eval()
    print(f"  acc={raw_ckpt.get('accuracy', float('nan')):.4f}")

    # Mean signals
    print("\nComputing mean signals...")
    mean_signals = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="mean", leave=False):
            for name in SENSOR_NAMES:
                x = batch[name].to(DEVICE).float().permute(0, 2, 1)
                x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
                mean_signals[name].append(x.cpu())
    mean_norm = {}
    for name in SENSOR_NAMES:
        i    = SENSOR_NAMES.index(name)
        m    = torch.cat(mean_signals[name]).mean(0, keepdim=True).to(DEVICE)
        mean_norm[name] = (m - norm_mean[i:i+1, :, None]) / norm_std[i:i+1, :, None]

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = [
        ("phone_acc",   ["phone_acc"]),
        ("watch_acc",   ["watch_acc"]),
        ("glasses_acc", ["glasses_acc"]),
        ("phone_all",   ["phone_acc","phone_gyro","phone_grav","phone_lacc"]),
        ("watch_all",   ["watch_acc","watch_gyro"]),
        ("only_watch",  ["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"]),
        ("only_phone",  ["watch_acc","watch_gyro","glasses_acc"]),
    ]

    results_guided = {}
    results_mean   = {}

    for pat, missing_sensors in scenarios:
        missing_idx = [SENSOR_NAMES.index(s) for s in missing_sensors]

        guided_probs = []
        mean_probs   = []
        all_labels   = []

        for batch in tqdm(test_loader, desc=f"{pat:<14}", leave=False):
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            # Stack + normalize
            parts = []
            for name in SENSOR_NAMES:
                x = batch[name].to(DEVICE).float().permute(0, 2, 1)
                x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
                parts.append(x)
            stacked = torch.stack(parts, dim=1)   # (B, K, C, T_COMMON)
            stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                         / norm_std[None, :, :, None]

            observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
            for i in missing_idx:
                observed_mask[:, i] = 0.0

            # --- Guided imputation ---
            imputed_norm = ddim_impute_guided(
                diff_model, noisy_clf, stacked_norm, observed_mask,
                labels, alpha_bar, T_diff, DDIM_STEPS, GUIDANCE_W,
            )
            imputed = imputed_norm * norm_std[None, :, :, None] \
                    + norm_mean[None, :, :, None]

            guided_signals = {}
            for i, name in enumerate(SENSOR_NAMES):
                sig = imputed[:, i]
                guided_signals[name] = F.interpolate(
                    sig, size=NATIVE_LENS[name], mode='linear', align_corners=False,
                )

            # --- Mean fill ---
            mean_signals_batch = {}
            for i, name in enumerate(SENSOR_NAMES):
                if i in missing_idx:
                    mn = mean_norm[name] * norm_std[i:i+1, :, None] \
                       + norm_mean[i:i+1, :, None]
                    sig = mn.expand(B, -1, -1)
                else:
                    sig = stacked[:, i]
                mean_signals_batch[name] = F.interpolate(
                    sig, size=NATIVE_LENS[name], mode='linear', align_corners=False,
                )

            with torch.no_grad():
                gp = F.softmax(raw_clf(guided_signals, SENSOR_NAMES), dim=1)
                mp = F.softmax(raw_clf(mean_signals_batch, SENSOR_NAMES), dim=1)

            guided_probs.extend(gp.cpu().numpy())
            mean_probs.extend(mp.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results_guided[pat] = compute_metrics(all_labels, guided_probs, n_classes)
        results_mean[pat]   = compute_metrics(all_labels, mean_probs,   n_classes)

    # ============================================================
    # Print results
    # ============================================================
    fmt = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}"

    print(f"\n{'='*100}")
    print(f"CLASSIFIER-GUIDED DIFFUSION vs MEAN-FILL  |  C-LSTM-A {task}  |  w={GUIDANCE_W}")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Guided':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 60)
    wins = 0
    for pat, _ in scenarios:
        g = results_guided[pat]
        m = results_mean[pat]
        winner = "Guided" if g["acc"] >= m["acc"] else "Mean"
        delta  = f"Δ={g['acc']-m['acc']:+.3f}"
        if g["acc"] >= m["acc"]:
            wins += 1
        print(f"  {pat:<14} {fmt(g):>14} {fmt(m):>14}   → {winner}  {delta}")
    print(f"\n  Guided wins: {wins}/{len(scenarios)}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
