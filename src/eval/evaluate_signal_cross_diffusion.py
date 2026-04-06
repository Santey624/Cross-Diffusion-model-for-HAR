# ============================================================
# Evaluate Cross-Sensor Signal Diffusion Imputation (no VAE)
#
# Pipeline:
#   available signals → cross-sensor diffusion → missing signals
#   all signals → C-LSTM-A (raw) → classification
#
# Compares:
#   - Baseline:    all sensors real
#   - CrossDiff:   missing sensor via cross-sensor signal diffusion
#   - Mean-fill:   missing sensor = training mean signal
#
# Usage:
#   python -m src.eval.evaluate_signal_cross_diffusion
#   python -m src.eval.evaluate_signal_cross_diffusion --state
#   python -m src.eval.evaluate_signal_cross_diffusion --state --augmented-cross
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
from src.models.signal_cross_diffusion import create_signal_cross_diffusion, T_COMMON
from src.models.sensor_vae import SENSOR_NAMES, SENSOR_SPECS
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE        = "--state"           in sys.argv
AUGMENTED_CROSS       = "--augmented-cross"       in sys.argv
AUGMENTED_CROSS_RECON = "--augmented-cross-recon" in sys.argv
USE_RECON             = "--recon"                 in sys.argv

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
tag = "state" if USE_STATE else "behavioral"
if AUGMENTED_CROSS_RECON:
    aug = "_augment_cross_recon"
elif AUGMENTED_CROSS:
    aug = "_augment_cross"
else:
    aug = ""

DIFF_DIR        = Path("checkpoints/signal_cross_diffusion_recon" if USE_RECON
                       else "checkpoints/signal_cross_diffusion")
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_{tag}{aug}/best_model.pt"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

NATIVE_LENS = {name: SENSOR_SPECS[name]["seq_len"] for name in SENSOR_NAMES}
DDIM_STEPS  = 25
BATCH_SIZE  = 32


# ============================================================
# Noise schedule
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float()


def stack_signals(batch, device):
    """Returns (B, K, C, T_COMMON)"""
    parts = []
    for name in SENSOR_NAMES:
        x = batch[name].to(device).permute(0, 2, 1).float()
        x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
        parts.append(x)
    return torch.stack(parts, dim=1)


# ============================================================
# DDIM sampling (cross-sensor)
# ============================================================
@torch.no_grad()
def ddim_impute(model, stacked_norm, observed_mask, alpha_bar, T, ddim_steps):
    """
    stacked_norm: (B, K, C, T_COMMON) — observed=normalized real, missing=noise
    observed_mask: (B, K)
    Returns: (B, K, C, T_COMMON) — imputed (only missing sensors changed)
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
        t_batch    = torch.full((B,), t_now, device=device, dtype=torch.long)

        # Pin observed sensors to real values
        noisy = stacked_norm.clone()
        for i in missing_idx:
            noisy[:, i] = z[:, i]

        noise_pred = model(noisy, t_batch, observed_mask)

        ab_now  = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        for i in missing_idx:
            pred_x0 = ((z[:, i] - torch.sqrt(1 - ab_now) * noise_pred[:, i])
                       / torch.sqrt(ab_now)).clamp(-5, 5)
            z[:, i] = (torch.sqrt(ab_next) * pred_x0
                       + torch.sqrt(1 - ab_next) * noise_pred[:, i])

    # Return full stacked with imputed missing sensors
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
    print(f"Signal Cross-Sensor Diffusion Eval (no VAE)  |  C-LSTM-A {task}  |  aug_cross={AUGMENTED_CROSS}")
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
    print(f"Loading Signal Cross-Sensor Diffusion...")
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
    T_diff    = diff_ckpt["T"]
    betas     = cosine_beta_schedule(T_diff)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    norm_stats = torch.load(DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
    norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)  # (K, C)
    norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)  # (K, C)
    print(f"  loss={diff_ckpt['loss']:.6f}, epoch={diff_ckpt['epoch']}")

    # Classifier
    print(f"Loading C-LSTM-A (raw) from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print("  NOT FOUND. Train first:")
        s = " --state" if USE_STATE else ""
        a = " --augment-cross" if AUGMENTED_CROSS else ""
        print(f"  python -m src.train.train_clstm_raw{s}{a}")
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

    # Mean signals (training set)
    print("\nComputing mean signals...")
    mean_signals = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="mean precompute", leave=False):
            for name in SENSOR_NAMES:
                mean_signals[name].append(batch[name].permute(0, 2, 1).float())
    mean_signals = {
        k: torch.cat(v).mean(0, keepdim=True).to(DEVICE)
        for k, v in mean_signals.items()
    }

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = {
        "all_real":               ([], "real"),
        "phone_acc+crossdiff":    (["phone_acc"], "crossdiff"),
        "phone_acc+mean":         (["phone_acc"], "mean"),
        "watch_acc+crossdiff":    (["watch_acc"], "crossdiff"),
        "watch_acc+mean":         (["watch_acc"], "mean"),
        "glasses_acc+crossdiff":  (["glasses_acc"], "crossdiff"),
        "glasses_acc+mean":       (["glasses_acc"], "mean"),
        "phone_all+crossdiff":    (["phone_acc","phone_gyro","phone_grav","phone_lacc"], "crossdiff"),
        "phone_all+mean":         (["phone_acc","phone_gyro","phone_grav","phone_lacc"], "mean"),
        "watch_all+crossdiff":    (["watch_acc","watch_gyro"], "crossdiff"),
        "watch_all+mean":         (["watch_acc","watch_gyro"], "mean"),
        "only_watch+crossdiff":   (["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"], "crossdiff"),
        "only_watch+mean":        (["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"], "mean"),
        "only_phone+crossdiff":   (["watch_acc","watch_gyro","glasses_acc"], "crossdiff"),
        "only_phone+mean":        (["watch_acc","watch_gyro","glasses_acc"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []
        missing_idx = [SENSOR_NAMES.index(s) for s in missing_sensors]

        for batch in tqdm(test_loader, desc=f"{scenario_name:<28}", leave=False):
            labels  = batch["label"].to(DEVICE)
            B       = labels.size(0)
            signals = {
                name: batch[name].to(DEVICE).permute(0, 2, 1).float()
                for name in SENSOR_NAMES
            }

            with torch.no_grad():
                if mode == "real":
                    pass

                elif mode == "crossdiff":
                    # Normalize → stack → diffusion impute → denormalize → native len
                    stacked = stack_signals(batch, DEVICE)   # (B, K, C, T_COMMON)
                    stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                                 / norm_std[None, :, :, None]

                    observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                    for i in missing_idx:
                        observed_mask[:, i] = 0.0

                    imputed_norm = ddim_impute(
                        diff_model, stacked_norm, observed_mask,
                        alpha_bar, T_diff, DDIM_STEPS,
                    )
                    # Denormalize and resize back to native lengths
                    imputed = imputed_norm * norm_std[None, :, :, None] \
                            + norm_mean[None, :, :, None]

                    for i in missing_idx:
                        name = SENSOR_NAMES[i]
                        gen  = imputed[:, i]   # (B, C, T_COMMON)
                        signals[name] = F.interpolate(
                            gen, size=NATIVE_LENS[name],
                            mode='linear', align_corners=False,
                        )

                else:  # mean
                    for name in missing_sensors:
                        signals[name] = mean_signals[name].expand(B, -1, -1)

                logits = classifier(signals, SENSOR_NAMES)
                probs  = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # Results
    # ============================================================
    print(f"\n{'='*100}")
    print(f"SIGNAL CROSS-SENSOR DIFFUSION EVAL  |  C-LSTM-A {task}")
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
    print("  COMPARISON: Signal Cross-Sensor Diffusion vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'CrossDiff':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        d = results.get(f"{pat}+crossdiff")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("CrossDiff", d), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = f"  Δ={d['acc']-m['acc']:+.3f}" if d and m else ""
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
