# ============================================================
# Evaluate Cross-Sensor Signal Imputation (no VAE, no Diffusion)
#
# Pipeline:
#   available signals → cross-sensor imputation → missing signals
#   all signals → C-LSTM-A (raw) → classification
#
# Compares:
#   - Baseline:    all sensors real
#   - CrossImp:    missing sensor via cross-sensor imputation (MSE)
#   - Mean-fill:   missing sensor = training mean signal
#
# Usage:
#   python -m src.eval.evaluate_signal_cross_imputation
#   python -m src.eval.evaluate_signal_cross_imputation --state
# ============================================================

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.models.clstm_classifier import create_clstm_classifier
from src.models.signal_cross_imputation import create_signal_cross_imputation, T_COMMON
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

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
tag = "state" if USE_STATE else "behavioral"

IMP_DIR         = Path("checkpoints/signal_cross_imputation")
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_{tag}/best_model.pt"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

NATIVE_LENS = {name: SENSOR_SPECS[name]["seq_len"] for name in SENSOR_NAMES}
BATCH_SIZE  = 128


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
    print(f"Signal Cross-Sensor Imputation Eval (MSE, no VAE)  |  C-LSTM-A {task}")
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

    # Imputation model
    print(f"Loading Signal Cross-Sensor Imputation model...")
    imp_ckpt  = torch.load(IMP_DIR / "best_model.pt", map_location=DEVICE)
    cfg       = imp_ckpt["config"]
    imp_model = create_signal_cross_imputation(
        n_sensors=cfg["n_sensors"],
        in_channels=cfg["in_channels"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    imp_model.load_state_dict(imp_ckpt["model_state"])
    imp_model.eval()
    norm_stats = torch.load(IMP_DIR / "normalization_stats.pt", map_location=DEVICE)
    norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
    norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
    print(f"  loss={imp_ckpt['loss']:.6f}, epoch={imp_ckpt['epoch']}")

    # Classifier
    print(f"Loading C-LSTM-A (raw) from {CLASSIFIER_CKPT}...")
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
        "phone_acc+crossimp":     (["phone_acc"], "crossimp"),
        "phone_acc+mean":         (["phone_acc"], "mean"),
        "watch_acc+crossimp":     (["watch_acc"], "crossimp"),
        "watch_acc+mean":         (["watch_acc"], "mean"),
        "glasses_acc+crossimp":   (["glasses_acc"], "crossimp"),
        "glasses_acc+mean":       (["glasses_acc"], "mean"),
        "phone_all+crossimp":     (["phone_acc","phone_gyro","phone_grav","phone_lacc"], "crossimp"),
        "phone_all+mean":         (["phone_acc","phone_gyro","phone_grav","phone_lacc"], "mean"),
        "watch_all+crossimp":     (["watch_acc","watch_gyro"], "crossimp"),
        "watch_all+mean":         (["watch_acc","watch_gyro"], "mean"),
        "only_watch+crossimp":    (["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"], "crossimp"),
        "only_watch+mean":        (["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"], "mean"),
        "only_phone+crossimp":    (["watch_acc","watch_gyro","glasses_acc"], "crossimp"),
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

                elif mode == "crossimp":
                    # Stack → normalize → zero missing → impute → denormalize → native len
                    parts = []
                    for name in SENSOR_NAMES:
                        x = signals[name].float()
                        x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
                        parts.append(x)
                    stacked = torch.stack(parts, dim=1)   # (B, K, C, T_COMMON)

                    stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                                 / norm_std[None, :, :, None]

                    observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                    inp = stacked_norm.clone()
                    for i in missing_idx:
                        observed_mask[:, i] = 0.0
                        inp[:, i] = 0.0

                    pred_norm = imp_model(inp, observed_mask)   # (B, K, C, T_COMMON)

                    # Denormalize and resize back to native lengths
                    pred = pred_norm * norm_std[None, :, :, None] \
                         + norm_mean[None, :, :, None]

                    for i in missing_idx:
                        name = SENSOR_NAMES[i]
                        gen  = pred[:, i]   # (B, C, T_COMMON)
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
    print(f"SIGNAL CROSS-SENSOR IMPUTATION EVAL  |  C-LSTM-A {task}")
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
    print("  COMPARISON: Cross-Sensor Imputation vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'CrossImp':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        d = results.get(f"{pat}+crossimp")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("CrossImp", d), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = f"  Δ={d['acc']-m['acc']:+.3f}" if d and m else ""
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
