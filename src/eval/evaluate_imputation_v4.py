# ============================================================
# Evaluate Imputation Quality — VAE V4 Direct Imputation
#
# Pipeline:
#   sensor signal → VAE V4 encode (available sensors)
#                 → V4.impute() for missing sensors
#                 → C-LSTM-A
#
# Compares:
#   - Baseline:    all sensors real (V4 encode → decode)
#   - V4 impute:   missing sensor via V4.impute() (PoE z_shared + z_private=0)
#   - Mean-fill:   missing sensor replaced by mean decoded signal from training set
#
# Usage:
#   python -m src.eval.evaluate_imputation_v4
#   python -m src.eval.evaluate_imputation_v4 --state
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

from src.models.sensor_vae_v4 import SensorSharedPrivateVAE, SENSOR_NAMES
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset, CogAgeLabeledDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STATE = "--state" in sys.argv

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v4/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

if USE_STATE:
    CLASSIFIER_CKPT = "checkpoints/clstm_v4_state/best_model.pt"
    DATA_ROOTS      = {"state": "data/cogage/python/arrays/state"}
else:
    CLASSIFIER_CKPT = "checkpoints/clstm_v4_behavioral/best_model.pt"
    DATA_ROOTS      = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

BATCH_SIZE = 32


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
    print(f"\n{'='*70}")
    print(f"Imputation Eval — VAE V4 Direct Imputation | C-LSTM-A {task}")
    print(f"{'='*70}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print(f"Loading test dataset ({', '.join(DATA_ROOTS.keys())})...")
    if USE_STATE:
        test_ds   = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing", normalizer)
        train_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
    else:
        test_ds   = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)
        train_ds  = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    n_classes = test_ds.n_classes

    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"  Test samples: {len(test_ds)}, Classes: {n_classes}")

    # VAE V4
    print("\nLoading VAE V4...")
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    cfg_vae  = vae_ckpt["config"]
    vae = SensorSharedPrivateVAE(
        d_shared=cfg_vae["d_shared"],
        d_private=cfg_vae["d_private"],
        t_lat=cfg_vae["t_lat"],
    ).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    print(f"  d_shared={cfg_vae['d_shared']}, d_private={cfg_vae['d_private']}, "
          f"t_lat={cfg_vae['t_lat']}, epoch={vae_ckpt['epoch']}, "
          f"impute={vae_ckpt['test_impute']:.6f}")

    # C-LSTM-A Classifier
    print(f"\nLoading C-LSTM-A from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print(f"  NOT FOUND. Train first with:")
        print(f"  python -m src.train.train_clstm_v4{' --state' if USE_STATE else ''}")
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
    print(f"  Loaded (best acc: {clf_ckpt.get('accuracy', float('nan')):.4f})")

    # Pre-compute mean decoded signals from training set (for mean-fill baseline)
    print("\nComputing mean decoded signals from training set...")
    mean_signals = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="mean-fill precompute", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs, _, _ = vae(sensor_data, mask_ratio=0.0)
            for name in SENSOR_NAMES:
                # recon: (B, T, C) → (B, C, T)
                mean_signals[name].append(outputs[name]["recon"].permute(0, 2, 1).cpu())
    mean_signals = {
        k: torch.cat(v).mean(0, keepdim=True).to(DEVICE)
        for k, v in mean_signals.items()
    }

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = {
        # Baseline
        "all_real":         ([], "real"),
        # Single sensor missing
        "phone_acc+v4":     (["phone_acc"], "v4"),
        "phone_acc+mean":   (["phone_acc"], "mean"),
        "watch_acc+v4":     (["watch_acc"], "v4"),
        "watch_acc+mean":   (["watch_acc"], "mean"),
        "glasses_acc+v4":   (["glasses_acc"], "v4"),
        "glasses_acc+mean": (["glasses_acc"], "mean"),
        # Full device missing
        "phone_all+v4":     (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "v4"),
        "phone_all+mean":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "mean"),
        "watch_all+v4":     (["watch_acc", "watch_gyro"], "v4"),
        "watch_all+mean":   (["watch_acc", "watch_gyro"], "mean"),
        # Only one device available
        "only_watch+v4":    (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "v4"),
        "only_watch+mean":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "mean"),
        "only_phone+v4":    (["watch_acc", "watch_gyro", "glasses_acc"], "v4"),
        "only_phone+mean":  (["watch_acc", "watch_gyro", "glasses_acc"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []

        for batch in tqdm(test_loader, desc=f"{scenario_name:<24}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            with torch.no_grad():
                avail_data = {k: v for k, v in sensor_data.items()
                              if k not in missing_sensors}

                if mode == "real":
                    # All sensors real
                    outputs, _, _ = vae(sensor_data, mask_ratio=0.0)
                    decoded = {
                        name: out["recon"].permute(0, 2, 1)
                        for name, out in outputs.items()
                    }

                elif mode == "v4":
                    # Available sensors: encode + decode via V4
                    outputs, _, _ = vae(avail_data, mask_ratio=0.0)
                    decoded = {
                        name: out["recon"].permute(0, 2, 1)
                        for name, out in outputs.items()
                    }
                    # Missing sensors: V4 direct imputation
                    for name in missing_sensors:
                        imputed = vae.impute(avail_data, name)      # (B, T, C)
                        decoded[name] = imputed.permute(0, 2, 1)    # (B, C, T)

                else:  # mean
                    # Available sensors: encode + decode via V4
                    outputs, _, _ = vae(avail_data, mask_ratio=0.0)
                    decoded = {
                        name: out["recon"].permute(0, 2, 1)
                        for name, out in outputs.items()
                    }
                    # Missing sensors: mean signal from training set
                    for name in missing_sensors:
                        decoded[name] = mean_signals[name].expand(B, -1, -1)

                logits = classifier(decoded, SENSOR_NAMES)
                probs  = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # Results Table
    # ============================================================
    print(f"\n{'='*100}")
    print(f"IMPUTATION EVAL — VAE V4 Direct Imputation  |  C-LSTM-A {task}")
    print(f"{'='*100}")
    print(f"  {'Scenario':<26} {'Acc':>7} {'AF1':>7} {'MAP':>7} {'AUC':>7}")
    print("  " + "-" * 60)
    for name, m in results.items():
        print(f"  {name:<26} {m['acc']:>7.4f} {m['af1']:>7.4f} "
              f"{m['map']:>7.4f} {m['auc']:>7.4f}")

    # Grouped comparison
    groups = [
        ("phone_acc",  ["phone_acc"]),
        ("watch_acc",  ["watch_acc"]),
        ("glasses_acc",["glasses_acc"]),
        ("phone_all",  ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"]),
        ("watch_all",  ["watch_acc", "watch_gyro"]),
        ("only_watch", ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"]),
        ("only_phone", ["watch_acc", "watch_gyro", "glasses_acc"]),
    ]

    real = results.get("all_real", {})
    fmt  = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}" if m else "    —    "

    print(f"\n{'='*100}")
    print("  COMPARISON: V4 Impute vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'V4 Impute':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        v = results.get(f"{pat}+v4")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("V4", v), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = ""
        if v and m:
            diff_acc = v["acc"] - m["acc"]
            delta = f"  Δ={diff_acc:+.3f}"
        print(f"  {pat:<14} {fmt(real):>14} {fmt(v):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
