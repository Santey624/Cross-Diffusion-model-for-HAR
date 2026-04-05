# ============================================================
# Evaluate Signal-Level Class-Conditional Diffusion Imputation
#
# Pipeline (no VAE):
#   1. Available sensors → C-LSTM-A (raw) → predicted class
#   2. Predicted class → signal diffusion → missing sensor signal
#   3. All raw signals → C-LSTM-A → final prediction
#
# Compares:
#   - Baseline:    all sensors real
#   - Signal-cond: missing sensor via signal class-cond diffusion
#   - Zero-fill:   missing sensor set to zeros
#   - Mean-fill:   missing sensor = training mean signal
#
# Usage:
#   python -m src.eval.evaluate_signal_diffusion
#   python -m src.eval.evaluate_signal_diffusion --state
#   python -m src.eval.evaluate_signal_diffusion --augmented   (augmented classifier)
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
from src.models.signal_class_diffusion import (
    create_signal_class_diffusion, SENSOR_T, T_MODEL,
)
from src.models.sensor_vae import SENSOR_NAMES, SENSOR_SPECS
from src.data.cogage_labeled_dataset import (
    get_combined_labeled_dataset, CogAgeLabeledDataset,
)
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE  = "--state"     in sys.argv
AUGMENTED  = "--augmented" in sys.argv

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
tag = "state" if USE_STATE else "behavioral"
aug = "_augment" if AUGMENTED else ""

DIFF_DIR        = Path(f"checkpoints/signal_class_diffusion_{tag}")
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_{tag}{aug}/best_model.pt"

if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

NATIVE_LENS = {name: SENSOR_SPECS[name]["seq_len"] for name in SENSOR_NAMES}
DDIM_STEPS  = 50
BATCH_SIZE  = 32


# ============================================================
# Noise schedule + DDIM
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float()


@torch.no_grad()
def ddim_sample_signal(model, class_label, sensor_id, alpha_bar, T, ddim_steps,
                        t_sensor=T_MODEL):
    B      = class_label.shape[0]
    device = class_label.device
    x      = torch.randn(B, 3, t_sensor, device=device)
    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)
    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch    = torch.full((B,), t_now, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch, class_label, sensor_id)
        ab_now     = alpha_bar[t_now]
        ab_next    = alpha_bar[t_next]
        pred_x0    = ((x - torch.sqrt(1 - ab_now) * noise_pred)
                      / torch.sqrt(ab_now)).clamp(-5, 5)
        x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
    return x


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
    print(f"Signal Diffusion Eval (no VAE)  |  C-LSTM-A {task}  |  aug={AUGMENTED}")
    print(f"{'='*75}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    if USE_STATE:
        test_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing",  normalizer)
        train_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
    else:
        test_ds  = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)
        train_ds = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    n_classes = test_ds.n_classes

    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    # Signal Diffusion
    print(f"Loading Signal Diffusion from {DIFF_DIR}...")
    diff_ckpt  = torch.load(DIFF_DIR / "best_model.pt", map_location=DEVICE)
    cfg_diff   = diff_ckpt["config"]
    diff_model = create_signal_class_diffusion(
        n_sensors=cfg_diff["n_sensors"],
        n_classes=cfg_diff["n_classes"],
        in_channels=cfg_diff["in_channels"],
        base_ch=cfg_diff["base_ch"],
        emb_dim=cfg_diff["emb_dim"],
    ).to(DEVICE)
    diff_model.load_state_dict(diff_ckpt["model_state"])
    diff_model.eval()
    T_diff    = diff_ckpt["T"]
    betas     = cosine_beta_schedule(T_diff)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    print(f"  loss={diff_ckpt['loss']:.6f}, epoch={diff_ckpt['epoch']}")

    # Classifier
    print(f"Loading C-LSTM-A from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print("  NOT FOUND. Train first:")
        s = " --state" if USE_STATE else ""
        a = " --augment" if AUGMENTED else ""
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

    # Mean signals from training set
    print("\nComputing mean signals from training set...")
    mean_signals = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="mean precompute", leave=False):
            for name in SENSOR_NAMES:
                mean_signals[name].append(batch[name].permute(0, 2, 1).float())
    mean_signals = {
        k: torch.cat(v).mean(0, keepdim=True).to(DEVICE)
        for k, v in mean_signals.items()
    }

    def predict_class_from_available(signals, missing_sensors):
        """Quick class prediction using available sensors + zero for missing."""
        sig = dict(signals)
        B   = next(iter(sig.values())).shape[0]
        for name in missing_sensors:
            sig[name] = torch.zeros(B, 3, NATIVE_LENS[name], device=DEVICE)
        logits = classifier(sig, SENSOR_NAMES)
        return logits.argmax(dim=1)

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = {
        "all_real":               ([], "real"),
        "phone_acc+sigdiff":      (["phone_acc"], "sigdiff"),
        "phone_acc+mean":         (["phone_acc"], "mean"),
        "watch_acc+sigdiff":      (["watch_acc"], "sigdiff"),
        "watch_acc+mean":         (["watch_acc"], "mean"),
        "glasses_acc+sigdiff":    (["glasses_acc"], "sigdiff"),
        "glasses_acc+mean":       (["glasses_acc"], "mean"),
        "phone_all+sigdiff":      (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "sigdiff"),
        "phone_all+mean":         (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "mean"),
        "watch_all+sigdiff":      (["watch_acc", "watch_gyro"], "sigdiff"),
        "watch_all+mean":         (["watch_acc", "watch_gyro"], "mean"),
        "only_watch+sigdiff":     (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "sigdiff"),
        "only_watch+mean":        (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "mean"),
        "only_phone+sigdiff":     (["watch_acc", "watch_gyro", "glasses_acc"], "sigdiff"),
        "only_phone+mean":        (["watch_acc", "watch_gyro", "glasses_acc"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []

        for batch in tqdm(test_loader, desc=f"{scenario_name:<28}", leave=False):
            labels  = batch["label"].to(DEVICE)
            B       = labels.size(0)
            signals = {
                name: batch[name].to(DEVICE).permute(0, 2, 1).float()
                for name in SENSOR_NAMES
            }

            with torch.no_grad():
                if mode == "real":
                    pass  # use signals as-is

                elif mode == "sigdiff":
                    pred_class = predict_class_from_available(signals, missing_sensors)
                    for name in missing_sensors:
                        sidx = torch.full((B,), SENSOR_NAMES.index(name),
                                          dtype=torch.long, device=DEVICE)
                        t_sensor = SENSOR_T.get(name, T_MODEL)
                        gen = ddim_sample_signal(
                            diff_model, pred_class, sidx,
                            alpha_bar, T_diff, DDIM_STEPS,
                            t_sensor=t_sensor,
                        )   # (B, 3, t_sensor)
                        if gen.shape[-1] != NATIVE_LENS[name]:
                            gen = F.interpolate(gen, size=NATIVE_LENS[name],
                                                mode='linear', align_corners=False)
                        signals[name] = gen

                else:  # mean
                    for name in missing_sensors:
                        signals[name] = mean_signals[name].expand(B, -1, -1)

                logits = classifier(signals, SENSOR_NAMES)
                probs  = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # Results Table
    # ============================================================
    print(f"\n{'='*100}")
    print(f"SIGNAL DIFFUSION EVAL (no VAE)  |  C-LSTM-A {task}  |  augmented={AUGMENTED}")
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
    print("  COMPARISON: Signal Diffusion vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'SigDiff':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        d = results.get(f"{pat}+sigdiff")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("SigDiff", d), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = f"  Δ={d['acc']-m['acc']:+.3f}" if d and m else ""
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
