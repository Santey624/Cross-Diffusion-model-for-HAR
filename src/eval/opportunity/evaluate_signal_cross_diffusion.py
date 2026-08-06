# ============================================================
# Evaluate Cross-Sensor Signal Diffusion Imputation (no VAE)  —  Opportunity
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
#   python -m src.eval.opportunity.evaluate_signal_cross_diffusion
#   python -m src.eval.opportunity.evaluate_signal_cross_diffusion --track locomotion
#   python -m src.eval.opportunity.evaluate_signal_cross_diffusion --track locomotion --augmented-cross
#
# --augmented-cross loads the robustness-trained classifier
#   checkpoints/clstm_raw_opportunity_{track}_augment_cross/
# and writes separate metrics files so original results are preserved:
#   eval_outputs/opportunity/{track}_augment_cross_metrics.{json,csv}
# ============================================================

import sys
import csv
import json
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
from src.data.opportunity.opportunity_constants import (
    OPP_SENSOR_NAMES, OPP_SENSOR_FILES, OPP_DEVICE_GROUPS, DEFAULT_LABEL_TRACK,
)
from src.data.opportunity.opportunity_labeled_dataset import OpportunityLabeledDataset

SENSOR_NAMES  = OPP_SENSOR_NAMES
DEVICE_GROUPS = OPP_DEVICE_GROUPS


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label track (analog of CogAge's --state). Default: locomotion.
TRACK = DEFAULT_LABEL_TRACK
for i, a in enumerate(sys.argv):
    if a == "--track" and i + 1 < len(sys.argv):
        TRACK = sys.argv[i + 1]
        break
tag = TRACK

# Use robustness-trained classifier (trained with diffusion imputation)
AUGMENTED_CROSS = "--augmented-cross" in sys.argv
clf_suffix = "_augment_cross" if AUGMENTED_CROSS else ""
metrics_tag = f"{tag}_augment_cross" if AUGMENTED_CROSS else tag

OPP_ROOT        = "data/opportunity/arrays"
DIFF_DIR        = Path("checkpoints/opportunity_signal_cross_diffusion")
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_opportunity_{tag}{clf_suffix}/best_model.pt"

NATIVE_LENS = {name: OPP_SENSOR_FILES[name][1] for name in SENSOR_NAMES}
DDIM_STEPS  = 25
BATCH_SIZE  = 32

OUT_DIR = Path("eval_outputs/opportunity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return {
        "acc": float(acc),
        "af1": float(af1),
        "map": float(map_score),
        "auc": float(auc_score),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    task = f"Opportunity ({TRACK})"
    clf_mode = "augment_cross" if AUGMENTED_CROSS else "real_only"
    print(f"\n{'='*75}")
    print(f"Signal Cross-Sensor Diffusion Eval (no VAE)  |  C-LSTM-A {task}")
    print(f"Classifier: {clf_mode}  |  ckpt: {CLASSIFIER_CKPT}")
    print(f"{'='*75}\n")

    test_ds  = OpportunityLabeledDataset(OPP_ROOT, "testing",  label_track=TRACK)
    train_ds = OpportunityLabeledDataset(OPP_ROOT, "training", label_track=TRACK)
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
        print("  NOT FOUND. Train the Opportunity raw classifier first.")
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
    scenarios = {"all_real": ([], "real")}
    # Single sensor missing
    for name in SENSOR_NAMES:
        scenarios[f"{name}+crossdiff"] = ([name], "crossdiff")
        scenarios[f"{name}+mean"]      = ([name], "mean")
    # Device-level missing
    for dev, members in DEVICE_GROUPS.items():
        scenarios[f"{dev}_all+crossdiff"] = (list(members), "crossdiff")
        scenarios[f"{dev}_all+mean"]      = (list(members), "mean")
    # Only one device available
    for dev, members in DEVICE_GROUPS.items():
        others = [s for s in SENSOR_NAMES if s not in members]
        scenarios[f"only_{dev}+crossdiff"] = (others, "crossdiff")
        scenarios[f"only_{dev}+mean"]      = (others, "mean")

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []
        missing_idx = [SENSOR_NAMES.index(s) for s in missing_sensors]

        for batch in tqdm(test_loader, desc=f"{scenario_name:<28}", leave=False):
            labels  = batch["label"].to(DEVICE)
            B       = labels.size(0)

            # Build signal dict — native lengths
            signals = {}
            for name in SENSOR_NAMES:
                x = batch[name].to(DEVICE).permute(0, 2, 1).float()
                signals[name] = x

            with torch.no_grad():
                if mode == "real":
                    pass

                elif mode == "crossdiff":
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

    groups = []
    # Single sensors
    for name in SENSOR_NAMES:
        groups.append((name, [name]))
    # Device-level
    for dev, members in DEVICE_GROUPS.items():
        groups.append((f"{dev}_all", list(members)))
    # Only one device available
    for dev, members in DEVICE_GROUPS.items():
        others = [s for s in SENSOR_NAMES if s not in members]
        groups.append((f"only_{dev}", others))

    real = results.get("all_real", {})
    fmt  = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}" if m else "    —    "

    print(f"\n{'='*100}")
    print("  COMPARISON: Signal Cross-Sensor Diffusion vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<16} {'Baseline':>14} {'CrossDiff':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 74)
    print(f"  {'all_real':<16} {fmt(real):>14}")

    cd_wins = 0
    comparison_rows = []
    for pat, _ in groups:
        d = results.get(f"{pat}+crossdiff")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("CrossDiff", d), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        if winner == "CrossDiff":
            cd_wins += 1
        delta_val = (d["acc"] - m["acc"]) if d and m else None
        delta  = f"  Δ={delta_val:+.3f}" if delta_val is not None else ""
        print(f"  {pat:<16} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14}   → {winner}{delta}")
        comparison_rows.append({
            "pattern": pat,
            "winner": winner,
            "delta_acc": None if delta_val is None else float(delta_val),
        })

    print(f"\n  CrossDiff wins: {cd_wins}/{len(groups)}")
    print(f"{'='*100}\n")

    # ============================================================
    # Persist results
    # ============================================================
    json_path = OUT_DIR / f"{metrics_tag}_metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "track": TRACK,
            "classifier": clf_mode,
            "classifier_ckpt": CLASSIFIER_CKPT,
            "n_classes": int(n_classes),
            "scenarios": results,
            "comparison": comparison_rows,
            "crossdiff_wins": int(cd_wins),
            "n_groups": int(len(groups)),
        }, f, indent=2)
    print(f"Saved: {json_path}")

    csv_path = OUT_DIR / f"{metrics_tag}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "acc", "af1", "map", "auc"])
        for name, m in results.items():
            writer.writerow([name, m["acc"], m["af1"], m["map"], m["auc"]])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
