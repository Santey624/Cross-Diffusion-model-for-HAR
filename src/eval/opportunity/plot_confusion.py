# ============================================================
# Confusion Matrix Plots for C-LSTM-A  —  Opportunity
#
# Few classes  (<= 8):  annotated heatmap with class ids
# Many classes (> 8):   heatmap + top-N confused pairs bar chart
#
# Usage:
#   python -m src.eval.opportunity.plot_confusion
#   python -m src.eval.opportunity.plot_confusion --track locomotion
#   python -m src.eval.opportunity.plot_confusion --track locomotion --scenario watch_all
#
# --scenario options (when using diffusion):
#   any single sensor name (e.g. back_acc, rua_gyro, ...),
#   back_all, right_arm_all, left_arm_all, shoes_all,
#   only_back, only_right_arm, only_left_arm, only_shoes
#   (omit --scenario to use all sensors / baseline)
# ============================================================

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

OPP_ROOT        = "data/opportunity/arrays"
DIFF_DIR        = Path("checkpoints/opportunity_signal_cross_diffusion")
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_opportunity_{tag}/best_model.pt"

NATIVE_LENS = {name: OPP_SENSOR_FILES[name][1] for name in SENSOR_NAMES}
BATCH_SIZE  = 32
DDIM_STEPS  = 25
TOP_N_PAIRS = 25   # top confused pairs shown in many-class bar chart
ANNOT_MAX   = 8    # <= this many classes -> annotated heatmap with ids

# Parse --scenario
SCENARIO = None
for i, a in enumerate(sys.argv):
    if a == "--scenario" and i + 1 < len(sys.argv):
        SCENARIO = sys.argv[i + 1]
        break

OUT_DIR = Path("eval_outputs/confusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Scenario -> missing sensor indices
# ============================================================
SCENARIO_MAP = {name: [SENSOR_NAMES.index(name)] for name in SENSOR_NAMES}
for dev, members in DEVICE_GROUPS.items():
    SCENARIO_MAP[f"{dev}_all"] = [SENSOR_NAMES.index(s) for s in members]
for dev, members in DEVICE_GROUPS.items():
    others = [s for s in SENSOR_NAMES if s not in members]
    SCENARIO_MAP[f"only_{dev}"] = [SENSOR_NAMES.index(s) for s in others]


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
    parts = []
    for name in SENSOR_NAMES:
        x = batch[name].to(device).permute(0, 2, 1).float()
        x = F.interpolate(x, size=T_COMMON, mode='linear', align_corners=False)
        parts.append(x)
    return torch.stack(parts, dim=1)   # (B, K, C, T_COMMON)


# ============================================================
# DDIM imputation (same as the metrics eval)
# ============================================================
@torch.no_grad()
def ddim_impute(model, stacked_norm, observed_mask, alpha_bar, T, ddim_steps):
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
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

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

    result = stacked_norm.clone()
    for i in missing_idx:
        result[:, i] = z[:, i]
    return result


# ============================================================
# Collect predictions
# ============================================================
@torch.no_grad()
def collect_preds(test_loader, classifier, diff_model, alpha_bar,
                  norm_mean, norm_std, missing_idx, n_classes):
    all_preds  = []
    all_labels = []

    use_diffusion = diff_model is not None and missing_idx is not None

    for batch in tqdm(test_loader, desc="Inference", leave=False):
        labels = batch["label"].numpy()
        B      = labels.shape[0]

        # Start with raw signals at native lengths (no interpolation degradation)
        signals_dict = {
            name: batch[name].to(DEVICE).permute(0, 2, 1).float()
            for name in SENSOR_NAMES
        }

        if use_diffusion:
            stacked = stack_signals(batch, DEVICE)           # (B, K, C, T_COMMON)
            stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                         / norm_std[None, :, :, None]

            observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
            for i in missing_idx:
                observed_mask[:, i] = 0.0
            imputed = ddim_impute(diff_model, stacked_norm, observed_mask,
                                  alpha_bar, diff_model_T, DDIM_STEPS)
            imputed_denorm = imputed * norm_std[None, :, :, None] \
                           + norm_mean[None, :, :, None]
            # Only replace missing sensors
            for i in missing_idx:
                name = SENSOR_NAMES[i]
                signals_dict[name] = F.interpolate(
                    imputed_denorm[:, i], size=NATIVE_LENS[name],
                    mode='linear', align_corners=False,
                )

        logits = classifier(signals_dict, SENSOR_NAMES)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


# ============================================================
# MAIN
# ============================================================
def main():
    global diff_model_T

    print(f"\n{'='*65}")
    print(f"Confusion Matrix  |  {tag}  |  scenario={SCENARIO or 'baseline (all sensors)'}")
    print(f"{'='*65}\n")

    test_ds = OpportunityLabeledDataset(OPP_ROOT, "testing", label_track=TRACK)
    n_classes = test_ds.n_classes
    idx_to_label = test_ds.idx_to_label   # contiguous idx -> raw label int

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Classifier
    print(f"Loading classifier from {CLASSIFIER_CKPT}...")
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
    print(f"  acc={clf_ckpt.get('accuracy', float('nan')):.4f}")

    # Diffusion (only needed when scenario is set)
    diff_model = None
    alpha_bar  = None
    norm_mean  = norm_std = None
    missing_idx = SCENARIO_MAP.get(SCENARIO) if SCENARIO else None

    if missing_idx is not None:
        print(f"Loading diffusion from {DIFF_DIR}...")
        diff_ckpt  = torch.load(DIFF_DIR / "best_model.pt", map_location=DEVICE)
        cfg        = diff_ckpt["config"]
        diff_model_T = diff_ckpt["T"]
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
        norm_stats = torch.load(DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
        norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
        norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
        betas      = cosine_beta_schedule(diff_ckpt["T"])
        alpha_bar  = torch.cumprod(1.0 - betas, dim=0)
    else:
        diff_model_T = 1000
        # Load norm_mean / norm_std even for baseline (harmless; unused)
        if (DIFF_DIR / "normalization_stats.pt").exists():
            norm_stats = torch.load(DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
            norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
            norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
        else:
            norm_mean = torch.zeros(len(SENSOR_NAMES), 3, device=DEVICE)
            norm_std  = torch.ones(len(SENSOR_NAMES),  3, device=DEVICE)

    # Run inference
    labels, preds = collect_preds(
        test_loader, classifier, diff_model, alpha_bar,
        norm_mean, norm_std, missing_idx, n_classes,
    )

    acc = (labels == preds).mean()
    print(f"\nAccuracy: {acc:.4f}  ({int((labels==preds).sum())}/{len(labels)})")

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))

    scenario_str = SCENARIO or "baseline"
    fname_base   = f"{tag}_{scenario_str}"

    # ============================================================
    # FEW CLASSES: annotated heatmap
    # ============================================================
    if n_classes <= ANNOT_MAX:
        class_names = [str(idx_to_label[i]) for i in range(n_classes)]

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.5 else "black")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title(f"{tag} Confusion Matrix — {scenario_str}  (acc={acc:.3f})", fontsize=13)
        plt.tight_layout()
        out_path = OUT_DIR / f"{fname_base}_confusion.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # ============================================================
    # MANY CLASSES: heatmap + top confused pairs
    # ============================================================
    else:
        # --- heatmap (row-normalised) ---
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(18, 16))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title(f"{tag} Confusion Matrix ({n_classes} classes) — {scenario_str}  (acc={acc:.3f})", fontsize=13)
        plt.tight_layout()
        out_path_hm = OUT_DIR / f"{fname_base}_confusion_heatmap.png"
        plt.savefig(out_path_hm, dpi=150)
        plt.close()
        print(f"Saved: {out_path_hm}")

        # --- top confused pairs ---
        pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    pairs.append((cm[i, j], idx_to_label[i], idx_to_label[j]))
        pairs.sort(reverse=True)
        top = pairs[:TOP_N_PAIRS]

        labels_pair = [f"{t}→{p}" for _, t, p in top]
        counts      = [c for c, _, _ in top]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(len(top)), counts, color="steelblue")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels_pair, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("# misclassified samples", fontsize=11)
        ax.set_title(f"Top-{TOP_N_PAIRS} Confused Pairs — {scenario_str}  (acc={acc:.3f})", fontsize=12)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va='center', fontsize=8)
        plt.tight_layout()
        out_path_pairs = OUT_DIR / f"{fname_base}_top_confused_pairs.png"
        plt.savefig(out_path_pairs, dpi=150)
        plt.close()
        print(f"Saved: {out_path_pairs}")

        # Print top pairs to terminal too
        print(f"\nTop-{TOP_N_PAIRS} most confused pairs (true→predicted):")
        for cnt, true_lbl, pred_lbl in top:
            print(f"  {true_lbl:>4} → {pred_lbl:<4}  ({cnt} samples)")


if __name__ == "__main__":
    main()
