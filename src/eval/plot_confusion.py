# ============================================================
# Confusion Matrix Plots for C-LSTM-A
#
# State  (6 classes):  standard heatmap with class names
# Behave (55 classes): heatmap + top-N confused pairs bar chart
#
# Usage:
#   python -m src.eval.plot_confusion --state
#   python -m src.eval.plot_confusion
#   python -m src.eval.plot_confusion --state  --recon-v3  --scenario watch_all
#   python -m src.eval.plot_confusion          --recon-v3  --scenario watch_all
#
# --scenario options (when using diffusion):
#   phone_acc, phone_gyro, phone_grav, phone_lacc,
#   watch_acc, watch_gyro, glasses_acc,
#   phone_all, watch_all, only_watch, only_phone, only_glasses
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

USE_RECON_V3 = "--recon-v3" in sys.argv
USE_RECON_V2 = "--recon-v2" in sys.argv
USE_RECON    = "--recon"    in sys.argv
DIFF_DIR = Path(
    "checkpoints/signal_cross_diffusion_recon_v3" if USE_RECON_V3 else
    "checkpoints/signal_cross_diffusion_recon_v2" if USE_RECON_V2 else
    "checkpoints/signal_cross_diffusion_recon"    if USE_RECON    else
    "checkpoints/signal_cross_diffusion"
)

tag = "state" if USE_STATE else "behavioral"
CLASSIFIER_CKPT = f"checkpoints/clstm_raw_{tag}/best_model.pt"

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
if USE_STATE:
    DATA_ROOTS = {"state": "data/cogage/python/arrays/state"}
else:
    DATA_ROOTS = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

BATCH_SIZE  = 32
DDIM_STEPS  = 25
TOP_N_PAIRS = 25   # top confused pairs shown in behavioral bar chart

# Parse --scenario
SCENARIO = None
for i, a in enumerate(sys.argv):
    if a == "--scenario" and i + 1 < len(sys.argv):
        SCENARIO = sys.argv[i + 1]
        break

OUT_DIR = Path("eval_outputs/confusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Scenario → missing sensor indices
# ============================================================
SCENARIO_MAP = {
    "phone_acc":   [SENSOR_NAMES.index("phone_acc")],
    "phone_gyro":  [SENSOR_NAMES.index("phone_gyro")],
    "phone_grav":  [SENSOR_NAMES.index("phone_grav")],
    "phone_lacc":  [SENSOR_NAMES.index("phone_lacc")],
    "watch_acc":   [SENSOR_NAMES.index("watch_acc")],
    "watch_gyro":  [SENSOR_NAMES.index("watch_gyro")],
    "glasses_acc": [SENSOR_NAMES.index("glasses_acc")],
    "phone_all":   [SENSOR_NAMES.index(s) for s in ["phone_acc","phone_gyro","phone_grav","phone_lacc"]],
    "watch_all":   [SENSOR_NAMES.index(s) for s in ["watch_acc","watch_gyro"]],
    "only_watch":  [SENSOR_NAMES.index(s) for s in ["phone_acc","phone_gyro","phone_grav","phone_lacc","glasses_acc"]],
    "only_phone":  [SENSOR_NAMES.index(s) for s in ["watch_acc","watch_gyro","glasses_acc"]],
    "only_glasses":[SENSOR_NAMES.index(s) for s in ["phone_acc","phone_gyro","phone_grav","phone_lacc","watch_acc","watch_gyro"]],
}

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
# DDIM imputation
# ============================================================
@torch.no_grad()
def ddim_impute(model, stacked_norm, observed_mask, alpha_bar, T, ddim_steps):
    B, K, C, T_len = stacked_norm.shape
    missing_idx = (observed_mask[0] == 0).nonzero(as_tuple=True)[0].tolist()
    z = stacked_norm.clone()
    for i in missing_idx:
        z[:, i] = torch.randn_like(z[:, i])

    timesteps = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long)
    for idx in range(len(timesteps)):
        t_now  = timesteps[idx]
        t_next = timesteps[idx + 1] if idx + 1 < len(timesteps) else torch.tensor(-1)
        ab_now  = alpha_bar[t_now].to(stacked_norm.device)
        ab_next = alpha_bar[t_next].to(stacked_norm.device) if t_next >= 0 else torch.tensor(1.0)

        t_batch    = t_now.expand(B).to(stacked_norm.device)
        noise_pred = model(z, t_batch, observed_mask)

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
        stacked = stack_signals(batch, DEVICE)           # (B, K, C, T_COMMON)
        B, K, C, T_len = stacked.shape

        stacked_norm = (stacked - norm_mean[None, :, :, None]) \
                     / norm_std[None, :, :, None]

        # Start with raw signals at native lengths (no interpolation degradation)
        signals_dict = {
            name: batch[name].to(DEVICE).permute(0, 2, 1).float()
            for name in SENSOR_NAMES
        }

        if use_diffusion:
            observed_mask = torch.ones(B, K, device=DEVICE)
            for i in missing_idx:
                observed_mask[:, i] = 0.0
            imputed = ddim_impute(diff_model, stacked_norm, observed_mask,
                                  alpha_bar, diff_model_T, DDIM_STEPS)
            imputed_denorm = imputed * norm_std[None, :, :, None] \
                           + norm_mean[None, :, :, None]
            # Only replace missing sensors
            for i in missing_idx:
                name = SENSOR_NAMES[i]
                native_len = SENSOR_SPECS[name]["seq_len"]
                signals_dict[name] = F.interpolate(
                    imputed_denorm[:, i], size=native_len,
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

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    if USE_STATE:
        test_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing", normalizer)
    else:
        test_ds = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    n_classes = test_ds.n_classes
    idx_to_label = test_ds.idx_to_label   # contiguous idx → raw label int

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Classifier
    print(f"Loading classifier from {CLASSIFIER_CKPT}...")
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
        # Still need norm for baseline (just use identity)
        norm_stats = None
        # Load norm_mean / norm_std even for baseline (needed in collect_preds)
        if (DIFF_DIR / "normalization_stats.pt").exists():
            norm_stats = torch.load(DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
            norm_mean  = torch.stack([norm_stats[k]["mean"] for k in SENSOR_NAMES]).to(DEVICE)
            norm_std   = torch.stack([norm_stats[k]["std"]  for k in SENSOR_NAMES]).to(DEVICE)
        else:
            # Fallback: unit norm
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
    # STATE: standard heatmap
    # ============================================================
    if USE_STATE:
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
        ax.set_title(f"State Confusion Matrix — {scenario_str}  (acc={acc:.3f})", fontsize=13)
        plt.tight_layout()
        out_path = OUT_DIR / f"{fname_base}_confusion.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # ============================================================
    # BEHAVIORAL: heatmap + top confused pairs
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
        ax.set_title(f"Behavioral Confusion Matrix ({n_classes} classes) — {scenario_str}  (acc={acc:.3f})", fontsize=13)
        plt.tight_layout()
        out_path_hm = OUT_DIR / f"{fname_base}_confusion_heatmap.png"
        plt.savefig(out_path_hm, dpi=150)
        plt.close()
        print(f"Saved: {out_path_hm}")

        # --- top confused pairs ---
        # Extract off-diagonal counts
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
