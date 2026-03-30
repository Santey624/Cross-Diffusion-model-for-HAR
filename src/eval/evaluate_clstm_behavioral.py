# ============================================================
# Evaluate C-LSTM-A Classifier — Behavioral Activities
#
# Option 2 pipeline:
#   latent (imputed/real) -> VAE decode -> C-LSTM-A
#
# Metrics: AF1 (macro F1), Acc, MAP (mean avg precision), AUC
#
# Flags:
#   --v3      Use V3 diffusion checkpoint
#   --robust  Load robust C-LSTM-A checkpoint
# ============================================================

import sys
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    average_precision_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion_v2 import create_sensor_diffusion_v2
from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG  (overridden by CLI flags below)
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE_CHECKPOINT    = "checkpoints/sensor_vae_combined_best.pt"
NORMALIZER_PATH   = "data/sensor_normalizer_combined.npz"
DIFFUSION_DIR     = Path("checkpoints/sensor_diffusion_v2_pretrain")
CLASSIFIER_CKPT   = "checkpoints/clstm_behavioral/best_model.pt"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh":  "data/cogage/python/arrays/bbh",
}

DDIM_STEPS  = 50
BATCH_SIZE  = 32


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float()


def make_schedule(T, schedule_type):
    betas = cosine_beta_schedule(T) if schedule_type == "cosine" \
            else torch.linspace(1e-4, 0.02, T)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return {"alpha_bar": alpha_bar}


# ============================================================
# DDIM SAMPLING
# ============================================================
@torch.no_grad()
def ddim_sample(model, stacked_latents, observed_mask, alpha_bar, T, ddim_steps=50):
    B, K, D, T_len = stacked_latents.shape
    device = stacked_latents.device
    missing = 1.0 - observed_mask

    z = observed_mask[:, :, None, None] * stacked_latents + \
        missing[:, :, None, None] * torch.randn_like(stacked_latents)

    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2 * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy = observed_mask[:, :, None, None] * stacked_latents + \
                missing[:, :, None, None] * z
        noise_pred = model(noisy, t_batch, observed_mask)

        ab_now, ab_next = alpha_bar[t_now], alpha_bar[t_next]
        pred_x0 = ((z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)).clamp(-5, 5)
        z_new = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
        z = observed_mask[:, :, None, None] * stacked_latents + missing[:, :, None, None] * z_new

    return z


# ============================================================
# DECODE LATENTS -> SIGNALS (B, C, T)
# ============================================================
def decode_to_signals(vae, latents_dict):
    decoded = {}
    for name in SENSOR_NAMES:
        z = latents_dict[name]
        sig = vae.decode_sensor(name, z)       # (B, T, C)
        decoded[name] = sig.permute(0, 2, 1)   # (B, C, T)
    return decoded


# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(all_labels, all_probs, n_classes):
    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)    # (N, n_classes)
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
    USE_V3   = "--v3" in sys.argv
    ROBUST   = "--robust" in sys.argv
    global DIFFUSION_DIR, CLASSIFIER_CKPT

    if USE_V3:
        DIFFUSION_DIR   = Path("checkpoints/sensor_diffusion_v3")
    if ROBUST:
        CLASSIFIER_CKPT = CLASSIFIER_CKPT.replace(
            "clstm_behavioral", "clstm_behavioral_robust"
        )

    diff_version = "v3" if USE_V3 else "v2"

    print(f"\n{'='*70}")
    print(f"C-LSTM-A BEHAVIORAL EVAL | Diffusion: {diff_version} | Robust: {ROBUST}")
    print(f"{'='*70}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print("Loading test dataset (BLHO + BBH)...")
    test_ds = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    n_classes = test_ds.n_classes
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_ds)}, Classes: {n_classes}")

    # VAE
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE)["model_state"])
    vae.eval()

    # Diffusion
    print(f"Loading Diffusion {diff_version}...")
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T_diff = diff_ckpt["T"]
    cfg    = diff_ckpt["config"]
    if USE_V3:
        diffusion = create_sensor_diffusion_v3(
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"], dropout=0.0,
        ).to(DEVICE)
    else:
        diffusion = create_sensor_diffusion_v2(
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"], dropout=0.0,
        ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()

    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T_diff, diff_ckpt["schedule"])

    # C-LSTM-A Classifier
    print(f"\nLoading C-LSTM-A from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print(f"Checkpoint not found: {CLASSIFIER_CKPT}")
        print("Please train it first with: python -m src.train.train_clstm_behavioral")
        return
    clf_ckpt = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
    cfg_clf  = clf_ckpt["config"]
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
    print(f"C-LSTM-A loaded (best acc: {clf_ckpt.get('accuracy', '?'):.4f})")

    # Mean latents for mean-fill baseline
    print("\nComputing mean latents...")
    mean_latents = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in test_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                mean_latents[k].append(outputs[k]["mu"].cpu())
    mean_latents = {k: torch.cat(v).mean(0, keepdim=True).to(DEVICE) for k, v in mean_latents.items()}

    # ============================================================
    # Evaluation scenarios
    # ============================================================
    scenarios = {
        "all_real":         ([], "real"),
        "phone_acc+diff":   (["phone_acc"], "diff"),
        "phone_acc+mean":   (["phone_acc"], "mean"),
        "phone_acc+zero":   (["phone_acc"], "zero"),
        "watch_acc+diff":   (["watch_acc"], "diff"),
        "watch_acc+mean":   (["watch_acc"], "mean"),
        "watch_acc+zero":   (["watch_acc"], "zero"),
        "glasses_acc+diff": (["glasses_acc"], "diff"),
        "glasses_acc+mean": (["glasses_acc"], "mean"),
        "glasses_acc+zero": (["glasses_acc"], "zero"),
        "phone_all+diff":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "diff"),
        "phone_all+mean":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "mean"),
        "phone_all+zero":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "zero"),
        "watch_all+diff":   (["watch_acc", "watch_gyro"], "diff"),
        "watch_all+mean":   (["watch_acc", "watch_gyro"], "mean"),
        "watch_all+zero":   (["watch_acc", "watch_gyro"], "zero"),
        "only_watch+diff":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "diff"),
        "only_watch+mean":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "mean"),
        "only_watch+zero":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "zero"),
        "only_phone+diff":  (["watch_acc", "watch_gyro", "glasses_acc"], "diff"),
        "only_phone+mean":  (["watch_acc", "watch_gyro", "glasses_acc"], "mean"),
        "only_phone+zero":  (["watch_acc", "watch_gyro", "glasses_acc"], "zero"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []

        for batch in tqdm(test_loader, desc=f"{scenario_name}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            if mode == "real":
                final_latents = latents

            elif mode == "diff":
                latents_norm = {
                    name: (latents[name] - norm_stats[name]["mean"].to(DEVICE))
                          / norm_stats[name]["std"].to(DEVICE)
                    for name in SENSOR_NAMES
                }
                stacked = torch.stack([latents_norm[n] for n in SENSOR_NAMES], dim=1)
                observed = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for s in missing_sensors:
                    observed[:, SENSOR_NAMES.index(s)] = 0.0

                imputed = ddim_sample(diffusion, stacked, observed,
                                      sched["alpha_bar"], T_diff, DDIM_STEPS)
                final_latents = {
                    name: imputed[:, i] * norm_stats[name]["std"].to(DEVICE)
                                       + norm_stats[name]["mean"].to(DEVICE)
                    for i, name in enumerate(SENSOR_NAMES)
                }

            elif mode == "mean":
                final_latents = dict(latents)
                for name in missing_sensors:
                    final_latents[name] = mean_latents[name].expand(B, -1, -1)

            else:  # zero
                final_latents = dict(latents)
                for name in missing_sensors:
                    final_latents[name] = torch.zeros_like(latents[name])

            with torch.no_grad():
                decoded  = decode_to_signals(vae, final_latents)
                logits   = classifier(decoded, SENSOR_NAMES)
                probs    = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_labels, all_probs, n_classes)
        results[scenario_name] = metrics

    # ============================================================
    # Print Results Table
    # ============================================================
    print(f"\n{'='*100}")
    print(f"C-LSTM-A — BEHAVIORAL ACTIVITIES  |  Diffusion: {diff_version}")
    print(f"{'='*100}")
    print(f"  {'Scenario':<22} {'Acc':>7} {'AF1':>7} {'MAP':>7} {'AUC':>7}")
    print("  " + "-" * 56)

    for name, m in results.items():
        print(f"  {name:<22} {m['acc']:>7.4f} {m['af1']:>7.4f} "
              f"{m['map']:>7.4f} {m['auc']:>7.4f}")

    # Grouped comparison table (Diff vs Mean vs Zero)
    pattern_groups = [
        ("phone_acc",   ["phone_acc"]),
        ("watch_acc",   ["watch_acc"]),
        ("glasses_acc", ["glasses_acc"]),
        ("phone_all",   ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"]),
        ("watch_all",   ["watch_acc", "watch_gyro"]),
        ("only_watch",  ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"]),
        ("only_phone",  ["watch_acc", "watch_gyro", "glasses_acc"]),
    ]

    real = results.get("all_real", {})

    print(f"\n{'='*100}")
    print("  COMPARISON: Diff vs Mean vs Zero  (Acc | AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Present':>14} {'Diff':>14} {'Mean':>14} {'Zero':>14}   Winner")
    print("  " + "-" * 86)
    fmt = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}" if m else "    —    "
    print(f"  {'all_real':<14} {fmt(real):>14}")
    for pat, _ in pattern_groups:
        d = results.get(f"{pat}+diff")
        m = results.get(f"{pat}+mean")
        z = results.get(f"{pat}+zero")
        candidates = {k: v["acc"] for k, v in [("Diff", d), ("Mean", m), ("Zero", z)] if v}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14} {fmt(z):>14}   → {winner}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
