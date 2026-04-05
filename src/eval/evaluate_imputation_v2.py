# ============================================================
# Evaluate Imputation Quality — V2 VAE + V3 Diffusion on V2 Latents
#
# Pipeline:
#   sensor signal → VAE V2 encode → latent → diffusion impute → VAE V2 decode → C-LSTM-A
#
# Compares:
#   - Baseline:   all sensors real
#   - Diffusion:  missing sensor imputed via diffusion in latent space
#   - Mean-fill:  missing sensor latent replaced by training mean
#   - Zero-fill:  missing sensor latent set to zeros
#
# Scenarios:
#   single sensor missing, full device missing, only one device available
#
# Usage:
#   python -m src.eval.evaluate_imputation_v2
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

import sys

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset, CogAgeLabeledDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STATE = "--state" in sys.argv

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"
DIFFUSION_DIR   = Path("checkpoints/sensor_diffusion_v3_v2")
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

if USE_STATE:
    CLASSIFIER_CKPT = "checkpoints/clstm_state/best_model.pt"
    DATA_ROOTS      = {"state": "data/cogage/python/arrays/state"}
else:
    CLASSIFIER_CKPT = "checkpoints/clstm_behavioral/best_model.pt"
    DATA_ROOTS      = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

DDIM_STEPS = 50
BATCH_SIZE = 32


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

    z = (observed_mask[:, :, None, None] * stacked_latents
         + missing[:, :, None, None] * torch.randn_like(stacked_latents))

    alpha_bar = alpha_bar.to(device)
    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy = (observed_mask[:, :, None, None] * stacked_latents
                 + missing[:, :, None, None] * z)
        noise_pred = model(noisy, t_batch, observed_mask)

        ab_now  = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]
        pred_x0 = ((z - torch.sqrt(1 - ab_now) * noise_pred)
                   / torch.sqrt(ab_now)).clamp(-5, 5)
        z_new = (torch.sqrt(ab_next) * pred_x0
                 + torch.sqrt(1 - ab_next) * noise_pred)
        z = (observed_mask[:, :, None, None] * stacked_latents
             + missing[:, :, None, None] * z_new)

    return z


# ============================================================
# DECODE LATENTS → SIGNALS (B, C, T)
# ============================================================
def decode_to_signals(vae, latents_dict):
    decoded = {}
    for name in SENSOR_NAMES:
        sig = vae.decode_sensor(name, latents_dict[name])  # (B, T, C)
        decoded[name] = sig.permute(0, 2, 1)               # (B, C, T)
    return decoded


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
    print(f"Imputation Eval — VAE V2 + Diffusion V3 | C-LSTM-A {task}")
    print(f"{'='*70}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print(f"Loading test dataset ({', '.join(DATA_ROOTS.keys())})...")
    if USE_STATE:
        test_ds   = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing", normalizer)
        n_classes = test_ds.n_classes
    else:
        test_ds   = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
        n_classes = test_ds.n_classes
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    print(f"  Test samples: {len(test_ds)}, Classes: {n_classes}")

    # VAE V2
    print("\nLoading VAE V2...")
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    cfg_vae  = vae_ckpt["config"]
    vae = SensorMultiModalVAE(
        latent_dim=cfg_vae["latent_dim"],
        t_shared=cfg_vae["t_shared"],
    ).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    print(f"  latent_dim={cfg_vae['latent_dim']}, t_shared={cfg_vae['t_shared']}, "
          f"epoch={vae_ckpt['epoch']}, recon={vae_ckpt['test_recon']:.6f}")

    # Diffusion V3 on V2 latents
    print("\nLoading Diffusion V3 (V2 latents)...")
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    cfg_diff  = diff_ckpt["config"]
    diffusion = create_sensor_diffusion_v3(
        n_sensors=len(SENSOR_NAMES),
        latent_dim=cfg_diff.get("latent_dim", cfg_vae["latent_dim"]),
        d_model=cfg_diff["d_model"],
        num_heads=cfg_diff["num_heads"],
        num_blocks=cfg_diff["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()
    T_diff = diff_ckpt["T"]
    sched  = make_schedule(T_diff, diff_ckpt["schedule"])
    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    print(f"  T={T_diff}, phase={diff_ckpt['phase']}, loss={diff_ckpt['loss']:.6f}")

    # C-LSTM-A Classifier
    print(f"\nLoading C-LSTM-A from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print(f"  NOT FOUND. Train first with:")
        print(f"  python -m src.train.train_clstm_behavioral")
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

    # Mean latents for mean-fill baseline (computed from test set)
    print("\nComputing mean latents...")
    mean_latents = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in test_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                mean_latents[k].append(outputs[k]["mu"].cpu())
    mean_latents = {
        k: torch.cat(v).mean(0, keepdim=True).to(DEVICE)
        for k, v in mean_latents.items()
    }

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = {
        # Baseline
        "all_real":         ([], "real"),
        # Single sensor missing
        "phone_acc+diff":   (["phone_acc"], "diff"),
        "phone_acc+mean":   (["phone_acc"], "mean"),
        "watch_acc+diff":   (["watch_acc"], "diff"),
        "watch_acc+mean":   (["watch_acc"], "mean"),
        "glasses_acc+diff": (["glasses_acc"], "diff"),
        "glasses_acc+mean": (["glasses_acc"], "mean"),
        # Full device missing
        "phone_all+diff":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "diff"),
        "phone_all+mean":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "mean"),
        "watch_all+diff":   (["watch_acc", "watch_gyro"], "diff"),
        "watch_all+mean":   (["watch_acc", "watch_gyro"], "mean"),
        # Only one device available
        "only_watch+diff":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "diff"),
        "only_watch+mean":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "mean"),
        "only_phone+diff":  (["watch_acc", "watch_gyro", "glasses_acc"], "diff"),
        "only_phone+mean":  (["watch_acc", "watch_gyro", "glasses_acc"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []

        for batch in tqdm(test_loader, desc=f"{scenario_name:<22}", leave=False):
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
                    n: (latents[n] - norm_stats[n]["mean"].to(DEVICE))
                       / norm_stats[n]["std"].to(DEVICE)
                    for n in SENSOR_NAMES
                }
                stacked  = torch.stack([latents_norm[n] for n in SENSOR_NAMES], dim=1)
                observed = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for s in missing_sensors:
                    observed[:, SENSOR_NAMES.index(s)] = 0.0

                imputed = ddim_sample(
                    diffusion, stacked, observed,
                    sched["alpha_bar"], T_diff, DDIM_STEPS,
                )
                final_latents = {
                    n: imputed[:, i] * norm_stats[n]["std"].to(DEVICE)
                                    + norm_stats[n]["mean"].to(DEVICE)
                    for i, n in enumerate(SENSOR_NAMES)
                }

            else:  # mean
                final_latents = dict(latents)
                for name in missing_sensors:
                    final_latents[name] = mean_latents[name].expand(B, -1, -1)

            with torch.no_grad():
                decoded = decode_to_signals(vae, final_latents)
                logits  = classifier(decoded, SENSOR_NAMES)
                probs   = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # Results Table
    # ============================================================
    print(f"\n{'='*100}")
    print(f"IMPUTATION EVAL — VAE V2 + Diffusion V3  |  C-LSTM-A {task}")
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
    print("  COMPARISON: Diffusion vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'Diffusion':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        d = results.get(f"{pat}+diff")
        m = results.get(f"{pat}+mean")
        candidates = {k: v["acc"] for k, v in [("Diff", d), ("Mean", m)] if v}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = ""
        if d and m:
            diff_acc = d["acc"] - m["acc"]
            delta = f"  Δ={diff_acc:+.3f}"
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
