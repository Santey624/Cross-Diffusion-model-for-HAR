# ============================================================
# Evaluate Class-Conditional Diffusion Imputation
#
# Pipeline:
#   1. Available sensors → VAE V2 encode → decode → C-LSTM-A → predicted class
#   2. Predicted class + noise → class-cond diffusion → latent → VAE V2 decode
#   3. All decoded signals (real + imputed) → C-LSTM-A → final prediction
#
# Compares:
#   - Baseline:       all sensors real
#   - Class-cond:     missing sensor via class-conditional diffusion
#   - Mean-fill:      missing sensor latent = training mean
#   - Cross-sensor:   missing sensor via V3 diffusion (previous approach)
#
# Usage:
#   python -m src.eval.evaluate_class_cond_diffusion
#   python -m src.eval.evaluate_class_cond_diffusion --state
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

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.class_conditional_diffusion import create_class_conditional_diffusion
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset, CogAgeLabeledDataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_STATE = "--state" in sys.argv

VAE_CHECKPOINT  = "checkpoints/sensor_vae_v2/best_model.pt"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

tag = "state" if USE_STATE else "behavioral"
CLASS_DIFF_DIR  = Path(f"checkpoints/class_cond_diffusion_{tag}")

USE_AUGMENTED = "--augmented" in sys.argv

if USE_STATE:
    CLASSIFIER_CKPT = (f"checkpoints/clstm_classcond_state/best_model.pt" if USE_AUGMENTED
                       else "checkpoints/clstm_state/best_model.pt")
    DATA_ROOTS      = {"state": "data/cogage/python/arrays/state"}
else:
    CLASSIFIER_CKPT = (f"checkpoints/clstm_classcond_behavioral/best_model.pt" if USE_AUGMENTED
                       else "checkpoints/clstm_behavioral/best_model.pt")
    DATA_ROOTS      = {
        "blho": "data/cogage/python/arrays/blho",
        "bbh":  "data/cogage/python/arrays/bbh",
    }

DDIM_STEPS = 50
BATCH_SIZE = 32


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
# DDIM sampling  (class-conditional)
# ============================================================
@torch.no_grad()
def ddim_sample_class(model, class_label, alpha_bar, T,
                      latent_dim, t_lat, ddim_steps=50):
    """
    Generate a latent from scratch conditioned on class_label.

    Returns: (B, D, T_lat)
    """
    B      = class_label.shape[0]
    device = class_label.device

    z = torch.randn(B, latent_dim, t_lat, device=device)
    alpha_bar = alpha_bar.to(device)

    tau = (torch.linspace(0, 1, ddim_steps + 1, device=device) ** 2
           * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noise_pred = model(z, t_batch, class_label)

        ab_now  = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]
        pred_x0 = ((z - torch.sqrt(1 - ab_now) * noise_pred)
                   / torch.sqrt(ab_now)).clamp(-5, 5)
        z = (torch.sqrt(ab_next) * pred_x0
             + torch.sqrt(1 - ab_next) * noise_pred)

    return z


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
    print(f"Imputation Eval — Class-Conditional Diffusion | C-LSTM-A {task}")
    print(f"{'='*75}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print(f"Loading dataset ({', '.join(DATA_ROOTS.keys())})...")
    if USE_STATE:
        test_ds  = CogAgeLabeledDataset(DATA_ROOTS["state"], "testing",  normalizer)
        train_ds = CogAgeLabeledDataset(DATA_ROOTS["state"], "training", normalizer)
    else:
        test_ds  = get_combined_labeled_dataset(DATA_ROOTS, "testing",  normalizer)
        train_ds = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    n_classes = test_ds.n_classes

    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    print(f"  Test: {len(test_ds)}, Classes: {n_classes}")

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
    latent_dim = cfg_vae["latent_dim"]
    t_lat      = cfg_vae["t_shared"]
    print(f"  latent_dim={latent_dim}, t_lat={t_lat}")

    # Class-Conditional Diffusion
    print(f"\nLoading Class-Conditional Diffusion from {CLASS_DIFF_DIR}...")
    if not (CLASS_DIFF_DIR / "best_model.pt").exists():
        print("  NOT FOUND. Train first:")
        print(f"  python -m src.train.train_class_conditional_diffusion{' --state' if USE_STATE else ''}")
        return
    diff_ckpt = torch.load(CLASS_DIFF_DIR / "best_model.pt", map_location=DEVICE)
    cfg_diff  = diff_ckpt["config"]
    diff_model = create_class_conditional_diffusion(
        latent_dim=cfg_diff["latent_dim"],
        t_lat=cfg_diff["t_lat"],
        n_classes=cfg_diff["n_classes"],
        d_model=cfg_diff["d_model"],
        n_blocks=cfg_diff["n_blocks"],
        emb_dim=cfg_diff["emb_dim"],
    ).to(DEVICE)
    diff_model.load_state_dict(diff_ckpt["model_state"])
    diff_model.eval()
    T_diff    = diff_ckpt["T"]
    betas     = cosine_beta_schedule(T_diff)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    norm_stats = torch.load(CLASS_DIFF_DIR / "normalization_stats.pt", map_location=DEVICE)
    print(f"  loss={diff_ckpt['loss']:.6f}, epoch={diff_ckpt['epoch']}")

    # C-LSTM-A (trained on V2 decoded signals, no missing sensors)
    print(f"\nLoading C-LSTM-A from {CLASSIFIER_CKPT}...")
    if not Path(CLASSIFIER_CKPT).exists():
        print("  NOT FOUND.")
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

    # Mean latents for mean-fill baseline
    print("\nComputing mean latents...")
    mean_latents = {k: [] for k in SENSOR_NAMES}
    with torch.no_grad():
        for batch in train_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                mean_latents[k].append(outputs[k]["mu"].cpu())
    mean_latents = {
        k: torch.cat(v).mean(0, keepdim=True).to(DEVICE)
        for k, v in mean_latents.items()
    }

    # ============================================================
    # Helper: classify from subset of sensors → predicted class
    # ============================================================
    def predict_class_from_available(sensor_data, missing_sensors):
        """
        Encode available sensors, decode, classify → (B,) predicted labels.
        """
        avail = {k: v for k, v in sensor_data.items() if k not in missing_sensors}
        outputs = vae(avail)
        # For missing sensors, use mean latent so classifier sees all 7 inputs
        latents = {k: outputs[k]["mu"] for k in avail}
        for name in missing_sensors:
            B = next(iter(latents.values())).shape[0]
            latents[name] = mean_latents[name].expand(B, -1, -1)

        decoded = {}
        for name in SENSOR_NAMES:
            sig = vae.decode_sensor(name, latents[name])  # (B, T, C)
            decoded[name] = sig.permute(0, 2, 1)           # (B, C, T)

        logits = classifier(decoded, SENSOR_NAMES)
        return logits.argmax(dim=1)   # (B,)

    # ============================================================
    # Scenarios
    # ============================================================
    scenarios = {
        "all_real":              ([], "real"),
        "phone_acc+classcond":   (["phone_acc"], "classcond"),
        "phone_acc+mean":        (["phone_acc"], "mean"),
        "watch_acc+classcond":   (["watch_acc"], "classcond"),
        "watch_acc+mean":        (["watch_acc"], "mean"),
        "glasses_acc+classcond": (["glasses_acc"], "classcond"),
        "glasses_acc+mean":      (["glasses_acc"], "mean"),
        "phone_all+classcond":   (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "classcond"),
        "phone_all+mean":        (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"], "mean"),
        "watch_all+classcond":   (["watch_acc", "watch_gyro"], "classcond"),
        "watch_all+mean":        (["watch_acc", "watch_gyro"], "mean"),
        "only_watch+classcond":  (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "classcond"),
        "only_watch+mean":       (["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"], "mean"),
        "only_phone+classcond":  (["watch_acc", "watch_gyro", "glasses_acc"], "classcond"),
        "only_phone+mean":       (["watch_acc", "watch_gyro", "glasses_acc"], "mean"),
    }

    results = {}

    for scenario_name, (missing_sensors, mode) in scenarios.items():
        all_probs  = []
        all_labels = []

        for batch in tqdm(test_loader, desc=f"{scenario_name:<28}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

                if mode == "real":
                    final_latents = latents

                elif mode == "classcond":
                    # Step 1: predict activity from available sensors
                    pred_class = predict_class_from_available(sensor_data, missing_sensors)

                    # Step 2: generate missing sensor latents conditioned on predicted class
                    final_latents = dict(latents)
                    for name in missing_sensors:
                        z_gen = ddim_sample_class(
                            diff_model, pred_class, alpha_bar, T_diff,
                            latent_dim, t_lat, DDIM_STEPS,
                        )   # (B, D, T_lat) — normalized
                        # Denormalize
                        final_latents[name] = (
                            z_gen * norm_stats[name]["std"].to(DEVICE)[None, :, None]
                            + norm_stats[name]["mean"].to(DEVICE)[None, :, None]
                        )

                else:  # mean
                    final_latents = dict(latents)
                    for name in missing_sensors:
                        final_latents[name] = mean_latents[name].expand(B, -1, -1)

                # Decode all sensors
                decoded = {}
                for name in SENSOR_NAMES:
                    sig = vae.decode_sensor(name, final_latents[name])  # (B, T, C)
                    decoded[name] = sig.permute(0, 2, 1)                 # (B, C, T)

                logits = classifier(decoded, SENSOR_NAMES)
                probs  = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # Results Table
    # ============================================================
    print(f"\n{'='*100}")
    print(f"CLASS-CONDITIONAL DIFFUSION EVAL  |  C-LSTM-A {task}")
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
        ("phone_all",  ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"]),
        ("watch_all",  ["watch_acc", "watch_gyro"]),
        ("only_watch", ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc", "glasses_acc"]),
        ("only_phone", ["watch_acc", "watch_gyro", "glasses_acc"]),
    ]

    real = results.get("all_real", {})
    fmt  = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}" if m else "    —    "

    print(f"\n{'='*100}")
    print("  COMPARISON: Class-Cond Diffusion vs Mean-Fill  (Acc / AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Baseline':>14} {'Class-Cond':>14} {'Mean-Fill':>14}   Winner")
    print("  " + "-" * 72)
    print(f"  {'all_real':<14} {fmt(real):>14}")

    for pat, _ in groups:
        c = results.get(f"{pat}+classcond")
        m = results.get(f"{pat}+mean")
        candidates = {k: val["acc"] for k, val in [("ClassCond", c), ("Mean", m)] if val}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        delta  = ""
        if c and m:
            delta = f"  Δ={c['acc'] - m['acc']:+.3f}"
        print(f"  {pat:<14} {fmt(real):>14} {fmt(c):>14} {fmt(m):>14}   → {winner}{delta}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
