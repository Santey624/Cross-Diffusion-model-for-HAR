# ============================================================
# Evaluate Latent Classifiers (MLP / Transformer / C-LSTM-A on latents)
# Same scenarios as C-LSTM-A decoded eval: diff / mean / zero imputation
#
# Flags:
#   --model clstm|transformer|mlp   (default: clstm)
#   --state                         State activities (6 classes)
#   --robust                        Load robust checkpoint
#   --v1                            Use V1 VAE + V1 diffusion (D=8, T=32)
# ============================================================

import sys
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_conditional_diffusion_v3 import create_sensor_diffusion_v3
from src.models.activity_classifier import create_activity_classifier
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CLI FLAGS
# ============================================================
MODEL_TYPE = "clstm"
for i, a in enumerate(sys.argv):
    if a == "--model" and i + 1 < len(sys.argv):
        MODEL_TYPE = sys.argv[i + 1]

ROBUST    = "--robust" in sys.argv
USE_STATE = "--state"  in sys.argv
USE_V1    = "--v1"     in sys.argv

# ============================================================
# PATHS
# ============================================================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
DDIM_STEPS      = 50
BATCH_SIZE      = 32

vae_tag = "v1" if USE_V1 else "v2"
task    = "state" if USE_STATE else "behavioral"
rob_tag = "_robust" if ROBUST else ""

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt" if USE_V1 \
                 else "checkpoints/sensor_vae_v2/best_model.pt"

# V1 diffusion for V1 latents, V2 diffusion for V2 latents
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion_v3") if USE_V1 \
                else Path("checkpoints/sensor_diffusion_v3_v2")

if MODEL_TYPE == "clstm":
    CLASSIFIER_CKPT = f"checkpoints/clstm_latents_{task}{rob_tag}_{vae_tag}/best_model.pt"
else:
    CLASSIFIER_CKPT = f"checkpoints/latent_{MODEL_TYPE}_{task}{rob_tag}_{vae_tag}/best_model.pt"

DATA_ROOTS = {"state": "data/cogage/python/arrays/state"} if USE_STATE else {
    "blho": "data/cogage/python/arrays/blho",
    "bbh":  "data/cogage/python/arrays/bbh",
}


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t   = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    ab    = f_t / f_t[0]
    betas = torch.clamp(1 - ab[1:] / ab[:-1], 1e-6, 0.999)
    return betas.float()

def make_schedule(T, schedule_type):
    betas = cosine_beta_schedule(T) if schedule_type == "cosine" \
            else torch.linspace(1e-4, 0.02, T)
    return {"alpha_bar": torch.cumprod(1.0 - betas, dim=0)}


# ============================================================
# DDIM SAMPLING (latent space)
# ============================================================
@torch.no_grad()
def ddim_sample(model, stacked, observed_mask, alpha_bar, T, steps=50):
    B, K, D, Tl = stacked.shape
    device  = stacked.device
    missing = 1.0 - observed_mask
    alpha_bar = alpha_bar.to(device)

    z   = observed_mask[:, :, None, None] * stacked + \
          missing[:, :, None, None] * torch.randn_like(stacked)
    tau = (torch.linspace(0, 1, steps + 1, device=device) ** 2 * (T - 1)).long().flip(0)

    for i in range(len(tau) - 1):
        t_now, t_next = tau[i], tau[i + 1]
        t_b = torch.full((B,), t_now, device=device, dtype=torch.long)
        noisy = observed_mask[:, :, None, None] * stacked + missing[:, :, None, None] * z
        noise_pred = model(noisy, t_b, observed_mask)
        ab_now, ab_next = alpha_bar[t_now], alpha_bar[t_next]
        pred_x0 = ((z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)).clamp(-5, 5)
        z_new   = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
        z = observed_mask[:, :, None, None] * stacked + missing[:, :, None, None] * z_new

    return z


# ============================================================
# METRICS
# ============================================================
def compute_metrics(labels, probs, n_classes):
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    af1 = f1_score(labels, preds, average="macro", zero_division=0)
    oh  = label_binarize(labels, classes=list(range(n_classes)))
    try:    map_s = average_precision_score(oh, probs, average="macro")
    except: map_s = float("nan")
    try:    auc_s = roc_auc_score(oh, probs, average="macro", multi_class="ovr")
    except: auc_s = float("nan")
    return {"acc": acc, "af1": af1, "map": map_s, "auc": auc_s}


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*70}")
    print(f"Latent Classifier Eval | Model: {MODEL_TYPE.upper()} | "
          f"Task: {task} | VAE: {vae_tag} | Robust: {ROBUST}")
    print(f"{'='*70}\n")

    normalizer  = SensorNormalizer.load(NORMALIZER_PATH)
    test_ds     = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)
    n_classes   = test_ds.n_classes
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_ds)}, Classes: {n_classes}")

    # VAE
    ckpt_vae = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    if USE_V1:
        latent_dim, t_shared = 8, 32
        vae = SensorMultiModalVAE().to(DEVICE)
    else:
        cfg = ckpt_vae["config"]
        latent_dim, t_shared = cfg["latent_dim"], cfg["t_shared"]
        vae = SensorMultiModalVAE(latent_dim=latent_dim, t_shared=t_shared).to(DEVICE)
    vae.load_state_dict(ckpt_vae["model_state"])
    vae.eval()
    print(f"VAE loaded (latent_dim={latent_dim}, t_shared={t_shared})")

    # Diffusion
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    dcfg      = diff_ckpt["config"]
    diff_latent_dim = diff_ckpt["model_state"]["sensor_embeddings.weight"].shape[1]
    diffusion = create_sensor_diffusion_v3(
        d_model=dcfg["d_model"], num_heads=dcfg["num_heads"],
        num_blocks=dcfg["num_blocks"], dropout=0.0,
        latent_dim=diff_latent_dim,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()
    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched      = make_schedule(diff_ckpt["T"], diff_ckpt["schedule"])
    print(f"Diffusion loaded (loss={diff_ckpt.get('loss', '?'):.4f})")

    # Classifier
    if not Path(CLASSIFIER_CKPT).exists():
        print(f"\nCheckpoint not found: {CLASSIFIER_CKPT}")
        print("Train it first!")
        return

    clf_ckpt = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
    print(f"\nLoading {MODEL_TYPE.upper()} from {CLASSIFIER_CKPT}...")
    print(f"  Best acc: {clf_ckpt.get('acc', '?'):.4f}")

    if MODEL_TYPE == "clstm":
        # Infer architecture from checkpoint weights
        w = clf_ckpt["model_state"]
        cnn_channels = w["sensor_cnns.0.0.weight"].shape[0]
        lstm_hidden  = w["lstm.weight_hh_l0"].shape[1]
        d_attn       = w["head.0.weight"].shape[0]
        classifier = create_clstm_classifier(
            n_sensors=len(SENSOR_NAMES),
            n_classes=n_classes,
            in_channels=latent_dim,
            cnn_channels=cnn_channels, lstm_hidden=lstm_hidden,
            d_attn=d_attn, n_heads=4, n_layers=2,
            pool_size=16, dropout=0.0,
        ).to(DEVICE)
        print(f"  CLSTM arch: cnn={cnn_channels}, lstm={lstm_hidden}, d_attn={d_attn}")
    else:
        clf_cfg = {}
        if MODEL_TYPE == "transformer":
            # Infer d_model from checkpoint
            w = clf_ckpt["model_state"]
            d_model = w["transformer.layers.0.self_attn.in_proj_weight"].shape[1]
            clf_cfg = {"d_model": d_model, "n_heads": 4, "n_layers": 2, "dropout": 0.0}
            print(f"  Transformer arch: d_model={d_model}")
        else:  # mlp
            clf_cfg = {"hidden_dims": [512, 256, 128], "dropout": 0.0}
        classifier = create_activity_classifier(
            model_type=MODEL_TYPE,
            n_sensors=len(SENSOR_NAMES),
            latent_dim=latent_dim,
            t_shared=t_shared,
            n_classes=n_classes,
            **clf_cfg,
        ).to(DEVICE)

    classifier.load_state_dict(clf_ckpt["model_state"])
    classifier.eval()

    # Mean latents for mean-fill baseline
    print("\nComputing mean latents...")
    sums  = {k: 0.0 for k in SENSOR_NAMES}
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            for k in SENSOR_NAMES:
                sums[k] = sums[k] + outputs[k]["mu"].mean(0, keepdim=True)
            count += 1
    mean_latents = {k: (sums[k] / count).to(DEVICE) for k in SENSOR_NAMES}

    # ============================================================
    # SCENARIOS
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

        for batch in tqdm(test_loader, desc=scenario_name, leave=False):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels      = batch["label"].to(DEVICE)
            B           = labels.size(0)

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

                imputed = ddim_sample(diffusion, stacked, observed,
                                      sched["alpha_bar"], diff_ckpt["T"], DDIM_STEPS)
                final_latents = {
                    n: imputed[:, i] * norm_stats[n]["std"].to(DEVICE)
                                    + norm_stats[n]["mean"].to(DEVICE)
                    for i, n in enumerate(SENSOR_NAMES)
                }

            elif mode == "mean":
                final_latents = dict(latents)
                for n in missing_sensors:
                    final_latents[n] = mean_latents[n].expand(B, -1, -1)

            else:  # zero
                final_latents = dict(latents)
                for n in missing_sensors:
                    final_latents[n] = torch.zeros_like(latents[n])

            with torch.no_grad():
                logits = classifier(final_latents, SENSOR_NAMES)
                probs  = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        results[scenario_name] = compute_metrics(all_labels, all_probs, n_classes)

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    model_label = f"{MODEL_TYPE.upper()} (latent/{vae_tag})"
    print(f"\n{'='*100}")
    print(f"{model_label} — {task.upper()} | Diffusion: {vae_tag}")
    print(f"{'='*100}")
    print(f"  {'Scenario':<22} {'Acc':>7} {'AF1':>7} {'MAP':>7} {'AUC':>7}")
    print("  " + "-" * 56)
    for name, m in results.items():
        print(f"  {name:<22} {m['acc']:>7.4f} {m['af1']:>7.4f} {m['map']:>7.4f} {m['auc']:>7.4f}")

    pattern_groups = [
        "phone_acc", "watch_acc", "glasses_acc",
        "phone_all", "watch_all", "only_watch", "only_phone",
    ]
    real = results.get("all_real", {})
    fmt  = lambda m: f"{m['acc']:.3f}/{m['af1']:.3f}" if m else "      —     "

    print(f"\n{'='*100}")
    print("  COMPARISON: Diff vs Mean vs Zero  (Acc | AF1)")
    print(f"{'='*100}")
    print(f"  {'Pattern':<14} {'Present':>14} {'Diff':>14} {'Mean':>14} {'Zero':>14}   Winner")
    print("  " + "-" * 86)
    print(f"  {'all_real':<14} {fmt(real):>14}")
    for pat in pattern_groups:
        d = results.get(f"{pat}+diff")
        m = results.get(f"{pat}+mean")
        z = results.get(f"{pat}+zero")
        candidates = {k: v["acc"] for k, v in [("Diff", d), ("Mean", m), ("Zero", z)] if v}
        winner = max(candidates, key=candidates.get) if candidates else "—"
        print(f"  {pat:<14} {fmt(real):>14} {fmt(d):>14} {fmt(m):>14} {fmt(z):>14}   → {winner}")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
