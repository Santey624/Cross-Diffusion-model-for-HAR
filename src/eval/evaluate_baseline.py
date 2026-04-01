# ============================================================
# Baseline Evaluation — All Sensors Present (no imputation)
# Evaluates MLP, Transformer, C-LSTM-A on latents
#
# Flags:
#   --model clstm|transformer|mlp   (default: clstm)
#   --state                         State activities (6 classes)
#   --v1                            Use V1 VAE (D=8, T=32)
# ============================================================

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.activity_classifier import create_activity_classifier
from src.models.clstm_classifier import create_clstm_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# FLAGS
# ============================================================
MODEL_TYPE = "clstm"
for i, a in enumerate(sys.argv):
    if a == "--model" and i + 1 < len(sys.argv):
        MODEL_TYPE = sys.argv[i + 1]

USE_STATE = "--state" in sys.argv
USE_V1    = "--v1"    in sys.argv

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"
BATCH_SIZE      = 64

vae_tag = "v1" if USE_V1 else "v2"
task    = "state" if USE_STATE else "behavioral"

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt" if USE_V1 \
                 else "checkpoints/sensor_vae_v2/best_model.pt"

if MODEL_TYPE == "clstm":
    CKPT = f"checkpoints/clstm_latents_{task}_{vae_tag}/best_model.pt"
else:
    CKPT = f"checkpoints/latent_{MODEL_TYPE}_{task}_{vae_tag}/best_model.pt"

DATA_ROOTS = {"state": "data/cogage/python/arrays/state"} if USE_STATE else {
    "blho": "data/cogage/python/arrays/blho",
    "bbh":  "data/cogage/python/arrays/bbh",
}


def main():
    print(f"\n{'='*60}")
    print(f"Baseline Eval | {MODEL_TYPE.upper()} | {task} | VAE {vae_tag}")
    print(f"{'='*60}\n")

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

    # Classifier
    if not Path(CKPT).exists():
        print(f"Checkpoint not found: {CKPT}")
        return

    clf_ckpt = torch.load(CKPT, map_location=DEVICE)
    print(f"Loading {MODEL_TYPE.upper()} from {CKPT}")
    print(f"  Best train acc: {clf_ckpt.get('acc', '?'):.4f}")

    if MODEL_TYPE == "clstm":
        classifier = create_clstm_classifier(
            n_sensors=len(SENSOR_NAMES), n_classes=n_classes,
            in_channels=latent_dim, cnn_channels=64, lstm_hidden=64,
            d_attn=128, n_heads=4, n_layers=2, pool_size=16, dropout=0.0,
        ).to(DEVICE)
    elif MODEL_TYPE == "transformer":
        classifier = create_activity_classifier(
            model_type="transformer", n_sensors=len(SENSOR_NAMES),
            latent_dim=latent_dim, t_shared=t_shared, n_classes=n_classes,
            d_model=128, n_heads=4, n_layers=2, dropout=0.0,
        ).to(DEVICE)
    else:  # mlp
        classifier = create_activity_classifier(
            model_type="mlp", n_sensors=len(SENSOR_NAMES),
            latent_dim=latent_dim, t_shared=t_shared, n_classes=n_classes,
            hidden_dims=[512, 256, 128], dropout=0.0,
        ).to(DEVICE)

    classifier.load_state_dict(clf_ckpt["model_state"])
    classifier.eval()

    # Eval
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels      = batch["label"].to(DEVICE)
            outputs     = vae(sensor_data)
            latents     = {k: outputs[k]["mu"] for k in SENSOR_NAMES}
            probs       = F.softmax(classifier(latents, SENSOR_NAMES), dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds_arr  = probs_arr.argmax(axis=1)
    oh         = label_binarize(labels_arr, classes=list(range(n_classes)))

    acc = accuracy_score(labels_arr, preds_arr)
    af1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    try:    map_s = average_precision_score(oh, probs_arr, average="macro")
    except: map_s = float("nan")
    try:    auc_s = roc_auc_score(oh, probs_arr, average="macro", multi_class="ovr")
    except: auc_s = float("nan")

    print(f"\n{'='*60}")
    print(f"  ALL SENSORS PRESENT — {task.upper()} ({n_classes} classes)")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {af1:.4f}")
    print(f"  MAP      : {map_s:.4f}")
    print(f"  AUC      : {auc_s:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
