import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import CogAgeContractDataset
from src.data.modality_registry import MODALITIES
from src.models.vae_har import VAEHAR


@torch.no_grad()
def evaluate_vae_fold(test_subject: int, device_filter=["phone"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = f"processed/behavior_blho/splits_loso/fold_test_subject_{test_subject}.json"
    norm_path = f"processed/behavior_blho/norm_loso/norm_fold_test_subject_{test_subject}.json"
    samples_path = "processed/behavior_blho/samples.pt"

    with open(split_path, "r") as f:
        split = json.load(f)

    test_ds = CogAgeContractDataset(
        samples_path=samples_path,
        indices=split["test_indices"],
        norm_path=norm_path,
        device_filter=device_filter,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    modality_names = [m for m, s in MODALITIES.items() if s.device in device_filter]
    num_classes = max(s["label"] for s in torch.load(samples_path)) + 1

    # Load trained model
    model = VAEHAR(modality_names, num_classes).to(device)
    ckpt = Path(f"runs/vae_blho/fold_test_subject_{test_subject}/best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    for batch in test_loader:
        modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
        mask = {k: v.to(device) for k, v in batch["mask"].items()}
        labels = batch["label"].to(device)

        logits, _, _, _, _ = model(modalities, mask)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def main():
    results = []
    for subj in [1, 2, 3, 4]:
        metrics = evaluate_vae_fold(subj)
        print(f"[VAE][Fold {subj}] acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f}")
        results.append(metrics["macro_f1"])

    print("VAE LOSO Macro-F1:", results)
    print("Mean:", np.mean(results))


if __name__ == "__main__":
    main()
