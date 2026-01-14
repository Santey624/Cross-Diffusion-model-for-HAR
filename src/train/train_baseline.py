# src/train/train_baseline.py

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import CogAgeContractDataset
from src.data.modality_registry import MODALITIES
from src.models.baseline_har import BaselineHAR
from src.eval.metrics import compute_metrics, compute_confusion


def _to_device_batch(batch: Dict, device: torch.device) -> Dict:
    # batch["modalities"] is dict: m -> Tensor[B, T, C]
    modalities = {m: x.to(device) for m, x in batch["modalities"].items()}
    # batch["mask"] is dict: m -> Tensor[B]
    mask = {m: x.to(device) for m, x in batch["mask"].items()}
    labels = batch["label"].to(device)
    return {"modalities": modalities, "mask": mask, "label": labels}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict, List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        b = _to_device_batch(batch, device)
        logits = model(b["modalities"], b["mask"])
        pred = torch.argmax(logits, dim=1)
        y_true.extend(b["label"].cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return compute_metrics(y_true, y_pred), y_true, y_pred


def train_one_fold(
    fold_subject: int,
    num_classes: int,
    device_filter: List[str] = ["phone"],
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    d_per_mod: int = 128,
    dropout: float = 0.3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_path = "processed/behavior_blho/samples.pt"
    split_path = Path(f"processed/behavior_blho/splits_loso/fold_test_subject_{fold_subject}.json")
    norm_path = Path(f"processed/behavior_blho/norm_loso/norm_fold_test_subject_{fold_subject}.json")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_ds = CogAgeContractDataset(
        samples_path=samples_path,
        indices=split["train_indices"],
        norm_path=str(norm_path),
        device_filter=device_filter,
    )
    test_ds = CogAgeContractDataset(
        samples_path=samples_path,
        indices=split["test_indices"],
        norm_path=str(norm_path),
        device_filter=device_filter,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # modality names actually used for modeling (e.g., phone only)
    modality_names = [m for m, spec in MODALITIES.items() if spec.device in device_filter]

    model = BaselineHAR(
        modality_names=modality_names,
        num_classes=num_classes,
        d_per_mod=d_per_mod,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    run_dir = Path(f"runs/baseline_blho/fold_test_subject_{fold_subject}")
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            b = _to_device_batch(batch, device)
            optim.zero_grad(set_to_none=True)
            logits = model(b["modalities"], b["mask"])
            loss = criterion(logits, b["label"])
            loss.backward()
            optim.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        metrics, y_true, y_pred = evaluate(model, test_loader, device)

        print(
            f"[Fold test={fold_subject}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_acc={metrics['accuracy']:.4f} test_macroF1={metrics['macro_f1']:.4f}"
        )

        # save best
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            with open(run_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            cm = compute_confusion(y_true, y_pred).tolist()
            with open(run_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
                json.dump(cm, f)

    return best_f1


def infer_num_classes() -> int:
    # Quick scan labels from samples.pt
    samples = torch.load("processed/behavior_blho/samples.pt")
    labels = [int(s["label"]) for s in samples]
    return max(labels) + 1


def main():
    num_classes = infer_num_classes()
    print("Inferred num_classes:", num_classes)

    # smartphone-only baseline
    fold_f1s = []
    for test_subj in [1, 2, 3, 4]:
        f1 = train_one_fold(
            fold_subject=test_subj,
            num_classes=num_classes,
            device_filter=["phone"],
            epochs=25,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,
            d_per_mod=128,
            dropout=0.3,
        )
        fold_f1s.append(f1)

    mean_f1 = sum(fold_f1s) / len(fold_f1s)
    print("LOSO macro-F1 per fold:", fold_f1s)
    print("Mean LOSO macro-F1:", mean_f1)


if __name__ == "__main__":
    main()
