import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import CogAgeContractDataset
from src.data.modality_registry import MODALITIES
from src.models.vae_har import VAEHAR
from src.train.vae_losses import kl_divergence


def train_vae_fold(test_subject: int, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Fold-spezifischer Output-Ordner
    run_dir = Path(f"runs/vae_blho/fold_test_subject_{test_subject}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    with open(f"processed/behavior_blho/splits_loso/fold_test_subject_{test_subject}.json") as f:
        split = json.load(f)

    norm_path = f"processed/behavior_blho/norm_loso/norm_fold_test_subject_{test_subject}.json"

    train_ds = CogAgeContractDataset(
        "processed/behavior_blho/samples.pt",
        split["train_indices"],
        norm_path,
        device_filter=["phone"],
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    modality_names = [m for m, s in MODALITIES.items() if s.device == "phone"]
    num_classes = max(s["label"] for s in torch.load("processed/behavior_blho/samples.pt")) + 1

    model = VAEHAR(modality_names, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for b in train_loader:
            modalities = {k: v.to(device) for k, v in b["modalities"].items()}
            mask = {k: v.to(device) for k, v in b["mask"].items()}
            y = b["label"].to(device)

            opt.zero_grad()
            logits, _, mu, logvar, _ = model(modalities, mask)

            cls = criterion(logits, y)
            kl = kl_divergence(mu, logvar)

            loss = cls + 1e-3 * kl
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Fold {test_subject}] epoch {ep:02d} loss={avg_loss:.4f}")

        # 🔹 Bestes Modell speichern
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    return model


if __name__ == "__main__":
    print("Starting VAE LOSO training...")

    for test_subject in [1, 2, 3, 4]:
        train_vae_fold(test_subject=test_subject, epochs=25)
