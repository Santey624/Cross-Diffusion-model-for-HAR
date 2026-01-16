from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer
from torch.utils.data import ConcatDataset
from src.models.vae import MultiModalVAE
from src.losses.vae_loss import vae_loss
import torch


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
BETA = 1e-3

NORMALIZER_PATH = "data/combined_normalizer.npz"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}


# ============================================================
# DATASETS
# ============================================================
print("Loading normalizer...")
normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)

# -------- TRAIN (Session #1) --------
ds_blho_train = CogAgeVAEDataset(DATA_ROOTS["blho"], "training", normalizer)
ds_bbh_train = CogAgeVAEDataset(DATA_ROOTS["bbh"], "training", normalizer)
ds_state_train = CogAgeVAEDataset(DATA_ROOTS["state"], "training", normalizer)

train_dataset = ConcatDataset([
    ds_blho_train,
    ds_bbh_train,
    ds_state_train
])

# -------- TEST (Session #2) --------
ds_blho_test = CogAgeVAEDataset(DATA_ROOTS["blho"], "testing", normalizer)
ds_bbh_test = CogAgeVAEDataset(DATA_ROOTS["bbh"], "testing", normalizer)
ds_state_test = CogAgeVAEDataset(DATA_ROOTS["state"], "testing", normalizer)

test_dataset = ConcatDataset([
    ds_blho_test,
    ds_bbh_test,
    ds_state_test
])

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples : {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================
model = MultiModalVAE(
    z_device=32,
    z_fused=64
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ============================================================
# TRAIN / TEST LOOP
# ============================================================
for epoch in range(1, EPOCHS + 1):

    # --------------------
    # TRAIN
    # --------------------
    model.train()
    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0

    for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}"):
        phone = batch["phone"].to(DEVICE)
        watch = batch["watch"].to(DEVICE)
        glasses = batch["glasses"].to(DEVICE)

        output = model(phone, watch, glasses)

        loss, recon, kl = vae_loss(
            batch={
                "phone": phone,
                "watch": watch,
                "glasses": glasses
            },
            output=output,
            beta=BETA
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()

    train_loss /= len(train_loader)
    train_recon /= len(train_loader)
    train_kl /= len(train_loader)

    # --------------------
    # TEST
    # --------------------
    model.eval()
    test_loss = 0.0
    test_recon = 0.0
    test_kl = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[Test ] Epoch {epoch}/{EPOCHS}"):
            phone = batch["phone"].to(DEVICE)
            watch = batch["watch"].to(DEVICE)
            glasses = batch["glasses"].to(DEVICE)

            output = model(phone, watch, glasses)

            loss, recon, kl = vae_loss(
                batch={
                    "phone": phone,
                    "watch": watch,
                    "glasses": glasses
                },
                output=output,
                beta=BETA
            )

            test_loss += loss.item()
            test_recon += recon.item()
            test_kl += kl.item()

    test_loss /= len(test_loader)
    test_recon /= len(test_loader)
    test_kl /= len(test_loader)

    # --------------------
    # LOGGING
    # --------------------
    print(
        f"\nEpoch {epoch:03d} | "
        f"Train: loss={train_loss:.4f}, recon={train_recon:.4f}, kl={train_kl:.4f} | "
        f"Test:  loss={test_loss:.4f}, recon={test_recon:.4f}, kl={test_kl:.4f}\n"
    )

    # --------------------
    # CHECKPOINT
    # --------------------
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        CHECKPOINT_DIR / f"vae_epoch_{epoch:03d}.pt"
    )

print("✅ VAE training finished.")
