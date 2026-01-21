# ============================================================
# Temporal MultiModal VAE – GPU Optimized Training
# ============================================================

from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import torch

from src.models.temporal_vae import TemporalMultiModalVAE
from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer
from src.losses.temporal_vae_loss import temporal_vae_loss


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32              # ↑ größer für stabilere Recon
EPOCHS = 50                  # etwas länger
LR = 1e-3

BETA = 5e-5                  # etwas sanfter
KL_WARMUP_EPOCHS = 30        # längerer Warmup → bessere Recon

NUM_WORKERS = 4              # an CPU anpassen
PIN_MEMORY = True

NORMALIZER_PATH = "data/combined_normalizer.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============================================================
# CUDA / AMP SETTINGS
# ============================================================

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()


# ============================================================
# DATASETS
# ============================================================

print("Loading normalizer...")
normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)

train_dataset = ConcatDataset([
    CogAgeVAEDataset(DATA_ROOTS["blho"], "training", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["bbh"], "training", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["state"], "training", normalizer),
])

test_dataset = ConcatDataset([
    CogAgeVAEDataset(DATA_ROOTS["blho"], "testing", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["bbh"], "testing", normalizer),
    CogAgeVAEDataset(DATA_ROOTS["state"], "testing", normalizer),
])

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples : {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# MODEL
# ============================================================

model = TemporalMultiModalVAE(
    z_phone=32,
    z_watch=32,
    z_glasses=16
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ============================================================
# TRAIN / EVAL LOOP
# ============================================================

for epoch in range(1, EPOCHS + 1):

    # ---------- TRAIN ----------
    model.train()
    warmup = min(epoch / KL_WARMUP_EPOCHS, 1.0)
    beta_eff = BETA * warmup * warmup

    train_tot = train_rec = train_kl = 0.0

    for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}"):

        phone = batch["phone"].to(DEVICE, non_blocking=True)
        watch = batch["watch"].to(DEVICE, non_blocking=True)
        glasses = batch["glasses"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(phone, watch, glasses)
            loss, parts = temporal_vae_loss(
                outputs=outputs,
                batch={"phone": phone, "watch": watch, "glasses": glasses},
                beta=beta_eff
            )

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_tot += loss.item()
        train_rec += parts["recon"].item()
        train_kl += parts["kl"].item()

    train_tot /= len(train_loader)
    train_rec /= len(train_loader)
    train_kl /= len(train_loader)

    # ---------- EVAL ----------
    model.eval()
    test_tot = test_rec = test_kl = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[Eval ] Epoch {epoch}/{EPOCHS}"):

            phone = batch["phone"].to(DEVICE, non_blocking=True)
            watch = batch["watch"].to(DEVICE, non_blocking=True)
            glasses = batch["glasses"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(phone, watch, glasses)
                loss, parts = temporal_vae_loss(
                    outputs=outputs,
                    batch={"phone": phone, "watch": watch, "glasses": glasses},
                    beta=beta_eff
                )

            test_tot += loss.item()
            test_rec += parts["recon"].item()
            test_kl += parts["kl"].item()

    test_tot /= len(test_loader)
    test_rec /= len(test_loader)
    test_kl /= len(test_loader)

    print(
        f"\nEpoch {epoch:03d} | beta={beta_eff:.2e}\n"
        f"Train: loss={train_tot:.4f}, recon={train_rec:.4f}, kl={train_kl:.4f}\n"
        f"Test : loss={test_tot:.4f}, recon={test_rec:.4f}, kl={test_kl:.4f}\n"
    )

    # ---------- CHECKPOINT ----------
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "beta": beta_eff,
        },
        CHECKPOINT_DIR / f"vae_gpu_epoch_{epoch:03d}.pt"
    )

print("✅ Temporal MultiModal VAE (GPU) training finished.")
