# ============================================================
# Train Sensor-Level Multimodal VAE — Shared Latent Space
# Shared encoder/decoder, per-sensor projections, alignment loss
# ============================================================

from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from src.losses.sensor_vae_loss import sensor_vae_loss


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

BETA = 5e-5
KL_WARMUP_EPOCHS = 30
ALIGN_WEIGHT = 0.1

NUM_WORKERS = 4
PIN_MEMORY = True

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

NORMALIZER_PATH = "data/sensor_normalizer.npz"


# ============================================================
# COMPUTE OR LOAD NORMALIZER
# ============================================================
def compute_sensor_normalizer():
    """Compute per-sensor normalization stats from all training data."""
    print("Computing sensor normalizer stats...")

    all_data = {k: [] for k in SENSOR_NAMES}

    for name, root in DATA_ROOTS.items():
        ds = CogAgeSensorDataset(root, split="training")
        for key in SENSOR_NAMES:
            all_data[key].append(ds.data[key])  # (N, T, 3)

    stats = {}
    for key in SENSOR_NAMES:
        concat = np.concatenate(all_data[key], axis=0)  # (N_total, T, 3)
        mean = concat.mean(axis=(0, 1))  # (3,)
        std = concat.std(axis=(0, 1)).clip(min=1e-8)  # (3,)
        stats[key] = (mean, std)
        print(f"  {key:15s}: mean={mean.round(4)}, std={std.round(4)}")

    normalizer = SensorNormalizer(stats)
    normalizer.save(NORMALIZER_PATH)
    print(f"Saved normalizer to {NORMALIZER_PATH}")
    return normalizer


# ============================================================
# MAIN
# ============================================================
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

# Load or compute normalizer
try:
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)
    print(f"Loaded normalizer from {NORMALIZER_PATH}")
except FileNotFoundError:
    normalizer = compute_sensor_normalizer()

# Datasets
print("Loading datasets...")
train_dataset = ConcatDataset([
    CogAgeSensorDataset(DATA_ROOTS["blho"], "training", normalizer),
    CogAgeSensorDataset(DATA_ROOTS["bbh"], "training", normalizer),
    CogAgeSensorDataset(DATA_ROOTS["state"], "training", normalizer),
])

test_dataset = ConcatDataset([
    CogAgeSensorDataset(DATA_ROOTS["blho"], "testing", normalizer),
    CogAgeSensorDataset(DATA_ROOTS["bbh"], "testing", normalizer),
    CogAgeSensorDataset(DATA_ROOTS["state"], "testing", normalizer),
])

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
)


# Model
print("\nCreating SensorMultiModalVAE (shared latent space)...")
model = SensorMultiModalVAE().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params / 1e6:.2f}M")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Training loop
print(f"\nStarting training for {EPOCHS} epochs on {DEVICE}...")
print(f"Alignment weight: {ALIGN_WEIGHT}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):

    # ---------- TRAIN ----------
    model.train()
    warmup = min(epoch / KL_WARMUP_EPOCHS, 1.0)
    beta_eff = BETA * warmup * warmup

    train_tot = train_rec = train_kl = train_align = 0.0

    for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}", leave=False):
        sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(sensor_data)
            loss, parts = sensor_vae_loss(outputs, sensor_data,
                                          beta=beta_eff, align_weight=ALIGN_WEIGHT)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_tot += loss.item()
        train_rec += parts["recon"].item()
        train_kl += parts["kl"].item()
        train_align += parts["align"].item()

    train_tot /= len(train_loader)
    train_rec /= len(train_loader)
    train_kl /= len(train_loader)
    train_align /= len(train_loader)

    # ---------- EVAL ----------
    model.eval()
    test_tot = test_rec = test_kl = test_align = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[Eval ] Epoch {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}

            with torch.cuda.amp.autocast():
                outputs = model(sensor_data)
                loss, parts = sensor_vae_loss(outputs, sensor_data,
                                              beta=beta_eff, align_weight=ALIGN_WEIGHT)

            test_tot += loss.item()
            test_rec += parts["recon"].item()
            test_kl += parts["kl"].item()
            test_align += parts["align"].item()

    test_tot /= len(test_loader)
    test_rec /= len(test_loader)
    test_kl /= len(test_loader)
    test_align /= len(test_loader)

    print(
        f"\nEpoch {epoch:03d} | beta={beta_eff:.2e}\n"
        f"Train: loss={train_tot:.4f}, recon={train_rec:.4f}, kl={train_kl:.4f}, align={train_align:.4f}\n"
        f"Test : loss={test_tot:.4f}, recon={test_rec:.4f}, kl={test_kl:.4f}, align={test_align:.4f}\n"
    )

    # ---------- CHECKPOINT ----------
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "beta": beta_eff,
        },
        CHECKPOINT_DIR / f"sensor_vae_epoch_{epoch:03d}.pt"
    )

print(f"{'='*60}")
print("Sensor VAE training finished.")
print(f"{'='*60}")
