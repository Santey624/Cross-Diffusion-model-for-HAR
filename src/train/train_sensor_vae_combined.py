# ============================================================
# Train Sensor-Level VAE on CogAge + WISDM Combined
# WISDM: only phone_acc, phone_gyro, watch_acc, watch_gyro
# CogAge: all 7 sensors
# ============================================================

from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from src.losses.sensor_vae_loss_masked import sensor_vae_loss_masked


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

BETA = 1e-3
KL_WARMUP_EPOCHS = 40
ALIGN_WEIGHT = 0.05

NUM_WORKERS = 4
PIN_MEMORY = True

COGAGE_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

WISDM_ROOT = "data/wisdm/arrays"

# Sensors with real WISDM data
WISDM_REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}
ALL_SENSORS = set(SENSOR_NAMES)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"


# ============================================================
# COMPUTE NORMALIZER (from CogAge only — target domain)
# ============================================================
def compute_sensor_normalizer():
    """Compute per-sensor normalization stats from CogAge training data."""
    print("Computing sensor normalizer stats (CogAge only)...")

    all_data = {k: [] for k in SENSOR_NAMES}

    for name, root in COGAGE_ROOTS.items():
        ds = CogAgeSensorDataset(root, split="training")
        for key in SENSOR_NAMES:
            all_data[key].append(ds.data[key])

    stats = {}
    for key in SENSOR_NAMES:
        concat = np.concatenate(all_data[key], axis=0)
        mean = concat.mean(axis=(0, 1))
        std = concat.std(axis=(0, 1)).clip(min=1e-8)
        stats[key] = (mean, std)
        print(f"  {key:15s}: mean={mean.round(4)}, std={std.round(4)}")

    normalizer = SensorNormalizer(stats)
    normalizer.save(NORMALIZER_PATH)
    print(f"Saved normalizer to {NORMALIZER_PATH}")
    return normalizer


# ============================================================
# CUSTOM DATASET WRAPPER to tag source
# ============================================================
class TaggedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and adds a 'source' tag to each sample."""

    def __init__(self, dataset, source_tag):
        self.dataset = dataset
        self.source_tag = source_tag

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["_source"] = self.source_tag
        return sample


# ============================================================
# MAIN
# ============================================================
def main():
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n{'='*60}")
    print("Training Sensor VAE on CogAge + WISDM")
    print(f"Beta: {BETA}, Align: {ALIGN_WEIGHT}")
    print(f"{'='*60}\n")

    # Normalizer
    try:
        normalizer = SensorNormalizer.load(NORMALIZER_PATH)
        print(f"Loaded normalizer from {NORMALIZER_PATH}")
    except FileNotFoundError:
        normalizer = compute_sensor_normalizer()

    # CogAge datasets
    print("\nLoading CogAge datasets...")
    cogage_train = ConcatDataset([
        TaggedDataset(
            CogAgeSensorDataset(COGAGE_ROOTS["blho"], "training", normalizer),
            "cogage"
        ),
        TaggedDataset(
            CogAgeSensorDataset(COGAGE_ROOTS["bbh"], "training", normalizer),
            "cogage"
        ),
        TaggedDataset(
            CogAgeSensorDataset(COGAGE_ROOTS["state"], "training", normalizer),
            "cogage"
        ),
    ])
    print(f"CogAge train: {len(cogage_train)}")

    cogage_test = ConcatDataset([
        TaggedDataset(
            CogAgeSensorDataset(COGAGE_ROOTS["blho"], "testing", normalizer),
            "cogage"
        ),
        TaggedDataset(
            CogAgeSensorDataset(COGAGE_ROOTS["bbh"], "testing", normalizer),
            "cogage"
        ),
        TaggedDataset(
            CogAgeSensorDataset(COGAGE_ROOTS["state"], "testing", normalizer),
            "cogage"
        ),
    ])
    print(f"CogAge test: {len(cogage_test)}")

    # WISDM dataset
    print("\nLoading WISDM dataset...")
    wisdm_path = Path(WISDM_ROOT)
    if wisdm_path.exists():
        wisdm_train = TaggedDataset(
            CogAgeSensorDataset(WISDM_ROOT, "training", normalizer),
            "wisdm"
        )
        print(f"WISDM train: {len(wisdm_train)}")
    else:
        print(f"WISDM not found at {WISDM_ROOT}. Run download_wisdm.py first!")
        print("Training on CogAge only...")
        wisdm_train = None

    # Combined train dataset
    if wisdm_train is not None:
        train_dataset = ConcatDataset([cogage_train, wisdm_train])
    else:
        train_dataset = cogage_train

    print(f"\nTotal train: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        cogage_test, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Model
    print("\nCreating SensorMultiModalVAE...")
    model = SensorMultiModalVAE().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"{'='*60}\n")

    best_recon = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        warmup = min(epoch / KL_WARMUP_EPOCHS, 1.0)
        beta_eff = BETA * warmup * warmup

        train_tot = train_rec = train_kl = train_align = 0.0
        n_cogage = n_wisdm = 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
            source = batch["_source"]

            # Determine valid sensors per sample
            # If any sample in batch is WISDM, restrict to WISDM sensors
            # (simplified: use batch-level source since we mix)
            has_wisdm = any(s == "wisdm" for s in source)
            has_cogage = any(s == "cogage" for s in source)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(sensor_data)

                if has_wisdm and not has_cogage:
                    # Pure WISDM batch
                    valid = WISDM_REAL_SENSORS
                    n_wisdm += 1
                elif has_cogage and not has_wisdm:
                    # Pure CogAge batch
                    valid = ALL_SENSORS
                    n_cogage += 1
                else:
                    # Mixed batch — use all sensors but weight
                    # CogAge contributes to all, WISDM to 4
                    valid = ALL_SENSORS
                    n_cogage += 1

                loss, parts = sensor_vae_loss_masked(
                    outputs, sensor_data,
                    beta=beta_eff, align_weight=ALIGN_WEIGHT,
                    valid_sensors=valid,
                )

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

        # ---------- EVAL (CogAge only) ----------
        model.eval()
        test_tot = test_rec = test_kl = test_align = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] Epoch {epoch}/{EPOCHS}", leave=False):
                sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}

                with torch.cuda.amp.autocast():
                    outputs = model(sensor_data)
                    loss, parts = sensor_vae_loss_masked(
                        outputs, sensor_data,
                        beta=beta_eff, align_weight=ALIGN_WEIGHT,
                        valid_sensors=ALL_SENSORS,
                    )

                test_tot += loss.item()
                test_rec += parts["recon"].item()
                test_kl += parts["kl"].item()
                test_align += parts["align"].item()

        test_tot /= len(test_loader)
        test_rec /= len(test_loader)
        test_kl /= len(test_loader)
        test_align /= len(test_loader)

        print(
            f"\nEpoch {epoch:03d} | beta={beta_eff:.2e} | CogAge batches: {n_cogage}, WISDM batches: {n_wisdm}\n"
            f"Train: loss={train_tot:.4f}, recon={train_rec:.4f}, kl={train_kl:.4f}, align={train_align:.4f}\n"
            f"Test : loss={test_tot:.4f}, recon={test_rec:.4f}, kl={test_kl:.4f}, align={test_align:.4f}\n"
        )

        # ---------- CHECKPOINT ----------
        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "beta": beta_eff,
                    "test_recon": test_rec,
                },
                CHECKPOINT_DIR / f"sensor_vae_combined_epoch_{epoch:03d}.pt"
            )

        if test_rec < best_recon:
            best_recon = test_rec
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "beta": beta_eff,
                    "test_recon": test_rec,
                },
                CHECKPOINT_DIR / "sensor_vae_combined_best.pt"
            )

    print(f"{'='*60}")
    print(f"Training finished. Best test recon: {best_recon:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
