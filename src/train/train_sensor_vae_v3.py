# ============================================================
# Train Sensor VAE V3 — Strong Alignment for Imputation
#
# Key difference to V2:
#   - ALIGN_WEIGHT: 0.05 → 1.0
#   - Alignment on FULL latent sequence (B, D, T), not time-mean
#   - Cosine similarity term added (directional alignment)
#   Goal: phone_acc ↔ watch_acc ↔ glasses_acc latents become
#         strongly correlated so diffusion imputation can beat mean-fill
#
# Same architecture as V2: D=16, T=64
# ============================================================

from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import torch

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from src.losses.sensor_vae_loss_masked import sensor_vae_loss_masked


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM   = 16
T_SHARED     = 64

BATCH_SIZE       = 32
EPOCHS           = 150
LR               = 1e-3
BETA             = 1e-3
KL_WARMUP_EPOCHS = 50
ALIGN_WEIGHT     = 1.0   # V2 was 0.05 — strong alignment for imputation

NUM_WORKERS = 4
PIN_MEMORY  = True

COGAGE_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}
WISDM_ROOT = "data/wisdm/arrays"

WISDM_REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}
ALL_SENSORS = set(SENSOR_NAMES)

OUT_DIR = Path("checkpoints/sensor_vae_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"


# ============================================================
# DATASET WRAPPER
# ============================================================
class TaggedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, source_tag):
        self.dataset    = dataset
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
    print("Training Sensor VAE V3 (D=16, T=64) — Strong Alignment")
    print(f"Beta: {BETA}, Align: {ALIGN_WEIGHT}, Warmup: {KL_WARMUP_EPOCHS} epochs")
    print(f"Alignment: full sequence MSE + cosine similarity")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print("\nLoading CogAge datasets...")
    cogage_train = ConcatDataset([
        TaggedDataset(CogAgeSensorDataset(COGAGE_ROOTS["blho"],  "training", normalizer), "cogage"),
        TaggedDataset(CogAgeSensorDataset(COGAGE_ROOTS["bbh"],   "training", normalizer), "cogage"),
        TaggedDataset(CogAgeSensorDataset(COGAGE_ROOTS["state"], "training", normalizer), "cogage"),
    ])
    cogage_test = ConcatDataset([
        TaggedDataset(CogAgeSensorDataset(COGAGE_ROOTS["blho"],  "testing", normalizer), "cogage"),
        TaggedDataset(CogAgeSensorDataset(COGAGE_ROOTS["bbh"],   "testing", normalizer), "cogage"),
        TaggedDataset(CogAgeSensorDataset(COGAGE_ROOTS["state"], "testing", normalizer), "cogage"),
    ])
    print(f"CogAge train: {len(cogage_train)}, test: {len(cogage_test)}")

    print("\nLoading WISDM dataset...")
    wisdm_path = Path(WISDM_ROOT)
    if wisdm_path.exists():
        wisdm_train = TaggedDataset(
            CogAgeSensorDataset(WISDM_ROOT, "training", normalizer), "wisdm"
        )
        print(f"WISDM train: {len(wisdm_train)}")
        train_dataset = ConcatDataset([cogage_train, wisdm_train])
    else:
        print("WISDM not found — training on CogAge only")
        train_dataset = cogage_train

    print(f"Total train: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        cogage_test, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    print(f"\nCreating SensorMultiModalVAE V3 (latent_dim={LATENT_DIM}, t_shared={T_SHARED})...")
    model = SensorMultiModalVAE(latent_dim=LATENT_DIM, t_shared=T_SHARED).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"{'='*60}\n")

    best_recon = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        warmup   = min(epoch / KL_WARMUP_EPOCHS, 1.0)
        beta_eff = BETA * warmup * warmup

        train_tot = train_rec = train_kl = train_align = 0.0
        n_cogage = n_wisdm = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
            source = batch["_source"]

            has_wisdm  = any(s == "wisdm"  for s in source)
            has_cogage = any(s == "cogage" for s in source)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(sensor_data)

                if has_wisdm and not has_cogage:
                    valid = WISDM_REAL_SENSORS
                    n_wisdm += 1
                else:
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

            train_tot   += loss.item()
            train_rec   += parts["recon"].item()
            train_kl    += parts["kl"].item()
            train_align += parts["align"].item()

        train_tot   /= len(train_loader)
        train_rec   /= len(train_loader)
        train_kl    /= len(train_loader)
        train_align /= len(train_loader)

        # ---------- EVAL ----------
        model.eval()
        test_tot = test_rec = test_kl = test_align = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
                with torch.cuda.amp.autocast():
                    outputs = model(sensor_data)
                    loss, parts = sensor_vae_loss_masked(
                        outputs, sensor_data,
                        beta=beta_eff, align_weight=ALIGN_WEIGHT,
                        valid_sensors=ALL_SENSORS,
                    )
                test_tot   += loss.item()
                test_rec   += parts["recon"].item()
                test_kl    += parts["kl"].item()
                test_align += parts["align"].item()

        test_tot   /= len(test_loader)
        test_rec   /= len(test_loader)
        test_kl    /= len(test_loader)
        test_align /= len(test_loader)

        print(
            f"Epoch {epoch:03d} | beta={beta_eff:.2e} | "
            f"cogage={n_cogage} wisdm={n_wisdm}\n"
            f"  Train: loss={train_tot:.4f} recon={train_rec:.4f} "
            f"kl={train_kl:.4f} align={train_align:.4f}\n"
            f"  Test : loss={test_tot:.4f} recon={test_rec:.4f} "
            f"kl={test_kl:.4f} align={test_align:.4f}\n"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "beta": beta_eff,
            "test_recon": test_rec,
            "config": {"latent_dim": LATENT_DIM, "t_shared": T_SHARED,
                       "align_weight": ALIGN_WEIGHT},
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if epoch >= KL_WARMUP_EPOCHS and test_rec < best_recon:
            best_recon = test_rec
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best recon: {best_recon:.6f}")

    print(f"{'='*60}")
    print(f"Training finished. Best test recon: {best_recon:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
