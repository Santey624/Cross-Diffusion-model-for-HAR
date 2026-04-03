# ============================================================
# Train Sensor VAE V4 — Shared+Private with Product of Experts
#
# Architecture change vs V3:
#   V1-V3: each sensor encoded INDEPENDENTLY → low cross-sensor R²
#   V4:    shared encoder → Product of Experts → z_shared for ALL sensors
#          + per-sensor private encoder → z_private (sensor-specific)
#
# Why this fixes imputation:
#   z_shared is inferred from ALL present sensors jointly.
#   Cross-sensor R²(z_shared) ≈ 1.0 by design.
#   Missing sensor imputed via: decode(z_shared_from_others, 0)
#
# Latent: D_SHARED=8 + D_PRIVATE=8 = 16 total (same as V2/V3)
# Training trick: random mask_ratio=0.3 → PoE uses random sensor subsets
#                 → robust z_shared inference from any subset
# ============================================================

from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import torch

from src.models.sensor_vae_v4 import SensorSharedPrivateVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from src.losses.sensor_vae_v4_loss import sensor_vae_v4_loss


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_SHARED     = 8
D_PRIVATE    = 8
T_LAT        = 64

BATCH_SIZE       = 32
EPOCHS           = 150
LR               = 1e-3
BETA_SHARED      = 1e-3
BETA_PRIVATE     = 1e-3
KL_WARMUP_EPOCHS = 50
MASK_RATIO       = 0.3   # fraction of sensors randomly dropped from PoE per batch

NUM_WORKERS = 4
PIN_MEMORY  = True

COGAGE_ROOTS = {
    "blho":  "data/cogage/python/arrays/blho",
    "bbh":   "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}
WISDM_ROOT = "data/wisdm/arrays"

WISDM_REAL_SENSORS = {"phone_acc", "phone_gyro", "watch_acc", "watch_gyro"}
ALL_SENSORS        = set(SENSOR_NAMES)

OUT_DIR = Path("checkpoints/sensor_vae_v4")
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
    print("Training Sensor VAE V4 — Shared+Private (Product of Experts)")
    print(f"D_shared={D_SHARED}, D_private={D_PRIVATE}, T_lat={T_LAT}")
    print(f"Beta_shared={BETA_SHARED}, Beta_private={BETA_PRIVATE}")
    print(f"Mask_ratio={MASK_RATIO} (random PoE subset for robustness)")
    print(f"Warmup: {KL_WARMUP_EPOCHS} epochs")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    print("Loading CogAge datasets...")
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

    print("Loading WISDM dataset...")
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

    print(f"\nCreating SensorSharedPrivateVAE V4...")
    model = SensorSharedPrivateVAE(
        d_shared=D_SHARED, d_private=D_PRIVATE, t_lat=T_LAT
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
    print(f"Shared encoder params (shared): {sum(p.numel() for p in model.shared_encoder.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"{'='*60}\n")

    best_recon = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        warmup        = min(epoch / KL_WARMUP_EPOCHS, 1.0)
        beta_eff      = BETA_SHARED * warmup * warmup
        beta_priv_eff = BETA_PRIVATE * warmup * warmup

        t_tot = t_rec = t_kls = t_klp = 0.0
        n_cogage = n_wisdm = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
            source      = batch["_source"]

            has_wisdm  = any(s == "wisdm"  for s in source)
            has_cogage = any(s == "cogage" for s in source)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs, mu_shared, logvar_shared = model(
                    sensor_data, mask_ratio=MASK_RATIO
                )

                valid = WISDM_REAL_SENSORS if (has_wisdm and not has_cogage) else ALL_SENSORS
                if has_wisdm and not has_cogage:
                    n_wisdm += 1
                else:
                    n_cogage += 1

                loss, parts = sensor_vae_v4_loss(
                    outputs, sensor_data, mu_shared, logvar_shared,
                    beta_shared=beta_eff, beta_private=beta_priv_eff,
                    valid_sensors=valid,
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_tot += loss.item()
            t_rec += parts["recon"].item()
            t_kls += parts["kl_s"].item()
            t_klp += parts["kl_p"].item()

        N = len(train_loader)
        t_tot /= N; t_rec /= N; t_kls /= N; t_klp /= N

        # ---------- EVAL ----------
        model.eval()
        e_tot = e_rec = e_kls = e_klp = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
                with torch.cuda.amp.autocast():
                    outputs, mu_shared, logvar_shared = model(sensor_data, mask_ratio=0.0)
                    loss, parts = sensor_vae_v4_loss(
                        outputs, sensor_data, mu_shared, logvar_shared,
                        beta_shared=beta_eff, beta_private=beta_priv_eff,
                        valid_sensors=ALL_SENSORS,
                    )
                e_tot += loss.item()
                e_rec += parts["recon"].item()
                e_kls += parts["kl_s"].item()
                e_klp += parts["kl_p"].item()

        M = len(test_loader)
        e_tot /= M; e_rec /= M; e_kls /= M; e_klp /= M

        print(
            f"Epoch {epoch:03d} | beta_s={beta_eff:.2e} | cogage={n_cogage} wisdm={n_wisdm}\n"
            f"  Train: loss={t_tot:.4f} recon={t_rec:.4f} kl_s={t_kls:.4f} kl_p={t_klp:.4f}\n"
            f"  Test : loss={e_tot:.4f} recon={e_rec:.4f} kl_s={e_kls:.4f} kl_p={e_klp:.4f}\n"
        )

        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "beta_shared":     beta_eff,
            "beta_private":    beta_priv_eff,
            "test_recon":      e_rec,
            "config": {
                "d_shared":   D_SHARED,
                "d_private":  D_PRIVATE,
                "t_lat":      T_LAT,
                "mask_ratio": MASK_RATIO,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        if e_rec < best_recon:
            best_recon = e_rec
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best recon: {best_recon:.6f}")

    print(f"{'='*60}")
    print(f"Training finished. Best test recon: {best_recon:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
