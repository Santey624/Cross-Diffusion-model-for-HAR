# ============================================================
# Train Sensor VAE V4 — Shared+Private with Product of Experts
#
# Architecture change vs V3:
#   V1-V3: each sensor encoded INDEPENDENTLY → low cross-sensor R²
#   V4:    shared encoder → Product of Experts → z_shared for ALL sensors
#          + per-sensor private encoder → z_private (sensor-specific)
#
# V4 additions:
#   - Direct imputation loss: encode others → decode missing sensor
#   - Semi-supervised: classifier head on z_shared → activity labels
#     behavioral (blho+bbh, 55 classes) + state (6 classes)
#     Forces z_shared to encode activity, not sensor-specific features
#
# D_SHARED=16 (increased for 55-class behavioral classification)
# ============================================================

from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sensor_vae_v4 import SensorSharedPrivateVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.cogage_labeled_dataset import CogAgeLabeledDataset, get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer
from src.losses.sensor_vae_v4_loss import sensor_vae_v4_loss


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_SHARED     = 16   # increased from 8 for semi-supervised classification
D_PRIVATE    = 8
T_LAT        = 64

BATCH_SIZE       = 32
EPOCHS           = 150
LR               = 1e-3
BETA_SHARED      = 1e-3
BETA_PRIVATE     = 2e-2   # forces z_private near prior (kl_p target: 20–40)
IMPUTE_WEIGHT    = 1.0    # direct imputation loss: encode others → decode missing
SUPERVISED_WEIGHT = 0.1   # classification loss on z_shared (behavioral + state)
KL_WARMUP_EPOCHS = 50
MASK_RATIO       = 0.5    # decoder must work without z_private more often

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
        if "label" not in sample:
            sample["label"] = -1  # no label available (WISDM)
        return sample


# ============================================================
# MAIN
# ============================================================
def main():
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n{'='*60}")
    print("Training Sensor VAE V4 — Shared+Private + Semi-supervised")
    print(f"D_shared={D_SHARED}, D_private={D_PRIVATE}, T_lat={T_LAT}")
    print(f"Beta_shared={BETA_SHARED}, Beta_private={BETA_PRIVATE}")
    print(f"Impute_weight={IMPUTE_WEIGHT}, Supervised_weight={SUPERVISED_WEIGHT}")
    print(f"Mask_ratio={MASK_RATIO}, Warmup: {KL_WARMUP_EPOCHS} epochs")
    print(f"{'='*60}\n")

    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # ---- Datasets ----
    print("Loading CogAge datasets (with labels)...")

    # behavioral: blho + bbh — consistent label mapping across both
    behavioral_train = get_combined_labeled_dataset(
        {"blho": COGAGE_ROOTS["blho"], "bbh": COGAGE_ROOTS["bbh"]},
        "training", normalizer,
    )
    behavioral_test = get_combined_labeled_dataset(
        {"blho": COGAGE_ROOTS["blho"], "bbh": COGAGE_ROOTS["bbh"]},
        "testing", normalizer,
    )
    n_classes_behavioral = behavioral_train.n_classes
    print(f"  Behavioral train: {len(behavioral_train)}, classes: {n_classes_behavioral}")

    # state: separate subset, 6 classes
    state_train = CogAgeLabeledDataset(COGAGE_ROOTS["state"], "training", normalizer)
    state_test  = CogAgeLabeledDataset(COGAGE_ROOTS["state"], "testing",  normalizer)
    n_classes_state = state_train.n_classes
    print(f"  State train:      {len(state_train)}, classes: {n_classes_state}")

    cogage_train = ConcatDataset([
        TaggedDataset(behavioral_train, "behavioral"),
        TaggedDataset(state_train,      "state"),
    ])
    cogage_test = ConcatDataset([
        TaggedDataset(behavioral_test, "behavioral"),
        TaggedDataset(state_test,      "state"),
    ])
    print(f"  CogAge total train: {len(cogage_train)}, test: {len(cogage_test)}")

    print("Loading WISDM dataset...")
    wisdm_path = Path(WISDM_ROOT)
    if wisdm_path.exists():
        wisdm_train = TaggedDataset(
            CogAgeSensorDataset(WISDM_ROOT, "training", normalizer), "wisdm"
        )
        print(f"  WISDM train: {len(wisdm_train)}")
        train_dataset = ConcatDataset([cogage_train, wisdm_train])
    else:
        print("  WISDM not found — training on CogAge only")
        train_dataset = cogage_train

    print(f"  Total train: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        cogage_test, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # ---- Model ----
    print(f"\nCreating SensorSharedPrivateVAE V4...")
    model = SensorSharedPrivateVAE(
        d_shared=D_SHARED, d_private=D_PRIVATE, t_lat=T_LAT
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # Classifier heads: GlobalAvgPool(z_shared) → Linear → n_classes
    clf_behavioral = nn.Linear(D_SHARED, n_classes_behavioral).to(DEVICE)
    clf_state      = nn.Linear(D_SHARED, n_classes_state).to(DEVICE)
    print(f"Classifier heads: behavioral ({n_classes_behavioral} classes), state ({n_classes_state} classes)")

    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(clf_behavioral.parameters()) +
        list(clf_state.parameters()),
        lr=LR,
    )

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"{'='*60}\n")

    best_impute = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        clf_behavioral.train()
        clf_state.train()

        warmup        = min(epoch / KL_WARMUP_EPOCHS, 1.0)
        beta_eff      = BETA_SHARED * warmup * warmup
        beta_priv_eff = BETA_PRIVATE * warmup * warmup

        t_tot = t_rec = t_kls = t_klp = t_imp = t_sup = 0.0
        n_cogage = n_wisdm = 0

        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}", leave=False):
            sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
            source      = batch["_source"]
            labels      = batch["label"]

            has_wisdm  = any(s == "wisdm"  for s in source)
            has_cogage = any(s != "wisdm"  for s in source)

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
                    align_weight=0.0, valid_sensors=valid,
                )

                # --- Imputation loss ---
                impute_sensors = list(valid)
                target = impute_sensors[torch.randint(len(impute_sensors), (1,)).item()]
                others = {k: sensor_data[k] for k in impute_sensors if k != target}
                imputed    = model.impute(others, target)
                impute_loss = F.mse_loss(imputed, sensor_data[target])
                loss = loss + IMPUTE_WEIGHT * impute_loss

                # --- Supervised loss on z_shared ---
                # GlobalAvgPool over time → (B, D_SHARED)
                z_feat = mu_shared.mean(dim=2)
                sup_loss = torch.tensor(0.0, device=DEVICE)

                beh_mask = torch.tensor([s == "behavioral" for s in source])
                if beh_mask.any():
                    lbl = labels[beh_mask].to(DEVICE)
                    logits = clf_behavioral(z_feat[beh_mask.to(DEVICE)])
                    sup_loss = sup_loss + F.cross_entropy(logits, lbl)

                st_mask = torch.tensor([s == "state" for s in source])
                if st_mask.any():
                    lbl = labels[st_mask].to(DEVICE)
                    logits = clf_state(z_feat[st_mask.to(DEVICE)])
                    sup_loss = sup_loss + F.cross_entropy(logits, lbl)

                loss = loss + SUPERVISED_WEIGHT * sup_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_tot += loss.item()
            t_rec += parts["recon"].item()
            t_kls += parts["kl_s"].item()
            t_klp += parts["kl_p"].item()
            t_imp += impute_loss.item()
            t_sup += sup_loss.item()

        N = len(train_loader)
        t_tot /= N; t_rec /= N; t_kls /= N; t_klp /= N; t_imp /= N; t_sup /= N

        # ---------- EVAL ----------
        model.eval()
        clf_behavioral.eval()
        clf_state.eval()

        e_tot = e_rec = e_kls = e_klp = e_imp = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[Eval ] {epoch}/{EPOCHS}", leave=False):
                sensor_data = {k: batch[k].to(DEVICE, non_blocking=True) for k in SENSOR_NAMES}
                with torch.cuda.amp.autocast():
                    outputs, mu_shared, logvar_shared = model(sensor_data, mask_ratio=0.0)
                    loss, parts = sensor_vae_v4_loss(
                        outputs, sensor_data, mu_shared, logvar_shared,
                        beta_shared=beta_eff, beta_private=beta_priv_eff,
                        align_weight=0.0, valid_sensors=ALL_SENSORS,
                    )
                    # Eval imputation: average over all sensors
                    imp_loss_sum = 0.0
                    for tgt in SENSOR_NAMES:
                        others  = {k: sensor_data[k] for k in SENSOR_NAMES if k != tgt}
                        imputed = model.impute(others, tgt)
                        imp_loss_sum += F.mse_loss(imputed, sensor_data[tgt]).item()
                    imp_loss_mean = imp_loss_sum / len(SENSOR_NAMES)

                e_tot += loss.item()
                e_rec += parts["recon"].item()
                e_kls += parts["kl_s"].item()
                e_klp += parts["kl_p"].item()
                e_imp += imp_loss_mean

        M = len(test_loader)
        e_tot /= M; e_rec /= M; e_kls /= M; e_klp /= M; e_imp /= M

        print(
            f"Epoch {epoch:03d} | beta_s={beta_eff:.2e} | cogage={n_cogage} wisdm={n_wisdm}\n"
            f"  Train: loss={t_tot:.4f} recon={t_rec:.4f} kl_s={t_kls:.4f} "
            f"kl_p={t_klp:.4f} impute={t_imp:.4f} sup={t_sup:.4f}\n"
            f"  Test : loss={e_tot:.4f} recon={e_rec:.4f} kl_s={e_kls:.4f} "
            f"kl_p={e_klp:.4f} impute={e_imp:.4f}\n"
        )

        ckpt = {
            "epoch":              epoch,
            "model_state":        model.state_dict(),
            "clf_behavioral":     clf_behavioral.state_dict(),
            "clf_state":          clf_state.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "beta_shared":        beta_eff,
            "beta_private":       beta_priv_eff,
            "test_recon":         e_rec,
            "test_impute":        e_imp,
            "config": {
                "d_shared":            D_SHARED,
                "d_private":           D_PRIVATE,
                "t_lat":               T_LAT,
                "mask_ratio":          MASK_RATIO,
                "n_classes_behavioral": n_classes_behavioral,
                "n_classes_state":      n_classes_state,
            },
        }

        if epoch % 25 == 0:
            torch.save(ckpt, OUT_DIR / f"epoch_{epoch:03d}.pt")

        # Best model = lowest imputation loss after warmup (not recon)
        if epoch >= KL_WARMUP_EPOCHS and e_imp < best_impute:
            best_impute = e_imp
            torch.save(ckpt, OUT_DIR / "best_model.pt")
            print(f"  --> New best impute: {best_impute:.6f}")

    print(f"{'='*60}")
    print(f"Training finished. Best test impute: {best_impute:.6f}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
