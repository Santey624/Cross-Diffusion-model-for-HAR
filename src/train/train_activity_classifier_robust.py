# ============================================================
# Train Robust Activity Classifier on Sensor Latents
# Mixed training: real latents + imputed latents (data augmentation)
# ============================================================

from pathlib import Path
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.models.sensor_joint_diffusion_v2 import create_sensor_diffusion_v2
from src.models.activity_classifier import create_activity_classifier
from src.data.cogage_labeled_dataset import get_combined_labeled_dataset
from src.data.sensor_normalizer import SensorNormalizer


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_combined_best.pt"
DIFFUSION_DIR = Path("checkpoints/sensor_diffusion_v2_pretrain")
NORMALIZER_PATH = "data/sensor_normalizer_combined.npz"

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

OUT_DIR = Path("checkpoints/activity_classifier_robust")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_TYPE = "transformer"
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3

# Training
BATCH_SIZE = 32  # Smaller because diffusion is expensive
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Imputation augmentation
IMPUTE_PROB = 0.5  # Probability to impute in each batch
MIN_MISSING = 1
MAX_MISSING = 3
DDIM_STEPS = 20  # Fast sampling during training

# Device groups for realistic missing patterns
DEVICE_GROUPS = {
    "phone": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
    "watch": ["watch_acc", "watch_gyro"],
    "glasses": ["glasses_acc"],
}


# ============================================================
# COSINE SCHEDULE
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    return betas.float()


def make_schedule(T, schedule_type):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"alpha_bar": alpha_bar}


# ============================================================
# FAST DDIM SAMPLER (V2)
# ============================================================
@torch.no_grad()
def ddim_sample_v2_fast(model, stacked_latents, observed_mask, alpha_bar, T, ddim_steps=20):
    B, K, D, T_len = stacked_latents.shape
    device = stacked_latents.device
    missing_mask = 1.0 - observed_mask

    z = stacked_latents.clone()
    noise_init = torch.randn_like(stacked_latents)
    z = observed_mask[:, :, None, None] * z + missing_mask[:, :, None, None] * noise_init

    alpha_bar = alpha_bar.to(device)
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noisy_input = observed_mask[:, :, None, None] * stacked_latents + \
                      missing_mask[:, :, None, None] * z

        noise_pred = model(noisy_input, t_batch, observed_mask)

        ab_now = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        pred_x0 = (z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

        dir_zt = torch.sqrt(1 - ab_next) * noise_pred
        z_new = torch.sqrt(ab_next) * pred_x0 + dir_zt

        z = observed_mask[:, :, None, None] * stacked_latents + \
            missing_mask[:, :, None, None] * z_new

    return z


def sample_missing_pattern():
    """
    Sample a realistic missing pattern.
    Options:
    - Single sensor missing
    - Whole device missing
    - Random 1-3 sensors
    """
    pattern_type = random.choice(["single", "device", "random"])

    if pattern_type == "single":
        return [random.choice(SENSOR_NAMES)]

    elif pattern_type == "device":
        device = random.choice(list(DEVICE_GROUPS.keys()))
        return DEVICE_GROUPS[device]

    else:  # random
        n_missing = random.randint(MIN_MISSING, MAX_MISSING)
        return random.sample(SENSOR_NAMES, n_missing)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Training Robust Activity Classifier")
    print(f"Imputation augmentation: {IMPUTE_PROB*100:.0f}% of batches")
    print(f"DDIM steps: {DDIM_STEPS}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Load datasets
    print("Loading datasets...")
    train_dataset = get_combined_labeled_dataset(DATA_ROOTS, "training", normalizer)
    test_dataset = get_combined_labeled_dataset(DATA_ROOTS, "testing", normalizer)

    n_classes = train_dataset.n_classes
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Classes: {n_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Load VAE (frozen)
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Load diffusion (frozen)
    print("Loading diffusion model...")
    diff_ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T = diff_ckpt["T"]
    schedule_type = diff_ckpt["schedule"]
    cfg = diff_ckpt["config"]

    diffusion = create_sensor_diffusion_v2(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout=0.0,
    ).to(DEVICE)
    diffusion.load_state_dict(diff_ckpt["model_state"])
    diffusion.eval()
    for p in diffusion.parameters():
        p.requires_grad = False

    # Load normalization stats
    norm_stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)
    sched = make_schedule(T, schedule_type)

    # Create classifier
    print("Creating classifier...")
    classifier = create_activity_classifier(
        model_type=MODEL_TYPE,
        n_classes=n_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    impute_count = 0
    real_count = 0

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            labels = batch["label"].to(DEVICE)
            B = labels.size(0)

            # Encode with VAE
            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            # Decide: impute or not
            do_impute = random.random() < IMPUTE_PROB

            if do_impute:
                impute_count += 1
                missing = sample_missing_pattern()

                # Normalize for diffusion
                latents_norm = {}
                for name in SENSOR_NAMES:
                    mean = norm_stats[name]["mean"].to(DEVICE)
                    std = norm_stats[name]["std"].to(DEVICE)
                    latents_norm[name] = (latents[name] - mean) / std

                stacked = torch.stack([latents_norm[name] for name in SENSOR_NAMES], dim=1)
                observed_mask = torch.ones(B, len(SENSOR_NAMES), device=DEVICE)
                for name in missing:
                    observed_mask[:, SENSOR_NAMES.index(name)] = 0.0

                with torch.no_grad():
                    imputed = ddim_sample_v2_fast(
                        diffusion, stacked, observed_mask,
                        sched["alpha_bar"], T, DDIM_STEPS,
                    )

                final_latents = {}
                for i, name in enumerate(SENSOR_NAMES):
                    mean = norm_stats[name]["mean"].to(DEVICE)
                    std = norm_stats[name]["std"].to(DEVICE)
                    final_latents[name] = imputed[:, i] * std + mean
            else:
                real_count += 1
                final_latents = latents

            # Classify
            optimizer.zero_grad()
            logits = classifier(final_latents, SENSOR_NAMES)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += B

            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Eval (on real latents only — baseline)
        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
                labels = batch["label"].to(DEVICE)

                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

                logits = classifier(latents, SENSOR_NAMES)
                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Test acc={test_acc:.4f} | Impute/Real: {impute_count}/{real_count}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state': classifier.state_dict(),
                'acc': test_acc,
                'n_classes': n_classes,
                'model_type': MODEL_TYPE,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label,
            }, OUT_DIR / "best_model.pt")

    torch.save({
        'epoch': EPOCHS,
        'model_state': classifier.state_dict(),
        'acc': test_acc,
        'n_classes': n_classes,
        'model_type': MODEL_TYPE,
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
    }, OUT_DIR / "final_model.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Total batches — Imputed: {impute_count}, Real: {real_count}")
    print(f"Saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
