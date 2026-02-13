# ============================================================
# Train Sensor-Level Joint Diffusion Model WITH DATA AUGMENTATION
# Online augmentation: Raw data -> Augment -> VAE encode -> Diffusion train
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import random
import math

from src.models.sensor_joint_diffusion import create_sensor_diffusion_model
from src.models.sensor_vae import SensorMultiModalVAE, SENSOR_NAMES
from src.data.cogage_sensor_dataset import CogAgeSensorDataset
from src.data.sensor_normalizer import SensorNormalizer
from src.data.sensor_augmentation import create_sensor_augmentation, MixupAugmentation


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/sensor_vae_best.pt"
NORMALIZER_PATH = "data/sensor_normalizer.npz"
OUT_DIR = Path("checkpoints/sensor_diffusion_aug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOTS = {
    "blho": "data/cogage/python/arrays/blho",
    "bbh": "data/cogage/python/arrays/bbh",
    "state": "data/cogage/python/arrays/state",
}

# Diffusion
T = 1000
SCHEDULE = "cosine"

# Model
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_CONV_BLOCKS = 6
NUM_ATTN_BLOCKS = 2
DROPOUT = 0.1

# Training
BATCH_SIZE = 64  # Smaller because of VAE encoding
EPOCHS = 800  # More epochs to see augmented data
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

# Min-SNR loss weighting
MIN_SNR_GAMMA = 5.0

# Augmentation
AUG_MODE = "default"  # "light", "default", "strong"
USE_LATENT_MIXUP = True
MIXUP_ALPHA = 0.2


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


def make_schedule(T, schedule_type, device):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T).to(device)
    else:
        betas = torch.linspace(1e-4, 0.02, T, device=device)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "snr": alpha_bar / (1.0 - alpha_bar),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*70}")
    print("Training Sensor Diffusion WITH DATA AUGMENTATION")
    print(f"{'='*70}")
    print(f"Schedule: {SCHEDULE}, T: {T}")
    print(f"Hidden: {HIDDEN_DIM}, Conv: {NUM_CONV_BLOCKS}, Attn: {NUM_ATTN_BLOCKS}")
    print(f"Augmentation: {AUG_MODE}, Latent Mixup: {USE_LATENT_MIXUP}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")

    # Load normalizer
    normalizer = SensorNormalizer.load(NORMALIZER_PATH)

    # Create augmentation
    augmentation = create_sensor_augmentation(mode=AUG_MODE, enabled=True)
    mixup = MixupAugmentation(alpha=MIXUP_ALPHA, enabled=USE_LATENT_MIXUP)

    # Load datasets WITH augmentation
    print("Loading datasets with augmentation...")
    train_dataset = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"], "training", normalizer, augmentation),
        CogAgeSensorDataset(DATA_ROOTS["bbh"], "training", normalizer, augmentation),
        CogAgeSensorDataset(DATA_ROOTS["state"], "training", normalizer, augmentation),
    ])
    print(f"Train samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )

    # Load VAE (frozen)
    print("\nLoading VAE...")
    vae = SensorMultiModalVAE().to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"VAE from epoch {vae_ckpt.get('epoch', '?')}")

    # Compute normalization stats from first pass (no augmentation for stats)
    print("\nComputing latent normalization stats...")
    norm_dataset = ConcatDataset([
        CogAgeSensorDataset(DATA_ROOTS["blho"], "training", normalizer, None),
        CogAgeSensorDataset(DATA_ROOTS["bbh"], "training", normalizer, None),
        CogAgeSensorDataset(DATA_ROOTS["state"], "training", normalizer, None),
    ])
    norm_loader = DataLoader(norm_dataset, batch_size=128, shuffle=False, num_workers=4)

    latent_sum = {k: 0.0 for k in SENSOR_NAMES}
    latent_sq_sum = {k: 0.0 for k in SENSOR_NAMES}
    n_total = 0

    with torch.no_grad():
        for batch in tqdm(norm_loader, desc="Computing stats"):
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            outputs = vae(sensor_data)
            B = sensor_data[SENSOR_NAMES[0]].size(0)
            n_total += B

            for name in SENSOR_NAMES:
                mu = outputs[name]["mu"]  # (B, D, T)
                latent_sum[name] += mu.sum(dim=(0, 2)).cpu()
                latent_sq_sum[name] += (mu ** 2).sum(dim=(0, 2)).cpu()

    # Compute mean/std per channel
    norm_stats = {}
    for name in SENSOR_NAMES:
        T_len = 32  # All latents have T_SHARED=32
        mean = latent_sum[name] / (n_total * T_len)
        var = latent_sq_sum[name] / (n_total * T_len) - mean ** 2
        std = torch.sqrt(var.clamp_min(1e-6))
        norm_stats[name] = {
            "mean": mean.view(1, -1, 1),
            "std": std.view(1, -1, 1),
        }
        print(f"  {name:15s}: mean={mean.mean():.4f}, std={std.mean():.4f}")

    torch.save(norm_stats, OUT_DIR / "normalization_stats.pt")

    # Create diffusion model
    print("\nCreating diffusion model...")
    model = create_sensor_diffusion_model(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_conv_blocks=NUM_CONV_BLOCKS,
        num_attn_blocks=NUM_ATTN_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Diffusion schedule
    sched = make_schedule(T, SCHEDULE, DEVICE)
    sqrt_ab = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]
    snr = sched["snr"]

    best_loss = float('inf')

    # Training
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        per_sensor_loss = {k: 0.0 for k in SENSOR_NAMES}
        per_sensor_count = {k: 0 for k in SENSOR_NAMES}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            sensor_data = {k: batch[k].to(DEVICE) for k in SENSOR_NAMES}
            B = sensor_data[SENSOR_NAMES[0]].size(0)

            # Encode with VAE (frozen)
            with torch.no_grad():
                outputs = vae(sensor_data)
                latents = {k: outputs[k]["mu"] for k in SENSOR_NAMES}

            # Normalize latents
            latents_norm = {}
            for name in SENSOR_NAMES:
                mean = norm_stats[name]["mean"].to(DEVICE)
                std = norm_stats[name]["std"].to(DEVICE)
                latents_norm[name] = (latents[name] - mean) / std

            # Optional: Latent-space mixup
            if USE_LATENT_MIXUP and random.random() < 0.5:
                latents_norm, _ = mixup(latents_norm)

            # Random target sensor
            target_name = random.choice(SENSOR_NAMES)
            z0 = latents_norm[target_name]

            # Sample timestep and noise
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(z0)
            z_t = sqrt_ab[t].view(-1, 1, 1) * z0 + sqrt_1_ab[t].view(-1, 1, 1) * noise

            # Build conditions (all except target)
            conditions = {
                k: (latents_norm[k] if k != target_name else None)
                for k in SENSOR_NAMES
            }

            # Forward
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                noise_pred = model(
                    target_modality=target_name,
                    z_t=z_t,
                    t=t,
                    conditions=conditions,
                )

                # Min-SNR weighted loss
                snr_t = snr[t]
                weight = torch.clamp(snr_t, max=MIN_SNR_GAMMA) / snr_t
                per_sample_loss = ((noise_pred - noise) ** 2).mean(dim=(1, 2))
                loss = (weight * per_sample_loss).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            per_sensor_loss[target_name] += loss.item()
            per_sensor_count[target_name] += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        lr_sched.step()
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches

        if epoch % 20 == 0 or epoch == 1:
            lr = lr_sched.get_last_lr()[0]
            sensor_losses = " | ".join(
                f"{k[:6]}:{per_sensor_loss[k] / max(per_sensor_count[k], 1):.4f}"
                for k in SENSOR_NAMES
            )
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | {sensor_losses} | LR: {lr:.2e}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'T': T,
                'schedule': SCHEDULE,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_heads': NUM_HEADS,
                    'num_conv_blocks': NUM_CONV_BLOCKS,
                    'num_attn_blocks': NUM_ATTN_BLOCKS,
                    'dropout': DROPOUT,
                },
            }, OUT_DIR / "best_model.pt")

        # Periodic save
        if epoch % 100 == 0 or epoch == EPOCHS:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'T': T,
                'schedule': SCHEDULE,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_heads': NUM_HEADS,
                    'num_conv_blocks': NUM_CONV_BLOCKS,
                    'num_attn_blocks': NUM_ATTN_BLOCKS,
                    'dropout': DROPOUT,
                },
            }, OUT_DIR / f"epoch_{epoch:03d}.pt")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE - Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {OUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
