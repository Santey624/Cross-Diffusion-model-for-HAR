# ============================================================
# Train Sensor-Level Joint Diffusion Model
# 7 sensor modalities, cosine schedule, min-SNR loss
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import math

from src.models.sensor_joint_diffusion import create_sensor_diffusion_model, DEFAULT_SENSOR_SPECS


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENTS_DIR = Path("data/sensor_latents")
OUT_DIR = Path("checkpoints/sensor_diffusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion
T = 1000
SCHEDULE = "cosine"

# Model
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_CONV_BLOCKS = 4
NUM_ATTN_BLOCKS = 3
DROPOUT = 0.1

# Training
BATCH_SIZE = 128
EPOCHS = 500
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

# Min-SNR loss weighting
MIN_SNR_GAMMA = 5.0

# Sensor names and device groups
SENSOR_NAMES = list(DEFAULT_SENSOR_SPECS.keys())

DEVICE_GROUPS = {
    "phone": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
    "watch": ["watch_acc", "watch_gyro"],
    "glasses": ["glasses_acc"],
}

SENSOR_TO_DEVICE = {}
for dev, sensors in DEVICE_GROUPS.items():
    for s in sensors:
        SENSOR_TO_DEVICE[s] = dev


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
# HIERARCHICAL CONDITION MASKING
# ============================================================
def mask_conditions(target_name, batch_data):
    """
    Hierarchical masking strategy:
    - 10% drop ALL conditions (for CFG)
    - 20% drop entire device group
    - 20% drop 1-3 random individual sensors
    - 50% full conditioning (all except target)
    """
    r = random.random()

    if r < 0.10:
        # Drop all
        return {k: None for k in SENSOR_NAMES}

    elif r < 0.30:
        # Drop entire device group (not the target's device)
        target_device = SENSOR_TO_DEVICE[target_name]
        other_devices = [d for d in DEVICE_GROUPS if d != target_device]
        if other_devices:
            drop_device = random.choice(other_devices)
            drop_set = set(DEVICE_GROUPS[drop_device])
        else:
            drop_set = set()

        return {
            k: (None if k == target_name or k in drop_set else batch_data[k])
            for k in SENSOR_NAMES
        }

    elif r < 0.50:
        # Drop 1-3 random individual sensors
        available = [k for k in SENSOR_NAMES if k != target_name]
        n_drop = random.randint(1, min(3, len(available)))
        to_drop = set(random.sample(available, n_drop))

        return {
            k: (None if k == target_name or k in to_drop else batch_data[k])
            for k in SENSOR_NAMES
        }

    else:
        # Full conditioning
        return {
            k: (batch_data[k] if k != target_name else None)
            for k in SENSOR_NAMES
        }


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Training Sensor-Level Joint Diffusion Model")
    print(f"Schedule: {SCHEDULE}, T: {T}")
    print(f"Hidden: {HIDDEN_DIM}, Conv: {NUM_CONV_BLOCKS}, Attn: {NUM_ATTN_BLOCKS}")
    print(f"Min-SNR gamma: {MIN_SNR_GAMMA}")
    print(f"Sensors: {len(SENSOR_NAMES)}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading sensor latents...")
    latents = {}
    for name in SENSOR_NAMES:
        path = LATENTS_DIR / f"train_latents_{name}_mu.pt"
        latents[name] = torch.load(path)
        print(f"  {name:15s}: {latents[name].shape}")

    N = latents[SENSOR_NAMES[0]].shape[0]
    print(f"Samples: {N}")

    # Normalize per sensor (mean over batch and time)
    print("\nNormalizing latents...")
    norm_stats = {}
    latents_norm = {}
    for name in SENSOR_NAMES:
        z = latents[name]
        mean = z.mean(dim=(0, 2), keepdim=True)
        std = z.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        latents_norm[name] = (z - mean) / std
        norm_stats[name] = {"mean": mean, "std": std}

    # Save normalization stats
    torch.save(norm_stats, OUT_DIR / "normalization_stats.pt")

    # Create dataset (stack all latents into a single TensorDataset)
    tensor_list = [latents_norm[name] for name in SENSOR_NAMES]
    ds = TensorDataset(*tensor_list)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    pin_memory=True, num_workers=4)

    # Model
    print("\nCreating model...")
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

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch_tensors in pbar:
            # Reconstruct dict from TensorDataset
            batch_data = {
                name: batch_tensors[i].to(DEVICE)
                for i, name in enumerate(SENSOR_NAMES)
            }
            B = batch_tensors[0].shape[0]

            # Random target sensor
            target_name = random.choice(SENSOR_NAMES)
            z0 = batch_data[target_name]

            # Sample timestep and noise
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(z0)
            z_t = sqrt_ab[t].view(-1, 1, 1) * z0 + sqrt_1_ab[t].view(-1, 1, 1) * noise

            # Mask conditions
            conditions = mask_conditions(target_name, batch_data)

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

            pbar.set_postfix(loss=f"{loss.item():.4f}", target=target_name)

        lr_sched.step()
        n_batches = len(dl)
        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
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

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
