# ============================================================
# Train Sensor Guidance Classifier (Regressor)
# Given noisy z_t of one sensor, predict clean latents of all others
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import math

from src.models.sensor_guidance_classifier import create_guidance_classifier
from src.models.sensor_vae import SENSOR_NAMES


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENTS_DIR = Path("data/sensor_latents")
OUT_DIR = Path("checkpoints/guidance_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use same normalization as diffusion
NORM_STATS_PATH = Path("checkpoints/uncond_diffusion/normalization_stats.pt")

# Diffusion schedule (must match the unconditional DM)
T = 1000
SCHEDULE = "cosine"

# Model
HIDDEN_DIM = 128
NUM_CONV_BLOCKS = 4
DROPOUT = 0.1

# Training
BATCH_SIZE = 128
EPOCHS = 200
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

SENSOR_NAME_TO_IDX = {name: i for i, name in enumerate(SENSOR_NAMES)}


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
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Training Sensor Guidance Classifier")
    print(f"Hidden: {HIDDEN_DIM}, Conv: {NUM_CONV_BLOCKS}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading sensor latents...")
    latents = {}
    for name in SENSOR_NAMES:
        path = LATENTS_DIR / f"train_latents_{name}_mu.pt"
        latents[name] = torch.load(path)
        print(f"  {name:15s}: {latents[name].shape}")

    # Load normalization stats (same as unconditional DM)
    print(f"\nLoading normalization stats from {NORM_STATS_PATH}...")
    norm_stats = torch.load(NORM_STATS_PATH, map_location=DEVICE)

    latents_norm = {}
    for name in SENSOR_NAMES:
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        latents_norm[name] = (latents[name] - mean.cpu()) / std.cpu()

    # Dataset
    tensor_list = [latents_norm[name] for name in SENSOR_NAMES]
    ds = TensorDataset(*tensor_list)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    pin_memory=True, num_workers=4)

    # Model
    print("\nCreating classifier...")
    model = create_guidance_classifier(
        hidden_dim=HIDDEN_DIM,
        num_conv_blocks=NUM_CONV_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Diffusion schedule (for adding noise to inputs)
    sched = make_schedule(T, SCHEDULE, DEVICE)
    sqrt_ab = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]

    best_loss = float('inf')

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch_tensors in pbar:
            batch_data = {
                name: batch_tensors[i].to(DEVICE)
                for i, name in enumerate(SENSOR_NAMES)
            }
            B = batch_tensors[0].shape[0]

            # Random target sensor
            target_name = random.choice(SENSOR_NAMES)
            sensor_idx = SENSOR_NAME_TO_IDX[target_name]
            z0 = batch_data[target_name]

            # Add noise to target (simulate what the DM sees)
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(z0)
            z_t = sqrt_ab[t].view(-1, 1, 1) * z0 + sqrt_1_ab[t].view(-1, 1, 1) * noise

            # Forward: predict all sensor latents from noisy target
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                preds = model(z_t=z_t, t=t, sensor_idx=sensor_idx)
                # preds: (B, 7, D, T)

                # Loss: MSE on all sensors EXCEPT the target
                loss = 0.0
                count = 0
                for i, name in enumerate(SENSOR_NAMES):
                    if name == target_name:
                        continue
                    pred_i = preds[:, i]  # (B, D, T)
                    gt_i = batch_data[name]  # (B, D, T)
                    loss = loss + F.mse_loss(pred_i, gt_i)
                    count += 1
                loss = loss / count

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", sensor=target_name)

        lr_sched.step()
        avg_loss = epoch_loss / len(dl)

        if epoch % 10 == 0 or epoch == 1:
            lr = lr_sched.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_conv_blocks': NUM_CONV_BLOCKS,
                    'dropout': DROPOUT,
                },
            }, OUT_DIR / "best_model.pt")

        if epoch % 50 == 0 or epoch == EPOCHS:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_conv_blocks': NUM_CONV_BLOCKS,
                    'dropout': DROPOUT,
                },
            }, OUT_DIR / f"epoch_{epoch:03d}.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
