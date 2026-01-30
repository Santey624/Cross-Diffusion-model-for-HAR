# ============================================================
# Train Joint Temporal Diffusion v2
# Same 3D Conv1D + CrossAttention architecture
# Fixes: cosine schedule, min-SNR loss, better sampling
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import math

from src.models.joint_diffusion import create_joint_diffusion_model


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENTS_DIR = Path("data/latents")
OUT_DIR = Path("checkpoints/joint_diffusion_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion
T = 1000
SCHEDULE = "cosine"  # Key fix #1: cosine instead of linear

# Model (same architecture that got 1.879)
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_CONV_BLOCKS = 4    # slightly more than before (was 3)
NUM_ATTN_BLOCKS = 3    # slightly more (was 2)
DROPOUT = 0.1

# Training
BATCH_SIZE = 128
EPOCHS = 500           # longer training (was 300)
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

# Min-SNR loss weighting (Key fix #2)
MIN_SNR_GAMMA = 5.0

# Condition masking: simple 10% chance of dropping ALL conditions
CFG_DROP_PROB = 0.1


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
    print(f"\n{'='*60}")
    print("Training Joint Temporal Diffusion v2")
    print(f"Schedule: {SCHEDULE}, T: {T}")
    print(f"Hidden: {HIDDEN_DIM}, Conv: {NUM_CONV_BLOCKS}, Attn: {NUM_ATTN_BLOCKS}")
    print(f"Min-SNR gamma: {MIN_SNR_GAMMA}")
    print(f"CFG drop prob: {CFG_DROP_PROB}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading latents...")
    phone_latents = torch.load(LATENTS_DIR / "train_latents_phone_mu.pt")
    watch_latents = torch.load(LATENTS_DIR / "train_latents_watch_mu.pt")
    glasses_latents = torch.load(LATENTS_DIR / "train_latents_glasses_mu.pt")

    N = phone_latents.shape[0]
    print(f"Phone:   {phone_latents.shape}")
    print(f"Watch:   {watch_latents.shape}")
    print(f"Glasses: {glasses_latents.shape}")
    print(f"Samples: {N}")

    # Normalize per channel (mean over batch and time)
    print("\nNormalizing latents...")
    phone_mean = phone_latents.mean(dim=(0, 2), keepdim=True)
    phone_std = phone_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    phone_norm = (phone_latents - phone_mean) / phone_std

    watch_mean = watch_latents.mean(dim=(0, 2), keepdim=True)
    watch_std = watch_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    watch_norm = (watch_latents - watch_mean) / watch_std

    glasses_mean = glasses_latents.mean(dim=(0, 2), keepdim=True)
    glasses_std = glasses_latents.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    glasses_norm = (glasses_latents - glasses_mean) / glasses_std

    # Save normalization stats
    torch.save({
        'phone': {'mean': phone_mean, 'std': phone_std},
        'watch': {'mean': watch_mean, 'std': watch_std},
        'glasses': {'mean': glasses_mean, 'std': glasses_std},
    }, OUT_DIR / "normalization_stats.pt")

    # Dataset
    ds = TensorDataset(phone_norm, watch_norm, glasses_norm)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                    pin_memory=True, num_workers=4)

    # Model
    print("\nCreating model...")
    model = create_joint_diffusion_model(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_conv_blocks=NUM_CONV_BLOCKS,
        num_attn_blocks=NUM_ATTN_BLOCKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # Optimizer + scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    # Diffusion schedule
    sched = make_schedule(T, SCHEDULE, DEVICE)
    sqrt_ab = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]
    snr = sched["snr"]

    modality_names = ["phone", "watch", "glasses"]
    best_loss = float('inf')

    # ============================================================
    # TRAINING
    # ============================================================
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        per_mod_loss = {k: 0.0 for k in modality_names}
        per_mod_count = {k: 0 for k in modality_names}

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for phone_b, watch_b, glasses_b in pbar:
            phone_b = phone_b.to(DEVICE)
            watch_b = watch_b.to(DEVICE)
            glasses_b = glasses_b.to(DEVICE)
            B = phone_b.shape[0]

            batch_data = {"phone": phone_b, "watch": watch_b, "glasses": glasses_b}

            # Random target
            target_name = random.choice(modality_names)
            z0 = batch_data[target_name]

            # Sample timestep and noise
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(z0)
            z_t = sqrt_ab[t].view(-1, 1, 1) * z0 + sqrt_1_ab[t].view(-1, 1, 1) * noise

            # Build conditions (simple: pass all others, or drop all for CFG)
            drop_all = random.random() < CFG_DROP_PROB
            if drop_all:
                phone_cond = None
                watch_cond = None
                glasses_cond = None
            else:
                phone_cond = phone_b if target_name != "phone" else None
                watch_cond = watch_b if target_name != "watch" else None
                glasses_cond = glasses_b if target_name != "glasses" else None

            # Forward
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                noise_pred = model(
                    target_modality=target_name,
                    z_t=z_t,
                    t=t,
                    phone_latent=phone_cond,
                    watch_latent=watch_cond,
                    glasses_latent=glasses_cond,
                )

                # Min-SNR weighted loss (Key fix #2)
                snr_t = snr[t]
                weight = torch.clamp(snr_t, max=MIN_SNR_GAMMA) / snr_t  # (B,)
                per_sample_loss = ((noise_pred - noise) ** 2).mean(dim=(1, 2))  # (B,)
                loss = (weight * per_sample_loss).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            per_mod_loss[target_name] += loss.item()
            per_mod_count[target_name] += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", target=target_name)

        lr_sched.step()
        n_batches = len(dl)
        avg_loss = epoch_loss / n_batches
        avg_mod = {k: per_mod_loss[k] / max(per_mod_count[k], 1) for k in modality_names}

        if epoch % 10 == 0 or epoch == 1:
            lr = lr_sched.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"P: {avg_mod['phone']:.4f} "
                  f"W: {avg_mod['watch']:.4f} "
                  f"G: {avg_mod['glasses']:.4f} | "
                  f"LR: {lr:.2e}")

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

        # Periodic
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
