# ============================================================
# Train Joint Flat Conditional Diffusion v2
# Single model, modality-specific projections, cosine schedule
# ============================================================

from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.flat_diffusion import (
    JointFlatDenoiser, make_schedule, FLAT_DIMS
)

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENTS_DIR = Path("data/latents")
CHECKPOINT_DIR = Path("checkpoints/flat_diffusion")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Diffusion
T = 1000
SCHEDULE = "cosine"

# Model
HIDDEN_DIM = 1024
NUM_LAYERS = 8
TIME_DIM = 256
DROPOUT = 0.1

# Training
EPOCHS = 500
BATCH_SIZE = 256
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# CFG: probability of dropping ALL conditions
CFG_DROP_PROB = 0.1

# Min-SNR loss weighting
MIN_SNR_GAMMA = 5.0


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("Training Joint Flat Conditional Diffusion v2")
    print(f"Schedule: {SCHEDULE}, T: {T}")
    print(f"Hidden: {HIDDEN_DIM}, Layers: {NUM_LAYERS}")
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
    print(f"Total samples: {N}")

    # Flatten: (N, D, T') → (N, D*T')
    phone_flat = phone_latents.reshape(N, -1)
    watch_flat = watch_latents.reshape(N, -1)
    glasses_flat = glasses_latents.reshape(N, -1)

    # Normalize
    stats = {}
    for name, flat in [("phone", phone_flat), ("watch", watch_flat), ("glasses", glasses_flat)]:
        mean = flat.mean(0)
        std = flat.std(0).clamp(min=1e-6)
        stats[name] = {"mean": mean, "std": std}
        print(f"{name}: flat_dim={flat.shape[1]}, mean={flat.mean():.4f}, std={flat.std():.4f}")

    phone_norm = (phone_flat - stats["phone"]["mean"]) / stats["phone"]["std"]
    watch_norm = (watch_flat - stats["watch"]["mean"]) / stats["watch"]["std"]
    glasses_norm = (glasses_flat - stats["glasses"]["mean"]) / stats["glasses"]["std"]

    # Build schedule
    sched = make_schedule(T, SCHEDULE)
    for k, v in sched.items():
        sched[k] = v.to(DEVICE)

    # Save config
    torch.save({
        "T": T,
        "schedule": SCHEDULE,
        "stats": stats,
        "config": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "time_dim": TIME_DIM,
            "dropout": DROPOUT,
        },
    }, CHECKPOINT_DIR / "config.pt")

    # Dataset
    dataset = TensorDataset(phone_norm, watch_norm, glasses_norm)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = JointFlatDenoiser(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        time_dim=TIME_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    sqrt_ab = sched["sqrt_alpha_bar"]
    sqrt_1_ab = sched["sqrt_one_minus_alpha_bar"]
    snr = sched["snr"]

    modality_names = ["phone", "watch", "glasses"]
    best_loss = float('inf')

    # ============================================================
    # TRAINING
    # ============================================================
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        per_mod_loss = {k: 0.0 for k in modality_names}
        per_mod_count = {k: 0 for k in modality_names}
        n_batches = 0

        for phone_b, watch_b, glasses_b in loader:
            phone_b = phone_b.to(DEVICE)
            watch_b = watch_b.to(DEVICE)
            glasses_b = glasses_b.to(DEVICE)
            B = phone_b.shape[0]

            batch_data = {"phone": phone_b, "watch": watch_b, "glasses": glasses_b}

            # Random target
            target_name = random.choice(modality_names)
            target_data = batch_data[target_name]

            # Build conditions (pass other modalities)
            cond_kwargs = {}
            drop_all = random.random() < CFG_DROP_PROB
            for name in modality_names:
                if name != target_name and not drop_all:
                    cond_kwargs[f"{name}_cond"] = batch_data[name]

            # Diffusion forward
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target_data)
            z_t = sqrt_ab[t].unsqueeze(1) * target_data + sqrt_1_ab[t].unsqueeze(1) * noise

            # Predict noise
            noise_pred = model(z_t, t, target_name, **cond_kwargs)

            # Min-SNR weighted loss
            snr_t = snr[t]
            weight = torch.clamp(snr_t, max=MIN_SNR_GAMMA) / snr_t
            loss = (weight.unsqueeze(1) * (noise_pred - noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            per_mod_loss[target_name] += loss.item()
            per_mod_count[target_name] += 1
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_per_mod = {k: (per_mod_loss[k] / max(per_mod_count[k], 1))
                       for k in modality_names}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"P: {avg_per_mod['phone']:.4f} "
                  f"W: {avg_per_mod['watch']:.4f} "
                  f"G: {avg_per_mod['glasses']:.4f} | "
                  f"LR: {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, CHECKPOINT_DIR / "best_model.pt")

        if (epoch + 1) % 100 == 0:
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, CHECKPOINT_DIR / f"epoch_{epoch+1:03d}.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
