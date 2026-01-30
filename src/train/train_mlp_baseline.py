# ============================================================
# MLP Baseline: Direct mapping condition latents → target latents
# Purpose: Diagnose if the cross-modal prediction task is feasible
# ============================================================

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENTS_DIR = Path("data/latents")
CHECKPOINT_DIR = Path("checkpoints/mlp_baseline")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Target modality to impute
TARGET = "phone"  # Change to "watch" or "glasses" as needed

EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# MLP Model
# ============================================================
class LatentMLP(nn.Module):
    """Flatten temporal latents and map condition → target."""

    def __init__(self, cond_dim, target_dim, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Conv1D Model (preserves temporal structure)
# ============================================================
class LatentConvMapper(nn.Module):
    """Map condition latents to target latents preserving temporal info."""

    def __init__(self, cond_channels, target_channels, target_seq_len, hidden=256):
        super().__init__()
        self.target_channels = target_channels
        self.target_seq_len = target_seq_len

        # Encode conditions (flatten all to single sequence)
        self.cond_encoder = nn.Sequential(
            nn.Conv1d(cond_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )

        # Adaptive pooling to target length
        self.adapt = nn.AdaptiveAvgPool1d(target_seq_len)

        # Decode to target
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, target_channels, 3, padding=1),
        )

    def forward(self, cond):
        # cond: (B, C_cond, T_cond)
        h = self.cond_encoder(cond)
        h = self.adapt(h)
        return self.decoder(h)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"MLP Baseline Training")
    print(f"Target: {TARGET}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Load latents
    print("Loading latents...")
    phone_latents = torch.load(LATENTS_DIR / "train_latents_phone_mu.pt")    # (N, 32, 100)
    watch_latents = torch.load(LATENTS_DIR / "train_latents_watch_mu.pt")    # (N, 32, 34)
    glasses_latents = torch.load(LATENTS_DIR / "train_latents_glasses_mu.pt")  # (N, 16, 10)

    N = phone_latents.shape[0]
    print(f"Phone:   {phone_latents.shape}")
    print(f"Watch:   {watch_latents.shape}")
    print(f"Glasses: {glasses_latents.shape}")
    print(f"Samples: {N}")

    # Latent statistics
    print(f"\n--- Latent Statistics ---")
    for name, lat in [("Phone", phone_latents), ("Watch", watch_latents), ("Glasses", glasses_latents)]:
        print(f"{name}: mean={lat.mean():.4f}, std={lat.std():.4f}, "
              f"min={lat.min():.4f}, max={lat.max():.4f}")

    # Split train/val
    n_train = int(0.9 * N)
    indices = torch.randperm(N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Setup target and conditions
    if TARGET == "phone":
        target_all = phone_latents       # (N, 32, 100)
        # Concat watch + glasses along channel dim, pad to same temporal length
        cond_modalities = [("watch", watch_latents), ("glasses", glasses_latents)]
        target_channels = 32
        target_seq_len = 100
    elif TARGET == "watch":
        target_all = watch_latents
        cond_modalities = [("phone", phone_latents), ("glasses", glasses_latents)]
        target_channels = 32
        target_seq_len = 34
    elif TARGET == "glasses":
        target_all = glasses_latents
        cond_modalities = [("phone", phone_latents), ("watch", watch_latents)]
        target_channels = 16
        target_seq_len = 10

    # ---- Approach 1: Flatten MLP ----
    print(f"\n{'='*60}")
    print("Approach 1: Flatten MLP")
    print(f"{'='*60}")

    # Flatten all latents
    target_flat = target_all.reshape(N, -1)  # (N, D*T)
    target_dim = target_flat.shape[1]

    cond_parts = []
    for name, lat in cond_modalities:
        cond_parts.append(lat.reshape(N, -1))
    cond_flat = torch.cat(cond_parts, dim=1)  # (N, sum of D*T)
    cond_dim = cond_flat.shape[1]

    print(f"Condition dim (flat): {cond_dim}")
    print(f"Target dim (flat):    {target_dim}")

    # Normalize
    cond_mean = cond_flat[train_idx].mean(0)
    cond_std = cond_flat[train_idx].std(0).clamp(min=1e-6)
    target_mean = target_flat[train_idx].mean(0)
    target_std = target_flat[train_idx].std(0).clamp(min=1e-6)

    cond_norm = (cond_flat - cond_mean) / cond_std
    target_norm = (target_flat - target_mean) / target_std

    train_dataset = TensorDataset(cond_norm[train_idx], target_norm[train_idx])
    val_dataset = TensorDataset(cond_norm[val_idx], target_norm[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    mlp = LatentMLP(cond_dim, target_dim, hidden=1024).to(DEVICE)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP parameters: {n_params:,}")

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Train
        mlp.train()
        train_loss = 0
        for cond_b, target_b in train_loader:
            cond_b, target_b = cond_b.to(DEVICE), target_b.to(DEVICE)
            pred = mlp(cond_b)
            loss = F.mse_loss(pred, target_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * cond_b.shape[0]
        train_loss /= len(train_dataset)
        scheduler.step()

        # Val
        mlp.eval()
        val_loss = 0
        with torch.no_grad():
            for cond_b, target_b in val_loader:
                cond_b, target_b = cond_b.to(DEVICE), target_b.to(DEVICE)
                pred = mlp(cond_b)
                loss = F.mse_loss(pred, target_b)
                val_loss += loss.item() * cond_b.shape[0]
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": mlp.state_dict(),
                "cond_mean": cond_mean, "cond_std": cond_std,
                "target_mean": target_mean, "target_std": target_std,
                "target": TARGET,
                "val_loss": best_val_loss,
            }, CHECKPOINT_DIR / f"mlp_{TARGET}_best.pt")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print(f"\nBest MLP Val Loss (normalized): {best_val_loss:.6f}")

    # Evaluate in original scale
    mlp.eval()
    with torch.no_grad():
        val_cond = cond_norm[val_idx].to(DEVICE)
        val_target = target_flat[val_idx].to(DEVICE)

        pred_norm = mlp(val_cond)
        # Denormalize
        pred = pred_norm * target_std.to(DEVICE) + target_mean.to(DEVICE)

        latent_mse = F.mse_loss(pred, val_target).item()

    print(f"\n{'='*60}")
    print(f"MLP RESULTS (Original Scale):")
    print(f"  Latent MSE: {latent_mse:.6f}")
    print(f"{'='*60}")

    # ---- Approach 2: Conv1D Mapper ----
    print(f"\n{'='*60}")
    print("Approach 2: Conv1D Mapper (temporal)")
    print(f"{'='*60}")

    # Pad and concat conditions along channel dim
    max_seq_len = max(lat.shape[2] for _, lat in cond_modalities)
    cond_padded_parts = []
    for name, lat in cond_modalities:
        if lat.shape[2] < max_seq_len:
            padded = F.interpolate(lat, size=max_seq_len, mode='linear', align_corners=False)
        else:
            padded = lat
        cond_padded_parts.append(padded)
    cond_temporal = torch.cat(cond_padded_parts, dim=1)  # (N, C_total, T_max)

    cond_channels = cond_temporal.shape[1]
    print(f"Condition: ({cond_channels} channels, {max_seq_len} time)")
    print(f"Target: ({target_channels} channels, {target_seq_len} time)")

    # Normalize per-channel
    cond_t_mean = cond_temporal[train_idx].mean(dim=(0, 2), keepdim=True)
    cond_t_std = cond_temporal[train_idx].std(dim=(0, 2), keepdim=True).clamp(min=1e-6)
    target_t_mean = target_all[train_idx].mean(dim=(0, 2), keepdim=True)
    target_t_std = target_all[train_idx].std(dim=(0, 2), keepdim=True).clamp(min=1e-6)

    cond_t_norm = (cond_temporal - cond_t_mean) / cond_t_std
    target_t_norm = (target_all - target_t_mean) / target_t_std

    train_dataset2 = TensorDataset(cond_t_norm[train_idx], target_t_norm[train_idx])
    val_dataset2 = TensorDataset(cond_t_norm[val_idx], target_t_norm[val_idx])
    train_loader2 = DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader2 = DataLoader(val_dataset2, batch_size=BATCH_SIZE)

    conv_model = LatentConvMapper(cond_channels, target_channels, target_seq_len, hidden=256).to(DEVICE)
    optimizer2 = torch.optim.AdamW(conv_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)

    n_params2 = sum(p.numel() for p in conv_model.parameters())
    print(f"Conv parameters: {n_params2:,}")

    best_val_loss2 = float('inf')

    for epoch in range(EPOCHS):
        conv_model.train()
        train_loss = 0
        for cond_b, target_b in train_loader2:
            cond_b, target_b = cond_b.to(DEVICE), target_b.to(DEVICE)
            pred = conv_model(cond_b)
            loss = F.mse_loss(pred, target_b)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            train_loss += loss.item() * cond_b.shape[0]
        train_loss /= len(train_dataset2)
        scheduler2.step()

        conv_model.eval()
        val_loss = 0
        with torch.no_grad():
            for cond_b, target_b in val_loader2:
                cond_b, target_b = cond_b.to(DEVICE), target_b.to(DEVICE)
                pred = conv_model(cond_b)
                loss = F.mse_loss(pred, target_b)
                val_loss += loss.item() * cond_b.shape[0]
        val_loss /= len(val_dataset2)

        if val_loss < best_val_loss2:
            best_val_loss2 = val_loss
            torch.save({
                "model_state": conv_model.state_dict(),
                "cond_t_mean": cond_t_mean, "cond_t_std": cond_t_std,
                "target_t_mean": target_t_mean, "target_t_std": target_t_std,
                "target": TARGET,
                "val_loss": best_val_loss2,
            }, CHECKPOINT_DIR / f"conv_{TARGET}_best.pt")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Best: {best_val_loss2:.6f}")

    # Evaluate Conv in original scale
    conv_model.eval()
    with torch.no_grad():
        val_cond = cond_t_norm[val_idx].to(DEVICE)
        val_target = target_all[val_idx].to(DEVICE)

        pred_norm = conv_model(val_cond)
        pred = pred_norm * target_t_std.to(DEVICE) + target_t_mean.to(DEVICE)

        conv_latent_mse = F.mse_loss(pred, val_target).item()

    print(f"\n{'='*60}")
    print(f"Conv1D RESULTS (Original Scale):")
    print(f"  Latent MSE: {conv_latent_mse:.6f}")
    print(f"{'='*60}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY - Target: {TARGET}")
    print(f"{'='*60}")
    print(f"  MLP Baseline Latent MSE:       {latent_mse:.6f}")
    print(f"  Conv1D Baseline Latent MSE:    {conv_latent_mse:.6f}")
    print(f"  Joint Diffusion Latent MSE:    1.879 (from eval)")
    print(f"  VAE Reconstruction MSE:        0.006 (from eval)")
    print(f"{'='*60}")
    print(f"\nIf MLP/Conv MSE << Diffusion MSE:")
    print(f"  → Diffusion training/sampling is broken")
    print(f"If MLP/Conv MSE ≈ Diffusion MSE:")
    print(f"  → Cross-modal prediction itself is hard")
    print(f"If MLP/Conv MSE >> Diffusion MSE:")
    print(f"  → Diffusion is doing well, task is just hard")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
