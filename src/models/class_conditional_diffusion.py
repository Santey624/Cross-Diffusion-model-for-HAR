# ============================================================
# Class-Conditional Diffusion Model for Sensor Latent Imputation
#
# Generates a plausible sensor latent z given activity class label.
# Operates per-sensor in VAE V2 latent space (D=16, T=64).
#
# Architecture: 1D Conv UNet with FiLM conditioning
#   - Timestep t     → sinusoidal embedding → MLP → scale/shift
#   - Class label c  → embedding → MLP → scale/shift
#   - Combined conditioning via FiLM on each conv block
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal timestep embedding
# ============================================================
def sinusoidal_embedding(t, dim):
    """t: (B,) long → (B, dim) float"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    ).float()
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


# ============================================================
# FiLM conditioning block
# ============================================================
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift per channel."""
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, n_channels * 2)

    def forward(self, x, cond):
        """
        x:    (B, C, T)
        cond: (B, cond_dim)
        """
        gamma, beta = self.proj(cond).chunk(2, dim=1)   # each (B, C)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


# ============================================================
# Residual Conv Block with FiLM conditioning
# ============================================================
class CondResBlock(nn.Module):
    def __init__(self, channels, cond_dim, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.film1 = FiLM(cond_dim, channels)
        self.film2 = FiLM(cond_dim, channels)

    def forward(self, x, cond):
        h = F.silu(self.film1(self.norm1(self.conv1(x)), cond))
        h = self.film2(self.norm2(self.conv2(h)), cond)
        return x + h


# ============================================================
# Main Model
# ============================================================
class ClassConditionalDiffusion(nn.Module):
    """
    Per-sensor class-conditional denoising diffusion model.

    Args:
        latent_dim:   D in VAE latent (B, D, T_lat)
        t_lat:        T in VAE latent
        n_classes:    number of activity classes
        d_model:      internal channel width
        n_blocks:     number of CondResBlocks
        emb_dim:      dimension of timestep + class embeddings
    """

    def __init__(self, latent_dim=16, t_lat=64, n_classes=55,
                 d_model=128, n_blocks=6, emb_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_lat      = t_lat
        self.n_classes  = n_classes
        cond_dim = emb_dim * 2

        # Timestep embedding
        self.t_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.emb_dim = emb_dim

        # Class embedding
        self.class_emb = nn.Embedding(n_classes, emb_dim)

        # Input projection: latent_dim → d_model
        self.in_proj  = nn.Conv1d(latent_dim, d_model, 1)

        # Residual blocks
        self.blocks = nn.ModuleList([
            CondResBlock(d_model, cond_dim) for _ in range(n_blocks)
        ])

        # Output projection: d_model → latent_dim
        self.out_proj = nn.Sequential(
            nn.GroupNorm(min(8, d_model), d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, latent_dim, 1),
        )

        # Initialize output near zero (standard for diffusion)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, z_t, t, class_label):
        """
        Args:
            z_t:         (B, D, T_lat)  — noisy latent
            t:           (B,)           — diffusion timestep (long)
            class_label: (B,)           — activity class index (long)

        Returns:
            noise_pred:  (B, D, T_lat)
        """
        # Build conditioning vector
        t_emb = self.t_emb(sinusoidal_embedding(t, self.emb_dim))   # (B, emb_dim)
        c_emb = self.class_emb(class_label)                          # (B, emb_dim)
        cond  = torch.cat([t_emb, c_emb], dim=1)                     # (B, 2*emb_dim)

        x = self.in_proj(z_t)            # (B, d_model, T_lat)
        for block in self.blocks:
            x = block(x, cond)
        return self.out_proj(x)          # (B, D, T_lat)


def create_class_conditional_diffusion(latent_dim, t_lat, n_classes,
                                        d_model=128, n_blocks=6, emb_dim=128):
    return ClassConditionalDiffusion(
        latent_dim=latent_dim,
        t_lat=t_lat,
        n_classes=n_classes,
        d_model=d_model,
        n_blocks=n_blocks,
        emb_dim=emb_dim,
    )
