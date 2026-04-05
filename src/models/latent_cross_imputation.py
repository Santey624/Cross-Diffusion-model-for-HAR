# ============================================================
# Latent Cross-Sensor Imputation Network
#
# Operates on VAE latents (B, K, D, T_SHARED).
# Direct MSE regression in latent space, trained jointly
# with frozen VAE decoder via signal reconstruction loss.
#
# No diffusion, no noise schedule — single forward pass.
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, K, T, d)"""
        B, K, T, d = x.shape
        xf = x.reshape(B * K, T, d)
        out, _ = self.attn(xf, xf, xf)
        xf = self.norm(xf + self.drop(out))
        return xf.reshape(B, K, T, d)


class CrossSensorAttn(nn.Module):
    """Missing sensors (queries) attend to observed sensors (keys/values)."""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, observed_mask):
        """x: (B, K, T, d) | observed_mask: (B, K)"""
        B, K, T, d = x.shape
        xf       = x.permute(0, 2, 1, 3).reshape(B * T, K, d)
        key_mask = (observed_mask == 0)[:, None, :].expand(B, T, K).reshape(B * T, K)
        out, _   = self.attn(xf, xf, xf, key_padding_mask=key_mask)
        xf = self.norm(xf + self.drop(out))
        return xf.reshape(B, T, K, d).permute(0, 2, 1, 3)


class ResBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.conv    = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm    = nn.LayerNorm(d_model)
        self.t_attn  = TemporalAttention(d_model, num_heads, dropout)
        self.cs_attn = CrossSensorAttn(d_model, num_heads, dropout)
        self.gate    = nn.Linear(d_model, d_model)
        self.filt    = nn.Linear(d_model, d_model)
        self.res     = nn.Linear(d_model, d_model)
        self.skip    = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)
        self.act     = nn.SiLU()

    def forward(self, x, observed_mask):
        """x: (B, K, T, d)"""
        B, K, T, d = x.shape
        h = x.reshape(B * K, T, d).permute(0, 2, 1)
        h = self.conv(h).permute(0, 2, 1).reshape(B, K, T, d)
        h = self.act(self.drop(self.norm(h)))
        h = self.t_attn(h)
        h = self.cs_attn(h, observed_mask)
        h = torch.sigmoid(self.gate(h)) * torch.tanh(self.filt(h))
        return (x + self.res(h)) / math.sqrt(2.0), self.skip(h)


class LatentCrossImputation(nn.Module):
    """
    Cross-sensor imputation in VAE latent space.

    Given observed sensor latents (B, K, D, T_SHARED), predicts
    missing sensor latents. Trained via signal reconstruction loss
    with a frozen VAE decoder.

    Args:
        n_sensors:  number of sensors (7)
        latent_dim: VAE latent dimension D (8)
        t_shared:   VAE latent time steps T_SHARED (32)
        d_model:    internal attention dimension
        num_heads:  attention heads
        num_blocks: number of ResBlocks
    """
    def __init__(self, n_sensors=7, latent_dim=8, t_shared=32,
                 d_model=128, num_heads=4, num_blocks=6, dropout=0.1):
        super().__init__()
        self.n_sensors  = n_sensors
        self.latent_dim = latent_dim
        self.t_shared   = t_shared
        self.d_model    = d_model

        # Learnable sensor-type embedding
        self.sensor_emb = nn.Embedding(n_sensors, d_model)

        # Input projection: (latent_dim + 1 mask) → d_model
        self.in_proj = nn.Linear(latent_dim + 1, d_model)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(d_model, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, latents, observed_mask):
        """
        latents:       (B, K, D, T_SHARED) — observed=real latent, missing=zeros
        observed_mask: (B, K) — 1=observed, 0=missing

        Returns: pred (B, K, D, T_SHARED)
        """
        B, K, D, T = latents.shape
        device = latents.device

        # (B, K, D, T) → (B, K, T, D)
        x = latents.permute(0, 1, 3, 2)

        # Append mask channel: (B, K, T, D+1)
        mask_ch = observed_mask[:, :, None, None].expand(B, K, T, 1)
        x = torch.cat([x, mask_ch], dim=-1)

        # Project to d_model
        h = self.in_proj(x)                           # (B, K, T, d_model)

        # Add sensor-type embedding
        sidx = torch.arange(K, device=device)
        h = h + self.sensor_emb(sidx)[None, :, None, :]

        # Blocks
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, observed_mask)
            skip_sum = skip_sum + skip

        h = skip_sum / math.sqrt(len(self.blocks))
        h = self.out_norm(h)
        out = self.out_proj(h)                        # (B, K, T, D)
        return out.permute(0, 1, 3, 2)               # (B, K, D, T)


def create_latent_cross_imputation(n_sensors=7, latent_dim=8, t_shared=32,
                                    d_model=128, num_heads=4, num_blocks=6,
                                    dropout=0.1):
    return LatentCrossImputation(
        n_sensors=n_sensors, latent_dim=latent_dim, t_shared=t_shared,
        d_model=d_model, num_heads=num_heads,
        num_blocks=num_blocks, dropout=dropout,
    )
