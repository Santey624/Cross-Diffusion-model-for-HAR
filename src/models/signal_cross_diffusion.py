# ============================================================
# Cross-Sensor Conditional Diffusion on Raw Signals (no VAE)
#
# p(x_missing | x_observed) — directly at signal level
#
# Architecture: same as V3 latent diffusion but operates on
# raw sensor signals interpolated to T_COMMON=256.
# All sensors have 3 channels (acc/gyro/grav).
#
# Key: missing sensors (queries) cross-attend to observed
# sensors (keys/values) — same mechanism as V3, more signal.
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

T_COMMON = 256   # all sensors interpolated to this for cross-attention


# ============================================================
# Building blocks (reused from V3)
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return F.pad(emb, (0, self.dim % 2))


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


class ConditionalCrossSensorAttn(nn.Module):
    """Missing sensors (queries) attend ONLY to observed sensors (keys/values)."""
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
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.film  = nn.Linear(d_model, d_model * 2)
        self.t_attn  = TemporalAttention(d_model, num_heads, dropout)
        self.cs_attn = ConditionalCrossSensorAttn(d_model, num_heads, dropout)
        self.gate    = nn.Linear(d_model, d_model)
        self.filt    = nn.Linear(d_model, d_model)
        self.res     = nn.Linear(d_model, d_model)
        self.skip    = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)
        self.act     = nn.SiLU()

    def forward(self, x, t_emb, observed_mask):
        """x: (B, K, T, d) | t_emb: (B, d_model)"""
        B, K, T, d = x.shape
        h = x.reshape(B * K, T, d).permute(0, 2, 1)
        h = self.conv1(h).permute(0, 2, 1).reshape(B, K, T, d)
        h = self.norm1(h)
        sc, sh = self.film(t_emb).chunk(2, dim=1)
        h = h * (1 + sc[:, None, None, :]) + sh[:, None, None, :]
        h = self.act(self.drop(h))
        h = self.t_attn(h)
        h = self.cs_attn(h, observed_mask)
        h = torch.sigmoid(self.gate(h)) * torch.tanh(self.filt(h))
        return (x + self.res(h)) / math.sqrt(2.0), self.skip(h)


# ============================================================
# Main model
# ============================================================
class SignalCrossDiffusion(nn.Module):
    """
    Cross-sensor conditional diffusion on raw signals.

    p(x_missing | x_observed) at signal level.
    All sensors interpolated to T_COMMON for cross-attention.

    Args:
        n_sensors:   number of sensor types (7)
        in_channels: signal channels (3)
        d_model:     internal dimension
        num_heads:   attention heads
        num_blocks:  number of ResBlocks
    """
    def __init__(self, n_sensors=7, in_channels=3, d_model=128,
                 t_dim=128, num_heads=4, num_blocks=6, dropout=0.1):
        super().__init__()
        self.n_sensors   = n_sensors
        self.in_channels = in_channels
        self.d_model     = d_model

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj  = nn.Sequential(
            nn.Linear(t_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        # Learnable sensor-type embedding (added to signal features)
        self.sensor_emb = nn.Embedding(n_sensors, d_model)

        # Input projection: (in_channels + 1 mask) → d_model
        self.in_proj = nn.Linear(in_channels + 1, d_model)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(d_model, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_signals, t, observed_mask):
        """
        noisy_signals: (B, K, C, T_COMMON) — observed=clean, missing=noisy
        t:             (B,)
        observed_mask: (B, K) — 1=observed, 0=missing

        Returns: noise_pred (B, K, C, T_COMMON)
        """
        B, K, C, T = noisy_signals.shape
        device = noisy_signals.device

        t_emb = self.t_proj(self.t_embed(t))          # (B, d_model)

        # (B, K, C, T) → (B, K, T, C)
        x = noisy_signals.permute(0, 1, 3, 2)

        # Append mask channel: (B, K, T, C+1)
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
            h, skip = block(h, t_emb, observed_mask)
            skip_sum = skip_sum + skip

        h = skip_sum / math.sqrt(len(self.blocks))
        h = self.out_norm(h)
        out = self.out_proj(h)                        # (B, K, T, C)
        return out.permute(0, 1, 3, 2)               # (B, K, C, T)


def create_signal_cross_diffusion(n_sensors=7, in_channels=3, d_model=128,
                                   num_heads=4, num_blocks=6, dropout=0.1):
    return SignalCrossDiffusion(
        n_sensors=n_sensors, in_channels=in_channels,
        d_model=d_model, t_dim=128,
        num_heads=num_heads, num_blocks=num_blocks, dropout=dropout,
    )
