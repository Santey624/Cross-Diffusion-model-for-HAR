# ============================================================
# Latent Cross-Sensor Diffusion with Reconstruction Loss
#
# p(z_missing | z_observed) in VAE V2 latent space.
# Trained with: noise_pred_loss + λ * signal_recon_loss
#
# During training: z0 is recovered from noise prediction (differentiable),
# decoded by frozen VAE decoder, MSE compared to original signal.
# During inference: standard DDIM sampling.
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.norm1   = nn.LayerNorm(d_model)
        self.film    = nn.Linear(d_model, d_model * 2)
        self.t_attn  = TemporalAttention(d_model, num_heads, dropout)
        self.cs_attn = CrossSensorAttn(d_model, num_heads, dropout)
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
        h = self.conv(h).permute(0, 2, 1).reshape(B, K, T, d)
        h = self.norm1(h)
        sc, sh = self.film(t_emb).chunk(2, dim=1)
        h = h * (1 + sc[:, None, None, :]) + sh[:, None, None, :]
        h = self.act(self.drop(h))
        h = self.t_attn(h)
        h = self.cs_attn(h, observed_mask)
        h = torch.sigmoid(self.gate(h)) * torch.tanh(self.filt(h))
        return (x + self.res(h)) / math.sqrt(2.0), self.skip(h)


class LatentCrossDiffusionRecon(nn.Module):
    """
    Cross-sensor conditional diffusion in VAE latent space.
    Trained with noise_pred_loss + lambda * signal_recon_loss.

    Latent shape: (B, K, D, T_SHARED) — K sensors, D latent dim, T_SHARED time steps.
    """
    def __init__(self, n_sensors=7, latent_dim=16, t_shared=64,
                 d_model=128, t_dim=128, num_heads=4, num_blocks=6, dropout=0.1):
        super().__init__()
        self.n_sensors  = n_sensors
        self.latent_dim = latent_dim

        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj  = nn.Sequential(
            nn.Linear(t_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        self.sensor_emb = nn.Embedding(n_sensors, d_model)

        # Input: latent_dim + 1 mask channel → d_model
        self.in_proj = nn.Linear(latent_dim + 1, d_model)

        self.blocks = nn.ModuleList([
            ResBlock(d_model, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_latents, t, observed_mask):
        """
        noisy_latents: (B, K, D, T_SHARED) — observed=clean, missing=noisy
        t:             (B,)
        observed_mask: (B, K) — 1=observed, 0=missing

        Returns: noise_pred (B, K, D, T_SHARED)
        """
        B, K, D, T = noisy_latents.shape
        device = noisy_latents.device

        t_emb = self.t_proj(self.t_embed(t))          # (B, d_model)

        # (B, K, D, T) → (B, K, T, D)
        x = noisy_latents.permute(0, 1, 3, 2)

        # Append mask channel: (B, K, T, D+1)
        mask_ch = observed_mask[:, :, None, None].expand(B, K, T, 1)
        x = torch.cat([x, mask_ch], dim=-1)

        h = self.in_proj(x)                           # (B, K, T, d_model)

        sidx = torch.arange(K, device=device)
        h = h + self.sensor_emb(sidx)[None, :, None, :]

        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, t_emb, observed_mask)
            skip_sum = skip_sum + skip

        h = skip_sum / math.sqrt(len(self.blocks))
        h = self.out_norm(h)
        out = self.out_proj(h)                        # (B, K, T, D)
        return out.permute(0, 1, 3, 2)               # (B, K, D, T)


def create_latent_cross_diffusion_recon(n_sensors=7, latent_dim=16, t_shared=64,
                                         d_model=128, num_heads=4, num_blocks=6,
                                         dropout=0.1):
    return LatentCrossDiffusionRecon(
        n_sensors=n_sensors, latent_dim=latent_dim, t_shared=t_shared,
        d_model=d_model, t_dim=128,
        num_heads=num_heads, num_blocks=num_blocks, dropout=dropout,
    )
