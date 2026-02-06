# ============================================================
# Unconditional Sensor-Level Diffusion Model
# Learns p(z_t | t, sensor_type) without condition inputs
# Used with classifier guidance at inference time
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal Time Embedding
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ============================================================
# Residual Conv Block with FiLM conditioning
# ============================================================
class ResConvBlock(nn.Module):
    def __init__(self, channels, t_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.film = nn.Linear(t_dim, channels * 2)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        scale, shift = self.film(t_emb).chunk(2, dim=1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + x


# ============================================================
# Self-Attention Block
# ============================================================
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + self.dropout(attn_out))
        return x_t.transpose(1, 2)


# ============================================================
# Unconditional Sensor Diffusion
# ============================================================
class SensorUncondDiffusion(nn.Module):
    """
    Unconditional diffusion for sensor latents.

    Input: z_t (B, D, T_SHARED) + sensor_type_idx + timestep
    Output: noise prediction (B, D, T_SHARED)

    The sensor_type embedding tells the model which sensor distribution
    it is denoising, since different sensors have different stats even
    in the shared latent space.
    """

    def __init__(
        self,
        n_sensors=7,
        latent_dim=8,
        hidden_dim=256,
        t_dim=128,
        num_heads=4,
        num_conv_blocks=6,
        num_attn_blocks=2,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_sensors = n_sensors

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Sensor type embedding
        self.sensor_embeddings = nn.Embedding(n_sensors, latent_dim)

        # Input projection: just z_t + sensor embedding = latent_dim channels
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, 1)

        # Conv blocks
        self.conv_blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_conv_blocks)
        ])

        # Self-attention blocks
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_attn_blocks)
        ])

        self.attn_after_conv = num_conv_blocks // (num_attn_blocks + 1)

        # Output projection (zero-initialized)
        self.output_proj = nn.Conv1d(hidden_dim, latent_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t, t, sensor_idx):
        """
        Args:
            z_t: (B, D, T_SHARED) noisy latent
            t: (B,) timestep
            sensor_idx: int — which sensor type

        Returns:
            noise prediction (B, D, T_SHARED)
        """
        device = z_t.device

        # Time embedding
        t_emb = self.t_proj(self.t_embed(t))  # (B, hidden_dim)

        # Add sensor type embedding
        emb = self.sensor_embeddings(
            torch.tensor(sensor_idx, device=device)
        )  # (latent_dim,)
        x = z_t + emb[None, :, None]

        # Project to hidden dim
        h = self.input_proj(x)  # (B, hidden_dim, T_SHARED)

        # Process through conv + attention blocks
        attn_idx = 0
        for i, conv_block in enumerate(self.conv_blocks):
            h = conv_block(h, t_emb)
            if (i + 1) % max(self.attn_after_conv, 1) == 0 and attn_idx < len(self.attn_blocks):
                h = self.attn_blocks[attn_idx](h)
                attn_idx += 1

        # Output
        noise_pred = self.output_proj(h)
        return noise_pred


def create_uncond_diffusion_model(
    n_sensors=7,
    latent_dim=8,
    hidden_dim=256,
    num_heads=4,
    num_conv_blocks=6,
    num_attn_blocks=2,
    dropout=0.1,
):
    return SensorUncondDiffusion(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        t_dim=128,
        num_heads=num_heads,
        num_conv_blocks=num_conv_blocks,
        num_attn_blocks=num_attn_blocks,
        dropout=dropout,
    )
