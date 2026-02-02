# ============================================================
# Sensor-Level Joint Temporal Diffusion Model
# Shared latent space: all latents (B, D, T_SHARED) — no interpolation
# Concatenation-based conditioning + sensor embeddings
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
        # x: (B, C, T) -> (B, T, C) for attention
        x_t = x.transpose(1, 2)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + self.dropout(attn_out))
        return x_t.transpose(1, 2)  # back to (B, C, T)


# ============================================================
# Sensor Joint Diffusion — Shared Latent Space
# ============================================================
class SensorJointDiffusion(nn.Module):
    """
    Joint diffusion for 7 sensor modalities in shared latent space.

    All latents are (B, D, T_SHARED) — same shape, no interpolation needed.

    Conditioning:
    1. Each condition + z_t gets a learnable sensor embedding
    2. All concatenated along channel dim
    3. Processed through Conv1d + Self-Attention blocks
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
        sensor_names=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_sensors = n_sensors

        # Sensor names
        if sensor_names is None:
            from src.models.sensor_vae import SENSOR_NAMES
            self.sensor_names = SENSOR_NAMES
        else:
            self.sensor_names = sensor_names

        self.sensor_name_to_idx = {name: i for i, name in enumerate(self.sensor_names)}
        self.missing_idx = n_sensors  # index for "missing" embedding

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable sensor-type embeddings (+1 for "missing")
        self.sensor_embeddings = nn.Embedding(n_sensors + 1, latent_dim)

        # Input projection: all sensors concatenated along channel dim
        total_input_dim = latent_dim * n_sensors
        self.input_proj = nn.Conv1d(total_input_dim, hidden_dim, 1)

        # Conv blocks (with FiLM time conditioning)
        self.conv_blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_conv_blocks)
        ])

        # Self-attention blocks
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_attn_blocks)
        ])

        # Insert attention after every N conv blocks
        self.attn_after_conv = num_conv_blocks // (num_attn_blocks + 1)

        # Single output projection (shared latent space = same output dim)
        self.output_proj = nn.Conv1d(hidden_dim, latent_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, target_modality, z_t, t, conditions):
        """
        Args:
            target_modality: str
            z_t: (B, D, T_SHARED) noisy target
            t: (B,) timestep
            conditions: dict {name: (B, D, T_SHARED) or None}

        Returns:
            noise prediction (B, D, T_SHARED)
        """
        B = z_t.shape[0]
        T_shared = z_t.shape[2]
        device = z_t.device

        # Time embedding
        t_emb = self.t_proj(self.t_embed(t))  # (B, hidden_dim)

        # Target with sensor embedding
        target_idx = self.sensor_name_to_idx[target_modality]
        target_emb = self.sensor_embeddings(
            torch.tensor(target_idx, device=device)
        )  # (latent_dim,)
        z_target = z_t + target_emb[None, :, None]

        channel_list = [z_target]

        for name in self.sensor_names:
            if name == target_modality:
                continue

            cond = conditions.get(name, None)
            if cond is not None:
                # Add sensor embedding — no interpolation needed (same shape)
                idx = self.sensor_name_to_idx[name]
                emb = self.sensor_embeddings(torch.tensor(idx, device=device))
                cond = cond + emb[None, :, None]
            else:
                # Missing: use learned "missing" embedding
                emb = self.sensor_embeddings(torch.tensor(self.missing_idx, device=device))
                cond = emb[None, :, None].expand(B, -1, T_shared)

            channel_list.append(cond)

        # Concatenate: (B, D * n_sensors, T_SHARED)
        x = torch.cat(channel_list, dim=1)

        # Project to hidden dim
        h = self.input_proj(x)  # (B, hidden_dim, T_SHARED)

        # Process through conv + attention blocks
        attn_idx = 0
        for i, conv_block in enumerate(self.conv_blocks):
            h = conv_block(h, t_emb)

            if (i + 1) % max(self.attn_after_conv, 1) == 0 and attn_idx < len(self.attn_blocks):
                h = self.attn_blocks[attn_idx](h)
                attn_idx += 1

        # Output projection (single shared projection)
        noise_pred = self.output_proj(h)

        return noise_pred


# ============================================================
# Default config
# ============================================================
DEFAULT_LATENT_DIM = 8
DEFAULT_N_SENSORS = 7


def create_sensor_diffusion_model(
    n_sensors=DEFAULT_N_SENSORS,
    latent_dim=DEFAULT_LATENT_DIM,
    hidden_dim=256,
    num_heads=4,
    num_conv_blocks=6,
    num_attn_blocks=2,
    dropout=0.1,
    sensor_names=None,
):
    return SensorJointDiffusion(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        t_dim=128,
        num_heads=num_heads,
        num_conv_blocks=num_conv_blocks,
        num_attn_blocks=num_attn_blocks,
        dropout=dropout,
        sensor_names=sensor_names,
    )
