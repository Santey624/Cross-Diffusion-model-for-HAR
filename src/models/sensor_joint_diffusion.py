# ============================================================
# Sensor-Level Joint Temporal Diffusion Model v2
# Concatenation-based conditioning (interpolate + channel concat)
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
# Sensor Joint Diffusion v2 — Concatenation Conditioning
# ============================================================
class SensorJointDiffusion(nn.Module):
    """
    Joint diffusion for 7 sensor modalities.

    Conditioning approach:
    1. Each condition latent is interpolated to target temporal length
    2. All conditions are concatenated along channel dim with z_t
    3. A learnable per-sensor embedding is added to distinguish sensors
    4. Processed through Conv1d + Self-Attention blocks
    """

    def __init__(
        self,
        modality_specs,
        hidden_dim=256,
        t_dim=128,
        num_heads=4,
        num_conv_blocks=6,
        num_attn_blocks=2,
        dropout=0.1,
    ):
        super().__init__()
        self.modality_specs = modality_specs
        self.hidden_dim = hidden_dim
        self.sensor_names = list(modality_specs.keys())
        self.n_sensors = len(self.sensor_names)

        # All sensors have the same latent_dim (z=8)
        self.latent_dim = list(modality_specs.values())[0][0]

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable sensor-type embeddings (added to each condition channel)
        # +1 for the "missing" embedding (when a condition is None)
        self.sensor_embeddings = nn.Embedding(self.n_sensors + 1, self.latent_dim)
        self.sensor_name_to_idx = {name: i for i, name in enumerate(self.sensor_names)}
        self.missing_idx = self.n_sensors  # index for "missing" embedding

        # Input projection: z_t (latent_dim) + n_conditions * latent_dim -> hidden_dim
        # Each condition is latent_dim channels after interpolation
        # Total input = latent_dim * (1 + n_conditions) where n_conditions = n_sensors - 1
        total_input_dim = self.latent_dim * self.n_sensors  # target + all conditions
        self.input_proj = nn.Conv1d(total_input_dim, hidden_dim, 1)

        # Conv blocks (with FiLM time conditioning)
        self.conv_blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_conv_blocks)
        ])

        # Self-attention blocks (interleaved with conv)
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_attn_blocks)
        ])

        # Insert attention after every N conv blocks
        self.attn_after_conv = num_conv_blocks // (num_attn_blocks + 1)

        # Per-modality output projections (initialized to zero)
        self.output_projs = nn.ModuleDict({
            name: nn.Conv1d(hidden_dim, latent_dim, 1)
            for name, (latent_dim, _) in modality_specs.items()
        })
        for proj in self.output_projs.values():
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, target_modality, z_t, t, conditions):
        """
        Args:
            target_modality: str
            z_t: (B, D, T_target) noisy target
            t: (B,) timestep
            conditions: dict {name: (B, D, T_i) or None}

        Returns:
            noise prediction (B, D, T_target)
        """
        B = z_t.shape[0]
        T_target = z_t.shape[2]
        device = z_t.device

        # Time embedding
        t_emb = self.t_proj(self.t_embed(t))  # (B, hidden_dim)

        # Build input: concat target + all conditions along channel dim
        # Target gets its sensor embedding added
        target_idx = self.sensor_name_to_idx[target_modality]
        target_emb = self.sensor_embeddings(
            torch.tensor(target_idx, device=device)
        )  # (latent_dim,)
        z_target = z_t + target_emb[None, :, None]  # (B, D, T_target)

        channel_list = [z_target]

        for name in self.sensor_names:
            if name == target_modality:
                continue

            cond = conditions.get(name, None)
            if cond is not None:
                # Interpolate condition to target temporal length
                if cond.shape[2] != T_target:
                    cond_interp = F.interpolate(
                        cond.float(), size=T_target, mode='linear', align_corners=False
                    )
                else:
                    cond_interp = cond

                # Add sensor embedding
                idx = self.sensor_name_to_idx[name]
                emb = self.sensor_embeddings(torch.tensor(idx, device=device))
                cond_interp = cond_interp + emb[None, :, None]
            else:
                # Missing condition: use learned "missing" embedding, broadcast to shape
                emb = self.sensor_embeddings(torch.tensor(self.missing_idx, device=device))
                cond_interp = emb[None, :, None].expand(B, -1, T_target)

            channel_list.append(cond_interp)

        # Concatenate: (B, D * n_sensors, T_target)
        x = torch.cat(channel_list, dim=1)

        # Project to hidden dim
        h = self.input_proj(x)  # (B, hidden_dim, T_target)

        # Process through conv + attention blocks
        attn_idx = 0
        for i, conv_block in enumerate(self.conv_blocks):
            h = conv_block(h, t_emb)

            # Insert self-attention periodically
            if (i + 1) % max(self.attn_after_conv, 1) == 0 and attn_idx < len(self.attn_blocks):
                h = self.attn_blocks[attn_idx](h)
                attn_idx += 1

        # Output projection
        noise_pred = self.output_projs[target_modality](h)

        return noise_pred


# ============================================================
# Default sensor specs (matching z=8 VAE)
# ============================================================
DEFAULT_SENSOR_SPECS = {
    "phone_acc":   (8, 100),
    "phone_gyro":  (8, 100),
    "phone_grav":  (8, 100),
    "phone_lacc":  (8, 100),
    "watch_acc":   (8, 34),
    "watch_gyro":  (8, 34),
    "glasses_acc": (8, 10),
}


def create_sensor_diffusion_model(
    modality_specs=None,
    hidden_dim=256,
    num_heads=4,
    num_conv_blocks=6,
    num_attn_blocks=2,
    dropout=0.1,
):
    specs = modality_specs or DEFAULT_SENSOR_SPECS
    return SensorJointDiffusion(
        modality_specs=specs,
        hidden_dim=hidden_dim,
        t_dim=128,
        num_heads=num_heads,
        num_conv_blocks=num_conv_blocks,
        num_attn_blocks=num_attn_blocks,
        dropout=dropout,
    )
