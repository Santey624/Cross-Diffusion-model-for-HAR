# ============================================================
# Sensor Conditional Diffusion V3
# Proper conditional diffusion: p(z_missing | z_observed)
#
# Key difference to V2 (inpainting):
#   V2: All sensors in one tensor, observed pinned, self-attention
#   V3: Missing sensors (queries) cross-attend to observed sensors
#       (keys/values only). Observed sensors never polluted by noise.
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
# Temporal Attention: attends across time for each sensor
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, K, T, d) — attend across T for each sensor."""
        B, K, T, d = x.shape
        x_flat = x.reshape(B * K, T, d)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + self.dropout(attn_out))
        return x_flat.reshape(B, K, T, d)


# ============================================================
# Conditional Cross-Sensor Attention
# Missing sensors (queries) attend ONLY to observed sensors (keys/values)
# ============================================================
class ConditionalCrossSensorAttn(nn.Module):
    """
    Implements p(z_missing | z_observed) via cross-attention:
      - Query: all sensors
      - Key/Value: ONLY observed sensors (missing are blocked via key_padding_mask)

    This ensures that missing sensors receive information from observed sensors
    but not from other noisy missing sensors.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, observed_mask):
        """
        x: (B, K, T, d)
        observed_mask: (B, K) — 1.0=observed, 0.0=missing
        """
        B, K, T, d = x.shape

        # Reshape to (B*T, K, d) for per-timestep cross-sensor attention
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, K, d)

        # key_padding_mask: (B*T, K), True = block this sensor as key
        # Block missing sensors (noise) from being used as keys
        key_mask = (observed_mask == 0)  # (B, K), True=missing
        key_mask_BT = key_mask[:, None, :].expand(B, T, K).reshape(B * T, K)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat,
                                key_padding_mask=key_mask_BT)
        x_flat = self.norm(x_flat + self.dropout(attn_out))
        return x_flat.reshape(B, T, K, d).permute(0, 2, 1, 3)


# ============================================================
# Residual Block V3: Temporal Attn + Conditional Cross-Sensor Attn
# ============================================================
class ResidualBlock3(nn.Module):
    def __init__(self, d_model, t_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.film = nn.Linear(t_dim, d_model * 2)

        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        self.cond_cross_attn = ConditionalCrossSensorAttn(d_model, num_heads, dropout)

        self.gate = nn.Linear(d_model, d_model)
        self.filter = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        self.res_proj = nn.Linear(d_model, d_model)
        self.skip_proj = nn.Linear(d_model, d_model)

    def forward(self, x, t_emb, observed_mask):
        """
        x: (B, K, T, d)
        t_emb: (B, t_dim)
        observed_mask: (B, K)
        """
        B, K, T, d = x.shape
        residual = x

        # 1. Temporal conv per sensor
        h = x.reshape(B * K, T, d).permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1).reshape(B, K, T, d)
        h = self.norm1(h)

        # 2. FiLM conditioning from timestep
        scale, shift = self.film(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        h = self.act(h)
        h = self.dropout(h)

        # 3. Temporal attention (across time per sensor)
        h = self.temporal_attn(h)

        # 4. Conditional cross-sensor attention (missing queries, observed keys)
        h = self.cond_cross_attn(h, observed_mask)

        # 5. Gate-filter
        gate = torch.sigmoid(self.gate(h))
        filt = torch.tanh(self.filter(h))
        h = gate * filt

        # 6. Residual + skip
        res_out = self.res_proj(h)
        skip_out = self.skip_proj(h)
        return (residual + res_out) / math.sqrt(2.0), skip_out


# ============================================================
# Sensor Conditional Diffusion V3
# ============================================================
class SensorConditionalDiffusionV3(nn.Module):
    """
    Conditional diffusion model for multimodal sensor imputation.

    Models p(z_missing | z_observed) via:
    - Cross-attention where observed sensors are keys/values
    - Missing sensors are queries — they receive information from observed
    - Noising process only conceptually on missing sensors
      (observed are pinned during DDIM sampling)

    Interface identical to V2 for drop-in compatibility.
    """

    def __init__(
        self,
        n_sensors=7,
        latent_dim=8,
        d_model=128,
        t_dim=128,
        num_heads=4,
        num_blocks=4,
        dropout=0.1,
        sensor_names=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_sensors = n_sensors

        if sensor_names is None:
            from src.models.sensor_vae import SENSOR_NAMES
            self.sensor_names = SENSOR_NAMES
        else:
            self.sensor_names = sensor_names

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable sensor-type embeddings
        self.sensor_embeddings = nn.Embedding(n_sensors, latent_dim)

        # Input projection: latent_dim + 1 (mask channel) -> d_model
        self.input_proj = nn.Linear(latent_dim + 1, d_model)

        # V3 blocks: use ConditionalCrossSensorAttn instead of FeatureAttention
        self.blocks = nn.ModuleList([
            ResidualBlock3(d_model, d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_sensors, t, observed_mask):
        """
        Args:
            noisy_sensors: (B, K, D, T) — observed=clean, missing=noisy
            t: (B,) diffusion timestep
            observed_mask: (B, K) — 1.0=observed, 0.0=missing

        Returns:
            noise_pred: (B, K, D, T)
        """
        B, K, D, T = noisy_sensors.shape
        device = noisy_sensors.device

        # Time embedding
        t_emb = self.t_proj(self.t_embed(t))  # (B, d_model)

        # Sensor-type embeddings
        sensor_idx = torch.arange(K, device=device)
        sensor_emb = self.sensor_embeddings(sensor_idx)  # (K, D)
        x = noisy_sensors + sensor_emb[None, :, :, None]  # (B, K, D, T)

        # Reshape to (B, K, T, D)
        x = x.permute(0, 1, 3, 2)

        # Append mask channel: (B, K, T, D+1)
        mask_channel = observed_mask[:, :, None, None].expand(B, K, T, 1)
        x = torch.cat([x, mask_channel], dim=-1)

        # Project to d_model
        h = self.input_proj(x)  # (B, K, T, d_model)

        # Process through V3 blocks (pass observed_mask for conditional cross-attn)
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, t_emb, observed_mask)
            skip_sum = skip_sum + skip

        h = skip_sum / math.sqrt(len(self.blocks))
        h = self.output_norm(h)
        noise_pred = self.output_proj(h)  # (B, K, T, D)

        # Reshape back to (B, K, D, T)
        noise_pred = noise_pred.permute(0, 1, 3, 2)
        return noise_pred


def create_sensor_diffusion_v3(
    n_sensors=7,
    latent_dim=8,
    d_model=128,
    num_heads=4,
    num_blocks=4,
    dropout=0.1,
    sensor_names=None,
):
    return SensorConditionalDiffusionV3(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        d_model=d_model,
        t_dim=128,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        sensor_names=sensor_names,
    )
