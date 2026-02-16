# ============================================================
# Sensor-Level Joint Temporal Diffusion Model V2
# Key improvements over V1:
#   1. 2D Attention: Temporal + Feature (cross-sensor) attention
#   2. Multi-sensor masking: arbitrary number of sensors can be missing
#   3. Mask-based conditioning: observed mask channel added
#   4. Predicts noise for ALL missing sensors simultaneously
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
        """
        x: (B, K, T, d) — K sensors, T timesteps
        Attend across T for each sensor independently.
        """
        B, K, T, d = x.shape
        # Merge B and K: (B*K, T, d)
        x_flat = x.reshape(B * K, T, d)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + self.dropout(attn_out))
        return x_flat.reshape(B, K, T, d)


# ============================================================
# Feature (Cross-Sensor) Attention: attends across sensors for each timestep
# ============================================================
class FeatureAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, K, T, d) — K sensors, T timesteps
        Attend across K sensors for each timestep independently.
        """
        B, K, T, d = x.shape
        # Merge B and T: (B*T, K, d)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, K, d)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + self.dropout(attn_out))
        return x_flat.reshape(B, T, K, d).permute(0, 2, 1, 3)


# ============================================================
# Residual Block with 2D Attention + FiLM conditioning
# ============================================================
class ResidualBlock2D(nn.Module):
    def __init__(self, d_model, t_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Temporal convolution (across time for each sensor)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FiLM conditioning from timestep
        self.film = nn.Linear(t_dim, d_model * 2)

        # 2D Attention
        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        self.feature_attn = FeatureAttention(d_model, num_heads, dropout)

        # Gate-filter mechanism (like CSDI)
        self.gate = nn.Linear(d_model, d_model)
        self.filter = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # Output projection for residual + skip
        self.res_proj = nn.Linear(d_model, d_model)
        self.skip_proj = nn.Linear(d_model, d_model)

    def forward(self, x, t_emb):
        """
        x: (B, K, T, d)
        t_emb: (B, t_dim)
        Returns: residual (B, K, T, d), skip (B, K, T, d)
        """
        B, K, T, d = x.shape
        residual = x

        # 1. Temporal conv (per sensor)
        h = x.reshape(B * K, T, d).permute(0, 2, 1)  # (B*K, d, T)
        h = self.conv1(h)
        h = h.permute(0, 2, 1).reshape(B, K, T, d)  # back to (B, K, T, d)
        h = self.norm1(h)

        # 2. FiLM conditioning
        scale, shift = self.film(t_emb).chunk(2, dim=1)  # (B, d) each
        h = h * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        h = self.act(h)
        h = self.dropout(h)

        # 3. 2D Attention: temporal then feature
        h = self.temporal_attn(h)
        h = self.feature_attn(h)

        # 4. Gate-Filter mechanism
        gate = torch.sigmoid(self.gate(h))
        filt = torch.tanh(self.filter(h))
        h = gate * filt

        # 5. Residual + skip connections
        res_out = self.res_proj(h)
        skip_out = self.skip_proj(h)

        return (residual + res_out) / math.sqrt(2.0), skip_out


# ============================================================
# Sensor Joint Diffusion V2 — 2D Attention + Multi-Mask
# ============================================================
class SensorJointDiffusionV2(nn.Module):
    """
    Joint diffusion for 7 sensor modalities in shared latent space.

    Key improvements:
    - 2D Attention: temporal + cross-sensor per timestep
    - Multi-mask: predicts noise for ALL missing sensors at once
    - Mask channel: model knows which sensors are observed vs missing
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

        # Sensor names
        if sensor_names is None:
            from src.models.sensor_vae import SENSOR_NAMES
            self.sensor_names = SENSOR_NAMES
        else:
            self.sensor_names = sensor_names

        self.sensor_name_to_idx = {name: i for i, name in enumerate(self.sensor_names)}

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

        # Residual blocks with 2D attention
        self.blocks = nn.ModuleList([
            ResidualBlock2D(d_model, d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection: d_model -> latent_dim
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_sensors, t, observed_mask):
        """
        Args:
            noisy_sensors: (B, K, D, T) — all sensors stacked.
                           Missing sensors filled with noise, observed with real latents.
            t: (B,) diffusion timestep
            observed_mask: (B, K) — 1.0 if sensor is observed, 0.0 if missing

        Returns:
            noise_pred: (B, K, D, T) — predicted noise for ALL sensors
        """
        B, K, D, T = noisy_sensors.shape
        device = noisy_sensors.device

        # Time embedding
        t_emb = self.t_proj(self.t_embed(t))  # (B, d_model)

        # Add sensor embeddings to each sensor
        sensor_idx = torch.arange(K, device=device)
        sensor_emb = self.sensor_embeddings(sensor_idx)  # (K, D)
        x = noisy_sensors + sensor_emb[None, :, :, None]  # (B, K, D, T)

        # Reshape to (B, K, T, D) for easier processing
        x = x.permute(0, 1, 3, 2)  # (B, K, T, D)

        # Add mask channel: (B, K, T, D+1)
        mask_channel = observed_mask[:, :, None, None].expand(B, K, T, 1)
        x = torch.cat([x, mask_channel], dim=-1)  # (B, K, T, D+1)

        # Project to d_model
        h = self.input_proj(x)  # (B, K, T, d_model)

        # Process through residual blocks with skip connections
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, t_emb)
            skip_sum = skip_sum + skip

        # Normalize accumulated skip connections
        h = skip_sum / math.sqrt(len(self.blocks))
        h = self.output_norm(h)

        # Output projection
        noise_pred = self.output_proj(h)  # (B, K, T, D)

        # Reshape back to (B, K, D, T)
        noise_pred = noise_pred.permute(0, 1, 3, 2)

        return noise_pred


# ============================================================
# Default config
# ============================================================
DEFAULT_LATENT_DIM = 8
DEFAULT_N_SENSORS = 7


def create_sensor_diffusion_v2(
    n_sensors=DEFAULT_N_SENSORS,
    latent_dim=DEFAULT_LATENT_DIM,
    d_model=128,
    num_heads=4,
    num_blocks=4,
    dropout=0.1,
    sensor_names=None,
):
    return SensorJointDiffusionV2(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        d_model=d_model,
        t_dim=128,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        sensor_names=sensor_names,
    )
