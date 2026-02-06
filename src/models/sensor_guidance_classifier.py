# ============================================================
# Sensor Guidance Classifier (Regressor)
# Given noisy z_t + timestep + sensor_type, predicts the
# clean latents of all OTHER sensors.
# Used for classifier guidance at inference time.
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
# Residual Conv Block with FiLM
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
# Sensor Guidance Classifier
# ============================================================
class SensorGuidanceClassifier(nn.Module):
    """
    Regressor for classifier guidance.

    Given a noisy sensor latent z_t, timestep t, and which sensor it is,
    predicts the CLEAN latents of all 7 sensors.

    During training: loss = MSE on all sensors EXCEPT the target.
    During inference: gradient of MSE(predicted, observed) w.r.t. z_t
                     provides the guidance signal.
    """

    def __init__(
        self,
        n_sensors=7,
        latent_dim=8,
        hidden_dim=128,
        t_dim=128,
        num_conv_blocks=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim

        # Sensor type embedding
        self.sensor_embeddings = nn.Embedding(n_sensors, latent_dim)

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, 1)

        # Conv blocks
        self.conv_blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_conv_blocks)
        ])

        # Output: predict all 7 sensor latents
        self.output_proj = nn.Conv1d(hidden_dim, n_sensors * latent_dim, 1)

    def forward(self, z_t, t, sensor_idx):
        """
        Args:
            z_t: (B, D, T_SHARED) noisy latent of target sensor
            t: (B,) timestep
            sensor_idx: int — which sensor type z_t belongs to

        Returns:
            predictions: (B, n_sensors, D, T_SHARED) — predicted clean latents
                         for ALL 7 sensors
        """
        B, D, T = z_t.shape
        device = z_t.device

        # Time embedding
        t_emb = self.t_proj(self.t_embed(t))  # (B, hidden_dim)

        # Add sensor type embedding
        emb = self.sensor_embeddings(
            torch.tensor(sensor_idx, device=device)
        )  # (latent_dim,)
        x = z_t + emb[None, :, None]

        # Process
        h = self.input_proj(x)  # (B, hidden_dim, T)

        for block in self.conv_blocks:
            h = block(h, t_emb)

        # Output: (B, n_sensors * D, T)
        out = self.output_proj(h)

        # Reshape to (B, n_sensors, D, T)
        out = out.view(B, self.n_sensors, D, T)

        return out

    def compute_guidance_loss(self, z_t, t, sensor_idx, observed_conditions, sensor_names):
        """
        Compute guidance loss for observed conditions.

        Args:
            z_t: (B, D, T) — noisy target (requires_grad=True)
            t: (B,) timestep
            sensor_idx: int — target sensor index
            observed_conditions: dict {name: (B, D, T)} — clean latents of observed sensors
            sensor_names: list of sensor names (order matches output indices)

        Returns:
            loss: scalar — MSE between predicted and observed conditions
        """
        preds = self.forward(z_t, t, sensor_idx)  # (B, 7, D, T)

        loss = 0.0
        count = 0
        for i, name in enumerate(sensor_names):
            if name in observed_conditions and observed_conditions[name] is not None:
                pred_i = preds[:, i]  # (B, D, T)
                target_i = observed_conditions[name]
                loss = loss + F.mse_loss(pred_i, target_i)
                count += 1

        return loss / max(count, 1)


def create_guidance_classifier(
    n_sensors=7,
    latent_dim=8,
    hidden_dim=128,
    num_conv_blocks=4,
    dropout=0.1,
):
    return SensorGuidanceClassifier(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        t_dim=128,
        num_conv_blocks=num_conv_blocks,
        dropout=dropout,
    )
