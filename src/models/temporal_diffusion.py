# ============================================================
# Temporal Conditional Diffusion Models
# For modality imputation in latent space
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal Time Embedding (same as before)
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer timesteps
        returns: (B, dim)
        """
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
# Temporal Conv Block with Time Conditioning
# ============================================================
class TemporalConvBlock(nn.Module):
    """
    Conv1D block with time conditioning via FiLM (Feature-wise Linear Modulation)
    """
    def __init__(self, in_channels: int, out_channels: int, t_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # FiLM conditioning: time embedding -> scale and shift
        self.film = nn.Linear(t_dim, out_channels * 2)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        t_emb: (B, t_dim)
        returns: (B, out_channels, T)
        """
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)

        # FiLM conditioning
        scale, shift = self.film(t_emb).chunk(2, dim=1)  # (B, out_channels) each
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = self.act(h)
        h = self.dropout(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        # Residual
        return h + self.residual(x)


# ============================================================
# Conditional Temporal Denoiser
# ============================================================
class ConditionalTemporalDenoiser(nn.Module):
    """
    Temporal denoiser that takes noisy target latent + condition latents

    Args:
        target_dim: latent dimension of target modality (e.g., 32 for phone)
        target_len: temporal length of target modality (e.g., 100 for phone)
        condition_dims: list of latent dims for conditioning modalities [32, 16] for watch+glasses
        condition_lens: list of temporal lengths for conditioning modalities [34, 10]
        hidden_dim: hidden dimension for conv layers
        t_dim: time embedding dimension
        num_blocks: number of temporal conv blocks
        dropout: dropout rate
    """
    def __init__(
        self,
        target_dim: int,
        target_len: int,
        condition_dims: list[int],
        condition_lens: list[int],
        hidden_dim: int = 128,
        t_dim: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.target_dim = target_dim
        self.target_len = target_len
        self.condition_dims = condition_dims
        self.condition_lens = condition_lens

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)

        # Input projection: target + interpolated conditions
        # We'll interpolate all conditions to target_len
        total_input_dim = target_dim + sum(condition_dims)
        self.input_proj = nn.Conv1d(total_input_dim, hidden_dim, 1)

        # Temporal conv blocks
        self.blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, hidden_dim, t_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, target_dim, 1)

        # Initialize output projection to zero (helps training stability)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        conditions: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            z_t: noisy target latent (B, target_dim, target_len)
            t: diffusion timestep (B,)
            conditions: list of condition latents [(B, D1, T1), (B, D2, T2), ...]

        Returns:
            predicted noise (B, target_dim, target_len)
        """
        B = z_t.shape[0]  # noqa: F841

        # Time embedding
        t_emb = self.t_embed(t)  # (B, t_dim)

        # Interpolate all conditions to target_len
        conditions_interp = []
        for cond, cond_len in zip(conditions, self.condition_lens):
            # cond: (B, D_cond, T_cond)
            if cond.shape[-1] != self.target_len:
                # Interpolate to target_len
                cond_interp = F.interpolate(cond, size=self.target_len, mode='linear', align_corners=False)
            else:
                cond_interp = cond
            conditions_interp.append(cond_interp)

        # Concatenate target and conditions along channel dimension
        x = torch.cat([z_t] + conditions_interp, dim=1)  # (B, total_input_dim, target_len)

        # Input projection
        h = self.input_proj(x)  # (B, hidden_dim, target_len)

        # Temporal conv blocks with time conditioning
        for block in self.blocks:
            h = block(h, t_emb)

        # Output projection
        noise_pred = self.output_proj(h)  # (B, target_dim, target_len)

        return noise_pred


# ============================================================
# Factory functions for each modality
# ============================================================

def create_phone_denoiser(hidden_dim: int = 128, num_blocks: int = 4, dropout: float = 0.1):
    """
    Denoiser for phone modality, conditioned on watch + glasses
    """
    return ConditionalTemporalDenoiser(
        target_dim=32,
        target_len=100,
        condition_dims=[32, 16],  # watch, glasses
        condition_lens=[34, 10],
        hidden_dim=hidden_dim,
        t_dim=128,
        num_blocks=num_blocks,
        dropout=dropout,
    )


def create_watch_denoiser(hidden_dim: int = 128, num_blocks: int = 4, dropout: float = 0.1):
    """
    Denoiser for watch modality, conditioned on phone + glasses
    """
    return ConditionalTemporalDenoiser(
        target_dim=32,
        target_len=34,
        condition_dims=[32, 16],  # phone, glasses
        condition_lens=[100, 10],
        hidden_dim=hidden_dim,
        t_dim=128,
        num_blocks=num_blocks,
        dropout=dropout,
    )


def create_glasses_denoiser(hidden_dim: int = 128, num_blocks: int = 4, dropout: float = 0.1):
    """
    Denoiser for glasses modality, conditioned on phone + watch
    """
    return ConditionalTemporalDenoiser(
        target_dim=16,
        target_len=10,
        condition_dims=[32, 32],  # phone, watch
        condition_lens=[100, 34],
        hidden_dim=hidden_dim,
        t_dim=128,
        num_blocks=num_blocks,
        dropout=dropout,
    )
