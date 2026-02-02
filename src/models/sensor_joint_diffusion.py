# ============================================================
# Sensor-Level Joint Temporal Diffusion Model
# Dynamic modality specs, Conv1D + CrossAttention
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
# Temporal Positional Encoding
# ============================================================
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# Cross-Modal Attention Block
# ============================================================
class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, T_q, D = query.shape
        T_kv = key.shape[1]

        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out)


# ============================================================
# Temporal Conv Block with FiLM
# ============================================================
class TemporalConvBlock(nn.Module):
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
# Sensor Joint Diffusion Model
# ============================================================
class SensorJointDiffusion(nn.Module):
    """
    Joint diffusion model for 7 sensor modalities.
    Configurable via modality_specs dict.
    """

    def __init__(
        self,
        modality_specs,
        hidden_dim=256,
        t_dim=128,
        num_heads=4,
        num_conv_blocks=4,
        num_attn_blocks=3,
        dropout=0.1,
    ):
        """
        modality_specs: dict {name: (latent_dim, seq_len)}
            e.g. {"phone_acc": (8, 100), "watch_acc": (8, 34), ...}
        """
        super().__init__()
        self.modality_specs = modality_specs
        self.hidden_dim = hidden_dim

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)

        # Per-modality input projections
        self.input_projs = nn.ModuleDict({
            name: nn.Conv1d(latent_dim, hidden_dim, 1)
            for name, (latent_dim, _) in modality_specs.items()
        })

        # Positional encoding (max_len covers concatenated conditions)
        self.pos_encoding = TemporalPositionalEncoding(hidden_dim, max_len=500)

        # Cross-modal attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attn_blocks)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_attn_blocks)
        ])

        # Temporal conv blocks
        self.conv_blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, t_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_conv_blocks)
        ])

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
            target_modality: str, which sensor to denoise
            z_t: (B, D, T') noisy target latent
            t: (B,) diffusion timestep
            conditions: dict {name: Tensor (B, D, T') or None}

        Returns:
            predicted noise (B, D, T')
        """
        B = z_t.shape[0]

        # Time embedding
        t_emb = self.t_embed(t)

        # Project target
        target_hidden = self.input_projs[target_modality](z_t)

        # Project and concatenate conditions
        cond_list = []
        for name, latent in conditions.items():
            if latent is not None and name != target_modality:
                cond_list.append(self.input_projs[name](latent))

        if cond_list:
            condition_concat = torch.cat(cond_list, dim=2)
        else:
            condition_concat = torch.zeros(B, self.hidden_dim, 1, device=z_t.device)

        # (B, C, T) -> (B, T, C) for attention
        target_seq = target_hidden.transpose(1, 2)
        condition_seq = condition_concat.transpose(1, 2)

        # Add positional encoding
        target_seq = self.pos_encoding(target_seq)
        condition_seq = self.pos_encoding(condition_seq)

        # Cross-attention blocks
        h = target_seq
        for attn, norm in zip(self.cross_attn_blocks, self.attn_norms):
            attn_out = attn(h, condition_seq, condition_seq)
            h = norm(h + attn_out)

        # Back to (B, C, T)
        h = h.transpose(1, 2)

        # Temporal conv blocks with time conditioning
        for conv_block in self.conv_blocks:
            h = conv_block(h, t_emb)

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
    num_conv_blocks=4,
    num_attn_blocks=3,
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
