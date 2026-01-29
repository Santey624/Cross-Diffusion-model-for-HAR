# ============================================================
# Joint Temporal Diffusion Model with Cross-Modal Attention
# Unified model for all modalities with cross-modal interactions
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal Time Embedding
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
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
    """Sinusoidal positional encoding for temporal sequences"""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# Cross-Modal Attention Block
# ============================================================
class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention between target and condition modalities
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        """
        query: (B, T_q, D)
        key, value: (B, T_kv, D)
        returns: (B, T_q, D)
        """
        B, T_q, D = query.shape
        T_kv = key.shape[1]

        # Project and reshape
        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_q, d)
        K = self.k_proj(key).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, T_kv, d)
        V = self.v_proj(value).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T_kv, d)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T_q, T_kv)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, T_q, d)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)  # (B, T_q, D)

        return self.out_proj(out)


# ============================================================
# Temporal Conv Block with FiLM
# ============================================================
class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, t_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)

        # FiLM conditioning
        self.film = nn.Linear(t_dim, channels * 2)

        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        t_emb: (B, t_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # FiLM
        scale, shift = self.film(t_emb).chunk(2, dim=1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + x  # Residual


# ============================================================
# Joint Temporal Diffusion Model
# ============================================================
class JointTemporalDiffusion(nn.Module):
    """
    Unified diffusion model for all modalities with cross-modal attention

    Args:
        hidden_dim: hidden dimension for processing
        t_dim: time embedding dimension
        num_heads: number of attention heads
        num_conv_blocks: number of temporal conv blocks
        num_attn_blocks: number of cross-attention blocks
        dropout: dropout rate
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        t_dim: int = 128,
        num_heads: int = 4,
        num_conv_blocks: int = 3,
        num_attn_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Modality specs: (latent_dim, seq_len)
        self.modality_specs = {
            'phone': (32, 100),
            'watch': (32, 34),
            'glasses': (16, 10),
        }

        # Time embedding
        self.t_embed = SinusoidalTimeEmbedding(t_dim)

        # Input projections for each modality (to common hidden_dim)
        self.input_projs = nn.ModuleDict({
            'phone': nn.Conv1d(32, hidden_dim, 1),
            'watch': nn.Conv1d(32, hidden_dim, 1),
            'glasses': nn.Conv1d(16, hidden_dim, 1),
        })

        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(hidden_dim, max_len=200)

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

        # Output projections for each modality
        self.output_projs = nn.ModuleDict({
            'phone': nn.Conv1d(hidden_dim, 32, 1),
            'watch': nn.Conv1d(hidden_dim, 32, 1),
            'glasses': nn.Conv1d(hidden_dim, 16, 1),
        })

        # Initialize output projections to zero
        for proj in self.output_projs.values():
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        target_modality: str,
        z_t: torch.Tensor,
        t: torch.Tensor,
        phone_latent: torch.Tensor = None,
        watch_latent: torch.Tensor = None,
        glasses_latent: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            target_modality: which modality to denoise ("phone", "watch", or "glasses")
            z_t: noisy target latent (B, D_target, T_target)
            t: diffusion timestep (B,)
            phone_latent: phone condition (B, 32, 100) or None if missing
            watch_latent: watch condition (B, 32, 34) or None if missing
            glasses_latent: glasses condition (B, 16, 10) or None if missing

        Returns:
            predicted noise (B, D_target, T_target)
        """
        B = z_t.shape[0]

        # Time embedding
        t_emb = self.t_embed(t)  # (B, t_dim)

        # Project target to hidden dim
        target_hidden = self.input_projs[target_modality](z_t)  # (B, hidden_dim, T_target)
        target_seq_len = target_hidden.shape[-1]

        # Prepare condition modalities
        condition_hiddens = []
        condition_names = []

        if phone_latent is not None and target_modality != 'phone':
            phone_hidden = self.input_projs['phone'](phone_latent)  # (B, hidden_dim, 100)
            condition_hiddens.append(phone_hidden)
            condition_names.append('phone')

        if watch_latent is not None and target_modality != 'watch':
            watch_hidden = self.input_projs['watch'](watch_latent)  # (B, hidden_dim, 34)
            condition_hiddens.append(watch_hidden)
            condition_names.append('watch')

        if glasses_latent is not None and target_modality != 'glasses':
            glasses_hidden = self.input_projs['glasses'](glasses_latent)  # (B, hidden_dim, 10)
            condition_hiddens.append(glasses_hidden)
            condition_names.append('glasses')

        # Concatenate all condition modalities along temporal dimension
        if len(condition_hiddens) > 0:
            condition_concat = torch.cat(condition_hiddens, dim=2)  # (B, hidden_dim, T_cond_total)
        else:
            # No conditions available (shouldn't happen in practice)
            condition_concat = torch.zeros(B, self.hidden_dim, 1, device=z_t.device)

        # Convert to (B, T, D) for attention
        target_seq = target_hidden.transpose(1, 2)  # (B, T_target, hidden_dim)
        condition_seq = condition_concat.transpose(1, 2)  # (B, T_cond, hidden_dim)

        # Add positional encoding
        target_seq = self.pos_encoding(target_seq)
        condition_seq = self.pos_encoding(condition_seq)

        # Cross-modal attention blocks
        h = target_seq
        for attn, norm in zip(self.cross_attn_blocks, self.attn_norms):
            # Cross-attention: target attends to conditions
            attn_out = attn(h, condition_seq, condition_seq)
            h = norm(h + attn_out)  # Residual + norm

        # Convert back to (B, C, T) for conv
        h = h.transpose(1, 2)  # (B, hidden_dim, T_target)

        # Temporal conv blocks with time conditioning
        for conv_block in self.conv_blocks:
            h = conv_block(h, t_emb)

        # Output projection
        noise_pred = self.output_projs[target_modality](h)  # (B, D_target, T_target)

        return noise_pred


# ============================================================
# Factory function
# ============================================================
def create_joint_diffusion_model(
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_conv_blocks: int = 3,
    num_attn_blocks: int = 2,
    dropout: float = 0.1,
):
    """Create a joint diffusion model for all modalities"""
    return JointTemporalDiffusion(
        hidden_dim=hidden_dim,
        t_dim=128,
        num_heads=num_heads,
        num_conv_blocks=num_conv_blocks,
        num_attn_blocks=num_attn_blocks,
        dropout=dropout,
    )
