# ============================================================
# Flat Joint Conditional Diffusion Model
# Single model for all modalities, works in flattened latent space
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Cosine Schedule (Nichol & Dhariwal 2021)
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule as proposed in 'Improved DDPM'."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    return betas.float()


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def make_schedule(T, schedule_type="cosine"):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = linear_beta_schedule(T)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # For SNR weighting
    snr = alpha_bar / (1.0 - alpha_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "snr": snr,
    }


# ============================================================
# Sinusoidal Time Embedding
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


# ============================================================
# ResBlock with FiLM
# ============================================================
class ResBlock(nn.Module):
    """Residual block with FiLM time conditioning."""

    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # FiLM: time → scale + shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )

    def forward(self, x, t_emb):
        scale_shift = self.time_proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift  # FiLM
        h = F.silu(h)
        h = self.linear1(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.linear2(h)
        h = self.dropout(h)

        return x + h


# ============================================================
# Joint Flat Conditional Denoiser
# ============================================================
MODALITY_IDS = {"phone": 0, "watch": 1, "glasses": 2}

FLAT_DIMS = {
    "phone":   32 * 100,  # 3200
    "watch":   32 * 34,   # 1088
    "glasses": 16 * 10,   # 160
}

# Max flat dim (for padding targets to same size)
MAX_FLAT_DIM = max(FLAT_DIMS.values())  # 3200

# Total condition dim (all modalities concatenated)
TOTAL_COND_DIM = sum(FLAT_DIMS.values())  # 4448


class JointFlatDenoiser(nn.Module):
    """
    Single denoiser for all modalities in flattened latent space.

    Input:
        z_t:             (B, max_flat_dim) - zero-padded noisy target
        t:               (B,)             - diffusion timestep
        cond:            (B, total_cond_dim) - all modalities concatenated
                         (zeros for missing/target modality)
        modality_id:     (B,)             - which modality is target (0/1/2)

    Output:
        noise prediction (B, max_flat_dim)
    """

    def __init__(self, hidden_dim=1024, num_layers=8, time_dim=256, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Modality embedding (3 modalities)
        self.modality_emb = nn.Embedding(3, hidden_dim)

        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(TOTAL_COND_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Input projection (from max_flat_dim)
        self.input_proj = nn.Linear(MAX_FLAT_DIM, hidden_dim)

        # Main backbone: ResBlocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(ResBlock(hidden_dim, time_dim, dropout))

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, MAX_FLAT_DIM)

        # Init output to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t, t, cond, modality_id):
        """
        z_t:         (B, max_flat_dim) - zero-padded noisy target
        t:           (B,)
        cond:        (B, total_cond_dim)
        modality_id: (B,) LongTensor - 0=phone, 1=watch, 2=glasses
        """
        t_emb = self.time_mlp(t)                    # (B, time_dim)
        c_emb = self.cond_encoder(cond)              # (B, hidden_dim)
        m_emb = self.modality_emb(modality_id)       # (B, hidden_dim)

        h = self.input_proj(z_t)       # (B, hidden_dim)
        h = h + c_emb + m_emb         # Combine all conditions

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.output_norm(h)
        return self.output_proj(h)     # (B, max_flat_dim)


# ============================================================
# Helper: pad/unpad target to max_flat_dim
# ============================================================
def pad_to_max(x, target_modality):
    """Pad flat target to MAX_FLAT_DIM with zeros."""
    dim = FLAT_DIMS[target_modality]
    if dim == MAX_FLAT_DIM:
        return x
    return F.pad(x, (0, MAX_FLAT_DIM - dim))


def unpad_from_max(x, target_modality):
    """Extract actual target dims from padded output."""
    dim = FLAT_DIMS[target_modality]
    return x[:, :dim]


# ============================================================
# Build condition vector
# ============================================================
def build_condition(phone_flat=None, watch_flat=None, glasses_flat=None,
                    target_modality="phone", device="cuda"):
    """
    Build concatenated condition vector.
    Target modality is zeroed out, available modalities are included.

    Returns: (B, TOTAL_COND_DIM)
    """
    B = None
    for x in [phone_flat, watch_flat, glasses_flat]:
        if x is not None:
            B = x.shape[0]
            break

    parts = []
    for name, dim in [("phone", FLAT_DIMS["phone"]),
                      ("watch", FLAT_DIMS["watch"]),
                      ("glasses", FLAT_DIMS["glasses"])]:
        data = {"phone": phone_flat, "watch": watch_flat, "glasses": glasses_flat}[name]

        if name == target_modality or data is None:
            parts.append(torch.zeros(B, dim, device=device))
        else:
            parts.append(data.to(device))

    return torch.cat(parts, dim=1)


# ============================================================
# DDIM Sampler
# ============================================================
@torch.no_grad()
def ddim_sample(model, cond, modality_id, target_modality, sched, T,
                ddim_steps=200, eta=0.0, clip_range=5.0, device="cuda"):
    """
    DDIM sampling with quadratic timestep spacing.
    """
    B = cond.shape[0]

    # Quadratic timestep spacing
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    # Start from pure noise (padded to max dim)
    z = torch.randn(B, MAX_FLAT_DIM, device=device)

    alpha_bar = sched["alpha_bar"].to(device)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]

        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        eps_pred = model(z, t_batch, cond, modality_id)

        ab_now = alpha_bar[t_now]
        pred_x0 = (z - torch.sqrt(1 - ab_now) * eps_pred) / torch.sqrt(ab_now)

        if clip_range > 0:
            pred_x0 = torch.clamp(pred_x0, -clip_range, clip_range)

        ab_next = alpha_bar[t_next]

        if eta > 0:
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_now)) * torch.sqrt(1 - ab_now / ab_next)
            dir_zt = torch.sqrt(1 - ab_next - sigma ** 2) * eps_pred
            noise = sigma * torch.randn_like(z)
        else:
            dir_zt = torch.sqrt(1 - ab_next) * eps_pred
            noise = 0

        z = torch.sqrt(ab_next) * pred_x0 + dir_zt + noise

    # Unpad to actual target dim
    return unpad_from_max(z, target_modality)
