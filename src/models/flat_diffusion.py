# ============================================================
# Flat Joint Conditional Diffusion Model v2
# Single model, modality-specific projections (no padding!)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Cosine Schedule (Nichol & Dhariwal 2021)
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    return betas.float()


def make_schedule(T, schedule_type="cosine"):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = torch.linspace(1e-4, 0.02, T)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "snr": alpha_bar / (1.0 - alpha_bar),
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
    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )

    def forward(self, x, t_emb):
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.linear1(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.linear2(h)
        h = self.dropout(h)

        return x + h


# ============================================================
# Constants
# ============================================================
MODALITY_IDS = {"phone": 0, "watch": 1, "glasses": 2}

FLAT_DIMS = {
    "phone":   32 * 100,  # 3200
    "watch":   32 * 34,   # 1088
    "glasses": 16 * 10,   # 160
}


# ============================================================
# Joint Flat Denoiser v2 - modality-specific projections
# ============================================================
class JointFlatDenoiser(nn.Module):
    """
    Single denoiser for all modalities.
    Each modality has its own input/output projection to/from hidden_dim.
    No padding needed!
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

        # Modality embedding
        self.modality_emb = nn.Embedding(3, hidden_dim)

        # Per-modality INPUT projections (noisy target → hidden)
        self.input_projs = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim) for name, dim in FLAT_DIMS.items()
        })

        # Per-modality OUTPUT projections (hidden → noise prediction)
        self.output_projs = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim) for name, dim in FLAT_DIMS.items()
        })

        # Condition encoders: one per modality
        self.cond_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            ) for name, dim in FLAT_DIMS.items()
        })

        # Condition fusion (combine multiple condition embeddings)
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Shared backbone
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, time_dim, dropout) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

        # Init output projections to zero
        for proj in self.output_projs.values():
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, z_t, t, target_modality, phone_cond=None,
                watch_cond=None, glasses_cond=None):
        """
        z_t:              (B, flat_dim_target) - noisy target (actual size, no padding!)
        t:                (B,)
        target_modality:  str - "phone", "watch", or "glasses"
        phone_cond:       (B, 3200) or None
        watch_cond:       (B, 1088) or None
        glasses_cond:     (B, 160) or None
        """
        B = z_t.shape[0]
        device = z_t.device

        # Time embedding
        t_emb = self.time_mlp(t)  # (B, time_dim)

        # Modality embedding
        mod_id = torch.full((B,), MODALITY_IDS[target_modality],
                            device=device, dtype=torch.long)
        m_emb = self.modality_emb(mod_id)  # (B, hidden_dim)

        # Input projection (modality-specific)
        h = self.input_projs[target_modality](z_t)  # (B, hidden_dim)

        # Encode conditions
        cond_embs = []
        for name, cond in [("phone", phone_cond), ("watch", watch_cond),
                           ("glasses", glasses_cond)]:
            if cond is not None and name != target_modality:
                cond_embs.append(self.cond_encoders[name](cond))

        # Fuse conditions
        if len(cond_embs) > 0:
            # Average condition embeddings then fuse
            cond_avg = torch.stack(cond_embs, dim=0).mean(dim=0)
            c_emb = self.cond_fusion(cond_avg)
        else:
            c_emb = torch.zeros(B, self.hidden_dim, device=device)

        # Combine
        h = h + c_emb + m_emb

        # Backbone
        for block in self.blocks:
            h = block(h, t_emb)

        h = self.output_norm(h)

        # Output projection (modality-specific)
        return self.output_projs[target_modality](h)


# ============================================================
# DDIM Sampler
# ============================================================
@torch.no_grad()
def ddim_sample(model, target_modality, sched, T,
                phone_cond=None, watch_cond=None, glasses_cond=None,
                ddim_steps=200, eta=0.0, clip_range=5.0, device="cuda"):
    """DDIM sampling with quadratic timestep spacing."""

    # Determine batch size from conditions
    B = None
    for c in [phone_cond, watch_cond, glasses_cond]:
        if c is not None:
            B = c.shape[0]
            break

    target_dim = FLAT_DIMS[target_modality]

    # Quadratic timestep spacing
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)

    # Start from noise (actual target dim, no padding!)
    z = torch.randn(B, target_dim, device=device)

    alpha_bar = sched["alpha_bar"].to(device)

    for i in range(len(tau) - 1):
        t_now = tau[i]
        t_next = tau[i + 1]

        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        eps_pred = model(z, t_batch, target_modality,
                         phone_cond=phone_cond,
                         watch_cond=watch_cond,
                         glasses_cond=glasses_cond)

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

    return z
