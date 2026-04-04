# ============================================================
# Class-Conditional Diffusion on Raw Sensor Signals (no VAE)
#
# Operates directly on sensor signals interpolated to T_MODEL=256.
# Conditioning: timestep + activity class + sensor_id
#
# Architecture: 1D UNet with FiLM conditioning
#   256 → 128 → 64 → 32  (encoder)
#         32 → 64 → 128 → 256  (decoder + skip connections)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Native lengths rounded up to nearest multiple of 8 (required by UNet stride-2 ops x3)
# Phone: 800 (800/8=100 ✓), Watch: 272 (268→272), Glasses: 80 (80/8=10 ✓)
SENSOR_T = {
    "phone_acc":   800,
    "phone_gyro":  800,
    "phone_grav":  800,
    "phone_lacc":  800,
    "watch_acc":   272,
    "watch_gyro":  272,
    "glasses_acc": 80,
}
T_MODEL = 256   # fallback if sensor not in SENSOR_T


def sinusoidal_embedding(t, dim):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
    ).float()
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class FiLM(nn.Module):
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, n_channels * 2)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1  = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.conv2  = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film1  = FiLM(cond_dim, out_ch)
        self.film2  = FiLM(cond_dim, out_ch)
        self.down   = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)
        self.skip_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = F.silu(self.film1(self.norm1(self.conv1(x)), cond))
        h = self.film2(self.norm2(self.conv2(h)), cond)
        h = h + self.skip_proj(x)
        return self.down(h), h    # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim):
        super().__init__()
        self.up     = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv1  = nn.Conv1d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.conv2  = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film1  = FiLM(cond_dim, out_ch)
        self.film2  = FiLM(cond_dim, out_ch)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        h = F.silu(self.film1(self.norm1(self.conv1(x)), cond))
        h = self.film2(self.norm2(self.conv2(h)), cond)
        return h


class SignalClassDiffusion(nn.Module):
    """
    1D UNet class-conditional diffusion on raw sensor signals.

    Args:
        in_channels:  signal channels (3 for all sensors)
        n_sensors:    number of sensor types (for sensor_id embedding)
        n_classes:    number of activity classes
        base_ch:      base channel width
        emb_dim:      embedding dimension for timestep/class/sensor
    """

    def __init__(self, in_channels=3, n_sensors=7, n_classes=55,
                 base_ch=64, emb_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.t_model     = T_MODEL
        cond_dim         = emb_dim * 3

        # Embeddings
        self.t_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.emb_dim     = emb_dim
        self.class_emb   = nn.Embedding(n_classes, emb_dim)
        self.sensor_emb  = nn.Embedding(n_sensors, emb_dim)

        ch = base_ch
        # Encoder
        self.in_conv = nn.Conv1d(in_channels, ch, 3, padding=1)
        self.down1   = DownBlock(ch,     ch*2,  cond_dim)   # 256→128
        self.down2   = DownBlock(ch*2,   ch*4,  cond_dim)   # 128→64
        self.down3   = DownBlock(ch*4,   ch*8,  cond_dim)   # 64→32

        # Bottleneck
        self.mid1 = nn.Conv1d(ch*8, ch*8, 3, padding=1)
        self.mid2 = nn.Conv1d(ch*8, ch*8, 3, padding=1)
        self.mid_norm1 = nn.GroupNorm(min(8, ch*8), ch*8)
        self.mid_norm2 = nn.GroupNorm(min(8, ch*8), ch*8)
        self.mid_film1 = FiLM(cond_dim, ch*8)
        self.mid_film2 = FiLM(cond_dim, ch*8)

        # Decoder
        self.up3   = UpBlock(ch*8,  ch*8,  ch*4,  cond_dim)  # 32→64
        self.up2   = UpBlock(ch*4,  ch*4,  ch*2,  cond_dim)  # 64→128
        self.up1   = UpBlock(ch*2,  ch*2,  ch,    cond_dim)  # 128→256

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv1d(ch, in_channels, 1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t, t, class_label, sensor_id):
        """
        x_t:        (B, C, T_MODEL)   noisy signal
        t:          (B,)              timestep
        class_label:(B,)              activity class
        sensor_id:  (B,)              sensor index (0-6)

        Returns:    (B, C, T_MODEL)   predicted noise
        """
        t_e   = self.t_emb(sinusoidal_embedding(t, self.emb_dim))
        c_e   = self.class_emb(class_label)
        s_e   = self.sensor_emb(sensor_id)
        cond  = torch.cat([t_e, c_e, s_e], dim=1)    # (B, 3*emb_dim)

        x = self.in_conv(x_t)

        x, s1 = self.down1(x, cond)
        x, s2 = self.down2(x, cond)
        x, s3 = self.down3(x, cond)

        x = F.silu(self.mid_film1(self.mid_norm1(self.mid1(x)), cond))
        x = self.mid_film2(self.mid_norm2(self.mid2(x)), cond)

        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)

        return self.out(x)


def create_signal_class_diffusion(n_sensors=7, n_classes=55,
                                   in_channels=3, base_ch=64, emb_dim=128):
    return SignalClassDiffusion(
        in_channels=in_channels,
        n_sensors=n_sensors,
        n_classes=n_classes,
        base_ch=base_ch,
        emb_dim=emb_dim,
    )
