# ============================================================
# Sensor-Level Multimodal VAE — Shared Latent Space
# Shared encoder/decoder weights, per-sensor input/output projections
# All latents: (B, D, T_SHARED) — uniform shape across sensors
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sensor specs
# ============================================================
SENSOR_SPECS = {
    "phone_acc":   {"in_channels": 3, "seq_len": 800},
    "phone_gyro":  {"in_channels": 3, "seq_len": 800},
    "phone_grav":  {"in_channels": 3, "seq_len": 800},
    "phone_lacc":  {"in_channels": 3, "seq_len": 800},
    "watch_acc":   {"in_channels": 3, "seq_len": 268},
    "watch_gyro":  {"in_channels": 3, "seq_len": 268},
    "glasses_acc": {"in_channels": 3, "seq_len": 80},
}

SENSOR_NAMES = list(SENSOR_SPECS.keys())

T_SHARED = 32
LATENT_DIM = 8
BASE_CHANNELS = 32


# ============================================================
# Utils
# ============================================================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ============================================================
# Shared Encoder: (B, base_channels, T_any) -> mu, logvar (B, D, T_SHARED)
# ============================================================
class SharedEncoder(nn.Module):
    def __init__(self, base_channels=32, latent_dim=8, t_shared=32):
        super().__init__()
        self.t_shared = t_shared
        self.net = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, latent_dim * 2, 5, stride=2, padding=2),
        )

    def forward(self, x):
        # x: (B, base_channels, T)
        h = self.net(x)  # (B, 2D, T/8)
        h = F.interpolate(h, size=self.t_shared, mode='linear', align_corners=False)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar  # each (B, D, T_SHARED)


# ============================================================
# Shared Decoder: z (B, D, T_SHARED) -> (B, base_channels, T_intermediate)
# ============================================================
class SharedDecoder(nn.Module):
    def __init__(self, latent_dim=8, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, base_channels, 4, stride=2, padding=1),
        )

    def forward(self, z):
        # z: (B, D, T_SHARED)
        return self.net(z)  # (B, base_channels, T_SHARED * 8)


# ============================================================
# Multimodal VAE with Shared Latent Space
# ============================================================
class SensorMultiModalVAE(nn.Module):
    """
    Shared-weight VAE for 7 sensor modalities.

    Architecture:
    - Per-sensor input projection: Conv1d(3, base_channels, 1)
    - Shared encoder: Conv1d stride-2 x3 + interpolate to T_SHARED
    - Shared decoder: ConvTranspose1d x3
    - Per-sensor output projection: Conv1d(base_channels, 3, 1) + interpolate to seq_len

    All latents are (B, D, T_SHARED) — same shape regardless of sensor.
    """

    def __init__(self, sensor_specs=None, latent_dim=LATENT_DIM,
                 base_channels=BASE_CHANNELS, t_shared=T_SHARED):
        super().__init__()
        specs = sensor_specs or SENSOR_SPECS
        self.sensor_names = list(specs.keys())
        self.sensor_specs = specs
        self.t_shared = t_shared

        # Per-sensor input projections (3 -> base_channels)
        self.input_projs = nn.ModuleDict({
            name: nn.Conv1d(params["in_channels"], base_channels, 1)
            for name, params in specs.items()
        })

        # Shared encoder and decoder
        self.encoder = SharedEncoder(base_channels, latent_dim, t_shared)
        self.decoder = SharedDecoder(latent_dim, base_channels)

        # Per-sensor output projections (base_channels -> 3)
        self.output_projs = nn.ModuleDict({
            name: nn.Conv1d(base_channels, params["in_channels"], 1)
            for name, params in specs.items()
        })

    def encode_sensor(self, name, x):
        """Encode a single sensor signal.
        x: (B, T, C) -> mu, logvar: (B, D, T_SHARED)
        """
        x = x.transpose(1, 2)                     # (B, C, T)
        h = self.input_projs[name](x)             # (B, base_channels, T)
        mu, logvar = self.encoder(h)               # (B, D, T_SHARED)
        return mu, logvar

    def decode_sensor(self, name, z):
        """Decode a latent back to sensor signal.
        z: (B, D, T_SHARED) -> recon: (B, T, C)
        """
        h = self.decoder(z)                        # (B, base_channels, T_SHARED*8)
        h = self.output_projs[name](h)            # (B, C, T_SHARED*8)
        seq_len = self.sensor_specs[name]["seq_len"]
        h = F.interpolate(h, size=seq_len, mode='linear', align_corners=False)
        return h.transpose(1, 2)                   # (B, T, C)

    def forward(self, sensor_data):
        """
        sensor_data: dict {name: (B, T, C)}
        Returns: dict {name: {"recon", "mu", "logvar", "z"}}
        """
        outputs = {}
        for name in self.sensor_names:
            if name in sensor_data:
                mu, logvar = self.encode_sensor(name, sensor_data[name])
                z = reparameterize(mu, logvar)
                recon = self.decode_sensor(name, z)
                outputs[name] = {
                    "recon": recon,
                    "mu": mu,
                    "logvar": logvar,
                    "z": z,
                }
        return outputs
