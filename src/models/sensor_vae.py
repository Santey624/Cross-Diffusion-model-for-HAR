# ============================================================
# Sensor-Level Multimodal VAE
# 7 independent single-sensor VAEs (each 3ch, z=8)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utils
# ============================================================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ============================================================
# Encoder: (B, T, C) -> mu, logvar (B, D, T')
# ============================================================
class TemporalConvEncoder1D(nn.Module):
    def __init__(self, in_channels, latent_dim, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, latent_dim * 2, 5, stride=2, padding=2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, C, T)
        h = self.net(x)                # (B, 2D, T')
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


# ============================================================
# Decoder: z (B, D, T') -> recon (B, T, C)
# ============================================================
class TemporalConvDecoder1D(nn.Module):
    def __init__(self, latent_dim, out_channels, out_length, base_channels=32):
        super().__init__()
        self.out_length = out_length
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.net(z)
        T = x.shape[-1]
        if T > self.out_length:
            x = x[..., :self.out_length]
        elif T < self.out_length:
            x = F.pad(x, (0, self.out_length - T))
        return x.transpose(1, 2)  # (B, T, C)


# ============================================================
# Single-Sensor VAE
# ============================================================
class SingleSensorVAE(nn.Module):
    def __init__(self, in_channels, seq_len, latent_dim, base_channels=32):
        super().__init__()
        self.encoder = TemporalConvEncoder1D(in_channels, latent_dim, base_channels)
        self.decoder = TemporalConvDecoder1D(latent_dim, in_channels, seq_len, base_channels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}


# ============================================================
# Sensor specs
# ============================================================
SENSOR_SPECS = {
    "phone_acc":   {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "phone_gyro":  {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "phone_grav":  {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "phone_lacc":  {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "watch_acc":   {"in_channels": 3, "seq_len": 268, "latent_dim": 8, "base_channels": 32},
    "watch_gyro":  {"in_channels": 3, "seq_len": 268, "latent_dim": 8, "base_channels": 32},
    "glasses_acc": {"in_channels": 3, "seq_len": 80,  "latent_dim": 8, "base_channels": 32},
}

SENSOR_NAMES = list(SENSOR_SPECS.keys())


# ============================================================
# Multimodal VAE (7 sensors)
# ============================================================
class SensorMultiModalVAE(nn.Module):
    """
    Multimodal VAE with one SingleSensorVAE per sensor.

    Forward input:  dict {sensor_name: (B, T, 3)}
    Forward output: dict {sensor_name: {"recon", "mu", "logvar", "z"}}
    """

    def __init__(self, sensor_specs=None):
        super().__init__()
        specs = sensor_specs or SENSOR_SPECS
        self.sensor_names = list(specs.keys())
        self.vaes = nn.ModuleDict({
            name: SingleSensorVAE(**params)
            for name, params in specs.items()
        })

    def forward(self, sensor_data):
        outputs = {}
        for name in self.sensor_names:
            if name in sensor_data:
                outputs[name] = self.vaes[name](sensor_data[name])
        return outputs
