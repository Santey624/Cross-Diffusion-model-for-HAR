# ============================================================
# Sensor Shared+Private Multimodal VAE (V4)
#
# Architecture:
#   - z_shared  (D_SHARED, T_LAT): activity latent shared across ALL sensors
#                                   via Product of Experts (PoE)
#   - z_private (D_PRIVATE, T_LAT): sensor-specific latent per sensor
#
# Key property:
#   Cross-sensor R²(z_shared) ≈ 1.0 by design, because z_shared is
#   inferred from ALL available sensors jointly — not independently.
#
# Imputation:
#   missing sensor → decode(z_shared_from_others, prior_private)
#   No diffusion needed for z_shared; diffusion improves z_private.
#
# Latent dims: D_SHARED=8, D_PRIVATE=8 → total=16 (same as V2/V3)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


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

T_LAT        = 64
D_SHARED     = 8
D_PRIVATE    = 8
BASE_CHANNELS = 32


# ============================================================
# Utils
# ============================================================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar.clamp(-8, 8))
    return mu + std * torch.randn_like(std)


def product_of_experts(mu_list, logvar_list):
    """
    Combine Gaussian posteriors via Product of Experts.
    Adds a unit-variance N(0,1) prior to ensure non-degenerate result.

    Args:
        mu_list:     list of (B, D, T) tensors
        logvar_list: list of (B, D, T) tensors

    Returns:
        mu_poe, logvar_poe — each (B, D, T)
    """
    # precision = 1 / sigma² = exp(-logvar)
    precisions = [torch.exp(-lv.clamp(-8, 8)) for lv in logvar_list]

    # Include N(0,1) prior: precision = 1, contribution to mu = 0
    total_prec = torch.ones_like(precisions[0])          # prior
    total_prec = total_prec + sum(precisions)             # + experts

    mu_poe = sum(m * p for m, p in zip(mu_list, precisions)) / total_prec
    logvar_poe = -torch.log(total_prec.clamp(min=1e-8))

    return mu_poe, logvar_poe


# ============================================================
# Building blocks
# ============================================================
class ConvEncoder(nn.Module):
    """(B, in_ch, T_any) → mu, logvar each (B, out_dim, T_lat)"""

    def __init__(self, in_channels, out_dim, t_lat):
        super().__init__()
        self.t_lat = t_lat
        ch = in_channels
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch,     5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch * 2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(ch * 2, out_dim * 2, 5, stride=2, padding=2),
        )

    def forward(self, x):
        h = self.net(x)
        h = F.interpolate(h, size=self.t_lat, mode='linear', align_corners=False)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class ConvDecoder(nn.Module):
    """(B, in_dim, T_lat) → (B, out_ch, T_lat * 8)"""

    def __init__(self, in_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_dim,          out_channels * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(out_channels * 2, out_channels,    4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(out_channels,     out_channels,    4, stride=2, padding=1),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# Main model
# ============================================================
class SensorSharedPrivateVAE(nn.Module):
    """
    Shared+Private Multimodal VAE for sensor imputation.

    z_shared  — D_SHARED dims, same for all sensors (activity).
    z_private — D_PRIVATE dims, per-sensor (device-specific patterns).

    During forward():
      - All present sensors predict (mu_s, logvar_s) via shared encoder.
      - PoE combines predictions → single z_shared.
      - Each sensor also predicts z_private independently.
      - Decoder: concat(z_shared, z_private) → reconstruction.

    Random masking during training (mask_ratio) forces z_shared to be
    inferable from any subset of sensors (robustness for imputation).
    """

    def __init__(self, sensor_specs=None, d_shared=D_SHARED, d_private=D_PRIVATE,
                 base_channels=BASE_CHANNELS, t_lat=T_LAT):
        super().__init__()
        specs = sensor_specs or SENSOR_SPECS
        self.sensor_names  = list(specs.keys())
        self.sensor_specs  = specs
        self.t_lat         = t_lat
        self.d_shared      = d_shared
        self.d_private     = d_private

        # Per-sensor input projection: (B, 3, T) → (B, base_ch, T)
        self.input_projs = nn.ModuleDict({
            name: nn.Conv1d(params["in_channels"], base_channels, 1)
            for name, params in specs.items()
        })

        # Shared encoder — SAME weights for all sensors
        self.shared_encoder = ConvEncoder(base_channels, d_shared, t_lat)

        # Private encoders — SEPARATE weights per sensor
        self.private_encoders = nn.ModuleDict({
            name: ConvEncoder(base_channels, d_private, t_lat)
            for name in specs.keys()
        })

        # Decoder: z_shared + z_private → base_ch features
        self.decoder = ConvDecoder(d_shared + d_private, base_channels)

        # Per-sensor output projection: (B, base_ch, T) → (B, C, T)
        self.output_projs = nn.ModuleDict({
            name: nn.Conv1d(base_channels, params["in_channels"], 1)
            for name, params in specs.items()
        })

    # ----------------------------------------------------------
    def _project_input(self, name, x):
        """x: (B, T, C) → (B, base_ch, T)"""
        return self.input_projs[name](x.transpose(1, 2))

    def encode_sensor(self, name, x):
        """
        x: (B, T, C)
        Returns: mu_s, logvar_s, mu_p, logvar_p — each (B, D, T_lat)
        """
        h = self._project_input(name, x)
        mu_s, logvar_s = self.shared_encoder(h)
        mu_p, logvar_p = self.private_encoders[name](h)
        return mu_s, logvar_s, mu_p, logvar_p

    def decode_sensor(self, name, z_shared, z_private):
        """
        z_shared:  (B, D_shared,  T_lat)
        z_private: (B, D_private, T_lat)
        Returns:   (B, T, C)
        """
        z   = torch.cat([z_shared, z_private], dim=1)   # (B, D_s+D_p, T_lat)
        h   = self.decoder(z)                            # (B, base_ch, T_lat*8)
        h   = self.output_projs[name](h)                 # (B, C, T_lat*8)
        seq = self.sensor_specs[name]["seq_len"]
        h   = F.interpolate(h, size=seq, mode='linear', align_corners=False)
        return h.transpose(1, 2)                         # (B, T, C)

    # ----------------------------------------------------------
    def forward(self, sensor_data, mask_ratio=0.0):
        """
        Args:
            sensor_data: dict {name: (B, T, C)}
            mask_ratio:  probability of randomly dropping a sensor from PoE
                         (0.0 = no masking, 0.5 = drop half on average)

        Returns:
            outputs:        dict {name: {"recon","mu_s","logvar_s","mu_p","logvar_p","z_s","z_p"}}
            mu_shared:      PoE posterior mean   (B, D_shared, T_lat)
            logvar_shared:  PoE posterior logvar (B, D_shared, T_lat)
        """
        present = [n for n in self.sensor_names if n in sensor_data]

        # Encode all present sensors
        enc = {}
        for name in present:
            mu_s, logvar_s, mu_p, logvar_p = self.encode_sensor(name, sensor_data[name])
            enc[name] = (mu_s, logvar_s, mu_p, logvar_p)

        # Optionally mask some sensors from PoE (training robustness)
        poe_names = present
        if mask_ratio > 0.0 and self.training and len(present) > 1:
            poe_names = [n for n in present
                         if torch.rand(1).item() > mask_ratio]
            if len(poe_names) == 0:
                poe_names = [present[0]]  # always keep at least one

        # Product of Experts → shared posterior
        mu_s_list     = [enc[n][0] for n in poe_names]
        logvar_s_list = [enc[n][1] for n in poe_names]
        mu_shared, logvar_shared = product_of_experts(mu_s_list, logvar_s_list)
        z_shared = reparameterize(mu_shared, logvar_shared)

        # Decode each present sensor
        outputs = {}
        for name in present:
            _, _, mu_p, logvar_p = enc[name]
            z_p   = reparameterize(mu_p, logvar_p)
            recon = self.decode_sensor(name, z_shared, z_p)
            outputs[name] = {
                "recon":    recon,
                "mu_s":     enc[name][0],
                "logvar_s": enc[name][1],
                "mu_p":     mu_p,
                "logvar_p": logvar_p,
                "z_s":      z_shared,
                "z_p":      z_p,
            }

        return outputs, mu_shared, logvar_shared

    # ----------------------------------------------------------
    def impute(self, sensor_data, target_name):
        """
        Impute a missing sensor from available ones.

        Args:
            sensor_data:  dict of available sensors (target NOT included)
            target_name:  which sensor to impute

        Returns: (B, T, C) imputed signal
        """
        present = [n for n in self.sensor_names
                   if n in sensor_data and n != target_name]

        if len(present) == 0:
            B   = next(iter(sensor_data.values())).shape[0]
            seq = self.sensor_specs[target_name]["seq_len"]
            C   = self.sensor_specs[target_name]["in_channels"]
            return torch.zeros(B, seq, C, device=next(self.parameters()).device)

        mu_s_list, logvar_s_list = [], []
        for name in present:
            h = self._project_input(name, sensor_data[name])
            mu_s, logvar_s = self.shared_encoder(h)
            mu_s_list.append(mu_s)
            logvar_s_list.append(logvar_s)

        mu_shared, logvar_shared = product_of_experts(mu_s_list, logvar_s_list)
        z_shared = reparameterize(mu_shared, logvar_shared)

        # Prior for missing sensor's private latent
        z_private = torch.zeros(z_shared.shape[0], self.d_private, self.t_lat,
                                device=z_shared.device)

        return self.decode_sensor(target_name, z_shared, z_private)
