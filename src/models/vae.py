# src/models/vae.py

import torch
import torch.nn as nn
from src.models.encoders import ConvEncoder1D
from src.models.decoders import ConvDecoder1D


class MultiModalVAE(nn.Module):
    def __init__(self, z_device=32, z_fused=64):
        super().__init__()

        # Encoders
        self.enc_phone = ConvEncoder1D(12, z_device)
        self.enc_watch = ConvEncoder1D(6, z_device)
        self.enc_glasses = ConvEncoder1D(3, z_device)

        # Fusion
        self.fusion_mu = nn.Linear(3 * z_device, z_fused)
        self.fusion_logvar = nn.Linear(3 * z_device, z_fused)

        # Decoders
        self.dec_phone = ConvDecoder1D(
        latent_dim=z_fused,
        out_channels=12,
        out_length=800,
        base_channels=256,   # groß
        seed_len=25
        )

        self.dec_watch = ConvDecoder1D(
            latent_dim=z_fused,
            out_channels=6,
            out_length=268,
            base_channels=128,   # mittel
            seed_len=15
        )

        self.dec_glasses = ConvDecoder1D(
            latent_dim=z_fused,
            out_channels=3,
            out_length=80,
            base_channels=64,    # klein
            seed_len=10
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, phone, watch, glasses):
        mu_p, lv_p = self.enc_phone(phone)
        mu_w, lv_w = self.enc_watch(watch)
        mu_g, lv_g = self.enc_glasses(glasses)

        h = torch.cat([mu_p, mu_w, mu_g], dim=-1)

        mu_f = self.fusion_mu(h)
        logvar_f = self.fusion_logvar(h)

        z = self.reparameterize(mu_f, logvar_f)

        recon_phone = self.dec_phone(z)
        recon_watch = self.dec_watch(z)
        recon_glasses = self.dec_glasses(z)

        return {
            "recon_phone": recon_phone,
            "recon_watch": recon_watch,
            "recon_glasses": recon_glasses,
            "mu": mu_f,
            "logvar": logvar_f,
        }
