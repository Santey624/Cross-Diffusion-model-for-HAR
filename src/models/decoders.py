import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvDecoder1D(nn.Module):
    def __init__(self, latent_dim, out_channels, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, base_channels*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.net(z)              # (B, C, T)
        return x.transpose(1, 2)