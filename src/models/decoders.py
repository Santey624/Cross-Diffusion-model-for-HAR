# src/models/decoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDecoder1D(nn.Module):
    def __init__(self, latent_dim, out_channels, out_length):
        super().__init__()

        self.out_length = out_length

        self.fc = nn.Linear(latent_dim, 256)

        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, out_channels, kernel_size=4, stride=2)
        )

    def forward(self, z):
        h = self.fc(z).unsqueeze(-1)  # (B, 256, 1)
        x = self.net(h)

        # crop / pad to exact length
        if x.shape[-1] > self.out_length:
            x = x[..., :self.out_length]
        elif x.shape[-1] < self.out_length:
            pad = self.out_length - x.shape[-1]
            x = F.pad(x, (0, pad))

        return x.transpose(1, 2)  # (B, T, C)
