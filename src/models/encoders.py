# src/models/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# src/models/encoders.py

class TemporalConvEncoder1D(nn.Module):
    def __init__(self, in_channels, latent_dim, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels*2, latent_dim*2, 5, stride=2, padding=2),
        )

    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        h = self.net(x)              # (B, 2D, T')
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar            # (B, D, T')

