# src/models/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B, 256, 1)
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (B, T, C) → Conv1D erwartet (B, C, T)
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
