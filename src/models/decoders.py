import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDecoder1D(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        out_length: int,
        base_channels: int,
        seed_len: int = 25
    ):
        super().__init__()

        self.out_length = out_length
        self.seed_len = seed_len
        self.base_channels = base_channels

        # ---- Latent → temporal seed ----
        self.fc = nn.Linear(latent_dim, base_channels * seed_len)

        # ---- Upsampling network ----
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(base_channels // 8, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        returns: (B, T, C)
        """
        B = z.size(0)

        # (B, latent_dim) → (B, C, seed_len)
        h = self.fc(z).view(B, self.base_channels, self.seed_len)

        x = self.net(h)

        # ---- Crop / pad to exact target length ----
        if x.shape[-1] > self.out_length:
            x = x[..., :self.out_length]
        elif x.shape[-1] < self.out_length:
            pad = self.out_length - x.shape[-1]
            x = F.pad(x, (0, pad))

        return x.transpose(1, 2)  # (B, T, C)
