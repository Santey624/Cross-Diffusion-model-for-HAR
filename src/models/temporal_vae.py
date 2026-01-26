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
# Encoder
# ============================================================
class TemporalConvEncoder1D(nn.Module):
    """
    Input : (B, T, C)
    Output: mu, logvar -> (B, D, T')
    """
    def __init__(self, in_channels, latent_dim, base_channels=64):
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
# Decoder
# ============================================================
class TemporalConvDecoder1D(nn.Module):
    """
    Input : z (B, D, T')
    Output: recon (B, T, C)
    """
    def __init__(self, latent_dim, out_channels, out_length, base_channels=64):
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
        x = self.net(z)  # (B, C, T_dec)

        # crop / pad
        T = x.shape[-1]
        if T > self.out_length:
            x = x[..., :self.out_length]
        elif T < self.out_length:
            x = F.pad(x, (0, self.out_length - T))

        return x.transpose(1, 2)  # (B, T, C)
    


# ============================================================
# Single-Modal Temporal VAE  ✅ DAS HAT DIR GEFEHLT
# ============================================================
class TemporalSingleModalVAE(nn.Module):
    def __init__(self, in_channels, seq_len, latent_dim, base_channels=64):
        super().__init__()

        self.encoder = TemporalConvEncoder1D(
            in_channels, latent_dim, base_channels
        )
        self.decoder = TemporalConvDecoder1D(
            latent_dim, in_channels, seq_len, base_channels
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z
        }


# ============================================================
# Multimodal Temporal VAE (Variante B)
# ============================================================
class TemporalMultiModalVAE(nn.Module):
    def __init__(self, z_phone=32, z_watch=32, z_glasses=16):
        super().__init__()

        self.phone = TemporalSingleModalVAE(
            in_channels=12, seq_len=800, latent_dim=z_phone, base_channels=64
        )
        self.watch = TemporalSingleModalVAE(
            in_channels=6, seq_len=268, latent_dim=z_watch, base_channels=64
        )
        self.glasses = TemporalSingleModalVAE(
            in_channels=3, seq_len=80, latent_dim=z_glasses, base_channels=32
        )

    def forward(self, phone, watch, glasses):
        return {
            "phone": self.phone(phone),
            "watch": self.watch(watch),
            "glasses": self.glasses(glasses),
        }
    
    def encode_mu(self, phone, watch, glasses):
        """
        Returns concatenated encoder means (mu) for diffusion.
        Shape: [B, z_phone + z_watch + z_glasses]
        """
        out = self.forward(phone, watch, glasses)

        mu_phone = out["phone"]["mu"]        # (B, Dp, T')
        mu_watch = out["watch"]["mu"]        # (B, Dw, T')
        mu_glasses = out["glasses"]["mu"]    # (B, Dg, T')

        # temporal aggregation (VERY IMPORTANT)
        mu_phone = mu_phone.mean(dim=2)
        mu_watch = mu_watch.mean(dim=2)
        mu_glasses = mu_glasses.mean(dim=2)

        mu = torch.cat([mu_phone, mu_watch, mu_glasses], dim=1)
        return mu
