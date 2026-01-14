import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.data.modality_registry import MODALITIES
from src.models.baseline_har import Conv1DModalityEncoder


class VAEHAR(nn.Module):
    def __init__(
        self,
        modality_names: List[str],
        num_classes: int,
        d_per_mod: int = 128,
        z_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modality_names = modality_names

        # -------- Encoders (same as baseline) --------
        self.encoders = nn.ModuleDict()
        self.null_emb = nn.ParameterDict()
        for m in modality_names:
            in_ch = MODALITIES[m].channels
            self.encoders[m] = Conv1DModalityEncoder(in_ch, d_out=d_per_mod)
            self.null_emb[m] = nn.Parameter(torch.zeros(d_per_mod))

        feat_dim = d_per_mod * len(modality_names)

        # -------- VAE heads --------
        self.fc_mu = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)

        # -------- Classifier (unchanged idea) --------
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # -------- Decoder (simple, per-modality MLP) --------
        self.decoders = nn.ModuleDict()
        for m in modality_names:
            T = MODALITIES[m].expected_length
            C = MODALITIES[m].channels
            self.decoders[m] = nn.Sequential(
                nn.Linear(z_dim, 256),
                nn.ReLU(),
                nn.Linear(256, T * C),
            )

    def encode(self, modalities: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
        feats = []
        for m in self.modality_names:
            x = modalities[m]           # [B,T,C]
            msk = mask[m].float().view(-1, 1)
            f = self.encoders[m](x)     # [B,d]
            null = self.null_emb[m].unsqueeze(0).expand_as(f)
            f = msk * f + (1.0 - msk) * null
            feats.append(f)
        h = torch.cat(feats, dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon = {}
        for m in self.modality_names:
            T = MODALITIES[m].expected_length
            C = MODALITIES[m].channels
            xhat = self.decoders[m](z).view(-1, T, C)
            recon[m] = xhat
        return recon

    def forward(self, modalities, mask):
        mu, logvar = self.encode(modalities, mask)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        recon = self.decode(z)
        return logits, recon, mu, logvar, z
