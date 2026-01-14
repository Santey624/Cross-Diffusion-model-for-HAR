# src/models/baseline_har.py

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from src.data.modality_registry import MODALITIES


class Conv1DModalityEncoder(nn.Module):
    """
    Input:  [B, T, C]
    Output: [B, D]
    """
    def __init__(self, in_channels: int, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        h = self.net(x)              # [B, D, T]
        h = self.pool(h).squeeze(-1) # [B, D]
        return h


class BaselineHAR(nn.Module):
    """
    Multimodal baseline HAR:
      modalities -> per-modality CNN encoder -> concat -> z -> MLP classifier

    We keep masks explicit: if mask[m]=0, we use a learned "null embedding" for that modality.
    """
    def __init__(
        self,
        modality_names: List[str],
        num_classes: int,
        d_per_mod: int = 128,
        mlp_hidden: Tuple[int, int] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modality_names = modality_names
        self.d_per_mod = d_per_mod

        # Encoders and null embeddings per modality
        self.encoders = nn.ModuleDict()
        self.null_emb = nn.ParameterDict()

        for m in modality_names:
            in_ch = MODALITIES[m].channels
            self.encoders[m] = Conv1DModalityEncoder(in_channels=in_ch, d_out=d_per_mod)
            self.null_emb[m] = nn.Parameter(torch.zeros(d_per_mod))

        z_dim = d_per_mod * len(modality_names)

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, mlp_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden[0], mlp_hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden[1], num_classes),
        )

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor],
        return_z: bool = False,
    ):
        feats = []
        for m in self.modality_names:
            x = modalities[m]              # [B, T, C]
            msk = mask[m].float()          # [B] or [B,]
            # Encode present
            f = self.encoders[m](x)        # [B, D]
            # Replace missing with learned null embedding
            null = self.null_emb[m].unsqueeze(0).expand_as(f)  # [B, D]
            msk2 = msk.view(-1, 1)         # [B,1]
            f = msk2 * f + (1.0 - msk2) * null
            feats.append(f)

        z = torch.cat(feats, dim=1)        # [B, D_total]
        logits = self.classifier(z)        # [B, num_classes]
        if return_z:
            return logits, z
        return logits
