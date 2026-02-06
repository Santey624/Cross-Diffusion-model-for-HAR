# ============================================================
# Activity Classifier on Sensor Latents
# MLP/Transformer that takes 7 sensor latents → activity class
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentActivityClassifier(nn.Module):
    """
    MLP classifier on concatenated sensor latents.

    Input: 7 latents of shape (B, D, T_SHARED) → flatten → MLP → n_classes
    """

    def __init__(
        self,
        n_sensors=7,
        latent_dim=8,
        t_shared=32,
        n_classes=54,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.t_shared = t_shared

        # Input: 7 * 8 * 32 = 1792 features
        input_dim = n_sensors * latent_dim * t_shared

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, latents_dict, sensor_names):
        """
        Args:
            latents_dict: dict {sensor_name: (B, D, T_SHARED)}
            sensor_names: list of sensor names (order matters)

        Returns:
            logits: (B, n_classes)
        """
        # Concatenate all latents: (B, 7*D*T)
        latent_list = [latents_dict[name].flatten(1) for name in sensor_names]
        x = torch.cat(latent_list, dim=1)

        return self.mlp(x)


class LatentActivityTransformer(nn.Module):
    """
    Transformer classifier on sensor latents.

    Each sensor latent is a token, attention learns cross-sensor patterns.
    """

    def __init__(
        self,
        n_sensors=7,
        latent_dim=8,
        t_shared=32,
        n_classes=54,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.n_sensors = n_sensors

        # Project each latent (D*T) to d_model
        self.input_proj = nn.Linear(latent_dim * t_shared, d_model)

        # Learnable sensor embeddings
        self.sensor_embeddings = nn.Embedding(n_sensors, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, latents_dict, sensor_names):
        """
        Args:
            latents_dict: dict {sensor_name: (B, D, T_SHARED)}
            sensor_names: list of sensor names

        Returns:
            logits: (B, n_classes)
        """
        B = next(iter(latents_dict.values())).shape[0]
        device = next(iter(latents_dict.values())).device

        # Project each sensor latent
        tokens = []
        for i, name in enumerate(sensor_names):
            z = latents_dict[name].flatten(1)  # (B, D*T)
            proj = self.input_proj(z)  # (B, d_model)
            emb = self.sensor_embeddings(torch.tensor(i, device=device))
            tokens.append(proj + emb)

        # Stack: (B, n_sensors, d_model)
        x = torch.stack(tokens, dim=1)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_sensors, d_model)

        # Transformer
        x = self.transformer(x)

        # Take CLS output
        cls_out = x[:, 0]

        return self.head(cls_out)


def create_activity_classifier(
    model_type="mlp",
    n_sensors=7,
    latent_dim=8,
    t_shared=32,
    n_classes=54,
    **kwargs
):
    if model_type == "mlp":
        return LatentActivityClassifier(
            n_sensors=n_sensors,
            latent_dim=latent_dim,
            t_shared=t_shared,
            n_classes=n_classes,
            **kwargs
        )
    elif model_type == "transformer":
        return LatentActivityTransformer(
            n_sensors=n_sensors,
            latent_dim=latent_dim,
            t_shared=t_shared,
            n_classes=n_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
