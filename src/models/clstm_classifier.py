# ============================================================
# C-LSTM-A Classifier on Decoded Sensor Signals (Option 2)
#
# Pipeline: latent z -> VAE decode -> (B, C, T) -> C-LSTM-A
#
# Architecture per sensor:
#   (B, 3, T) -> Conv1D x2 -> AdaptivePool -> BiLSTM -> (B, d_feat)
# Fusion:
#   Stack 7 sensor features -> Transformer (CLS token) -> n_classes
# ============================================================

import torch
import torch.nn as nn


class SensorBranch(nn.Module):
    """Per-sensor CNN + BiLSTM branch."""

    def __init__(self, in_channels=3, cnn_channels=64, lstm_hidden=64, pool_size=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        # BiLSTM: out_dim = lstm_hidden * 2
        self.lstm = nn.LSTM(
            cnn_channels * 2, lstm_hidden,
            batch_first=True, bidirectional=True,
        )
        self.out_dim = lstm_hidden * 2

    def forward(self, x):
        """x: (B, C, T) -> (B, out_dim)"""
        h = self.cnn(x)           # (B, cnn_ch*2, T//4)
        h = self.pool(h)          # (B, cnn_ch*2, pool_size)
        h = h.permute(0, 2, 1)   # (B, pool_size, cnn_ch*2)
        _, (hidden, _) = self.lstm(h)
        return torch.cat([hidden[0], hidden[1]], dim=1)  # (B, out_dim)


class CLSTMAttentionClassifier(nn.Module):
    """
    C-LSTM-A: Per-sensor CNN+BiLSTM branches + Cross-sensor Transformer.

    Input:  signals_dict {sensor_name: (B, C=3, T)}  (C, T format)
    Output: logits (B, n_classes)
    """

    def __init__(
        self,
        n_sensors=7,
        in_channels=3,
        cnn_channels=64,
        lstm_hidden=64,
        d_attn=128,
        n_heads=4,
        n_layers=2,
        n_classes=54,
        pool_size=16,
        dropout=0.3,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        branch_out = lstm_hidden * 2  # 128 for bidir

        # Per-sensor branches (separate weights — sensors have different characteristics)
        self.branches = nn.ModuleList([
            SensorBranch(in_channels, cnn_channels, lstm_hidden, pool_size)
            for _ in range(n_sensors)
        ])

        # Project branch output to attention dimension
        self.proj = nn.Linear(branch_out, d_attn)
        self.proj_norm = nn.LayerNorm(d_attn)

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_attn))

        # Cross-sensor Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_attn,
            nhead=n_heads,
            dim_feedforward=d_attn * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_attn),
            nn.Dropout(dropout),
            nn.Linear(d_attn, n_classes),
        )

    def forward(self, signals_dict, sensor_names):
        """
        Args:
            signals_dict: {sensor_name: (B, C, T)}  — decoded signals in (C, T) format
            sensor_names: list of sensor names (order determines branch index)

        Returns:
            logits: (B, n_classes)
        """
        B = next(iter(signals_dict.values())).shape[0]

        tokens = []
        for i, name in enumerate(sensor_names):
            x = signals_dict[name]          # (B, C, T)
            feat = self.branches[i](x)      # (B, branch_out)
            proj = self.proj_norm(self.proj(feat))  # (B, d_attn)
            tokens.append(proj)

        # Stack sensor tokens: (B, n_sensors, d_attn)
        x = torch.stack(tokens, dim=1)

        # Prepend CLS token: (B, 1+n_sensors, d_attn)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Cross-sensor attention
        x = self.transformer(x)

        # Use CLS token output for classification
        cls_out = x[:, 0]  # (B, d_attn)
        return self.head(cls_out)


def create_clstm_classifier(
    n_sensors=7,
    n_classes=54,
    in_channels=3,
    cnn_channels=64,
    lstm_hidden=64,
    d_attn=128,
    n_heads=4,
    n_layers=2,
    pool_size=16,
    dropout=0.3,
):
    return CLSTMAttentionClassifier(
        n_sensors=n_sensors,
        in_channels=in_channels,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        d_attn=d_attn,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
        pool_size=pool_size,
        dropout=dropout,
    )
