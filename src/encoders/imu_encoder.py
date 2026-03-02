import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingWindowCNNBlock(nn.Module):
    """Multi-scale 1D CNN with three parallel kernel sizes."""
    def __init__(self, in_channels: int = 9, out_channels: int = 256):
        super().__init__()
        mid = out_channels // 3

        self.branch_5  = self._conv_block(in_channels, mid, kernel_size=5)
        self.branch_7  = self._conv_block(in_channels, mid, kernel_size=7)
        self.branch_11 = self._conv_block(in_channels, out_channels - 2 * mid, kernel_size=11)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def _conv_block(self, in_ch, out_ch, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → transpose to [B, C, T] for Conv1d
        x = x.permute(0, 2, 1)
        out5  = self.pool(self.branch_5(x)).squeeze(-1)
        out7  = self.pool(self.branch_7(x)).squeeze(-1)
        out11 = self.pool(self.branch_11(x)).squeeze(-1)
        return torch.cat([out5, out7, out11], dim=-1)  # [B, 256]


class ChannelTimeAttentionTransformer(nn.Module):
    """3-layer transformer that attends across both time and channel dimensions."""
    def __init__(self, embed_dim: int = 256, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256] — treat as single-token sequence
        x = x.unsqueeze(1)          # [B, 1, 256]
        out = self.transformer(x)   # [B, 1, 256]
        return self.norm(out.squeeze(1))  # [B, 256]


class SWCTNet(nn.Module):
    """
    IMU Encoder: SlidingWindowCNN + ChannelTimeAttentionTransformer
    Input:  [B, 1000, 9]  (20s window, 9 IMU channels)
    Output: h_imu [B, 256]
    """
    def __init__(self, in_channels: int = 9, output_dim: int = 256,
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.sw_cnn = SlidingWindowCNNBlock(in_channels, output_dim)
        self.ctat   = ChannelTimeAttentionTransformer(output_dim, n_heads, n_layers, dropout)

        # Activity classification head (used during supervised fine-tuning)
        self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sw_cnn(x)    # [B, 256]
        h = self.ctat(h)      # [B, 256]
        return h

    def build_classifier(self, n_classes: int):
        self.classifier = nn.Linear(256, n_classes)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward(x)
        assert self.classifier is not None, "Call build_classifier first."
        return self.classifier(h)


class ProjectionHead(nn.Module):
    """Contrastive learning projection head — not part of the encoder at inference."""
    def __init__(self, in_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)
