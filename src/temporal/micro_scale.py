import torch
import torch.nn as nn
import math

class DilatedResidualBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, :x.shape[2]]
        return self.relu(out + x)

class TCNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, dilations: list=None, dropout: float=0.1):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8]
        self.input_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.blocks = nn.ModuleList([DilatedResidualBlock(out_channels, kernel_size, d, dropout) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

class MicroScaleModel(nn.Module):

    def __init__(self, input_dim: int=128, tcn_channels: int=256, tcn_kernel: int=3, dilations: list=None, transformer_heads: int=8, transformer_layers: int=3, transformer_dim: int=256, ff_dim: int=512, sequence_len: int=180, output_dim: int=256, dropout: float=0.1):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8]
        self.tcn = TCNBlock(input_dim, tcn_channels, tcn_kernel, dilations, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_len + 1, transformer_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.output_dim = output_dim
        self.hr_head = nn.Linear(output_dim, 1)
        self.hrv_head = nn.Linear(output_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        return self.norm(x[:, 0])

    def predict(self, x: torch.Tensor):
        z = self.forward(x)
        return (self.hr_head(z), self.hrv_head(z))