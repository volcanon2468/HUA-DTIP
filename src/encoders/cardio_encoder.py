import torch
import torch.nn as nn

class ResBlock1D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad), nn.BatchNorm1d(out_channels))
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))

class TemporalSelfAttention(nn.Module):

    def __init__(self, embed_dim: int=128, n_heads: int=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        return self.norm((x + attn_out).squeeze(1))

class CardioEncoder(nn.Module):

    def __init__(self, max_in_channels: int=2, output_dim: int=128, attn_heads: int=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(max_in_channels, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.res1 = ResBlock1D(64, 64, kernel_size=7)
        self.res2 = ResBlock1D(64, 128, kernel_size=5)
        self.res3 = ResBlock1D(128, 128, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn = TemporalSelfAttention(embed_dim=output_dim, n_heads=attn_heads)
        self.hr_head = nn.Linear(output_dim, 1)

    def _pad_channels(self, x: torch.Tensor, target_ch: int) -> torch.Tensor:
        if x.shape[1] < target_ch:
            pad = torch.zeros(x.shape[0], target_ch - x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self._pad_channels(x, 2)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1)
        x = self.attn(x)
        return x

    def predict_hr(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward(x)
        return self.hr_head(h)