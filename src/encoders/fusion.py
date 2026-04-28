import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):

    def __init__(self, common_dim: int=128, n_heads: int=4, n_layers: int=2, ff_dim: int=256, output_dim: int=128):
        super().__init__()
        self.proj_imu = nn.Sequential(nn.Linear(256, common_dim), nn.LayerNorm(common_dim))
        self.proj_cardio = nn.Sequential(nn.Linear(128, common_dim), nn.LayerNorm(common_dim))
        self.proj_feat = nn.Sequential(nn.Linear(64, common_dim), nn.LayerNorm(common_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=common_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.weight_head = nn.Linear(common_dim, 1)
        self.output_dim = output_dim

    def forward(self, h_imu: torch.Tensor, h_cardio: torch.Tensor, h_feat: torch.Tensor) -> torch.Tensor:
        p_imu = self.proj_imu(h_imu).unsqueeze(1)
        p_cardio = self.proj_cardio(h_cardio).unsqueeze(1)
        p_feat = self.proj_feat(h_feat).unsqueeze(1)
        H = torch.cat([p_imu, p_cardio, p_feat], dim=1)
        H = self.transformer(H)
        weights = F.softmax(self.weight_head(H), dim=1)
        h_fused = (weights * H).sum(dim=1)
        return h_fused