import torch
import torch.nn as nn
import numpy as np


class NBEATSBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 backcast_dim: int, basis_type: str = "generic"):
        super().__init__()
        self.basis_type = basis_type
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.backcast_head = nn.Linear(hidden_dim, backcast_dim)
        self.forecast_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class MacroScaleModel(nn.Module):
    def __init__(self, input_dim: int = 640, block_dim: int = 256,
                 output_dim: int = 128, sequence_len: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_len = sequence_len

        flat_dim = input_dim * sequence_len

        self.trend_block      = NBEATSBlock(flat_dim, block_dim, output_dim, flat_dim, "trend")
        self.seasonality_block = NBEATSBlock(flat_dim, block_dim, output_dim, flat_dim, "seasonality")
        self.generic_block    = NBEATSBlock(flat_dim, block_dim, output_dim, flat_dim, "generic")

        self.output_proj = nn.Linear(output_dim * 3, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

        self.capacity_head   = nn.Linear(output_dim, 1)
        self.trajectory_head = nn.Linear(output_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.reshape(B, -1)

        bc1, fc1 = self.trend_block(x_flat)
        residual = x_flat - bc1
        bc2, fc2 = self.seasonality_block(residual)
        residual = residual - bc2
        _, fc3   = self.generic_block(residual)

        z = self.output_proj(torch.cat([fc1, fc2, fc3], dim=-1))
        return self.output_norm(z)

    def predict(self, x: torch.Tensor):
        z = self.forward(x)
        return self.capacity_head(z), self.trajectory_head(z)


def generate_synthetic_trajectories(meso_model: nn.Module, n_trajectories: int = 500,
                                     n_months: int = 6, device: str = "cpu") -> list:
    meso_model.eval()
    trajectories = []
    with torch.no_grad():
        for _ in range(n_trajectories):
            activity_level = np.random.uniform(0.3, 1.0)
            recovery_speed = np.random.uniform(0.5, 1.5)
            monthly_vecs = []
            weekly_state = torch.randn(1, 7, 512).to(device) * 0.1
            for month in range(n_months):
                weekly_state = weekly_state + activity_level * 0.05 * torch.randn_like(weekly_state)
                z_meso = meso_model(weekly_state)
                trend_stats = torch.randn(1, 128).to(device) * recovery_speed * 0.1
                monthly_vec = torch.cat([z_meso, trend_stats], dim=-1)
                monthly_vecs.append(monthly_vec)
            traj = torch.cat(monthly_vecs, dim=0)
            trajectories.append(traj.cpu())
    return trajectories
