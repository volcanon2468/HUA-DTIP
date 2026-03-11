import torch
import torch.nn as nn
import numpy as np
from collections import deque


class NoveltyAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 48, latent_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class AutoencoderNoveltyDetector:
    def __init__(self, input_dim: int = 48, latent_dim: int = 32,
                 percentile_threshold: float = 95.0, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = NoveltyAutoencoder(input_dim, latent_dim).to(self.device)
        self.percentile = percentile_threshold
        self.threshold = float("inf")
        self._ref_errors = []
        self._drift_log = []
        self._is_fitted = False

    def fit(self, ref_data: np.ndarray, epochs: int = 50, lr: float = 1e-3):
        self.model.train()
        X = torch.tensor(ref_data, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse = nn.MSELoss()

        for epoch in range(epochs):
            recon, _ = self.model(X)
            loss = mse(recon, X)
            opt.zero_grad(); loss.backward(); opt.step()

        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X)
            errors = torch.mean((recon - X) ** 2, dim=-1).cpu().numpy()
        self._ref_errors = errors.tolist()
        self.threshold = float(np.percentile(errors, self.percentile))
        self._is_fitted = True

    def score(self, feature_vec: np.ndarray) -> float:
        if not self._is_fitted:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feature_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            recon, _ = self.model(x)
            error = float(torch.mean((recon - x) ** 2).item())
        return error

    def update(self, feature_vec: np.ndarray, timestamp: float = None) -> dict:
        error = self.score(feature_vec)
        is_novel = error > self.threshold if self._is_fitted else False
        result = {
            "drift_detected": is_novel,
            "reconstruction_error": error,
            "threshold": self.threshold,
            "novelty_ratio": error / (self.threshold + 1e-8),
            "timestamp": timestamp,
        }
        if result["drift_detected"]:
            self._drift_log.append(result)
        return result

    def get_drift_log(self) -> list:
        return self._drift_log

    def retrain(self, new_data: np.ndarray, epochs: int = 20, lr: float = 5e-4):
        self.model.train()
        X = torch.tensor(new_data, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse = nn.MSELoss()
        for _ in range(epochs):
            recon, _ = self.model(X)
            loss = mse(recon, X)
            opt.zero_grad(); loss.backward(); opt.step()

        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X)
            errors = torch.mean((recon - X) ** 2, dim=-1).cpu().numpy()
        self.threshold = float(np.percentile(errors, self.percentile))
