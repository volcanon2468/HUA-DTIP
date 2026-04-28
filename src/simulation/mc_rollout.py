import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE

@dataclass
class InterventionPlan:
    intensity: float = 0.5
    duration_days: int = 7
    rest_extra_hours: float = 1.0
    nutrition_quality: float = 0.7
    sleep_consistency: float = 0.8

    def to_activity_tensor(self, device='cpu') -> torch.Tensor:
        return torch.tensor([self.intensity, self.duration_days / 28.0, self.rest_extra_hours / 8.0, self.nutrition_quality, self.sleep_consistency, min(1.0, self.intensity * self.duration_days / 14.0)], dtype=torch.float32, device=device).unsqueeze(0)

    def to_rest_tensor(self, device='cpu') -> torch.Tensor:
        return torch.tensor([self.rest_extra_hours / 8.0, self.sleep_consistency, 1.0 - self.intensity], dtype=torch.float32, device=device).unsqueeze(0)

@dataclass
class RolloutResult:
    z_mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z_std: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z_trajectories: np.ndarray = field(default_factory=lambda: np.zeros(1))
    hr_trajectory: np.ndarray = field(default_factory=lambda: np.zeros(1))
    hrv_trajectory: np.ndarray = field(default_factory=lambda: np.zeros(1))
    risk_scores: np.ndarray = field(default_factory=lambda: np.zeros(1))
    overtraining_prob: float = 0.0
    injury_prob: float = 0.0
    peaking_day: int = -1

class MCRolloutEngine:

    def __init__(self, vae: BayesianVAE, sde: LatentNeuralSDE, n_samples: int=200, device: str='cpu'):
        self.vae = vae
        self.sde = sde
        self.n_samples = n_samples
        self.device = torch.device(device)
        self.vae.to(self.device).eval()
        self.sde.to(self.device).eval()

    @torch.no_grad()
    def rollout(self, z0_mean: torch.Tensor, z0_std: torch.Tensor, plan: InterventionPlan, n_days: int=28) -> RolloutResult:
        z0_mean = z0_mean.to(self.device)
        z0_std = z0_std.to(self.device)
        activity = plan.to_activity_tensor(self.device).expand(self.n_samples, -1)
        rest = plan.to_rest_tensor(self.device).expand(self.n_samples, -1)
        z0_samples = z0_mean.unsqueeze(0) + z0_std.unsqueeze(0) * torch.randn(self.n_samples, z0_mean.shape[-1], device=self.device)
        ts = torch.linspace(0, float(n_days), n_days + 1, device=self.device)
        all_trajs = []
        for i in range(0, self.n_samples, 32):
            chunk = z0_samples[i:i + 32]
            a_chunk = activity[i:i + 32]
            r_chunk = rest[i:i + 32]
            zs = self.sde(chunk, a_chunk, r_chunk, ts)
            all_trajs.append(zs.cpu())
        all_trajs = torch.cat(all_trajs, dim=1)
        z_mean = all_trajs.mean(dim=1).numpy()
        z_std = all_trajs.std(dim=1).numpy()
        decoded_list = []
        for t in range(len(ts)):
            zt = torch.tensor(z_mean[t], dtype=torch.float32, device=self.device).unsqueeze(0)
            decoded = self.vae.decoder(zt)
            pred = self.vae.pred_head(zt, self.vae.activity_proj(decoded))
            decoded_list.append(pred.cpu().numpy())
        preds = np.stack(decoded_list, axis=0)
        hr_traj = preds[:, 0, 0]
        hrv_traj = preds[:, 0, 1:6]
        risk_scores = self._compute_risk(z_mean, z_std, hr_traj)
        overtraining_prob = float(np.mean(all_trajs[:, :, 0].numpy() < -2.0))
        injury_prob = float(np.mean(all_trajs[:, :, 1].numpy() > 2.0))
        peaking_day = int(np.argmax(hr_traj)) if len(hr_traj) > 0 else -1
        return RolloutResult(z_mean=z_mean, z_std=z_std, z_trajectories=all_trajs.numpy(), hr_trajectory=hr_traj, hrv_trajectory=hrv_traj, risk_scores=risk_scores, overtraining_prob=overtraining_prob, injury_prob=injury_prob, peaking_day=peaking_day)

    def _compute_risk(self, z_mean: np.ndarray, z_std: np.ndarray, hr_traj: np.ndarray) -> np.ndarray:
        n_steps = z_mean.shape[0]
        risk = np.zeros(n_steps)
        for t in range(n_steps):
            uncertainty = z_std[t].mean()
            hr_norm = min(1.0, abs(hr_traj[t] - 70.0) / 50.0) if t < len(hr_traj) else 0.0
            risk[t] = 0.5 * uncertainty + 0.5 * hr_norm
        return risk