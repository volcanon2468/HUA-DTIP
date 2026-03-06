import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE


class TwinGymEnv(gym.Env):
    def __init__(self, vae: BayesianVAE, sde: LatentNeuralSDE,
                 episode_len: int = 28, device: str = "cpu"):
        super().__init__()
        self.vae = vae
        self.sde = sde
        self.device = torch.device(device)
        self.episode_len = episode_len
        self.latent_dim = vae.latent_dim

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.latent_dim * 2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        self.t = 0
        self.z_mu = None
        self.z_std = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.z_mu  = torch.randn(1, self.latent_dim, device=self.device) * 0.5
        self.z_std = torch.ones(1, self.latent_dim, device=self.device) * 0.3
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        activity = torch.cat([action_t, (action_t[:, :1] * action_t[:, 1:2])], dim=-1)
        rest = torch.stack([action_t[:, 2], action_t[:, 4], 1.0 - action_t[:, 0]], dim=-1)

        ts = torch.tensor([0.0, 1.0], device=self.device)
        with torch.no_grad():
            z0 = self.z_mu + self.z_std * torch.randn_like(self.z_mu)
            zs = self.sde(z0, activity, rest, ts)
            z_next = zs[-1]

        self.z_mu  = z_next
        self.z_std = self.z_std * 0.99 + 0.01 * torch.abs(z_next - self.z_mu)
        self.t += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.t >= self.episode_len
        truncated = False
        info = {"z_mu": self.z_mu.cpu().numpy(), "z_std": self.z_std.cpu().numpy(), "day": self.t}

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return torch.cat([self.z_mu, self.z_std], dim=-1).squeeze(0).cpu().numpy()

    def _compute_reward(self, action: np.ndarray) -> float:
        z = self.z_mu.squeeze(0)
        capacity   = float(z[0])
        fatigue    = float(z[1])
        recovery   = float(z[2])
        cardio     = float(z[3])
        stability  = float(z[4])

        r_progress  = 0.3 * (capacity + cardio)
        r_safety    = -0.4 * max(0.0, fatigue - 1.5)
        r_recovery  = 0.2 * recovery
        r_stability = 0.1 * stability

        intensity = action[0]
        if fatigue > 2.0 and intensity > 0.7:
            r_safety -= 1.0
        if self.z_std.mean().item() > 1.5:
            r_safety -= 0.5

        return float(r_progress + r_safety + r_recovery + r_stability)


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim: int = 20, action_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),    nn.ReLU(inplace=True),
        )
        self.mu_head      = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = (action + 1.0) / 2.0
        return action, log_prob

    def deterministic(self, state: torch.Tensor):
        mu, _ = self.forward(state)
        return (torch.tanh(mu) + 1.0) / 2.0


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int = 20, action_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),                  nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),                  nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
