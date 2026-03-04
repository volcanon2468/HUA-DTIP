import torch
import torch.nn as nn
import torchsde


class ActivityEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden: int = 32, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RestEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, hidden: int = 16, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SDEFunc(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, latent_dim: int = 10, activity_dim: int = 64,
                 rest_dim: int = 32, drift_hidden: int = 128, diffusion_hidden: int = 32):
        super().__init__()
        input_dim = latent_dim + activity_dim + rest_dim + 1

        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, drift_hidden), nn.Tanh(),
            nn.Linear(drift_hidden, drift_hidden), nn.Tanh(),
            nn.Linear(drift_hidden, latent_dim),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + 1, diffusion_hidden), nn.Softplus(),
            nn.Linear(diffusion_hidden, latent_dim),
        )

        self.activity_dim = activity_dim
        self.rest_dim = rest_dim
        self.latent_dim = latent_dim

        self._a_t = None
        self._r_t = None

    def set_context(self, a_t: torch.Tensor, r_t: torch.Tensor):
        self._a_t = a_t
        self._r_t = r_t

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_vec = t.expand(z.shape[0], 1) if t.dim() == 0 else t.unsqueeze(-1)
        a = self._a_t if self._a_t is not None else torch.zeros(z.shape[0], self.activity_dim, device=z.device)
        r = self._r_t if self._r_t is not None else torch.zeros(z.shape[0], self.rest_dim, device=z.device)
        inp = torch.cat([z, a, r, t_vec], dim=-1)
        return self.drift_net(inp)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_vec = t.expand(z.shape[0], 1) if t.dim() == 0 else t.unsqueeze(-1)
        inp = torch.cat([z, t_vec], dim=-1)
        return self.diffusion_net(inp)


class LatentNeuralSDE(nn.Module):
    def __init__(self, latent_dim: int = 10, activity_input_dim: int = 6,
                 rest_input_dim: int = 3, drift_hidden: int = 128, diffusion_hidden: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        self.activity_enc = ActivityEncoder(activity_input_dim, 32, 64)
        self.rest_enc      = RestEncoder(rest_input_dim, 16, 32)
        self.sde_func      = SDEFunc(latent_dim, 64, 32, drift_hidden, diffusion_hidden)

    def forward(self, z0: torch.Tensor, activity: torch.Tensor, rest: torch.Tensor,
                ts: torch.Tensor) -> torch.Tensor:
        a_t = self.activity_enc(activity)
        r_t = self.rest_enc(rest)
        self.sde_func.set_context(a_t, r_t)

        zs = torchsde.sdeint(self.sde_func, z0, ts, method="euler", dt=1e-1)
        return zs

    def predict_trajectory(self, z0: torch.Tensor, activity: torch.Tensor, rest: torch.Tensor,
                           n_days: int = 7, n_samples: int = 50):
        ts = torch.linspace(0, float(n_days), n_days + 1, device=z0.device)
        trajectories = []
        for _ in range(n_samples):
            zs = self.forward(z0, activity, rest, ts)
            trajectories.append(zs)
        trajectories = torch.stack(trajectories, dim=0)
        return trajectories.mean(dim=0), trajectories.std(dim=0)
