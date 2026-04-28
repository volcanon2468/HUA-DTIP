import torch
import torch.nn as nn
import torch.nn.functional as F

class TwinEncoder(nn.Module):

    def __init__(self, input_dim: int=512, latent_dim: int=32, dropout: float=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return (mu, logvar)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

class TwinDecoder(nn.Module):

    def __init__(self, latent_dim: int=32, output_dim: int=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, output_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class PredictionHead(nn.Module):

    def __init__(self, latent_dim: int=32, activity_dim: int=64, output_dim: int=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + activity_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, output_dim))

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))

class BayesianVAE(nn.Module):

    def __init__(self, input_dim: int=512, latent_dim: int=32, decoder_out_dim: int=256, beta: float=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = TwinEncoder(input_dim, latent_dim)
        self.decoder = TwinDecoder(latent_dim, decoder_out_dim)
        self.activity_proj = nn.Linear(decoder_out_dim, 64)
        self.pred_head = PredictionHead(latent_dim, 64, 6)
        self.recon_proj = nn.Linear(decoder_out_dim, 48)

    def forward(self, z_temporal: torch.Tensor):
        mu, logvar = self.encoder(z_temporal)
        z = TwinEncoder.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        recon = self.recon_proj(decoded)
        activity_ctx = self.activity_proj(decoded)
        prediction = self.pred_head(z.detach(), activity_ctx.detach())
        return (z, mu, logvar, recon, prediction)

    def loss(self, z_temporal: torch.Tensor, hr_true: torch.Tensor, hrv_true: torch.Tensor):
        z, mu, logvar, recon, pred = self.forward(z_temporal)
        target_recon = z_temporal[:, :48]
        l_recon = F.mse_loss(recon, target_recon)
        l_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        l_pred = F.mse_loss(pred[:, 0], hr_true) + F.mse_loss(pred[:, 1:6], hrv_true)
        return (l_recon + self.beta * l_kl + 0.0001 * l_pred, {'recon': l_recon.item(), 'kl': l_kl.item(), 'pred': l_pred.item()})

    @torch.no_grad()
    def mc_sample(self, z_temporal: torch.Tensor, n_samples: int=100):
        samples = []
        for _ in range(n_samples):
            mu, logvar = self.encoder(z_temporal)
            z = TwinEncoder.reparameterize(mu, logvar)
            samples.append(z)
        samples = torch.stack(samples, dim=0)
        return (samples.mean(dim=0), samples.std(dim=0), samples)