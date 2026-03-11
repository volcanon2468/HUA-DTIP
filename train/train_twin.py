import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.temporal.micro_scale import MicroScaleModel
from src.temporal.meso_scale import MesoScaleModel
from src.temporal.macro_scale import MacroScaleModel
from src.temporal.hierarchical_fusion import HierarchicalFusion
from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion
from src.utils.seed import set_seed
from src.utils.logger import init_run, log_metrics, log_model, finish_run


class ZTemporalDataset(Dataset):
    def __init__(self, processed_dir: str, z_dim: int = 512):
        self.paths = sorted(glob.glob(
            os.path.join(processed_dir, "subject_*", "windows", "window_*.pt")
        ))
        self.z_dim = z_dim

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        w = torch.load(self.paths[idx], map_location="cpu")
        features = w["features"]
        hrv      = w["hrv"]
        hr       = features[20:21]
        return features, hrv, hr


class DaySequenceDataset(Dataset):
    def __init__(self, processed_dir: str, seq_len: int = 1):
        self.sequences = []
        all_paths = {}
        for p in sorted(glob.glob(os.path.join(processed_dir, "subject_*", "daily_summaries", "day_*.pt"))):
            parts = p.replace("\\", "/").split("/")
            sid_str = [x for x in parts if x.startswith("subject_")]
            if not sid_str:
                continue
            sid = int(sid_str[0].replace("subject_", ""))
            all_paths.setdefault(sid, []).append(p)

        for sid, paths in all_paths.items():
            for i in range(len(paths) - seq_len):
                self.sequences.append(paths[i: i + seq_len + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths = self.sequences[idx]
        seq = torch.stack([torch.load(p, map_location="cpu") for p in paths])
        return seq[:-1], seq[-1]


def _load_encoders_temporal(cfg, device):
    enc_imu    = SWCTNet().to(device)
    enc_cardio = CardioEncoder().to(device)
    enc_feat   = FeatureEncoder().to(device)
    enc_fusion = CrossModalFusion().to(device)
    micro      = MicroScaleModel().to(device)
    meso       = MesoScaleModel().to(device)
    macro      = MacroScaleModel().to(device)
    hier       = HierarchicalFusion().to(device)

    name_map = {
        "encoder_imu": enc_imu, "encoder_cardio": enc_cardio,
        "encoder_feature": enc_feat, "encoder_fusion": enc_fusion,
        "temporal_micro": micro, "temporal_meso": meso,
        "temporal_macro": macro, "temporal_fusion": hier,
    }
    for name, model in name_map.items():
        p = os.path.join(cfg.checkpoints.dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device, weights_only=True), strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return enc_imu, enc_cardio, enc_feat, enc_fusion, micro, meso, macro, hier


def train_vae(vae: BayesianVAE, loader: DataLoader, cfg: DictConfig, device: torch.device,
              feat_mean=None, feat_std=None):
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.training.twin.lr,
                           weight_decay=cfg.training.twin.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.training.twin.epochs)
    best_loss = float("inf"); patience_count = 0

    for epoch in range(cfg.training.twin.epochs):
        total_loss = 0.0; n = 0
        for features, hrv, hr in loader:
            features = torch.nan_to_num(features.to(device), nan=0.0)
            if feat_mean is not None:
                features = (features - feat_mean.to(device)) / feat_std.to(device)
            hr       = torch.nan_to_num(hr.to(device).squeeze(-1), nan=0.0)
            hrv      = torch.nan_to_num(hrv.to(device), nan=0.0)

            z_input = torch.cat([features, torch.zeros(features.shape[0], 512 - 48, device=device)], dim=-1)
            loss, parts = vae.loss(z_input, hr, hrv)

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * features.shape[0]
            n += features.shape[0]

        avg = total_loss / max(n, 1)
        scheduler.step()
        log_metrics({"vae/loss": avg, "vae/recon": parts["recon"],
                     "vae/kl": parts["kl"], "vae/pred": parts["pred"]}, step=epoch)

        if avg < best_loss:
            best_loss = avg; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.training.twin.patience:
                break

        if epoch % 10 == 0:
            print(f"  [VAE] epoch {epoch:3d}  loss={avg:.4f}  recon={parts['recon']:.4f}  kl={parts['kl']:.4f}")


def train_sde(sde: LatentNeuralSDE, vae: BayesianVAE, day_loader: DataLoader,
              cfg: DictConfig, device: torch.device):
    sde.train(); vae.eval()
    opt = torch.optim.Adam(sde.parameters(), lr=cfg.training.twin.sde.lr)
    mse = nn.MSELoss()
    best_loss = float("inf"); patience_count = 0

    for epoch in range(cfg.training.twin.sde.epochs):
        total_loss = 0.0; n = 0
        for x_seq, y_next in day_loader:
            x_seq  = torch.nan_to_num(x_seq.to(device), nan=0.0)
            y_next = torch.nan_to_num(y_next.to(device), nan=0.0)

            with torch.no_grad():
                z_input = torch.cat([x_seq[:, -1, :48],
                                     torch.zeros(x_seq.shape[0], 512 - 48, device=device)], dim=-1)
                mu, logvar = vae.encoder(z_input)
                z0 = vae.encoder.reparameterize(mu, logvar)

            B = z0.shape[0]
            activity = torch.zeros(B, 6, device=device)
            rest     = torch.zeros(B, 3, device=device)
            ts       = torch.tensor([0.0, 1.0], device=device)

            zs = sde(z0, activity, rest, ts)
            z_pred = zs[-1]

            with torch.no_grad():
                y_input = torch.cat([y_next[:, :48],
                                     torch.zeros(y_next.shape[0], 512 - 48, device=device)], dim=-1)
                mu_true, _ = vae.encoder(y_input)

            loss = mse(z_pred, mu_true)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * B; n += B

        avg = total_loss / max(n, 1)
        log_metrics({"sde/loss": avg}, step=epoch)
        if avg < best_loss:
            best_loss = avg; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 15:
                break
        if epoch % 10 == 0:
            print(f"  [SDE] epoch {epoch:3d}  loss={avg:.4f}")


def joint_finetune(vae: BayesianVAE, sde: LatentNeuralSDE,
                   loader: DataLoader, day_loader: DataLoader,
                   cfg: DictConfig, device: torch.device,
                   feat_mean=None, feat_std=None):
    for p in list(vae.parameters()) + list(sde.parameters()):
        p.requires_grad = True

    opt = torch.optim.Adam(
        list(vae.parameters()) + list(sde.parameters()),
        lr=cfg.training.twin.joint_finetune_lr,
    )
    mse = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0.0; n = 0
        for (features, hrv, hr), (x_seq, y_next) in zip(loader, day_loader):
            features = torch.nan_to_num(features.to(device), nan=0.0)
            if feat_mean is not None:
                features = (features - feat_mean.to(device)) / feat_std.to(device)
            hr       = torch.nan_to_num(hr.squeeze(-1).to(device), nan=0.0)
            hrv      = torch.nan_to_num(hrv.to(device), nan=0.0)
            x_seq    = torch.nan_to_num(x_seq.to(device), nan=0.0)
            y_next   = torch.nan_to_num(y_next.to(device), nan=0.0)

            z_input = torch.cat([features, torch.zeros(features.shape[0], 512 - 48, device=device)], dim=-1)
            vae_loss, _ = vae.loss(z_input, hr, hrv)

            B = min(x_seq.shape[0], 8)
            z_inp = torch.cat([x_seq[:B, -1, :48],
                                torch.zeros(B, 512 - 48, device=device)], dim=-1)
            mu, logvar = vae.encoder(z_inp)
            z0 = vae.encoder.reparameterize(mu, logvar)
            ts = torch.tensor([0.0, 1.0], device=device)
            zs = sde(z0, torch.zeros(B, 6, device=device), torch.zeros(B, 3, device=device), ts)
            y_inp = torch.cat([y_next[:B, :48], torch.zeros(B, 512 - 48, device=device)], dim=-1)
            mu_true, _ = vae.encoder(y_inp)
            sde_loss = mse(zs[-1], mu_true.detach())

            loss = vae_loss + 0.5 * sde_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        avg = total_loss / max(n, 1)
        log_metrics({"joint/loss": avg}, step=epoch)
        if epoch % 5 == 0:
            print(f"  [Joint] epoch {epoch:2d}  loss={avg:.4f}")


@hydra.main(config_path="../configs", config_name="training", version_base=None)
def main(cfg: DictConfig):
    data_cfg = OmegaConf.load("configs/data.yaml")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    init_run(cfg, name="twin-training")

    processed_dir = data_cfg.paths.processed
    feat_ds  = ZTemporalDataset(processed_dir)
    feat_loader = DataLoader(feat_ds, batch_size=cfg.training.twin.batch_size,
                             shuffle=True, num_workers=2, drop_last=True)
    day_ds = DaySequenceDataset(processed_dir, seq_len=1)
    day_loader = DataLoader(day_ds, batch_size=min(16, max(len(day_ds), 1)), shuffle=True, num_workers=2, drop_last=False)

    # Precompute global feature normalization stats
    print("Computing feature normalization statistics...")
    all_feats = []
    for features, hrv, hr in feat_loader:
        all_feats.append(torch.nan_to_num(features, nan=0.0))
    all_feats_cat = torch.cat(all_feats, dim=0)
    feat_mean = all_feats_cat.mean(dim=0, keepdim=True)
    feat_std  = all_feats_cat.std(dim=0, keepdim=True).clamp(min=1e-6)
    torch.save({"mean": feat_mean, "std": feat_std},
               os.path.join('checkpoints', 'feature_norm_stats.pt'))
    print(f"  Feature stats saved. Mean range: [{feat_mean.min():.2f}, {feat_mean.max():.2f}]")

    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)

    print("=== Training Beta-VAE ===")
    train_vae(vae, feat_loader, cfg, device, feat_mean=feat_mean, feat_std=feat_std)
    log_model(vae, "twin_vae", cfg)

    print("=== Step 2: Train Latent Neural SDE ===")
    train_sde(sde, vae, day_loader, cfg, device)
    log_model(sde, "twin_sde", cfg)

    print("=== Step 3: Joint Fine-Tuning ===")
    joint_finetune(vae, sde, feat_loader, day_loader, cfg, device,
                   feat_mean=feat_mean, feat_std=feat_std)
    log_model(vae, "twin_vae", cfg)
    log_model(sde, "twin_sde", cfg)

    print("Twin training complete.")
    finish_run()


if __name__ == "__main__":
    main()
