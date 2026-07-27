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


class WindowDataset(Dataset):
    def __init__(self, processed_dir: str):
        self.paths = sorted(
            glob.glob(os.path.join(processed_dir, 'subject_*', 'windows', 'window_*.pt'))
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        w = torch.load(self.paths[idx], map_location='cpu', weights_only=False)
        return {
            'imu'     : w['imu'],
            'cardio'  : w['cardio'],
            'features': w['features'],
            'hrv'     : w['hrv'],
        }


class DaySequenceDataset(Dataset):
    def __init__(self, processed_dir: str, seq_len: int = 1):
        self.sequences = []
        all_paths: dict = {}
        for p in sorted(glob.glob(
            os.path.join(processed_dir, 'subject_*', 'daily_summaries', 'day_*.pt')
        )):
            parts = p.replace('\\\\', '/').split('/')
            sid_str = [x for x in parts if x.startswith('subject_')]
            if not sid_str:
                continue
            sid = int(sid_str[0].replace('subject_', ''))
            all_paths.setdefault(sid, []).append(p)
        for sid, paths in all_paths.items():
            for i in range(len(paths) - seq_len):
                self.sequences.append(paths[i:i + seq_len + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths = self.sequences[idx]
        seq = torch.stack([torch.load(p, map_location='cpu') for p in paths])
        return seq[:-1], seq[-1]


def _encode_batch(batch, enc_imu, enc_cardio, enc_feat, enc_fusion, hier, device):
    imu    = torch.nan_to_num(batch['imu'].to(device), nan=0.0)
    cardio = torch.nan_to_num(batch['cardio'].to(device), nan=0.0)
    feats  = torch.nan_to_num(batch['features'].to(device), nan=0.0)

    h_imu    = enc_imu(imu)
    h_cardio = enc_cardio(cardio)
    h_feat   = enc_feat(feats)
    h_fused  = enc_fusion(h_imu, h_cardio, h_feat)

    z_temporal = hier(h_fused)
    return z_temporal


def _load_encoders_temporal(cfg, device):
    """Load and freeze the pre-trained encoder stack."""
    enc_imu    = SWCTNet().to(device)
    enc_cardio = CardioEncoder().to(device)
    enc_feat   = FeatureEncoder().to(device)
    enc_fusion = CrossModalFusion().to(device)
    micro  = MicroScaleModel().to(device)
    meso   = MesoScaleModel().to(device)
    macro  = MacroScaleModel().to(device)
    hier   = HierarchicalFusion().to(device)

    name_map = {
        'encoder_imu'     : enc_imu,
        'encoder_cardio'  : enc_cardio,
        'encoder_feature' : enc_feat,
        'encoder_fusion'  : enc_fusion,
        'temporal_micro'  : micro,
        'temporal_meso'   : meso,
        'temporal_macro'  : macro,
        'temporal_fusion' : hier,
    }
    for name, model in name_map.items():
        p = os.path.join(cfg.checkpoints.dir, f'{name}.pt')
        if os.path.exists(p):
            model.load_state_dict(
                torch.load(p, map_location=device, weights_only=True), strict=False
            )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return enc_imu, enc_cardio, enc_feat, enc_fusion, micro, meso, macro, hier


def train_vae(
    vae: BayesianVAE,
    loader: DataLoader,
    enc_imu, enc_cardio, enc_feat, enc_fusion, hier,
    cfg: DictConfig,
    device: torch.device,
):
    vae.train()
    opt = torch.optim.Adam(
        vae.parameters(), lr=cfg.training.twin.lr,
        weight_decay=cfg.training.twin.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.training.twin.epochs)
    best_loss = float('inf')
    patience_count = 0

    for epoch in range(cfg.training.twin.epochs):
        total_loss = 0.0
        n = 0
        for batch in loader:
            hrv = torch.nan_to_num(batch['hrv'].to(device), nan=0.0)
            hr  = torch.nan_to_num(batch['features'][:, 20].to(device), nan=0.0)

            with torch.no_grad():
                z_temporal = _encode_batch(
                    batch, enc_imu, enc_cardio, enc_feat, enc_fusion, hier, device
                )

            loss, parts = vae.loss(z_temporal, hr, hrv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * z_temporal.shape[0]
            n += z_temporal.shape[0]

        avg = total_loss / max(n, 1)
        scheduler.step()
        log_metrics({
            'vae/loss' : avg,
            'vae/recon': parts['recon'],
            'vae/kl'   : parts['kl'],
            'vae/pred' : parts['pred'],
        }, step=epoch)

        if avg < best_loss:
            best_loss = avg
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.training.twin.patience:
                break
        if epoch % 10 == 0:
            print(
                f'  [VAE] epoch {epoch:3d}  loss={avg:.4f}  '                f'recon={parts["recon"]:.4f}  kl={parts["kl"]:.4f}'
            )


def train_sde(
    sde: LatentNeuralSDE,
    vae: BayesianVAE,
    day_loader: DataLoader,
    enc_feat,
    cfg: DictConfig,
    device: torch.device,
):
    sde.train()
    vae.eval()

    daily_proj = nn.Linear(48, 512).to(device)
    nn.init.xavier_uniform_(daily_proj.weight)

    opt = torch.optim.Adam(
        list(sde.parameters()) + list(daily_proj.parameters()),
        lr=cfg.training.twin.sde.lr,
    )
    mse = nn.MSELoss()
    best_loss = float('inf')
    patience_count = 0

    for epoch in range(cfg.training.twin.sde.epochs):
        total_loss = 0.0
        n = 0
        for x_seq, y_next in day_loader:
            x_seq  = torch.nan_to_num(x_seq.to(device), nan=0.0)
            y_next = torch.nan_to_num(y_next.to(device), nan=0.0)

            with torch.no_grad():
                x_last_projected = daily_proj(x_seq[:, -1, :])
                mu, logvar = vae.encoder(x_last_projected)
                z0 = vae.encoder.reparameterize(mu, logvar)

            B = z0.shape[0]
            activity = torch.zeros(B, 6, device=device)
            rest     = torch.zeros(B, 3, device=device)
            ts = torch.tensor([0.0, 1.0], device=device)
            zs = sde(z0, activity, rest, ts)
            z_pred = zs[-1]

            with torch.no_grad():
                y_projected = daily_proj(y_next)
                mu_true, _ = vae.encoder(y_projected)

            loss = mse(z_pred, mu_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * B
            n += B

        avg = total_loss / max(n, 1)
        log_metrics({'sde/loss': avg}, step=epoch)
        if avg < best_loss:
            best_loss = avg
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 15:
                break
        if epoch % 10 == 0:
            print(f'  [SDE] epoch {epoch:3d}  loss={avg:.4f}')

    return daily_proj


def joint_finetune(
    vae: BayesianVAE,
    sde: LatentNeuralSDE,
    loader: DataLoader,
    day_loader: DataLoader,
    enc_imu, enc_cardio, enc_feat, enc_fusion, hier,
    cfg: DictConfig,
    device: torch.device,
    daily_proj: nn.Module = None,
):
    for p in list(vae.parameters()) + list(sde.parameters()):
        p.requires_grad = True
    if daily_proj is not None:
        for p in daily_proj.parameters():
            p.requires_grad = True

    trainable = list(vae.parameters()) + list(sde.parameters())
    if daily_proj is not None:
        trainable += list(daily_proj.parameters())
    opt = torch.optim.Adam(trainable, lr=cfg.training.twin.joint_finetune_lr)
    mse = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0.0
        n = 0
        for batch_w, (x_seq, y_next) in zip(loader, day_loader):
            hrv    = torch.nan_to_num(batch_w['hrv'].to(device), nan=0.0)
            hr     = torch.nan_to_num(batch_w['features'][:, 20].to(device), nan=0.0)
            x_seq  = torch.nan_to_num(x_seq.to(device), nan=0.0)
            y_next = torch.nan_to_num(y_next.to(device), nan=0.0)

            with torch.no_grad():
                z_temporal = _encode_batch(
                    batch_w, enc_imu, enc_cardio, enc_feat, enc_fusion, hier, device
                )

            vae_loss, _ = vae.loss(z_temporal, hr, hrv)

            B = min(x_seq.shape[0], 8)
            if daily_proj is not None:
                x_proj = daily_proj(x_seq[:B, -1, :])
            else:
                x_proj = torch.zeros(B, 512, device=device)
            mu_0, lv_0 = vae.encoder(x_proj)
            z0 = vae.encoder.reparameterize(mu_0, lv_0)
            ts = torch.tensor([0.0, 1.0], device=device)
            zs = sde(z0, torch.zeros(B, 6, device=device), torch.zeros(B, 3, device=device), ts)

            if daily_proj is not None:
                y_proj = daily_proj(y_next[:B])
            else:
                y_proj = torch.zeros(B, 512, device=device)
            mu_true, _ = vae.encoder(y_proj)
            sde_loss = mse(zs[-1], mu_true.detach())

            loss = vae_loss + 0.5 * sde_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        log_metrics({'joint/loss': avg}, step=epoch)
        if epoch % 5 == 0:
            print(f'  [Joint] epoch {epoch:2d}  loss={avg:.4f}')


@hydra.main(config_path='../configs', config_name='training', version_base=None)
def main(cfg: DictConfig):
    data_cfg = OmegaConf.load('configs/data.yaml')
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    init_run(cfg, name='twin-training')
    processed_dir = data_cfg.paths.processed

    (enc_imu, enc_cardio, enc_feat, enc_fusion,
     micro, meso, macro, hier) = _load_encoders_temporal(cfg, device)

    feat_ds  = WindowDataset(processed_dir)
    feat_loader = DataLoader(
        feat_ds,
        batch_size=cfg.training.twin.batch_size,
        shuffle=True, num_workers=2, drop_last=True,
    )
    day_ds  = DaySequenceDataset(processed_dir, seq_len=1)
    day_loader = DataLoader(
        day_ds,
        batch_size=min(16, max(len(day_ds), 1)),
        shuffle=True, num_workers=2, drop_last=False,
    )

    vae = BayesianVAE(input_dim=512, latent_dim=10).to(device)
    sde = LatentNeuralSDE().to(device)

    print('=== Step 1: Training Beta-VAE (input_dim=512, latent_dim=10) ===')
    train_vae(vae, feat_loader, enc_imu, enc_cardio, enc_feat, enc_fusion, hier, cfg, device)
    log_model(vae, 'twin_vae', cfg)

    print('=== Step 2: Train Latent Neural SDE ===')
    daily_proj = train_sde(sde, vae, day_loader, enc_feat, cfg, device)
    log_model(sde, 'twin_sde', cfg)
    torch.save(daily_proj.state_dict(),
               os.path.join(cfg.checkpoints.dir, 'daily_proj.pt'))

    print('=== Step 3: Joint Fine-Tuning ===')
    joint_finetune(
        vae, sde, feat_loader, day_loader,
        enc_imu, enc_cardio, enc_feat, enc_fusion, hier,
        cfg, device, daily_proj=daily_proj,
    )
    log_model(vae, 'twin_vae', cfg)
    log_model(sde, 'twin_sde', cfg)

    print('Twin training complete.')
    finish_run()


if __name__ == '__main__':
    main()
