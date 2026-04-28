import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from src.temporal.micro_scale import MicroScaleModel
from src.temporal.meso_scale import MesoScaleModel
from src.temporal.macro_scale import MacroScaleModel, generate_synthetic_trajectories
from src.temporal.hierarchical_fusion import HierarchicalFusion
from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion
from src.utils.seed import set_seed
from src.utils.logger import init_run, log_metrics, log_model, load_checkpoint, finish_run

class HourlyBufferDataset(Dataset):

    def __init__(self, processed_dir: str, buffer_len: int=180, subject_ids: list=None):
        self.buffer_len = buffer_len
        self.sequences = []
        pattern = os.path.join(processed_dir, 'subject_*', 'windows', 'window_*.pt')
        all_paths = {}
        for p in sorted(glob.glob(pattern)):
            parts = p.replace('\\', '/').split('/')
            sid_str = [x for x in parts if x.startswith('subject_')]
            if not sid_str:
                continue
            sid = int(sid_str[0].replace('subject_', ''))
            if subject_ids is not None and sid not in subject_ids:
                continue
            all_paths.setdefault(sid, []).append(p)
        for sid, paths in all_paths.items():
            for i in range(0, len(paths) - buffer_len, buffer_len // 2):
                self.sequences.append(paths[i:i + buffer_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        windows = [torch.load(p, map_location='cpu', weights_only=False) for p in self.sequences[idx]]
        for w in windows:
            if 'cardio' in w and w['cardio'].shape[-1] == 1:
                w['cardio'] = torch.nn.functional.pad(w['cardio'], (0, 1))
        imu_seq = torch.stack([w['imu'] for w in windows])
        cardio_seq = torch.stack([w['cardio'] for w in windows])
        feat_seq = torch.stack([w['features'] for w in windows])
        hr_seq = torch.stack([w['features'][20:21] for w in windows]).squeeze(-1)
        hrv_seq = torch.stack([w['hrv'] for w in windows])
        return (imu_seq, cardio_seq, feat_seq, hr_seq, hrv_seq)

class DailySequenceDataset(Dataset):

    def __init__(self, processed_dir: str, seq_len: int=7, subject_ids: list=None):
        self.sequences = []
        pattern = os.path.join(processed_dir, 'subject_*', 'daily_summaries', 'day_*.pt')
        all_paths = {}
        for p in sorted(glob.glob(pattern)):
            parts = p.replace('\\', '/').split('/')
            sid_str = [x for x in parts if x.startswith('subject_')]
            if not sid_str:
                continue
            sid = int(sid_str[0].replace('subject_', ''))
            if subject_ids is not None and sid not in subject_ids:
                continue
            all_paths.setdefault(sid, []).append(p)
        for sid, paths in all_paths.items():
            for i in range(len(paths) - seq_len):
                self.sequences.append((paths[i:i + seq_len], paths[i + seq_len]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_paths, target_path = self.sequences[idx]
        x = torch.stack([torch.load(p, map_location='cpu') for p in input_paths])
        y = torch.load(target_path, map_location='cpu')
        return (x, y)

def train_micro(micro_model, imu_enc, cardio_enc, feat_enc, fusion_mod, loader, cfg, device):
    micro_model.train()
    for enc in [imu_enc, cardio_enc, feat_enc, fusion_mod]:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam(micro_model.parameters(), lr=cfg.training.temporal.lr, weight_decay=cfg.training.temporal.weight_decay)
    mse = nn.MSELoss()
    patience_count = 0
    best_loss = float('inf')
    for epoch in range(cfg.training.temporal.epochs):
        total_loss = 0.0
        n = 0
        for imu_seq, cardio_seq, feat_seq, hr_seq, hrv_seq in loader:
            B, T, *_ = imu_seq.shape
            imu_seq = torch.nan_to_num(imu_seq.to(device), nan=0.0)
            cardio_seq = torch.nan_to_num(cardio_seq.to(device), nan=0.0)
            feat_seq = torch.nan_to_num(feat_seq.to(device), nan=0.0)
            hr_seq = torch.nan_to_num(hr_seq.to(device), nan=0.0)
            hrv_seq = torch.nan_to_num(hrv_seq.to(device), nan=0.0)
            with torch.no_grad():
                imu_flat = imu_seq.view(B * T, *imu_seq.shape[2:])
                card_flat = cardio_seq.view(B * T, *cardio_seq.shape[2:])
                feat_flat = feat_seq.view(B * T, *feat_seq.shape[2:])
                h_imu = imu_enc(imu_flat).view(B, T, -1)
                h_cardio = cardio_enc(card_flat).view(B, T, -1)
                h_feat = feat_enc(feat_flat).view(B, T, -1)
                h_fused = torch.stack([fusion_mod(h_imu[:, t], h_cardio[:, t], h_feat[:, t]) for t in range(T)], dim=1)
            z_micro = micro_model(h_fused)
            hr_pred, hrv_pred = micro_model.predict(h_fused)
            loss = mse(hr_pred.squeeze(-1), hr_seq[:, -5:].mean(dim=1)) + mse(hrv_pred, hrv_seq[:, -5:, :].mean(dim=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * B
            n += B
        avg = total_loss / max(n, 1)
        log_metrics({'micro/loss': avg}, step=epoch)
        if avg < best_loss:
            best_loss = avg
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.training.temporal.patience:
                break
        if epoch % 10 == 0:
            print(f'  [Micro] epoch {epoch:3d}  loss={avg:.4f}')

def train_meso(meso_model, loader, cfg, device):
    meso_model.train()
    opt = torch.optim.Adam(meso_model.parameters(), lr=cfg.training.temporal.lr)
    mse = nn.MSELoss()
    best_loss = float('inf')
    patience_count = 0
    for epoch in range(cfg.training.temporal.epochs):
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = torch.nan_to_num(x.to(device), nan=0.0)
            y = torch.nan_to_num(y.to(device), nan=0.0)
            next_day_pred, _ = meso_model.predict(x)
            loss = mse(next_day_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
            n += x.shape[0]
        avg = total_loss / max(n, 1)
        log_metrics({'meso/loss': avg}, step=epoch)
        if avg < best_loss:
            best_loss = avg
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.training.temporal.patience:
                break
        if epoch % 10 == 0:
            print(f'  [Meso] epoch {epoch:3d}  loss={avg:.4f}')

def train_macro(macro_model, meso_model, processed_dir, cfg, device):
    macro_model.train()
    opt = torch.optim.Adam(macro_model.parameters(), lr=cfg.training.temporal.lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    synthetic = generate_synthetic_trajectories(meso_model, n_trajectories=500, n_months=7, device=str(device))
    from torch.utils.data import TensorDataset
    traj_tensor = torch.stack(synthetic)
    traj_in = traj_tensor[:, :-1]
    traj_out = traj_tensor[:, -1, :1]
    ds = TensorDataset(traj_in, traj_out)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for epoch in range(60):
        total_loss = 0.0
        n = 0
        for x, y_cap in loader:
            x = torch.nan_to_num(x.to(device), nan=0.0)
            y_cap = torch.nan_to_num(y_cap.to(device), nan=0.0)
            cap_pred, _ = macro_model.predict(x)
            loss = mse(cap_pred, y_cap)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
            n += x.shape[0]
        avg = total_loss / max(n, 1)
        log_metrics({'macro/loss': avg}, step=epoch)
        if epoch % 10 == 0:
            print(f'  [Macro] epoch {epoch:3d}  loss={avg:.4f}')

@hydra.main(config_path='../configs', config_name='training', version_base=None)
def main(cfg: DictConfig):
    data_cfg = OmegaConf.load('configs/data.yaml')
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    init_run(cfg, name='temporal-training')
    processed_dir = data_cfg.paths.processed
    imu_enc = SWCTNet().to(device)
    cardio_enc = CardioEncoder().to(device)
    feat_enc = FeatureEncoder().to(device)
    fusion_mod = CrossModalFusion().to(device)
    for name, model in [('encoder_imu', imu_enc), ('encoder_cardio', cardio_enc), ('encoder_feature', feat_enc), ('encoder_fusion', fusion_mod)]:
        ckpt_path = os.path.join(cfg.checkpoints.dir, f'{name}.pt')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
    micro_model = MicroScaleModel().to(device)
    meso_model = MesoScaleModel().to(device)
    macro_model = MacroScaleModel().to(device)
    fusion_t = HierarchicalFusion().to(device)
    print('=== Training Micro-Scale ===')
    hourly_ds = HourlyBufferDataset(processed_dir)
    hourly_loader = DataLoader(hourly_ds, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    train_micro(micro_model, imu_enc, cardio_enc, feat_enc, fusion_mod, hourly_loader, cfg, device)
    log_model(micro_model, 'temporal_micro', cfg)
    print('=== Training Meso-Scale ===')
    daily_ds = DailySequenceDataset(processed_dir, seq_len=1)
    daily_loader = DataLoader(daily_ds, batch_size=min(16, max(len(daily_ds), 1)), shuffle=True, num_workers=2, drop_last=False)
    train_meso(meso_model, daily_loader, cfg, device)
    log_model(meso_model, 'temporal_meso', cfg)
    print('=== Training Macro-Scale ===')
    train_macro(macro_model, meso_model, processed_dir, cfg, device)
    log_model(macro_model, 'temporal_macro', cfg)
    print('=== Training Hierarchical Fusion ===')
    z_micro = torch.randn(16, 256).to(device)
    z_meso = torch.randn(16, 512).to(device)
    z_macro = torch.randn(16, 128).to(device)
    opt_f = torch.optim.Adam(fusion_t.parameters(), lr=cfg.training.temporal.lr)
    for _ in range(20):
        z_t = fusion_t(z_micro, z_meso, z_macro)
        loss = (z_t ** 2).mean()
        opt_f.zero_grad()
        loss.backward()
        opt_f.step()
    log_model(fusion_t, 'temporal_fusion', cfg)
    print('Temporal training complete.')
    finish_run()
if __name__ == '__main__':
    main()