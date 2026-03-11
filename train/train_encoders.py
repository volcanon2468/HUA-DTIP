import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig

from src.encoders.imu_encoder import SWCTNet, ProjectionHead
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion
from src.utils.seed import set_seed
from src.utils.logger import init_run, log_metrics, log_model, finish_run
from src.utils.metrics import activity_f1


class WindowDataset(Dataset):
    def __init__(self, processed_dir: str, subject_ids: list = None):
        self.paths = []
        pattern = os.path.join(processed_dir, "subject_*", "windows", "window_*.pt")
        for p in sorted(glob.glob(pattern)):
            parts = p.replace("\\", "/").split("/")
            sid_str = [x for x in parts if x.startswith("subject_")]
            if not sid_str:
                continue
            sid = int(sid_str[0].replace("subject_", ""))
            if subject_ids is not None and sid not in subject_ids:
                continue
            self.paths.append(p)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        cardio = data["cardio"]
        if cardio.shape[-1] == 1:
            data["cardio"] = F.pad(cardio, (0, 1))
        return data


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=-1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    labels = torch.cat([torch.arange(B, device=z.device) + B,
                        torch.arange(B, device=z.device)])
    return F.cross_entropy(sim, labels)


def imu_augment(window: torch.Tensor) -> torch.Tensor:
    aug = window + 0.01 * torch.randn_like(window)
    T = window.shape[0]
    start = random.randint(0, T // 4)
    end   = random.randint(3 * T // 4, T)
    crop  = aug[start:end]
    aug   = F.interpolate(crop.T.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0).T
    return aug


def pretrain_imu(encoder, proj_head, loader, cfg, device):
    encoder.train(); proj_head.train()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=cfg.training.encoders.lr, weight_decay=cfg.training.encoders.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.training.encoders.epochs)
    best_loss = float("inf")
    patience_count = 0

    for epoch in range(cfg.training.encoders.epochs):
        total_loss = 0.0; n = 0
        for batch in loader:
            imu = batch["imu"].to(device)
            aug1 = torch.stack([imu_augment(x) for x in imu])
            aug2 = torch.stack([imu_augment(x) for x in imu])
            z1 = proj_head(encoder(aug1))
            z2 = proj_head(encoder(aug2))
            loss = nt_xent_loss(z1, z2, cfg.training.encoders.contrastive.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(imu); n += len(imu)

        avg_loss = total_loss / max(n, 1)
        scheduler.step()
        log_metrics({"imu_pretrain/loss": avg_loss}, step=epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.training.encoders.patience:
                break
        if epoch % 10 == 0:
            print(f"  [IMU pretrain] epoch {epoch:3d}  loss={avg_loss:.4f}")

    return best_loss


def finetune_imu(encoder, loader, cfg, device, n_classes):
    all_labels_raw = []
    for batch in loader:
        labs = batch["label"]
        all_labels_raw.extend(labs[labs >= 0].tolist())
    unique_labels = sorted(set(all_labels_raw))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    actual_n_classes = len(unique_labels)
    print(f"  Found {actual_n_classes} unique classes: {unique_labels[:20]}")

    encoder.build_classifier(actual_n_classes)
    encoder.to(device).train()
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.training.encoders.lr * 0.1)
    ce  = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for epoch in range(30):
        all_logits, all_mapped = [], []
        for batch in loader:
            imu = batch["imu"].to(device)
            labels = batch["label"]
            valid = labels >= 0
            if not valid.any():
                continue
            mapped = torch.tensor([label_map.get(int(l), 0) for l in labels[valid]], device=device)
            logits = encoder.classify(imu[valid])
            loss = ce(logits, mapped)
            opt.zero_grad(); loss.backward(); opt.step()
            all_logits.append(logits.detach().cpu())
            all_mapped.append(mapped.cpu())

        if all_logits:
            f1 = activity_f1(torch.cat(all_logits), torch.cat(all_mapped))
            log_metrics({"imu_finetune/f1": f1}, step=epoch)
            best_f1 = max(best_f1, f1)
            if epoch % 10 == 0:
                print(f"  [IMU finetune] epoch {epoch:2d}  F1={f1:.4f}")

    return best_f1


def train_cardio(encoder, loader, cfg, device):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.training.encoders.lr)
    mse_loss = nn.MSELoss()
    best_mse = float("inf"); patience_count = 0

    for epoch in range(cfg.training.encoders.epochs):
        total_loss = 0.0; n = 0
        for batch in loader:
            cardio = batch["cardio"].to(device)
            hr_target = batch["features"][:, 20].to(device).unsqueeze(-1)
            hr_pred = encoder.predict_hr(cardio)
            loss = mse_loss(hr_pred, hr_target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(cardio); n += len(cardio)

        avg = total_loss / max(n, 1)
        log_metrics({"cardio_train/mse": avg}, step=epoch)
        if avg < best_mse:
            best_mse = avg; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.training.encoders.patience:
                break
        if epoch % 10 == 0:
            print(f"  [Cardio] epoch {epoch:3d}  MSE={avg:.4f}")

    return best_mse


def train_fusion(imu_enc, cardio_enc, feat_enc, fusion, loader, cfg, device):
    for p in list(imu_enc.parameters()) + list(cardio_enc.parameters()):
        p.requires_grad = False
    opt = torch.optim.Adam(list(feat_enc.parameters()) + list(fusion.parameters()),
                           lr=cfg.training.encoders.lr)
    mse_loss = nn.MSELoss()

    for epoch in range(50):
        total_loss = 0.0; n = 0
        for batch in loader:
            imu    = torch.nan_to_num(batch["imu"].to(device), nan=0.0)
            cardio = torch.nan_to_num(batch["cardio"].to(device), nan=0.0)
            feats  = torch.nan_to_num(batch["features"].to(device), nan=0.0)
            with torch.no_grad():
                h_imu    = imu_enc(imu)
                h_cardio = cardio_enc(cardio)
            h_feat  = feat_enc(feats)
            h_fused = fusion(h_imu, h_cardio, h_feat)
            loss = mse_loss(h_fused, h_cardio.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(imu); n += len(imu)

        avg = total_loss / max(n, 1)
        log_metrics({"fusion_train/mse": avg}, step=epoch)
        if epoch % 10 == 0:
            print(f"  [Fusion] epoch {epoch:2d}  MSE={avg:.4f}")

    for p in list(imu_enc.parameters()) + list(cardio_enc.parameters()):
        p.requires_grad = True


@hydra.main(config_path="../configs", config_name="training", version_base=None)
def main(cfg: DictConfig):
    from omegaconf import OmegaConf
    data_cfg  = OmegaConf.load("configs/data.yaml")

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    init_run(cfg, name="encoder-pretraining")

    processed_dir = data_cfg.paths.processed
    dataset = WindowDataset(processed_dir)
    loader  = DataLoader(dataset, batch_size=cfg.encoders.batch_size, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    imu_enc    = SWCTNet().to(device)
    proj_head  = ProjectionHead().to(device)
    cardio_enc = CardioEncoder().to(device)
    feat_enc   = FeatureEncoder().to(device)
    fusion_mod = CrossModalFusion().to(device)

    print("=== Phase 1: IMU Pre-training (SimCLR) ===")
    pretrain_imu(imu_enc, proj_head, loader, cfg, device)
    log_model(imu_enc, "encoder_imu_pretrained", cfg)

    print("=== Phase 2: IMU Fine-tuning ===")
    n_classes = data_cfg.mhealth.n_activity_classes
    finetune_imu(imu_enc, loader, cfg, device, n_classes)
    log_model(imu_enc, "encoder_imu", cfg)

    print("=== Phase 3: Cardio Encoder Training ===")
    train_cardio(cardio_enc, loader, cfg, device)
    log_model(cardio_enc, "encoder_cardio", cfg)

    print("=== Phase 4: Feature Encoder + Fusion Training ===")
    train_fusion(imu_enc, cardio_enc, feat_enc, fusion_mod, loader, cfg, device)
    log_model(feat_enc, "encoder_feature", cfg)
    log_model(fusion_mod, "encoder_fusion", cfg)

    print("Training complete.")
    finish_run()


if __name__ == "__main__":
    main()
