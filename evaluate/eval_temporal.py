import os
import csv
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.temporal.micro_scale import MicroScaleModel
from src.temporal.meso_scale import MesoScaleModel
from src.temporal.macro_scale import MacroScaleModel
from src.utils.metrics import mae
from train.train_temporal import HourlyBufferDataset, DailySequenceDataset
from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion


def eval_micro(micro_model, imu_enc, cardio_enc, feat_enc, fusion_mod, processed_dir, device):
    micro_model.eval()
    for m in [imu_enc, cardio_enc, feat_enc, fusion_mod]:
        m.eval()

    loader = DataLoader(HourlyBufferDataset(processed_dir), batch_size=8, shuffle=False)
    all_pred, all_true = [], []

    with torch.no_grad():
        for imu_seq, cardio_seq, feat_seq, hr_seq, hrv_seq in loader:
            B, T, *_ = imu_seq.shape
            imu_seq = imu_seq.to(device); cardio_seq = cardio_seq.to(device)
            feat_seq = feat_seq.to(device)
            imu_flat  = imu_seq.view(B * T, *imu_seq.shape[2:])
            card_flat = cardio_seq.view(B * T, *cardio_seq.shape[2:])
            feat_flat = feat_seq.view(B * T, *feat_seq.shape[2:])
            h_imu     = imu_enc(imu_flat).view(B, T, -1)
            h_cardio  = cardio_enc(card_flat).view(B, T, -1)
            h_feat    = feat_enc(feat_flat).view(B, T, -1)
            h_fused   = torch.stack([
                fusion_mod(h_imu[:, t], h_cardio[:, t], h_feat[:, t]) for t in range(T)
            ], dim=1)
            hr_pred, _ = micro_model.predict(h_fused)
            all_pred.append(hr_pred.squeeze(-1).cpu())
            all_true.append(hr_seq[:, -5:].mean(dim=1))

    mae_val = mae(torch.cat(all_pred), torch.cat(all_true)) if all_pred else float("nan")
    print(f"  Micro next-window HR MAE: {mae_val:.4f}")
    return mae_val


def eval_meso(meso_model, processed_dir, device):
    meso_model.eval()
    loader = DataLoader(DailySequenceDataset(processed_dir), batch_size=16, shuffle=False)
    all_pred, all_true = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred, _ = meso_model.predict(x)
            all_pred.append(pred[:, 20].cpu())
            all_true.append(y[:, 20])

    mae_val = mae(torch.cat(all_pred), torch.cat(all_true)) if all_pred else float("nan")
    print(f"  Meso next-day HR MAE: {mae_val:.4f}")
    return mae_val


def main():
    data_cfg  = OmegaConf.load("configs/data.yaml")
    train_cfg = OmegaConf.load("configs/training.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir  = data_cfg.paths.processed
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir    = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)

    imu_enc    = SWCTNet().to(device)
    cardio_enc = CardioEncoder().to(device)
    feat_enc   = FeatureEncoder().to(device)
    fusion_mod = CrossModalFusion().to(device)
    micro_model = MicroScaleModel().to(device)
    meso_model  = MesoScaleModel().to(device)

    for name, model in [("encoder_imu", imu_enc), ("encoder_cardio", cardio_enc),
                        ("encoder_feature", feat_enc), ("encoder_fusion", fusion_mod),
                        ("temporal_micro", micro_model), ("temporal_meso", meso_model)]:
        p = os.path.join(checkpoint_dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))

    micro_mae = eval_micro(micro_model, imu_enc, cardio_enc, feat_enc, fusion_mod, processed_dir, device)
    meso_mae  = eval_meso(meso_model, processed_dir, device)

    out_path = os.path.join(results_dir, "temporal_prediction_errors.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "target"])
        writer.writerow(["micro_next_window_hr_mae", f"{micro_mae:.4f}", "<5"])
        writer.writerow(["meso_next_day_hr_mae", f"{meso_mae:.4f}", "<8"])

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
