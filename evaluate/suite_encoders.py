import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion
from src.utils.metrics import activity_f1, mae

def run_encoder_eval(processed_dir: str, checkpoint_dir: str, data_cfg, device) -> dict:
    from evaluate.eval_encoders import loso_imu_eval, hr_mae_eval
    print('=== Encoder Suite ===')
    loso = loso_imu_eval(processed_dir, checkpoint_dir, n_subjects=data_cfg.mhealth.n_subjects, n_classes=data_cfg.mhealth.n_activity_classes, device=device)
    avg_f1 = np.mean([r['activity_f1'] for r in loso]) if loso else 0.0
    hr_mae_val = hr_mae_eval(processed_dir, checkpoint_dir, device=device)
    fusion_dim = 128
    imu_enc = SWCTNet().to(device)
    cardio_enc = CardioEncoder().to(device)
    feat_enc = FeatureEncoder().to(device)
    fusion_mod = CrossModalFusion().to(device)
    total_params = sum((sum((p.numel() for p in m.parameters())) for m in [imu_enc, cardio_enc, feat_enc, fusion_mod]))
    return {'imu_loso_f1': float(avg_f1), 'imu_loso_target': 0.85, 'hr_mae_bpm': float(hr_mae_val), 'hr_mae_target': 5.0, 'fusion_output_dim': fusion_dim, 'total_encoder_params': total_params, 'per_subject_f1': loso}
if __name__ == '__main__':
    data_cfg = OmegaConf.load('configs/data.yaml')
    train_cfg = OmegaConf.load('configs/training.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_encoder_eval(data_cfg.paths.processed, train_cfg.checkpoints.dir, data_cfg, device)
    print(json.dumps({k: v for k, v in results.items() if k != 'per_subject_f1'}, indent=2))