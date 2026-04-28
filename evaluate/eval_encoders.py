import os
import csv
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.utils.metrics import activity_f1, mae
from train.train_encoders import WindowDataset

def loso_imu_eval(processed_dir, checkpoint_dir, n_subjects=10, n_classes=12, device=torch.device('cpu')):
    results = []
    all_ids = list(range(1, n_subjects + 1))
    for test_id in all_ids:
        encoder = SWCTNet()
        ckpt_path = os.path.join(checkpoint_dir, 'encoder_imu.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            ckpt_n_classes = ckpt.get('classifier.weight', torch.empty(0)).shape[0]
            if ckpt_n_classes > 0:
                encoder.build_classifier(ckpt_n_classes)
            else:
                encoder.build_classifier(n_classes)
            encoder.to(device)
            encoder.load_state_dict(ckpt)
        else:
            encoder.build_classifier(n_classes)
            encoder.to(device)
        encoder.eval()
        test_loader = DataLoader(WindowDataset(processed_dir, [test_id]), batch_size=64, shuffle=False)
        all_logits, all_labels = ([], [])
        with torch.no_grad():
            for batch in test_loader:
                imu = batch['imu'].to(device)
                labels = batch['label']
                valid = labels >= 0
                if not valid.any():
                    continue
                logits = encoder.classify(imu[valid])
                all_logits.append(logits.cpu())
                all_labels.append(labels[valid])
        f1 = activity_f1(torch.cat(all_logits), torch.cat(all_labels)) if all_logits else 0.0
        results.append({'subject_id': test_id, 'activity_f1': f1})
        print(f'  Subject {test_id:2d} — F1: {f1:.4f}')
    return results

def hr_mae_eval(processed_dir, checkpoint_dir, device=torch.device('cpu')):
    encoder = CardioEncoder().to(device)
    ckpt_path = os.path.join(checkpoint_dir, 'encoder_cardio.pt')
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder.eval()
    loader = DataLoader(WindowDataset(processed_dir), batch_size=64, shuffle=False)
    all_pred, all_true = ([], [])
    with torch.no_grad():
        for batch in loader:
            cardio = batch['cardio'].to(device)
            hr_true = batch['features'][:, 20]
            hr_pred = encoder.predict_hr(cardio).squeeze(-1).cpu()
            all_pred.append(hr_pred)
            all_true.append(hr_true)
    if not all_pred:
        return float('nan')
    mae_val = mae(torch.cat(all_pred), torch.cat(all_true))
    print(f'  HR MAE: {mae_val:.2f} bpm  (target: <5 bpm)')
    return mae_val

def main():
    data_cfg = OmegaConf.load('configs/data.yaml')
    train_cfg = OmegaConf.load('configs/training.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_dir = data_cfg.paths.processed
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)
    print('=== IMU Encoder: LOSO Activity F1 ===')
    loso_results = loso_imu_eval(processed_dir, checkpoint_dir, n_subjects=data_cfg.mhealth.n_subjects, n_classes=data_cfg.mhealth.n_activity_classes, device=device)
    f1_path = os.path.join(results_dir, 'encoder_activity_f1.csv')
    with open(f1_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'activity_f1'])
        writer.writeheader()
        writer.writerows(loso_results)
    avg_f1 = sum((r['activity_f1'] for r in loso_results)) / len(loso_results)
    print(f'  Mean LOSO F1: {avg_f1:.4f}  (target: >0.85)')
    print('\n=== Cardio Encoder: HR MAE ===')
    hr_mae_val = hr_mae_eval(processed_dir, checkpoint_dir, device=device)
    mae_path = os.path.join(results_dir, 'encoder_hr_mae.csv')
    with open(mae_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'target'])
        writer.writerow(['hr_mae_bpm', f'{hr_mae_val:.4f}', '<5'])
        writer.writerow(['imu_avg_f1', f'{avg_f1:.4f}', '>0.85'])
    print(f'\nResults saved to {results_dir}')
if __name__ == '__main__':
    main()