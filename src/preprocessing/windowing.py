import json
import os
import numpy as np
import torch
from typing import Optional
from src.preprocessing.signal_cleaning import handle_missing

def create_windows(signal: np.ndarray, window_size: int=1000, stride: int=500):
    windows = []
    T = len(signal)
    start = 0
    while start + window_size <= T:
        windows.append(signal[start:start + window_size].copy())
        start += stride
    return windows

def per_subject_zscore(windows: list):
    stacked = np.concatenate(windows, axis=0)
    mu = stacked.mean(axis=0, keepdims=True)
    sigma = stacked.std(axis=0, keepdims=True) + 1e-08
    normalized = [(w - mu) / sigma for w in windows]
    return (normalized, (mu.squeeze(), sigma.squeeze()))

def assess_quality(window: np.ndarray) -> float:
    nan_frac = np.isnan(window).mean()
    if nan_frac > 0.3:
        return 0.0
    completeness = 1.0 - nan_frac
    w = np.nan_to_num(window)
    q1, q99 = (np.quantile(w, 0.01), np.quantile(w, 0.99))
    sat_frac = ((w <= q1) | (w >= q99)).mean()
    sat_score = max(0.0, 1.0 - sat_frac * 5)
    var_score = min(1.0, float(np.std(w)) / 0.01)
    return float(completeness * 0.4 + sat_score * 0.3 + var_score * 0.3)

def process_subject(imu: np.ndarray, cardio: np.ndarray, features_fn, hrv_fn, subject_id: int, out_dir: str, window_size: int=1000, stride: int=500, label_seq: Optional[np.ndarray]=None, timestamps: Optional[np.ndarray]=None):
    windows_dir = os.path.join(out_dir, f'subject_{subject_id}', 'windows')
    os.makedirs(windows_dir, exist_ok=True)
    imu_wins = create_windows(imu, window_size, stride)
    cardio_wins = create_windows(cardio, window_size, stride)
    imu_wins_norm, (imu_mu, imu_sigma) = per_subject_zscore(imu_wins)
    card_wins_norm, _ = per_subject_zscore(cardio_wins)
    saved = 0
    for t, (imu_w, card_w) in enumerate(zip(imu_wins_norm, card_wins_norm)):
        quality = assess_quality(imu_w)
        if np.isnan(card_w).any():
            card_w, discard = handle_missing(card_w)
            if discard:
                continue
        feat = features_fn(imu_w, card_w)
        hrv_vec = hrv_fn(card_w)
        label = int(label_seq[t * stride] if label_seq is not None else -1)
        ts = float(timestamps[t * stride]) if timestamps is not None else float(t)
        window_data = {'imu': torch.tensor(imu_w, dtype=torch.float32), 'cardio': torch.tensor(card_w, dtype=torch.float32), 'features': torch.tensor(feat, dtype=torch.float32), 'hrv': torch.tensor(hrv_vec, dtype=torch.float32), 'label': label, 'timestamp': ts, 'quality': quality}
        torch.save(window_data, os.path.join(windows_dir, f'window_{t:05d}.pt'))
        saved += 1
    return {'subject_id': subject_id, 'windows_saved': saved, 'imu_mu': imu_mu, 'imu_sigma': imu_sigma}