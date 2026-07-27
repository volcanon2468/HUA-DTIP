import json
import os
import glob
import numpy as np
import torch

MICRO_SURROGATE_DIM = 256
HRV_STATS_DIM = 128
CONTEXT_DIM = 64
CIRCADIAN_DIM = 64
DAILY_VECTOR_DIM = MICRO_SURROGATE_DIM + HRV_STATS_DIM + CONTEXT_DIM + CIRCADIAN_DIM

FEAT_DIM = 48
HRV_DIM = 5
CARDIO_FEAT_START = 20
CARDIO_FEAT_END = 40
SMA_IDX = 3
STEP_COUNT_IDX = 1


def build_daily_summary(windows_dir: str) -> np.ndarray:
    paths = sorted(glob.glob(os.path.join(windows_dir, 'window_*.pt')))
    if not paths:
        return np.zeros(DAILY_VECTOR_DIM, dtype=np.float32)
    features_list, hrv_list, qualities, labels, timestamps = ([], [], [], [], [])
    for p in paths:
        w = torch.load(p, map_location='cpu', weights_only=False)
        features_list.append(w['features'].numpy())
        hrv_list.append(w['hrv'].numpy())
        qualities.append(float(w['quality']))
        labels.append(int(w['label']))
        timestamps.append(float(w['timestamp']))
    feats = np.stack(features_list)
    hrvs = np.stack(hrv_list)
    q_weights = np.array(qualities) + 1e-6
    q_weights /= q_weights.sum()
    feat_mean = (feats * q_weights[:, None]).sum(axis=0)
    feat_std = feats.std(axis=0)

    micro_surrogate = np.zeros(MICRO_SURROGATE_DIM, dtype=np.float32)
    micro_surrogate[:FEAT_DIM] = feat_mean
    micro_surrogate[FEAT_DIM:FEAT_DIM * 2] = feat_std

    hrv_mean = hrvs.mean(axis=0)
    hrv_std = hrvs.std(axis=0)
    cardio_mean = feat_mean[CARDIO_FEAT_START:CARDIO_FEAT_END]
    cardio_std = feat_std[CARDIO_FEAT_START:CARDIO_FEAT_END]
    daily_stats = np.zeros(HRV_STATS_DIM, dtype=np.float32)
    daily_stats[:HRV_DIM] = hrv_mean
    daily_stats[HRV_DIM:HRV_DIM * 2] = hrv_std
    daily_stats[HRV_DIM * 2:HRV_DIM * 2 + (CARDIO_FEAT_END - CARDIO_FEAT_START)] = cardio_mean
    daily_stats[HRV_DIM * 2 + (CARDIO_FEAT_END - CARDIO_FEAT_START):HRV_DIM * 2 + 2 * (CARDIO_FEAT_END - CARDIO_FEAT_START)] = cardio_std

    context = np.zeros(CONTEXT_DIM, dtype=np.float32)
    context[0] = feat_mean[STEP_COUNT_IDX]
    context[1] = float(np.array(qualities).mean())
    context[2] = float((np.array(labels) != np.roll(np.array(labels), 1)).mean())
    context[3] = feat_mean[SMA_IDX]

    circadian = np.zeros(CIRCADIAN_DIM, dtype=np.float32)
    ts_arr = np.array(timestamps)
    if ts_arr.max() > ts_arr.min():
        ts_norm = (ts_arr - ts_arr.min()) / (ts_arr.max() - ts_arr.min() + 1e-8)
        bin_idx = np.clip((ts_norm * CIRCADIAN_DIM).astype(int), 0, CIRCADIAN_DIM - 1)
        sma_vals = feats[:, SMA_IDX]
        for i, (bi, sv) in enumerate(zip(bin_idx, sma_vals)):
            circadian[bi] += sv * q_weights[i]

    daily_vector = np.concatenate([micro_surrogate, daily_stats, context, circadian])
    assert len(daily_vector) == DAILY_VECTOR_DIM
    return daily_vector.astype(np.float32)


def save_daily_summary(subject_id: int, day_idx: int, vector: np.ndarray, out_dir: str):
    day_dir = os.path.join(out_dir, f'subject_{subject_id}', 'daily_summaries')
    os.makedirs(day_dir, exist_ok=True)
    torch.save(torch.tensor(vector), os.path.join(day_dir, f'day_{day_idx:03d}.pt'))


def save_metadata(subject_id: int, meta: dict, out_dir: str):
    meta_dir = os.path.join(out_dir, f'subject_{subject_id}')
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)