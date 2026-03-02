import json
import os
import glob
import numpy as np
import torch


def build_daily_summary(windows_dir: str) -> np.ndarray:
    """
    Aggregates all window .pt files in a day's windows directory into a 512-dim daily vector.
    Composition:
      - 256: micro-scale surrogate (mean + std of 48 features → 96 dims + 160 zeros for now)
      - 128: daily stats from cardio/HRV
      - 64:  context features (step count, fatigue proxy, etc.)
      - 64:  circadian features (activity pattern across day)
    """
    paths = sorted(glob.glob(os.path.join(windows_dir, "window_*.pt")))
    if not paths:
        return np.zeros(512, dtype=np.float32)

    features_list, hrv_list, qualities, labels, timestamps = [], [], [], [], []
    for p in paths:
        w = torch.load(p, map_location="cpu")
        features_list.append(w["features"].numpy())
        hrv_list.append(w["hrv"].numpy())
        qualities.append(float(w["quality"]))
        labels.append(int(w["label"]))
        timestamps.append(float(w["timestamp"]))

    feats = np.stack(features_list)    # [N, 48]
    hrvs  = np.stack(hrv_list)         # [N, 5]

    # Quality-weighted aggregation
    q_weights = np.array(qualities) + 1e-6
    q_weights /= q_weights.sum()

    feat_mean = (feats * q_weights[:, None]).sum(axis=0)  # [48]
    feat_std  = feats.std(axis=0)                          # [48]

    # --- 256-dim micro-surrogate ---
    micro_surrogate = np.zeros(256, dtype=np.float32)
    micro_surrogate[:48] = feat_mean
    micro_surrogate[48:96] = feat_std

    # --- 128-dim daily stats (cardio/HRV) ---
    hrv_mean = hrvs.mean(axis=0)  # [5]
    hrv_std  = hrvs.std(axis=0)   # [5]
    # Cardio features are indices 20-39 in the 48-dim vector
    cardio_mean = feat_mean[20:40]  # [20]
    cardio_std  = feat_std[20:40]   # [20]
    daily_stats = np.zeros(128, dtype=np.float32)
    daily_stats[:5]  = hrv_mean
    daily_stats[5:10] = hrv_std
    daily_stats[10:30] = cardio_mean
    daily_stats[30:50] = cardio_std

    # --- 64-dim context ---
    context = np.zeros(64, dtype=np.float32)
    context[0] = feat_mean[1]   # step_count
    context[1] = float(np.array(qualities).mean())  # avg quality
    context[2] = float((np.array(labels) != np.roll(np.array(labels), 1)).mean())  # activity variability
    context[3] = feat_mean[3]   # sma (activity level proxy)

    # --- 64-dim circadian (activity across 24h bins) ---
    circadian = np.zeros(64, dtype=np.float32)
    ts_arr = np.array(timestamps)
    if ts_arr.max() > ts_arr.min():
        # Normalize timestamps to [0, 1] and bin into 64 slots
        ts_norm = (ts_arr - ts_arr.min()) / (ts_arr.max() - ts_arr.min() + 1e-8)
        bin_idx = np.clip((ts_norm * 64).astype(int), 0, 63)
        sma_vals = feats[:, 3]  # SMA as activity indicator
        for i, (bi, sv) in enumerate(zip(bin_idx, sma_vals)):
            circadian[bi] += sv * q_weights[i]

    daily_vector = np.concatenate([micro_surrogate, daily_stats, context, circadian])
    assert len(daily_vector) == 512
    return daily_vector.astype(np.float32)


def save_daily_summary(subject_id: int, day_idx: int, vector: np.ndarray, out_dir: str):
    day_dir = os.path.join(out_dir, f"subject_{subject_id}", "daily_summaries")
    os.makedirs(day_dir, exist_ok=True)
    torch.save(torch.tensor(vector), os.path.join(day_dir, f"day_{day_idx:03d}.pt"))


def save_metadata(subject_id: int, meta: dict, out_dir: str):
    meta_dir = os.path.join(out_dir, f"subject_{subject_id}")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
