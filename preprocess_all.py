import os
import sys
import glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.signal_cleaning import bandpass_filter, handle_missing, compute_snr
from src.preprocessing.windowing import create_windows, per_subject_zscore, assess_quality
from src.preprocessing.feature_extraction import extract_all_features
from src.preprocessing.hrv import compute_hrv_neurokit
from src.preprocessing.daily_summary import build_daily_summary, save_daily_summary, save_metadata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

WINDOW_SIZE = 1000
STRIDE = 500
SAMPLE_RATE = 50


def save_window(win_dir, t, imu_win, cardio_win, label=-1, timestamp=0.0):
    imu_t = torch.tensor(imu_win, dtype=torch.float32)
    cardio_t = torch.tensor(cardio_win, dtype=torch.float32)
    quality = assess_quality(imu_win)

    features = extract_all_features(imu_win, cardio_win)
    ecg_col = cardio_win[:, 0] if cardio_win.shape[1] > 0 else np.zeros(WINDOW_SIZE)
    hrv = compute_hrv_neurokit(ecg_col, SAMPLE_RATE)

    window_data = {
        "imu": imu_t,
        "cardio": cardio_t,
        "features": torch.tensor(features, dtype=torch.float32),
        "hrv": torch.tensor(hrv, dtype=torch.float32),
        "label": int(label),
        "quality": float(quality),
        "timestamp": float(timestamp),
    }
    torch.save(window_data, os.path.join(win_dir, f"window_{t:05d}.pt"))


def build_and_save_daily(sid, win_dir, n_windows):
    if n_windows <= 0:
        return
    windows_per_day = min(200, max(1, n_windows))
    n_days = max(1, n_windows // windows_per_day)
    for d in range(n_days):
        vector = build_daily_summary(win_dir)
        save_daily_summary(sid, d, vector, PROCESSED_DIR)


def pad_to_9_channels(data, n_channels=9):
    if data.shape[1] >= n_channels:
        return data[:, :n_channels]
    pad = np.zeros((len(data), n_channels - data.shape[1]))
    return np.hstack([data, pad])


def pad_to_2_channels(data, n_channels=2):
    if data.shape[1] >= n_channels:
        return data[:, :n_channels]
    pad = np.zeros((len(data), n_channels - data.shape[1]))
    return np.hstack([data, pad])


def interp_nans(arr):
    for col in range(arr.shape[1]):
        nans = np.isnan(arr[:, col])
        if nans.all():
            arr[:, col] = 0.0
            continue
        if nans.any():
            valid = ~nans
            arr[nans, col] = np.interp(
                np.where(nans)[0], np.where(valid)[0], arr[valid, col]
            )
    return arr


def process_subject(sid, imu_data, cardio_data, labels=None, dataset_name="unknown"):
    imu_data, _ = handle_missing(imu_data)
    imu_data = pad_to_9_channels(imu_data)
    cardio_data = pad_to_2_channels(cardio_data)

    imu_windows = create_windows(imu_data, WINDOW_SIZE, STRIDE)
    cardio_windows = create_windows(cardio_data, WINDOW_SIZE, STRIDE)

    if labels is not None:
        label_windows = create_windows(labels.reshape(-1, 1), WINDOW_SIZE, STRIDE)
    else:
        label_windows = None

    imu_windows, _ = per_subject_zscore(imu_windows)

    n_windows = min(len(imu_windows), len(cardio_windows))
    if label_windows is not None:
        n_windows = min(n_windows, len(label_windows))

    if n_windows == 0:
        return 0

    win_dir = os.path.join(PROCESSED_DIR, f"subject_{sid}", "windows")
    os.makedirs(win_dir, exist_ok=True)

    for t in range(n_windows):
        label = int(np.median(label_windows[t])) if label_windows is not None else -1
        save_window(win_dir, t, imu_windows[t], cardio_windows[t],
                    label=label, timestamp=float(t * STRIDE / SAMPLE_RATE))

    build_and_save_daily(sid, win_dir, n_windows)
    save_metadata(sid, {"dataset": dataset_name, "n_windows": n_windows, "sample_rate": SAMPLE_RATE}, PROCESSED_DIR)
    return n_windows


def process_mhealth():
    print("\n=== Processing MHEALTH ===")
    mhealth_dir = os.path.join(RAW_DIR, "mhealth")
    files = sorted(glob.glob(os.path.join(mhealth_dir, "**", "mHealth_subject*.log"), recursive=True))

    if not files:
        print("  No MHEALTH files found — skipping")
        return

    print(f"  Found {len(files)} subject files")
    for fpath in files:
        fname = os.path.basename(fpath)
        sid = int(fname.replace("mHealth_subject", "").replace(".log", ""))
        print(f"  Subject {sid}...", end=" ", flush=True)

        data = np.loadtxt(fpath)
        imu_data = data[:, 0:9]
        ecg_data = data[:, 9:11] if data.shape[1] > 10 else np.zeros((len(data), 2))
        labels = data[:, -1].astype(int)

        n = process_subject(sid, imu_data, ecg_data, labels, "mhealth")
        print(f"{n} windows")


def process_pamap2():
    print("\n=== Processing PAMAP2 ===")
    pamap2_dir = os.path.join(RAW_DIR, "pamap2")
    files = sorted(glob.glob(os.path.join(pamap2_dir, "**", "subject10*.dat"), recursive=True))

    if not files:
        print("  No PAMAP2 files found — skipping")
        return

    print(f"  Found {len(files)} subject files")
    for fpath in files:
        fname = os.path.basename(fpath)
        sid = int(fname.replace("subject", "").replace(".dat", ""))
        print(f"  Subject {sid}...", end=" ", flush=True)

        try:
            data = np.loadtxt(fpath, dtype=float)
        except Exception:
            data = np.genfromtxt(fpath, dtype=float)

        mask = data[:, 1] != 0
        data = data[mask]

        if len(data) < WINDOW_SIZE:
            print(f"too few samples ({len(data)}) — skipping")
            continue

        imu_cols = list(range(4, min(13, data.shape[1])))
        imu_data = data[:, imu_cols]
        imu_data = interp_nans(imu_data)

        hr_data = data[:, 2:3] if data.shape[1] > 2 else np.zeros((len(data), 1))
        hr_data = interp_nans(hr_data)
        cardio = np.hstack([hr_data, np.zeros((len(hr_data), 1))])

        labels = data[:, 1].astype(int)

        n = process_subject(sid, imu_data, cardio, labels, "pamap2")
        print(f"{n} windows")


def process_ppg():
    print("\n=== Processing 4-Week PPG ===")
    ppg_dir = os.path.join(RAW_DIR, "ppg_4week")
    hrv_file = os.path.join(ppg_dir, "sensor_hrv.csv")

    if not os.path.exists(hrv_file):
        print("  No PPG sensor_hrv.csv found — skipping")
        return

    import pandas as pd
    print("  Loading sensor_hrv.csv...")
    df = pd.read_csv(hrv_file)

    devices = df["deviceId"].unique()
    print(f"  Found {len(devices)} unique devices (subjects)")

    for i, dev_id in enumerate(sorted(devices)):
        sid = 500 + i
        sub_df = df[df["deviceId"] == dev_id].copy()
        sub_df = sub_df.sort_values("ts_start")

        acc_cols = [c for c in sub_df.columns if c.startswith("acc_")]
        hr_col = "HR" if "HR" in sub_df.columns else None

        if not acc_cols:
            print(f"  Device {dev_id}: no acc columns — skipping")
            continue

        acc_data = sub_df[acc_cols].values.astype(float)
        acc_data = interp_nans(np.nan_to_num(acc_data, nan=0.0))

        hr_data = sub_df[hr_col].values.astype(float).reshape(-1, 1) if hr_col else np.zeros((len(acc_data), 1))
        hr_data = np.nan_to_num(hr_data, nan=70.0)

        hrv_cols = ["sdnn", "rmssd", "lf", "hf", "lf/hf"]
        hrv_available = [c for c in hrv_cols if c in sub_df.columns]

        n_samples = len(acc_data)
        if n_samples < 20:
            continue

        repeats = max(1, WINDOW_SIZE // n_samples + 1)
        imu_expanded = np.tile(acc_data, (repeats, 1))[:WINDOW_SIZE * (n_samples // 1 + 1)]
        cardio_expanded = np.tile(np.hstack([hr_data, np.zeros((len(hr_data), 1))]), (repeats, 1))[:len(imu_expanded)]

        imu_expanded = pad_to_9_channels(imu_expanded)
        cardio_expanded = pad_to_2_channels(cardio_expanded)

        imu_windows = create_windows(imu_expanded, WINDOW_SIZE, STRIDE)
        cardio_windows = create_windows(cardio_expanded, WINDOW_SIZE, STRIDE)
        imu_windows, _ = per_subject_zscore(imu_windows)

        n_windows = min(len(imu_windows), len(cardio_windows))
        if n_windows == 0:
            continue

        win_dir = os.path.join(PROCESSED_DIR, f"subject_{sid}", "windows")
        os.makedirs(win_dir, exist_ok=True)

        for t in range(n_windows):
            save_window(win_dir, t, imu_windows[t], cardio_windows[t],
                        label=-1, timestamp=float(t * STRIDE / SAMPLE_RATE))

        build_and_save_daily(sid, win_dir, n_windows)
        save_metadata(sid, {"dataset": "ppg_4week", "n_windows": n_windows}, PROCESSED_DIR)
        print(f"  Subject {sid} (device {dev_id[:8]}): {n_windows} windows")


def process_stroke_rehab():
    print("\n=== Processing Stroke Rehab ===")
    sr_dir = os.path.join(RAW_DIR, "stroke_rehab", "raw")
    if not os.path.exists(sr_dir):
        print("  No stroke_rehab/raw directory — skipping")
        return

    patient_dirs = sorted([d for d in os.listdir(sr_dir) if d.startswith("imu")])
    print(f"  Found {len(patient_dirs)} patients")

    for pi, patient_id in enumerate(patient_dirs):
        sid = 400 + pi
        all_data = []

        for visit in ["visit1", "visit2"]:
            imu_dir = os.path.join(sr_dir, patient_id, visit, "imu")
            if not os.path.exists(imu_dir):
                continue
            for csv_file in sorted(os.listdir(imu_dir)):
                if not csv_file.endswith(".csv"):
                    continue
                fpath = os.path.join(imu_dir, csv_file)
                try:
                    import pandas as pd
                    df = pd.read_csv(fpath, header=None)
                    numeric_cols = df.select_dtypes(include=[np.number]).values
                    if len(numeric_cols) > 0:
                        all_data.append(numeric_cols)
                except Exception:
                    continue

        if not all_data:
            print(f"  Patient {patient_id}: no usable data — skipping")
            continue

        combined = np.vstack(all_data)
        if len(combined) < WINDOW_SIZE:
            print(f"  Patient {patient_id}: too few samples — skipping")
            continue

        imu_data = pad_to_9_channels(combined)
        imu_data = interp_nans(np.nan_to_num(imu_data, nan=0.0))
        cardio = np.zeros((len(imu_data), 2))

        n = process_subject(sid, imu_data, cardio, None, "stroke_rehab")
        print(f"  Patient {patient_id} (subject {sid}): {n} windows")


def process_capture24():
    print("\n=== Processing CAPTURE-24 ===")
    cap_dir = os.path.join(RAW_DIR, "capture24")
    gz_files = sorted(glob.glob(os.path.join(cap_dir, "P*.csv.gz")))

    if not gz_files:
        feat_file = os.path.join(cap_dir, "feats_all.csv.gz")
        if os.path.exists(feat_file):
            import pandas as pd
            print("  Using precomputed features from feats_all.csv.gz...")
            df = pd.read_csv(feat_file)
            numeric = df.select_dtypes(include=[np.number])
            if len(numeric.columns) >= 3:
                n_subjects = min(151, len(df) // 50)
                rows_per_subject = len(df) // max(1, n_subjects)
                for i in range(n_subjects):
                    sid = 300 + i
                    start = i * rows_per_subject
                    end = start + rows_per_subject
                    sub = numeric.iloc[start:end].values
                    if len(sub) < WINDOW_SIZE:
                        sub = np.tile(sub, (WINDOW_SIZE // len(sub) + 1, 1))[:WINDOW_SIZE * 2]
                    imu_data = pad_to_9_channels(sub[:, :min(9, sub.shape[1])])
                    imu_data = np.nan_to_num(imu_data, nan=0.0)
                    cardio = np.zeros((len(imu_data), 2))
                    n = process_subject(sid, imu_data, cardio, None, "capture24")
                    if n > 0:
                        print(f"  Subject {sid}: {n} windows")
            return

        print("  No CAPTURE-24 data found — skipping")
        return

    print(f"  Found {len(gz_files)} raw accelerometer files")
    import pandas as pd
    for fi, fpath in enumerate(gz_files):
        sid = 300 + fi
        print(f"  Processing {os.path.basename(fpath)}...", end=" ", flush=True)
        try:
            df = pd.read_csv(fpath, nrows=100000)
            xyz = df[["x", "y", "z"]].values.astype(float)
            xyz = np.nan_to_num(xyz, nan=0.0)
            imu_data = pad_to_9_channels(xyz)
            cardio = np.zeros((len(imu_data), 2))
            n = process_subject(sid, imu_data, cardio, None, "capture24")
            print(f"{n} windows")
        except Exception as e:
            print(f"failed: {e}")


def process_mex():
    print("\n=== Processing MEx ===")
    mex_dir = os.path.join(RAW_DIR, "mex")
    acw_dir = os.path.join(mex_dir, "acw")

    if not os.path.exists(acw_dir):
        print("  No MEx acw directory — skipping")
        return

    subject_dirs = sorted(os.listdir(acw_dir))
    print(f"  Found {len(subject_dirs)} subjects in acw/")

    for si, subj_id in enumerate(subject_dirs):
        sid = 200 + si
        subj_dir = os.path.join(acw_dir, subj_id)
        if not os.path.isdir(subj_dir):
            continue

        all_data = []
        for csv_file in sorted(os.listdir(subj_dir)):
            if not csv_file.endswith(".csv"):
                continue
            fpath = os.path.join(subj_dir, csv_file)
            try:
                import pandas as pd
                df = pd.read_csv(fpath, header=None)
                numeric_cols = df.select_dtypes(include=[np.number])
                if len(numeric_cols.columns) >= 3:
                    all_data.append(numeric_cols.values)
            except Exception:
                continue

        if not all_data:
            continue

        combined = np.vstack(all_data)
        if len(combined) < WINDOW_SIZE:
            continue

        imu_data = pad_to_9_channels(combined[:, :min(3, combined.shape[1])])
        imu_data = np.nan_to_num(imu_data, nan=0.0)
        cardio = np.zeros((len(imu_data), 2))

        n = process_subject(sid, imu_data, cardio, None, "mex")
        print(f"  Subject {sid} (MEx {subj_id}): {n} windows")


def check_result():
    print(f"\n{'='*60}")
    print("  PREPROCESSING RESULTS")
    print(f"{'='*60}")
    subjects = sorted(glob.glob(os.path.join(PROCESSED_DIR, "subject_*")))
    total_windows = 0
    total_daily = 0
    for s in subjects:
        wins = glob.glob(os.path.join(s, "windows", "window_*.pt"))
        days = glob.glob(os.path.join(s, "daily_summaries", "day_*.pt"))
        total_windows += len(wins)
        total_daily += len(days)
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Total windows:  {total_windows}")
    print(f"  Total daily summaries: {total_daily}")

    if total_windows > 0:
        sample_path = glob.glob(os.path.join(PROCESSED_DIR, "subject_*", "windows", "window_00000.pt"))
        if sample_path:
            sample = torch.load(sample_path[0], map_location="cpu")
            print(f"\n  Sample window keys: {list(sample.keys())}")
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k:12s} shape={list(v.shape)} dtype={v.dtype}")
                else:
                    print(f"    {k:12s} = {v}")

    return total_windows > 0


if __name__ == "__main__":
    print("HUA-DTIP — Preprocessing Pipeline")
    print("=" * 60)

    process_mhealth()
    process_pamap2()
    process_ppg()
    process_stroke_rehab()
    process_capture24()
    process_mex()

    check_result()
    print("\nDone! Next step: python -m train.train_encoders")
