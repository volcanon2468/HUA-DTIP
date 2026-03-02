import os
import numpy as np
import hydra
from omegaconf import DictConfig

from src.preprocessing.dataset_loaders import (
    MHEALTHDataset, PAMAP2Dataset, FourWeekPPGDataset,
    IMU_COLS, ECG_COLS, PAMAP2_IMU_COLS,
)
from src.preprocessing.signal_cleaning import bandpass_filter, highpass_filter, handle_missing
from src.preprocessing.windowing import process_subject
from src.preprocessing.feature_extraction import extract_all_features
from src.preprocessing.hrv import compute_hrv_neurokit
from src.preprocessing.daily_summary import build_daily_summary, save_daily_summary, save_metadata


def _hrv_fn(cardio: np.ndarray) -> np.ndarray:
    return compute_hrv_neurokit(cardio, fs=50.0)


def _feat_fn(imu: np.ndarray, cardio: np.ndarray) -> np.ndarray:
    return extract_all_features(imu, cardio)


def preprocess_mhealth(cfg: DictConfig, out_dir: str):
    print("Preprocessing MHEALTH...")
    ds = MHEALTHDataset(cfg.paths.mhealth)
    for df in ds.get_all_dfs():
        sid = int(df["subject_id"].iloc[0])
        imu_raw = df[IMU_COLS].values.astype(np.float32)
        ecg_raw = df[ECG_COLS].values.astype(np.float32)
        labels  = df["activity_label"].values

        imu_filt = bandpass_filter(imu_raw, fs=50.0, low=0.5, high=20.0)
        ecg_filt = highpass_filter(ecg_raw, fs=50.0, cutoff=0.5)

        stats = process_subject(
            imu_filt, ecg_filt, _feat_fn, _hrv_fn,
            subject_id=sid, out_dir=out_dir,
            window_size=cfg.window.size, stride=cfg.window.stride,
            label_seq=labels,
        )
        meta = {"subject_id": sid, "dataset": "mhealth", "n_windows": stats["windows_saved"]}
        save_metadata(sid, meta, out_dir)
        print(f"  Subject {sid}: {stats['windows_saved']} windows")


def preprocess_pamap2(cfg: DictConfig, out_dir: str):
    print("Preprocessing PAMAP2...")
    ds = PAMAP2Dataset(cfg.paths.pamap2)
    for df in ds.get_all_dfs():
        sid = int(df["subject_id"].iloc[0])
        imu_raw = df[PAMAP2_IMU_COLS].values.astype(np.float32)
        hr_col  = df["heart_rate"].values.astype(np.float32)
        cardio_raw = hr_col[:, None]
        imu_filt   = bandpass_filter(imu_raw, fs=100.0, low=0.5, high=20.0)

        stats = process_subject(
            imu_filt, cardio_raw, _feat_fn, _hrv_fn,
            subject_id=sid, out_dir=out_dir,
            window_size=cfg.window.size, stride=cfg.window.stride,
        )
        meta = {"subject_id": sid, "dataset": "pamap2", "n_windows": stats["windows_saved"]}
        save_metadata(sid, meta, out_dir)
        print(f"  Subject {sid}: {stats['windows_saved']} windows")


def preprocess_ppg_4week(cfg: DictConfig, out_dir: str):
    print("Preprocessing 4-Week PPG...")
    ds = FourWeekPPGDataset(cfg.paths.ppg_4week)
    for sid, days in ds.records.items():
        for day_idx, (day_key, df) in enumerate(sorted(days.items())):
            if "ppg" not in df.columns:
                continue
            ppg = df["ppg"].values.astype(np.float32)
            imu = np.zeros((len(ppg), 9), dtype=np.float32)
            if "acc_x" in df.columns:
                for i, col in enumerate(["acc_x", "acc_y", "acc_z"]):
                    if col in df.columns:
                        imu[:, i] = df[col].values.astype(np.float32)

            process_subject(
                imu, ppg[:, None], _feat_fn, _hrv_fn,
                subject_id=sid, out_dir=out_dir,
                window_size=cfg.window.size, stride=cfg.window.stride,
            )

            windows_dir = os.path.join(out_dir, f"subject_{sid}", "windows")
            daily_vec   = build_daily_summary(windows_dir)
            save_daily_summary(sid, day_idx, daily_vec, out_dir)

        meta = {"subject_id": sid, "dataset": "ppg_4week"}
        save_metadata(sid, meta, out_dir)

    print(f"  {len(ds.records)} subjects processed.")


@hydra.main(config_path="../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig):
    out_dir = cfg.paths.processed
    os.makedirs(out_dir, exist_ok=True)
    preprocess_mhealth(cfg, out_dir)
    preprocess_pamap2(cfg, out_dir)
    preprocess_ppg_4week(cfg, out_dir)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
