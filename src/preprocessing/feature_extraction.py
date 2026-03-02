import numpy as np


def extract_imu_features(imu: np.ndarray) -> np.ndarray:
    """
    imu: [T, 9] — [wrist_acc_xyz, wrist_gyro_xyz, ankle_acc_xyz]
    Returns: 20-dim feature vector
    """
    acc = imu[:, :3]   # wrist accelerometer
    gyro = imu[:, 3:6] # wrist gyroscope

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # Signal magnitude area
    sma = acc_mag.mean()

    # Mean absolute deviation
    mad = np.abs(acc_mag - acc_mag.mean()).mean()

    # Energy proxy (sum of squares of FFT amplitudes)
    energy = np.mean(np.fft.rfft(acc_mag).__abs__() ** 2)

    # Step-related heuristics (zero-crossing rate of vertical axis)
    vert = acc[:, 2]
    zero_crosses = ((vert[:-1] * vert[1:]) < 0).sum()
    cadence = zero_crosses / (len(vert) / 50.0)  # crossings per second = rough step freq

    step_count = cadence * (len(acc) / 50.0) / 2  # rough steps in window

    # Step regularity: autocorrelation at ~1s lag (50 samples at 50Hz)
    lag = min(50, len(acc_mag) - 1)
    step_regularity = float(np.corrcoef(acc_mag[:-lag], acc_mag[lag:])[0, 1])

    # Gait symmetry: correlation between left half and right half acc magnitude
    half = len(acc_mag) // 2
    gait_symmetry = float(np.corrcoef(acc_mag[:half], acc_mag[half:half * 2])[0, 1])

    # Stride length proxy: peak-to-peak amplitude
    stride_length = float(acc_mag.max() - acc_mag.min())

    # Postural score: mean tilt (angle of gravity vector from vertical)
    gravity = acc.mean(axis=0)
    g_norm = np.linalg.norm(gravity) + 1e-8
    postural_score = float(np.arccos(np.clip(gravity[2] / g_norm, -1, 1)))

    # Transition count (large magnitude changes)
    diff_mag = np.abs(np.diff(acc_mag))
    transition_count = float((diff_mag > diff_mag.mean() + 2 * diff_mag.std()).sum())

    # Per-axis stats
    acc_mean = acc.mean(axis=0)   # [3]
    acc_std = acc.std(axis=0)     # [3]
    gyro_mean = gyro.mean(axis=0) # [3]
    gyro_std = gyro.std(axis=0)   # [3]

    features = np.array([
        cadence, step_count, step_regularity, sma, mad,
        energy, stride_length, gait_symmetry,
        *acc_mean, *acc_std, *gyro_mean, *gyro_std,
        transition_count, postural_score,
    ], dtype=np.float32)

    assert len(features) == 20, f"IMU feature dim mismatch: {len(features)}"
    return features


def extract_cardio_features(cardio: np.ndarray, hr: float = None, fs: float = 50.0) -> np.ndarray:
    """
    cardio: [T, C] — ECG or PPG channels
    hr: optional scalar heart rate from metadata
    Returns: 20-dim feature vector
    """
    sig = cardio[:, 0]  # Use first channel as primary

    # Amplitude envelope
    ppg_amplitude = float(np.percentile(sig, 95) - np.percentile(sig, 5))

    # Rise/fall time heuristic: time from 20% to 80% of range
    rng = sig.max() - sig.min() + 1e-8
    rising = sig > (sig.min() + 0.2 * rng)
    ppg_rise_time = float(rising.sum() / fs)
    falling = sig < (sig.min() + 0.8 * rng)
    ppg_fall_time = float(falling.sum() / fs)

    # ----- HR features -----
    # Detect peaks via zero-crossing of derivative
    dsig = np.diff(sig)
    peaks = np.where((dsig[:-1] > 0) & (dsig[1:] <= 0))[0]

    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs  # inter-peak intervals in seconds
        hr_mean = float(60.0 / rr.mean()) if rr.mean() > 0 else (hr or 70.0)
        hr_std  = float(60.0 * rr.std() / (rr.mean() ** 2 + 1e-8))
        hr_min  = float(60.0 / rr.max() if rr.max() > 0 else hr_mean)
        hr_max  = float(60.0 / rr.min() if rr.min() > 0 else hr_mean)
        hr_range = hr_max - hr_min
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        sdnn  = float(rr.std())
    else:
        hr_mean = hr or 70.0
        hr_std = hr_min = hr_max = hr_range = rmssd = sdnn = 0.0

    # Spectral features for HF/LF
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    lf_power = float(power[lf_mask].sum())
    hf_power = float(power[hf_mask].sum())
    lf_hf_ratio = lf_power / (hf_power + 1e-8)

    # Proxies
    hr_max_est = 220.0  # age-adjusted not available, use 220 default
    hr_reserve_pct = max(0.0, (hr_mean - 60.0) / (hr_max_est - 60.0 + 1e-8))
    recovery_rate_proxy = max(0.0, 1.0 - hr_mean / 150.0)
    resp_rate_proxy = float((power[(freqs >= 0.1) & (freqs < 0.4)]).argmax() * (fs / len(sig)) * 60)
    spo2_proxy = min(1.0, ppg_amplitude / (ppg_amplitude + 0.1))
    cardiac_output_proxy = hr_mean * ppg_amplitude / 100.0
    parasympathetic_index = hf_power / (lf_power + hf_power + 1e-8)
    sympathetic_index = lf_power / (lf_power + hf_power + 1e-8)

    features = np.array([
        hr_mean, hr_std, hr_min, hr_max, hr_range,
        rmssd, sdnn, lf_power, hf_power, lf_hf_ratio,
        hr_reserve_pct, recovery_rate_proxy,
        ppg_amplitude, ppg_rise_time, ppg_fall_time,
        resp_rate_proxy, spo2_proxy, cardiac_output_proxy,
        parasympathetic_index, sympathetic_index,
    ], dtype=np.float32)

    assert len(features) == 20
    return features


def extract_quality_features(imu: np.ndarray, cardio: np.ndarray) -> np.ndarray:
    """Returns 8 quality-indicator features."""
    from src.preprocessing.signal_cleaning import compute_snr

    imu_snr = compute_snr(imu[:, 0])
    cardio_snr = compute_snr(cardio[:, 0])
    ppg_snr = cardio_snr  # same channel if only PPG

    # Completeness: fraction of non-NaN samples
    completeness = float(1.0 - np.isnan(imu).mean())

    # Motion artifact: high-frequency energy ratio in IMU
    freqs = np.fft.rfftfreq(len(imu), d=1.0 / 50.0)
    power = np.abs(np.fft.rfft(imu[:, 0])) ** 2
    artifact = float(power[freqs > 10].sum() / (power.sum() + 1e-8))

    # Baseline wander: power below 0.5 Hz
    wander = float(power[freqs < 0.5].sum() / (power.sum() + 1e-8))

    # RR interval count from cardio channel 0
    sig = cardio[:, 0]
    dsig = np.diff(sig)
    peaks = np.where((dsig[:-1] > 0) & (dsig[1:] <= 0))[0]
    rr_count = float(max(0, len(peaks) - 1))

    # Beat quality: fraction of peaks with amplitude > 50% of max
    beat_quality = 0.0
    if len(peaks) > 0:
        peak_amps = sig[peaks]
        beat_quality = float((peak_amps > 0.5 * peak_amps.max()).mean())

    return np.array([
        imu_snr, cardio_snr, ppg_snr, completeness,
        artifact, wander, rr_count, beat_quality,
    ], dtype=np.float32)


def extract_all_features(imu: np.ndarray, cardio: np.ndarray, hr: float = None) -> np.ndarray:
    """
    Combines IMU (20) + cardio (20) + quality (8) = 48-dim feature vector.
    """
    feat_imu = extract_imu_features(imu)
    feat_cardio = extract_cardio_features(cardio, hr=hr)
    feat_quality = extract_quality_features(imu, cardio)
    full = np.concatenate([feat_imu, feat_cardio, feat_quality])
    assert len(full) == 48, f"Feature dim mismatch: {len(full)}"
    return full
