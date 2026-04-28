import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

def bandpass_filter(signal: np.ndarray, fs: float, low: float=0.5, high: float=20.0, order: int=4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

def highpass_filter(signal: np.ndarray, fs: float, cutoff: float=0.5, order: int=4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal, axis=0)

def lowpass_filter(signal: np.ndarray, fs: float, cutoff: float=20.0, order: int=4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal, axis=0)

def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    g = gcd(target_fs, orig_fs)
    up, down = (target_fs // g, orig_fs // g)
    return resample_poly(signal, up, down, axis=0)

def compute_snr(signal: np.ndarray, noise_band_high: float=1.0, fs: float=50.0) -> float:
    fft = np.fft.rfft(signal, axis=0)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    power = np.abs(fft) ** 2
    noise_mask = freqs < noise_band_high
    noise_power = power[noise_mask].mean() + 1e-10
    signal_power = power[~noise_mask].mean() + 1e-10
    return float(10 * np.log10(signal_power / noise_power))

def handle_missing(signal: np.ndarray, threshold_interp: float=0.1, threshold_discard: float=0.3):
    if signal.ndim == 1:
        signal = signal[:, None]
        squeeze = True
    else:
        squeeze = False
    n = len(signal)
    nan_mask = np.isnan(signal).any(axis=1)
    missing_frac = nan_mask.mean()
    if missing_frac == 0:
        return (signal.squeeze(-1) if squeeze else signal, False)
    max_gap = 0
    curr = 0
    for v in nan_mask:
        curr = curr + 1 if v else 0
        max_gap = max(max_gap, curr)
    if max_gap / n > threshold_discard:
        return (signal.squeeze(-1) if squeeze else signal, True)
    x = np.arange(n, dtype=float)
    for ch in range(signal.shape[1]):
        valid = ~np.isnan(signal[:, ch])
        if valid.sum() > 1:
            signal[:, ch] = np.interp(x, x[valid], signal[valid, ch])
    return (signal.squeeze(-1) if squeeze else signal, False)

def remove_motion_artifact(ppg: np.ndarray, acc: np.ndarray, fs: float=50.0) -> np.ndarray:
    acc_mag = np.linalg.norm(acc, axis=-1, keepdims=True) if acc.ndim > 1 else acc[:, None]
    acc_lp = lowpass_filter(acc_mag, fs, cutoff=5.0)
    ppg_clean = ppg - acc_lp[:len(ppg)] * np.corrcoef(ppg.ravel(), acc_lp[:len(ppg)].ravel())[0, 1]
    return ppg_clean