import numpy as np
try:
    import neurokit2 as nk
    _NK_AVAILABLE = True
except ImportError:
    _NK_AVAILABLE = False

def compute_hrv_neurokit(ecg_or_ppg: np.ndarray, fs: float=50.0) -> np.ndarray:
    sig = ecg_or_ppg[:, 0] if ecg_or_ppg.ndim > 1 else ecg_or_ppg
    if _NK_AVAILABLE:
        try:
            signals, info = nk.ecg_process(sig, sampling_rate=int(fs))
            hrv_time = nk.hrv_time(signals, sampling_rate=int(fs), show=False)
            hrv_freq = nk.hrv_frequency(signals, sampling_rate=int(fs), show=False)
            sdnn = float(hrv_time.get('HRV_SDNN', [0.0])[0])
            rmssd = float(hrv_time.get('HRV_RMSSD', [0.0])[0])
            lf = float(hrv_freq.get('HRV_LF', [0.0])[0])
            hf = float(hrv_freq.get('HRV_HF', [0.0])[0])
            lf_hf = lf / (hf + 1e-08)
            return np.array([sdnn, rmssd, lf, hf, lf_hf], dtype=np.float32)
        except Exception:
            pass
    return _hrv_numpy_fallback(sig, fs)

def _hrv_numpy_fallback(sig: np.ndarray, fs: float) -> np.ndarray:
    dsig = np.diff(sig)
    peaks = np.where((dsig[:-1] > 0) & (dsig[1:] <= 0))[0]
    if len(peaks) < 3:
        return np.zeros(5, dtype=np.float32)
    rr = np.diff(peaks) / fs
    sdnn = float(rr.std())
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    lf = float(power[(freqs >= 0.04) & (freqs < 0.15)].sum())
    hf = float(power[(freqs >= 0.15) & (freqs < 0.4)].sum())
    lf_hf = lf / (hf + 1e-08)
    return np.array([sdnn, rmssd, lf, hf, lf_hf], dtype=np.float32)