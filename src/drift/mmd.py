import torch
import numpy as np
from collections import deque


class MMDDetector:
    def __init__(self, ref_window_size: int = 500, test_window_size: int = 100,
                 kernel_bandwidth: float = 1.0, threshold: float = 0.05):
        self.ref_window_size  = ref_window_size
        self.test_window_size = test_window_size
        self.bandwidth = kernel_bandwidth
        self.threshold = threshold
        self.ref_buffer  = deque(maxlen=ref_window_size)
        self.test_buffer = deque(maxlen=test_window_size)
        self._drift_log = []
        self._fitted = False

    def fit(self, ref_data: np.ndarray):
        self.ref_buffer.clear()
        for row in ref_data:
            self.ref_buffer.append(row)
        self._fitted = True

    def update(self, feature_vec: np.ndarray, timestamp: float = None) -> dict:
        if not self._fitted:
            self.ref_buffer.append(feature_vec)
            if len(self.ref_buffer) >= self.ref_window_size:
                self._fitted = True
            return {"drift_detected": False, "mmd_value": 0.0, "timestamp": timestamp}

        self.test_buffer.append(feature_vec)
        if len(self.test_buffer) < self.test_window_size:
            return {"drift_detected": False, "mmd_value": 0.0, "timestamp": timestamp}

        ref  = np.array(self.ref_buffer)
        test = np.array(self.test_buffer)
        mmd_val = self._compute_mmd(ref, test)

        result = {
            "drift_detected": mmd_val > self.threshold,
            "mmd_value": float(mmd_val),
            "timestamp": timestamp,
        }
        if result["drift_detected"]:
            self._drift_log.append(result)
        return result

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        n = min(len(X), 200)
        m = min(len(Y), 100)
        idx_x = np.random.choice(len(X), n, replace=False)
        idx_y = np.random.choice(len(Y), m, replace=False)
        X_s = X[idx_x]
        Y_s = Y[idx_y]

        gamma = 1.0 / (2.0 * self.bandwidth ** 2)

        K_xx = np.exp(-gamma * np.sum((X_s[:, None] - X_s[None, :]) ** 2, axis=-1))
        K_yy = np.exp(-gamma * np.sum((Y_s[:, None] - Y_s[None, :]) ** 2, axis=-1))
        K_xy = np.exp(-gamma * np.sum((X_s[:, None] - Y_s[None, :]) ** 2, axis=-1))

        np.fill_diagonal(K_xx, 0)
        np.fill_diagonal(K_yy, 0)

        mmd = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2 * K_xy.sum() / (n * m)
        return max(0.0, mmd)

    def get_drift_log(self) -> list:
        return self._drift_log

    def adapt(self):
        if len(self.test_buffer) > 0:
            for sample in list(self.test_buffer):
                self.ref_buffer.append(sample)
            self.test_buffer.clear()
