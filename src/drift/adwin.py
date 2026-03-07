import numpy as np
from collections import deque

try:
    from river.drift import ADWIN as RiverADWIN
    _RIVER_AVAILABLE = True
except ImportError:
    _RIVER_AVAILABLE = False


class ADWINDetector:
    def __init__(self, delta: float = 0.002, n_features: int = 48):
        self.delta = delta
        self.n_features = n_features
        self.detectors = []
        self._drift_log = []
        self.reset()

    def reset(self):
        self.detectors = []
        if _RIVER_AVAILABLE:
            self.detectors = [RiverADWIN(delta=self.delta) for _ in range(self.n_features)]
        else:
            self.detectors = [_NumpyADWIN(self.delta) for _ in range(self.n_features)]
        self._drift_log = []

    def update(self, feature_vec: np.ndarray, timestamp: float = None) -> dict:
        assert len(feature_vec) == self.n_features
        drifted_features = []
        for i, val in enumerate(feature_vec):
            if _RIVER_AVAILABLE:
                self.detectors[i].update(float(val))
                if self.detectors[i].drift_detected:
                    drifted_features.append(i)
            else:
                if self.detectors[i].update(float(val)):
                    drifted_features.append(i)

        result = {
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "n_drifted": len(drifted_features),
            "timestamp": timestamp,
        }
        if result["drift_detected"]:
            self._drift_log.append(result)
        return result

    def get_drift_log(self) -> list:
        return self._drift_log


class _NumpyADWIN:
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque(maxlen=2000)
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.window.append(value)
        self.drift_detected = False
        if len(self.window) < 20:
            return False

        arr = np.array(self.window)
        n = len(arr)
        best_cut = -1
        best_eps = 0.0

        for cut in range(10, n - 10, max(1, n // 20)):
            left  = arr[:cut]
            right = arr[cut:]
            diff  = abs(left.mean() - right.mean())
            m     = 1.0 / (1.0 / len(left) + 1.0 / len(right))
            eps   = np.sqrt(np.log(2.0 / self.delta) / (2.0 * m))
            if diff > eps and diff > best_eps:
                best_eps = diff
                best_cut = cut

        if best_cut > 0:
            self.drift_detected = True
            for _ in range(best_cut):
                self.window.popleft()
            return True
        return False
