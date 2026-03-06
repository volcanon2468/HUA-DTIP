import numpy as np
from dataclasses import dataclass


@dataclass
class SafetyBounds:
    max_intensity_when_fatigued: float = 0.4
    fatigue_threshold: float = 1.5
    max_consecutive_high_days: int = 3
    min_rest_hours: float = 0.5
    max_hr_deviation: float = 30.0
    min_sleep_consistency: float = 0.5


class SafetyGuard:
    def __init__(self, bounds: SafetyBounds = None):
        self.bounds = bounds or SafetyBounds()
        self.consecutive_high_days = 0
        self.episode_violations = 0

    def reset(self):
        self.consecutive_high_days = 0
        self.episode_violations = 0

    def check_and_clip(self, action: np.ndarray, z_mu: np.ndarray, z_std: np.ndarray) -> np.ndarray:
        action = action.copy()
        fatigue   = float(z_mu[1]) if len(z_mu) > 1 else 0.0
        intensity = action[0]

        if fatigue > self.bounds.fatigue_threshold:
            action[0] = min(action[0], self.bounds.max_intensity_when_fatigued)
            action[2] = max(action[2], 0.5)
            self.episode_violations += 1

        if intensity > 0.7:
            self.consecutive_high_days += 1
        else:
            self.consecutive_high_days = 0

        if self.consecutive_high_days >= self.bounds.max_consecutive_high_days:
            action[0] = min(action[0], 0.3)
            action[2] = max(action[2], 0.6)
            self.consecutive_high_days = 0
            self.episode_violations += 1

        action[2] = max(action[2], self.bounds.min_rest_hours / 8.0)
        action[4] = max(action[4], self.bounds.min_sleep_consistency)

        uncertainty = float(z_std.mean()) if z_std is not None else 0.0
        if uncertainty > 2.0:
            action[0] = min(action[0], 0.3)
            self.episode_violations += 1

        action = np.clip(action, 0.0, 1.0)
        return action

    def compute_penalty(self, action: np.ndarray, z_mu: np.ndarray, z_std: np.ndarray) -> float:
        penalty = 0.0
        fatigue = float(z_mu[1]) if len(z_mu) > 1 else 0.0

        if fatigue > self.bounds.fatigue_threshold and action[0] > self.bounds.max_intensity_when_fatigued:
            penalty += 2.0 * (action[0] - self.bounds.max_intensity_when_fatigued)

        if self.consecutive_high_days >= self.bounds.max_consecutive_high_days:
            penalty += 1.5

        uncertainty = float(z_std.mean()) if z_std is not None else 0.0
        if uncertainty > 2.0 and action[0] > 0.5:
            penalty += 1.0

        return penalty

    def get_stats(self) -> dict:
        return {
            "episode_violations": self.episode_violations,
            "consecutive_high_days": self.consecutive_high_days,
        }
