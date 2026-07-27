import numpy as np


class MultiObjectiveReward:
    def __init__(
        self,
        w_progress : float = 0.30,
        w_safety   : float = 0.40,
        w_recovery : float = 0.20,
        w_stability: float = 0.10,
    ):
        self.w_progress  = w_progress
        self.w_safety    = w_safety
        self.w_recovery  = w_recovery
        self.w_stability = w_stability

    def compute(
        self,
        z_mu     : np.ndarray,
        z_std    : np.ndarray,
        action   : np.ndarray,
        prev_z_mu: np.ndarray = None,
    ) -> float:
        capacity  = float(z_mu[0])
        fatigue   = float(z_mu[1])
        recovery  = float(z_mu[2])
        cardio    = float(z_mu[3])
        stability = float(z_mu[4])
        intensity = float(action[0])

        r_progress = capacity + 0.5 * cardio
        if prev_z_mu is not None:
            r_progress += max(0.0, float(z_mu[0]) - float(prev_z_mu[0])) * 2.0

        r_safety = -0.40 * max(0.0, fatigue - 1.5)
        if fatigue > 1.5 and intensity > 0.7:
            r_safety -= 1.0
        if float(z_std.mean()) > 1.5:
            r_safety -= 0.5

        r_recovery = recovery * (1.0 - intensity) + float(action[2]) * 0.5

        r_stability = stability * 0.5 + (1.0 - abs(intensity - 0.5))

        total = (
            self.w_progress  * r_progress
            + self.w_safety    * r_safety
            + self.w_recovery  * r_recovery
            + self.w_stability * r_stability
            + 2.5
        )
        return float(total)

    def decompose(
        self,
        z_mu  : np.ndarray,
        z_std : np.ndarray,
        action: np.ndarray,
    ) -> dict:
        capacity  = float(z_mu[0])
        fatigue   = float(z_mu[1])
        recovery  = float(z_mu[2])
        cardio    = float(z_mu[3])
        stability = float(z_mu[4])
        intensity = float(action[0])

        r_progress  = capacity + 0.5 * cardio
        r_safety    = -0.40 * max(0.0, fatigue - 1.5)
        if fatigue > 1.5 and intensity > 0.7:
            r_safety -= 1.0
        if float(z_std.mean()) > 1.5:
            r_safety -= 0.5
        r_recovery  = recovery * (1.0 - intensity) + float(action[2]) * 0.5
        r_stability = stability * 0.5 + (1.0 - abs(intensity - 0.5))

        return {
            'progress' : r_progress,
            'safety'   : r_safety,
            'recovery' : r_recovery,
            'stability': r_stability,
        }
