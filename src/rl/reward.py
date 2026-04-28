import torch
import numpy as np

class MultiObjectiveReward:

    def __init__(self, w_progress: float=0.35, w_safety: float=0.3, w_recovery: float=0.2, w_adherence: float=0.15):
        self.w_progress = w_progress
        self.w_safety = w_safety
        self.w_recovery = w_recovery
        self.w_adherence = w_adherence

    def compute(self, z_mu: np.ndarray, z_std: np.ndarray, action: np.ndarray, prev_z_mu: np.ndarray=None) -> float:
        capacity = float(z_mu[0])
        fatigue = float(z_mu[1])
        recovery = float(z_mu[2])
        cardio = float(z_mu[3])
        stability = float(z_mu[4])
        r_progress = capacity + 0.5 * cardio
        if prev_z_mu is not None:
            r_progress += max(0.0, z_mu[0] - prev_z_mu[0]) * 2.0
        r_safety = 0.0
        r_safety -= max(0.0, fatigue - 1.0) * 2.0
        r_safety -= max(0.0, z_std.mean() - 1.0)
        if fatigue > 2.0:
            r_safety -= 3.0
        r_recovery = recovery * (1.0 - action[0]) + action[2] * 0.5
        adherence = 1.0 - abs(action[0] - 0.5)
        consistency = stability * 0.5
        r_adherence = adherence + consistency
        total = self.w_progress * r_progress + self.w_safety * r_safety + self.w_recovery * r_recovery + self.w_adherence * r_adherence
        return float(total)

    def decompose(self, z_mu: np.ndarray, z_std: np.ndarray, action: np.ndarray) -> dict:
        capacity = float(z_mu[0])
        fatigue = float(z_mu[1])
        recovery = float(z_mu[2])
        cardio = float(z_mu[3])
        stability = float(z_mu[4])
        return {'progress': capacity + 0.5 * cardio, 'safety': -max(0.0, fatigue - 1.0) * 2.0 - max(0.0, z_std.mean() - 1.0), 'recovery': recovery * (1.0 - action[0]) + action[2] * 0.5, 'adherence': 1.0 - abs(action[0] - 0.5) + stability * 0.5}