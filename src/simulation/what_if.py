import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, asdict

from src.simulation.mc_rollout import MCRolloutEngine, InterventionPlan, RolloutResult
from src.twin.bayesian_vae import BayesianVAE


@dataclass
class ScenarioResult:
    name:              str
    plan:              dict
    overtraining_prob: float
    injury_prob:       float
    peaking_day:       int
    avg_hr:            float
    avg_hrv_sdnn:      float
    max_risk:          float
    fitness_score:     float


def _fitness_score(result: RolloutResult, plan: InterventionPlan) -> float:
    hr_improvement = max(0.0, (result.hr_trajectory[-1] - result.hr_trajectory[0]) / 20.0)
    hrv_improvement = max(0.0, float(result.hrv_trajectory[-1, 0] - result.hrv_trajectory[0, 0]) / 30.0)
    risk_penalty = float(result.risk_scores.mean())
    fatigue_penalty = result.overtraining_prob + result.injury_prob
    return max(0.0, hr_improvement + hrv_improvement - risk_penalty - fatigue_penalty)


class WhatIfEngine:
    def __init__(self, engine: MCRolloutEngine, vae: BayesianVAE):
        self.engine = engine
        self.vae    = vae

    def query(self, z0: torch.Tensor, plan_kwargs: dict, n_days: int = 28) -> ScenarioResult:
        plan = InterventionPlan(**plan_kwargs)
        mu, logvar = self.vae.encoder(z0)
        std = torch.exp(0.5 * logvar)

        result = self.engine.rollout(mu.squeeze(0), std.squeeze(0), plan, n_days)

        return ScenarioResult(
            name=f"intensity={plan.intensity:.2f}_dur={plan.duration_days}d",
            plan=plan_kwargs,
            overtraining_prob=result.overtraining_prob,
            injury_prob=result.injury_prob,
            peaking_day=result.peaking_day,
            avg_hr=float(result.hr_trajectory.mean()),
            avg_hrv_sdnn=float(result.hrv_trajectory[:, 0].mean()),
            max_risk=float(result.risk_scores.max()),
            fitness_score=_fitness_score(result, plan),
        )

    def compare_scenarios(self, z0: torch.Tensor, scenarios: List[dict], n_days: int = 28) -> List[ScenarioResult]:
        results = []
        for scenario_kwargs in scenarios:
            kw = dict(scenario_kwargs)
            name = kw.pop("name", None)
            r = self.query(z0, kw, n_days)
            if name:
                r.name = name
            results.append(r)
        return sorted(results, key=lambda r: r.fitness_score, reverse=True)

    def grid_search(self, z0: torch.Tensor,
                    intensities: list = None, durations: list = None,
                    n_days: int = 28) -> List[ScenarioResult]:
        intensities = intensities or [0.3, 0.5, 0.7, 0.9]
        durations   = durations   or [7, 14, 21]
        scenarios = []
        for intensity in intensities:
            for dur in durations:
                scenarios.append({
                    "name": f"I{intensity:.1f}_D{dur}",
                    "intensity": intensity,
                    "duration_days": dur,
                })
        return self.compare_scenarios(z0, scenarios, n_days)
