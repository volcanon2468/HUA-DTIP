import torch
import numpy as np
from typing import List
from dataclasses import asdict

from src.simulation.what_if import WhatIfEngine, ScenarioResult


PRESET_SCENARIOS = [
    {"name": "conservative",  "intensity": 0.3, "duration_days": 21, "rest_extra_hours": 2.0,
     "nutrition_quality": 0.8, "sleep_consistency": 0.9},
    {"name": "moderate",      "intensity": 0.5, "duration_days": 14, "rest_extra_hours": 1.0,
     "nutrition_quality": 0.7, "sleep_consistency": 0.8},
    {"name": "aggressive",    "intensity": 0.8, "duration_days": 14, "rest_extra_hours": 0.5,
     "nutrition_quality": 0.6, "sleep_consistency": 0.7},
    {"name": "peak_week",     "intensity": 0.9, "duration_days": 7,  "rest_extra_hours": 0.0,
     "nutrition_quality": 0.9, "sleep_consistency": 0.6},
    {"name": "recovery_week", "intensity": 0.2, "duration_days": 7,  "rest_extra_hours": 3.0,
     "nutrition_quality": 0.9, "sleep_consistency": 0.95},
]


def rank_interventions(engine: WhatIfEngine, z0: torch.Tensor, n_days: int = 28,
                       custom_scenarios: list = None) -> List[ScenarioResult]:
    scenarios = [dict(s) for s in PRESET_SCENARIOS]
    if custom_scenarios:
        scenarios.extend(custom_scenarios)

    ranked = engine.compare_scenarios(z0, scenarios, n_days)
    return ranked


def print_ranking(ranked: List[ScenarioResult]):
    print(f"\n{'Rank':<5} {'Name':<20} {'FitnessScore':<14} {'OvertPr':<10} {'InjPr':<8} {'RiskMax':<10} {'PeakDay'}")
    print("-" * 80)
    for i, r in enumerate(ranked, 1):
        print(f"{i:<5} {r.name:<20} {r.fitness_score:<14.4f} {r.overtraining_prob:<10.4f} "
              f"{r.injury_prob:<8.4f} {r.max_risk:<10.4f} {r.peaking_day}")


def top_k_interventions(ranked: List[ScenarioResult], k: int = 3) -> List[ScenarioResult]:
    safe = [r for r in ranked if r.overtraining_prob < 0.15 and r.injury_prob < 0.10]
    if not safe:
        safe = ranked
    return safe[:k]


def build_periodized_plan(top1: ScenarioResult, n_weeks: int = 12) -> list:
    base = top1.plan.copy()
    plan_blocks = []
    for week in range(n_weeks):
        phase_frac = week / n_weeks
        intensity = base.get("intensity", 0.5) * (0.7 + 0.3 * phase_frac)
        if week % 4 == 3:
            intensity *= 0.6
        block = {
            "week": week + 1,
            "intensity": round(min(1.0, intensity), 2),
            "duration_days": 7,
            "rest_extra_hours": base.get("rest_extra_hours", 1.0) * max(0.5, 1.0 - phase_frac * 0.3),
            "nutrition_quality": base.get("nutrition_quality", 0.7),
            "sleep_consistency": base.get("sleep_consistency", 0.8),
        }
        plan_blocks.append(block)
    return plan_blocks
