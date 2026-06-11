from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from src.rl.safety import SafetyGuard
from src.simulation.intervention_ranking import build_periodized_plan, top_k_interventions
from src.simulation.what_if import ScenarioResult

@dataclass
class Stage10Config:
    uncertainty_conservative_threshold: float = 0.8
    critical_fatigue_score: float = 85.0
    max_consecutive_high_days: int = 3
    low_injury_risk_threshold: float = 0.1
    moderate_injury_risk_threshold: float = 0.2
    high_confidence_threshold: float = 80.0
    medium_confidence_threshold: float = 60.0
    attribution_steps: int = 32
    n_top_reasons: int = 3

@dataclass
class PatientSnapshot:
    subject_id: str
    source_window: str
    z_mu: np.ndarray
    z_std: np.ndarray
    capacity_score: float
    fatigue_score: float
    confidence_pct: float
    avg_uncertainty: float
    sleep_hours_last_3_nights: float
    weekly_training_load: float
    hrv_7day_trend: float
    day_of_week: int
    high_intensity_streak: int = 0
    drift_diagnosis: Optional[Dict[str, Any]] = None

@dataclass
class SafetyGateResult:
    mode: str
    passed: bool
    reasons: List[str] = field(default_factory=list)

@dataclass
class AttributionReason:
    feature: str
    attribution: float
    direction: str
    explanation: str

@dataclass
class RecommendationReport:
    status: str
    selected_plan_name: str
    selected_plan: Dict[str, Any]
    raw_action: List[float]
    safe_action: List[float]
    safety_gate: SafetyGateResult
    top_reasons: List[AttributionReason]
    counterfactual: str
    expected_outcome: str
    injury_risk_label: str
    injury_risk_pct: float
    overtraining_risk_pct: float
    weekly_plan: List[Dict[str, Any]]
    plain_language: str
    ranked_scenarios: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

def _sigmoid_score(value: float) -> float:
    return float(100.0 / (1.0 + np.exp(-float(value))))

def _confidence_from_uncertainty(avg_uncertainty: float) -> float:
    return float(np.clip((1.0 - avg_uncertainty) * 100.0, 0.0, 100.0))

def make_snapshot(*, subject_id: str, source_window: str, z_mu: np.ndarray, z_std: np.ndarray, sleep_hours_last_3_nights: float, weekly_training_load: float, hrv_7day_trend: float, day_of_week: int, high_intensity_streak: int=0, drift_diagnosis: Optional[Dict[str, Any]]=None) -> PatientSnapshot:
    z_mu = np.asarray(z_mu, dtype=np.float32).reshape(-1)
    z_std = np.asarray(z_std, dtype=np.float32).reshape(-1)
    capacity_score = _sigmoid_score(z_mu[0]) if z_mu.size else 50.0
    fatigue_score = _sigmoid_score(z_mu[1]) if z_mu.size > 1 else 50.0
    avg_uncertainty = float(np.mean(z_std)) if z_std.size else 1.0
    return PatientSnapshot(subject_id=subject_id, source_window=source_window, z_mu=z_mu, z_std=z_std, capacity_score=capacity_score, fatigue_score=fatigue_score, confidence_pct=_confidence_from_uncertainty(avg_uncertainty), avg_uncertainty=avg_uncertainty, sleep_hours_last_3_nights=float(sleep_hours_last_3_nights), weekly_training_load=float(np.clip(weekly_training_load, 0.0, 1.0)), hrv_7day_trend=float(np.clip(hrv_7day_trend, -1.0, 1.0)), day_of_week=int(day_of_week) % 7, high_intensity_streak=int(high_intensity_streak), drift_diagnosis=drift_diagnosis)

class _DecisionScoreModel(nn.Module):

    def __init__(self, actor: Optional[nn.Module], actor_state_dim: int):
        super().__init__()
        self.actor = actor
        self.actor_state_dim = actor_state_dim

    def forward(self, state68: torch.Tensor) -> torch.Tensor:
        actor_state = state68[:, :self.actor_state_dim]
        if self.actor is None:
            action = torch.stack([torch.sigmoid(actor_state[:, 0] - actor_state[:, 1]), torch.full_like(actor_state[:, 0], 0.5), torch.sigmoid(actor_state[:, 1]), torch.full_like(actor_state[:, 0], 0.7), torch.full_like(actor_state[:, 0], 0.8), torch.sigmoid(actor_state[:, 0])], dim=-1)
        else:
            action = self.actor.deterministic(actor_state)
        score = 0.55 * action[:, 0] + 0.15 * action[:, 3] + 0.15 * action[:, 4] - 0.1 * action[:, 2] + 0.05 * action[:, 5]
        extras = state68[:, self.actor_state_dim:]
        if extras.shape[1] >= 4:
            sleep_norm = extras[:, 0]
            load = extras[:, 1]
            hrv_trend = extras[:, 2]
            day_phase = extras[:, 3]
            score = score + 0.1 * sleep_norm - 0.12 * load + 0.1 * hrv_trend
            score = score + 0.02 * torch.cos(day_phase * 2.0 * torch.pi)
        return score

class RecommendationEngine:

    def __init__(self, actor: Optional[nn.Module]=None, actor_state_dim: int=64, device: str='cpu', config: Optional[Stage10Config]=None):
        self.actor = actor
        self.actor_state_dim = int(actor_state_dim)
        self.device = torch.device(device)
        self.config = config or Stage10Config()
        self.safety_guard = SafetyGuard()
        if self.actor is not None:
            self.actor.to(self.device).eval()

    def build_policy_state(self, snapshot: PatientSnapshot) -> torch.Tensor:
        base = np.concatenate([snapshot.z_mu, snapshot.z_std]).astype(np.float32)
        if base.size < self.actor_state_dim:
            base = np.pad(base, (0, self.actor_state_dim - base.size))
        else:
            base = base[:self.actor_state_dim]
        context = np.array([np.clip(snapshot.sleep_hours_last_3_nights / 9.0, 0.0, 1.0), np.clip(snapshot.weekly_training_load, 0.0, 1.0), np.clip(snapshot.hrv_7day_trend, -1.0, 1.0), snapshot.day_of_week / 6.0], dtype=np.float32)
        state = np.concatenate([base, context]).astype(np.float32)
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def feature_names(self, latent_dim: int) -> List[str]:
        semantic = {0: 'capacity', 1: 'fatigue', 2: 'recovery', 3: 'cardio', 4: 'stability'}
        mu = [f"mu_{semantic.get(i, f'latent_{i:02d}')}" for i in range(latent_dim)]
        sigma = [f"sigma_{semantic.get(i, f'latent_{i:02d}')}" for i in range(latent_dim)]
        names = (mu + sigma)[:self.actor_state_dim]
        if len(names) < self.actor_state_dim:
            names.extend([f'padding_{i:02d}' for i in range(self.actor_state_dim - len(names))])
        names.extend(['sleep_hours_last_3_nights', 'weekly_training_load', 'HRV_7day_trend', 'day_of_week_phase'])
        return names

    def propose_action(self, snapshot: PatientSnapshot) -> np.ndarray:
        state = self.build_policy_state(snapshot)[:, :self.actor_state_dim]
        rule_action = self._rule_action(snapshot)
        if self.actor is None:
            return rule_action
        with torch.no_grad():
            actor_action = self.actor.deterministic(state).squeeze(0).detach().cpu().numpy()
        action = 0.45 * actor_action.astype(np.float32) + 0.55 * rule_action
        return np.clip(action.astype(np.float32), 0.0, 1.0)

    def _rule_action(self, snapshot: PatientSnapshot) -> np.ndarray:
        capacity = snapshot.capacity_score / 100.0
        fatigue = snapshot.fatigue_score / 100.0
        recovery_bonus = max(0.0, snapshot.hrv_7day_trend) * 0.12
        load_penalty = snapshot.weekly_training_load * 0.18
        intensity = np.clip(0.22 + 0.52 * capacity - 0.42 * fatigue + recovery_bonus - load_penalty, 0.05, 0.75)
        rest = np.clip(0.18 + 0.48 * fatigue + 0.35 * snapshot.avg_uncertainty, 0.15, 0.85)
        duration = np.clip(0.35 + 0.25 * capacity - 0.2 * fatigue, 0.2, 0.65)
        sleep = np.clip(snapshot.sleep_hours_last_3_nights / 9.0, 0.55, 0.95)
        return np.array([intensity, duration, rest, 0.78, sleep, np.clip(intensity * duration * 1.5, 0.05, 0.9)], dtype=np.float32)

    def run_safety_gate(self, snapshot: PatientSnapshot) -> SafetyGateResult:
        reasons: List[str] = []
        uncertainty_trigger = False
        drift_trigger = False
        rest_trigger = False
        if snapshot.avg_uncertainty > self.config.uncertainty_conservative_threshold:
            uncertainty_trigger = True
            reasons.append(f'Average latent uncertainty is {snapshot.avg_uncertainty:.2f}, above {self.config.uncertainty_conservative_threshold:.2f}; conservative mode is required.')
        drift = snapshot.drift_diagnosis or {}
        drift_action = drift.get('recommended_action', 'no_action')
        drift_severity = float(drift.get('severity', 0.0))
        if drift_action != 'no_action' or drift_severity > 0.05:
            drift_trigger = True
            reasons.append(f'Drift monitor requested {drift_action} at severity {drift_severity:.2f}; recommendations are paused until recalibration.')
        if snapshot.fatigue_score > self.config.critical_fatigue_score:
            rest_trigger = True
            reasons.append(f'Fatigue is {snapshot.fatigue_score:.0f}/100, above the critical {self.config.critical_fatigue_score:.0f}/100 threshold.')
        if snapshot.high_intensity_streak >= self.config.max_consecutive_high_days:
            rest_trigger = True
            reasons.append(f'{snapshot.high_intensity_streak} consecutive high-intensity days were detected; a deload/rest day is mandatory.')
        if drift_trigger:
            mode = 'paused'
            passed = False
        elif rest_trigger:
            mode = 'rest_day'
            passed = False
        elif uncertainty_trigger:
            mode = 'conservative'
            passed = False
        else:
            mode = 'normal'
            passed = True
        if not reasons and passed:
            reasons.append('No Stage 10 safety gate was triggered.')
        return SafetyGateResult(mode=mode, passed=passed, reasons=reasons)

    def enforce_action(self, raw_action: np.ndarray, snapshot: PatientSnapshot, gate: SafetyGateResult) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).copy()
        if gate.mode == 'paused':
            return np.array([0.0, 0.0, 1.0, 0.8, 0.9, 0.0], dtype=np.float32)
        if gate.mode == 'rest_day':
            action[:] = np.array([0.0, 0.0, 1.0, 0.8, 0.9, 0.0], dtype=np.float32)
        elif gate.mode == 'conservative':
            action[0] = min(action[0], 0.25)
            action[1] = min(action[1], 0.25)
            action[2] = max(action[2], 0.75)
            action[4] = max(action[4], 0.85)
            action[5] = min(action[5], 0.25)
        self.safety_guard.reset()
        self.safety_guard.consecutive_high_days = max(0, snapshot.high_intensity_streak - 1)
        return self.safety_guard.check_and_clip(action, snapshot.z_mu, snapshot.z_std)

    def action_to_plan(self, action: np.ndarray) -> Dict[str, Any]:
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        duration_days = int(np.clip(round(float(action[1]) * 28.0), 1, 28))
        return {'name': 'stage10_safe_policy', 'intensity': round(float(action[0]), 3), 'duration_days': duration_days, 'rest_extra_hours': round(float(action[2]) * 8.0, 2), 'nutrition_quality': round(float(action[3]), 3), 'sleep_consistency': round(float(action[4]), 3)}

    def explain(self, snapshot: PatientSnapshot) -> List[AttributionReason]:
        state = self.build_policy_state(snapshot)
        model = _DecisionScoreModel(self.actor, self.actor_state_dim).to(self.device).eval()
        baseline = torch.zeros_like(state)
        try:
            from captum.attr import IntegratedGradients
            attrs = IntegratedGradients(model).attribute(state, baselines=baseline, n_steps=self.config.attribution_steps)
        except Exception:
            attrs = self._manual_integrated_gradients(model, state, baseline)
        names = self.feature_names(len(snapshot.z_mu))
        values = attrs.squeeze(0).detach().cpu().numpy()
        order = np.argsort(np.abs(values))[::-1]
        preferred_order = [idx for idx in order if idx < len(names) and 'latent_' not in names[int(idx)] and (not names[int(idx)].startswith('padding_'))]
        fallback_order = [idx for idx in order if idx < len(names) and 'latent_' in names[int(idx)] and (not names[int(idx)].startswith('padding_'))]
        reasons: List[AttributionReason] = []
        for idx in preferred_order + fallback_order:
            if len(reasons) >= self.config.n_top_reasons:
                break
            feature = names[int(idx)] if idx < len(names) else f'feature_{idx}'
            if feature.startswith('padding_'):
                continue
            attribution = float(values[idx])
            direction = 'supports' if attribution >= 0 else 'reduces'
            reasons.append(AttributionReason(feature=feature, attribution=attribution, direction=direction, explanation=self._reason_sentence(feature, attribution, snapshot)))
        if not reasons:
            reasons.append(AttributionReason(feature='safety_gate', attribution=0.0, direction='supports', explanation='The deterministic safety gate dominated the final recommendation.'))
        return reasons

    def _manual_integrated_gradients(self, model: nn.Module, state: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        total_grad = torch.zeros_like(state)
        steps = max(2, self.config.attribution_steps)
        for alpha in torch.linspace(0.0, 1.0, steps, device=state.device):
            scaled = (baseline + alpha * (state - baseline)).detach().requires_grad_(True)
            score = model(scaled).sum()
            grad = torch.autograd.grad(score, scaled)[0]
            total_grad += grad
        avg_grad = total_grad / float(steps)
        return (state - baseline) * avg_grad

    def _reason_sentence(self, feature: str, attribution: float, snapshot: PatientSnapshot) -> str:
        polarity = 'pushed the system toward more activity' if attribution >= 0 else 'pushed the system toward caution'
        if feature == 'mu_fatigue':
            return f'Fatigue state was influential and {polarity}; current fatigue is {snapshot.fatigue_score:.0f}/100.'
        if feature == 'mu_capacity':
            return f'Capacity state was influential and {polarity}; current capacity is {snapshot.capacity_score:.0f}/100.'
        if feature == 'mu_recovery':
            return f'Recovery state was influential and {polarity}; HRV trend is {snapshot.hrv_7day_trend:+.2f}.'
        if feature == 'mu_cardio':
            return f'Cardio state was influential and {polarity}; the plan keeps cardio work controlled.'
        if feature == 'mu_stability':
            return f'Physiological stability was influential and {polarity}; the plan keeps progression gradual.'
        if feature.startswith('sigma_'):
            return f"Uncertainty in {feature.replace('sigma_', '')} affected confidence and {polarity}."
        if feature == 'sleep_hours_last_3_nights':
            return f'Recent sleep averaged {snapshot.sleep_hours_last_3_nights:.1f} hours and {polarity}.'
        if feature == 'weekly_training_load':
            return f'Weekly training load was {snapshot.weekly_training_load:.2f} on a 0-1 scale and {polarity}.'
        if feature == 'HRV_7day_trend':
            return f'HRV trend was {snapshot.hrv_7day_trend:+.2f} and {polarity}.'
        if feature == 'day_of_week_phase':
            return f'The current day-of-week context affected the plan and {polarity}.'
        return f'{feature} had a measurable contribution and {polarity}.'

    def recommend(self, snapshot: PatientSnapshot, ranked_scenarios: Optional[Sequence[ScenarioResult]]=None, n_weeks: int=12) -> RecommendationReport:
        raw_action = self.propose_action(snapshot)
        gate = self.run_safety_gate(snapshot)
        safe_action = self.enforce_action(raw_action, snapshot, gate)
        action_plan = self.action_to_plan(safe_action)
        reasons = self.explain(snapshot)
        selected = self._policy_scenario(ranked_scenarios, action_plan['name'])
        selected_plan_name = action_plan['name']
        selected_plan = dict(action_plan)
        ranked_dicts: List[Dict[str, Any]] = []
        if ranked_scenarios:
            ranked_dicts = [self._scenario_to_dict(r) for r in ranked_scenarios]
        if gate.mode in ('paused', 'rest_day', 'conservative'):
            selected_plan_name = action_plan['name']
            selected_plan = dict(action_plan)
        weekly_plan = self._weekly_plan(selected, selected_plan, n_weeks, gate.mode)
        expected = self._expected_outcome(snapshot, selected)
        injury_pct = float(selected.injury_prob * 100.0) if selected is not None else self._fallback_injury_pct(snapshot, safe_action)
        overtrain_pct = float(selected.overtraining_prob * 100.0) if selected is not None else self._fallback_overtraining_pct(snapshot, safe_action)
        injury_label = self._risk_label(injury_pct / 100.0)
        counterfactual = self._counterfactual(selected, ranked_scenarios)
        status = self._status_line(snapshot)
        plain = self._render_plain_language(snapshot=snapshot, gate=gate, selected_plan_name=selected_plan_name, selected_plan=selected_plan, safe_action=safe_action, reasons=reasons, expected_outcome=expected, injury_label=injury_label, injury_pct=injury_pct, overtraining_pct=overtrain_pct, counterfactual=counterfactual, weekly_plan=weekly_plan)
        return RecommendationReport(status=status, selected_plan_name=selected_plan_name, selected_plan=selected_plan, raw_action=[round(float(x), 4) for x in raw_action], safe_action=[round(float(x), 4) for x in safe_action], safety_gate=gate, top_reasons=reasons, counterfactual=counterfactual, expected_outcome=expected, injury_risk_label=injury_label, injury_risk_pct=injury_pct, overtraining_risk_pct=overtrain_pct, weekly_plan=weekly_plan, plain_language=plain, ranked_scenarios=ranked_dicts)

    def _select_scenario(self, ranked: Optional[Sequence[ScenarioResult]]) -> Optional[ScenarioResult]:
        if not ranked:
            return None
        safe = top_k_interventions(list(ranked), k=1)
        return safe[0] if safe else ranked[0]

    def _policy_scenario(self, ranked: Optional[Sequence[ScenarioResult]], policy_name: str) -> Optional[ScenarioResult]:
        if not ranked:
            return None
        for scenario in ranked:
            if scenario.name == policy_name:
                return scenario
        return None

    def _scenario_to_dict(self, scenario: ScenarioResult) -> Dict[str, Any]:
        data = asdict(scenario)
        data['overtraining_prob'] = round(float(data['overtraining_prob']), 6)
        data['injury_prob'] = round(float(data['injury_prob']), 6)
        data['max_risk'] = round(float(data['max_risk']), 6)
        data['fitness_score'] = round(float(data['fitness_score']), 6)
        return data

    def _weekly_plan(self, selected: Optional[ScenarioResult], selected_plan: Dict[str, Any], n_weeks: int, mode: str) -> List[Dict[str, Any]]:
        if mode == 'paused':
            return []
        if selected is not None:
            return build_periodized_plan(selected, n_weeks=n_weeks)
        pseudo = ScenarioResult(name=selected_plan.get('name', 'stage10_safe_policy'), plan={k: v for k, v in selected_plan.items() if k != 'name'}, overtraining_prob=0.0, injury_prob=0.0, peaking_day=0, avg_hr=0.0, avg_hrv_sdnn=0.0, max_risk=0.0, fitness_score=0.0)
        return build_periodized_plan(pseudo, n_weeks=n_weeks)

    def _status_line(self, snapshot: PatientSnapshot) -> str:
        confidence_label = self._confidence_label(snapshot.confidence_pct)
        return f'Physical capacity is {self._score_word(snapshot.capacity_score)} ({snapshot.capacity_score:.0f}/100), fatigue is {self._score_word(snapshot.fatigue_score)} ({snapshot.fatigue_score:.0f}/100), and confidence is {confidence_label} ({snapshot.confidence_pct:.0f}%).'

    def _expected_outcome(self, snapshot: PatientSnapshot, selected: Optional[ScenarioResult]) -> str:
        if selected is None:
            cap_next = np.clip(snapshot.capacity_score + 1.0 - snapshot.fatigue_score / 100.0, 0.0, 100.0)
            fatigue_next = np.clip(snapshot.fatigue_score - 3.0, 0.0, 100.0)
            return f'Expected tomorrow: capacity around {cap_next:.0f}/100 and fatigue around {fatigue_next:.0f}/100, assuming adherence and no new drift.'
        score_gain = min(8.0, max(0.5, selected.fitness_score * 10.0))
        cap_low = np.clip(snapshot.capacity_score + score_gain * 0.5, 0.0, 100.0)
        cap_high = np.clip(snapshot.capacity_score + score_gain, 0.0, 100.0)
        fatigue_delta = 6.0 if selected.plan.get('intensity', 0.5) < 0.35 else 2.0
        fat_low = np.clip(snapshot.fatigue_score - fatigue_delta, 0.0, 100.0)
        fat_high = np.clip(snapshot.fatigue_score - fatigue_delta * 0.4, 0.0, 100.0)
        return f'Expected tomorrow: capacity moves toward {cap_low:.0f}-{cap_high:.0f}/100, with fatigue around {fat_low:.0f}-{fat_high:.0f}/100.'

    def _counterfactual(self, selected: Optional[ScenarioResult], ranked: Optional[Sequence[ScenarioResult]]) -> str:
        if not ranked:
            return 'Counterfactual comparison unavailable because no Stage 6 simulations were supplied.'
        rest = next((r for r in ranked if r.name == 'recovery_week'), None)
        alternatives = [r for r in ranked if selected is None or r.name != selected.name]
        high_load = max(alternatives, key=lambda r: float(r.plan.get('intensity', 0.0)) * float(r.plan.get('duration_days', 1))) if alternatives else None
        selected_name = selected.name if selected is not None else 'the selected plan'
        parts = []
        if rest is not None and (selected is None or rest.name != selected.name):
            parts.append(f'If you rest instead, the recovery-week plan has {rest.injury_prob * 100:.1f}% injury risk and fitness score {rest.fitness_score:.3f}; it is safer but slower than {selected_name}.')
        if high_load is not None:
            parts.append(f"The highest-load rejected scenario was {high_load.name}: intensity {float(high_load.plan.get('intensity', 0.0)):.2f}, max risk {high_load.max_risk:.2f}, injury risk {high_load.injury_prob * 100:.1f}%.")
        return ' '.join(parts) if parts else 'The selected plan was also the lowest-risk simulated alternative.'

    def _render_plain_language(self, *, snapshot: PatientSnapshot, gate: SafetyGateResult, selected_plan_name: str, selected_plan: Dict[str, Any], safe_action: np.ndarray, reasons: Sequence[AttributionReason], expected_outcome: str, injury_label: str, injury_pct: float, overtraining_pct: float, counterfactual: str, weekly_plan: Sequence[Dict[str, Any]]) -> str:
        if gate.mode == 'paused':
            recommendation = 'RECOMMENDATION: Pause automated training changes today. Continue only baseline daily activity and wait for recalibration because the drift monitor invalidated the current twin.'
        elif gate.mode == 'rest_day':
            recommendation = 'RECOMMENDATION: Take a rest day. Use light mobility or flexibility only, add recovery time, and do not increase training load today.'
        elif gate.mode == 'conservative':
            recommendation = 'RECOMMENDATION: Use conservative mode: mild flexibility or easy walking only, no activity increase, and prioritize extra rest plus stable sleep.'
        else:
            intensity_pct = int(round(selected_plan.get('intensity', float(safe_action[0])) * 100))
            rest_hours = float(selected_plan.get('rest_extra_hours', float(safe_action[2]) * 8.0))
            recommendation = f'RECOMMENDATION: Use the {selected_plan_name} plan at about {intensity_pct}% normalized intensity, with {rest_hours:.1f} extra rest hours, cardio emphasis, supporting strength, and flexibility work.'
        top_reasons = '\n'.join((f'({idx}) {reason.explanation}' for idx, reason in enumerate(reasons, start=1)))
        safety_lines = '\n'.join((f'- {reason}' for reason in gate.reasons))
        first_weeks = '\n'.join((f"Week {w['week']}: intensity {w['intensity']:.2f}, rest +{w['rest_extra_hours']:.1f}h, sleep consistency {w['sleep_consistency']:.2f}" for w in weekly_plan[:4]))
        if not first_weeks:
            first_weeks = 'No 12-week plan generated while recommendations are paused.'
        return f'STATUS: {self._status_line(snapshot)}\n\nSAFETY GATE:\n{safety_lines}\n\n{recommendation}\n\nEXPECTED OUTCOME: {expected_outcome}\n\nTOP 3 REASONS:\n{top_reasons}\n\nINJURY RISK: {injury_label} ({injury_pct:.1f}%). OVERTRAINING RISK: {overtraining_pct:.1f}%.\n\nWHAT HAPPENS IF YOU CHOOSE AN ALTERNATIVE: {counterfactual}\n\n12-WEEK PLAN PREVIEW:\n{first_weeks}\n\nAUDIT NOTE: This is a research prototype recommendation assembled from deterministic Stage 10 rules, SAC policy output, simulation ranking, and integrated-gradient attribution.'

    def _risk_label(self, injury_prob: float) -> str:
        if injury_prob < self.config.low_injury_risk_threshold:
            return 'LOW'
        if injury_prob < self.config.moderate_injury_risk_threshold:
            return 'MODERATE'
        return 'HIGH'

    def _fallback_injury_pct(self, snapshot: PatientSnapshot, action: np.ndarray) -> float:
        return float(np.clip(100.0 * (0.03 + 0.12 * action[0] + 0.2 * snapshot.avg_uncertainty), 0.0, 100.0))

    def _fallback_overtraining_pct(self, snapshot: PatientSnapshot, action: np.ndarray) -> float:
        fatigue = snapshot.fatigue_score / 100.0
        load = float(action[0] * max(action[1], 0.1))
        return float(np.clip(100.0 * (0.02 + 0.3 * fatigue * load), 0.0, 100.0))

    def _score_word(self, score: float) -> str:
        if score >= 70.0:
            return 'good'
        if score >= 40.0:
            return 'moderate'
        return 'low'

    def _confidence_label(self, confidence_pct: float) -> str:
        if confidence_pct >= self.config.high_confidence_threshold:
            return 'HIGH'
        if confidence_pct >= self.config.medium_confidence_threshold:
            return 'MEDIUM'
        return 'LOW'

def scenario_names(ranked: Iterable[ScenarioResult]) -> List[str]:
    return [r.name for r in ranked]