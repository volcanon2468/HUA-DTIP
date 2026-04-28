import os
import json
import csv
import torch
import numpy as np
from omegaconf import OmegaConf
from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.simulation.mc_rollout import MCRolloutEngine, InterventionPlan
from src.simulation.what_if import WhatIfEngine
from src.simulation.intervention_ranking import rank_interventions, top_k_interventions, print_ranking, build_periodized_plan

def eval_rollout_consistency(engine: MCRolloutEngine, z0_mean: torch.Tensor, z0_std: torch.Tensor, n_repeats: int=5) -> dict:
    plan = InterventionPlan(intensity=0.5, duration_days=14)
    hr_finals = []
    for _ in range(n_repeats):
        result = engine.rollout(z0_mean, z0_std, plan, n_days=14)
        hr_finals.append(result.hr_trajectory[-1])
    hr_finals = np.array(hr_finals)
    return {'consistency_std': float(hr_finals.std()), 'consistency_mean': float(hr_finals.mean()), 'consistency_cv': float(hr_finals.std() / (hr_finals.mean() + 1e-08))}

def eval_risk_calibration(engine: MCRolloutEngine, vae: BayesianVAE, z0: torch.Tensor) -> dict:
    with torch.no_grad():
        mu, logvar = vae.encoder(z0)
        std = torch.exp(0.5 * logvar)
    high_plan = InterventionPlan(intensity=0.95, duration_days=21)
    low_plan = InterventionPlan(intensity=0.2, duration_days=7, rest_extra_hours=3.0)
    hi_result = engine.rollout(mu.squeeze(0), std.squeeze(0), high_plan, n_days=21)
    low_result = engine.rollout(mu.squeeze(0), std.squeeze(0), low_plan, n_days=7)
    overtraining_high = hi_result.overtraining_prob
    overtraining_low = low_result.overtraining_prob
    return {'high_intensity_overtraining_prob': overtraining_high, 'low_intensity_overtraining_prob': overtraining_low, 'risk_discriminability': overtraining_high - overtraining_low}

def main():
    data_cfg = OmegaConf.load('configs/data.yaml')
    train_cfg = OmegaConf.load('configs/training.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)
    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    for name, model in [('twin_vae', vae), ('twin_sde', sde)]:
        p = os.path.join(checkpoint_dir, f'{name}.pt')
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
    vae.eval()
    sde.eval()
    mc_engine = MCRolloutEngine(vae, sde, n_samples=200, device=str(device))
    what_if = WhatIfEngine(mc_engine, vae)
    z0_feat = torch.randn(1, 48, device=device)
    z0_input = torch.cat([z0_feat, torch.zeros(1, 464, device=device)], dim=-1)
    with torch.no_grad():
        mu, logvar = vae.encoder(z0_input)
        std = torch.exp(0.5 * logvar)
    print('=== Rollout Consistency ===')
    consistency = eval_rollout_consistency(mc_engine, mu.squeeze(0), std.squeeze(0))
    for k, v in consistency.items():
        print(f'  {k}: {v:.4f}')
    print('\n=== Risk Calibration ===')
    risk_cal = eval_risk_calibration(mc_engine, vae, z0_input)
    for k, v in risk_cal.items():
        print(f'  {k}: {v:.4f}')
    print('\n=== Intervention Ranking ===')
    ranked = rank_interventions(what_if, z0_input)
    print_ranking(ranked)
    top3 = top_k_interventions(ranked, k=3)
    print('\n=== 12-Week Periodized Plan (Top-1) ===')
    weekly_plan = build_periodized_plan(top3[0], n_weeks=12)
    for block in weekly_plan:
        print(f"  Week {block['week']:2d}: intensity={block['intensity']:.2f}  rest+={block['rest_extra_hours']:.1f}h")
    out_consistency = os.path.join(results_dir, 'simulation_consistency.csv')
    with open(out_consistency, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k, v in {**consistency, **risk_cal}.items():
            w.writerow([k, f'{v:.6f}'])
    out_ranking = os.path.join(results_dir, 'intervention_rankings.json')
    with open(out_ranking, 'w') as f:
        json.dump([{'name': r.name, 'fitness_score': r.fitness_score, 'overtraining_prob': r.overtraining_prob, 'injury_prob': r.injury_prob, 'peaking_day': r.peaking_day} for r in ranked], f, indent=2)
    out_plan = os.path.join(results_dir, 'periodized_plan_top1.json')
    with open(out_plan, 'w') as f:
        json.dump({'top_intervention': top3[0].name, 'weekly_blocks': weekly_plan}, f, indent=2)
    print(f'\nAll simulation results saved to {results_dir}')
if __name__ == '__main__':
    main()