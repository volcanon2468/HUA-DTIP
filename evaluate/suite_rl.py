import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.rl.sac_networks import TwinGymEnv, SquashedGaussianActor
from src.rl.reward import MultiObjectiveReward
from src.rl.safety import SafetyGuard
from evaluate.eval_rl import evaluate_policy, compare_with_baseline

def run_rl_eval(checkpoint_dir: str, device) -> dict:
    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    for name, model in [('twin_vae', vae), ('twin_sde', sde)]:
        p = os.path.join(checkpoint_dir, f'{name}.pt')
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
    actor = SquashedGaussianActor(64, 6).to(device)
    p = os.path.join(checkpoint_dir, 'rl_actor.pt')
    if os.path.exists(p):
        actor.load_state_dict(torch.load(p, map_location=device))
    actor.eval()
    env = TwinGymEnv(vae, sde, episode_len=28, device=str(device))
    safety = SafetyGuard()
    reward_fn = MultiObjectiveReward()
    print('=== RL Suite ===')
    eval_results = evaluate_policy(actor, env, safety, reward_fn, n_eval=50)
    comparison = compare_with_baseline(actor, env, safety, reward_fn, n_eval=30)
    sac_reward = comparison.get('sac_policy', {}).get('mean_reward', 0)
    random_reward = comparison.get('random_policy', {}).get('mean_reward', 0)
    return {'sac_mean_reward': float(eval_results['mean_reward']), 'sac_std_reward': float(eval_results['std_reward']), 'sac_mean_violations': float(eval_results['mean_violations']), 'random_mean_reward': float(random_reward), 'improvement_over_random': float(sac_reward - random_reward), 'reward_components': eval_results.get('components', {}), 'comparison': comparison, 'actor_params': sum((p.numel() for p in actor.parameters()))}
if __name__ == '__main__':
    train_cfg = OmegaConf.load('configs/training.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_rl_eval(train_cfg.checkpoints.dir, device)
    print(json.dumps({k: v for k, v in results.items() if k != 'comparison'}, indent=2))