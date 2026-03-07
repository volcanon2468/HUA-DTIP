import os
import csv
import json
import numpy as np
import torch
from omegaconf import OmegaConf

from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.rl.sac_networks import TwinGymEnv, SquashedGaussianActor
from src.rl.reward import MultiObjectiveReward
from src.rl.safety import SafetyGuard


def evaluate_policy(actor, env, safety, reward_fn, n_eval: int = 50, deterministic: bool = True):
    device = next(actor.parameters()).device
    ep_rewards, ep_violations, ep_lengths = [], [], []
    reward_components = {"progress": [], "safety": [], "recovery": [], "adherence": []}

    for _ in range(n_eval):
        state, _ = env.reset()
        safety.reset()
        ep_reward = 0.0

        for step in range(env.episode_len):
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic:
                    action_np = actor.deterministic(s_t).squeeze(0).cpu().numpy()
                else:
                    action_np, _ = actor.sample(s_t)
                    action_np = action_np.squeeze(0).cpu().numpy()

            z_mu  = state[:env.latent_dim]
            z_std = state[env.latent_dim:]
            action_np = safety.check_and_clip(action_np, z_mu, z_std)

            next_state, _, terminated, truncated, info = env.step(action_np)
            shaped = reward_fn.compute(info["z_mu"].squeeze(), info["z_std"].squeeze(), action_np)
            decomposed = reward_fn.decompose(info["z_mu"].squeeze(), info["z_std"].squeeze(), action_np)
            for k, v in decomposed.items():
                reward_components[k].append(v)

            ep_reward += shaped
            state = next_state
            if terminated or truncated:
                break

        ep_rewards.append(ep_reward)
        ep_violations.append(safety.episode_violations)
        ep_lengths.append(step + 1)

    return {
        "mean_reward":     float(np.mean(ep_rewards)),
        "std_reward":      float(np.std(ep_rewards)),
        "mean_violations": float(np.mean(ep_violations)),
        "max_violations":  int(np.max(ep_violations)),
        "mean_length":     float(np.mean(ep_lengths)),
        "components": {k: float(np.mean(v)) for k, v in reward_components.items()},
    }


def compare_with_baseline(actor, env, safety, reward_fn, n_eval: int = 50):
    device = next(actor.parameters()).device
    results = {}

    results["sac_policy"] = evaluate_policy(actor, env, safety, reward_fn, n_eval, deterministic=True)

    class RandomActor:
        def deterministic(self, s): return torch.rand(1, 6)
        def parameters(self): return iter([torch.zeros(1)])
    results["random_policy"] = evaluate_policy(
        RandomActor(), env, SafetyGuard(), reward_fn, n_eval, deterministic=True
    )

    class FixedActor:
        def deterministic(self, s): return torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.7, 0.8]])
        def parameters(self): return iter([torch.zeros(1)])
    results["fixed_moderate"] = evaluate_policy(
        FixedActor(), env, SafetyGuard(), reward_fn, n_eval, deterministic=True
    )

    return results


def main():
    train_cfg = OmegaConf.load("configs/training.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir    = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)

    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    for name, model in [("twin_vae", vae), ("twin_sde", sde)]:
        p = os.path.join(checkpoint_dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))

    state_dim  = 20
    action_dim = 6
    actor = SquashedGaussianActor(state_dim, action_dim).to(device)
    p = os.path.join(checkpoint_dir, "rl_actor.pt")
    if os.path.exists(p):
        actor.load_state_dict(torch.load(p, map_location=device))
    actor.eval()

    env       = TwinGymEnv(vae, sde, episode_len=28, device=str(device))
    safety    = SafetyGuard()
    reward_fn = MultiObjectiveReward()

    print("=== Policy Evaluation (50 episodes) ===")
    eval_results = evaluate_policy(actor, env, safety, reward_fn, n_eval=50)
    print(f"  Mean reward:     {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"  Mean violations: {eval_results['mean_violations']:.1f}")
    print(f"  Components:      {eval_results['components']}")

    print("\n=== Baseline Comparison ===")
    comparison = compare_with_baseline(actor, env, safety, reward_fn, n_eval=50)
    for policy_name, metrics in comparison.items():
        print(f"  {policy_name:<20} reward={metrics['mean_reward']:.3f}  violations={metrics['mean_violations']:.1f}")

    out_eval = os.path.join(results_dir, "rl_policy_evaluation.json")
    with open(out_eval, "w") as f:
        json.dump(eval_results, f, indent=2)

    out_compare = os.path.join(results_dir, "rl_baseline_comparison.json")
    with open(out_compare, "w") as f:
        json.dump(comparison, f, indent=2)

    out_csv = os.path.join(results_dir, "rl_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "mean_reward", "std_reward", "mean_violations", "target"])
        for name, m in comparison.items():
            w.writerow([name, f"{m['mean_reward']:.4f}", f"{m['std_reward']:.4f}",
                        f"{m['mean_violations']:.1f}", ">random"])

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
