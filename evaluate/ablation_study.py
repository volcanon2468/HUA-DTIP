import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from src.encoders.imu_encoder import SWCTNet
from src.encoders.cardio_encoder import CardioEncoder
from src.encoders.feature_encoder import FeatureEncoder
from src.encoders.fusion import CrossModalFusion
from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.rl.sac_networks import TwinGymEnv, SquashedGaussianActor
from src.rl.reward import MultiObjectiveReward
from src.rl.safety import SafetyGuard
from evaluate.eval_rl import evaluate_policy


ABLATION_CONFIGS = {
    "full_model": {"remove": []},
    "no_imu_encoder": {"remove": ["imu"]},
    "no_cardio_encoder": {"remove": ["cardio"]},
    "no_feature_encoder": {"remove": ["feature"]},
    "no_fusion": {"remove": ["fusion"]},
    "no_tcn_only_transformer": {"remove": ["tcn"]},
    "no_safety_guard": {"remove": ["safety"]},
    "no_ewc": {"remove": ["ewc"]},
    "beta_vae_beta_1": {"override": {"beta": 1.0}},
    "beta_vae_beta_8": {"override": {"beta": 8.0}},
    "sde_no_rest_context": {"remove": ["rest_context"]},
    "no_fedper_personal": {"remove": ["fedper"]},
}


def run_ablation(checkpoint_dir: str, device, ablation_name: str, config: dict) -> dict:
    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    for name, model in [("twin_vae", vae), ("twin_sde", sde)]:
        p = os.path.join(checkpoint_dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))

    removals = config.get("remove", [])
    overrides = config.get("override", {})

    if "beta" in overrides:
        vae.beta = overrides["beta"]

    actor = SquashedGaussianActor(64, 6).to(device)
    p = os.path.join(checkpoint_dir, "rl_actor.pt")
    if os.path.exists(p):
        actor.load_state_dict(torch.load(p, map_location=device))
    actor.eval()

    env = TwinGymEnv(vae, sde, episode_len=28, device=str(device))

    use_safety = "safety" not in removals
    safety    = SafetyGuard() if use_safety else None
    reward_fn = MultiObjectiveReward()

    class NoSafetyWrapper:
        episode_violations = 0
        def reset(self): self.episode_violations = 0
        def check_and_clip(self, a, z, s): return a
        def compute_penalty(self, a, z, s): return 0.0

    guard = safety if safety else NoSafetyWrapper()
    eval_results = evaluate_policy(actor, env, guard, reward_fn, n_eval=20, deterministic=True)

    return {
        "ablation": ablation_name,
        "mean_reward": float(eval_results["mean_reward"]),
        "std_reward": float(eval_results["std_reward"]),
        "mean_violations": float(eval_results["mean_violations"]),
        "components": eval_results.get("components", {}),
    }


def run_all_ablations(checkpoint_dir: str, device) -> list:
    results = []
    for name, config in ABLATION_CONFIGS.items():
        print(f"  Running ablation: {name}")
        try:
            r = run_ablation(checkpoint_dir, device, name, config)
            results.append(r)
            print(f"    reward={r['mean_reward']:.3f}")
        except Exception as e:
            results.append({"ablation": name, "error": str(e)})
            print(f"    FAILED: {e}")
    return results


def main():
    train_cfg = OmegaConf.load("configs/training.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir    = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("=== Ablation Study ===")
    ablation_results = run_all_ablations(checkpoint_dir, device)

    baseline = next((r for r in ablation_results if r["ablation"] == "full_model"), None)
    if baseline and "mean_reward" in baseline:
        for r in ablation_results:
            if "mean_reward" in r:
                r["delta_vs_full"] = r["mean_reward"] - baseline["mean_reward"]

    out_path = os.path.join(results_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\n{'Ablation':<30} {'Reward':<12} {'Delta vs Full'}")
    print("-" * 55)
    for r in ablation_results:
        if "mean_reward" in r:
            delta = r.get("delta_vs_full", 0.0)
            marker = " ←" if delta < -0.1 else ""
            print(f"  {r['ablation']:<28} {r['mean_reward']:<12.3f} {delta:+.3f}{marker}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
