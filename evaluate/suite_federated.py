import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf

from evaluate.eval_federated import eval_personalization, eval_cold_start


def run_federated_eval(processed_dir: str, checkpoint_dir: str, device) -> dict:
    print("=== Federated Suite ===")
    personal = eval_personalization(processed_dir, checkpoint_dir, device)
    cold     = eval_cold_start(processed_dir, checkpoint_dir, device, n_warmup_samples=20)

    avg_improvement = np.mean([r["improvement_pct"] for r in personal]) if personal else 0.0
    avg_cold_mae    = np.mean([r["cold_start_mae"] for r in cold]) if cold else float("nan")
    n_subjects      = len(personal)

    return {
        "n_subjects_evaluated": n_subjects,
        "avg_personalization_improvement_pct": float(avg_improvement),
        "improvement_target_pct": 15.0,
        "avg_cold_start_mae": float(avg_cold_mae),
        "cold_start_warmup_samples": 20,
        "per_subject_personalization": personal,
        "per_subject_cold_start": cold,
    }


if __name__ == "__main__":
    data_cfg  = OmegaConf.load("configs/data.yaml")
    train_cfg = OmegaConf.load("configs/training.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_federated_eval(data_cfg.paths.processed, train_cfg.checkpoints.dir, device)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("per_subject_personalization", "per_subject_cold_start")}, indent=2))
