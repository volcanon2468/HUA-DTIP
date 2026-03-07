import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf

from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.utils.metrics import mse, coverage_probability


def run_twin_eval(processed_dir: str, checkpoint_dir: str, device) -> dict:
    from evaluate.eval_twin import eval_reconstruction, eval_trajectory
    from train.train_twin import ZTemporalDataset, DaySequenceDataset
    from torch.utils.data import DataLoader

    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    for name, model in [("twin_vae", vae), ("twin_sde", sde)]:
        p = os.path.join(checkpoint_dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))

    feat_loader = DataLoader(ZTemporalDataset(processed_dir), batch_size=64, shuffle=False)
    day_loader  = DataLoader(DaySequenceDataset(processed_dir), batch_size=16, shuffle=False)

    print("=== Twin Suite ===")
    recon_mse = eval_reconstruction(vae, feat_loader, device)
    traj      = eval_trajectory(vae, sde, day_loader, device)

    vae_params = sum(p.numel() for p in vae.parameters())
    sde_params = sum(p.numel() for p in sde.parameters())

    return {
        "reconstruction_mse": float(recon_mse),
        "reconstruction_target": 0.05,
        "trajectory_mae": float(traj["trajectory_mae"]),
        "trajectory_target": 0.12,
        "uncertainty_calibration": float(traj["uncertainty_calibration"]),
        "calibration_target": 0.90,
        "vae_params": vae_params,
        "sde_params": sde_params,
        "latent_dim": 10,
    }


if __name__ == "__main__":
    data_cfg  = OmegaConf.load("configs/data.yaml")
    train_cfg = OmegaConf.load("configs/training.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_twin_eval(data_cfg.paths.processed, train_cfg.checkpoints.dir, device)
    print(json.dumps(results, indent=2))
