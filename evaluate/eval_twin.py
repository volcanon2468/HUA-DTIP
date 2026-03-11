import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.utils.metrics import mse, coverage_probability
from train.train_twin import ZTemporalDataset, DaySequenceDataset


def eval_reconstruction(vae: BayesianVAE, loader: DataLoader, device,
                        feat_mean=None, feat_std=None) -> float:
    """Measure reconstruction MSE in normalized feature space."""
    vae.eval()
    total_mse = 0.0; n = 0
    with torch.no_grad():
        for features, hrv, hr in loader:
            features = torch.nan_to_num(features.to(device), nan=0.0)
            if feat_mean is not None:
                features = (features - feat_mean.to(device)) / feat_std.to(device)
            z_input = torch.cat([features, torch.zeros(features.shape[0], 512 - 48, device=device)], dim=-1)
            z, mu, logvar, recon, _ = vae(z_input)
            target = z_input[:, :48]
            total_mse += torch.mean((recon - target) ** 2).item() * features.shape[0]
            n += features.shape[0]
    avg = total_mse / max(n, 1)
    print(f"  State reconstruction MSE: {avg:.6f}  (target: <0.05)")
    return avg


def eval_trajectory(vae: BayesianVAE, sde: LatentNeuralSDE,
                    day_loader: DataLoader, device, n_days: int = 7) -> dict:
    vae.eval(); sde.eval()
    all_mae, all_coverage = [], []

    with torch.no_grad():
        for x_seq, y_next in day_loader:
            x_seq  = torch.nan_to_num(x_seq.to(device))
            y_next = torch.nan_to_num(y_next.to(device))

            z_input = torch.cat([x_seq[:, -1, :48],
                                 torch.zeros(x_seq.shape[0], 512 - 48, device=device)], dim=-1)
            mu0, _ = vae.encoder(z_input)

            activity = torch.zeros(mu0.shape[0], 6, device=device)
            rest     = torch.zeros(mu0.shape[0], 3, device=device)

            z_mean, z_std = sde.predict_trajectory(mu0, activity, rest, n_days=1, n_samples=50)

            y_inp   = torch.cat([y_next[:, :48], torch.zeros(y_next.shape[0], 512 - 48, device=device)], dim=-1)
            mu_true, _ = vae.encoder(y_inp)

            err = torch.abs(z_mean[-1] - mu_true).mean(dim=-1)
            all_mae.append(err.cpu().numpy())

            cov = coverage_probability(
                z_mean[-1].cpu().numpy(),
                z_std[-1].cpu().numpy(),
                mu_true.cpu().numpy(),
                z=2.1,
            )
            all_coverage.append(cov)

    mae_val  = float(np.concatenate(all_mae).mean()) if all_mae else float("nan")
    cov_val  = float(np.array(all_coverage).mean()) if all_coverage else float("nan")
    print(f"  7-day trajectory MAE: {mae_val:.4f}  (target: <0.12)")
    print(f"  Uncertainty calibration (95% coverage): {cov_val:.4f}")
    return {"trajectory_mae": mae_val, "uncertainty_calibration": cov_val}


def main():
    data_cfg  = OmegaConf.load("configs/data.yaml")
    train_cfg = OmegaConf.load("configs/training.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir  = data_cfg.paths.processed
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir    = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)

    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    for name, model in [("twin_vae", vae), ("twin_sde", sde)]:
        p = os.path.join(checkpoint_dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))

    # Load feature normalization stats
    feat_mean, feat_std = None, None
    norm_path = os.path.join(checkpoint_dir, "feature_norm_stats.pt")
    if os.path.exists(norm_path):
        stats = torch.load(norm_path, map_location=device)
        feat_mean = stats["mean"]
        feat_std = stats["std"]
        print("  Loaded feature normalization stats.")

    feat_loader = DataLoader(ZTemporalDataset(processed_dir), batch_size=64, shuffle=False)
    day_loader  = DataLoader(DaySequenceDataset(processed_dir), batch_size=16, shuffle=False)

    print("=== Twin State Reconstruction ===")
    recon_mse = eval_reconstruction(vae, feat_loader, device, feat_mean=feat_mean, feat_std=feat_std)

    print("\n=== Trajectory Prediction ===")
    traj_metrics = eval_trajectory(vae, sde, day_loader, device)

    out_recon = os.path.join(results_dir, "twin_state_reconstruction.csv")
    with open(out_recon, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "target"])
        w.writerow(["reconstruction_mse", f"{recon_mse:.6f}", "<0.05"])

    out_traj = os.path.join(results_dir, "state_evolution_trajectory.csv")
    with open(out_traj, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "target"])
        w.writerow(["trajectory_mae", f"{traj_metrics['trajectory_mae']:.4f}", "<0.12"])
        w.writerow(["uncertainty_calibration", f"{traj_metrics['uncertainty_calibration']:.4f}", ">0.90"])

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
