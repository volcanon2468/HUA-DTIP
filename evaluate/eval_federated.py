import os
import csv
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.federated.fedper_client import FedPerClient
from src.federated.clustering import SubjectClusterer
from src.utils.metrics import mae
from train.train_federated import FederatedModel, SubjectDataset, get_available_subjects, _collate_fn


def eval_personalization(processed_dir: str, checkpoint_dir: str,
                         device: torch.device, n_personal_epochs: int = 10) -> list:
    global_model = FederatedModel().to(device)
    p = os.path.join(checkpoint_dir, "federated_global.pt")
    if os.path.exists(p):
        global_model.load_state_dict(torch.load(p, map_location=device))

    available = get_available_subjects(processed_dir)
    results = []

    for sid in available:
        ds = SubjectDataset(processed_dir, sid)
        if len(ds) < 10:
            continue

        n_train = max(1, int(0.7 * len(ds)))
        train_paths = ds.paths[:n_train]
        test_paths  = ds.paths[n_train:]

        global_only_model = copy.deepcopy(global_model)
        global_only_model.eval()
        test_loader = DataLoader(
            SubjectDataset.__new__(SubjectDataset),
            batch_size=32, shuffle=False, collate_fn=_collate_fn,
        )
        test_ds = SubjectDataset.__new__(SubjectDataset)
        test_ds.paths = test_paths
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=_collate_fn)

        global_mae = _evaluate_model(global_only_model, test_loader, device)

        personal_model = copy.deepcopy(global_model)
        client = FedPerClient(personal_model, ["personal_head"], client_id=sid, lr=5e-4)
        train_ds = SubjectDataset.__new__(SubjectDataset)
        train_ds.paths = train_paths
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=_collate_fn)
        client.personalize(train_loader, nn.MSELoss(), device, n_epochs=n_personal_epochs)
        personal_mae = _evaluate_model(personal_model, test_loader, device)

        results.append({
            "subject_id": sid,
            "global_mae": global_mae,
            "personalized_mae": personal_mae,
            "improvement_pct": (global_mae - personal_mae) / (global_mae + 1e-8) * 100,
        })
        print(f"  Subject {sid:3d}  global={global_mae:.4f}  personal={personal_mae:.4f}  "
              f"Delta={results[-1]['improvement_pct']:.1f}%")

    return results


def eval_cold_start(processed_dir: str, checkpoint_dir: str,
                    device: torch.device, n_warmup_samples: int = 20) -> list:
    global_model = FederatedModel().to(device)
    p = os.path.join(checkpoint_dir, "federated_global.pt")
    if os.path.exists(p):
        global_model.load_state_dict(torch.load(p, map_location=device))

    available = get_available_subjects(processed_dir)
    results = []

    for sid in available:
        ds = SubjectDataset(processed_dir, sid)
        if len(ds) < n_warmup_samples + 10:
            continue

        warmup_ds = SubjectDataset.__new__(SubjectDataset)
        warmup_ds.paths = ds.paths[:n_warmup_samples]
        test_ds = SubjectDataset.__new__(SubjectDataset)
        test_ds.paths = ds.paths[n_warmup_samples:]

        personal_model = copy.deepcopy(global_model)
        client = FedPerClient(personal_model, ["personal_head"], client_id=sid, lr=1e-3)
        warmup_loader = DataLoader(warmup_ds, batch_size=min(16, n_warmup_samples),
                                   shuffle=True, collate_fn=_collate_fn)
        client.personalize(warmup_loader, nn.MSELoss(), device, n_epochs=5)

        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=_collate_fn)
        cold_mae = _evaluate_model(personal_model, test_loader, device)

        results.append({
            "subject_id": sid,
            "warmup_samples": n_warmup_samples,
            "cold_start_mae": cold_mae,
        })
        print(f"  Subject {sid:3d}  cold-start MAE ({n_warmup_samples} warmup): {cold_mae:.4f}")

    return results


def _evaluate_model(model, loader, device) -> float:
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            x = torch.nan_to_num(batch["features"].to(device))
            y = torch.nan_to_num(batch["hrv"].to(device))
            pred = model(x)
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())
    if not all_pred:
        return float("nan")
    return mae(torch.cat(all_pred), torch.cat(all_true))


def main():
    data_cfg  = OmegaConf.load("configs/data.yaml")
    train_cfg = OmegaConf.load("configs/training.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir  = data_cfg.paths.processed
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir    = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("=== Personalization Evaluation ===")
    personal_results = eval_personalization(processed_dir, checkpoint_dir, device)

    print("\n=== Cold-Start Evaluation (20 warmup samples) ===")
    cold_results = eval_cold_start(processed_dir, checkpoint_dir, device, n_warmup_samples=20)

    out_personal = os.path.join(results_dir, "federated_personalization.csv")
    with open(out_personal, "w", newline="") as f:
        if personal_results:
            w = csv.DictWriter(f, fieldnames=personal_results[0].keys())
            w.writeheader(); w.writerows(personal_results)

    out_cold = os.path.join(results_dir, "federated_cold_start.csv")
    with open(out_cold, "w", newline="") as f:
        if cold_results:
            w = csv.DictWriter(f, fieldnames=cold_results[0].keys())
            w.writeheader(); w.writerows(cold_results)

    if personal_results:
        avg_imp = np.mean([r["improvement_pct"] for r in personal_results])
        print(f"\n  Avg personalization improvement: {avg_imp:.1f}%  (target: >15%)")
    if cold_results:
        avg_cold = np.mean([r["cold_start_mae"] for r in cold_results])
        print(f"  Avg cold-start MAE (20 warmup): {avg_cold:.4f}  (target: <baseline)")

    out_summary = os.path.join(results_dir, "federated_summary.json")
    with open(out_summary, "w") as f:
        json.dump({
            "personalization": personal_results,
            "cold_start": cold_results,
        }, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
