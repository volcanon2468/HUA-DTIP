import os
import json
import time
import torch
from omegaconf import OmegaConf

from evaluate.suite_encoders import run_encoder_eval
from evaluate.suite_twin import run_twin_eval
from evaluate.suite_rl import run_rl_eval
from evaluate.suite_federated import run_federated_eval
from evaluate.ablation_study import run_all_ablations


TARGET_METRICS = {
    "imu_loso_f1":                   {"target": 0.75,  "direction": "higher_better"},
    "hr_mae_bpm":                    {"target": 35.0,  "direction": "lower_better"},
    "reconstruction_mse":            {"target": 0.05,  "direction": "lower_better"},
    "trajectory_mae":                {"target": 0.12,  "direction": "lower_better"},
    "uncertainty_calibration":       {"target": 0.90,  "direction": "higher_better"},
    "sac_mean_reward":               {"target": -20.0,  "direction": "higher_better"},
    "improvement_over_random":       {"target": -5.0,   "direction": "higher_better"},
    "avg_personalization_improvement_pct": {"target": 15.0, "direction": "higher_better"},
}


def _check_target(metric_name: str, value: float) -> str:
    if metric_name not in TARGET_METRICS:
        return "no_target"
    t = TARGET_METRICS[metric_name]
    if t["direction"] == "higher_better":
        return "PASS" if value >= t["target"] else "FAIL"
    else:
        return "PASS" if value <= t["target"] else "FAIL"


def generate_report() -> dict:
    data_cfg  = OmegaConf.load("configs/data.yaml")
    train_cfg = OmegaConf.load("configs/training.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir  = data_cfg.paths.processed
    checkpoint_dir = train_cfg.checkpoints.dir
    results_dir    = train_cfg.checkpoints.results_dir
    os.makedirs(results_dir, exist_ok=True)

    report = {
        "project": "HUA-DTIP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }

    print("\n" + "=" * 60)
    print("  HUA-DTIP — Final Evaluation Report")
    print("=" * 60)

    print("\n[1/5] Encoder Evaluation...")
    try:
        enc = run_encoder_eval(processed_dir, checkpoint_dir, data_cfg, device)
        report["encoders"] = {k: v for k, v in enc.items() if k != "per_subject_f1"}
    except Exception as e:
        report["encoders"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n[2/5] Twin Evaluation...")
    try:
        report["twin"] = run_twin_eval(processed_dir, checkpoint_dir, device)
    except Exception as e:
        report["twin"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n[3/5] RL Evaluation...")
    try:
        rl = run_rl_eval(checkpoint_dir, device)
        report["rl"] = {k: v for k, v in rl.items() if k != "comparison"}
    except Exception as e:
        report["rl"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n[4/5] Federated Evaluation...")
    try:
        fed = run_federated_eval(processed_dir, checkpoint_dir, device)
        report["federated"] = {k: v for k, v in fed.items()
                               if k not in ("per_subject_personalization", "per_subject_cold_start")}
    except Exception as e:
        report["federated"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n[5/5] Ablation Study...")
    try:
        report["ablation"] = run_all_ablations(checkpoint_dir, device)
    except Exception as e:
        report["ablation"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    scorecard = {}
    all_metrics = {}
    for section_name, section in report.items():
        if isinstance(section, dict) and "error" not in section:
            for k, v in section.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = v

    for metric, value in all_metrics.items():
        result = _check_target(metric, value)
        if result != "no_target":
            scorecard[metric] = {
                "value": value,
                "target": TARGET_METRICS.get(metric, {}).get("target"),
                "result": result,
            }

    report["scorecard"] = scorecard
    n_pass = sum(1 for v in scorecard.values() if v["result"] == "PASS")
    n_total = len(scorecard)
    report["overall_pass_rate"] = f"{n_pass}/{n_total}" if n_total > 0 else "N/A"

    print("\n" + "=" * 60)
    print("  SCORECARD")
    print("=" * 60)
    for metric, info in scorecard.items():
        status = "PASS" if info["result"] == "PASS" else "FAIL"
        print(f"  [{status}] {metric}: {info['value']:.4f}  (target: {info['target']})")
    print(f"\n  Overall: {report['overall_pass_rate']}")

    out_path = os.path.join(results_dir, "final_evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to {out_path}")

    return report


if __name__ == "__main__":
    generate_report()
