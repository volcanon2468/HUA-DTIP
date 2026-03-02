# HUA-DTIP
**Holistic User Activity Digital Twin with Intervention Planning**

A multi-modal wearable health digital twin system combining hierarchical temporal modeling,
Bayesian state estimation (β-VAE), latent Neural SDE state evolution, SAC reinforcement learning,
adaptive drift detection, and privacy-preserving federated learning.

## Datasets
Place raw data under `data/raw/`:
- `mhealth/` — MHEALTH dataset `.log` files
- `pamap2/` — PAMAP2 dataset `.dat` files
- `ppg_4week/` — 4-Week PPG/HRV CSV files
- `stroke_rehab/` — Stroke rehabilitation IMU recordings
- `capture24/` — CAPTURE-24 per-subject CSVs
- `mex/` — MEx exercise recordings

## Quick Start
```bash
pip install -r requirements.txt

# Preprocess all datasets
python train/train_encoders.py

# Train in order: encoders → temporal → twin → rl → drift → federated
python train/train_temporal.py
python train/train_twin.py
python train/train_rl.py
python train/train_drift.py
python train/train_federated.py

# Final evaluation
python evaluate/compile_report.py
```

## Config
All hyperparameters live in `configs/`. Override from CLI:
```bash
python train/train_encoders.py training.lr=5e-4 model.imu_encoder.heads=4
```
