# HUA-DTIP: Perception-Grounded Digital Twin (Phase 1)

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg">
</div>

<br/>

**HUA-DTIP (Phase 1)** builds a continuous-time digital twin of human physiological state from multi-modal wearable sensor data. It combines modality-specific representation learning, hierarchical temporal modeling, and generative latent-state modeling to produce a current latent health state and a forecasted trajectory with uncertainty, decodable back into interpretable HR/HRV predictions.

---

## Key Features & Capabilities

- **Digital Twin VAE & Neural SDE**: A Bayesian β-VAE (β = 0.1, latent dim = **10**) encodes the **512-dimensional hierarchical temporal representation** into a probabilistic latent space. A Latent Neural SDE (latent dim = 10) models continuous forward evolution of these states.
- **Multi-Modal Fusion**: Fuses IMU and cardio (PPG/ECG) signals with a 48-D handcrafted tabular feature vector using specialised encoders (`SWCTNet`, `CardioEncoder`, `FeatureEncoder`) combined via `CrossModalFusion`, trained end-to-end via downstream task losses (HAR classification + HR regression).
- **Hierarchical Temporal Modeling**: Captures short-term (`MicroScaleModel`), medium-term (`MesoScaleModel`, 7-day windows), and long-term (`MacroScaleModel`) dependencies, fused via `HierarchicalFusion` into the final 512-D representation.

---

## System Architecture (Phase 1 Pillars)

1. **Multi-Modal Encoders (`src/encoders/`)**
   Raw sensors → modality-specific embeddings via `SWCTNet` (IMU), `CardioEncoder` (cardio signal, 2-channel input), `FeatureEncoder` (48-D handcrafted feature vector). `CrossModalFusion` combines all three representations. Fusion is trained end-to-end using downstream HAR and HR losses, not a standalone reconstruction objective.

2. **Hierarchical Temporal Modeling (`src/temporal/`)**
   `MicroScaleModel`, `MesoScaleModel`, and `MacroScaleModel` operate at short-, medium-, and long-term time scales respectively. `HierarchicalFusion` combines all three into the final **512-D** temporal representation, which is the input to the digital twin.

3. **Neural SDE Twin State Evolution (`src/twin/`)**
   `BayesianVAE` (input dim = **512**, latent dim = **10**, β = 0.1) encodes the `HierarchicalFusion` output into a probabilistic 10-D latent space. `LatentNeuralSDE` (latent dim = 10) models continuous forward evolution of this latent state.

---

## Project Structure

```text
HUA-DTIP/
├── configs/             # YAML configurations (model, training, data)
├── data/                # Data directory (see Datasets below)
│   ├── raw/             # Raw dataset files
│   └── processed/       # Cached preprocessed artifacts (windows, daily summaries)
├── evaluate/            # Evaluation suites for encoders and digital twin
├── src/
│   ├── encoders/        # Multi-modal embedding networks
│   ├── preprocessing/   # Data loaders, windowing, feature extraction, HRV
│   ├── temporal/        # Hierarchical time-series modeling
│   ├── twin/            # Bayesian VAE & Latent Neural SDE
│   └── utils/           # Metrics, logging, seeding
├── train/               # Training entrypoints
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

---

## Installation & Prerequisites

**Prerequisites:**
- Python 3.11+
- CUDA-enabled GPU (recommended for SDE and Transformer training)

**Setup Instructions:**

1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/your-org/HUA-DTIP.git
   cd HUA-DTIP
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Datasets

Place raw files into the corresponding subdirectories under `data/raw/`, matching the paths configured in `configs/data.yaml`:

- **MHEALTH** (`mhealth/`): Mobile Health dataset `.log` files.
- **PAMAP2** (`pamap2/`): Physical Activity Monitoring `.dat` files.
- **4-Week PPG/HRV** (`ppg_4week/`): Longitudinal PPG CSV files.

Data is loaded and preprocessed via `src/preprocessing/run_preprocessing.py`, which handles windowing (window = 1000 samples, stride = 500), per-subject z-score normalisation, and window label assignment by majority vote. Only the 4-Week PPG dataset currently produces daily summary artifacts consumed by the meso/macro-scale temporal training stage.

---

## Getting Started (Training Pipeline)

Run stages in order — each depends on checkpoints saved by the previous stage:

1. **Preprocess raw sensor data**
   ```bash
   python -m src.preprocessing.run_preprocessing
   ```

2. **Pretrain Multi-Modal Encoders** (IMU contrastive pretraining + fine-tuning, Cardio HR regression, Feature+Fusion task-supervised training)
   ```bash
   python train/train_encoders.py
   ```

3. **Train Hierarchical Temporal Models** (Micro/Meso/Macro + HierarchicalFusion → 512-D representation)
   ```bash
   python train/train_temporal.py
   ```

4. **Train the Digital Twin** (Bayesian VAE on the 512-D representation → 10-D latent, then Latent Neural SDE, then joint fine-tuning)
   ```bash
   python train/train_twin.py
   ```

---

## Configuration Management

Hyperparameters, dataset paths, and network dimensions are managed via YAML configs in `configs/`:

- `data.yaml`: Window sizes, normalisation methods, dataset paths.
- `training.yaml`: Learning rates, contrastive temperatures, epochs, early stopping, VAE beta.

**Overriding from CLI:**
```bash
python train/train_encoders.py training.encoders.lr=5e-4
```

---

## Evaluation & Reporting

```bash
# Evaluate encoder quality (LOSO activity F1, HR MAE)
python evaluate/suite_encoders.py

# Evaluate Digital Twin reconstruction and trajectory forecasting
python evaluate/suite_twin.py
```

**Target metrics (embedded in the evaluation scripts):**

| Metric | Target |
|---|---|
| Activity classification F1 (LOSO) | > 0.85 |
| Heart rate MAE | < 5 bpm |
| Twin state reconstruction MSE (on the 512-D representation) | < 0.05 |
| 7-day trajectory prediction MAE | < 0.12 |
| Uncertainty calibration (95% coverage) | > 0.90 |