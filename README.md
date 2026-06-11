# HUA-DTIP: Holistic User Activity Digital Twin with Intervention Planning

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
</div>

<br/>

**HUA-DTIP** is an advanced machine learning framework for generating predictive, continuous-time digital twins of human health and activity from multi-modal wearable sensor data. By combining representation learning, generative modeling, and reinforcement learning, the system can monitor health state trajectories, predict future states, detect data drift, train securely across decentralized clients, and plan optimal, safe clinical interventions.

---

## ✨ Key Features & Capabilities

- 🧬 **Digital Twin VAE & Neural SDEs**: Uses Bayesian β-VAEs coupled with Latent Neural Stochastic Differential Equations (SDEs) to model the continuous-time evolution of a patient's health state, smoothly handling irregular sampling.
- 🔗 **Multi-Modal Fusion**: Fuses 1D signals (IMU, ECG, PPG) with tabular features using Transformer-based Cross-Modal Fusion and specialized encoders (e.g., SWCTNet).
- 🕒 **Hierarchical Temporal Modeling**: Captures short-term (micro-scale TCNs), medium-term (meso-scale Self-Attention), and long-term (macro-scale N-BEATS) temporal dependencies.
- 🤖 **Reinforcement Learning Intervention Planning**: Employs Soft Actor-Critic (SAC) networks with rigorous safety guardrails to propose optimal activity and therapy interventions.
- 🌍 **Privacy-Preserving Federated Learning**: Trains securely across decentralized patient cohorts using FL strategies like FedProx and FedPer (Personalization layers).
- 🚨 **Concept Drift & Novelty Detection**: Adapts online to changing physiological distributions using MMD (Maximum Mean Discrepancy), ADWIN, and Novelty Autoencoders.
- 🏥 **Clinical Recommendation Engine**: Translates complex latent trajectories into plain-language clinical recommendations, complete with counterfactual analysis and uncertainty-bounded confidence scores.

---

## 🏗️ System Architecture (The 7 Pillars)

The HUA-DTIP pipeline is organized into 7 highly cohesive subsystems:

1. **Multi-Modal Encoders (`src/encoders/`)**
   Processes raw sensors into latent embeddings. Uses `CardioEncoder` (ResBlock1D + Self-Attention) for ECG/PPG, `SWCTNet` for sliding-window IMU modeling, and `FeatureEncoder` for discrete signals, fused via a Cross-Modal Transformer.
2. **Hierarchical Temporal Modeling (`src/temporal/`)**
   Fuses representations across varying time scales. `MicroScaleModel` uses Dilated TCNs for high-frequency dynamics; `MesoScaleModel` applies attention over rolling windows; `MacroScaleModel` forecasts long-range trends.
3. **Neural SDE Twin State Evolution (`src/twin/`)**
   The core digital twin. A `BayesianVAE` encodes historical trajectories into a probabilistic latent space, while a `LatentNeuralSDE` models the continuous forward evolution of these states under different conditions.
4. **Reinforcement Learning & Policy Network (`src/rl/`)**
   Models clinical interventions as an MDP. A `TwinGymEnv` simulates patient responses, while `SquashedGaussianActor` and `TwinCritic` (SAC) optimize multi-objective rewards (health gain vs. injury risk), bounded by a `SafetyGuard`.
5. **Federated Learning (`src/federated/`)**
   Enables training across 200+ distinct subjects locally without centralizing raw data. Uses `SubjectClusterer` for cohort grouping, `FedPerClient` for local personalization, and `FedProxServer` for proximal aggregation.
6. **Concept Drift (`src/drift/`)**
   Monitors incoming streams for statistical shifts. Includes `ADWINDetector` for mean shifts, `AutoencoderNoveltyDetector` for structural anomalies, and `MMDDetector` for distribution discrepancies, guided by an EWCR-based `DriftManager`.
7. **Recommendation Engine (`src/recommendation/`)**
   The user-facing interface. Evaluates counterfactual plans using Monte Carlo rollouts (`MC_RolloutEngine`), scores risk, and generates actionable, explainable `RecommendationReport` objects for clinicians.

---

## 📂 Project Structure

```text
HUA-DTIP/
├── configs/             # YAML configurations (model, training, data)
├── data/                # Data directory (see Datasets below)
│   ├── raw/             # Raw dataset files
│   └── processed/       # Cached preprocessed artifacts
├── evaluate/            # Evaluation suites and ablation studies
├── src/                 # Core architecture source code
│   ├── drift/           # Concept drift & novelty detection
│   ├── encoders/        # Multi-modal embedding networks
│   ├── federated/       # Federated learning strategies
│   ├── preprocessing/   # Data loaders, signal cleaning, HRV extraction
│   ├── recommendation/  # Clinical decision engine
│   ├── rl/              # Reinforcement learning env & agents
│   ├── simulation/      # Twin simulation & what-if analysis
│   ├── temporal/        # Hierarchical time-series modeling
│   ├── twin/            # Generative VAE & Latent SDEs
│   └── utils/           # Metrics, logging (WandB), configuration
├── train/               # Main entrypoints for the training pipeline
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

---

## 🚀 Installation & Prerequisites

**Prerequisites:**
- Python 3.11+
- CUDA-enabled GPU (Highly recommended for SDE and Transformer training)

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
   *Core libraries include `torch`, `torchsde`, `stable-baselines3`, `captum`, `flwr`, and `river`.*

---

## 📊 Datasets

The system is evaluated against multiple robust open-source health and activity datasets. To run the full pipeline, place the raw dataset files into their corresponding subdirectories under `data/raw/`:

- **MHEALTH** (`mhealth/`): Mobile Health dataset `.log` files.
- **PAMAP2** (`pamap2/`): Physical Activity Monitoring `.dat` files.
- **4-Week PPG/HRV** (`ppg_4week/`): Longitudinal PPG CSV files.
- **Stroke Rehabilitation** (`stroke_rehab/`): Clinical IMU recordings.
- **CAPTURE-24** (`capture24/`): Free-living activity per-subject CSVs.
- **MEx** (`mex/`): Exercise form recordings.

Data is loaded via `src/preprocessing/dataset_loaders.py`, which handles missing data imputation, bandpass filtering, windowing, and per-subject z-score normalization.

---

## 🏃 Getting Started (Training Pipeline)

The system relies on a sequential training pipeline where each downstream task relies on the representations learned by the previous step. Run them in the following order:

1. **Pretrain Multi-Modal Encoders** (Representation Learning)
   ```bash
   python train/train_encoders.py
   ```
2. **Train Hierarchical Temporal Models** (Time-series fusion)
   ```bash
   python train/train_temporal.py
   ```
3. **Train the Digital Twin** (Bayesian VAE & Latent Neural SDE)
   ```bash
   python train/train_twin.py
   ```
4. **Train the RL Policy** (SAC Intervention Planning)
   ```bash
   python train/train_rl.py
   ```
5. **Train Federated Learning Agents** (Privacy-Preserving Aggregation)
   ```bash
   python train/train_federated.py
   ```

*(Drift components are evaluated iteratively or during online rollout simulations via `evaluate/` scripts).*

---

## ⚙️ Configuration Management

All hyperparameters, dataset paths, and neural network dimensions are managed via YAML configurations located in the `configs/` directory:
- `data.yaml`: Window sizes, normalization methods, HRV feature lists, and dataset specifics.
- `model.yaml`: Transformer heads, SDE drift dimensions, VAE latent sizes, etc.
- `training.yaml`: Learning rates, contrastive temperatures, epochs, and early stopping.

**Overriding from CLI:**
You can seamlessly override any parameter from the command line:
```bash
python train/train_encoders.py training.encoders.lr=5e-4 model.imu_encoder.transformer_heads=4
```

---

## 🔬 Evaluation & Reporting

To validate the model, run the dedicated evaluation suites located in `evaluate/`. These scripts generate metrics like RMSE, Pearson R, Coverage, and F1 scores, logging directly to the console and WandB.

```bash
# Evaluate representation quality and downstream linear-probe
python evaluate/suite_encoders.py

# Evaluate Digital Twin reconstruction and SDE trajectory accuracy
python evaluate/suite_twin.py

# Evaluate the RL policy performance against baselines
python evaluate/suite_rl.py

# Run ablation studies
python evaluate/ablation_study.py

# Generate final comprehensive system/clinical report
python evaluate/final_report.py
```

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome!

**License:** This project is licensed under the **MIT License**. See the `LICENSE` file for more information.

*(If you use HUA-DTIP in your research, please cite our corresponding publication.)*
