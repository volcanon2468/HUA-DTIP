# HUA-DTIP: Holistic User Activity Digital Twin with Intervention Planning

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg">
</div>

<br/>

**HUA-DTIP** is an advanced machine learning framework for generating predictive, continuous-time digital twins of human health and activity from multi-modal wearable sensor data. By combining representation learning, generative modeling, and reinforcement learning, the system monitors health state trajectories, predicts future states, detects data drift, trains securely across decentralised clients, and plans optimal, safe clinical interventions.

---

##  Key Features & Capabilities

-  **Digital Twin VAE & Neural SDEs**: A Bayesian β-VAE (β = 0.1, latent dim = **10**) encodes the **512-dimensional hierarchical temporal representation** (not the raw 48-D feature vector) into a probabilistic latent space. A Latent Neural SDE (latent dim = 10) models continuous forward evolution of these states under activity and rest inputs.
-  **Multi-Modal Fusion**: Fuses 1D signals (IMU, ECG, PPG) with 48 tabular features using Transformer-based Cross-Modal Fusion with specialised encoders (SWCTNet, CardioEncoder, FeatureEncoder). Fusion is trained **end-to-end via downstream task losses** (HAR cross-entropy + HR regression), not via any standalone modality-reconstruction objective.
-  **Hierarchical Temporal Modeling**: Captures short-term (MicroScale: Dilated TCN + Transformer), medium-term (MesoScale: Self-Attention over 7-day windows with recency weighting), and long-term (MacroScale: N-BEATS) temporal dependencies. Fused via `HierarchicalFusion` into a 512-D representation.
-  **RL Intervention Planning**: Soft Actor-Critic (SAC) policy trained on twin rollouts. Reward is computed via `MultiObjectiveReward` (weights: progress 0.30, safety 0.40, recovery 0.20, stability 0.10, plus a +2.5 per-step baseline). Bounded by a `SafetyGuard`.
-  **Privacy-Preserving Federated Learning**: Trains across 249 subjects without centralising raw data using FedProx proximal aggregation and FedPer personalisation layers. Subjects are clustered into cohorts based on 96-dimensional activity profiles (48-D mean + 48-D std across all handcrafted features).
-  **Concept Drift & Novelty Detection**: Adapts online to changing physiological distributions using MMD, ADWIN, and a Novelty Autoencoder (input dim = 48, operating on the handcrafted feature vector, independent of the VAE pipeline). EWC regularisation preserves task knowledge during updates using negative log-likelihood-based Fisher Information.
-  **Clinical Recommendation Engine**: Translates latent trajectories into plain-language clinical recommendations with counterfactual analysis, Monte Carlo rollouts, intervention ranking, and uncertainty-bounded confidence scores.

---

## ️ System Architecture (The 7 Pillars)

The HUA-DTIP pipeline is organised into 7 cohesive subsystems:

1. **Multi-Modal Encoders (`src/encoders/`)**
   Raw sensors → modality-specific embeddings. `SWCTNet` (sliding-window IMU CNN + Transformer, output 256-D), `CardioEncoder` (ResBlock1D + Self-Attention, output 128-D), `FeatureEncoder` (tabular MLP over the 48-D handcrafted vector, output 64-D). `CrossModalFusion` combines all three (256+128+64) into a single 128-D representation via a 2-layer Transformer with learned modality-importance weighting. **Fusion has no reconstruction heads** — it is trained end-to-end using downstream HAR and HR losses.

2. **Hierarchical Temporal Modeling (`src/temporal/`)**
   Fuses the 128-D fused representation across time scales. `MicroScaleModel` uses Dilated TCNs + Transformer for high-frequency dynamics; `MesoScaleModel` applies recency-weighted attention over 7-day windows; `MacroScaleModel` forecasts long-range trends with N-BEATS decomposition. `HierarchicalFusion` combines all three into the final **512-D** temporal representation, which is the input to the digital twin.

3. **Neural SDE Twin State Evolution (`src/twin/`)**
   The core digital twin. `BayesianVAE` (input dim = **512**, latent dim = **10**, β = 0.1) encodes the 512-D `HierarchicalFusion` output into a probabilistic 10-D latent space. The decoder reconstructs the **full 512-D input** (no slicing). `LatentNeuralSDE` (latent dim = 10) models continuous forward evolution using activity and rest context vectors as conditioning signals.

4. **Reinforcement Learning & Policy Network (`src/rl/`)**
   Models clinical interventions as an MDP. `TwinGymEnv` simulates patient responses; observation space = latent dim × 2 = **20-D** (10 mean + 10 std, derived directly from `vae.latent_dim` at runtime — never hardcoded). The environment computes an internal reward for consistency, but the SAC training signal is driven by `MultiObjectiveReward` (weights: progress 0.30, safety 0.40, recovery 0.20, stability 0.10, plus a +2.5 per-step baseline), called explicitly in the training loop using the latent state returned in `info`. `SquashedGaussianActor` and `TwinCritic` (SAC, hidden = 256, state_dim = 20) optimise this objective, bounded by `SafetyGuard`.

5. **Federated Learning (`src/federated/`)**
   Trains across 249 distinct subjects locally. `SubjectClusterer` groups cohorts using 96-D profiles (48-D mean + 48-D std of the handcrafted feature vector). `FedPerClient` handles local personalisation layers. `FedProxServer` handles proximal aggregation.

6. **Concept Drift (`src/drift/`)**
   Monitors streams for statistical shifts. `ADWINDetector` for mean shifts, `AutoencoderNoveltyDetector` (input dim = 48) for structural anomalies, `MMDDetector` for distribution discrepancies. `EWCRegularizer` uses negative log-likelihood-based Fisher Information for correct parameter importance estimation.

7. **Recommendation Engine (`src/recommendation/`)**
   The user-facing interface. `MCRolloutEngine` runs Monte Carlo trajectory rollouts, `WhatIfEngine` evaluates counterfactual plans, `rank_interventions` scores and orders plans, and `build_periodized_plan` generates structured weekly training blocks.

---

##  Key Dimensions at a Glance

| Component | Input Dim | Output / Latent Dim |
|---|---|---|
| IMU encoder (SWCTNet) | raw IMU window (T × 9) | 256-D |
| Cardio encoder | raw PPG/ECG window (T × 2) | 128-D |
| Feature encoder | handcrafted features | 64-D |
| Handcrafted feature vector | — | **48-D** |
| CrossModalFusion | 256 + 128 + 64 | 128-D |
| HierarchicalFusion (micro + meso + macro) | 128-D (per window, aggregated) | **512-D** |
| BayesianVAE | **512** (HierarchicalFusion output) | latent **10** |
| LatentNeuralSDE | latent 10 | latent 10 |
| SAC observation space | latent × 2 | **20-D** |
| Subject profile (clustering) | 48-D feature vector | **96-D** (mean + std) |
| Novelty Autoencoder (drift) | 48-D feature vector | reconstruction of 48-D |
| MesoScale window | 128-D fused representation | 512-D (7-day summary) |

> ️ **Common source of confusion:** the 48-D handcrafted feature vector and the 512-D `HierarchicalFusion` output are two *different* stages of the pipeline. The 48-D vector feeds `FeatureEncoder` early on (Pillar 1) and separately feeds the drift/clustering modules directly. The 512-D representation is a *downstream* product of the full encoder + temporal stack, and it — not the 48-D vector — is what feeds `BayesianVAE`.

---

##  Project Structure

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

##  Installation & Prerequisites

**Prerequisites:**
- Python 3.11+
- CUDA-enabled GPU (highly recommended for SDE and Transformer training)

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
   *Core libraries: `torch`, `torchsde`, `stable-baselines3`, `captum`, `flwr`, `river`, `scikit-learn`, `omegaconf`.*

---

##  Datasets

The system is evaluated against multiple open-source health and activity datasets. Place raw files into corresponding subdirectories under `data/raw/`:

- **MHEALTH** (`mhealth/`): Mobile Health dataset `.log` files.
- **PAMAP2** (`pamap2/`): Physical Activity Monitoring `.dat` files.
- **4-Week PPG/HRV** (`ppg_4week/`): Longitudinal PPG CSV files.
- **Stroke Rehabilitation** (`stroke_rehab/`): Clinical IMU recordings.
- **CAPTURE-24** (`capture24/`): Free-living activity per-subject CSVs.
- **MEx** (`mex/`): Exercise form recordings.

Data is loaded via `src/preprocessing/dataset_loaders.py`, which handles missing data imputation, bandpass filtering, windowing (window = 1000 samples, stride = 500), and per-subject z-score normalisation. Window labels are assigned by **majority vote** across all samples in the window span.

---

##  Getting Started (Training Pipeline)

The system uses a sequential training pipeline. Each stage depends on representations learned by the previous step:

1. **Pretrain Multi-Modal Encoders** (Representation Learning: IMU contrastive pretraining + fine-tuning, Cardio HR regression, Feature+Fusion task-supervised training)
   ```bash
   python train/train_encoders.py
   ```
2. **Train Hierarchical Temporal Models** (Micro/Meso/Macro + HierarchicalFusion → 512-D representation)
   ```bash
   python train/train_temporal.py
   ```
3. **Train the Digital Twin** (Bayesian VAE on the 512-D representation → 10-D latent, then Latent Neural SDE, then joint fine-tuning)
   ```bash
   python train/train_twin.py
   ```
4. **Train the RL Policy** (SAC over the 20-D twin observation space, reward from `MultiObjectiveReward`)
   ```bash
   python train/train_rl.py
   ```
5. **Train Federated Learning Agents** (Privacy-Preserving Aggregation across 249 subjects)
   ```bash
   python train/train_federated.py
   ```

*(Drift components are evaluated iteratively or during online rollout simulations via `evaluate/` scripts.)*

---

## ️ Configuration Management

All hyperparameters, dataset paths, and neural network dimensions are managed via YAML configurations in `configs/`:

- `data.yaml`: Window sizes, normalisation methods, HRV feature lists, dataset specifics.
- `model.yaml`: Transformer heads, SDE drift dimensions, VAE dimensions (**input=512, latent=10, β=0.1**), SAC dimensions (**state=20, action=6**).
- `training.yaml`: Learning rates, contrastive temperatures, epochs, early stopping, VAE beta.

**Overriding from CLI:**
```bash
python train/train_encoders.py training.encoders.lr=5e-4 model.imu_encoder.transformer_heads=4
```

---

##  Evaluation & Reporting

Run evaluation suites from `evaluate/`. Scripts generate metrics (RMSE, Pearson R, Coverage Probability, Macro-F1, Trajectory MAE), logging to console and WandB.

```bash
# Evaluate encoder quality (LOSO activity F1, HR MAE)
python evaluate/suite_encoders.py

# Evaluate Digital Twin reconstruction (MSE < 0.05) and trajectory (MAE < 0.12)
python evaluate/suite_twin.py

# Evaluate RL policy performance against random and rule-based baselines
python evaluate/suite_rl.py

# Run ablation studies across component subsets
python evaluate/ablation_study.py

# Generate final comprehensive clinical report
python evaluate/final_report.py
```

**Target metrics (from project report):**

| Metric | Target |
|---|---|
| Activity classification F1 (LOSO) | > 0.85 |
| Heart rate MAE | < 5 bpm |
| Twin state reconstruction MSE (on the 512-D representation) | < 0.05 |
| 7-day trajectory prediction MAE | < 0.12 |
| Uncertainty calibration (95% coverage) | > 0.90 |

---
