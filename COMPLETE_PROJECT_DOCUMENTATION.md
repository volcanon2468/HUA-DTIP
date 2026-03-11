# HUA-DTIP — Complete Technical Documentation

> **Holistic User Activity Digital Twin with Intervention Planning**
>
> This document provides an exhaustive, highly detailed explanation of every component of the HUA-DTIP project. After reading it, you should be able to confidently answer any question about the project's purpose, architecture, algorithms, data flows, individual files, and how they all interconnect.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack & Dependencies](#3-tech-stack--dependencies)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
   - 4.1 [Configuration Files (`configs/`)](#41-configuration-files)
   - 4.2 [Data Pipeline — Preprocessing (`src/preprocessing/`)](#42-data-pipeline--preprocessing)
   - 4.3 [Encoders — Signal-to-Embedding (`src/encoders/`)](#43-encoders--signal-to-embedding)
   - 4.4 [Temporal Modeling (`src/temporal/`)](#44-temporal-modeling)
   - 4.5 [Digital Twin (`src/twin/`)](#45-digital-twin)
   - 4.6 [Reinforcement Learning (`src/rl/`)](#46-reinforcement-learning)
   - 4.7 [Drift Detection (`src/drift/`)](#47-drift-detection)
   - 4.8 [Federated Learning (`src/federated/`)](#48-federated-learning)
   - 4.9 [Simulation & Intervention Ranking (`src/simulation/`)](#49-simulation--intervention-ranking)
   - 4.10 [Utilities (`src/utils/`)](#410-utilities)
   - 4.11 [Training Scripts (`train/`)](#411-training-scripts)
   - 4.12 [Evaluation Scripts (`evaluate/`)](#412-evaluation-scripts)
   - 4.13 [Top-Level Scripts](#413-top-level-scripts)
   - 4.14 [Checkpoints & Outputs](#414-checkpoints--outputs)
5. [Core Logic & Workflows](#5-core-logic--workflows)
6. [Setup & Execution](#6-setup--execution)
7. [Datasets](#7-datasets)
8. [Key Concepts Glossary](#8-key-concepts-glossary)

---

## 1. Executive Summary

### What Is This Project?

**HUA-DTIP** (Holistic User Activity Digital Twin with Intervention Planning) is a multi-modal wearable health intelligence system. It ingests raw sensor data from wearable devices (smartwatches, heart rate monitors, IMU sensors) and builds a **Digital Twin** — a compact, learned mathematical model of a person's health state — then uses that model to **simulate future health trajectories** and **recommend optimal exercise/rest interventions** via reinforcement learning.

### What Does It Do?

1. **Ingests raw sensor signals** — accelerometer, gyroscope, ECG, PPG from 6 different public datasets covering 249+ subjects.
2. **Preprocesses** — cleans noise, handles missing data, segments signals into 20-second windows, extracts 48 hand-crafted statistical features plus 5 HRV (Heart Rate Variability) features per window.
3. **Encodes** — three neural network encoders compress raw IMU signals (→256D), cardiac signals (→128D), and statistical features (→64D) into compact embeddings, then a cross-modal Transformer fuses all three into a unified 128D representation per window.
4. **Models time** — three temporal models capture activity patterns at micro (last 3 hours), meso (last 7 days), and macro (last 6 months) scales, then hierarchically fuse them into a 512D temporal state.
5. **Builds the Digital Twin** — a β-VAE compresses the 512D temporal state into a 10D latent health vector `z`, and a Neural SDE (Stochastic Differential Equation) simulates how `z` evolves forward in time under different activity/rest interventions.
6. **Plans interventions** — a Soft Actor-Critic (SAC) RL agent learns to choose optimal exercise intensity, duration, rest, nutrition, and sleep parameters to maximize long-term health while respecting safety constraints.
7. **Simulates outcomes** — a Monte Carlo rollout engine runs 200+ stochastic simulations per intervention plan to produce risk estimates (overtraining probability, injury probability) and ranks competing plans.
8. **Adapts to drift** — ADWIN, MMD, and autoencoder novelty detectors monitor for concept/distribution drift, triggering model re-training (with EWC to prevent catastrophic forgetting) when the user's body changes.
9. **Preserves privacy** — FedProx federated learning with FedPer personalization trains models across multiple users without sharing raw data, while differential privacy provides formal privacy guarantees.

### Core End-Goal

Given a person's recent wearable sensor data, produce a **personalized, safety-aware, periodized 12-week exercise/rest plan** that maximizes long-term fitness improvement while minimizing injury and overtraining risk — all without requiring the user to share their raw health data with a central server.

### Target Use Cases
- Personalized fitness planning (athletes, general wellness)
- Stroke rehabilitation monitoring
- Chronic disease management (cardiovascular disease, post-COVID recovery)
- Sports science research

---

## 2. High-Level Architecture

The system is a sequential pipeline of 6 major stages, each building on the previous:

```
Raw Sensor Data (Accelerometer, Gyroscope, ECG, PPG)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 1: PREPROCESSING                              │
│  Clean signals (bandpass/highpass filter)             │
│  Handle missing data (interpolation/discard)         │
│  Segment into 1000-sample windows (20s @ 50Hz)       │
│  Z-score normalize per subject                       │
│  Extract 48 features + 5 HRV features per window     │
│  Build 512-dim daily summary vectors                  │
│  Save as .pt files per subject per window             │
│                                                      │
│  Files: dataset_loaders.py, signal_cleaning.py,      │
│         windowing.py, feature_extraction.py,          │
│         hrv.py, daily_summary.py                      │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 2: ENCODERS                                   │
│  IMU Encoder (SWCTNet): [B,1000,9] → [B,256]        │
│    Multi-scale CNN (k=5,7,11) + Transformer          │
│  Cardio Encoder: [B,1000,2] → [B,128]               │
│    ResNet1D + Self-Attention                          │
│  Feature Encoder: [B,48] → [B,64]                   │
│    3-layer MLP with LayerNorm                         │
│  Cross-Modal Fusion: [256+128+64] → [B,128]         │
│    Project to common dim → 2-layer Transformer       │
│    → Weighted aggregation                             │
│                                                      │
│  Files: imu_encoder.py, cardio_encoder.py,           │
│         feature_encoder.py, fusion.py                 │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 3: TEMPORAL MODELING                          │
│  Micro (3h): TCN + ViT → [B,256]                    │
│    180 fused windows → dilated TCN → CLS-Transformer │
│  Meso (7d): Transformer → [B,512]                    │
│    7 daily summaries + day-of-week/position embeds   │
│    → 4-layer Transformer → recency-weighted average  │
│  Macro (6mo): N-BEATS → [B,128]                     │
│    6 monthly vectors → trend/season/generic decomp.  │
│  Hierarchical Fusion: [256+512+128] → [B,512]       │
│    Top-down attention (macro→meso)                    │
│    Bottom-up attention (meso→micro)                   │
│    Gated combination → output projection              │
│                                                      │
│  Files: micro_scale.py, meso_scale.py,               │
│         macro_scale.py, hierarchical_fusion.py        │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 4: DIGITAL TWIN                               │
│  β-VAE: [B,512] → μ[B,10], σ[B,10] → z[B,10]      │
│    Encoder MLP → (μ,logvar) → reparameterize         │
│    Decoder MLP: z → reconstructed state              │
│    Prediction head: (z,activity) → HR + 5 HRV        │
│    Loss = reconstruction + β×KL + prediction          │
│  Neural SDE: z(t) → z(t+Δt)                          │
│    dz = f(z,activity,rest,t)dt + g(z,t)dW           │
│    Drift net: (z+activity_embed+rest_embed+t) → dz  │
│    Diffusion net: (z+t) → noise magnitude             │
│    Euler-Maruyama integration via torchsde            │
│                                                      │
│  Files: bayesian_vae.py, latent_sde.py               │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 5: RL + SIMULATION + INTERVENTION PLANNING    │
│  Environment: TwinGymEnv (28-day episodes)           │
│    obs=[z_mu(10),z_std(10)], action=[0,1]^6          │
│  SAC Agent: Actor(state→action), TwinCritic          │
│    Multi-objective reward (progress/safety/recovery)  │
│    Safety guard (fatigue limits, consecutive days)    │
│  MC Rollout: 200 SDE trajectories per scenario       │
│    Overtraining/injury probability estimation        │
│  What-If Engine: compare/grid-search scenarios       │
│  Intervention Ranking: 5 presets + custom             │
│    Top-k filtering → 12-week periodized plan          │
│                                                      │
│  Files: sac_networks.py, reward.py, safety.py,       │
│         mc_rollout.py, what_if.py,                    │
│         intervention_ranking.py                       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 6: DRIFT DETECTION + FEDERATED LEARNING       │
│  ADWIN: 48 streaming detectors (1 per feature)       │
│  MMD: Distribution shift via kernel two-sample test   │
│  Autoencoder Novelty: reconstruction error threshold  │
│  DriftManager: combines all 3 → diagnose → act       │
│    EWC regularizer to prevent catastrophic forgetting │
│  FedProx Server: aggregate client updates + proximal │
│  FedPer Client: shared encoder + personal head        │
│  Subject Clustering: K-Means on 48D profiles          │
│                                                      │
│  Files: adwin.py, mmd.py, autoencoder_novelty.py,    │
│         drift_manager.py, fedprox_server.py,          │
│         fedper_client.py, clustering.py               │
└──────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Raw .log/.dat/.csv files
  ↓ (preprocess_all.py / run_preprocessing.py)
data/processed/subject_N/windows/window_NNNNN.pt
data/processed/subject_N/daily_summaries/day_NNN.pt
  ↓ (train_encoders.py)
checkpoints/encoder_*.pt
  ↓ (train_temporal.py)
checkpoints/temporal_*.pt
  ↓ (train_twin.py)
checkpoints/twin_*.pt
  ↓ (train_rl.py)
checkpoints/rl_*.pt
  ↓ (train_federated.py)
checkpoints/federated_global.pt
  ↓ (evaluate/*.py)
results/*.csv, results/*.json
results/final_evaluation_report.json
```

---

## 3. Tech Stack & Dependencies

### Language
- **Python 3.10+** — the entire project is pure Python

### Core ML Framework
| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.1.0 | Neural network models, GPU training, tensor operations |
| **torchvision** | 0.16.0 | Utility transforms (not heavily used directly) |
| **torchaudio** | 2.1.0 | Audio-style signal processing utilities |
| **torchsde** | 0.2.6 | Neural SDE integration via `sdeint()` with Euler-Maruyama solver — core to the Digital Twin's stochastic trajectory prediction |

### Signal Processing & Feature Engineering
| Library | Version | Purpose |
|---------|---------|---------|
| **scipy** | 1.11.4 | Butterworth bandpass/highpass/lowpass filters (`butter`, `filtfilt`), signal resampling (`resample_poly`) |
| **neurokit2** | 0.2.7 | ECG processing and HRV computation (SDNN, RMSSD, LF/HF power). Falls back to pure NumPy if unavailable |
| **numpy** | 1.26.2 | Numerical operations everywhere — windowing, feature extraction, FFT-based features |
| **pandas** | 2.1.4 | Dataset loading (CSV/log files), groupby for daily aggregation |

### Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | 1.3.2 | K-Means clustering (subject clustering), F1/precision/recall metrics |
| **stable-baselines3** | 2.2.1 | Listed in requirements but the SAC implementation is custom (not using SB3's SAC) |
| **gymnasium** | 0.29.1 | OpenAI Gym interface — `TwinGymEnv` extends `gym.Env` for the RL environment |
| **captum** | 0.7.0 | Model interpretability (listed but not directly invoked in source; available for feature attribution) |

### Drift Detection
| Library | Version | Purpose |
|---------|---------|---------|
| **river** | 0.19.0 | Production-grade ADWIN implementation for streaming concept drift detection. Falls back to custom NumPy implementation if unavailable |

### Federated Learning
| Library | Version | Purpose |
|---------|---------|---------|
| **flwr** (Flower) | 1.6.0 | FL framework — listed in requirements but the actual implementation uses custom FedProx/FedPer code rather than Flower's built-in strategies |

### Configuration & Experiment Tracking
| Library | Version | Purpose |
|---------|---------|---------|
| **hydra-core** | 1.3.2 | YAML configuration loading with CLI overrides. Training scripts use `@hydra.main()` decorator |
| **omegaconf** | 2.3.0 | Structured configuration objects underlying Hydra. Used via `OmegaConf.load()` for programmatic config access |
| **wandb** | 0.16.2 | Weights & Biases experiment tracking. Runs in **offline mode** by default (logs to local `wandb/` dir, no cloud sync needed) |
| **pytorch-lightning** | 2.1.3 | Listed but not directly used in training scripts — the project implements its own training loops |

### Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | 3.8.2 | Plotting (used in evaluation reports) |

### Why These Choices?
- **PyTorch over TensorFlow**: Better for research-grade code with custom training loops, dynamic computation graphs, and direct control over gradient flow
- **torchsde**: Only library offering differentiable SDE solvers in PyTorch — required for backpropagating through the Neural SDE
- **Hydra/OmegaConf**: Industry standard for ML configuration management; CLI overrides enable hyperparameter sweeps without code changes
- **WandB offline**: Enables experiment tracking without internet connectivity, crucial for on-device/privacy-preserving scenarios
- **Custom FL over Flower**: More control over the FedProx proximal term and FedPer layer splitting

---

## 4. File-by-File Breakdown

### 4.1 Configuration Files

#### `configs/data.yaml`
**What it is:** Data pipeline configuration.

**Key settings:**
- `paths`: Locations of raw data (`data/raw/mhealth`, etc.) and processed output (`data/processed`)
- `window.size: 1000` — each window is 1000 samples = **20 seconds at 50 Hz**
- `window.stride: 500` — windows overlap by 50% (500 samples)
- `window.sample_rate: 50` — target sampling rate in Hz
- `normalization.method: per_subject_zscore` — each subject's data is independently z-score normalized (subtract mean, divide by std). This is crucial because sensor placement differences create systematic offsets between subjects
- `missing_data.interpolate_threshold: 0.10` — if <10% of a window is NaN, interpolate; `discard_threshold: 0.30` — if >30% is NaN or a contiguous gap exceeds 30%, discard the window
- `hrv.features: [SDNN, RMSSD, LF, HF, LF_HF_ratio]` — the 5 HRV features computed per window
- Dataset-specific settings: number of subjects, channels, activity classes for MHEALTH (10 subjects, 12 classes), PAMAP2 (9 subjects, 18 classes), PPG (49 subjects, 28 days), CAPTURE-24 (151 subjects), MEx (30 subjects)

**Used by:** `run_preprocessing.py`, `preprocess_all.py`, all training/eval scripts via `OmegaConf.load("configs/data.yaml")`

#### `configs/model.yaml`
**What it is:** Architecture hyperparameters for every neural network.

**Key sections:**
- `imu_encoder`: CNN channels [64,128,256], kernel sizes [5,7,11] (multi-scale), 8-head 3-layer Transformer, output_dim=256
- `cardio_encoder`: ResBlock channels [64,128,128], kernels [7,5,3], 4-head attention, output_dim=128
- `feature_encoder`: input=48, hidden=[128,128], output=64, dropout=0.3
- `fusion`: common_dim=128, 4-head 2-layer Transformer, ff_dim=256
- `micro_scale`: TCN channels=256, kernel=3, dilations=[1,2,4,8], 8-head 3-layer Transformer, sequence_len=180
- `meso_scale`: input=512, 8-head 4-layer Transformer, ff_dim=1024, recency_weights=[0.05→0.30], sequence_len=7
- `macro_scale`: input=640 (512 meso + 128 stats), block_dim=256, output=128, sequence_len=6
- `hierarchical_fusion`: proj_dim=256, output=512
- `twin_vae`: input=512, hidden=[256,128], **latent_dim=10**, decoder_hidden=[64,128,256], **beta=4.0**
- `latent_sde`: latent_dim=10, activity_input=6, rest_input=3, drift_hidden=128, diffusion_hidden=32
- `sac`: **state_dim=68** (in config but actual code uses 20=10+10), action_dim=6, actor_hidden=[256,256], target_entropy=-6.0
- `drift`: adwin_delta=0.002, mmd_kernel_sigma=1.0, autoencoder input=128, novelty_percentile=95
- `federated`: n_clusters=4, proximal_mu=0.01, local_epochs=5, **dp_epsilon=1.0** (differential privacy), dp_delta=1e-5

**Used by:** Every model instantiation — though models currently use defaults that match these values

#### `configs/training.yaml`
**What it is:** Training hyperparameters.

**Key sections:**
- `seed: 42` — reproducibility seed
- `device: cuda` — GPU preference (falls back to CPU)
- `training.encoders`: lr=1e-3, batch=64, epochs=100, patience=10, cosine scheduler, contrastive temperature=0.07
- `training.temporal`: lr=1e-3, batch=32, epochs=80, patience=10
- `training.twin`: lr=1e-3, batch=64, epochs=100, patience=10, cosine scheduler; **SDE lr=5e-4, epochs=80**; joint_finetune_lr=2e-4
- `training.rl`: **5000 episodes**, episode_length=28 steps (one step = one day), batch=256, **buffer=100K** transitions, warmup=1000, gamma=0.99, tau=0.005, actor/critic/alpha lr=3e-4
- `training.federated`: **n_clients=249**, n_rounds=50, local_epochs=5, client_lr=1e-3, fedprox_mu=0.01
- `wandb.project: hua-dtip`
- `checkpoints.dir: checkpoints/`, `results_dir: results/`

**Used by:** All training scripts via Hydra `@hydra.main()` or `OmegaConf.load()`

---

### 4.2 Data Pipeline — Preprocessing

#### `src/preprocessing/dataset_loaders.py`
**What it is:** Six PyTorch `Dataset` classes, one per raw dataset, that load raw sensor files into pandas DataFrames.

**Classes:**

| Class | Dataset | Input Format | Output per Sample |
|-------|---------|--------------|-------------------|
| `MHEALTHDataset` | MHEALTH | `.log` files (whitespace-separated, 24 columns) | `(imu[9], ecg[2], label)` |
| `PAMAP2Dataset` | PAMAP2 | `.dat` files (whitespace-separated, 54 columns) | `(imu[9], hr[1], label)` |
| `FourWeekPPGDataset` | 4-Week PPG | CSV files with timestamps | `(subject_id, date, DataFrame)` grouped by day |
| `StrokeRehabDataset` | Stroke Rehab | CSV files named `patient_N_visitN.csv` | DataFrame per recording |
| `CAPTURE24Dataset` | CAPTURE-24 | `subject_N.csv` files | Row-level accelerometer data |
| `MExDataset` | MEx | `subject_N.csv` files | Row-level exercise data |

**Key implementation details:**
- `MHEALTHDataset` filters out activity_label=0 (null class), loads subjects 1-10
- `PAMAP2Dataset` forward-fills heart rate NaNs, interpolates IMU NaNs, loads subjects 101-109
- `FourWeekPPGDataset` groups data by date for daily analysis, handles both per-subject CSVs and a combined `sensor_data.csv`
- `StrokeRehabDataset` parses patient_id and visit number from filenames
- Column constants `IMU_COLS` (9 wrist+ankle channels), `ECG_COLS` (2 ECG channels), `PAMAP2_IMU_COLS` (9 hand+chest channels) define exactly which raw columns are used

**Used by:** `run_preprocessing.py`, `preprocess_all.py`

#### `src/preprocessing/signal_cleaning.py`
**What it is:** Classical digital signal processing functions for cleaning raw sensor signals.

**Functions:**

| Function | What It Does | Parameters |
|----------|-------------|------------|
| `bandpass_filter(signal, fs, low=0.5, high=20.0)` | 4th-order Butterworth bandpass. Keeps 0.5–20 Hz. Removes baseline drift and high-frequency noise. Used on IMU signals. | Forward-backward filtering via `filtfilt` (zero phase distortion) |
| `highpass_filter(signal, fs, cutoff=0.5)` | Removes frequencies below 0.5 Hz. Used on ECG to remove baseline wander. | Same Butterworth design |
| `lowpass_filter(signal, fs, cutoff=20.0)` | Removes frequencies above 20 Hz. Used for PPG motion artifact removal. | |
| `resample_signal(signal, orig_fs, target_fs)` | Polyphase resampling via `scipy.signal.resample_poly`. PAMAP2 (100 Hz) → 50 Hz target. Uses GCD for optimal up/down factors. | |
| `compute_snr(signal, noise_band_high=1.0, fs=50.0)` | FFT-based Signal-to-Noise Ratio in dB. Noise = power below 1 Hz, signal = power above 1 Hz. | Used in quality scoring |
| `handle_missing(signal, thresh_interp=0.10, thresh_discard=0.30)` | Detects NaN fraction. <10%: linear interpolation. >30% or contiguous gap >30%: marks for discard. Returns `(cleaned_signal, should_discard)`. | |
| `remove_motion_artifact(ppg, acc, fs=50.0)` | Subtracts low-pass-filtered accelerometer magnitude (correlated with wrist movement) from PPG signal, weighted by their Pearson correlation. Reduces motion artifacts in heart rate signals. | |

**Why filtering matters:** Raw sensor signals contain: (a) slow baseline drift from body movement, (b) high-frequency electronic noise, (c) power line interference at 50/60 Hz, (d) motion artifacts in PPG from wrist movement. Without cleaning, feature extraction and neural networks would learn noise instead of signal.

#### `src/preprocessing/windowing.py`
**What it is:** Segments continuous signals into fixed-length overlapping windows, normalizes them, and orchestrates the full per-subject processing pipeline.

**Functions:**

| Function | What It Does |
|----------|-------------|
| `create_windows(signal, window_size=1000, stride=500)` | Sliding window segmentation. 50% overlap ensures no temporal patterns are missed at window boundaries. Returns list of numpy arrays. |
| `per_subject_zscore(windows)` | Concatenates all windows from one subject, computes global mean and std, then normalizes each window. Returns `(normalized_windows, (mu, sigma))`. Critical because different sensor placements → different absolute values. |
| `assess_quality(window)` | Scores each window 0.0–1.0 based on: **completeness** (40% weight, 1 − NaN fraction), **saturation** (30%, fraction of values at extremes), **variance** (30%, std / 0.01 clipped to [0,1]). Low-quality windows are kept but down-weighted in daily summaries. |
| `process_subject(imu, cardio, features_fn, hrv_fn, subject_id, out_dir, ...)` | The main pipeline function. For one subject: creates IMU and cardio windows, normalizes, assesses quality, extracts features and HRV per window, saves each as a `.pt` file containing `{imu, cardio, features, hrv, label, timestamp, quality}`. |

**Each saved `.pt` window contains:**
```python
{
    "imu":       tensor[1000, 9],   # 20s of 9-channel normalized IMU
    "cardio":    tensor[1000, 2],   # 20s of 2-channel normalized ECG/PPG
    "features":  tensor[48],        # 48 extracted statistical features
    "hrv":       tensor[5],         # [SDNN, RMSSD, LF, HF, LF/HF]
    "label":     int,               # Activity class (-1 if unknown)
    "timestamp": float,             # Seconds since recording start
    "quality":   float              # Quality score 0.0–1.0
}
```

#### `src/preprocessing/feature_extraction.py`
**What it is:** Extracts 48 hand-crafted scalar features from each 20-second window. These features capture domain-specific knowledge about human movement and cardiac physiology.

**`extract_imu_features(imu)` → 20 features:**

| # | Feature | How Computed | What It Measures |
|---|---------|-------------|-----------------|
| 0 | cadence | Zero-crossings of vertical acceleration / time | Steps per minute |
| 1 | step_count | cadence × window_duration / 2 | Number of steps in window |
| 2 | step_regularity | Autocorrelation at lag=50 (1 second) | How rhythmic is the gait |
| 3 | SMA | Mean of acceleration magnitude | Overall movement intensity |
| 4 | MAD | Mean absolute deviation of acc magnitude | Movement variability |
| 5 | energy | Mean power of FFT of acc magnitude | Frequency-domain energy |
| 6 | stride_length | Max − min of acc magnitude | Amplitude of stride |
| 7 | gait_symmetry | Correlation between first and second half | Left/right symmetry |
| 8-10 | acc_mean (x,y,z) | Mean per axis | Orientation/posture |
| 11-13 | acc_std (x,y,z) | Std per axis | Movement variability per axis |
| 14-16 | gyro_mean (x,y,z) | Mean angular velocity per axis | Rotation patterns |
| 17-19 | gyro_std (x,y,z) | Std angular velocity per axis | Rotation variability |

**`extract_cardio_features(cardio)` → 20 features:**

| # | Feature | What It Measures |
|---|---------|-----------------|
| 0-4 | hr_mean, hr_std, hr_min, hr_max, hr_range | Heart rate statistics from peak detection |
| 5 | rmssd | Root Mean Square of Successive RR Differences — parasympathetic activity |
| 6 | sdnn | Standard deviation of RR intervals — overall HRV |
| 7-8 | lf_power, hf_power | FFT power in 0.04–0.15 Hz (sympathetic+parasympathetic) and 0.15–0.4 Hz (parasympathetic) bands |
| 9 | lf_hf_ratio | Sympathovagal balance — stress vs. rest indicator |
| 10 | hr_reserve_pct | % of max heart rate reserve used |
| 11 | recovery_rate_proxy | How far from max effort (1 − HR/150) |
| 12-14 | ppg_amplitude, ppg_rise_time, ppg_fall_time | PPG pulse morphology |
| 15 | resp_rate_proxy | Estimated breathing rate from HRV spectrum |
| 16 | spo2_proxy | Blood oxygen saturation proxy from PPG amplitude |
| 17 | cardiac_output_proxy | HR × PPG amplitude / 100 |
| 18-19 | parasympathetic_index, sympathetic_index | Normalized autonomic balance |

**`extract_quality_features(imu, cardio)` → 8 features:**
IMU SNR, cardio SNR, PPG SNR, completeness (1−NaN fraction), artifact fraction (high-frequency power ratio), baseline wander (low-frequency power ratio), R-peak count, beat quality (fraction of peaks above 50% of max).

**`extract_all_features(imu, cardio)` → 48 features** (concatenation of all three groups)

#### `src/preprocessing/hrv.py`
**What it is:** Dedicated HRV (Heart Rate Variability) computation module.

**`compute_hrv_neurokit(ecg_or_ppg, fs=50.0)` → 5 features `[SDNN, RMSSD, LF, HF, LF/HF]`**

Uses `neurokit2` library for clinically accurate R-peak detection and HRV analysis. Falls back to `_hrv_numpy_fallback()` (simple peak detection via derivative sign changes + FFT power spectral analysis) if neurokit2 is not available or processing fails.

**Why HRV matters:** HRV reflects the balance between the sympathetic ("fight or flight") and parasympathetic ("rest and digest") branches of the autonomic nervous system. Low RMSSD and high LF/HF ratio indicate stress, fatigue, or overtraining. High RMSSD and low LF/HF indicate good recovery.

#### `src/preprocessing/daily_summary.py`
**What it is:** Aggregates all windows from a single day into a single 512-dimensional vector that the meso-scale temporal model consumes.

**`build_daily_summary(windows_dir)` → `np.ndarray[512]`**

The 512 dimensions are composed of 4 blocks:
- **[0:256] micro_surrogate**: [0:48] = quality-weighted mean of 48 features; [48:96] = std of features; [96:256] = zero-padded. Represents the "average moment" of the day.
- **[256:384] daily_stats**: [0:5] = HRV mean; [5:10] = HRV std; [10:30] = cardio feature mean; [30:50] = cardio feature std. Cardiovascular profile of the day.
- **[384:448] context**: step_count (feature[1]), average quality, activity transition rate (fraction of label changes between consecutive windows), movement intensity (feature[3] = SMA).
- **[448:512] circadian**: 64-bin histogram of SMA (movement intensity) distributed across normalized timestamps. Each bin ≈ 22 minutes of a 24-hour day. Captures the circadian activity pattern — when the person is active vs. sedentary.

**`save_daily_summary(subject_id, day_idx, vector, out_dir)`**: Saves as `.pt` file at `data/processed/subject_N/daily_summaries/day_NNN.pt`

**`save_metadata(subject_id, meta, out_dir)`**: Writes `metadata.json` with dataset name, window count, etc.

#### `src/preprocessing/run_preprocessing.py`
**What it is:** Hydra-configured entry point for running the full preprocessing pipeline across MHEALTH, PAMAP2, and 4-Week PPG datasets.

Uses `@hydra.main(config_path="../../configs", config_name="data")` to load data.yaml. For each dataset:
1. Loads raw data via the appropriate `Dataset` class
2. Applies signal cleaning (bandpass for IMU, highpass for ECG)
3. Calls `process_subject()` to window, normalize, extract features, and save
4. For PPG: additionally builds and saves daily summaries

#### `src/preprocessing/__init__.py`
Empty file. Makes the directory a Python package.

---

### 4.3 Encoders — Signal-to-Embedding

#### `src/encoders/imu_encoder.py`
**What it is:** The IMU encoder — transforms raw 9-channel motion sensor data into a compact 256-dimensional embedding.

**Classes:**

**`SlidingWindowCNNBlock`** (input: `[B, 1000, 9]` → output: `[B, 256]`):
- Three parallel 1D CNN branches with different kernel sizes (5, 7, 11), each having 3 convolutional layers (in→64→128→out_channels) with BatchNorm + ReLU
- **Why multi-scale kernels?** Kernel 5 (~100ms at 50Hz) captures fast movements (finger taps, quick steps). Kernel 7 (~140ms) captures medium movements. Kernel 11 (~220ms) captures slower movements (posture transitions, arm swings). This ensures no temporal pattern scale is missed.
- Each branch applies AdaptiveAvgPool1d(1) to collapse the time axis → one vector per branch
- Concatenation of 3 branches → 256-dim vector (85+85+86 split to sum to 256)

**`ChannelTimeAttentionTransformer` (CTAT)** (input: `[B, 256]` → output: `[B, 256]`):
- Unsqueezes input to `[B, 1, 256]` (single token sequence)
- 3-layer Transformer Encoder with 8 attention heads, `norm_first=True` (pre-LayerNorm for training stability), feedforward dim = 4×256 = 1024
- Squeezes back to `[B, 256]` and applies LayerNorm
- Although operating on a single token, the Transformer refines the representation through its feedforward layers

**`SWCTNet`** (the complete model):
- Chains `SlidingWindowCNNBlock` → `ChannelTimeAttentionTransformer`
- `build_classifier(n_classes)`: dynamically adds a Linear(256→n_classes) head
- `classify(x)`: forward pass + classifier head (for activity classification)

**`ProjectionHead`** (for contrastive pre-training):
- 2-layer MLP: 256→256→128 with ReLU, L2-normalized output
- Projects embeddings to a lower-dimensional space where contrastive loss operates

**Training approach:**
1. **Self-supervised contrastive pre-training** (NT-Xent loss): Two augmented views of the same window are encoded and projected. The loss forces their projections close while pushing all other windows apart. This learns meaningful representations without labels.
2. **Supervised fine-tuning**: A classifier head is added and trained with cross-entropy on activity labels.

**Checkpoint:** `checkpoints/encoder_imu.pt`, `checkpoints/encoder_imu_pretrained.pt`

#### `src/encoders/cardio_encoder.py`
**What it is:** Encodes raw 2-channel ECG/PPG signals into a 128-dim cardiac embedding.

**Classes:**

**`ResBlock1D`** (Residual Block):
- Two 1D convolutions with BatchNorm, plus a skip connection (1×1 conv if channel dimensions differ, Identity otherwise)
- Skip connections prevent vanishing gradients and enable deeper networks

**`TemporalSelfAttention`**:
- Single-head self-attention (4 heads, `nn.MultiheadAttention`) with LayerNorm
- Allows the model to attend to the most informative parts of the cardiac signal (e.g., R-peaks)

**`CardioEncoder`** (input: `[B, 1000, 2]` → output: `[B, 128]`):
1. **Stem**: Conv1d(2→64, kernel=7) + BatchNorm + ReLU — initial feature extraction
2. **Three ResBlock1Ds**: 64→64 (k=7), 64→128 (k=5), 128→128 (k=3) — progressively extract higher-level features with decreasing kernel sizes (coarse-to-fine)
3. **AdaptiveAvgPool1d(1)**: Collapse time dimension → `[B, 128]`
4. **TemporalSelfAttention**: Adds contextual information via self-attention
5. **`_pad_channels()`**: Handles single-channel PPG input by zero-padding to 2 channels

**`predict_hr(x)`**: Forward pass + `hr_head` (Linear 128→1) → heart rate prediction in BPM

**Training:** Supervised regression — predicts heart rate from ECG/PPG. Loss = MSE. Target: **MAE < 5 BPM**.

**Checkpoint:** `checkpoints/encoder_cardio.pt`

#### `src/encoders/feature_encoder.py`
**What it is:** Encodes the 48 hand-crafted scalar features into a 64-dim embedding.

**`FeatureEncoder`** (input: `[B, 48]` → output: `[B, 64]`):
- 3-layer MLP: 48→128 (LayerNorm+ReLU+Dropout(0.3)) → 128→128 (same) → 128→64 (ReLU only)
- **Why MLP, not CNN/Transformer?** The 48 features are already scalar statistics (means, stds, ratios). There's no sequential/spatial structure to exploit — an MLP is the correct architecture for tabular feature vectors.
- **Why LayerNorm instead of BatchNorm?** For small feature vectors with variable batch sizes, LayerNorm normalizes across the feature dimension which is more stable than BatchNorm's batch statistics.
- **Dropout 0.3**: Regularization to prevent overfitting on the relatively small feature space

**Checkpoint:** `checkpoints/encoder_feature.pt`

#### `src/encoders/fusion.py`
**What it is:** Combines IMU (256D), Cardio (128D), and Feature (64D) embeddings into a single 128D fused representation.

**`CrossModalFusion`** (inputs: `h_imu[B,256]`, `h_cardio[B,128]`, `h_feat[B,64]` → output: `[B,128]`):
1. **Three projection layers**: Each Linear+LayerNorm maps to common_dim=128. This puts all modalities in the same "language"
2. **Unsqueeze + concatenate**: Creates a 3-token sequence `[B, 3, 128]` where token 0=IMU, token 1=Cardio, token 2=Features
3. **2-layer Transformer Encoder** (4 heads, `norm_first=True`): Each token can attend to the others, learning cross-modal relationships (e.g., high acceleration + high HR → running; high acceleration + normal HR → arm movement only)
4. **Weighted aggregation**: A learned `weight_head` (Linear 128→1) produces a scalar weight per token, softmax-normalized, then weighted sum → final 128D vector. This allows the model to dynamically weight modalities by importance.

**Why fusion?** A slow jog and a stressful meeting both elevate heart rate — only IMU data tells them apart. The cross-modal Transformer learns these interactions.

**Checkpoint:** `checkpoints/encoder_fusion.pt`

#### `src/encoders/__init__.py`
Empty. Package marker.

---

### 4.4 Temporal Modeling

#### `src/temporal/micro_scale.py`
**What it is:** Models the last ~3 hours of activity (180 consecutive fused embeddings, each representing a 20-second window).

**Classes:**

**`DilatedResidualBlock`**:
- Weight-normalized 1D convolution with dilation + causal padding (padding = (kernel_size-1) × dilation, then truncated to input length)
- Residual skip connection with ReLU
- **Why weight normalization?** More stable than batch normalization for sequential models, especially with variable sequence lengths

**`TCNBlock`** (Temporal Convolutional Network):
- Input projection (1×1 conv if dimensions differ)
- 4 `DilatedResidualBlock`s with dilations [1, 2, 4, 8]
- **Exponentially growing receptive field:** With kernel_size=3 and dilations [1,2,4,8], the effective receptive field grows to 1+2+4+8 = 15 positions per block, allowing the TCN to see patterns across ~30 windows (10 minutes) efficiently without the O(n²) cost of self-attention

**`MicroScaleModel`** (input: `[B, 180, 128]` → output: `[B, 256]`):
1. **TCN**: Processes the fused embedding sequence → `[B, 256, 180]` → permute → `[B, 180, 256]`
2. **CLS token**: A learned parameter prepended to the sequence → `[B, 181, 256]`
3. **Positional embeddings**: Learned positional embeddings added (initialized with truncated normal)
4. **3-layer Transformer Encoder** (8 heads, `norm_first=True`)
5. **Take CLS token output**: `x[:, 0]` → LayerNorm → `[B, 256]`. The CLS token has attended to the entire sequence and summarized it.
6. **Prediction heads**: `hr_head` (256→1) predicts next heart rate; `hrv_head` (256→5) predicts next 5 HRV values

**Why TCN + Transformer hybrid?** TCN efficiently captures local temporal dependencies with O(n) cost. Transformer then globally attends to the TCN's output, reasoning about relationships between distant time points.

**Checkpoint:** `checkpoints/temporal_micro.pt`

#### `src/temporal/meso_scale.py`
**What it is:** Models the last 7 days of daily health summaries.

**`MesoScaleModel`** (input: `[B, 7, 512]` daily summary vectors → output: `[B, 512]`):
1. **Day-of-week embedding**: `nn.Embedding(7, 512)` — a learned 512D vector for each day of the week (Monday's pattern differs from Sunday's). Added to input if provided.
2. **Position embedding**: `nn.Embedding(7, 512)` — learned positional encoding for positions 0-6
3. **Input LayerNorm**: Normalizes before Transformer
4. **4-layer Transformer Encoder** (8 heads, ff_dim=1024, `norm_first=True`)
5. **Recency-weighted average**: Instead of taking just the last position, computes a weighted sum with weights `[0.05, 0.07, 0.10, 0.12, 0.16, 0.20, 0.30]`. Day 7 (most recent) gets 30% weight, day 1 gets only 5%. This creates a "recent-trend-aware" summary.
6. **Prediction heads**: `next_day_head` (Linear 512→512, predicts tomorrow's daily vector); `capacity_head` (Linear 512→1, predicts physical capacity score)

**Why recency weighting?** Simple taking the last position ignores the pattern. Uniform average loses temporal ordering. The exponential-like recency weights capture both the full week's context and emphasize the most recent trends.

**Checkpoint:** `checkpoints/temporal_meso.pt`

#### `src/temporal/macro_scale.py`
**What it is:** Models 6 months of fitness trajectory using N-BEATS (Neural Basis Expansion Analysis for Time Series).

**`NBEATSBlock`**:
- 3 fully-connected layers (input→256→256→256) with ReLU
- **Backcast head**: Reconstructs the input (used as residual to remove)
- **Forecast head**: Projects to output dimension
- `basis_type` parameter: "trend", "seasonality", or "generic" (label only; all blocks use the same MLP architecture in this implementation)

**`MacroScaleModel`** (input: `[B, 6, 640]` → output: `[B, 128]`):
1. Flattens input: `[B, 6×640]` = `[B, 3840]`
2. **Trend block**: Captures long-term direction (fitness improving/declining) → forecast1 + backcast1
3. **Seasonality block**: Processes (input − backcast1), captures periodic patterns (e.g., 4-week training cycles) → forecast2 + backcast2
4. **Generic block**: Processes remaining residual → forecast3
5. **Concatenate** [forecast1, forecast2, forecast3] → Linear(128×3 → 128) + LayerNorm
6. **Prediction heads**: `capacity_head` (128→1, overall fitness), `trajectory_head` (128→3, trend direction)

**Input dimension 640 = 512 (meso output) + 128 (additional monthly statistics) per month**

**`generate_synthetic_trajectories(meso_model, n_trajectories=500, n_months=7)`**: Since real multi-month data is scarce, generates synthetic 7-month trajectories by: (a) initializing random weekly states, (b) running them through the trained meso model with random activity levels and recovery speeds, (c) concatenating meso outputs with random trend statistics into monthly vectors. This data augmentation enables macro model training even with limited real long-term data.

**Checkpoint:** `checkpoints/temporal_macro.pt`

#### `src/temporal/hierarchical_fusion.py`
**What it is:** Fuses micro (256D), meso (512D), and macro (128D) temporal representations into a single 512D state.

**`HierarchicalFusion`** (inputs: `z_micro[B,256]`, `z_meso[B,512]`, `z_macro[B,128]` → output: `[B,512]`):
1. **Project all to common dimension**: Three Linear+LayerNorm layers → all 256D
2. **Top-down attention (macro → meso)**: `MultiheadAttention(4 heads)` where meso is the query and macro is key/value. Macro context guides interpretation of the weekly pattern. Result added to meso (residual connection).
3. **Bottom-up attention (meso → micro)**: The macro-informed meso attends to micro. Macro+meso context guides interpretation of recent-hour data. Result added to micro (residual).
4. **Gated combination**: Concatenate all three context-enriched vectors `[B, 768]` → sigmoid gate × concatenation → selective filtering of each scale's contribution
5. **Output projection**: Linear(768→512) + LayerNorm + ReLU

**Why bidirectional hierarchical attention?** "I ran 10km this morning" (micro) should be interpreted differently if I've been training for a marathon for 3 months (macro) vs. if I've been sedentary. The top-down then bottom-up flow propagates long-term context into short-term interpretation.

**Checkpoint:** `checkpoints/temporal_fusion.pt`

---

### 4.5 Digital Twin

#### `src/twin/bayesian_vae.py`
**What it is:** The core Digital Twin component — a β-VAE that compresses the 512D temporal state into a 10D latent health state.

**Classes:**

**`TwinEncoder`** (512 → μ[10], logvar[10]):
- MLP: 512→256 (LayerNorm+ReLU+Dropout(0.2)) → 128 (same)
- Two parallel heads: `mu_head` (Linear 128→10) outputs the mean, `logvar_head` (Linear 128→10) outputs log-variance
- **`reparameterize(mu, logvar)`**: `z = mu + exp(0.5 × logvar) × ε`, where `ε ~ N(0,1)`. This "reparameterization trick" allows gradients to flow through the stochastic sampling operation during backpropagation.

**`TwinDecoder`** (10 → 256):
- MLP: 10→64→128→256 with ReLU activations

**`PredictionHead`** (10+64 → 6):
- Concatenates the 10D latent z with a 64D activity context (from decoder) → MLP → predicts HR (1 value) + 5 HRV metrics

**`BayesianVAE`** (the complete model):
- **Forward pass**: z_temporal[512] → encoder → (μ,logvar) → reparameterize → z[10] → decoder → reconstructed[256]; also: decoder output → activity_proj → 64D context → pred_head(z, context) → predictions[6]
- **Loss function**: `L_recon + β × L_KL + L_pred` where:
  - `L_recon` = MSE between reconstructed and original[:256] (first 256 dimensions of input)
  - `L_KL` = −0.5 × mean(1 + logvar − μ² − exp(logvar)) — KL divergence from standard normal
  - `L_pred` = MSE(predicted HR, true HR) + MSE(predicted HRV, true HRV)
  - **β = 4.0**: Higher than standard VAE (β=1). Forces more **disentanglement** — pushing different health concepts (fatigue, cardiovascular fitness, recovery) into separate latent dimensions. Crucial for RL agent interpretability.

**`mc_sample(z_temporal, n_samples=100)`**: Monte Carlo uncertainty estimation. Runs the encoder 100 times, getting 100 different z samples from the learned distribution. Returns mean (best estimate) and std (uncertainty). High std = model is unsure about the health state.

**The 10 latent dimensions conceptually represent:**
| Index | Concept |
|-------|---------|
| z[0] | Physical capacity |
| z[1] | Fatigue level |
| z[2] | Recovery state |
| z[3] | Cardiovascular fitness |
| z[4] | Stability/consistency |
| z[5-9] | Other learned health facets |

**Checkpoint:** `checkpoints/twin_vae.pt`

#### `src/twin/latent_sde.py`
**What it is:** A Neural Stochastic Differential Equation that predicts how the 10D health state evolves over time given planned interventions.

**Core equation:** `dz = f(z, activity, rest, t)dt + g(z, t)dW`
- **f** = drift function (deterministic trend direction)
- **g** = diffusion function (stochastic noise magnitude)
- **dW** = Wiener process (Brownian motion)

**Classes:**

**`ActivityEncoder`** (6→64): MLP that converts raw 6D activity vector → 64D activity embedding
**`RestEncoder`** (3→32): MLP that converts raw 3D rest vector → 32D rest embedding

**`SDEFunc`** (implements the `torchsde` interface):
- `sde_type = "ito"` — Itô interpretation of the SDE
- `noise_type = "diagonal"` — independent noise per latent dimension
- **Drift net `f(t, z)`**: Concatenates [z(10), activity_embed(64), rest_embed(32), t(1)] → Linear(107→128) → Tanh → Linear(128→128) → Tanh → Linear(128→10). The Tanh activations bound the drift, preventing numerical instability.
- **Diffusion net `g(t, z)`**: Concatenates [z(10), t(1)] → Linear(11→32) → **Softplus** → Linear(32→10). Softplus ensures positive diffusion (noise magnitude must be non-negative).
- `set_context(a_t, r_t)`: Activity and rest encodings are set externally before integration begins

**`LatentNeuralSDE`** (the complete model):
- **`forward(z0, activity, rest, ts)`**: Encodes activity and rest, sets context, then calls `torchsde.sdeint(sde_func, z0, ts, method="euler", dt=0.1)` — Euler-Maruyama numerical integration. Each time step: z(t+dt) = z(t) + f(...)×dt + g(...)×√dt×ε
- **`predict_trajectory(z0, activity, rest, n_days=7, n_samples=50)`**: Runs 50 independent SDE simulations (each with different Brownian motion noise), returns mean and std across trajectories. This gives a **distributional forecast** with uncertainty bands.

**Why SDE instead of ODE?** Health evolution is inherently stochastic — the same workout produces different outcomes on different days due to sleep quality, hydration, stress, minor illness. The diffusion term models this irreducible randomness.

**Checkpoint:** `checkpoints/twin_sde.pt`

---

### 4.6 Reinforcement Learning

#### `src/rl/sac_networks.py`
**What it is:** The RL environment, actor, and critic networks for SAC (Soft Actor-Critic).

**`TwinGymEnv`** (extends `gymnasium.Env`):
- **Observation space**: Box[-5,5] shape=(20,) = [z_mu(10), z_std(10)]
- **Action space**: Box[0,1] shape=(6,) — 6 continuous action dimensions:

| Index | Meaning | Range |
|-------|---------|-------|
| 0 | Exercise intensity | 0=rest → 1=max effort |
| 1 | Session duration (normalized) | 0→1 (mapped to 0-28 days) |
| 2 | Rest/recovery hours (normalized) | 0→1 (mapped to 0-8 hours) |
| 3 | Nutrition quality | 0=poor → 1=excellent |
| 4 | Sleep consistency | 0=erratic → 1=very regular |
| 5 | Combined load proxy | intensity × duration |

- **`reset()`**: Initializes z_mu ~ N(0, 0.5²), z_std = 0.3 (uncertain), t=0
- **`step(action)`**: (a) Encode action into activity (first 6 dims) and rest (duration, sleep_consistency, 1−intensity) tensors. (b) Sample z0 from N(z_mu, z_std). (c) Run Neural SDE forward 1 day. (d) Update z_mu = z_next, z_std = 0.99×z_std + 0.01×|z_next − z_old|. (e) Compute reward. (f) Episode ends after 28 steps (days).
- **Built-in reward**: 0.3×(capacity+cardio) − 0.4×max(0, fatigue−1.5) + 0.2×recovery + 0.1×stability, with extra penalties for high intensity when fatigued or when uncertainty is high.

**`SquashedGaussianActor`** (state(20) → action(6)):
- 2 hidden layers (256 each, ReLU)
- Two output heads: `mu_head` (mean action), `log_std_head` (clamped to [-20, 2])
- **`sample(state)`**: Samples from Normal(μ, exp(log_std)), applies tanh squashing to bound actions to [-1,1], shifts to [0,1]. Computes corrected log-probability: `log_prob = log_N(x) − log(1 − tanh²(x))` (accounts for the nonlinear squashing).
- **`deterministic(state)`**: At evaluation time, returns tanh(μ) shifted to [0,1] — no exploration noise.

**`TwinCritic`** (state(20)+action(6) → Q-value):
- **Two independent Q-networks** (Q1, Q2), each: Linear(26→256) → ReLU → Linear(256→256) → ReLU → Linear(256→1)
- Taking min(Q1, Q2) prevents **overestimation bias** — a well-known failure mode in Q-learning
- `q1_forward()`: Returns only Q1 (for efficiency when Q2 isn't needed)

#### `src/rl/reward.py`
**What it is:** A multi-objective reward function that balances competing health goals.

**`MultiObjectiveReward`**:
```
Total = 0.35 × R_progress + 0.30 × R_safety + 0.20 × R_recovery + 0.15 × R_adherence
```

| Component | Weight | Formula | What It Rewards |
|-----------|--------|---------|----------------|
| R_progress | 0.35 | capacity + 0.5×cardio + 2×max(0, Δcapacity) | Improving fitness and cardiovascular health. Bonus for improvement over previous state |
| R_safety | 0.30 | −2×max(0, fatigue−1.0) −3.0 if fatigue>2.0 −max(0, uncertainty−1.0) | Penalizes overtraining, danger zones, and high model uncertainty (conservative behavior) |
| R_recovery | 0.20 | recovery×(1−intensity) + rest_hours×0.5 | Recovery when not pushing hard; rest quality |
| R_adherence | 0.15 | (1−|intensity−0.5|) + stability×0.5 | Moderate, consistent behavior. Intensity close to 0.5 is rewarded. |

**`decompose()`**: Returns individual component values for analysis/visualization.

#### `src/rl/safety.py`
**What it is:** Hard safety constraints that clip agent actions before execution.

**`SafetyBounds`** (dataclass with defaults):
- `max_intensity_when_fatigued: 0.4`, `fatigue_threshold: 1.5`
- `max_consecutive_high_days: 3`, `min_rest_hours: 0.5`
- `min_sleep_consistency: 0.5`

**`SafetyGuard`**:

| Rule | Trigger Condition | Action Taken |
|------|-------------------|-------------|
| Fatigue limit | z[1] > 1.5 | Cap intensity at 0.4, force rest ≥ 0.5 |
| Consecutive high days | 3+ days with intensity > 0.7 | Force intensity ≤ 0.3, force rest ≥ 0.6, reset counter |
| Minimum rest | Always | rest ≥ min_rest_hours/8 |
| High uncertainty | z_std.mean() > 2.0 | Cap intensity at 0.3 |
| Sleep consistency | Always | sleep_consistency ≥ 0.5 |

- **`check_and_clip(action, z_mu, z_std)`**: Applies all rules, clips to [0,1], increments violation counter
- **`compute_penalty(action, z_mu, z_std)`**: Returns a scalar penalty for the reward signal during training, so the agent learns *why* these rules exist
- **`reset()`**: Called at episode start; resets consecutive days counter and violation count

**Why both reward and safety?** The reward teaches soft preferences (gradual learning). Safety guards enforce hard constraints (immediate action). Together, the agent learns to avoid unsafe behavior while the guards catch edge cases during exploration.

---

### 4.7 Drift Detection

#### `src/drift/adwin.py`
**What it is:** ADWIN (ADaptive WINdowing) streaming concept drift detection — one detector per feature.

**`ADWINDetector`**:
- Maintains 48 independent ADWIN instances (one per feature dimension)
- Uses `river.drift.ADWIN` if available (production-grade C implementation), falls back to custom `_NumpyADWIN`

**`_NumpyADWIN`** (custom fallback):
- Maintains a sliding window (deque, max 2000 samples)
- At each update, tests every possible split point (sampled every n/20 steps for efficiency)
- Uses Hoeffding bound: if |mean_left − mean_right| > √(log(2/δ) / 2m), where m = harmonic mean of sub-window sizes and δ=0.002, drift is declared
- On drift: discards the old (left) portion of the window

**Output per update:** `{drift_detected: bool, drifted_features: [indices], n_drifted: int, timestamp: float}`

#### `src/drift/mmd.py`
**What it is:** Maximum Mean Discrepancy — a statistical two-sample test for distribution shift.

**`MMDDetector`**:
- Maintains reference buffer (500 samples) and test buffer (100 samples)
- `fit(ref_data)`: Initializes reference buffer from known-good data
- `update(feature_vec)`: Adds to test buffer; when full, computes MMD² between ref and test
- **MMD computation**: Uses RBF kernel k(x,y) = exp(−||x−y||² / 2σ²). MMD² = mean(K_xx) + mean(K_yy) − 2×mean(K_xy). Subsamples to 200 ref + 100 test for efficiency. Diagonal elements zeroed to avoid self-comparison bias.
- Threshold: 0.05 (configurable). MMD > threshold → drift detected
- **`adapt()`**: Appends test buffer to reference buffer (accepts new distribution as normal), clears test buffer

#### `src/drift/autoencoder_novelty.py`
**What it is:** Detects novel/out-of-distribution data via autoencoder reconstruction error.

**`NoveltyAutoencoder`** (nn.Module):
- Encoder: Linear(48→64) → ReLU → Linear(64→16) — compresses to 16D latent
- Decoder: Linear(16→64) → ReLU → Linear(64→48) — reconstructs
- On normal data: low reconstruction error. On novel data: high error.

**`AutoencoderNoveltyDetector`**:
- **`fit(ref_data, epochs=50)`**: Trains the autoencoder on reference data (MSE loss, Adam optimizer), then computes reconstruction errors on training data and sets threshold = 95th percentile
- **`score(feature_vec)`**: Returns reconstruction MSE for a single sample
- **`update(feature_vec)`**: Checks if error > threshold → novelty detected. Returns `{drift_detected, reconstruction_error, threshold, novelty_ratio=error/threshold}`
- **`retrain(new_data, epochs=20)`**: Re-trains on new data (for adaptation after drift)

#### `src/drift/drift_manager.py`
**What it is:** Orchestrates all three drift detectors and decides what action to take.

**`DriftDiagnoser`**:
- `FEATURE_GROUPS`: Maps feature indices to semantic groups: `imu_gait` (0-7), `imu_stats` (8-19), `cardio_hr` (20-29), `cardio_hrv` (30-39), `quality` (40-47)
- **`diagnose(adwin_result, mmd_result, ae_result)`**: Combines results, determines:
  - **Causes**: concept_drift (which features), distribution_shift (MMD value), novelty (reconstruction error)
  - **Severity** (0-1): Weighted sum of triggered detectors (ADWIN: 0.3×n_drifted/48, MMD: 0.4, AE: 0.3×novelty_ratio)
  - **Recommended action**:

| Severity | Action |
|----------|--------|
| > 0.5 | `full_retrain` — retrain model from scratch |
| 0.2 – 0.5 | `ewc_update` — update with EWC regularization |
| 0.05 – 0.2 | `adapt_reference` — just update reference distributions |
| < 0.05 | `no_action` |

**`EWCRegularizer`** (Elastic Weight Consolidation):
- **Purpose**: Prevents catastrophic forgetting when updating models on new data. Without EWC, retraining on new distribution data makes the model forget the old distribution.
- **`compute_fisher(model, data_loader)`**: Computes Fisher Information Matrix diagonal — squared gradients averaged over reference data. Measures how important each parameter is to the old data's performance.
- **`penalty(model)`**: `λ/2 × Σ F_i × (θ_i − θ*_i)²` — penalizes changes to important parameters (high Fisher value), allows changes to unimportant ones. λ=1000.0 by default.

**`DriftManager`**:
- Contains instances of all three detectors + `DriftDiagnoser` + optional `EWCRegularizer`
- **`fit_reference(ref_data)`**: Initializes MMD reference buffer and autoencoder
- **`update(feature_vec, timestamp)`**: Feeds to all 3 detectors, diagnoses, and automatically executes the recommended action (adapt reference buffers, retrain autoencoder, etc.)
- **`get_ewc_penalty(model)`**: Returns EWC penalty tensor to add to training loss
- **`get_history()`**: Returns full log of all drift events and diagnoses

---

### 4.8 Federated Learning

#### `src/federated/fedprox_server.py`
**What it is:** The central server in federated learning. Manages global model and aggregates client updates.

**`FedProxServer`**:
- **`distribute()`**: Returns a deep copy of the global model state dict to send to clients
- **`aggregate(client_states, client_weights)`**: Weighted average of client model parameters. Weights are proportional to each client's dataset size (clients with more data have more influence). Updates global model.
- **`get_proximal_term(local_model)`**: Computes `(μ/2) × Σ ||w_local − w_global||²` — the FedProx proximal penalty. This prevents any single client from straying too far from the global model, stabilizing training when clients have heterogeneous data distributions. μ=0.01.

**Why FedProx over FedAvg?** Standard FedAvg (simple averaging) works poorly when clients have very different data distributions (e.g., athletes vs. sedentary vs. heart patients). The proximal term acts as a "leash" — clients can adapt locally but can't diverge wildly.

#### `src/federated/fedper_client.py`
**What it is:** An individual client in federated learning with personal layers.

**`FedPerClient`**:
- Each client has a model split into **shared layers** (trained globally across all clients) and **personal layers** (trained only on that individual's data)
- `personal_layer_names`: List of layer name prefixes that are personal (e.g., `["personal_head"]`)

Key methods:
- **`receive_global(global_state)`**: Updates only shared layers from the server. Restores personal layers from local save.
- **`train_local(dataloader, loss_fn, proximal_fn)`**: Full local training for `local_epochs` (default 5). Includes FedProx proximal term if provided. Handles various batch formats (dict/tuple/tensor).
- **`get_shared_state()`**: Returns only shared layer parameters — this is what gets sent back to the server. Personal layers stay private.
- **`personalize(dataloader, n_epochs=10)`**: Freezes shared layers, fine-tunes only personal layers on the subject's own data. Used after global training to adapt the model to the individual. Learning rate halved (lr×0.5) for stability.

**Why FedPer?** A global model for heart rate prediction works for average physiology, but a person with bradycardia (low resting HR) or an elite athlete has fundamentally different baselines. Personal layers can adapt without poisoning the global model.

#### `src/federated/clustering.py`
**What it is:** Groups subjects with similar physiological profiles to enable cluster-aware federated learning.

**`SubjectClusterer`**:
- `n_clusters=5` — `KMeans` clustering from scikit-learn
- **`build_subject_profile(feature_windows)`**: Takes up to 50 windows, computes mean and std of first 24 features → concatenates to 48D profile. This captures "what is this person's average physiological state?"
- **`fit(subject_profiles)`**: Runs K-Means on all profiles, assigns cluster labels
- **`assign(profile)`**: Assigns a new subject to the nearest cluster centroid
- **`get_cluster_groups()`**: Returns dict mapping cluster_id → list of subject_ids

**Use case**: Different clusters could use different global models, so cluster-similar subjects benefit from each other's data without being diluted by dissimilar subjects.

---

### 4.9 Simulation & Intervention Ranking

#### `src/simulation/mc_rollout.py`
**What it is:** Monte Carlo rollout engine — simulates future health trajectories under a given intervention plan.

**`InterventionPlan`** (dataclass):
- Fields: `intensity` (0-1), `duration_days` (int), `rest_extra_hours` (float), `nutrition_quality` (0-1), `sleep_consistency` (0-1)
- **`to_activity_tensor()`**: Converts to 6D tensor: [intensity, duration/28, rest/8, nutrition, sleep, min(1, intensity×duration/14)]
- **`to_rest_tensor()`**: Converts to 3D tensor: [rest/8, sleep_consistency, 1−intensity]

**`RolloutResult`** (dataclass): Contains z_mean, z_std, z_trajectories, hr_trajectory, hrv_trajectory, risk_scores, overtraining_prob, injury_prob, peaking_day

**`MCRolloutEngine`**:
- **`rollout(z0_mean, z0_std, plan, n_days=28)`**:
  1. Sample 200 initial states from N(z0_mean, z0_std) — representing uncertainty about current state
  2. Run Neural SDE forward for n_days steps in batches of 32
  3. Collect 200 trajectory samples → compute mean and std at each time step
  4. Decode each day's mean z through VAE decoder + prediction head → HR and HRV trajectory
  5. Compute risk scores: `risk = 0.5×z_uncertainty + 0.5×|HR−70|/50`
  6. **Overtraining probability**: Fraction of trajectories where z[0] (capacity) drops below −2
  7. **Injury probability**: Fraction where z[1] (fatigue) exceeds 2
  8. **Peaking day**: Day with highest predicted HR

#### `src/simulation/what_if.py`
**What it is:** "What if I followed plan A vs. plan B?" — comparative scenario analysis.

**`WhatIfEngine`**:
- **`query(z0, plan_kwargs, n_days=28)`**: Encodes z0 through VAE to get (mu, std), runs MCRolloutEngine, computes fitness score = HR improvement + HRV improvement − risk − fatigue penalty
- **`compare_scenarios(z0, scenarios)`**: Runs query on a list of scenarios, sorts by fitness_score descending
- **`grid_search(z0, intensities, durations)`**: Exhaustively tests all combinations of intensity [0.3, 0.5, 0.7, 0.9] × duration [7, 14, 21 days]

#### `src/simulation/intervention_ranking.py`
**What it is:** Predefined intervention scenarios and plan generation.

**5 preset scenarios:**

| Name | Intensity | Duration | Rest Extra | Description |
|------|-----------|----------|------------|-------------|
| conservative | 0.3 | 21 days | +2h | Safe, slow progress |
| moderate | 0.5 | 14 days | +1h | Balanced approach |
| aggressive | 0.8 | 14 days | +0.5h | Fast but risky |
| peak_week | 0.9 | 7 days | 0h | Maximum effort |
| recovery_week | 0.2 | 7 days | +3h | Full rest |

**`rank_interventions(engine, z0)`**: Runs all presets + any custom scenarios, returns sorted by fitness score
**`top_k_interventions(ranked, k=3)`**: Filters out unsafe plans (overtraining_prob > 15% or injury_prob > 10%), returns top-k
**`build_periodized_plan(top1, n_weeks=12)`**: Creates a 12-week progressive overload plan:
- Intensity gradually increases from 70% to 100% of base
- **Every 4th week is a deload week** (intensity × 0.6) — standard sports science technique
- Rest hours gradually decrease as fitness improves

---

### 4.10 Utilities

#### `src/utils/config.py`
**What it is:** Configuration loading utilities.
- `load_configs()`: Loads and merges all 3 YAML config files via OmegaConf
- `get_device(cfg)`: Returns CUDA device if available, else CPU
- `get_processed_dir()`, `get_checkpoint_dir()`, `get_results_dir()`: Path accessors

#### `src/utils/logger.py`
**What it is:** Experiment tracking via Weights & Biases.
- `init_run(cfg, name)`: Initializes WandB run in **offline mode** (no internet needed). Logs all config as hyperparameters.
- `log_metrics(metrics, step)`: Logs metric dict at a step
- `log_model(model, name, cfg)`: Saves model state_dict to `checkpoints/{name}.pt` and logs as WandB artifact
- `save_checkpoint/load_checkpoint`: Generic checkpoint save/load
- `finish_run()`: Closes WandB run

#### `src/utils/metrics.py`
**What it is:** Standard evaluation metrics.
- `mae(pred, target)`: Mean Absolute Error
- `mse(pred, target)`: Mean Squared Error
- `pearson_r(pred, target)`: Pearson correlation coefficient
- `activity_f1/precision/recall(logits, labels)`: Macro-averaged classification metrics using scikit-learn
- `coverage_probability(mu, sigma, true, z=1.96)`: Fraction of true values within 95% CI (μ ± 1.96σ). Measures uncertainty calibration — a well-calibrated model should have ~95% coverage.

#### `src/utils/seed.py`
**What it is:** Reproducibility function.
- `set_seed(seed=42)`: Sets Python, NumPy, PyTorch, and CUDA random seeds. Also sets `torch.backends.cudnn.deterministic = True` and `benchmark = False` for exact reproducibility.

#### `src/__init__.py`
Empty package marker.

---

### 4.11 Training Scripts

#### `train/train_encoders.py`
**What it is:** Trains all 4 encoder components (IMU, Cardio, Feature, Fusion).

**`WindowDataset`**: Loads all `window_*.pt` files from processed directory, optionally filtered by subject IDs.

**`nt_xent_loss(z1, z2, temperature=0.07)`**: NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss. For a batch of B windows creating 2B augmented views:
- Computes all-pairs cosine similarity / temperature
- Masks self-similarity (diagonal)
- Positive pair = augmentations of the same window
- Loss pushes same-window pairs close, different-window pair apart
- Temperature=0.07 sharpens the distribution (harder negatives)

**`imu_augment(window)`**: Augmentation for contrastive learning — adds Gaussian noise (σ=0.01) + random crop-and-resize (interpolate a random 75-100% crop back to full length)

**Training phases:**

1. **Phase 1: IMU Self-Supervised Pre-training** (`pretrain_imu`):
   - Creates two augmented views of each batch
   - Trains SWCTNet + ProjectionHead with NT-Xent loss
   - Cosine annealing LR schedule, early stopping with patience=10
   - Saves `encoder_imu_pretrained.pt`

2. **Phase 2: IMU Supervised Fine-tuning** (`finetune_imu`):
   - Discovers actual unique classes in data (robust to partial datasets)
   - Builds label mapping (in case labels aren't 0-indexed)
   - Adds classifier head, trains 30 epochs with cross-entropy
   - Saves `encoder_imu.pt`

3. **Phase 3: Cardio Training** (`train_cardio`):
   - Trains CardioEncoder to predict HR from features[:, 20] (the hr_mean feature)
   - MSE loss, early stopping
   - Saves `encoder_cardio.pt`

4. **Phase 4: Feature Encoder + Fusion** (`train_fusion`):
   - **Freezes** IMU and Cardio encoders (requires_grad=False)
   - Trains FeatureEncoder + CrossModalFusion jointly
   - Loss: MSE(fused_output, cardio_embedding) — ensures fusion captures cardiac information
   - Saves `encoder_feature.pt`, `encoder_fusion.pt`

**Note:** The current `main()` function loads existing IMU and Cardio checkpoints and only runs Phase 4 (fusion training). Phases 1-3 are invoked from the function definitions but the actual `main()` skips to Phase 4 after attempting to load existing checkpoints.

#### `train/train_temporal.py`
**What it is:** Trains all 4 temporal model components.

**Custom Datasets:**
- **`HourlyBufferDataset`**: Creates sequences of 180 consecutive windows from each subject. Stride = buffer_len/2 for overlap. Returns `(imu_seq, cardio_seq, feat_seq, hr_seq, hrv_seq)`.
- **`DailySequenceDataset`**: Creates consecutive daily summary sequences of specified length. Returns `(input_days, target_day)`.

**Training stages:**

1. **Micro-Scale** (`train_micro`):
   - Freezes all 4 encoder models
   - Encodes each window in a 180-window sequence through the frozen encoder stack → fused embeddings
   - Trains MicroScaleModel to predict HR and HRV from the last 5 windows' average
   - MSE loss, patience=10

2. **Meso-Scale** (`train_meso`):
   - Trains MesoScaleModel to predict next-day daily summary from today's history
   - Note: seq_len=1 in practice (limited daily data from short-duration datasets)

3. **Macro-Scale** (`train_macro`):
   - Generates 500 synthetic 7-month trajectories via `generate_synthetic_trajectories(meso_model)`
   - Trains MacroScaleModel to predict physical capacity from 6-month input trajectories
   - 60 epochs, batch_size=32

4. **Hierarchical Fusion** (in `main()`):
   - Brief training on random data (20 optimization steps minimizing output L2 norm)
   - This is a lightweight initialization; real training would come from end-to-end fine-tuning

#### `train/train_twin.py`
**What it is:** Trains the Digital Twin (β-VAE + Neural SDE).

**Custom Datasets:**
- **`ZTemporalDataset`**: Loads window `.pt` files, returns `(features[48], hrv[5], hr[1])`
- **`DaySequenceDataset`**: Similar to temporal training — consecutive daily summary pairs

**Training phases:**

1. **Phase 1: β-VAE Training** (`train_vae`):
   - Input: features[48] zero-padded to 512D
   - Loss: reconstruction + β×KL + prediction (HR + HRV)
   - Cosine annealing LR, patience=10
   - Saves `twin_vae.pt`

2. **Phase 2: Neural SDE Training** (`train_sde`):
   - Freezes VAE
   - For consecutive day pairs: encode day_t → z0 via VAE, run SDE 1 step → z_pred, encode day_t+1 → z_true via VAE
   - Loss: MSE(z_pred, z_true) — learn to predict next-day latent state
   - Activity and rest are zeros (default, no specific intervention)
   - Patience=15
   - Saves `twin_sde.pt`

3. **Phase 3: Joint Fine-Tuning** (`joint_finetune`):
   - Unfreezes everything
   - Combined loss: VAE loss + 0.5×SDE loss
   - Reduced learning rate (2e-4)
   - 20 epochs
   - Saves both `twin_vae.pt` and `twin_sde.pt`

#### `train/train_rl.py`
**What it is:** Trains the SAC reinforcement learning agent.

**`ReplayBuffer`**: Circular buffer of 100K transitions (state, action, reward, next_state, done). Pre-allocated numpy arrays for efficiency.

**`soft_update(target, source, tau=0.005)`**: `target = τ×source + (1−τ)×target` — slowly tracks the online critic for stable Q-learning.

**Training loop (`train_sac`):**
1. Load trained VAE and SDE, create `TwinGymEnv`
2. Initialize actor, critic, target_critic (deep copy, frozen)
3. Initialize learnable log_alpha (SAC entropy temperature)

For each of 5000 episodes:
1. Reset environment and safety guard
2. For each of 28 steps:
   - Sample action from actor
   - Apply SafetyGuard clipping
   - Step environment
   - Compute multi-objective reward + safety penalty
   - Store transition in replay buffer
   - If buffer > warmup (1000): sample batch of 256, run SAC update:
     - **Critic update**: Bellman target = reward + γ×(min(Q1_next, Q2_next) − α×log_π_next). L2 loss on both Q1, Q2.
     - **Actor update**: Maximize min(Q1,Q2) − α×log_π
     - **Temperature update**: Adjust α to maintain entropy near target_entropy=-6
     - **Soft target update**: τ=0.005
3. Track 100-episode moving average; save best checkpoint

Saves: `rl_actor.pt`, `rl_critic.pt`

#### `train/train_federated.py`
**What it is:** Simulates federated learning across subjects.

**`SubjectDataset`**: Loads all windows for one subject.
**`FederatedModel`**: Simple model for federated training — `shared_encoder` (48→128→128→64) + `personal_head` (64→32→6)
**`_collate_fn(batch)`**: Custom batch collation — stacks features and constructs targets as (HR, HRV5) concatenation.

**Training flow:**
1. Discover available subjects (up to 249)
2. Create one `FedPerClient` per subject with a deep copy of the global `FederatedModel`
3. Build subject profiles and fit K-Means clusterer
4. For 50 rounds:
   - Server distributes global model
   - Random 30% of clients selected
   - Each selected client: receives global, trains locally (5 epochs), returns shared weights
   - Server aggregates (weighted by dataset size)
5. Saves `federated_global.pt`

---

### 4.12 Evaluation Scripts

#### `evaluate/eval_encoders.py`
**What it is:** Evaluates encoder quality.
- **LOSO (Leave-One-Subject-Out)**: For each of 10 MHEALTH subjects, evaluates IMU activity classification F1 with that subject as test set. **Target: mean F1 > 0.85**
- **HR MAE**: Cardio encoder HR prediction error across all windows. **Target: MAE < 5 BPM**
- Saves results to `results/encoder_activity_f1.csv` and `results/encoder_hr_mae.csv`

#### `evaluate/eval_temporal.py`
**What it is:** Evaluates temporal model predictions.
- **Micro**: Predicts HR from 180-window sequences. **Target: MAE < 5**
- **Meso**: Predicts next-day HR (feature index 20) from daily history. **Target: MAE < 8**
- Saves to `results/temporal_prediction_errors.csv`

#### `evaluate/eval_twin.py`
**What it is:** Evaluates Digital Twin accuracy.
- **Reconstruction**: Encodes and decodes health state vectors, measures MSE. **Target: MSE < 0.05**
- **Trajectory**: 1-day ahead SDE predictions vs. actual next-day state. Measures MAE and 95% coverage probability. **Target: MAE < 0.12, coverage > 0.90**
- Saves to `results/twin_state_reconstruction.csv` and `results/state_evolution_trajectory.csv`

#### `evaluate/eval_rl.py`
**What it is:** Evaluates the trained RL policy.
- Runs 50 deterministic episodes, reports mean/std reward, violations, component decomposition
- **Compares against baselines**: random policy (uniform random actions) and fixed moderate (always [0.5, 0.5, 0.5, 0.5, 0.7, 0.8])
- **Target: SAC reward > random reward**
- Saves to `results/rl_policy_evaluation.json`, `results/rl_baseline_comparison.json`, `results/rl_summary.csv`

#### `evaluate/eval_simulation.py`
**What it is:** Tests the simulation engine's reliability and discrimination.
- **Rollout consistency**: Same plan run 5 times — measures coefficient of variation in final HR. Low CV = deterministic enough to trust.
- **Risk calibration**: High-intensity plan (0.95 for 21 days) should have higher overtraining probability than low-intensity (0.2 for 7 days). Measures the gap.
- **Intervention ranking**: Runs all 5 presets, prints ranking table, builds 12-week plan from top result
- Saves to multiple result files

#### `evaluate/eval_federated.py`
**What it is:** Evaluates federated learning and personalization.
- **Personalization**: For each subject with ≥10 windows, compares global model MAE vs. personalized (FedPer fine-tuned) model MAE. **Target: >15% average improvement**
- **Cold-start**: Given only 20 warmup samples from a new subject, evaluates personalized model performance
- Saves to `results/federated_personalization.csv`, `results/federated_cold_start.csv`, `results/federated_summary.json`

#### `evaluate/ablation_study.py`
**What it is:** Systematically tests 12 model variants to measure each component's contribution.

| Ablation | What's Removed/Changed |
|----------|----------------------|
| full_model | Nothing — baseline |
| no_imu_encoder | IMU data removed |
| no_cardio_encoder | ECG/PPG removed |
| no_feature_encoder | Hand-crafted features removed |
| no_fusion | Cross-modal fusion removed |
| no_tcn_only_transformer | TCN removed from micro model |
| no_safety_guard | Safety constraints removed |
| no_ewc | EWC regularization removed |
| beta_vae_beta_1 | β=1 (standard VAE) |
| beta_vae_beta_8 | β=8 (over-regularized) |
| sde_no_rest_context | Rest context removed from SDE |
| no_fedper_personal | Personal layers removed |

Each runs 20 evaluation episodes. Delta from full_model shows each component's contribution.

#### `evaluate/final_report.py`
**What it is:** Master evaluation script that runs everything and produces a pass/fail scorecard.

Runs 5 evaluation suites in sequence (encoders, twin, RL, federated, ablation), checks every metric against targets:

| Metric | Target | Direction |
|--------|--------|-----------|
| imu_loso_f1 | ≥ 0.85 | Higher better |
| hr_mae_bpm | ≤ 5.0 | Lower better |
| reconstruction_mse | ≤ 0.05 | Lower better |
| trajectory_mae | ≤ 0.12 | Lower better |
| uncertainty_calibration | ≥ 0.90 | Higher better |
| sac_mean_reward | > 0.0 | Higher better |
| improvement_over_random | > 0.0 | Higher better |
| avg_personalization_improvement_pct | ≥ 15% | Higher better |

Outputs: `results/final_evaluation_report.json` with full results and scorecard.

#### `evaluate/suite_*.py` (suite_encoders, suite_twin, suite_rl, suite_federated)
**What they are:** Thin wrappers around the eval scripts that return results as dictionaries. Called by `final_report.py` for programmatic evaluation pipeline composition.

---

### 4.13 Top-Level Scripts

#### `download_datasets.py`
**What it is:** Downloads all 6 datasets from their public sources.

For each dataset:
1. Checks if already downloaded (skips if populated)
2. Downloads ZIP from URL (UCI repository, Figshare, Zenodo)
3. Extracts ZIP contents
4. Reports success/failure

Special handling:
- PAMAP2: Has a nested ZIP (`PAMAP2_Dataset.zip` inside the outer ZIP)
- PPG: Downloads individual CSV files (not a ZIP)
- MEx: Falls back to `ucimlrepo` Python package if direct ZIP fails
- CAPTURE-24: Downloads from Zenodo

Final `check_status()` prints a table of what's present/missing.

#### `preprocess_all.py`
**What it is:** A standalone preprocessing script (alternative to `run_preprocessing.py`) that handles all 6 datasets including edge cases.

Key differences from `run_preprocessing.py`:
- Handles PPG data by tiling short recordings to fill windows
- Processes Stroke Rehab by iterating patient directories and visits
- Processes CAPTURE-24 by using precomputed features or per-subject CSVs
- Generates synthetic subjects for CAPTURE-24 and MEx when real data format varies
- More robust NaN handling via `interp_nans()` helper
- Subject ID mapping: MHEALTH=1-10, PAMAP2=101-109, Stroke=400+, PPG=500+, CAPTURE-24=300+, MEx=200+

---

### 4.14 Checkpoints & Outputs

#### `checkpoints/` — Trained Model Weights

| File | Model Class | Approx. Parameters |
|------|------------|--------------------|
| `encoder_imu.pt` | SWCTNet | ~2M |
| `encoder_imu_pretrained.pt` | SWCTNet (pre-fine-tune) | ~2M |
| `encoder_cardio.pt` | CardioEncoder | ~500K |
| `encoder_feature.pt` | FeatureEncoder | ~50K |
| `encoder_fusion.pt` | CrossModalFusion | ~200K |
| `temporal_micro.pt` | MicroScaleModel | ~3M |
| `temporal_meso.pt` | MesoScaleModel | ~5M |
| `temporal_macro.pt` | MacroScaleModel | ~2M |
| `temporal_fusion.pt` | HierarchicalFusion | ~600K |
| `twin_vae.pt` | BayesianVAE | ~200K |
| `twin_sde.pt` | LatentNeuralSDE | ~100K |
| `rl_actor.pt` | SquashedGaussianActor | ~200K |
| `rl_critic.pt` | TwinCritic | ~300K |
| `federated_global.pt` | FederatedModel | ~25K |

#### `data/processed/` — Preprocessed Data
- `subject_N/windows/window_NNNNN.pt` — individual window tensors
- `subject_N/daily_summaries/day_NNN.pt` — 512D daily vectors
- `subject_N/metadata.json` — dataset name, window count

#### `results/` — Evaluation Outputs
- Various CSVs and JSONs from evaluation scripts
- `final_evaluation_report.json` — comprehensive results with pass/fail scorecard

#### `outputs/` — Hydra Run Logs
- `YYYY-MM-DD/HH-MM-SS/` — Hydra output directories with `.hydra/` config snapshots

#### `wandb/` — WandB Offline Logs
- `offline-run-*` directories containing event files, metadata, and logged artifacts

---

## 5. Core Logic & Workflows

### Workflow 1: End-to-End Training Pipeline

This is the most important workflow — how the system goes from raw data to a trained intervention planner.

**Step 1 — Data Download & Preprocessing:**
```
download_datasets.py → data/raw/{mhealth,pamap2,...}/
preprocess_all.py → data/processed/subject_*/windows/window_*.pt
                   → data/processed/subject_*/daily_summaries/day_*.pt
```

For each subject in each dataset:
1. Load raw data via dataset-specific loader
2. Apply bandpass/highpass filters to remove noise
3. Handle missing data (interpolation or discard)
4. Pad/resize to standard channel counts (9 IMU, 2 cardio)
5. Create overlapping windows (1000 samples, 500 stride)
6. Z-score normalize per subject
7. For each window: extract 48 features + 5 HRV features, assess quality, save as `.pt`
8. Aggregate windows into daily 512D summary vectors

**Step 2 — Encoder Training:**
```
train_encoders.py → checkpoints/encoder_*.pt
```

Phase 1 → 2 → 3 → 4:
1. Pre-train IMU encoder with contrastive learning (learn representations without labels)
2. Fine-tune IMU encoder for activity classification (add supervised signal)
3. Train cardio encoder on HR prediction (supervised regression)
4. Freeze encoders, train feature encoder + fusion (learn cross-modal relationships)

**Step 3 — Temporal Training:**
```
train_temporal.py → checkpoints/temporal_*.pt
```

1. Freeze encoders
2. Train micro: encode 180-window sequences → predict HR/HRV from recent activity
3. Train meso: daily summary sequences → predict next day
4. Generate synthetic multi-month trajectories; train macro on them
5. Brief hierarchical fusion initialization

**Step 4 — Digital Twin Training:**
```
train_twin.py → checkpoints/twin_*.pt
```

1. Train β-VAE: compress 512D→10D latent space with disentanglement (β=4.0)
2. Train Neural SDE: learn to predict next-day latent state from current state
3. Joint fine-tuning: unfreeze everything, train end-to-end with combined loss

**Step 5 — RL Training:**
```
train_rl.py → checkpoints/rl_*.pt
```

1. Load trained Digital Twin (VAE+SDE)
2. Create TwinGymEnv (28-day episodes)
3. Train SAC agent for 5000 episodes with:
   - Multi-objective reward (progress + safety + recovery + adherence)
   - Safety guard action clipping
   - Replay buffer (100K transitions)
   - Automatic entropy temperature tuning

**Step 6 — Federated Training:**
```
train_federated.py → checkpoints/federated_global.pt
```

1. Cluster subjects by physiological profile (K-Means, k=5)
2. Create FedPer clients (shared encoder + personal head per subject)
3. 50 rounds × 30% client sampling → FedProx aggregation
4. Each client trains locally with proximal regularization

### Workflow 2: Inference — Planning for a New User

Given a new user's sensor data (last 3 months):

```
1. Preprocess:
   Raw sensor data → window_*.pt + daily_summaries

2. Encode each window:
   IMU window → SWCTNet → h_imu[256]
   Cardio window → CardioEncoder → h_cardio[128]
   Feature vector → FeatureEncoder → h_feat[64]
   CrossModalFusion(h_imu, h_cardio, h_feat) → h_fused[128]

3. Temporal aggregation:
   180 h_fused → MicroScaleModel → z_micro[256]
   7 daily summaries → MesoScaleModel → z_meso[512]
   6 monthly summaries → MacroScaleModel → z_macro[128]
   HierarchicalFusion(z_micro, z_meso, z_macro) → z_temporal[512]

4. Digital Twin state:
   BayesianVAE.encoder(z_temporal) → μ[10], σ[10]
   Monte Carlo sampling (100 samples) → confidence-aware z

5. Drift check:
   DriftManager.update(features) → have we seen this pattern before?
   If drift: adapt or retrain

6. Personalize:
   FedPerClient.personalize(user_data) → adapt personal head

7. Simulate & rank:
   For each of 5 preset + custom scenarios:
     WhatIfEngine → MCRolloutEngine (200 × SDE trajectories)
     → RolloutResult(hr_trajectory, overtraining_prob, injury_prob)
   rank_interventions() → sorted by fitness score
   top_k_interventions() → filter unsafe (overtraining>15%, injury>10%)

8. Generate plan:
   build_periodized_plan(top1, n_weeks=12) → weekly schedule
   Output: "Week 1: intensity=0.35, +1h rest, ..."
           "Week 4: DELOAD (intensity=0.21)"
           ...
```

### Workflow 3: Drift Detection & Adaptation

When the system is deployed and receiving new data:

```
1. Each new window's 48 features → DriftManager.update()

2. Three parallel checks:
   a. ADWIN: per-feature streaming test (Hoeffding bound)
      → which individual features have shifted?
   b. MMD: distribution-level test (RBF kernel)
      → has the overall feature distribution changed?
   c. Autoencoder: reconstruction error vs. threshold
      → is this data truly novel (out-of-distribution)?

3. DriftDiagnoser combines results:
   → Identifies affected feature groups (gait, HR, HRV, quality)
   → Computes severity score (0-1)
   → Recommends action

4. Automated response:
   severity < 0.05  → no_action
   severity 0.05-0.2 → adapt_reference (update MMD reference buffer)
   severity 0.2-0.5  → ewc_update (retrain with EWC to prevent forgetting)
   severity > 0.5    → full_retrain (model is too far out-of-date)
```

---

## 6. Setup & Execution

### Prerequisites
- Python 3.10+
- NVIDIA GPU recommended (CUDA support)
- ~10 GB disk space for datasets

### Step 1: Install Dependencies
```bash
cd HUA-DTIP
pip install -r requirements.txt
```

### Step 2: Download Datasets
```bash
python download_datasets.py
```
This downloads 6 datasets (~5-8 GB total) to `data/raw/`. Check the final status table for any failures. If downloads fail, you can manually place data in the appropriate directories.

### Step 3: Preprocess Data
```bash
python preprocess_all.py
```
Or alternatively:
```bash
python -m src.preprocessing.run_preprocessing
```
This creates `data/processed/` with window `.pt` files and daily summaries for all subjects. Takes 10-30 minutes depending on hardware.

### Step 4: Train Models (in order)
```bash
# 1. Encoders (IMU → Cardio → Feature → Fusion)
python train/train_encoders.py

# 2. Temporal models (Micro → Meso → Macro → Fusion)
python train/train_temporal.py

# 3. Digital Twin (β-VAE → Neural SDE → Joint fine-tuning)
python train/train_twin.py

# 4. RL agent (5000 SAC episodes)
python train/train_rl.py

# 5. Federated learning simulation (50 rounds)
python train/train_federated.py
```

**Order matters!** Each stage depends on checkpoints from the previous stage. The scripts will attempt to load pre-existing checkpoints and skip unnecessary re-training.

### Step 5: Evaluate
```bash
# Individual evaluations
python evaluate/eval_encoders.py
python evaluate/eval_temporal.py
python evaluate/eval_twin.py
python evaluate/eval_rl.py
python evaluate/eval_simulation.py
python evaluate/eval_federated.py
python evaluate/ablation_study.py

# Or run everything + generate scorecard
python evaluate/final_report.py
```

Results are saved to `results/` as CSVs and JSONs.

### CLI Overrides (via Hydra)
Any config value can be overridden from the command line:
```bash
python train/train_encoders.py training.lr=5e-4 training.encoders.batch_size=128
python train/train_rl.py training.rl.n_episodes=10000
python train/train_twin.py device=cpu  # Force CPU training
```

---

## 7. Datasets

### 7.1 MHEALTH
- **Source**: UCI Machine Learning Repository
- **What**: 10 subjects, IMU (wrist+chest+ankle) + ECG, 50 Hz
- **Activities**: 12 classes (standing, sitting, lying, walking, climbing stairs, cycling, jogging, running, etc.)
- **Used for**: IMU encoder training (activity classification), cardio encoder (HR prediction)
- **Subject IDs in system**: 1–10

### 7.2 PAMAP2
- **Source**: UCI Machine Learning Repository
- **What**: 9 subjects (IDs 101-109), IMU (hand+chest+ankle) + HR monitor, 100 Hz (resampled to 50 Hz)
- **Activities**: 18 classes (more detailed than MHEALTH)
- **Used for**: Additional encoder training data
- **Subject IDs in system**: 101–109

### 7.3 4-Week PPG
- **Source**: Figshare
- **What**: 49 subjects, wrist PPG over 4 weeks (28 days), with HRV metrics
- **Used for**: Meso-scale temporal model (day-by-day patterns), Digital Twin (longitudinal data)
- **Why important**: Provides multi-week longitudinal data — most other datasets are single-session
- **Subject IDs in system**: 500+

### 7.4 Stroke Rehabilitation
- **Source**: Zenodo
- **What**: Stroke patients, IMU recordings across multiple visits
- **Used for**: Clinical rehabilitation testing scenarios
- **Subject IDs in system**: 400+

### 7.5 CAPTURE-24
- **Source**: Zenodo
- **What**: 151 subjects, 24-hour wrist accelerometer recordings
- **Used for**: Large-scale activity recognition, macro-scale patterns
- **Subject IDs in system**: 300+

### 7.6 MEx
- **Source**: UCI / Mendeley
- **What**: 30 subjects performing specific exercises
- **Used for**: Exercise-specific activity recognition
- **Subject IDs in system**: 200+

---

## 8. Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **IMU** | Inertial Measurement Unit — accelerometer + gyroscope (+ sometimes magnetometer). Measures linear acceleration and angular velocity. |
| **ECG** | Electrocardiogram — electrical activity of the heart. Provides R-R intervals for HRV computation. |
| **PPG** | Photoplethysmography — optical heart rate sensor (light through skin measures blood volume pulse). Used in smartwatches. |
| **HRV** | Heart Rate Variability — variation in time between consecutive heartbeats. High HRV = good health/recovery. Low HRV = stress/fatigue/overtraining. |
| **SDNN** | Standard Deviation of NN (normal-to-normal) intervals — overall HRV, reflects combined sympathetic + parasympathetic activity. |
| **RMSSD** | Root Mean Square of Successive Differences — short-term HRV, primarily reflects parasympathetic activity. |
| **LF/HF ratio** | Low Frequency to High Frequency HRV power ratio — sympathovagal balance. Higher = more stress/exercise activation. |
| **Window** | A fixed-length segment of a time series (1000 samples = 20 seconds at 50 Hz). The fundamental processing unit of the pipeline. |
| **Embedding / Latent vector** | A learned compact numerical representation that preserves meaningful information while discarding noise. |
| **Contrastive learning** | Self-supervised learning method where a model learns by comparing pairs — similar items should have similar representations, dissimilar items should have different ones. No labels needed. |
| **NT-Xent** | Normalized Temperature-scaled Cross-Entropy — the contrastive loss used in SimCLR-style pre-training. Temperature controls difficulty. |
| **Transformer** | Neural network using self-attention — each element "attends" to all others, learning which are most relevant. Quadratic in sequence length. |
| **TCN** | Temporal Convolutional Network — dilated causal convolutions for efficient sequence modeling. Linear in sequence length. |
| **Dilation** | In convolution: spacing between kernel elements. Exponentially increasing dilation gives exponentially growing receptive field. |
| **Residual connection** | Adding input directly to output (skip connection). Prevents vanishing gradients in deep networks. |
| **VAE** | Variational Autoencoder — encoder outputs a distribution (μ, σ²), not a point. Enables sampling and uncertainty quantification. |
| **β-VAE** | VAE with β > 1 in the KL divergence term. Higher β forces more disentanglement — different latent dimensions capture different factors of variation. |
| **Reparameterization trick** | z = μ + σ × ε (ε ~ N(0,1)). Makes sampling differentiable for backpropagation. |
| **KL divergence** | Measures how different two probability distributions are. In VAE: keeps latent distribution close to standard normal N(0,1). |
| **SDE** | Stochastic Differential Equation — ODE with a random noise term. Models systems with inherent randomness. |
| **Neural SDE** | SDE where drift and diffusion functions are neural networks learned from data. |
| **Euler-Maruyama** | Simplest numerical method for SDE integration: z(t+dt) = z(t) + f×dt + g×√dt×ε. |
| **Monte Carlo** | Running many random simulations to estimate a distribution. 200 SDE trajectories give a distribution over future health states. |
| **SAC** | Soft Actor-Critic — RL algorithm maximizing reward + entropy (exploration). State-of-the-art for continuous action spaces. |
| **Replay buffer** | Memory storing past (state, action, reward, next_state) tuples. Random sampling breaks temporal correlations in training. |
| **Soft target update** | target = τ×online + (1−τ)×target. Slowly tracks internet networks for stable Q-learning. |
| **ADWIN** | ADaptive WINdowing — streaming concept drift detection. Compares old and new parts of a sliding window using Hoeffding bound. |
| **MMD** | Maximum Mean Discrepancy — kernel-based two-sample test. Measures distribution difference. |
| **EWC** | Elastic Weight Consolidation — prevents catastrophic forgetting by penalizing changes to important parameters using Fisher Information. |
| **Fisher Information Matrix** | Measures parameter importance. High Fisher = parameter is critical for current task; penalize changing it. |
| **Federated Learning** | Collaborative training across multiple clients without sharing raw data. Only model weight updates are exchanged. |
| **FedAvg** | Federated Averaging — simple averaging of client model updates. |
| **FedProx** | FedAvg + proximal regularization. Prevents clients from diverging too far from the global model. Handles heterogeneous data. |
| **FedPer** | Federated Personalization — model split into shared (global) and personal (local) layers. |
| **Digital Twin** | A virtual model mirroring a real system's state, capable of simulating its future behavior. Here: a 10D latent model of a person's health. |
| **Intervention** | A planned health action: exercise intensity/duration, rest hours, nutrition quality, sleep schedule. |
| **Periodization** | Systematic variation of training load over time — build-up phases + recovery weeks. Standard sports science technique. |
| **Deload week** | A scheduled week of reduced intensity (typically every 4th week) to allow recovery and super-compensation. |
| **N-BEATS** | Neural Basis Expansion Analysis for Time Series — forecasting architecture decomposing signals into trend + seasonality + generic residual. |
| **CLS token** | Special classification token prepended to a Transformer sequence. Its output summarizes the entire sequence. |
| **Positional embedding** | Adds position information to Transformer inputs (which are inherently position-agnostic). |
| **LayerNorm** | Normalizes across features (not batch). More stable for variable sequences and small batches. |
| **BatchNorm** | Normalizes across the batch dimension. Effective for CNNs but sensitive to batch size. |
| **Weight normalization** | Reparameterizes w = g × v/‖v‖. Stabilizes TCN training. |
| **Dropout** | Randomly zeroes outputs during training (regularization). 0.3 = 30% dropped. |
| **AdaptiveAvgPool1d(1)** | Collapses any-length sequence to 1 timestep by averaging. Makes CNNs handle variable-length input. |
| **Hydra** | YAML configuration framework with CLI overrides. |
| **OmegaConf** | Structured config library underlying Hydra. |
| **WandB** | Weights & Biases — experiment tracking. Offline mode stores logs locally. |
| **LOSO** | Leave-One-Subject-Out — evaluation where one subject is the test set. Gold standard for wearable ML generalization. |
| **Coverage probability** | Fraction of true values within predicted confidence interval. A calibrated 95% CI should cover ~95% of values. |
| **Circadian rhythm** | ~24-hour biological cycle. Captured as a 64-bin activity histogram in daily summaries. |
| **Sympathetic** | "Fight-or-flight" branch of autonomic nervous system. Elevates HR, increases alertness. |
| **Parasympathetic** | "Rest-and-digest" branch. Lowers HR, promotes recovery. Reflected in high RMSSD and HF power. |

---

*This documentation covers every file, every class, every function, every algorithm, every data flow, and every design decision in the HUA-DTIP project. For questions about specific implementation details, refer to the relevant section above and cross-reference with the actual source code.*
