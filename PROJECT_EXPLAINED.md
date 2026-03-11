# HUA-DTIP — Complete Project Explanation

> **Who this document is for:** Someone with zero background in the project. After reading this, you will understand every file, every concept, every decision — what it is, why it exists, how it works, and what it produces.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Big Picture — System Architecture](#2-the-big-picture--system-architecture)
3. [Datasets](#3-datasets)
4. [Configuration Files](#4-configuration-files)
5. [Data Pipeline — Preprocessing (`src/preprocessing/`)](#5-data-pipeline--preprocessing)
6. [Encoders — Reading Raw Signals (`src/encoders/`)](#6-encoders--reading-raw-signals)
7. [Temporal Modeling — Understanding Time (`src/temporal/`)](#7-temporal-modeling--understanding-time)
8. [Digital Twin — Your Health Avatar (`src/twin/`)](#8-digital-twin--your-health-avatar)
9. [Reinforcement Learning — Planning Interventions (`src/rl/`)](#9-reinforcement-learning--planning-interventions)
10. [Drift Detection — Detecting Change (`src/drift/`)](#10-drift-detection--detecting-change)
11. [Federated Learning — Privacy-Preserving Training (`src/federated/`)](#11-federated-learning--privacy-preserving-training)
12. [Simulation & Intervention Ranking (`src/simulation/`)](#12-simulation--intervention-ranking)
13. [Utilities (`src/utils/`)](#13-utilities)
14. [Training Scripts (`train/`)](#14-training-scripts)
15. [Evaluation Scripts (`evaluate/`)](#15-evaluation-scripts)
16. [Checkpoints](#16-checkpoints)
17. [How Everything Connects — End-to-End Flow](#17-how-everything-connects--end-to-end-flow)
18. [Key Concepts Glossary](#18-key-concepts-glossary)

---

## 1. What is this project?

**HUA-DTIP** stands for **Holistic User Activity Digital Twin with Intervention Planning**.

### In plain English:

Imagine you wear a smartwatch and heart rate sensor all day. This project takes all those sensor readings and builds a **virtual model of your health** — called a *Digital Twin* — that:

1. **Understands what you are doing** (walking, running, sitting, exercising) from accelerometer/gyroscope data.
2. **Tracks your heart rate and heart health** from ECG/PPG signals.
3. **Builds a picture of your health at three time scales**: every few seconds (micro), every day (meso), every month (macro).
4. **Creates a compact numerical representation of your current health state** in a 10-dimensional "latent space" — think of it as 10 numbers that together describe your fitness, fatigue, recovery, cardiovascular health, etc.
5. **Simulates how your health state will evolve** over the coming days/weeks under different exercise/rest plans.
6. **Recommends the best intervention plan** — e.g., "Do moderate exercise for 14 days with 1 extra hour of rest per night" — using reinforcement learning.
7. **Adapts over time** when your body changes (concept drift detection).
8. **Respects privacy** by using federated learning, meaning your data never has to leave your device.

### Target Use Cases:
- Personalised fitness planning (athletes, rehabilitation patients, general wellness)
- Stroke rehabilitation monitoring
- Chronic disease management (CVD, post-COVID recovery)
- Sports science research

---

## 2. The Big Picture — System Architecture

The system is a **pipeline of 6 major stages**, each building on the previous one:

```
Raw Sensor Data
       │
       ▼
┌─────────────────────────────────────────────┐
│  Stage 1: PREPROCESSING                     │
│  Clean signals → extract features →         │
│  create time windows → daily summaries      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 2: ENCODERS                          │
│  IMU Encoder: raw motion → 256-dim vector   │
│  Cardio Encoder: ECG/PPG → 128-dim vector   │
│  Feature Encoder: 48 stats → 64-dim vector  │
│  Fusion: combine all 3 → 128-dim vector     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 3: TEMPORAL MODELING                 │
│  Micro: last 3 minutes → 256-dim state      │
│  Meso: last 7 days → 512-dim state          │
│  Macro: last 6 months → 128-dim state       │
│  Hierarchical Fusion → 512-dim state        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 4: DIGITAL TWIN                      │
│  β-VAE: compress 512-dim → 10-dim latent z  │
│  Neural SDE: evolve z forward in time       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 5: RL + SIMULATION                   │
│  SAC Agent: given current z, pick best      │
│  exercise/rest plan (6 action values)       │
│  MC Rollout: simulate 200 possible futures  │
│  Rank interventions by safety + fitness     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 6: DRIFT + FEDERATED LEARNING        │
│  ADWIN + MMD + Autoencoder Novelty:         │
│  detect when model needs re-training        │
│  FedProx + FedPer: train across 249         │
│  subjects without sharing raw data          │
└─────────────────────────────────────────────┘
```

---

## 3. Datasets

The project uses **6 public wearable health datasets**, each providing a different angle on human activity and physiology.

### 3.1 MHEALTH (`data/raw/mhealth/`)
- **What:** 10 subjects wearing IMU sensors on chest, ankle, wrist + ECG electrodes.
- **Signals:** 9-axis IMU (accelerometer + gyroscope + magnetometer), 2-channel ECG.
- **Labels:** 12 activity classes (standing, walking, running, cycling, etc.)
- **Used for:** Training the IMU encoder (activity classification) and cardio encoder (heart rate).
- **Format:** `.log` files, space-separated, one row per sample at 50 Hz.

### 3.2 PAMAP2 (`data/raw/pamap2/`)
- **What:** 9 subjects (IDs 101–109) wearing IMUs on hand, chest, ankle + heart rate monitor.
- **Signals:** 9-axis IMU per sensor + heart rate.
- **Labels:** 18 activity classes (more detailed than MHEALTH: walking, Nordic walking, stair climbing, etc.)
- **Used for:** Additional training data for IMU + cardio encoders.
- **Format:** `.dat` files, space-separated, ~100 Hz.

### 3.3 4-Week PPG (`data/raw/ppg_4week/`)
- **What:** 49 subjects wearing wrist PPG sensors over 4 weeks (28 days).
- **Signals:** PPG (photoplethysmography — optical heart rate sensor), possibly accelerometer.
- **Used for:** Training the meso-scale temporal model (day-by-day patterns) and the Digital Twin.
- **Why important:** Provides long-term longitudinal data — crucial for learning how health evolves over weeks.

### 3.4 Stroke Rehab (`data/raw/stroke_rehab/`)
- **What:** Stroke rehabilitation patients' IMU recordings across multiple visits.
- **Format:** CSV files named `patient_N_visitN.csv`.
- **Used for:** Testing the system on clinical rehabilitation scenarios.

### 3.5 CAPTURE-24 (`data/raw/capture24/`)
- **What:** 151 subjects with 24-hour wrist accelerometer recordings.
- **Used for:** Large-scale activity recognition and macro-scale temporal patterns.

### 3.6 MEx (`data/raw/mex/`)
- **What:** 30 subjects performing specific exercises with wearable sensors.
- **Used for:** Exercise-specific activity recognition.

---

## 4. Configuration Files

All hyperparameters live in `configs/`. The system uses **Hydra** (a configuration framework) to load them. You can override any value from the command line.

### `configs/data.yaml`
Defines **where data lives** and **how to process it**.
```yaml
paths:         # where raw and processed data folders are
window:
  size: 1000   # 1000 samples per window = 20 seconds at 50 Hz
  stride: 500  # windows overlap by 500 samples = 50% overlap
normalization:
  method: per_subject_zscore   # normalize each subject independently
hrv:
  features: [SDNN, RMSSD, LF, HF, LF_HF_ratio]  # 5 HRV metrics
mhealth:
  n_subjects: 10
  n_activity_classes: 12
```

### `configs/model.yaml`
Defines **every neural network's architecture**.
- `imu_encoder`: CNN channels, kernel sizes, Transformer depth.
- `cardio_encoder`: ResBlock structure.
- `feature_encoder`: MLP hidden dimensions.
- `fusion`: Cross-modal Transformer parameters.
- `micro_scale` / `meso_scale` / `macro_scale`: Temporal model sizes.
- `twin_vae`: β-VAE hidden layers, latent dimension (10), β regularisation (4.0).
- `latent_sde`: Neural SDE drift/diffusion network sizes.
- `sac`: Actor/Critic network sizes, action dimension (6), state dimension (68).
- `drift`: Drift detector thresholds.
- `federated`: Number of FL clusters, privacy budget (ε=1.0 differential privacy).

### `configs/training.yaml`
Defines **how to train** each component.
- Learning rates, batch sizes, epochs, patience (early stopping), schedulers.
- RL: 5000 episodes, 28 steps each, replay buffer of 100K transitions, discount γ=0.99.
- Federated: 249 clients, 50 rounds, 30% client sampling per round.
- WandB logging project name.

---

## 5. Data Pipeline — Preprocessing

**Location:** `src/preprocessing/`  
**Entry point:** `preprocess_all.py` (top-level) or `src/preprocessing/run_preprocessing.py`

The preprocessing pipeline converts raw sensor recordings into clean, normalised, feature-rich data windows that neural networks can learn from.

### Step-by-Step Pipeline:

#### Step 1 — Load Raw Data (`dataset_loaders.py`)

Six PyTorch `Dataset` classes, one per dataset:

| Class | Dataset | Output per sample |
|-------|---------|-------------------|
| `MHEALTHDataset` | MHEALTH `.log` files | `(imu[9], ecg[2], label)` |
| `PAMAP2Dataset` | PAMAP2 `.dat` files | `(imu[9], hr[1], label)` |
| `FourWeekPPGDataset` | PPG CSVs | `(subject_id, date, DataFrame)` — grouped by day |
| `StrokeRehabDataset` | Rehabilitation CSVs | DataFrame with patient_id and visit |
| `CAPTURE24Dataset` | 24h accelerometer CSVs | Row-level data |
| `MExDataset` | Exercise CSVs | Row-level data |

**IMU columns used:** wrist_acc_x/y/z, wrist_gyro_x/y/z, ankle_acc_x/y/z (9 channels)  
**ECG/Cardio columns:** ecg_1, ecg_2

#### Step 2 — Signal Cleaning (`signal_cleaning.py`)

Raw sensor signals are noisy. This module applies classical digital signal processing:

- **`bandpass_filter(signal, fs, low=0.5, high=20.0)`** — Butterworth band-pass filter. Keeps only frequencies between 0.5 and 20 Hz. This removes slow drift (baseline wander) and high-frequency noise. IMU signals use this.
- **`highpass_filter(signal, fs, cutoff=0.5)`** — Removes slow moving average from ECG (removes baseline wander from ECG).
- **`lowpass_filter(signal, fs, cutoff=20.0)`** — Cuts off high-frequency noise. Used in PPG motion artefact removal.
- **`resample_signal(orig_fs, target_fs)`** — Resamples to a common 50 Hz if sensors record at different rates (PAMAP2 uses 100 Hz).
- **`compute_snr(signal)`** — Computes signal-to-noise ratio in decibels. Used for quality scoring.
- **`handle_missing(signal)`** — Detects NaN values. If <10% missing: linear interpolation fixes the gaps. If >30% missing: marks the window as "discard". If >10% but more than 30% in a contiguous gap: also discard.
- **`remove_motion_artifact(ppg, acc)`** — Subtracts the low-frequency accelerometer signal (correlated with wrist movement) from the PPG, reducing motion artefacts in heart rate readings.

#### Step 3 — Windowing (`windowing.py`)

Continuous signals are split into overlapping fixed-length windows:
- **Window size:** 1000 samples = 20 seconds at 50 Hz.
- **Stride:** 500 samples = 50% overlap between consecutive windows.
- **`per_subject_zscore(windows)`** — Each subject's data is normalised independently: subtract mean, divide by std. This is important because different people have different baseline sensor readings, and different sensor placements create systematic offsets.
- **`assess_quality(window)`** — Scores each window 0–1 based on: (40%) completeness (no NaNs), (30%) signal not saturated at extremes, (30%) has sufficient variance (flat signal = useless). Windows with very low quality scores are kept but weighted less.
- **`process_subject(imu, cardio, features_fn, hrv_fn, ...)`** — Main function: runs windowing, normalisation, quality assessment, feature extraction, and saves each window as a `.pt` (PyTorch tensor dict) file.

Each saved window file contains:
```python
{
    "imu":       tensor[1000, 9],   # normalised IMU samples
    "cardio":    tensor[1000, 2],   # normalised ECG/PPG samples
    "features":  tensor[48],        # extracted statistical features
    "hrv":       tensor[5],         # HRV features (SDNN, RMSSD, LF, HF, LF/HF)
    "label":     int,               # activity class (-1 if unknown)
    "timestamp": float,             # seconds since start of recording
    "quality":   float              # quality score 0–1
}
```

#### Step 4 — Feature Extraction (`feature_extraction.py`)

From each 20-second window, 48 scalar features are extracted:

**20 IMU features** (`extract_imu_features`):
| Feature | What it measures |
|---------|-----------------|
| cadence | Step rate (steps/minute) |
| step_count | Number of steps |
| step_regularity | How regular the stride timing is (autocorrelation) |
| SMA | Signal Magnitude Area — overall movement intensity |
| MAD | Mean Absolute Deviation — movement variability |
| energy | FFT-based energy of acceleration |
| stride_length | Estimated stride amplitude |
| gait_symmetry | Left/right symmetry (correlation first/second half) |
| acc_mean (x,y,z) | Mean acceleration per axis |
| acc_std (x,y,z) | Std dev of acceleration per axis |
| gyro_mean (x,y,z) | Mean angular velocity per axis |
| gyro_std (x,y,z) | Std dev of angular velocity per axis |

**20 Cardio features** (`extract_cardio_features`):
| Feature | What it measures |
|---------|-----------------|
| hr_mean, hr_std, hr_min, hr_max, hr_range | Heart rate statistics |
| rmssd | Root Mean Square of Successive Differences (HRV — parasympathetic activity) |
| sdnn | Standard Deviation of NN intervals (overall HRV) |
| lf_power, hf_power | Low/High frequency HRV power |
| lf_hf_ratio | Sympathetic/parasympathetic balance |
| hr_reserve_pct | % of heart rate reserve used |
| recovery_rate_proxy | How far HR is from max effort |
| ppg_amplitude, ppg_rise_time, ppg_fall_time | PPG pulse morphology |
| resp_rate_proxy | Estimated breathing rate from HRV |
| spo2_proxy | Estimated oxygen saturation from PPG amplitude |
| cardiac_output_proxy | HR × amplitude |
| parasympathetic_index, sympathetic_index | Autonomic balance indices |

**8 Quality features** (`extract_quality_features`):
SNR (signal quality), completeness (fraction non-NaN), artefact fraction, baseline wander, R-peak count, beat quality.

#### Step 5 — HRV (`hrv.py`)

Computes 5 Heart Rate Variability (HRV) features from the ECG/PPG signal:
- **SDNN**: Standard deviation of RR intervals — overall HRV, reflects general health.
- **RMSSD**: Root mean square of successive RR interval differences — short-term variability, reflects parasympathetic (rest/digest) nervous system.
- **LF power**: Low-frequency power (0.04–0.15 Hz) — reflects sympathetic + parasympathetic activity.
- **HF power**: High-frequency power (0.15–0.4 Hz) — purely parasympathetic, correlates with breathing.
- **LF/HF ratio**: Sympathovagal balance — higher = more sympathetic activation (stress/exercise).

Uses `neurokit2` library if available (more accurate), falls back to a pure NumPy implementation.

#### Step 6 — Daily Summary (`daily_summary.py`)

After all windows for a day are processed, `build_daily_summary()` aggregates them into a single **512-dimensional daily vector**. This is the representation the meso-scale temporal model consumes.

The 512 dimensions are composed as:
- **[0:256]** — `micro_surrogate`: First 48 = quality-weighted mean of the 48 features; next 48 = std dev; rest padded with zeros. Represents the "average moment" of the day.
- **[256:384]** — `daily_stats`: HRV mean+std (10), cardio feature mean+std (40). Cardiovascular profile of the day.
- **[384:448]** — `context`: Step count, quality score, activity transition rate, movement intensity. Behavioural context.
- **[448:512]** — `circadian`: 64-bin histogram of activity intensity across the day (each bin = ~22 minutes). Captures the circadian rhythm pattern — when during the day the person is active.

---

## 6. Encoders — Reading Raw Signals

**Location:** `src/encoders/`

Encoders are neural networks that take raw sensor windows and produce compact, meaningful vector representations (called **embeddings** or **feature vectors**). These compact vectors capture what the model has learnt about health state.

### 6.1 IMU Encoder (`imu_encoder.py`) — `SWCTNet`

**Input:** A window of shape `[batch, 1000, 9]` (1000 timesteps, 9 IMU channels)  
**Output:** A vector of shape `[batch, 256]`

**Architecture:** Two-stage pipeline.

**Stage 1: `SlidingWindowCNNBlock`**
- Runs 3 parallel CNN branches with different kernel sizes (5, 7, and 11 samples).
- Each branch has 3 conv layers (64 → 128 → output channels), each followed by BatchNorm + ReLU.
- Different kernel sizes capture different movement frequencies:
  - Kernel 5: fast movements (quick steps, sharp turns)
  - Kernel 7: medium movements
  - Kernel 11: slower movements (posture changes)
- Each branch applies global average pooling → single vector.
- All three branches are **concatenated** → 256-dim vector.

**Stage 2: `ChannelTimeAttentionTransformer` (CTAT)**
- Takes the 256-dim CNN output and passes it through a 3-layer Transformer encoder.
- The Transformer has 8 attention heads, each attending to different "aspects" of the motion.
- Uses `norm_first=True` (pre-norm) for training stability.
- Applies final LayerNorm.

**Why this design?** CNNs are excellent at detecting local temporal patterns in motion (footstep rhythm, arm swing). The Transformer then "reasons" about the relationships between these patterns. Multi-scale kernels ensure no movement frequency is missed.

**Training:** Two-phase approach:
1. **Self-supervised pre-training with contrastive learning (NT-Xent loss):** Two augmented versions of the same window (random noise + random crop-and-resize) are encoded, and the model is trained to make their projections similar to each other while being different from all other windows in the batch. This forces the encoder to learn meaningful representations without needing labels.
2. **Supervised fine-tuning:** A linear classifier head is added and trained with cross-entropy loss to classify activities. Target: **F1 > 0.85** on MHEALTH.

**Checkpoint:** `checkpoints/encoder_imu.pt`, `checkpoints/encoder_imu_pretrained.pt`

### 6.2 Cardio Encoder (`cardio_encoder.py`) — `CardioEncoder`

**Input:** A window of shape `[batch, 1000, 2]` (1000 timesteps, 2 ECG/PPG channels)  
**Output:** A vector of shape `[batch, 128]`

**Architecture:**
- **Stem:** 1 conv layer (channels 2→64, kernel 7) + BatchNorm + ReLU.
- **3 `ResBlock1D` blocks:** Residual blocks (skip connections) with progressively more channels: 64→64, 64→128, 128→128.
  - Each ResBlock has 2 conv layers with the same kernel size + BatchNorm.
  - A skip/shortcut connection adds the input to the output, making it a "residual" block. This prevents vanishing gradients in deep networks.
- **Global Average Pooling:** Collapses the time dimension → single 128-dim vector.
- **`TemporalSelfAttention`:** Single self-attention layer (4 heads). Adds contextual information, allowing the model to attend to the most informative part of the cardiac signal.
- **HR prediction head:** A linear layer predicts heart rate from the embedding. Used as the training objective.

**Training:** Supervised regression — predicts heart rate (bpm) from the ECG/PPG window. Loss = MSE between predicted and ground truth HR. Target: **MAE < 5 bpm**.

**Checkpoint:** `checkpoints/encoder_cardio.pt`

### 6.3 Feature Encoder (`feature_encoder.py`) — `FeatureEncoder`

**Input:** A vector of shape `[batch, 48]` (the 48 hand-crafted features per window)  
**Output:** A vector of shape `[batch, 64]`

**Architecture:** 3-layer MLP (Multi-Layer Perceptron):
- 48 → 128 (LayerNorm + ReLU + Dropout 30%)
- 128 → 128 (LayerNorm + ReLU + Dropout 30%)
- 128 → 64 (ReLU)

**Why LayerNorm instead of BatchNorm?** For small features vectors and variable batch sizes, LayerNorm is more stable.

**Why this encoder?** Raw signals (IMU, ECG) contain time-series structure that CNNs/Transformers exploit. But the hand-crafted 48 features are already scalars — gait statistics, HRV metrics, quality scores. An MLP is the right tool for scalar feature vectors.

**Training:** Jointly trained with the fusion module.

**Checkpoint:** `checkpoints/encoder_feature.pt`

### 6.4 Cross-Modal Fusion (`fusion.py`) — `CrossModalFusion`

**Input:** Three embeddings: `h_imu[batch, 256]`, `h_cardio[batch, 128]`, `h_feat[batch, 64]`  
**Output:** A single fused vector `[batch, 128]`

**Architecture:**
- **Projection layers:** Each embedding is projected to a common dimension (128) via a linear layer + LayerNorm. This puts all modalities in the same "language."
- **Cross-modal Transformer:** The three projected vectors (now shape `[batch, 3, 128]`) are fed as a sequence of 3 tokens to a 2-layer Transformer encoder. Each modality can attend to the others, learning how motion, heart rate, and body statistics interrelate.
- **Weighted aggregation:** A learned linear layer produces a scalar attention weight for each of the 3 tokens, then a softmax-weighted sum produces the final fused vector.

**Why cross-modal fusion?** Motion and heart rate are related but carry different information. A slow jog and a stressful meeting might both elevate HR — but only motion data tells them apart. By fusing, the model can reason about these interactions.

**Checkpoint:** `checkpoints/encoder_fusion.pt`

---

## 7. Temporal Modeling — Understanding Time

**Location:** `src/temporal/`

Humans exist across multiple timescales simultaneously. A step happens in milliseconds. Recovery from a workout takes days. Seasonal fitness changes take months. The temporal module captures all three.

### 7.1 Micro-Scale Model (`micro_scale.py`) — `MicroScaleModel`

**What it models:** The last 3 hours of activity (~180 fused windows, each representing ~20 seconds of data).  
**Input:** A sequence `[batch, 180, 128]` of fused embeddings.  
**Output:** A single 256-dim vector representing the "current state of the last few hours."

**Architecture: TCN + Transformer hybrid**

**Stage 1: Temporal Convolutional Network (TCN)**
- `TCNBlock` contains a projection layer followed by 4 `DilatedResidualBlock`s with dilations [1, 2, 4, 8].
- Dilation = how many samples apart the conv kernel "looks." Dilation 1 = consecutive, dilation 8 = every 8th sample.
- This exponential dilation means the receptive field grows exponentially: 1, 2, 4, 8 → the model sees patterns at timescales from 1 window (20s) to 80+ windows (26+ minutes).
- Each block uses **weight normalisation** for training stability and has a skip connection.

**Stage 2: ViT-style Transformer with CLS token**
- A special learnable `[CLS]` token is prepended to the sequence.
- Learnable positional embeddings are added to each position.
- A 3-layer Transformer encoder (8 heads) processes the full sequence.
- The final `[CLS]` token output is the representation — it has attended to the entire sequence and summarized it.

**Prediction heads:** HR head (256 → 1) and HRV head (256 → 5) to predict next heart rate and HRV from the recent activity pattern.

**Why TCN + Transformer?** TCN efficiently captures local temporal dependencies without the quadratic cost of full self-attention. The Transformer then globally aggregates the patterns the TCN has detected.

**Checkpoint:** `checkpoints/temporal_micro.pt`

### 7.2 Meso-Scale Model (`meso_scale.py`) — `MesoScaleModel`

**What it models:** The last 7 days of daily summaries.  
**Input:** A sequence `[batch, 7, 512]` of daily summary vectors (the 512-dim outputs from `daily_summary.py`).  
**Output:** A single 512-dim vector representing the "weekly wellness state."

**Architecture: Transformer with recency bias**
- **Day-of-week embedding:** A learned 512-dim vector for each of the 7 days of the week. Monday's pattern is inherently different from Sunday's (work vs. rest). This is added to each position.
- **Position embedding:** An additional learnable embedding for position 0–6 within the 7-day window.
- **LayerNorm** on the input before the transformer.
- **4-layer Transformer Encoder** with 8 heads and 1024-dim feedforward layers.
- **Recency-weighted average:** Instead of taking just the final position, the model computes a weighted sum of all 7 outputs, where recent days get higher weights: `[0.05, 0.07, 0.10, 0.12, 0.16, 0.20, 0.30]`. Day 7 (most recent) gets 30% weight, day 1 gets 5%. This gives a "recent trend–aware" daily summary.

**Prediction heads:** `next_day_head` predicts tomorrow's daily summary vector; `capacity_head` predicts a single "physical capacity" score.

**Checkpoint:** `checkpoints/temporal_meso.pt`

### 7.3 Macro-Scale Model (`macro_scale.py`) — `MacroScaleModel`

**What it models:** The last 6 months of monthly health trajectories.  
**Input:** A sequence `[batch, 6, 640]` of monthly vectors (640 = 512 meso + 128 additional stats).  
**Output:** A single 128-dim vector representing the "long-term fitness trajectory."

**Architecture: N-BEATS inspired decomposition**

N-BEATS is a neural basis expansion architecture for forecasting. The model decomposes the signal into:
1. **Trend block:** captures upward/downward long-term trend (e.g., fitness gradually improving).
2. **Seasonality block:** captures periodic patterns (e.g., performance dips every 4th week = recovery week).
3. **Generic block:** learns any residual patterns.

Each `NBEATSBlock` has:
- 3 FC layers with ReLU: input → 256 → 256 → 256.
- **Backcast head:** Tries to reconstruct the input (used as a residual).
- **Forecast head:** Projects to the 128-dim output.

The macro model processes them in sequence:
1. Trend block produces forecast1 + backcast1.
2. Input - backcast1 = residual1.
3. Seasonality block processes residual1 → forecast2 + backcast2.
4. Residual2 → Generic block → forecast3.
5. Concatenate [forecast1, forecast2, forecast3] → project to 128.

**Why macro modelling?** Fitness is a long-term process. A runner's VO2max takes months to change. By modelling 6 months, the system can distinguish "temporary fatigue" from "long-term decline."

**Checkpoint:** `checkpoints/temporal_macro.pt`

### 7.4 Hierarchical Fusion (`hierarchical_fusion.py`) — `HierarchicalFusion`

**Input:** Three temporal vectors: `z_micro[batch, 256]`, `z_meso[batch, 512]`, `z_macro[batch, 128]`  
**Output:** A single unified state vector `[batch, 512]`

**Architecture:**
- All three scales are projected to the same dimension (256 via linear + LayerNorm).
- **Top-down attention:** Macro context attends to Meso (macro guides how to interpret the weekly pattern).
- **Bottom-up attention:** Macro-informed meso context attends to Micro (the week's context guides how to interpret the last few hours).
- **Gated combination:** A sigmoid gate is applied to the concatenation of all three scales, allowing the model to learn which scale is most informative in each context.
- **Output projection:** Projects 3×256 = 768 → 512.

**Why hierarchical + bidirectional attention?** "I ran 10km this morning" (micro) should be interpreted differently if I've been training for a marathon for 3 months (macro) vs. if I've been sedentary. The hierarchical attention captures these cross-scale dependencies.

**Checkpoint:** `checkpoints/temporal_fusion.pt`

---

## 8. Digital Twin — Your Health Avatar

**Location:** `src/twin/`

The Digital Twin is the core innovation of the project. It takes the 512-dim temporal state and compresses it into a compact 10-dimensional **latent health state** `z`, then learns to predict how `z` evolves over time.

### 8.1 Bayesian β-VAE (`bayesian_vae.py`) — `BayesianVAE`

**What it does:** Compresses the 512-dim temporal representation into a 10-dim latent space `z` — like compressing a 512-word description of your health into 10 key numbers.

**Input (to encoder):** `z_temporal` of shape `[batch, 512]`  
**Output:** `z` of shape `[batch, 10]` — the **health state vector**

**Architecture breakdown:**

**`TwinEncoder`:**
- MLP: 512 → 256 (LayerNorm + ReLU + Dropout) → 128 (LayerNorm + ReLU + Dropout).
- Two parallel linear heads: `mu_head` (outputs mean μ) and `logvar_head` (outputs log-variance log σ²).
- **Reparameterization trick:** Instead of sampling directly (which is not differentiable), compute: `z = μ + σ × ε` where `ε ~ N(0,1)`. This allows gradients to flow through the sampling operation.

**`TwinDecoder`:**
- MLP: 10 → 64 → 128 → 256.
- Reconstructs a 256-dim representation from the 10-dim latent z.

**`PredictionHead`:**
- Concatenates `z[10]` with an activity context vector `[64]` → MLP → 6-dim output.
- Predicts: HR + 5 HRV metrics.

**Loss function (combined):**
```
Total Loss = L_reconstruction + β × L_KL + L_prediction

L_reconstruction = MSE(reconstructed, original[:256])   # Can it recreate the health state?
L_KL = -0.5 × mean(1 + log σ² - μ² - σ²)             # Keep latent space "well-behaved"
L_prediction = MSE(predicted HR, true HR) + MSE(predicted HRV, true HRV)
```

**What is β?** β=4.0 is the β-VAE hyper-parameter. Higher β forces more **disentanglement** — it pushes different health dimensions (fatigue, cardiovascular fitness, recovery) into separate dimensions of `z`. β=1 is a standard VAE; β>1 encourages a cleaner, more interpretable latent space. This is crucial for the RL agent: the 10 dimensions of `z` should correspond to interpretable health concepts.

**Bayesian uncertainty estimation via Monte Carlo sampling (`mc_sample`):**
- Runs the encoder 100 times on the same input.
- Gets 100 different `z` samples (because the encoder outputs a distribution, each forward pass samples differently).
- Returns mean and standard deviation across the 100 samples.
- The standard deviation = **uncertainty**. If the model is confident, std is small. If the health state is unusual/out-of-distribution, std is large.

**Checkpoint:** `checkpoints/twin_vae.pt`

**The 10 latent dimensions conceptually represent:**
- `z[0]` → Physical capacity
- `z[1]` → Fatigue level
- `z[2]` → Recovery state
- `z[3]` → Cardiovascular fitness
- `z[4]` → Stability/consistency
- `z[5]–z[9]` → Other learned health facets

### 8.2 Latent Neural SDE (`latent_sde.py`) — `LatentNeuralSDE`

**What it does:** Given the current health state `z` and a planned intervention (exercise intensity, rest, nutrition, sleep), predicts how `z` will evolve over the coming days.

**Why an SDE (Stochastic Differential Equation)?** Health evolution is not deterministic. The same workout done on two different days produces slightly different outcomes — due to sleep quality the night before, hydration, stress, minor illness, etc. An SDE models this **randomness explicitly**:

```
dz = f(z, activity, rest, t) dt + g(z, t) dW
```
- `f` = **drift function**: the general trend/direction of change (deterministic).
- `g` = **diffusion function**: the magnitude of random fluctuations.
- `dW` = Wiener process (Brownian motion) = small random noise added at each step.

**Architecture:**

**`ActivityEncoder`**: Converts 6-dim activity vector (intensity, duration, ...) → 64-dim activity embedding.  
**`RestEncoder`**: Converts 3-dim rest vector (rest hours, sleep consistency, not-intensity) → 32-dim rest embedding.

**`SDEFunc`** (the trainable SDE):
- **Drift net `f`**: Takes `[z(10) ‖ activity_embed(64) ‖ rest_embed(32) ‖ t(1)] → [128] tanh → [128] tanh → [10]`. Predicts the direction of health change.
- **Diffusion net `g`**: Takes `[z(10) ‖ t(1)] → [32] Softplus → [10]`. Predicts how much random variation to add. Softplus ensures positivity (diffusion must be non-negative).
- Context (activity + rest) is set externally via `set_context()` before integration.

**Integration:** Uses `torchsde.sdeint` with Euler–Maruyama method — the simplest numerical SDE solver. At each time step `dt=0.1`, it computes: `z(t+dt) = z(t) + f(...) × dt + g(...) × √dt × ε`.

**`predict_trajectory`:** Runs 50 independent Monte Carlo simulations of the SDE, each giving a different trajectory due to the random noise. Returns mean and std across all trajectories → **distributional forecast** with uncertainty bands.

**Checkpoint:** `checkpoints/twin_sde.pt`

---

## 9. Reinforcement Learning — Planning Interventions

**Location:** `src/rl/`

Given the Digital Twin (the 10-dim health state and its predicted evolution), the RL agent must choose the best daily intervention plan to maximise long-term health outcomes safely.

### 9.1 The Environment (`sac_networks.py`) — `TwinGymEnv`

**What it models:** A 28-day episode. Each day, the agent picks an action; the Digital Twin simulates what happens to health state next.

**Observation space:** `[z_mu (10), z_std (10)] = 20-dim vector`. The agent sees both the current health mean and its uncertainty. High uncertainty should make the agent more conservative.

**Action space:** `[0,1]^6` — 6 continuous action dimensions:
| Index | Meaning |
|-------|---------|
| action[0] | Exercise intensity (0=rest, 1=max) |
| action[1] | Session duration (normalised to 28 days) |
| action[2] | Rest/recovery hours |
| action[3] | Nutrition quality |
| action[4] | Sleep consistency |
| action[5] | Combined load proxy (intensity × duration) |

**Step logic:**
1. The agent picks an action.
2. Activity and rest tensors are derived from the action.
3. The Neural SDE evolves `z` by 1 day: `z_next = SDE(z_current, activity, rest, [0,1])`.
4. The next observation is `[z_next_mu, z_next_std]`.
5. Update running uncertainty: `z_std = 0.99 × z_std + 0.01 × |z_next - z_old|`.

**Built-in reward (simple version):** Combines capacity, fatigue, recovery, cardiac, and stability from `z_mu`. Imposes large penalties for high intensity when fatigued.

### 9.2 Reward Function (`reward.py`) — `MultiObjectiveReward`

**What it does:** Computes a scalar reward signal from the resulting health state and action. Multi-objective because health is not one-dimensional.

```
Total reward = 0.35 × R_progress
             + 0.30 × R_safety      (negative — penalises fatigue/injury)
             + 0.20 × R_recovery
             + 0.15 × R_adherence
```

**Components:**
- **R_progress**: `z[0] (capacity) + 0.5 × z[3] (cardio)`. Improvement over previous state (×2 bonus if improving).
- **R_safety**: `-2 × max(0, fatigue−1.0)` (linear penalty for overtraining); `-3.0` if fatigue > 2.0 (hard danger zone); `-max(0, uncertainty−1.0)` (penalises high uncertainty → conservative behaviour).
- **R_recovery**: `z[2] (recovery) × (1 − intensity) + rest_hours × 0.5`. Recovery is rewarded when the agent isn't pushing hard.
- **R_adherence**: How close intensity is to 0.5 (moderate); plus a stability bonus.

### 9.3 Safety Guard (`safety.py`) — `SafetyGuard`

**What it does:** Hard constraints that clip or modify the agent's action before it's applied. The RL reward teaches "soft" preferences; safety guards enforce "hard" rules.

| Rule | Condition | Action |
|------|-----------|--------|
| Fatigue limit | If `z[1] > 1.5` | Cap intensity at 0.4; increase rest to ≥0.5 |
| Consecutive hard days | If 3+ consecutive high-intensity days | Force intensity ≤ 0.3, reset counter |
| Minimum rest | Always | Rest ≥ `min_rest_hours / 8` |
| Uncertainty too high | If `z_std.mean() > 2.0` | Cap intensity at 0.3 |
| Sleep consistency | Always | Sleep consistency ≥ 0.5 |

The guard also computes a safety **penalty** added to the reward during training (so the agent learns *why* these rules exist, not just what they are).

### 9.4 SAC Actor-Critic (`sac_networks.py`)

**SAC = Soft Actor-Critic** — a state-of-the-art off-policy RL algorithm for continuous action spaces. "Soft" means it maximises reward + entropy (exploration bonus).

**`SquashedGaussianActor`:**
- Input: state (20-dim).
- 2 hidden layers (256 neurons each, ReLU).
- Two output heads: `mu_head` (mean action) and `log_std_head` (action variability).
- Actions are sampled from the Gaussian, then squashed through tanh (→ range [-1,1]), then shifted to [0,1].
- The squashing means probabilities are computed with a correction: `log_prob = log p(x) - log(1 - tanh²(x))`.
- `deterministic()`: used at evaluation time — just take the mean action.

**`TwinCritic`:**
- Two independent Q-networks (Q1, Q2), both taking state+action concatenated (20+6=26 dim) → 256 → 256 → 1.
- Having two critics and taking the **minimum** Q-value prevents overestimation bias (a common failure mode in RL).

**Training loop (in `train_rl.py`):**

SAC uses a **replay buffer** — a cyclic memory of 100K past transitions `(state, action, reward, next_state, done)`. Training updates:

1. **Critic update:** `Q_target = reward + γ × (min(Q1_next, Q2_next) − α × log_π_next)`. Train critics to minimise MSE against this target.
2. **Actor update:** Maximise `mean(min(Q1,Q2) − α × log_π)` — do actions that are both high-value and diverse.
3. **Temperature α update:** Automatically adjusted to keep entropy near `target_entropy = -6`. This balances exploration vs. exploitation.
4. **Soft target update:** `target_critic ← 0.005 × critic + 0.995 × target_critic`. Slowly tracks the live critic for stability.

---

## 10. Drift Detection — Detecting Change

**Location:** `src/drift/`

Bodies change. A model trained on your summer data may fail in winter. Drift detection monitors whether the incoming data is still "normal" relative to what the model was trained on.

### 10.1 ADWIN (`adwin.py`) — `ADWINDetector`

**ADWIN = ADaptive WINdowing**. A streaming algorithm for concept drift detection.

**Idea:** It maintains a sliding window of observations and continually checks whether the distribution has changed between the "old" part and the "new" part of the window.

**How it works:**
- One `ADWINDetector` instance manages 48 independent ADWIN detectors, one per feature.
- For each incoming feature vector, it updates all 48 detectors.
- Each detector uses the Hoeffding bound: if the difference in means between any two sub-windows exceeds `√(log(2/δ) / 2m)` (where m is the harmonic mean of the two sub-window sizes and δ=0.002 is the confidence), drift is declared.
- When drift is detected, the old part of the window is discarded (the detector "forgets" the past distribution).

**Uses `river` library** if available (faster, production-grade); falls back to a pure NumPy implementation.

**Output per sample:**
```python
{
    "drift_detected": True/False,
    "drifted_features": [list of feature indices],
    "n_drifted": int
}
```

### 10.2 MMD Detector (`mmd.py`) — `MMDDetector`

**MMD = Maximum Mean Discrepancy**. A statistical test for detecting whether two distributions are different.

**Idea:** Maintain a large reference window (500 samples of "normal" behaviour) and a small test window (100 samples of recent behaviour). If their MMD statistic is above a threshold, the distributions have shifted.

**MMD computation:**
```
MMD² = E[k(x,x')] + E[k(y,y')] - 2×E[k(x,y)]
```
Where `k(x,y) = exp(-||x-y||² / 2σ²)` is the RBF kernel. Intuitively, MMD measures how different the average "kernel similarities within group X" vs. "between groups X and Y" vs. "within group Y" are. If X and Y come from the same distribution, these should all be equal → MMD≈0.

**`adapt()`:** When action is "adapt_reference", the test buffer is appended to the reference buffer (the model accepts the new distribution as normal).

### 10.3 Autoencoder Novelty Detector (`autoencoder_novelty.py`)

**Idea:** Train an autoencoder on "normal" data. When novel/unusual data arrives, the autoencoder will fail to reconstruct it well → high reconstruction error = novelty/drift.

- **`NoveltyAutoencoder`**: input(48) → encoder → latent(16) → decoder → reconstruction(48). A small MLP autoencoder.
- **Fitting**: Trained on reference data for 50 epochs. The reconstruction errors on training data are stored. The 95th percentile of these errors becomes the **novelty threshold**.
- **Detection**: New sample → reconstruction error > threshold → drift detected.
- **`retrain()`**: When adaptation is needed, retrain on new data and update the threshold.

### 10.4 Drift Manager (`drift_manager.py`) — `DriftManager` + `DriftDiagnoser` + `EWCRegularizer`

**`DriftDiagnoser`**: Combines results from all three detectors and determines:
- **What type of drift**: concept drift (ADWIN, feature-specific), distribution shift (MMD), novelty (autoencoder).
- **Which feature groups are affected**: `imu_gait`, `imu_stats`, `cardio_hr`, `cardio_hrv`, `quality` (based on which indices drifted in ADWIN).
- **Severity score** (0–1): sum of weighted detections.
- **Recommended action**:
  | Severity | Action |
  |----------|--------|
  | > 0.5 | `full_retrain` — retrain model from scratch |
  | 0.2–0.5 | `ewc_update` — update with EWC regularisation |
  | 0.05–0.2 | `adapt_reference` — just update the reference distribution |
  | < 0.05 | `no_action` |

**`EWCRegularizer` (Elastic Weight Consolidation):**
EWC solves **catastrophic forgetting** — when a neural network is updated on new data, it tends to "forget" the old data's patterns. EWC prevents this by:
1. Computing the **Fisher Information Matrix** (how important each parameter is to the old data) using the squared gradients.
2. Adding a penalty term to the new training loss: `λ/2 × Σ F_i × (θ_i − θ_star_i)²` where θ_star are the old optimal parameters.
3. This "elastic" regularisation lets the model adapt to new data while preserving performance on old data.

**`DriftManager.update(feature_vec)`**: The main entry point — feeds a feature vector to all three detectors, diagnoses drift, applies recommended action automatically, and logs diagnostic history.

---

## 11. Federated Learning — Privacy-Preserving Training

**Location:** `src/federated/`

Different users have different physiological profiles. Ideally, training on all users' data gives the best model — but sharing raw health data raises serious privacy concerns. Federated learning solves this by training collaboratively *without* sharing raw data.

**The protocol:**
1. A central server holds a global model.
2. Each client (each person, each device) has only their own data.
3. Each round: server sends the global model to a random subset of clients.
4. Each client trains the model locally on their own data.
5. Clients send back only the **weight updates** (not the data).
6. Server aggregates updates → new global model.

### 11.1 FedProx Server (`fedprox_server.py`) — `FedProxServer`

**`FedProx` = Federated Proximal** — an extension of FedAvg (Federated Averaging) that handles **data heterogeneity** (different users have very different data distributions).

**Standard FedAvg:** Simply average all client model updates.  
**Problem:** If one client has only running data and another only uses a wheelchair, simple averaging can pull the model in conflicting directions.  
**FedProx fix:** Add a proximal term to each client's local loss:
```
L_client = L_local + (μ/2) × ||w_local - w_global||²
```
This prevents any client from straying too far from the global model, stabilising training. μ=0.01 (a gentle regularisation).

**`distribute()`**: Sends a deep copy of the global model state to all clients.  
**`aggregate(client_states, client_weights)`**: Weighted average of client model updates (weighted by dataset size — clients with more data have more influence).

### 11.2 FedPer Client (`fedper_client.py`) — `FedPerClient`

**`FedPer` = Federated Personalisation** — each client has both shared layers (trained globally) and personal layers (trained only on that person's data).

**Why?** A global model for heart rate prediction might work decently for everyone, but a person with bradycardia (low resting HR) has fundamentally different baselines. Personal layers can adapt to individual physiology without poisoning the global model.

**Layer division in `FederatedModel`:**
- **Shared:** `shared_encoder` — the feature extraction (48→128→64). This learns general health representations.
- **Personal:** `personal_head` — the prediction head (64→32→6). This learns person-specific output patterns.

**`receive_global(global_state)`**: Updates only shared layers from the server. Personal layers are restored from the local save.  
**`train_local(dataloader, loss_fn, proximal_fn)`**: Trains all layers locally (with FedProx proximal term).  
**`get_shared_state()`**: Returns only the shared layers' parameters — what gets sent back to the server.  
**`personalize(dataloader, n_epochs=10)`**: Fine-tunes only the personal layers on the subject's own data (shared layers are frozen). Used after global training to adapt to the individual.

### 11.3 Subject Clustering (`clustering.py`) — `SubjectClusterer`

**Purpose:** With 249 subjects having very different profiles, a single global model may converge poorly. Clustering subjects into groups allows different clusters to have different global models.

**Algorithm:** K-means clustering (k=5 clusters) on subject **profiles**.

**Subject profile construction (`build_subject_profile`):**
- Take up to 50 windows from a subject.
- Stack their 48 feature vectors.
- Compute mean and std across time for the first 24 features.
- Concatenate → 48-dim profile.

This profile captures: "what is this person's average physiological state?" — a mix of their typical activity level, HR, HRV, gait patterns.

**`assign(profile) → cluster_id`**: Assigns a new subject to the nearest cluster centroid. This can be used to select which cluster's global model to start from.

---

## 12. Simulation & Intervention Ranking

**Location:** `src/simulation/`

Given a trained Digital Twin, this module answers the question: **"What would happen to my health if I followed plan A vs. plan B?"**

### 12.1 MC Rollout Engine (`mc_rollout.py`)

**`InterventionPlan`** (dataclass):
```python
intensity: float        # 0–1 exercise intensity
duration_days: int      # how many days to apply
rest_extra_hours: float # extra sleep/rest per day
nutrition_quality: float # diet quality
sleep_consistency: float # regular sleep schedule
# → converted to 6-dim activity tensor + 3-dim rest tensor
```

**`MCRolloutEngine.rollout(z0_mean, z0_std, plan, n_days=28) → RolloutResult`:**

1. Sample 200 initial states from `N(z0_mean, z0_std)` — representing uncertainty about the current state.
2. Run the Neural SDE forward for `n_days` steps for all 200 in batches of 32.
3. Collect all 200 trajectory samples → shape `[n_days+1, n_samples, 10]`.
4. Compute mean and std across samples at each time step.
5. Decode each day's mean z through the VAE decoder + prediction head → HR and HRV trajectory.
6. Compute risk scores at each time step: `risk = 0.5 × z_uncertainty + 0.5 × |HR - 70| / 50`.
7. **Overtraining probability:** fraction of trajectories where `z[0]` (capacity) < -2 at any time.
8. **Injury probability:** fraction of trajectories where `z[1]` (fatigue) > 2 at any time.
9. **Peaking day:** the day with highest predicted HR.

**`RolloutResult`:** Contains `z_mean, z_std, z_trajectories, hr_trajectory, hrv_trajectory, risk_scores, overtraining_prob, injury_prob, peaking_day`.

### 12.2 What-If Engine (`what_if.py`) — `WhatIfEngine`

**`WhatIfEngine.query(z0, plan_kwargs, n_days) → ScenarioResult`:**
1. Encode `z0` through VAE encoder to get μ and σ.
2. Run `MCRolloutEngine.rollout()`.
3. Compute **fitness score**: HR improvement + HRV improvement − risk − fatigue penalty.
4. Return a `ScenarioResult` with all key metrics.

**`compare_scenarios(z0, scenarios)`**: Runs `query` on a list of scenario dictionaries, sorts by `fitness_score` descending.

**`grid_search(z0, intensities, durations)`**: Exhaustively tests all combinations of intensity [0.3, 0.5, 0.7, 0.9] × duration [7, 14, 21 days].

### 12.3 Intervention Ranking (`intervention_ranking.py`)

**5 preset scenarios:**
| Name | Intensity | Duration | Rest | Description |
|------|-----------|----------|------|-------------|
| conservative | 0.3 | 21 days | +2h extra | Safe, slow progress |
| moderate | 0.5 | 14 days | +1h | Balanced approach |
| aggressive | 0.8 | 14 days | +0.5h | Fast but risky |
| peak_week | 0.9 | 7 days | 0h extra | Maximum effort |
| recovery_week | 0.2 | 7 days | +3h | Full rest and recovery |

**`rank_interventions(engine, z0)`**: Runs all 5 presets (plus any custom ones), returns sorted list by fitness score.

**`top_k_interventions(ranked, k=3)`**: Filters out unsafe plans (overtraining_prob > 15% or injury_prob > 10%), returns top-k from what remains.

**`build_periodized_plan(top1, n_weeks=12)`**: Takes the best-ranked intervention and creates a **12-week progressive overload plan**:
- Intensity gradually increases from 70% to 100% of the base intensity.
- Every 4th week is automatically set to 60% intensity (recovery/deload week — a standard strength training technique).
- Rest hours gradually decrease as intensity increases (as fitness improves, less recovery is needed on average days).

---

## 13. Utilities

**Location:** `src/utils/`

### `config.py`
- **`load_configs()`**: Loads and merges all 3 YAML config files into a single OmegaConf object.
- **`get_device(cfg)`**: Returns `torch.device("cuda")` if GPU available, otherwise `"cpu"`.
- Helper functions for getting checkpoint and results directory paths.

### `logger.py`
- **`init_run(cfg, name)`**: Initialises a WandB (Weights & Biases) run for experiment tracking. Runs in `offline` mode by default — logs are saved locally in `wandb/` and can be synced later.
- **`log_metrics(metrics, step)`**: Logs a dictionary of `{metric_name: value}` at a given training step.
- **`log_model(model, name, cfg)`**: Saves the model's `state_dict()` to `checkpoints/{name}.pt` and logs it as a WandB artifact.
- **`save_checkpoint / load_checkpoint`**: Save/load full checkpoints.
- **`finish_run()`**: Closes the WandB run.

### `metrics.py`
Standard evaluation metrics:
- `mae(pred, target)`: Mean Absolute Error.
- `mse(pred, target)`: Mean Squared Error.
- `pearson_r(pred, target)`: Pearson correlation coefficient.
- `activity_f1(logits, labels)`: Macro-averaged F1 for activity classification.
- `activity_precision / recall`: Precision/Recall for activity classification.
- `coverage_probability(mu, sigma, true, z=1.96)`: Fraction of true values that fall within the 95% confidence interval — measures uncertainty calibration.

### `seed.py`
Sets random seeds (Python, NumPy, PyTorch, CUDA) for reproducibility.

---

## 14. Training Scripts

**Location:** `train/`

### `train_encoders.py`

**Datasets used:** All preprocessed `window_*.pt` files.

**Phase 1 — IMU Self-Supervised Pre-training:**
- Augments each window twice (Gaussian noise + random crop/resize).
- Trains `SWCTNet` + `ProjectionHead` with NT-Xent loss (contrastive) for up to 100 epochs.
- Cosine annealing learning rate schedule.
- Early stopping with patience=10.
- **NT-Xent loss explained:** For a batch of B windows, create 2B augmented views. For pair (i, i+B), the loss pushes them together in projection space while pushing all other pairs apart. Temperature=0.07 sharpens the distribution (lower temperature = harder negatives).

**Phase 2 — IMU Supervised Fine-tuning:**
- Adds a classification head and trains for 30 epochs with cross-entropy loss.
- Handles unknown label (-1) by filtering out unlabelled samples.
- Auto-discovers actual number of unique classes in the data (robust to partial datasets).

**Phase 3 — Cardio Training:**
- Trains `CardioEncoder` to predict heart rate from ECG/PPG.
- MSE loss, early stopping.

**Phase 4 — Feature Encoder + Fusion:**
- Freezes IMU + Cardio encoders.
- Trains `FeatureEncoder` + `CrossModalFusion` jointly.
- MSE loss: fused output should be close to cardio embedding (ensures fusion is informative).

### `train_temporal.py`

**Datasets used:** `HourlyBufferDataset` (180-window sequences) for micro; `DailySequenceDataset` for meso/macro.

**Train micro:** Encodes each window in the sequence through the (frozen) encoder stack, passes the 180-window sequence through `MicroScaleModel`, and trains HR + HRV prediction.

**Train meso:** Trains `MesoScaleModel` to predict tomorrow's daily summary from today's 7-day history.

**Train macro:** Generates 500 synthetic 7-month trajectories using the trained meso model (see `generate_synthetic_trajectories`). Then trains `MacroScaleModel` on these synthetic trajectories to predict physical capacity.

### `train_twin.py`

**Phase 1 — VAE training:** Trains `BayesianVAE` on feature vectors padded to 512-dim. Optimises reconstruction + KL + prediction loss.

**Phase 2 — SDE training:** Freezes VAE. Trains `LatentNeuralSDE` to predict the next day's latent state from today's. Uses consecutive day pairs from daily summaries.

**Phase 3 — Joint fine-tuning:** Unfreezes everything and jointly trains VAE + SDE with a combined loss, learning rate reduced to `2e-4`.

### `train_rl.py`

Implements the full **SAC training loop** with a replay buffer:
1. Warm-up phase: collect 1000 random transitions before starting gradient updates.
2. For each of 5000 episodes:
   - Reset environment.
   - Collect 28 steps using the current actor (with `SafetyGuard` filtering actions).
   - Store transitions in replay buffer.
   - After each step (if warmed up): sample 256 transitions and run one SAC update step.
3. Saves best checkpoint when 100-episode moving average reward improves.

### `train_federated.py`

Simulates federated learning:
1. Discovers available subjects (up to 249).
2. Creates one `FedPerClient` per subject with a deep copy of the global `FederatedModel`.
3. Runs 50 rounds:
   - In each round, 30% of clients are randomly selected.
   - Each selected client: receives global model, trains locally (5 epochs), returns shared weights.
   - Server aggregates (weighted by dataset size).
4. Saves global model checkpoint.

---

## 15. Evaluation Scripts

**Location:** `evaluate/`

### `eval_encoders.py`
- **LOSO evaluation** (Leave-One-Subject-Out): For each of 10 MHEALTH subjects, evaluates activity classification F1 score using that subject as the test set. This is the gold standard for generalisation in wearable datasets. **Target: mean F1 > 0.85.**
- **HR MAE evaluation**: Computes mean absolute error of heart rate prediction across all windows. **Target: MAE < 5 bpm.**
- Saves results to `results/encoder_activity_f1.csv` and `results/encoder_hr_mae.csv`.

### `eval_temporal.py`
- **Micro-scale evaluation**: Predicts HR from 180-window sequences. **Target: MAE < 5.**
- **Meso-scale evaluation**: Predicts next-day HR from 7-day history. **Target: MAE < 8.**

### `eval_twin.py`
- **State Reconstruction**: Encodes then decodes health state vectors. **Target: MSE < 0.05.**
- **Trajectory Prediction**: Runs 1-day ahead SDE predictions, measures MAE vs. actual next-day state in latent space. Also measures **95% coverage** (do 95% of true values fall within predicted 95% CI?). **Target: MAE < 0.12, coverage > 0.90.**

### `eval_rl.py`
- Evaluates the trained SAC policy across 50 episodes.
- Compares against 2 baselines:
  - **Random policy**: uniformly random actions.
  - **Fixed moderate**: always `[0.5, 0.5, 0.5, 0.5, 0.7, 0.8]`.
- Reports mean/std reward, safety violations, reward component decomposition.
- **Target**: SAC policy reward > random policy reward.

### `eval_simulation.py`
- **Rollout consistency**: Runs the same plan 5 times, measures coefficient of variation in final HR. Low variance = deterministic enough to trust.
- **Risk calibration**: High-intensity plan should have higher overtraining probability than low-intensity plan. Measures the discriminability gap.
- **Intervention ranking**: Runs all 5 preset scenarios, prints the ranked table, builds a 12-week plan from the top result.

### `eval_federated.py`
- **Personalisation evaluation**: Compares global model MAE vs. personalised (FedPer) model MAE per subject. **Target: >15% average improvement from personalisation.**
- **Cold-start evaluation**: Given only 20 warmup samples from a new subject, how well does the personalised model perform?

### `ablation_study.py`
Systematically tests 12 variants of the model to determine the contribution of each component:

| Ablation | What's removed |
|----------|----------------|
| `no_imu_encoder` | IMU data removed |
| `no_cardio_encoder` | ECG/PPG removed |
| `no_feature_encoder` | Hand-crafted features removed |
| `no_fusion` | Cross-modal fusion removed |
| `no_tcn_only_transformer` | TCN removed (Transformer only) |
| `no_safety_guard` | Safety constraints removed |
| `no_ewc` | Elastic Weight Consolidation removed |
| `beta_vae_beta_1` | β=1 (standard VAE) |
| `beta_vae_beta_8` | β=8 (over-regularised VAE) |
| `sde_no_rest_context` | Rest context removed from SDE |
| `no_fedper_personal` | FedPer personal layers removed |

Each ablation runs 20 evaluation episodes and reports mean reward. The drop from `full_model` shows how much each component contributes.

### `final_report.py`
The master evaluation script that:
1. Runs all evaluation suites in sequence.
2. Checks every metric against its target (PASS/FAIL scorecard).
3. Saves a comprehensive `final_evaluation_report.json` with all results.

**Target metrics:**

| Metric | Target | Direction |
|--------|--------|-----------|
| IMU LOSO F1 | ≥ 0.85 | Higher better |
| HR MAE (bpm) | ≤ 5.0 | Lower better |
| Reconstruction MSE | ≤ 0.05 | Lower better |
| Trajectory MAE | ≤ 0.12 | Lower better |
| Uncertainty Calibration | ≥ 0.90 | Higher better |
| SAC Mean Reward | > 0.0 | Higher better |
| Improvement over random policy | > 0.0 | Higher better |
| Personalisation improvement | ≥ 15% | Higher better |

---

## 16. Checkpoints

**Location:** `checkpoints/`

All trained models are saved as PyTorch state dictionaries (`.pt` files):

| File | Module | Size (params approx.) |
|------|--------|----------------------|
| `encoder_imu.pt` | `SWCTNet` IMU encoder | ~2M |
| `encoder_imu_pretrained.pt` | Pre-trained (before fine-tune) | ~2M |
| `encoder_cardio.pt` | `CardioEncoder` | ~500K |
| `encoder_feature.pt` | `FeatureEncoder` | ~50K |
| `encoder_fusion.pt` | `CrossModalFusion` | ~200K |
| `temporal_micro.pt` | `MicroScaleModel` | ~3M |
| `temporal_meso.pt` | `MesoScaleModel` | ~5M |
| `temporal_macro.pt` | `MacroScaleModel` | ~2M |
| `temporal_fusion.pt` | `HierarchicalFusion` | ~600K |
| `twin_vae.pt` | `BayesianVAE` | ~200K |
| `twin_sde.pt` | `LatentNeuralSDE` | ~100K |
| `rl_actor.pt` | `SquashedGaussianActor` | ~200K |
| `rl_critic.pt` | `TwinCritic` | ~300K |

---

## 17. How Everything Connects — End-to-End Flow

Here is a complete example of how everything runs together, from raw sensor data to a personalised 12-week plan:

### Offline Training Phase (done once):

```
1. python download_datasets.py
   → Downloads 6 datasets to data/raw/

2. python preprocess_all.py  (or src/preprocessing/run_preprocessing.py)
   → Cleans signals, creates windows, extracts features
   → Saves window_*.pt files and daily_summaries to data/processed/

3. python train/train_encoders.py
   → Phase 1: Contrastive pretraining on IMU data
   → Phase 2: Fine-tune with activity labels
   → Phase 3: Train cardio encoder on HR prediction
   → Phase 4: Train feature encoder + cross-modal fusion
   → Saves: encoder_imu.pt, encoder_cardio.pt, encoder_feature.pt, encoder_fusion.pt

4. python train/train_temporal.py
   → Trains micro/meso/macro + hierarchical fusion
   → Saves: temporal_micro.pt, temporal_meso.pt, temporal_macro.pt, temporal_fusion.pt

5. python train/train_twin.py
   → Phase 1: Train β-VAE
   → Phase 2: Train Neural SDE
   → Phase 3: Joint fine-tuning
   → Saves: twin_vae.pt, twin_sde.pt

6. python train/train_rl.py
   → Trains SAC agent in TwinGymEnv
   → Saves: rl_actor.pt, rl_critic.pt

7. python train/train_federated.py
   → Simulates federated training across subjects
   → Saves: federated_global.pt (not in checkpoints/ list, saved to results/)
```

### Inference Phase (for new user):

```
Given: New user's sensor data (last 3 months)
       ↓
1. Preprocess → window_*.pt files + daily summaries
       ↓
2. IMU windows → SWCTNet → h_imu [256]
   Cardio windows → CardioEncoder → h_cardio [128]
   Feature vectors → FeatureEncoder → h_feat [64]
   CrossModalFusion(h_imu, h_cardio, h_feat) → h_fused [128]
       ↓
3. 180 consecutive h_fused → MicroScaleModel → z_micro [256]
   7 daily summaries → MesoScaleModel → z_meso [512]
   6 monthly summaries → MacroScaleModel → z_macro [128]
   HierarchicalFusion(z_micro, z_meso, z_macro) → z_state [512]
       ↓
4. BayesianVAE.encoder(z_state) → μ[10], σ[10]
   Monte Carlo sampling (100 samples) → uncertainty-aware z
       ↓
5. DriftManager.update(features) → is the data normal? adapt if needed
       ↓
6. FedPerClient.personalize() → adapt personal head to this user
       ↓
7. Intervention Ranking:
   WhatIfEngine.compare_scenarios(z0, PRESET_SCENARIOS)
   → Runs MCRolloutEngine with 200×SDE trajectories per scenario
   → Returns ranked list of plans with fitness score + risk scores
       ↓
8. top_k_interventions() → filter unsafe plans → take top-3
   build_periodized_plan(top1, n_weeks=12) → 12-week schedule
       ↓
   Output: "For the next 12 weeks, follow this plan:
            Week 1-3: intensity 0.35, +1h rest/night, ...
            Week 4: recovery week (intensity 0.21)
            ..."
```

---

## 18. Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **IMU** | Inertial Measurement Unit — accelerometer + gyroscope + sometimes magnetometer. Measures movement and orientation. |
| **ECG** | Electrocardiogram — measures electrical activity of the heart. Provides R-R intervals for HRV. |
| **PPG** | Photoplethysmography — optical sensor that shines light through skin to measure blood volume pulse. Used in smartwatches for heart rate. |
| **HRV** | Heart Rate Variability — variation in time between heartbeats. High HRV = good recovery/health. Low HRV = stress/fatigue/overtraining. |
| **Window** | A fixed-length segment of a time series (1000 samples = 20 seconds here). The fundamental processing unit. |
| **Embedding / latent vector** | A learned compact numerical representation of complex data. "Compression" that preserves meaningful information. |
| **Encoder** | A neural network that converts high-dimensional input (raw signal) into a compact embedding. |
| **Contrastive learning** | Self-supervised learning where a model learns by comparing similar and dissimilar pairs, without labels. |
| **NT-Xent loss** | Normalized Temperature-scaled Cross Entropy — the specific contrastive loss used for SimCLR-style pretraining. |
| **Transformer** | Neural network architecture using "attention" — learns which parts of a sequence are important for predicting each output position. |
| **Attention heads** | Parallel "views" in a Transformer. Each head learns to attend to different aspects of the data. |
| **TCN** | Temporal Convolutional Network — dilated causal convolutions for efficient sequence modelling. |
| **Dilation** | In convolution: spacing between kernel elements. Dilation 4 means the kernel "looks at" every 4th sample. |
| **Residual connection** | Adding the input directly to the output (skip connection). Prevents vanishing gradients in deep networks. |
| **VAE** | Variational Autoencoder — encodes data into a probability distribution (not just a point), enabling sampling and uncertainty estimation. |
| **β-VAE** | VAE with β > 1 in the KL term, encouraging a more disentangled, interpretable latent space. |
| **Latent space** | The compressed space where the VAE represents health states. 10-dimensional here. |
| **KL divergence** | Measure of how different two probability distributions are. The KL term in VAE loss keeps the latent distribution close to a standard normal. |
| **Reparameterization trick** | Mathematical trick to allow backpropagation through sampling: z = μ + σ × ε, ε ~ N(0,1). |
| **SDE** | Stochastic Differential Equation — a differential equation with a random noise term. Models systems with inherent randomness. |
| **Neural SDE** | SDE where the drift and diffusion functions are neural networks — learned from data. |
| **Monte Carlo** | Running many random simulations to estimate a distribution. 200 SDE trajectories give a distribution over future health states. |
| **RL** | Reinforcement Learning — learning by trial and error. Agent takes actions, receives rewards, learns to maximise cumulative reward. |
| **SAC** | Soft Actor-Critic — RL algorithm that maximises reward + entropy (encourages exploration). Stable for continuous action spaces. |
| **Replay buffer** | Memory that stores past agent experiences `(state, action, reward, next_state)`. Sampled randomly for training to break correlations. |
| **Soft target update** | Slowly updating target network: `target = τ × online + (1−τ) × target`. Stabilises Q-learning. |
| **Drift** | Statistical change in data distribution over time. In health: your body changes, and the model needs to adapt. |
| **ADWIN** | ADaptive WINdowing — streaming concept drift detection via sliding window comparison. |
| **MMD** | Maximum Mean Discrepancy — statistical distance between two distributions. Used to detect distribution shift. |
| **EWC** | Elastic Weight Consolidation — prevents catastrophic forgetting by penalising changes to important parameters when learning new data. |
| **Fisher Information Matrix** | Measures how much each parameter contributes to the model's output. Used by EWC to identify which weights to protect. |
| **Federated Learning** | Training across multiple users' devices without sharing raw data. Only model weight updates are shared. |
| **FedProx** | Federated learning with proximal regularisation — prevents clients from deviating too far from the global model. |
| **FedPer** | Federated Personalisation — splits model into shared (globally trained) and personal (individually trained) layers. |
| **Digital Twin** | A virtual model of a real-world system that mirrors its state and can simulate its future behaviour. Here: a virtual health model of a person. |
| **Intervention** | A planned health/fitness action: exercise plan, rest protocol, nutrition, sleep schedule. |
| **Periodisation** | Systematic variation of training load over time (build-up phases + recovery weeks) to maximise fitness while preventing overtraining. |
| **Overtraining** | Training too hard / too frequently without adequate recovery. Leads to performance decline and injury risk. |
| **N-BEATS** | Neural Basis Expansion Analysis for Time Series — forecasting architecture that decomposes signals into trend, seasonality, and generic components. |
| **CLS token** | Special "classification" token prepended to a Transformer sequence. Its output at the last layer summarises the entire sequence. |
| **Positional embedding** | Adds information about position (time step) to Transformer inputs, since self-attention is position-agnostic. |
| **Hydra** | Configuration framework that loads YAML files and allows CLI overrides. Used throughout training scripts. |
| **OmegaConf** | Object-oriented configuration library underlying Hydra. |
| **WandB** | Weights & Biases — experiment tracking platform. Logs metrics, models, hyperparameters. Runs offline here. |
| **LOSO** | Leave-One-Subject-Out — evaluation protocol where one subject is the test set and all others are training. Gold standard for wearable ML to ensure the model generalises to new people. |
| **Coverage probability** | In uncertainty quantification: what fraction of true values fall within the predicted confidence interval. A well-calibrated model with 95% CI should have coverage ≈ 0.95. |
| **Recency weights** | In MesoScaleModel: weights `[0.05, ..., 0.30]` that give more importance to recent days in the weekly summary. |
| **Sympathetic/Parasympathetic** | Two branches of the autonomic nervous system. Sympathetic = "fight-or-flight" (elevates HR, stress). Parasympathetic = "rest-and-digest" (lowers HR, recovers HRV). |
| **LF/HF ratio** | Sympathovagal balance metric. High ratio = more sympathetic (stressed/exercising). Low ratio = more parasympathetic (resting/recovering). |
| **Circadian rhythm** | Natural ~24-hour physiological cycle. Captured in daily summary as a 64-bin activity histogram. |
| **Softplus** | Activation function `log(1 + exp(x))` — smooth, always positive. Used in SDE diffusion net to ensure non-negative noise. |
| **Weight normalisation** | Re-parameterises weights as `w = g × v/||v||`. Stabilises training of TCN layers. |
| **LayerNorm** | Normalises across feature dimension (not batch). Better than BatchNorm for variable-length sequences and small batches. |
| **Dropout** | Randomly sets a fraction of outputs to zero during training. Regularisation technique to prevent overfitting. |
| **AdaptiveAvgPool1d(1)** | Compresses any-length sequence to 1 timestep by averaging. Used after CNN to get a single vector regardless of window length. |

---

*This document was generated to comprehensively explain every component of the HUA-DTIP project. Every concept, design decision, data flow, and algorithm is covered above.*
