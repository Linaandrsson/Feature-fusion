# Multi-Sensor Fusion Framework for Human Activity Recognition
*A robust framework for sensor fusion with noise resilience*

## 📋 Overview

This framework implements a complete pipeline for multi-sensor Human Activity Recognition (HAR) using the MHEALTH dataset. It trains sensor-specific CNNs to extract features, then fuses them using gated or concatenation-based methods. The system supports:

- **Multiple sensor types**: Accelerometers, gyroscopes, magnetometers, and ECG
- **Configurable sampling rates**: Mix sensors at different frequencies (10-50 Hz)
- **Noise robustness**: Train and test with realistic sensor corruptions
- **Feature-level fusion**: Gated attention or simple concatenation
- **Comprehensive logging**: Track experiments, corruptions, and performance

---

## 🏗️ Project Structure

```
Code/
├── 📊 data/                              # Raw and processed data
│   ├── MHEALTHDATASET/                   # Original MHEALTH dataset
│   └── Datagenerator_files/              # Processed time-series windows
│       └── {config}/                     # e.g., s1_w1_aug2, s2_w2_aug2
│           ├── splits.npz                # Train/val/test splits (shared)
│           └── {variant}/                # e.g., fs30_clean, fs50_AWGN_s0p3
│               ├── Acc_ankle.txt         # Sensor data files
│               ├── Acc_arm.txt
│               └── ExtractedFeatures/    # CNN embeddings (128-dim)
│
├── 🔧 Core Scripts
│   ├── DataGenerator_v3.py               # Generate windowed data with noise
│   ├── SplitIndexes.py                   # Create train/val/test splits
│   └── Sanity_check.py                   # Verify data generation
│
├── 📈 Dataset_analyzis/                  # Exploratory data analysis
│   ├── dataset_analyzis.py               # Dataset statistics
│   ├── DataPlotting_per_sensor.py        # Visualize sensor signals
│   └── DataPlotting_per_activity.py      # Visualize activities
│
├── 🔊 Noise_simulation/                  # Sensor corruption modules
│   ├── AWGN.py                           # Gaussian noise (hardware noise)
│   ├── Dropout.py                        # Sensor failure (constant value)
│   ├── WeakSignal.py                     # Signal attenuation (degradation)
│   └── Tremor.py                         # Physiological tremor (van der Pol)
│
├── 🧠 Feature extraction CNNs/           # CNN-based feature extractors
│   ├── Training/                         # Train CNNs per configuration
│   │   ├── s1_w1_aug2_fs10/              # 1s stride, 1s window, 10Hz
│   │   ├── s1_w1_aug2_fs20/              # ... 20Hz
│   │   ├── s1_w1_aug2_fs30/              # ... 30Hz
│   │   ├── s1_w1_aug2_fs50/              # ... 50Hz
│   │   ├── s2_w2_aug2_fs20/              # 2s stride, 2s window, 20Hz
│   │   └── ...
│   │       ├── config.py                 # Paths and hyperparameters
│   │       ├── Acc_ankle_CNN.py          # Train accelerometer CNN
│   │       ├── Gyro_arm_CNN.py           # Train gyroscope CNN
│   │       └── ...                       # One script per sensor
│   ├── Models/                           # Saved CNN models
│   │   └── {config}/{variant}/
│   │       └── feature_extractor_*.pth
│   └── Extraction/                       # Extract embeddings
│       ├── config_extraction.py          # Configure extraction
│       ├── extract_features.py           # Extract single sensor
│       └── extract_all_sensors.py        # Extract all sensors
│
├── 🔗 Feature fusion/                    # Multi-sensor fusion training
│   ├── fusion_concat_train_v3.py         # Main fusion training (latest)
│   ├── fusion_concat_fs_study.py         # Feature size ablation study
│   ├── fusion_concat_ablation_study.py   # Sensor ablation study
│   ├── analyze_fs_results.py             # Analyze feature size results
│   └── analyze_ablation_results.py       # Analyze ablation results
│
├── 📁 fusion_logs/                       # Experiment logs
│   ├── fusion_v3_experiments.jsonl       # Fusion training logs
│   ├── fs_combo_study.jsonl              # Feature size study logs
│   ├── fusion_v4_ablation.jsonl          # Ablation study logs
│   ├── corruption_metadata.jsonl         # Per-window corruption tracking
│   └── fs_study_reports/                 # Analysis reports
│
├── 📊 Visualization outputs
│   ├── gating_plots/                     # Gate weight visualizations
│   └── ablation_plots/                   # Ablation study plots
│
├── 📚 Documentation/
│   ├── DataGenerator_README.md           # Data generation guide
│   └── TREMOR_Guide.md                   # Tremor simulation guide
│
└── 🗂️ Baseline models/                  # Legacy baseline classifiers
    └── ...
```

---

## 🚀 Complete Workflow

### 1️⃣ Data Preparation

#### Step 1.1: Generate Windowed Data

**File:** [DataGenerator_v3.py](DataGenerator_v3.py)

Convert raw MHEALTH logs into windowed, z-scored time series with optional noise injection.

```bash
python DataGenerator_v3.py
```

**Key configurations** (edit in script):
```python
# Window parameters
WINDOW_SEC = 2.0        # Window duration
STRIDE_SEC = 2.0        # Stride (1.0 = 50% overlap with 2s window)
FS = 30                 # Target sampling rate (10, 20, 30, or 50 Hz)
AUG_SIZE = 2            # Augmentation copies per window

# Noise scenario (applied to raw signals)
SCENARIO_NOISE_TYPE = None  # or "AWGN", "DROPOUT", "WEAK_SIGNAL", "TREMOR"
```

**Output:** `data/Datagenerator_files/{config}/{variant}/`
- Example: `s2_w2_aug2/fs30_clean/` (2s window, 30Hz, clean data)

#### Step 1.2: Create Train/Val/Test Splits

**File:** [SplitIndexes.py](SplitIndexes.py)

Split data into stratified train/val/test sets (shared across variants).

```bash
python SplitIndexes.py
```

**Output:** `data/Datagenerator_files/{config}/splits.npz`

---

### 2️⃣ Feature Extraction

#### Step 2.1: Train CNNs per Sensor

**Directory:** [Feature extraction CNNs/Training/](Feature extraction CNNs/Training/)

Train one CNN per sensor to extract 128-dimensional embeddings.

```bash
# Navigate to appropriate config
cd "Feature extraction CNNs/Training/s2_w2_aug2_fs30"

# Train individual sensor
python Acc_ankle_CNN.py

# Or train all sensors
for script in *_CNN.py ECG_chest.py; do
    python "$script"
done
```

**Configuration** (edit `config.py`):
```python
parent_dir = Path("../../data/Datagenerator_files/s2_w2_aug2")
variant_name = "fs30_clean"  # Which data variant to train on
seq_len = 60  # 2 seconds * 30 Hz
```

**Models saved to:** `Feature extraction CNNs/Models/{config}/{variant}/feature_extractor_*.pth`

**Architecture:**
- Conv1D → Conv1D → Flatten → FC(128) → FC(12 classes)
- Embedding dimension: 128 (extracted from `fc_embed` layer)

#### Step 2.2: Extract Features

**Directory:** [Feature extraction CNNs/Extraction/](Feature extraction CNNs/Extraction/)

Use trained CNNs to extract embeddings from data.

```bash
cd "Feature extraction CNNs/Extraction"

# Configure extraction (edit config_extraction.py)
# - model_config: Which model set to use (e.g., "s2_w2_aug2")
# - model_variant: Which model variant (e.g., "fs30_clean")
# - variant_name: Which data to extract from (can differ from model_variant)

# Extract all sensors
python extract_all_sensors.py
```

**Output:** `data/Datagenerator_files/{config}/{variant}/ExtractedFeatures/`
- `{sensor}_features_train.npz` (features + labels)
- `{sensor}_features_val.npz`
- `{sensor}_features_test.npz`

---

### 3️⃣ Multi-Sensor Fusion

#### Fusion Training

**File:** [Feature fusion/fusion_concat_train_v3.py](Feature fusion/fusion_concat_train_v3.py)

Combine embeddings from multiple sensors with gated attention or concatenation.

```bash
cd "Feature fusion"
python fusion_concat_train_v3.py
```

**Key configurations** (edit in script):

```python
# Sensors and sampling rates
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest", "Mag_arm"]
sensor_fs = [30, 30, 30, 30]  # Can mix rates: [20, 30, 30, 50]

# Corruption scenarios (optional)
train_corruption_config = {
    "AWGN_s0p3": 0.3,                   # 30% AWGN on all sensors
    "DROPOUT_drop/Acc_ankle": 0.1,      # Additional 10% dropout on ankle
}
test_corruption_config = {}  # Test on clean data

# Fusion architecture
USE_GATING = True           # True = gated fusion, False = concatenation
gate_type = "sigmoid"       # "sigmoid" or "softmax"
gate_hidden = 64            # Gate MLP hidden size (0 = linear gate)
alpha_floor = 0.05          # Minimum gate weight

# Training
epochs = 200
lr = 1e-3
patience = 30
```

**Output:**
- Model: `fusion_logs/fusion_model_best.pth`
- Logs: `fusion_logs/fusion_v3_experiments.jsonl`
- Plots: `gating_plots/` (gate weights, training curves)

#### Ablation Studies

**Feature Size Study:**
```bash
python fusion_concat_fs_study.py    # Run experiments
python analyze_fs_results.py        # Analyze results
```

Tests different embedding dimensions per sensor (10, 20, 30, ...).

**Sensor Ablation Study:**
```bash
python fusion_concat_ablation_study.py    # Run experiments
python analyze_ablation_results.py        # Analyze results
```

Tests removing individual sensors or sensor groups.

---

## 📦 Key Components

### Data Generation (Noise Injection)

**Module:** [Noise_simulation/](Noise_simulation/)

Four noise types simulate realistic sensor issues:

| Noise Type | Module | Simulates | Use Case |
|------------|--------|-----------|----------|
| **AWGN** | [AWGN.py](Noise_simulation/AWGN.py) | Hardware noise | Thermal noise, measurement uncertainty |
| **Dropout** | [Dropout.py](Noise_simulation/Dropout.py) | Sensor failure | Connection loss, battery depletion |
| **WeakSignal** | [WeakSignal.py](Noise_simulation/WeakSignal.py) | Degradation | Poor contact, low battery, sensor aging |
| **Tremor** | [Tremor.py](Noise_simulation/Tremor.py) | Physiological tremor | Parkinsonian/essential tremor (4-12 Hz) |

**Usage example:**
```python
from Noise_simulation.AWGN import apply_awgn_rms_ratio

noisy_signal = apply_awgn_rms_ratio(raw_signal, rms_ratio=0.2, rng=rng)
```

### CNN Feature Extractors

**Directory:** [Feature extraction CNNs/](Feature extraction CNNs/)

- **One CNN per sensor:** Each sensor type has its own CNN (Acc_ankle, Gyro_arm, etc.)
- **Shared architecture:** All CNNs use the same Conv1D-based architecture
- **Embedding size:** 128-dimensional feature vectors
- **Multi-rate support:** Train separate models for different sampling rates

**Available sensors:**
- Accelerometers: `Acc_ankle`, `Acc_arm`, `Acc_chest` (3-axis)
- Gyroscopes: `Gyro_ankle`, `Gyro_arm` (3-axis)
- Magnetometers: `Mag_ankle`, `Mag_arm` (3-axis)
- ECG: `ECG` (2-lead)

### Fusion Models

**Directory:** [Feature fusion/](Feature fusion/)

Two fusion strategies:

1. **Concatenation Baseline:**
   - Concatenate all sensor embeddings: `[e1; e2; ...; eN]`
   - Feed to MLP classifier
   - Simple but treats all sensors equally

2. **Gated Fusion:**
   - Learn attention weights per sensor: `α_i = gate(e_i)`
   - Weighted combination: `h = Σ α_i * e_i`
   - **Sigmoid gate:** Independent weights (sensors can be high/low together)
   - **Softmax gate:** Competitive weights (sum to 1, sensors compete)
   - **Alpha floor:** Minimum weight (prevents complete sensor shutdown)

**Corruption handling:**
- **Global:** Apply same corruption to all sensors (`"AWGN_s0p3": 0.3`)
- **Per-sensor:** Target specific sensors (`"DROPOUT_drop/Acc_ankle": 0.1`)
- **Mixed:** Combine global and per-sensor corruptions
- **Multi-scenario:** Multiple corruption types simultaneously

---

## 📊 Dataset: MHEALTH

**Location:** [data/MHEALTHDATASET/](data/MHEALTHDATASET/)

- **Subjects:** 10 healthy individuals
- **Activities:** 12 classes (standing, walking, running, cycling, etc.)
- **Sensors:**
  - 3× BioHarness BT accelerometers (chest, left ankle, right arm)
  - 2× gyroscopes + magnetometers (ankle, arm)
  - 2-lead ECG (chest)
- **Sampling rate:** 50 Hz
- **Total samples:** ~660,000 frames

**Activity classes:**
0. Nothing (null class)
1. Standing still
2. Sitting and relaxing
3. Lying down
4. Walking
5. Climbing stairs
6. Waist bends forward
7. Frontal elevation of arms
8. Knees bending (crouching)
9. Cycling
10. Jogging
11. Running
12. Jump front & back

---

## 🔬 Experimental Configurations

### Naming Convention

Format: `s{stride}_w{window}_aug{aug_size}_fs{freq}`

Examples:
- `s1_w1_aug2_fs50`: 1s stride, 1s window, 2 augmentations, 50Hz
- `s2_w2_aug2_fs30`: 2s stride, 2s window, 2 augmentations, 30Hz

### Noise Scenarios

Format: `fs{freq}_{noise_type}_{params}_{sensors}`

Examples:
- `fs30_clean`: Clean data at 30Hz
- `fs50_AWGN_s0p3`: AWGN (σ=0.3) at 50Hz on all sensors
- `fs30_DROPOUT_drop_AAn`: Dropout on Acc_ankle at 30Hz
- `fs50_TREMOR_mu1p0`: Tremor (μ=1.0) at 50Hz

**Sensor abbreviations:**
- `AAn` = Acc_ankle
- `AAr` = Acc_arm
- `ACh` = Acc_chest
- `GAn` = Gyro_ankle
- `GAr` = Gyro_arm
- `MAn` = Mag_ankle
- `MAr` = Mag_arm
- `ECG` = ECG

---

## 📁 Output Files

### Data Files
```
data/Datagenerator_files/{config}/{variant}/
├── Acc_ankle.txt           # Shape: (N, seq_len, 3)
├── Acc_arm.txt
├── labels.txt              # Activity labels (N,)
├── base_info.txt           # Configuration summary
└── ExtractedFeatures/
    ├── Acc_ankle_features_train.npz
    ├── Acc_ankle_features_val.npz
    └── Acc_ankle_features_test.npz
```

### Model Checkpoints
```
Feature extraction CNNs/Models/{config}/{variant}/
└── feature_extractor_{sensor}.pth
    ├── model_state_dict    # Model weights
    ├── num_classes         # 12
    ├── seq_len             # Window size in samples
    └── num_channels        # Sensor channels
```

### Logs
```
fusion_logs/
├── fusion_v3_experiments.jsonl       # One line per fusion experiment
├── fs_combo_study.jsonl              # Feature size study results
├── fusion_v4_ablation.jsonl          # Ablation study results
├── corruption_metadata.jsonl         # Per-window corruption details
└── fs_study_reports/                 # Analysis reports (JSON & TXT)
    └── {condition}/{fusion_type}/
        └── results_*.{json,txt}
```

**Log format (JSONL):**
```json
{
  "timestamp": "2026-02-17_12:34:56",
  "sensors": ["Acc_ankle", "Acc_arm"],
  "sensor_fs": [30, 30],
  "fusion_type": "gated_sigmoid",
  "train_corruption": {"AWGN_s0p3": 0.3},
  "test_corruption": {},
  "best_val_acc": 0.9234,
  "test_acc": 0.9156,
  "train_time_sec": 1234.5
}
```

---

## 🛠️ Dependencies

```
torch >= 2.0
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

Install:
```bash
conda activate base
pip install torch numpy scipy pandas matplotlib seaborn scikit-learn tqdm
```

---

## 📈 Typical Results

### Single-Sensor Baselines
- Acc_ankle: ~85% accuracy
- Acc_arm: ~82% accuracy
- Acc_chest: ~88% accuracy
- Gyro_ankle: ~78% accuracy

### Fusion Results
- **Concat (clean):** ~92% accuracy
- **Gated (clean):** ~93% accuracy
- **Gated (30% AWGN training):** ~91% accuracy (robust to noise)

### Gating Insights
- **High weights:** Acc_chest, Acc_ankle (most informative)
- **Medium weights:** Acc_arm, Gyro_arm
- **Low weights:** Mag sensors (less informative for HAR)
- **Adaptation:** Weights adjust down for corrupted sensors during test

---

## 🔍 Analysis Tools

### Dataset Analysis

**Directory:** [Dataset_analyzis/](Dataset_analyzis/)

```bash
cd Dataset_analyzis

# Generate dataset statistics
python dataset_analyzis.py

# Visualize sensor signals per activity
python DataPlotting_per_activity.py

# Visualize all sensors for comparison
python DataPlotting_per_sensor.py
```

**Outputs:**
- `dataset_summary.txt`: Class distribution, samples per subject
- `plots_per_activity/`: Time series plots grouped by activity
- `plots_per_sensor/`: Sensor comparison plots

### Fusion Analysis

```bash
cd "Feature fusion"

# Analyze feature size study
python analyze_fs_results.py

# Analyze sensor ablation
python analyze_ablation_results.py
```

**Outputs:**
- Summary tables (best configurations)
- Performance comparisons
- Statistical analysis

---

## 🔄 Version History

### Latest Versions (Use These!)

- **DataGenerator:** `DataGenerator_v3.py` (supports per-sensor noise control)
- **Fusion:** `fusion_concat_train_v3.py` (multi-FS, per-sensor corruption, gating)
- **Extraction:** `extract_all_sensors.py` (batch processing)

### Deprecated Files

- `fusion_concat_train v1.py`, `v2.py` → Use `_v3.py`
- Legacy baseline models in `Baseline models/` (superseded by CNNs + Fusion)

---

## 🚨 Common Issues & Solutions

### Issue: "parent_dir NOT found"
**Solution:** Run `DataGenerator_v3.py` first to generate data.

### Issue: "splits.npz not found"
**Solution:** Run `SplitIndexes.py` after data generation.

### Issue: "Model checkpoint not found"
**Solution:** Train CNNs first in `Feature extraction CNNs/Training/{config}/`.

### Issue: "Label mismatch across sensors"
**Solution:** 
- Ensure all sensors use same `splits.npz`
- Verify data generation used same seed and configuration

### Issue: "Embedding dimension mismatch"
**Solution:** Check that all CNNs in fusion use same embedding size (128).

### Issue: Corruption probabilities sum > 1.0
**Solution:** Per-sensor corruption probabilities must sum to ≤ 1.0.

---

## 📚 Documentation

- [DataGenerator Guide](Documentation/DataGenerator_README.md) - Data generation details
- [Tremor Simulation](Documentation/TREMOR_Guide.md) - Tremor implementation
- [Feature Extraction](Feature extraction CNNs/README.md) - CNN training and extraction
- [Fusion Training](Feature fusion/README_v3.md) - Fusion configuration guide
- [Noise Modules](Noise_simulation/README.md) - Noise type details

---

## 📧 Contact & Citation

**Author:** Lina Andersson (Master's Thesis, NTNU)  
**Topic:** Robust Multi-Sensor Fusion for Human Activity Recognition

If you use this framework, please cite:
```
@mastersthesis{andersson2026fusion,
  title={Robust Multi-Sensor Fusion for Human Activity Recognition},
  author={Andersson, Lina},
  year={2026},
  school={Norwegian University of Science and Technology}
}
```

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| Generate data | `python DataGenerator_v3.py` |
| Create splits | `python SplitIndexes.py` |
| Train CNN (single) | `cd "Feature extraction CNNs/Training/{config}" && python Acc_ankle_CNN.py` |
| Train all CNNs | `for f in *_CNN.py ECG_chest.py; do python "$f"; done` |
| Extract features | `cd "Feature extraction CNNs/Extraction" && python extract_all_sensors.py` |
| Train fusion | `cd "Feature fusion" && python fusion_concat_train_v3.py` |
| Feature size study | `python fusion_concat_fs_study.py && python analyze_fs_results.py` |
| Ablation study | `python fusion_concat_ablation_study.py && python analyze_ablation_results.py` |
| Analyze dataset | `cd Dataset_analyzis && python dataset_analyzis.py` |

---

**Last updated:** February 17, 2026
