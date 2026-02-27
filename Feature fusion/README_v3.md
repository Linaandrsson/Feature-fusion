# Fusion Training v3 - User Guide

## Overview

`fusion_concat_train_v3.py` trains a multi-sensor fusion model for Human Activity Recognition (HAR). It combines embeddings from multiple sensors (extracted by CNNs) and trains a fusion classifier with optional gating mechanism.

**Key features:**
- Mix different sampling rates per sensor (e.g., Acc_ankle at 20Hz, Acc_arm at 30Hz)
- Train with controlled corruption (noise injection during training)
- Per-sensor corruption control
- Gated vs. concat fusion
- Automatic alignment verification
- Comprehensive logging and plotting

---

## Quick Start

### Basic Usage (All Clean Data)

```python
# In fusion_concat_train_v3.py, set:
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest"]
sensor_fs = [30, 30, 30]
train_corruption_config = {}
test_corruption_config = {}
USE_GATING = True
```

Run: `python fusion_concat_train_v3.py`

---

## Configuration Guide

### 1. Sensor Selection

```python
# Sensors to use
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest", "Mag_arm"]

# Sampling rate for each sensor (must match length of sensors)
sensor_fs = [30, 30, 30, 30]  # All at 30Hz
# OR mix different rates:
sensor_fs = [20, 30, 30, 50]  # Different FS per sensor
```

**Available sensors:**
- `Acc_ankle`, `Acc_arm`, `Acc_chest` (3-axis accelerometers)
- `Gyro_ankle`, `Gyro_arm` (3-axis gyroscopes)
- `Mag_ankle`, `Mag_arm` (3-axis magnetometers)
- `ECG` (1-channel ECG from chest)

**Available sampling rates:** 10, 20, 30, 50 Hz

---

### 2. Corruption Configuration

Control which sensors get corrupted during training and testing.

#### Simple: All Sensors Clean (Default)

```python
train_corruption_config = {}
test_corruption_config = {}
```

Result: All sensors use 100% clean data.

#### Global Corruption (Same for All Sensors)

```python
# Apply 30% AWGN to ALL sensors during training
train_corruption_config = {
    "AWGN_s0p3": 0.3
}
# Test on clean data
test_corruption_config = {}
```

Result:
- All sensors: 30% AWGN, 70% clean (training)
- All sensors: 100% clean (test)

#### Per-Sensor Corruption

```python
# Syntax: "scenario/Sensor_name": probability
train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.15,      # 15% AWGN on Acc_ankle
    "DROPOUT_drop/Acc_ankle": 0.10,    # 10% dropout on Acc_ankle
    "WEAK_SIGNAL_w0p2/Mag_arm": 0.05,  # 5% weak signal on Mag_arm
}
# Acc_ankle: 15% AWGN + 10% dropout + 75% clean
# Mag_arm: 5% weak signal + 95% clean
# Other sensors: 100% clean
```

#### Mix Global and Per-Sensor

```python
train_corruption_config = {
    "AWGN_s0p3": 0.2,                  # 20% AWGN on all sensors
    "DROPOUT_drop/Acc_ankle": 0.1,     # Additional 10% dropout only on ankle
}
# Result:
#   Acc_ankle: 20% AWGN + 10% dropout + 70% clean
#   Others: 20% AWGN + 80% clean
```

#### Test with Corruption

```python
test_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.5,  # Test ankle with 50% AWGN
}
# Result: Acc_ankle has 50% AWGN in test, others clean
```

**Available scenarios:**
- `AWGN_s0p3` - Additive White Gaussian Noise (σ=0.3)
- `DROPOUT_drop` - Random dropout corruption
- `WEAK_SIGNAL_w0p2` - Signal attenuation (scale=0.2)

**Important:** Probabilities per sensor must sum to ≤ 1.0. Remainder is automatically clean.

---

### 3. Fusion Model Configuration

#### Gated Fusion (Recommended)

```python
USE_GATING = True

# Gating parameters
gate_type = "sigmoid"       # "sigmoid" or "softmax"
gate_hidden = 64            # 0 = linear gate, >0 = MLP gate
gate_dropout = 0.1          # Dropout before gate
use_layernorm = True        # Layer normalization per sensor
alpha_floor = 0.05          # Minimum gate weight (prevents complete shutdown)
```

**Gate types:**
- `"sigmoid"`: Independent weights per sensor (0-1 each, can all be high/low)
- `"softmax"`: Competitive weights (sum to 1, sensors compete)

#### Concat Baseline

```python
USE_GATING = False
```

Simple concatenation of all sensor embeddings → MLP classifier.

---

### 4. Training Hyperparameters

```python
batch_size = 128
epochs = 200
lr = 1e-3
patience = 30           # Early stopping patience
min_delta = 1e-4        # Minimum improvement for early stopping

# Classifier head
head_hidden_dims = (64, 64)
head_dropout = 0.4
```

---

### 5. Paths and Logging

```python
# Data directory
base_data_dir = Path("/path/to/data/Datagenerator_files")
dataset_config = "s1_w1_aug2"

# Logging
LOG_DIR = Path("fusion_logs")
LOG_FILE = LOG_DIR / "fusion_v3_experiments.jsonl"
CORRUPTION_LOG_FILE = LOG_DIR / "corruption_metadata.jsonl"

# Plotting
SAVE_PLOTS = True
PLOT_DIR = Path("gating_plots")

# Detailed logging (per-sample corruption info)
LOG_SAMPLE_CORRUPTION = False  # Set True for debugging
```

---

## Example Configurations

### Example 1: Baseline Evaluation

Test pure concatenation with clean data.

```python
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest"]
sensor_fs = [30, 30, 30]
train_corruption_config = {}
test_corruption_config = {}
USE_GATING = False
```

### Example 2: Robustness Training

Train with 30% global noise, test on clean.

```python
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest", "Mag_arm"]
sensor_fs = [30, 30, 30, 30]
train_corruption_config = {
    "AWGN_s0p3": 0.3
}
test_corruption_config = {}
USE_GATING = True
```

### Example 3: Sensor-Specific Corruption

Simulate realistic failure modes.

```python
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest", "Mag_arm"]
sensor_fs = [30, 30, 30, 30]

# Ankle often has AWGN, arm has dropouts, magnetometer weak signal
train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.3,
    "DROPOUT_drop/Acc_arm": 0.15,
    "WEAK_SIGNAL_w0p2/Mag_arm": 0.2,
}

# Test with different corruption
test_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.5,  # Stress-test ankle
}

USE_GATING = True
gate_type = "sigmoid"
```

### Example 4: Multi-FS Setup

Different sensors at different sampling rates.

```python
sensors = ["Acc_ankle", "Acc_arm", "Gyro_arm", "Mag_arm"]
sensor_fs = [20, 30, 30, 50]  # Mix of rates

train_corruption_config = {
    "AWGN_s0p3": 0.2,  # Global noise
}

test_corruption_config = {}
USE_GATING = True
```

---

## Understanding the Output

### Training Output

```
======================================================================
Multi-FS Multi-Scenario Fusion Training (v3)
======================================================================

[0/7] Parsing corruption configurations...

Training corruption configuration:
  Acc_ankle:
    AWGN_s0p3: 30.0%
    clean: 70.0%
  Acc_arm: 100% clean
  ...

[1/7] Loading embeddings...
  ✓ Loaded Acc_ankle (fs30) - clean
  ✓ Loaded Acc_ankle (fs30) - AWGN_s0p3
  ...

[2/7] Verifying label alignment...
✓ Label alignment verified for train split (6 variants)

Dataset info:
  Sensors: 4
  Embedding dim: 128
  Classes: 12
  Train/Val/Test: 9849/2110/2111

[3/7] Creating datasets...
  Train dataset: 9849 samples
  Val dataset: 2110 samples (clean only)
  Test dataset: 2111 samples

Training corruption summary:
  Acc_ankle:
    clean: 6894 (70.0%)
    AWGN_s0p3: 2955 (30.0%)
  ...

[4/7] Creating fusion model...
  Mode: GATED (gate_type=sigmoid, gate_hidden=64)

[5/7] Training...
Epoch   1/200 | Train loss 1.2345 acc 0.6543 | Val loss 0.9876 acc 0.7234
...
Early stopping at epoch 67. Best val loss: 0.1234

[6/7] Evaluating on test set...

Test Accuracy: 0.9621

Gating weights (test set):
  Acc_ankle: 0.994 ± 0.007
  Acc_arm: 0.991 ± 0.010
  Acc_chest: 0.994 ± 0.006
  Mag_arm: 0.977 ± 0.030

[7/7] Logging results...
  ✓ Logged to fusion_logs/fusion_v3_experiments.jsonl
  ✓ Logged corruption metadata to fusion_logs/corruption_metadata.jsonl
  ✓ Saved gating_plots/test_gates_mean.png
  ✓ Saved gating_plots/test_gates_hist.png
```

### Interpreting Gating Weights

**High weight (>0.95):**
- Sensor is very reliable and informative
- Model strongly depends on this sensor
- Example: `Acc_ankle: 0.994` → Model uses 99.4% of ankle information

**Medium weight (0.7-0.95):**
- Sensor is useful but partially downweighted
- May indicate noise or redundancy
- Example: `Mag_arm: 0.977` → Still valuable but slightly less trusted

**Low weight (<0.7):**
- Sensor is deemed unreliable or uninformative
- Model has learned to ignore this sensor
- This happens when you have very noisy sensors or redundant information

**High variance (std >0.05):**
- Weight varies significantly across samples
- Model adapts gating per input
- Example: `Mag_arm: 0.977 ± 0.030` → More dynamic than others

**All weights near 1.0:**
- All sensors are valuable
- Gating hasn't learned to discriminate
- Consider: Is gating needed? Try concat baseline comparison

---

## Log Files

### fusion_v3_experiments.jsonl

One JSON line per run with:
```json
{
  "experiment": "fusion_v3",
  "sensors": ["Acc_ankle", "Acc_arm"],
  "sensor_fs": [30, 30],
  "train_corruption_config": {...},
  "test_corruption_config": {...},
  "fusion_mode": "gated",
  "val_accuracy": 0.9543,
  "test_accuracy": 0.9621,
  "gate_mean": {"Acc_ankle": 0.994, ...},
  "gate_std": {"Acc_ankle": 0.007, ...},
  "seed": 42,
  "best_val_loss": 0.1234
}
```

### corruption_metadata.jsonl

Corruption statistics per run:
```json
{
  "train_summary": {
    "Acc_ankle": {
      "clean": {"count": 6894, "percentage": 70.0},
      "AWGN_s0p3": {"count": 2955, "percentage": 30.0}
    }
  },
  "test_summary": {...},
  ...
}
```

---

## Plots

### test_gates_mean.png
Bar plot of average gating weights per sensor on test set.

### test_gates_hist.png
Distribution of gating weights across all test samples per sensor.

---

## Common Issues

### FileNotFoundError: Embeddings not found

**Problem:** Embeddings haven't been extracted for the specified sensor/FS/scenario.

**Solution:**
1. Check that data exists: `data/Datagenerator_files/s1_w1_aug2/fs30_clean/`
2. Extract embeddings using `extract_all_sensors.py`
3. Ensure `sensor_fs` matches available data

### ValueError: Corruption probabilities sum > 1.0

**Problem:** You specified too much corruption for a sensor.

**Solution:**
```python
# BAD:
train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.7,
    "DROPOUT_drop/Acc_ankle": 0.5,  # 0.7 + 0.5 = 1.2 > 1.0
}

# GOOD:
train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.5,
    "DROPOUT_drop/Acc_ankle": 0.3,  # 0.5 + 0.3 = 0.8, clean = 0.2
}
```

### Label mismatch error

**Problem:** Embeddings from different sources have different split orderings.

**Solution:** All embeddings must use the same `splits.npz` file. Re-extract with consistent configuration.

---

## Advanced Usage

### Experiment Grid Search

Create a script to test multiple configurations:

```python
import fusion_concat_train_v3 as v3

configs = [
    {"corruption": {}, "gating": False},  # Baseline concat
    {"corruption": {}, "gating": True},   # Gated clean
    {"corruption": {"AWGN_s0p3": 0.3}, "gating": True},  # Gated + noise
]

for config in configs:
    v3.train_corruption_config = config["corruption"]
    v3.USE_GATING = config["gating"]
    v3.main()
```

### Compare Gate Types

```python
for gate_type in ["sigmoid", "softmax"]:
    v3.gate_type = gate_type
    v3.main()
```

Compare: Do sigmoid and softmax learn different importance patterns?

### Ablation Study

Remove sensors one by one to measure contribution:

```python
all_sensors = ["Acc_ankle", "Acc_arm", "Acc_chest", "Mag_arm"]

for i, remove_sensor in enumerate(all_sensors):
    v3.sensors = [s for s in all_sensors if s != remove_sensor]
    v3.sensor_fs = [30] * len(v3.sensors)
    v3.main()
```

---

## Tips for Good Results

1. **Start with clean data baseline** - Establish upper bound performance
2. **Gradually add corruption** - Start with 10-20%, increase if needed
3. **Use validation set properly** - Always keep val clean for fair model selection
4. **Compare gating vs. concat** - Gating adds complexity; verify it helps
5. **Check alignment** - Script verifies labels match, but sanity check your data
6. **Monitor gating weights** - If all near 1.0, consider simpler concat model
7. **Use per-sensor corruption wisely** - Model realistic failure modes for your application

---

## Related Files

- `fusion_concat_train_v2.py` - Previous version (simpler, no per-sensor corruption)
- `extract_features.py` - Extract embeddings from trained CNNs
- `extract_all_sensors.py` - Batch extraction for all sensors
- `DataGenerator_v3.py` - Generate clean/noisy sensor data

---

## Citation / Contact

If you use this code, please cite the relevant paper and acknowledge the mHealth dataset.

For questions or issues, check the main project README or contact the maintainer.
