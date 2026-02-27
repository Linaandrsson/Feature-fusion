# DataGenerator v2 - User Guide

## Overview

DataGenerator v2 is a flexible tool for generating time-series datasets from the mHealth dataset with configurable window parameters, augmentation, and scenario noise simulation. It's designed for testing sensor fusion models under various corruption scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Parameters](#configuration-parameters)
3. [Scenario Noise Types](#scenario-noise-types)
4. [Output Format](#output-format)
5. [Usage Examples](#usage-examples)
6. [Understanding the Output](#understanding-the-output)
7. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Generate a Clean Dataset

```python
# In DataGenerator_v2.py, set:
SCENARIO_NOISE_TYPE = None
```

Run:
```bash
python DataGenerator_v2.py
```

Output folder: `data/Datagenerator_files/fs50_s1_w1_aug2/`

### Generate a Dataset with AWGN Noise

```python
SCENARIO_NOISE_TYPE = "AWGN"
AWGN_SIGMA = 0.3
SCENARIO_NOISE_SENSOR_POOL = ["Acc_chest", "Acc_ankle"]
SCENARIO_NOISE_NUM_SENSORS = 1
```

Output folder: `data/Datagenerator_files/fs50_s1_w1_aug2_AWGN_s0p3_n1/`

---

## Configuration Parameters

### Basic Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ORIGINAL_FS` | Original sampling rate in dataset (Hz) | `50` |
| `FS` | Target sampling rate (Hz) | `50` |
| `WINDOW_SEC` | Window length in seconds | `1.0` |
| `STRIDE_SEC` | Stride between windows in seconds | `1.0` |
| `AUG_SIZE` | Number of augmented copies per window | `2` |
| `NOISE_LEVEL` | Augmentation noise level (NOT scenario noise) | `0.01` |
| `NUM_SUBJECTS` | Number of subjects to process | `10` |

**Notes:**
- Set `FS` = `ORIGINAL_FS` to skip resampling
- Overlap = `1.0 - (STRIDE_SEC / WINDOW_SEC)`
- `STRIDE_SEC = WINDOW_SEC` → no overlap
- `STRIDE_SEC = 0.5 * WINDOW_SEC` → 50% overlap

### Seeds (Reproducibility)

| Parameter | Controls | Recommendation |
|-----------|----------|----------------|
| `SEED` | Augmentation noise | Keep constant for consistency |
| `NOISE_SCENARIO_SEED` | Scenario noise + sensor selection | Change to get different corruption patterns |

**Important:** Using the same seeds will produce identical datasets!

### Scenario Noise Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `SCENARIO_NOISE_TYPE` | Type of corruption: `None`, `"AWGN"`, `"DROPOUT"`, `"WEAK_SIGNAL"` | `"AWGN"` |
| `SCENARIO_NOISE_SENSOR_POOL` | List of sensors that CAN be corrupted | `["Acc_chest", "Acc_ankle", "Mag_arm"]` |
| `SCENARIO_NOISE_NUM_SENSORS` | How many sensors to corrupt per window | `1` or `2` |

**Noise Type Parameters:**

- **AWGN** (Additive White Gaussian Noise):
  ```python
  AWGN_RMS_RATIO = 0.2  # Noise RMS as fraction of signal RMS (applied to raw signal)
  ```
  Applied to raw sensor measurements before resampling and z-score normalization,
  simulating hardware measurement noise.

- **DROPOUT**:
  ```python
  DROPOUT_VALUE = 0.0  # Value to set (typically 0)
  ```
  Sets entire window to constant value for that sensor.

- **WEAK_SIGNAL**:
  ```python
  WEAK_SIGNAL_FACTOR = 0.2  # Multiplication factor (0.0 to 1.0)
  ```
  Multiplies signal by factor < 1 to simulate weak reception.

---

## Scenario Noise Types

### 1. None (Clean Dataset)

```python
SCENARIO_NOISE_TYPE = None
```

No scenario noise applied. Only augmentation noise is added.

**Use case:** Baseline performance, optimal conditions.

### 2. AWGN (Additive White Gaussian Noise)

```python
SCENARIO_NOISE_TYPE = "AWGN"
AWGN_RMS_RATIO = 0.2
```

Adds Gaussian noise with RMS proportional to the signal RMS per channel.
Applied to **raw sensor measurements** before resampling and z-score normalization.

**Use case:** Hardware measurement noise, sensor noise floor, electromagnetic interference.

**Interpretation:** Noise RMS = `rms_ratio` × Signal RMS (per channel)
- `rms_ratio = 0.1`: Low noise (10% of signal RMS)
- `rms_ratio = 0.2`: Moderate noise (20% of signal RMS) 
- `rms_ratio = 0.3`: High noise (30% of signal RMS)

### 3. DROPOUT

```python
SCENARIO_NOISE_TYPE = "DROPOUT"
DROPOUT_VALUE = 0.0
```

Sets entire window to constant value (all channels).

**Use case:** Sensor failure, disconnection, battery dead.

### 4. WEAK_SIGNAL

```python
SCENARIO_NOISE_TYPE = "WEAK_SIGNAL"
WEAK_SIGNAL_FACTOR = 0.2
```

Multiplies signal by factor < 1.

**Use case:** Poor contact, low battery, distance from body.

**Interpretation:**
- `factor = 0.5`: Signal at 50% strength
- `factor = 0.2`: Signal at 20% strength
- `factor = 0.1`: Very weak signal

---

## Output Format

### Directory Naming Convention

Pattern: `fs{FS}_s{STRIDE}_w{WINDOW}_aug{AUG_SIZE}[_{NOISE_TAG}]`

Examples:
- Clean: `fs50_s1_w1_aug2`
- AWGN: `fs50_s1_w1_aug2_AWGN_s0p3_n1`
- Dropout: `fs50_s1_w1_aug2_DROPOUT_drop_n2`
- Weak signal: `fs50_s1_w1_aug2_WEAK_SIGNAL_w0p2_n1`

Where:
- `s0p3` means sigma=0.3 (decimal point → 'p')
- `n1` means 1 sensor corrupted per window

### Files Generated

#### Per-Sensor Files (8 sensors × 2 files = 16 files)

**NPZ Format** (`Acc_chest.npz`, `ECG.npz`, etc.):
```python
data = np.load("Acc_chest.npz")
X = data['X']                    # (N, C, L) - sensor data
y = data['y']                    # (N,) - labels (0-11)
subject_id = data['subject_id']  # (N,) - subject IDs (1-10)
base_window_idx = data['base_window_idx']  # (N,) - base window index
fs = data['fs']                  # Sampling rate
window_len = data['window_len']  # Window length in samples
```

**TXT Format** (`Acc_chest.txt`, `ECG.txt`, etc.):
- Each row: `[flattened_channels, label]`
- Channels flattened as: `[ch1_t1, ch1_t2, ..., ch1_tL, ch2_t1, ..., chC_tL, label]`

#### Metadata Files

**`info.txt`**: Human-readable configuration summary

**`corruption_log.jsonl`**: Per-window corruption details
```json
{"window_idx": 0, "subject_id": 1, "label": 1, "corrupted_sensors": ["Acc_chest"]}
{"window_idx": 1, "subject_id": 1, "label": 1, "corrupted_sensors": []}
```

**`corruption_summary.json`**: Overall statistics
```json
{
  "total_windows": 5000,
  "corruption_counts": {
    "Acc_chest": 2500,
    "Acc_ankle": 2480
  },
  "corruption_percentages": {
    "Acc_chest": 50.0,
    "Acc_ankle": 49.6
  }
}
```

---

## Usage Examples

### Example 1: Clean Dataset for Baseline

```python
# Configuration
SCENARIO_NOISE_TYPE = None
FS = 50
WINDOW_SEC = 1.0
STRIDE_SEC = 1.0
AUG_SIZE = 2
```

**Result:** `fs50_s1_w1_aug2/`
- No corruption
- Each base window → 2 augmented samples

### Example 2: Test Robustness to One Noisy Sensor

```python
SCENARIO_NOISE_TYPE = "AWGN"
AWGN_SIGMA = 0.3
SCENARIO_NOISE_SENSOR_POOL = ["Acc_chest", "Acc_ankle"]
SCENARIO_NOISE_NUM_SENSORS = 1  # Only 1 sensor corrupted per window
```

**Result:** `fs50_s1_w1_aug2_AWGN_s0p3_n1/`
- Each window: randomly choose 1 sensor from pool
- ~50% windows have Acc_chest corrupted
- ~50% windows have Acc_ankle corrupted

### Example 3: Test with Multiple Sensor Failures

```python
SCENARIO_NOISE_TYPE = "DROPOUT"
SCENARIO_NOISE_SENSOR_POOL = ["Acc_chest", "Acc_ankle", "Gyro_ankle"]
SCENARIO_NOISE_NUM_SENSORS = 2  # 2 sensors fail per window
```

**Result:** `fs50_s1_w1_aug2_DROPOUT_drop_n2/`
- Each window: randomly choose 2 sensors from pool
- Those 2 sensors set to 0 for that window

### Example 4: Downsampled Data with Weak Signals

```python
ORIGINAL_FS = 50
FS = 10  # Downsample to 10 Hz
SCENARIO_NOISE_TYPE = "WEAK_SIGNAL"
WEAK_SIGNAL_FACTOR = 0.2
SCENARIO_NOISE_SENSOR_POOL = ["Mag_ankle", "Mag_arm"]
SCENARIO_NOISE_NUM_SENSORS = 1
```

**Result:** `fs10_s1_w1_aug2_WEAK_SIGNAL_w0p2_n1/`
- Data resampled to 10 Hz
- 1 magnetometer has weak signal (20% strength) per window

### Example 5: High Overlap for Dense Features

```python
WINDOW_SEC = 2.0
STRIDE_SEC = 0.5  # 75% overlap
AUG_SIZE = 1      # No augmentation
SCENARIO_NOISE_TYPE = None
```

**Result:** `fs50_s0p5_w2_aug1/`
- 2-second windows
- 0.5-second stride (75% overlap)
- Single copy per window

---

## Understanding the Output

### Index Alignment

**Key concept:** Same index across different sensor files = same time window.

Example with `AUG_SIZE = 2`:
```
Window 0 → samples [0, 1] in all sensor files
Window 1 → samples [2, 3] in all sensor files
Window 2 → samples [4, 5] in all sensor files
```

Use `base_window_idx` to map back:
```python
data = np.load("Acc_chest.npz")
sample_idx = 5
base_idx = data['base_window_idx'][sample_idx]  # → 2

# Look up which sensors were corrupted for this base window
import json
with open("corruption_log.jsonl") as f:
    log = [json.loads(line) for line in f]
    
corrupted = log[base_idx]['corrupted_sensors']
print(f"Sample {sample_idx} comes from base window {base_idx}")
print(f"Corrupted sensors: {corrupted}")
```

### Corruption Statistics

After generation, check `corruption_summary.json`:

```json
{
  "total_windows": 5472,
  "corruption_counts": {
    "Acc_chest": 2736,
    "Acc_ankle": 2736
  },
  "corruption_percentages": {
    "Acc_chest": 50.0,
    "Acc_ankle": 50.0
  }
}
```

This shows balanced corruption when `SCENARIO_NOISE_NUM_SENSORS = 1` and pool size = 2.

---

## Advanced Topics

### Adding New Noise Types

1. **Add parameters to config section:**
   ```python
   # Your new noise parameters
   MY_NOISE_PARAM = 0.5
   ```

2. **Create noise function:**
   ```python
   def apply_my_noise(win: np.ndarray, param: float) -> np.ndarray:
       """
       Apply custom noise.
       win: shape (L, C)
       """
       # Your implementation
       return modified_win
   ```

3. **Update `apply_scenario_noise()`:**
   ```python
   def apply_scenario_noise(...):
       if noise_type == "MY_NOISE":
           return apply_my_noise(win, MY_NOISE_PARAM)
       # ... existing types
   ```

4. **Update `make_scenario_tag()`:**
   ```python
   def make_scenario_tag():
       if SCENARIO_NOISE_TYPE == "MY_NOISE":
           tag += f"_my{MY_NOISE_PARAM}".replace(".", "p")
       # ... existing types
   ```

### Reproducibility Guarantee

The generator ensures:
1. **Same seed → same output** (bit-for-bit identical)
2. **Loop-order independence:** Adding/removing sensors doesn't change corruption for other sensors
3. **Deterministic RNG per (window, sensor):** Each window-sensor pair gets unique, reproducible noise

Seed formula:
```python
seed = (NOISE_SCENARIO_SEED + window_idx * 9176 + hash(sensor_name) % 100000) & 0xffffffff
```

### Memory Optimization

For large datasets:
- Process one sensor at a time (already implemented)
- Data is cached per subject during sensor processing
- Use `np.savez_compressed` for smaller file sizes

### Validation

Quick sanity checks:
```python
# Check alignment across sensors
data1 = np.load("Acc_chest.npz")
data2 = np.load("Acc_ankle.npz")

assert np.array_equal(data1['y'], data2['y'])
assert np.array_equal(data1['subject_id'], data2['subject_id'])
assert np.array_equal(data1['base_window_idx'], data2['base_window_idx'])
print("✓ Alignment verified")

# Check corruption was applied
if SCENARIO_NOISE_TYPE is not None:
    import json
    with open("corruption_log.jsonl") as f:
        log = [json.loads(line) for line in f]
    
    num_corrupted = sum(1 for entry in log if len(entry['corrupted_sensors']) > 0)
    print(f"Corrupted windows: {num_corrupted}/{len(log)}")
```

---

## Troubleshooting

### Issue: "No windows generated"

**Cause:** `WINDOW_SEC` is too large or `STRIDE_SEC` is too large.

**Solution:** Reduce `WINDOW_SEC` or check that dataset files exist.

### Issue: Different corruption patterns than expected

**Cause:** `NOISE_SCENARIO_SEED` was changed.

**Solution:** Use same seed for reproducibility, or intentionally change it for different patterns.

### Issue: Can't map sample to corruption info

**Problem:** With `AUG_SIZE > 1`, sample index ≠ window index.

**Solution:** Use `base_window_idx`:
```python
data = np.load("Acc_chest.npz")
base_idx = data['base_window_idx'][sample_idx]
# Look up log[base_idx]
```

### Issue: Memory error

**Solution:** 
- Reduce `NUM_SUBJECTS`
- Process in batches (modify code to process subjects 1-5, then 6-10)

---

## Summary

**DataGenerator v2** provides:
- ✅ Flexible windowing and resampling
- ✅ Configurable augmentation
- ✅ Multiple noise types for robustness testing
- ✅ Per-window corruption logging
- ✅ Reproducible and deterministic
- ✅ Index-aligned across sensors
- ✅ Easy to extend with new noise types

**Typical workflow:**
1. Configure parameters in `DataGenerator_v2.py`
2. Run: `python DataGenerator_v2.py`
3. Check output folder for generated data
4. Review `corruption_summary.json` for statistics
5. Use `base_window_idx` to map samples to corruption info

---

## Contact & Support

For issues or questions about the data generator, refer to the source code comments or the `info.txt` file in each generated dataset folder.
