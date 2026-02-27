## Tremor Dataset Generation Guide

This guide explains how to generate datasets with Parkinson's tremor labels using the new tremor generation system.

---

## 📋 Overview

The tremor dataset generation system consists of:

1. **`tremor_parkinson_config.py`**: Configuration for 10 Parkinson subjects with varying tremor severity
2. **`Tremor.py`**: Van der Pol oscillator-based tremor generation with frequency control
3. **`DataGenerator_Tremor.py`**: Main script to generate tremor-labeled datasets
4. **`test_tremor_dataset.py`**: Script to inspect and verify generated datasets

---

## 🎯 What's Generated

Each dataset contains windows labeled with **4 tremor parameters**:

| Label | Description | Range | Clean Value |
|-------|-------------|-------|-------------|
| `tremor_freq` | Tremor frequency (Hz) | 3.5-7.0 | 0.0 |
| `tremor_acc_rms` | Accelerometer RMS (m/s²) | 0.07-6.0 | 0.0 |
| `tremor_gyro_rms` | Gyroscope RMS (deg/s) | Calculated | 0.0 |
| `tremor_score` | Severity score | 0-4 | 0 |

### Severity Scores

- **Score 0**: No tremor (< 0.07 m/s²)
- **Score 1**: Mild (0.07-0.15 m/s²) 
- **Score 2**: Mild-Moderate (0.15-0.7 m/s²)
- **Score 3**: Moderate-Severe (0.7-2.5 m/s²)
- **Score 4**: Severe (2.5-6.0 m/s²)

---

## 🚀 Quick Start

### 1. Generate Clean Dataset (No Tremor)

```python
# In DataGenerator_Tremor.py, set:
GENERATE_TREMOR = False

# Then run:
python DataGenerator_Tremor.py
# Or uncomment at bottom: generate_tremor_dataset("s2_w2_tremor_clean")
```

**Result**: Dataset with all tremor labels = 0 (healthy baseline)

---

### 2. Generate Tremor Dataset

```python
# In DataGenerator_Tremor.py, set:
GENERATE_TREMOR = True
USE_TREMOR_JITTER = True
TREMOR_SEED = 42

# Then run:
python DataGenerator_Tremor.py
# Or uncomment at bottom: generate_tremor_dataset("s2_w2_tremor_parkinson")
```

**Result**: Dataset with Parkinson-specific tremor and per-window labels

---

## 📊 Data Format

### NPZ Files

Each sensor saves an NPZ file with:

```python
data = np.load("data/Tremor_datagenerator_files/s2_w2_tremor_parkinson/Acc_arm.npz")

# Standard arrays
X = data['X']              # (N, C, L) sensor data
y = data['y']              # (N,) activity labels (0-11)
subject_id = data['subject_id']  # (N,) subject IDs (1-10)
base_window_idx = data['base_window_idx']  # (N,) window index

# NEW: Tremor labels
tremor_freq = data['tremor_freq']      # (N,) frequency (Hz)
tremor_acc_rms = data['tremor_acc_rms']  # (N,) acc RMS (m/s²)
tremor_gyro_rms = data['tremor_gyro_rms']  # (N,) gyro RMS (deg/s)
tremor_score = data['tremor_score']    # (N,) severity score (0-4)

# Metadata
fs = data['fs']            # Sampling frequency
window_len = data['window_len']  # Window length in samples
```

### TXT Files

CSV format with **all labels included**: `(N, C×L+7)` where:

| Column Range | Content | Description |
|--------------|---------|-------------|
| 1 to C×L | Sensor data | Flattened channels (all samples from ch0, then ch1, then ch2) |
| C×L + 1 | Activity | Activity label (1-12, 1-indexed) |
| C×L + 2 | Subject ID | Subject identifier (1-10) |
| C×L + 3 | Window index | Base window index for tracking |
| C×L + 4 | Tremor freq | Tremor frequency (Hz) |
| C×L + 5 | Tremor acc RMS | Accelerometer RMS (m/s²) |
| C×L + 6 | Tremor gyro RMS | Gyroscope RMS (deg/s) |
| C×L + 7 | Tremor score | Severity score (0-4) |

**Example for Acc_ankle (C=3, L=100):**
- Total columns: 307
- Columns 1-300: Sensor data
- Column 301: Activity label
- Column 302: Subject ID
- Column 303: Window index
- Column 304: Tremor frequency
- Column 305: Tremor acc RMS
- Column 306: Tremor gyro RMS
- Column 307: Tremor score

---

## 🔍 Inspect Generated Dataset

```bash
python test_tremor_dataset.py
```

This will:
- Load and display tremor label statistics
- Show sample windows with labels
- Compare clean vs tremor datasets
- Provide usage examples

---

## ⚙️ Configuration Options

### Basic Parameters

```python
ORIGINAL_FS = 50      # Original sampling rate (Hz)
FS = 30               # Target sampling rate (Hz)
WINDOW_SEC = 2.0      # Window length (seconds)
STRIDE_SEC = 2.0      # Window stride (seconds)
AUG_SIZE = 2          # Augmentation copies per window
```

### Tremor Parameters

```python
# Main toggle
GENERATE_TREMOR = True/False  # Enable/disable tremor generation

# Van der Pol oscillator
TREMOR_MU = 1.0       # Nonlinearity (controls limit cycle)
TREMOR_SIGMA = 0.5    # Stochastic noise (controls irregularity)
TREMOR_DT = 0.001     # Integration time step

# Variability
USE_TREMOR_JITTER = True  # Window-to-window variability
TREMOR_JITTER_STD = 0.15  # 15% RMS variability

# Seeds
SEED = 0              # Augmentation seed
TREMOR_SEED = 42      # Tremor generation seed (change for different realizations)
```

---

## 📐 Tremor Model

### Subject Configuration

10 subjects with varying severity (from `tremor_parkinson_config.py`):

| Subjects | Severity | Acc RMS (m/s²) | Frequency (Hz) |
|----------|----------|----------------|----------------|
| 1-3 | Mild (Score 1) | 0.08-0.12 | 5.2-5.8 |
| 4-6 | Mild-Moderate (Score 2) | 0.30-0.45 | 4.8-5.3 |
| 7-8 | Moderate-Severe (Score 3) | 1.0-1.3 | 4.2-4.5 |
| 9-10 | Severe (Score 4) | 2.8-3.0 | 3.8-4.0 |

### RMS Calculation

Per-window tremor RMS is calculated as:

```
RMS_acc = A_subject[subj] × C[body_part] × beta[activity] × (1 + jitter)
RMS_gyro = k_g(RMS_acc) × RMS_acc
Frequency = FREQ_TREMOR[subj]  (constant per subject)
```

**Body part scaling (C)**:
- Arm: 1.0 (most affected)
- Ankle: 0.8
- Chest: 0.6

**Activity modulation (beta)**:
- Resting (sitting, standing): 1.0 (high tremor)
- Walking: 0.9
- Jogging: 0.7 (reduced tremor)
- Running: 0.65

---

## 🎓 Usage Examples

### Example 1: Filter by Severity

```python
data = np.load("Acc_arm.npz")
X = data['X']
tremor_score = data['tremor_score']

# Get only mild tremor windows
mild_windows = X[tremor_score == 1]

# Get severe tremor windows
severe_windows = X[tremor_score == 4]
```

### Example 1b: Load from TXT File

```python
import numpy as np

# Load TXT file (all data in one CSV)
txt_data = np.loadtxt("data/Tremor_datagenerator_files/s2_w2_tremor_parkinson/Acc_arm.txt", delimiter=",")

# For Acc sensors: C=3 channels, L=100 samples → C×L = 300
# Extract columns
sensor_data = txt_data[:, :300]          # Columns 1-300
activity = txt_data[:, 300] - 1          # Column 301 (convert to 0-indexed)
subject_id = txt_data[:, 301].astype(int)  # Column 302
window_idx = txt_data[:, 302].astype(int)  # Column 303
tremor_freq = txt_data[:, 303]           # Column 304
tremor_acc_rms = txt_data[:, 304]        # Column 305
tremor_gyro_rms = txt_data[:, 305]       # Column 306
tremor_score = txt_data[:, 306].astype(int)  # Column 307

# Reshape sensor data from (N, C×L) to (N, C, L)
N = sensor_data.shape[0]
X = sensor_data.reshape(N, 3, 100)  # (N, 3, 100)

print(f"Loaded {N} windows from TXT file")
print(f"Tremor freq range: [{tremor_freq.min():.1f}, {tremor_freq.max():.1f}] Hz")
```

### Example 2: Filter by Frequency

```python
tremor_freq = data['tremor_freq']

# Low frequency tremor (typical of severe cases)
low_freq_windows = X[(tremor_freq >= 3.5) & (tremor_freq < 4.5)]

# High frequency tremor (typical of mild cases)
high_freq_windows = X[(tremor_freq >= 5.5) & (tremor_freq <= 7.0)]
```

### Example 3: Regression Task

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Flatten data
X_flat = X.reshape(X.shape[0], -1)

# Predict tremor RMS from sensor data
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, data['tremor_acc_rms'], test_size=0.2
)

model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Example 4: Multi-Task Learning

```python
# Predict both activity and tremor severity
y_activity = data['y']
y_tremor = data['tremor_score']

# Stack targets
y_multi = np.column_stack([y_activity, y_tremor])

# Train model with multiple outputs
```

### Example 5: Analyze Tremor vs Activity

```python
import pandas as pd

df = pd.DataFrame({
    'activity': data['y'],
    'tremor_freq': data['tremor_freq'],
    'tremor_rms': data['tremor_acc_rms'],
    'score': data['tremor_score']
})

# Group by activity
grouped = df.groupby('activity').agg({
    'tremor_freq': 'mean',
    'tremor_rms': 'mean',
    'score': ['mean', 'std']
})

print(grouped)
```

---

## 🔬 Technical Details

### Tremor Generation Pipeline

1. **Window Extraction**: Extract windows from raw MHEALTH data
2. **Tremor Injection**: Apply tremor to RAW signal (before preprocessing)
3. **Resampling**: Downsample from 50 Hz to target FS (e.g., 30 Hz)
4. **Normalization**: Apply z-score normalization per channel
5. **Augmentation**: Create multiple copies with small noise
6. **Labeling**: Store tremor parameters for each window

### Why Apply Tremor Before Preprocessing?

- Tremor applies to raw sensor measurements (physical signal)
- Resampling after tremor preserves natural frequency characteristics
- Z-score normalization then makes amplitude-invariant features

### Frequency Control

The van der Pol oscillator generates tremor at specific frequencies:

```python
omega = 2 * π * freq_hz  # Convert Hz to angular frequency

# Oscillator equations:
dx1/dt = omega * x2
dx2/dt = omega * (mu * (1 - x1²) * x2 - x1) + sigma * dW/dt
```

This ensures tremor has exact frequency matching `FREQ_TREMOR[subject_id]`.

---

## 📁 Output Structure

Tremor datasets are stored separately from regular datasets due to their additional label structure:

```
data/
├── Datagenerator_files/          # Regular datasets (from DataGenerator_v3.py)
│   └── s2_w2_aug2/
│
└── Tremor_datagenerator_files/   # Tremor-labeled datasets (NEW)
    ├── s2_w2_tremor_clean/       # Clean dataset (no tremor, labels=0)
    │   ├── Acc_arm.npz
    │   ├── Acc_arm.txt
    │   ├── Acc_ankle.npz
    │   ├── ...
    │   └── info.txt
    │
    └── s2_w2_tremor_parkinson/   # Tremor dataset (with Parkinson model)
        ├── Acc_arm.npz
        ├── Acc_arm.txt
        ├── Acc_ankle.npz
        ├── ...
        ├── info.txt
        └── tremor_parkinson_params.txt  # Detailed tremor configuration
```

---

## 🛠️ Troubleshooting

### Issue: OneDrive Sync Timeout

**Solution**: The script has retry logic. If it fails, ensure OneDrive has fully synced the MHEALTHDATASET folder.

### Issue: Memory Error

**Solution**: Reduce `NUM_SUBJECTS` or process subjects in batches.

### Issue: Different Tremor Realization

**Solution**: Change `TREMOR_SEED` to get different tremor patterns with same statistics.

### Issue: Clean Dataset Still Has Tremor

**Solution**: Make sure `GENERATE_TREMOR = False` before running.

---

## 📚 References

- **Parkinson's Tremor**: Typically 3.5-7 Hz resting tremor
- **Van der Pol Oscillator**: Generates physiologically realistic tremor patterns
- **Subject Configuration**: Based on clinical severity scales

---

## ✅ Checklist

Before generating datasets:

- [ ] Set `GENERATE_TREMOR` appropriately (True/False)
- [ ] Choose `TREMOR_SEED` for reproducibility
- [ ] Configure `USE_TREMOR_JITTER` for variability
- [ ] Set output variant name
- [ ] Verify MHEALTH data is accessible

After generation:

- [ ] Run `test_tremor_dataset.py` to verify
- [ ] Check `info.txt` for configuration
- [ ] Inspect tremor label distributions
- [ ] Validate NPZ files load correctly

---

## 🎉 Summary

You now have a complete system to generate tremor-labeled datasets with:

✅ Subject-specific Parkinson's tremor (10 subjects, varying severity)  
✅ Controlled tremor frequency (3.5-7.0 Hz per subject)  
✅ Per-window labels: freq, acc_rms, gyro_rms, score  
✅ Clean baseline datasets (tremor labels = 0)  
✅ Compatible with existing DataGenerator_v3.py workflow  

Use these datasets for:
- Tremor severity classification
- Tremor parameter regression
- Activity recognition with tremor
- Robust model evaluation
- Multi-task learning

---

**Next Steps**: Run `DataGenerator_Tremor.py` and start training! 🚀
