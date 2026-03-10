# Tremor Generation Overview - Complete Pipeline

This document provides a comprehensive overview of the Parkinson's tremor generation pipeline, including all formulas, RMS calculations, and body part derivations.

---

## Table of Contents
1. [Data Source and Output Structure](#data-source-and-output-structure)
2. [Tremor Physics Model](#tremor-physics-model)
3. [Sampling Methods](#sampling-methods)
4. [Body Part Tremor Derivation](#body-part-tremor-derivation)
5. [Sensor-Specific Augmentation](#sensor-specific-augmentation)
6. [Mathematical Formulas](#mathematical-formulas)

---

## 1. Data Source and Output Structure

### 1.1 Input Data (Source Dataset)

**Location:** `data/MHEALTHDATASET/`

**Files:**
```
mHealth_subject1.log
mHealth_subject2.log
...
mHealth_subject10.log
```

**Format:** Space-separated values (23 sensor columns + 1 label column)

**Sensor columns (0-indexed):**
- **Chest sensors:**
  - Acc_chest: [0, 1, 2] (3-axis accelerometer)
  - ECG: [3, 4] (2-channel ECG)
- **Ankle sensors:**
  - Acc_ankle: [5, 6, 7]
  - Gyro_ankle: [8, 9, 10]
  - Mag_ankle: [11, 12, 13]
- **Arm sensors:**
  - Acc_arm: [14, 15, 16]
  - Gyro_arm: [17, 18, 19]
  - Mag_arm: [20, 21, 22]
- **Activity label:** Column 23 (values 1-12, or 0 for unlabeled)

**Original sampling rate:** 50 Hz (all sensors synchronized)

**Dataset characteristics:**
- 10 subjects performing 12 activities
- Activities: Standing, Sitting, Lying down, Walking, Climbing stairs, etc.
- Only windows with consistent activity labels (all frames ≥ 1) are used

---

### 1.2 Output Data (Generated Datasets)

**Base output directory:** `data/Tremor_datagenerator_files/`

**Dataset variants:** Generated with different tremor augmentation modes and sampling frequencies
```
data/Tremor_datagenerator_files/
├── s2_w2_fs50_tremor_clean/               # 50 Hz, no tremor (100% score=0)
├── s2_w2_fs50_tremor_mild_mod/            # 50 Hz, mild tremor (50% score=1, 50% score=2)
├── s2_w2_fs50_tremor_mod_severe/          # 50 Hz, severe tremor (50% score=3, 50% score=4)
├── s2_w2_fs30_tremor_clean/               # 30 Hz, no tremor
├── s2_w2_fs30_tremor_mild_mod/            # 30 Hz, mild tremor
├── s2_w2_fs30_tremor_mod_severe/          # 30 Hz, severe tremor
└── [other variants...]
```

**Naming convention:** `s{STRIDE}_w{WINDOW}_fs{FS}_tremor_{AUGMENT_MODE}`
- `s2`: 2-second stride
- `w2`: 2-second window
- `fs50`: 50 Hz sampling frequency
- `tremor_clean`: No tremor (augment_mode="clean")
- `tremor_mild_mod`: Mild/moderate tremor (augment_mode="mild_mod")
- `tremor_mod_severe`: Moderate/severe tremor (augment_mode="mod_severe")

**Why include sampling frequency in folder name?**
- Allows multiple datasets with different sampling rates (e.g., 50 Hz, 30 Hz, 20 Hz)
- Prevents accidental overwriting when regenerating with different FS
- Makes it clear which FS was used for each dataset

---

### 1.3 Output Files Per Variant

Each variant directory contains:

#### **Sensor data files (8 sensors × 2 formats = 16 files):**

**NPZ format (compressed NumPy arrays):**
```
Acc_ankle.npz
Acc_arm.npz
Acc_chest.npz
ECG.npz
Gyro_ankle.npz
Gyro_arm.npz
Mag_ankle.npz
Mag_arm.npz
```

**NPZ file structure:**
```python
npz = np.load("Acc_ankle.npz")

# Data arrays
npz['X']                 # Shape: (N, C, L) - sensor windows
                         # N = number of windows (~3,313 for s2_w2)
                         # C = number of channels (3 for most, 2 for ECG)
                         # L = window length in samples (100 for 2s @ 50Hz)

npz['y']                 # Shape: (N,) - activity labels (0-11, from original 1-12)
npz['subject_id']        # Shape: (N,) - subject IDs (1-10)
npz['base_window_idx']   # Shape: (N,) - window index in original subject data

# Tremor labels (per window)
npz['tremor_freq']       # Shape: (N,) - tremor frequency (Hz, 0 if no tremor)
npz['tremor_acc_rms']    # Shape: (N,) - accelerometer RMS (m/s²)
npz['tremor_gyro_rms']   # Shape: (N,) - gyroscope RMS (deg/s)
npz['tremor_score']      # Shape: (N,) - severity score (0-4)
```

**TXT format (CSV, for compatibility):**
```
Acc_ankle.txt
Acc_arm.txt
... (same 8 sensors)
```

**TXT file structure:** One row per window
```
[sensor_data_flattened] | activity | subject | base_idx | tremor_freq | tremor_acc_rms | tremor_gyro_rms | tremor_score
     C×L columns             1          1          1            1              1                 1               1
```

**Example row dimensions:**
- 3-axis sensors: 300 sensor columns + 7 label columns = **307 total**
- ECG (2-axis): 200 sensor columns + 7 label columns = **207 total**

---

#### **Configuration files:**

**`info.txt`** - Dataset metadata and documentation
```
Tremor-Labeled Dataset Configuration
================================================================================
Basic Parameters:
  - Original FS: 50 Hz
  - Target FS: 50 Hz
  - Window size: 2.0 s (100 samples)
  - Stride: 2.0 s (100 samples)
  - Overlap: 0.00
  - Augmentation copies (AUG_SIZE): 1
  - Augmentation noise level: 0.01
  - Augmentation seed: 0

Tremor Configuration:
  - Generate tremor: True
  - Tremor seed: 42
  - Use jitter: True
  - Jitter std: 0.15

Tremor Labels (per window):
  - tremor_freq: Tremor frequency (Hz) [3.5-7.0]
  - tremor_acc_rms: Accelerometer RMS (m/s²)
  - tremor_gyro_rms: Gyroscope RMS (deg/s)
  - tremor_score: Severity score [0-4]

Augmentation Strategy:
  - Tremor-free sensors: ['Acc_chest', 'ECG']
  - Rotation-based mag sensors: ['Mag_arm', 'Mag_ankle']
  - (Full policy documentation...)
```

**`tremor_parkinson_params.txt`** - Tremor model parameters (only if tremor generated)
```
Tremor Noise Parameters - Parkinson Patient Model
================================================================================
Van der Pol Oscillator:
  - mu: 1.0
  - sigma: 0.5
  - dt: 0.001

Sampling method: interval / subject
Augment mode: clean / mild_mod / mod_severe
Window-to-window jitter: enabled (std=0.15)
Scenario seed: 42
```

---

### 1.4 Data Processing Pipeline

**Flow:** Raw MHEALTH data → Windowing → Tremor injection → Resampling → Z-score normalization → Save

```
1. Load subject data from MHEALTHDATASET/
   ↓
2. Extract windows (2s, non-overlapping)
   ↓
3. Filter: Keep only windows with consistent activity labels
   ↓
4. Pre-compute tremor cache for all windows
   - Generate van der Pol states
   - Scale to target RMS per body part
   - Store in cache: (window_idx, body_part) → tremor_noise
   ↓
5. Process each sensor:
   - Apply tremor (additive for acc/gyro, rotation-based for mag)
   - Apply augmentation (AWGN or rotation for tremor-free sensors)
   - Resample to target FS (if needed)
   - Z-score normalize per channel within window
   ↓
6. Add tremor labels (freq, acc_rms, gyro_rms, score)
   ↓
7. Save to NPZ and TXT formats
   ↓
8. Write info.txt and tremor_parkinson_params.txt
```

---

### 1.5 Storage Requirements

**Approximate sizes per variant:**
- NPZ files: ~15-20 MB per sensor × 8 sensors = **~120-160 MB**
- TXT files: ~50-70 MB per sensor × 8 sensors = **~400-560 MB**
- Total per variant: **~500-700 MB**

**For 3 tremor variants (clean, mild, severe):**
- Total storage: **~1.5-2.1 GB**

---

### 1.6 Usage Example

**Loading data for training:**
```python
import numpy as np

# Load sensor data
data = np.load("data/Tremor_datagenerator_files/s2_w2_fs50_tremor_mod_severe/Acc_ankle.npz")

X = data['X']                # Shape: (3313, 3, 100)
y_activity = data['y']       # Activity labels (0-11)
y_tremor = data['tremor_score']  # Tremor severity (0-4)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_tremor, test_size=0.3, random_state=42, stratify=y_tremor
)

# Train model...
```

**Loading data for fusion:**
```python
# Load embeddings extracted by CNNs
variant = "s2_w2_fs50_tremor_mod_severe"
embeddings_dir = f"data/Tremor_datagenerator_files/{variant}/ExtractedFeatures"

acc_ankle = np.load(f"{embeddings_dir}/Acc_ankle_embeddings.npz")
acc_arm = np.load(f"{embeddings_dir}/Acc_arm_embeddings.npz")
# ... load all 8 sensors

# Concatenate for fusion (8 sensors × 128 dim = 1024 features)
X_train = np.concatenate([
    acc_ankle['train_embeddings'],
    acc_arm['train_embeddings'],
    # ... all 8 sensors
], axis=1)

y_train = acc_ankle['train_labels']  # Tremor scores (same across all sensors)
```

---

## 2. Tremor Physics Model

### Van der Pol Oscillator
Tremor signals are generated using a stochastic van der Pol oscillator:

```
dx₁/dt = ω · x₂
dx₂/dt = ω · (μ(1 - x₁²)x₂ - x₁) + σ · dW/dt
```

**Parameters:**
- **μ = 1.0**: Nonlinearity parameter (controls limit cycle amplitude)
- **σ = 0.5**: Stochastic noise intensity (controls irregularity)
- **ω = 2πf**: Angular frequency (f = tremor frequency in Hz)
- **dt = 0.001**: Integration time step

**States:**
- **x₁**: Position state (used for accelerometer tremor)
- **x₂**: Velocity state (used for gyroscope tremor)

The oscillator is integrated using **Euler-Maruyama** method and resampled to sensor frequency (50 Hz).

---

## 3. Sampling Methods

The pipeline supports two tremor parameter sampling strategies:

### 3.1 Subject-Based Sampling (`sampling_method="subject"`)

**Use case:** Patient-specific tremor simulation

**Formula:**
```
RMS_acc = A_subject[subj] × C[body_part] × β[activity] × (1 + jitter)
RMS_gyro = k_g(RMS_acc) × RMS_acc
Frequency = FREQ_TREMOR[subject_id]
```

**Parameters:**
- `A_subject[subj]`: Subject-specific baseline severity (accelerometer RMS)
- `C[body_part]`: Body part sensitivity factor
  - `C['arm'] = 1.0` (reference)
  - `C['ankle'] = 0.8`
  - `C['chest'] = 0.6`
- `β[activity]`: Activity modulation factor
  - High at rest (e.g., β=1.2 for sitting)
  - Reduced during movement (e.g., β=0.7 for walking)
- `jitter`: Random variation ~N(0, 0.15) for window-to-window variability

**Defined subjects:**
- Subject 1: A=0.8, freq=4.5 Hz
- Subject 2: A=1.5, freq=5.2 Hz
- Subject 5: A=2.3, freq=6.0 Hz
- *(See `tremor_parkinson_config.py` for full list)*

---

### 3.2 Interval-Based Sampling (`sampling_method="interval"`)

**Use case:** Data augmentation with controlled severity distribution

**Process:**
1. **Sample severity score** based on augment_mode:
   - `"clean"`: 100% score=0 (no tremor)
   - `"mild_mod"`: 50% score=1, 50% score=2
   - `"mod_severe"`: 50% score=3, 50% score=4

2. **Sample RMS from severity range:**
   ```
   RMS_acc ~ Uniform(RMS_min[score], RMS_max[score]) × (1 + variation)
   ```
   
   **Severity ranges (SCORE_RMS_RANGE):**
   - Score 0 (none): 0.0 m/s²
   - Score 1 (mild): 0.05 - 0.10 m/s²
   - Score 2 (mild-moderate): 0.25 - 0.50 m/s²
   - Score 3 (moderate-severe): 1.10 - 2.00 m/s²
   - Score 4 (severe): 4.65 - 5.10 m/s²

3. **Sample frequency:**
   ```
   Frequency ~ Uniform(3.5, 7.0) Hz
   ```

4. **Calculate gyroscope RMS:**
   ```
   k_g = choose_kg_from_rms_acc(RMS_acc)
   RMS_gyro = k_g × RMS_acc
   ```

**Variation factor:**
- `variation ~ N(0, 0.15)` adds natural variability

---

## 4. Body Part Tremor Derivation

### 4.1 ARM (Reference Body Part)

**Subject mode:**
```
RMS_acc_arm = A_subject[subj] × C['arm'] × β[activity] × (1 + jitter)
             = A_subject[subj] × 1.0 × β[activity] × (1 + jitter)
RMS_gyro_arm = k_g(RMS_acc_arm) × RMS_acc_arm
Freq_arm = FREQ_TREMOR[subject_id]
```

**Interval mode:**
```
Score ~ sample_from_augment_mode()
RMS_acc_arm ~ Uniform(SCORE_RMS_RANGE[score]) × (1 + variation)
RMS_gyro_arm = k_g(RMS_acc_arm) × RMS_acc_arm
Freq_arm ~ Uniform(3.5, 7.0) Hz
```

**Processing:** Additive tremor directly to Acc_arm and Gyro_arm signals.

---

### 4.2 ANKLE (Scaled from ARM)

#### **Subject mode:**
```
RMS_acc_ankle = A_subject[subj] × C['ankle'] × β[activity] × (1 + jitter)
               = A_subject[subj] × 0.8 × β[activity] × (1 + jitter)
RMS_gyro_ankle = k_g(RMS_acc_ankle) × RMS_acc_ankle
Freq_ankle = FREQ_TREMOR[subject_id]  # Same as arm
```

#### **Interval mode (severity-dependent scaling):**
```
# Step 1: Get arm parameters for same window
RMS_acc_arm = arm_cache['acc_target_rms']
RMS_gyro_arm = arm_cache['gyro_target_rms']
Score = arm_cache['score']
Freq_arm = arm_cache['freq_hz']

# Step 2: Apply ankle scaling ratio
ankle_ratio = ANKLE_RATIO_BY_SCORE[score]
RMS_acc_ankle = RMS_acc_arm × ankle_ratio
RMS_gyro_ankle = RMS_gyro_arm × ankle_ratio
Freq_ankle = Freq_arm  # Same frequency as arm
```

**Ankle scaling ratios (ANKLE_RATIO_BY_SCORE):**
- Score 0: 0.00 (no tremor)
- Score 1: 0.05 (5% of arm)
- Score 2: 0.12 (12% of arm)
- Score 3: 0.22 (22% of arm)
- Score 4: 0.35 (35% of arm)

**Rationale:** Lower extremity tremor is typically less severe than upper extremity tremor in Parkinson's disease. Scaling increases with severity.

**Processing:** Additive tremor to Acc_ankle and Gyro_ankle signals.

---

### 4.3 CHEST (Tremor-Free)

**Both modes:**
```
RMS_acc_chest = 0.0  # No tremor applied
RMS_gyro_chest = 0.0  # No tremor applied
```

**Rationale:** Chest-mounted sensors primarily capture transmitted vibrations from limb tremor, not independent chest tremor. To avoid label inconsistency, chest sensors receive **alternative augmentations** instead:

- **Mild severity (score 1-2):** AWGN (Additive White Gaussian Noise)
  ```
  AWGN_std = 0.05 × rms(clean_signal)
  ```

- **Severe severity (score 3-4):** Random 3D rotation
  ```
  θ ~ Uniform(-5°, +5°) per axis
  R = Rz(θz) · Ry(θy) · Rx(θx)
  signal_augmented = R · signal_clean
  ```

**Label assignment:**
- `tremor_rms_label = 0.0` (chest is tremor-free)
- `tremor_severity_score = 0` (even if arm/ankle have tremor)

---

## 5. Sensor-Specific Augmentation

### 5.1 Accelerometer (Additive Tremor)

**Sensors:** Acc_arm, Acc_ankle

**Process:**
1. Generate 3D tremor using van der Pol state x₁
2. Scale to target RMS:
   ```
   tremor_3d = unit_random_direction() × x₁(t)
   tremor_3d_scaled = tremor_3d × (RMS_target / rms(tremor_3d))
   ```
3. Add to clean signal:
   ```
   signal_augmented = signal_clean + tremor_3d_scaled
   ```

**Sensors:** Acc_chest - **No tremor** (receives AWGN or rotation instead)

---

### 5.2 Gyroscope (Additive Tremor)

**Sensors:** Gyro_arm, Gyro_ankle

**Process:**
1. Generate 3D tremor using van der Pol state x₂ (velocity state)
2. Scale to target RMS:
   ```
   RMS_gyro = k_g(RMS_acc) × RMS_acc
   tremor_3d = unit_random_direction() × x₂(t)
   tremor_3d_scaled = tremor_3d × (RMS_gyro / rms(tremor_3d))
   ```
3. Add to clean signal:
   ```
   signal_augmented = signal_clean + tremor_3d_scaled
   ```

**k_g relationship (gyro-to-acc ratio):**
```python
if RMS_acc == 0:
    k_g = 0
elif RMS_acc < 0.5:
    k_g = 14.0
elif RMS_acc < 2.0:
    k_g = 14.0
else:
    k_g = 13.0
```

---

### 5.3 Magnetometer (Rotation-Based Tremor)

**Sensors:** Mag_arm, Mag_ankle

**Process:** Gyroscope-driven cumulative rotation (physically realistic)

1. Convert gyro tremor to rotation angles:
   ```
   Δθ(t) = gyro_tremor(t) × dt
   # where gyro_tremor has same temporal structure as additive gyro tremor
   ```

2. Build incremental rotation matrix using **Rodrigues formula**:
   ```
   axis = gyro_tremor(t) / ||gyro_tremor(t)||
   angle = ||Δθ(t)||
   K = skew_symmetric_matrix(axis)
   dR(t) = I + sin(angle)·K + (1 - cos(angle))·K²
   ```

3. Apply cumulative rotation:
   ```
   R_cumulative(0) = I (identity)
   for each timestep t:
       R_cumulative(t) = dR(t) @ R_cumulative(t-1)
       mag_augmented(t) = R_cumulative(t) @ mag_clean(t)
   ```

**Key property:** Preserves magnetometer vector norm (physically correct)
```
||mag_augmented(t)|| = ||mag_clean(t)||  ∀t
```

**Rationale:** Tremor causes orientation perturbations, not changes in magnetic field magnitude.

---

### 5.4 ECG (Tremor-Free)

**Process:** Same as Acc_chest
- **No tremor applied** (tremor_rms_label = 0.0)
- Receives AWGN or rotation augmentation based on arm severity
- Only 2 channels (vs 3 for other sensors)

---

## 6. Mathematical Formulas

### 6.1 Van der Pol State Evolution

**Discrete integration (Euler-Maruyama):**
```
x₁[k+1] = x₁[k] + dt · ω · x₂[k]
x₂[k+1] = x₂[k] + dt · ω · (μ(1 - x₁[k]²)x₂[k] - x₁[k]) + σ√(dt) · ξ[k]
```
where ξ[k] ~ N(0,1) is white noise

---

### 6.2 Gyroscope-Accelerometer Relationship

**Model:** RMS_gyro scales linearly with RMS_acc, with severity-dependent factor

```
k_g(RMS_acc) = { 0      if RMS_acc = 0
               { 14.0   if 0 < RMS_acc < 2.0
               { 13.0   if RMS_acc ≥ 2.0

RMS_gyro = k_g(RMS_acc) · RMS_acc
```

**Units:**
- RMS_acc: m/s²
- RMS_gyro: deg/s
- k_g: (deg/s) / (m/s²)

---

### 6.3 Rodrigues Rotation Formula

**Given:** rotation axis **n** (unit vector) and angle θ

**Rotation matrix:**
```
R = I + sin(θ)·K + (1 - cos(θ))·K²

where K = [  0   -nz   ny ]
          [  nz   0   -nx ]
          [ -ny   nx   0  ]
```

**Applied to magnetometer:**
```
mag_rotated = R · mag_original
```

---

### 6.4 RMS Scaling

**To achieve target RMS from arbitrary signal:**
```
signal_scaled = signal × (RMS_target / RMS_current)

where RMS_current = √(mean(signal²))
```

---

### 6.5 Severity Score from RMS

**Classification based on accelerometer RMS:**
```
score(RMS_acc) = { 0  if RMS_acc = 0
                 { 1  if 0.05 ≤ RMS_acc < 0.25
                 { 2  if 0.25 ≤ RMS_acc < 1.10
                 { 3  if 1.10 ≤ RMS_acc < 4.65
                 { 4  if 4.65 ≤ RMS_acc
```

**Labels:**
- 0: None
- 1: Mild
- 2: Mild-Moderate
- 3: Moderate-Severe
- 4: Severe

---

## Summary of Processing Order

### Interval Mode (Window-by-Window)
```
For each window:
    1. Process ARM:
       - Sample score, RMS_acc, frequency
       - Calculate RMS_gyro = k_g(RMS_acc) × RMS_acc
       - Generate tremor (acc, gyro, mag=0)
       - Store in cache
    
    2. Process ANKLE:
       - Retrieve arm parameters from cache
       - Scale: RMS_ankle = RMS_arm × ANKLE_RATIO_BY_SCORE[score]
       - Use same frequency as arm
       - Generate tremor (acc, gyro, mag=0)
       - Store in cache
    
    3. Process CHEST:
       - Set RMS = 0 (no tremor)
       - Store in cache with score=0
    
    4. Apply to sensors:
       - Acc_arm/ankle: additive tremor
       - Gyro_arm/ankle: additive tremor
       - Mag_arm/ankle: rotation-based from gyro tremor
       - Acc_chest: AWGN or rotation (no tremor label)
       - ECG: AWGN or rotation (no tremor label)
```

### Subject Mode
```
For each window:
    1. Calculate RMS for each body part using subject-specific formula
    2. All body parts get independent tremor generation
    3. Same sensor application rules as interval mode
```

---

## Configuration Files

**Primary config:** `Noise_simulation/tremor_parkinson_config.py`

**Key constants:**
- `TREMOR_FREE_SENSORS = ['Acc_chest', 'ECG']`
- `ROTATION_BASED_MAG_SENSORS = ['Mag_arm', 'Mag_ankle']`
- `ANKLE_RATIO_BY_SCORE = {0: 0.0, 1: 0.05, 2: 0.12, 3: 0.22, 4: 0.35}`
- `SCORE_RMS_RANGE = {1: (0.05, 0.1), 2: (0.25, 0.5), 3: (1.1, 2.0), 4: (4.65, 5.1)}`
- `FREQ_RANGE_HZ = (3.5, 7.0)`

---

## Validation & Diagnostics

**Diagnostic function:** `run_tremor_sanity_checks()` in `DataGenerator_Tremor.py`

**Verifies:**
1. Rotation-based mag tremor preserves vector norms
2. Additive tremor matches target RMS
3. Tremor-free sensors have zero tremor
4. Ankle scaling ratios match expected values
5. Arm-ankle frequency synchronization

**Enable:** Set `RUN_SANITY_CHECKS = True` in DataGenerator_Tremor.py

---

**Last updated:** March 2026  
**Version:** 3.0 (Interval mode with ankle scaling fix)
