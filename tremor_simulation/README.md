# DataGenerator_Tremor — Tremor Dataset Generation

This folder contains the data generation pipeline for creating tremor-labeled datasets
from the raw MHEALTH sensor recordings. The primary script is `DataGenerator_Tremor.py`.

---

## Table of Contents

1. [Overview](#overview)
2. [Files in this folder](#files-in-this-folder)
3. [Configuration](#configuration)
4. [Pipeline Steps](#pipeline-steps)
5. [Output Structure](#output-structure)
6. [File Formats](#file-formats)
   - [NPZ format](#npz-format)
   - [TXT format](#txt-format)
7. [Sensors and Column Mapping](#sensors-and-column-mapping)
8. [Tremor Model](#tremor-model)
9. [Augmentation Strategy per Sensor](#augmentation-strategy-per-sensor)
10. [Subject Splits](#subject-splits)
11. [Reproducibility and Seeds](#reproducibility-and-seeds)

---

## Overview

`DataGenerator_Tremor.py` reads raw MHEALTH `.log` files (10 subjects × 12 activities),
segments them into non-overlapping windows, applies simulated Parkinson's tremor to
the relevant sensors, and saves the result as `.npz` and `.txt` files ready for
model training and evaluation.

Three **severity variants** can be generated in one run:

| Variant suffix | Score range | Acc RMS range (m/s²) | Description              |
|----------------|-------------|----------------------|--------------------------|
| `_clean`       | 0           | 0                    | No tremor                |
| `_mild_mod`    | 1–2         | 0.05–0.50            | Mild to mild-moderate    |
| `_mod_severe`  | 3–4         | 1.10–5.10            | Moderate-severe to severe|

The dataset name format is: `s{STRIDE}_w{WINDOW}_fs{FS}_tremor_{mode}`  
Example: `s4_w4_fs50_tremor_mod_severe`

---

## Files in this folder

| File | Description |
|------|-------------|
| `DataGenerator_Tremor.py` | Main data generation script |
| `DataGenerator_CleanAug.py` | Clean dataset generator (no tremor, various augmentations) |
| `Tremor.py` | Van der Pol tremor oscillator + physics-based tremor functions |
| `tremor_parkinson_config.py` | Parkinson tremor parameter configuration (severities, frequencies, body-part scaling, activity modulation) |

---

## Configuration

All key parameters are defined at the top of `DataGenerator_Tremor.py`:

### Basic parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ORIGINAL_FS` | 50 Hz | Sampling rate of the raw MHEALTH dataset |
| `FS` | 50 Hz | Target output sampling rate |
| `WINDOW_SEC` | 4.0 s | Window length in seconds |
| `STRIDE_SEC` | 4.0 s | Stride in seconds (no overlap at default) |
| `AUG_SIZE` | 1 | Number of augmented copies per window |
| `NOISE_LEVEL` | 0.01 | Std of small additive Gaussian noise applied after z-score |
| `NUM_SUBJECTS` | 10 | Number of subjects to process |

### Paths

| Variable | Resolved path | Description |
|----------|---------------|-------------|
| `DATA_PATH` | `data/MHEALTHDATASET/` | Raw MHEALTH `.log` files |
| `OUT_BASE` | `data/Tremor_datagenerator_files/` | Root output directory |

### Tremor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GENERATE_TREMOR` | `True` | Set `False` to produce a clean (no-tremor) dataset |
| `TREMOR_SEED` | 42 | Seed for tremor generation (change for independent realizations) |
| `USE_TREMOR_JITTER` | `True` | Add ±15% window-to-window variability to tremor amplitude |
| `TREMOR_JITTER_STD` | 0.15 | Jitter standard deviation (15%) |

Tremor sampling method and augmentation mode are set in `tremor_parkinson_config.py`:

```python
DEFAULT_SAMPLING_METHOD = "interval"   # "subject" or "interval"
DEFAULT_AUGMENT_MODE    = "mod_severe" # "clean", "mild_mod", or "mod_severe"
```

---

## Pipeline Steps

### Step 1 — Collect window specifications
For each subject file `mHealth_subject{N}.log`, the script scans valid labeled
rows (label ≥ 1) with a sliding window. Each window that contains only valid labels
is recorded as `(subject_id, activity_label, win_start, win_end)`.

Window length: `WINDOW_SEC × ORIGINAL_FS` samples  
Stride: `STRIDE_SEC × ORIGINAL_FS` samples

### Step 2 — Pre-compute tremor cache
`precompute_tremor_cache_with_parkinson_model()` generates a tremor noise dictionary
keyed by `(window_index, body_part)` for `arm`, `ankle`, and `chest`. Each entry holds:
- `acc_noise`: `(L, 3)` accelerometer tremor signal
- `gyro_noise`: `(L, 3)` gyroscope tremor signal
- `mag_noise`: `(L, 3)` magnetometer tremor signal (zeros for rotation-based sensors)
- `meta`: metadata dict with `freq_hz`, `acc_target_rms`, `gyro_target_rms`, `sampled_score`

For clean datasets (`GENERATE_TREMOR = False`) a zero-noise dummy cache is created.

Optional sanity checks are printed when `RUN_SANITY_CHECKS = True`.

### Step 3 — Process each sensor (HAR branch)
For each of the 8 sensors (see [Sensors and Column Mapping](#sensors-and-column-mapping)):

1. Extract raw window from subject data
2. Apply sensor-specific tremor or augmentation (see [Augmentation Strategy](#augmentation-strategy-per-sensor))
3. Resample to target `FS` with `scipy.signal.resample`
4. **Z-score normalize** per channel within the window
5. Add small `augment_window` noise (std = `NOISE_LEVEL`)
6. Repeat `AUG_SIZE` times (with different noise)
7. Save `.npz` and `.txt`

### Step 3.5 — Tremor-branch datasets (Acc_arm, Gyro_arm only)
Identical to Step 3 **except**:
- Z-score normalization is **skipped** (amplitude preserved for severity estimation)
- `augment_window` noise is **skipped**
- Only 1 copy per window (no AUG_SIZE repetition)

Output files: `Acc_arm_tremorbranch.{npz,txt}`, `Gyro_arm_tremorbranch.{npz,txt}`

### Step 4 — Write configuration files
- `info.txt` — human-readable summary of all generation parameters
- `tremor_parkinson_params.txt` — Van der Pol oscillator and tremor model parameters

---

## Output Structure

```
data/Tremor_datagenerator_files/
│
├── s4_w4_fs50_tremor_clean/          # No tremor (score = 0)
│   ├── Acc_ankle.npz / .txt
│   ├── Acc_arm.npz / .txt
│   ├── Acc_arm_tremorbranch.npz / .txt
│   ├── Acc_chest.npz / .txt
│   ├── ECG.npz / .txt
│   ├── Gyro_ankle.npz / .txt
│   ├── Gyro_arm.npz / .txt
│   ├── Gyro_arm_tremorbranch.npz / .txt
│   ├── Mag_ankle.npz / .txt
│   ├── Mag_arm.npz / .txt
│   ├── info.txt
│   └── tremor_parkinson_params.txt
│
├── s4_w4_fs50_tremor_mild_mod/       # Mild–moderate tremor (scores 1–2)
│   └── (same files as above)
│
└── s4_w4_fs50_tremor_mod_severe/     # Moderate–severe tremor (scores 3–4)
    └── (same files as above)
```

Each variant folder contains identical sensor files. Differences are in the
tremor amplitude/score labels and the actual signal content.

---

## File Formats

### NPZ format

Each `.npz` file is a compressed NumPy archive with the following arrays:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `X` | `(N, C, L)` | float32 | Sensor data |
| `y` | `(N,)` | int64 | Activity label, **0-indexed** (0–11) |
| `subject_id` | `(N,)` | int64 | Subject ID (1–10) |
| `base_window_idx` | `(N,)` | int64 | Sequential window index from Step 1 |
| `tremor_freq` | `(N,)` | float32 | Tremor frequency in Hz (0.0 for clean) |
| `tremor_acc_rms` | `(N,)` | float32 | Accelerometer tremor RMS in m/s² (0.0 for clean) |
| `tremor_gyro_rms` | `(N,)` | float32 | Gyroscope tremor RMS in deg/s (0.0 for clean) |
| `tremor_score` | `(N,)` | int8 | Severity score 0–4 |
| `fs` | scalar | int | Target sampling rate (Hz) |
| `window_len` | scalar | int | Window length in samples |
| `stride` | scalar | int | Stride in samples |
| `sensor_name` | str | — | Sensor identifier string |
| `sensor_cols` | `(C,)` | int64 | Original MHEALTH column indices |

**Dimensions:**  
- `N` = number of windows × `AUG_SIZE`  
- `C` = 3 for Acc/Gyro/Mag, 2 for ECG  
- `L` = `WINDOW_SEC × FS` (e.g., 200 at 4 s × 50 Hz)

**Loading example (Python):**
```python
import numpy as np

data = np.load("Acc_arm.npz")
X            = data["X"]           # (N, 3, 200)
y            = data["y"]           # (N,) — 0-indexed
subject_id   = data["subject_id"]  # (N,)
tremor_score = data["tremor_score"]# (N,)

# Split by subject (example: test subjects 5 and 10)
test_mask = np.isin(subject_id, [5, 10])
X_test = X[test_mask]
y_test = y[test_mask]
```

---

### TXT format

Each `.txt` file is a CSV where rows are windows and columns are:

```
[sensor data (flattened)] | [7 label columns]
```

#### 3-axis sensors (Acc_*, Gyro_*, Mag_*) — 607 columns total

| Columns | Range | Description |
|---------|-------|-------------|
| 1 – L | 1–200 | Channel 0 (x-axis), z-score normalized |
| L+1 – 2L | 201–400 | Channel 1 (y-axis), z-score normalized |
| 2L+1 – 3L | 401–600 | Channel 2 (z-axis), z-score normalized |
| 3L+1 | 601 | Activity label (**1-indexed**, 1–12) |
| 3L+2 | 602 | Subject ID (1–10) |
| 3L+3 | 603 | Base window index |
| 3L+4 | 604 | Tremor frequency (Hz) |
| 3L+5 | 605 | Tremor acc RMS (m/s²) |
| 3L+6 | 606 | Tremor gyro RMS (deg/s) |
| 3L+7 | 607 | Tremor severity score (0–4) |

Where `L = WINDOW_SEC × FS = 200` for the default 4 s / 50 Hz configuration.

#### ECG sensor — 407 columns total

| Columns | Range | Description |
|---------|-------|-------------|
| 1 – L | 1–200 | Lead I, z-score normalized |
| L+1 – 2L | 201–400 | Lead II, z-score normalized |
| 2L+1 – 2L+7 | 401–407 | Same 7 label columns as above |

#### Tremor-branch files (`*_tremorbranch.txt`)

Same column layout as the corresponding sensor, **but the sensor data is NOT
z-score normalized** — raw resampled amplitude is preserved.

#### Activity label mapping (TXT is 1-indexed; NPZ is 0-indexed)

| TXT value | NPZ value | Activity |
|-----------|-----------|----------|
| 1 | 0 | Standing still |
| 2 | 1 | Sitting and relaxing |
| 3 | 2 | Lying down |
| 4 | 3 | Walking |
| 5 | 4 | Climbing stairs |
| 6 | 5 | Waist bends forward |
| 7 | 6 | Frontal elevation of arms |
| 8 | 7 | Knees bending (crouching) |
| 9 | 8 | Cycling |
| 10 | 9 | Jogging |
| 11 | 10 | Running |
| 12 | 11 | Jump front & back |

**Loading example (Python):**
```python
import numpy as np

data = np.loadtxt("Acc_arm.txt", delimiter=",")
L = 200  # window length (4 s × 50 Hz)

sensor_flat    = data[:, :3*L]         # (N, 600) — z-score normalized
activity       = data[:, 3*L]          # 1-indexed
subject_id     = data[:, 3*L+1].astype(int)
base_window    = data[:, 3*L+2].astype(int)
tremor_freq    = data[:, 3*L+3]
tremor_acc_rms = data[:, 3*L+4]
tremor_gyro_rms= data[:, 3*L+5]
tremor_score   = data[:, 3*L+6].astype(int)

# Reshape to (N, C, L)
N = data.shape[0]
X = sensor_flat.reshape(N, 3, L)  # channel-block layout: X[:, 0, :] = x-axis
```

---

## Sensors and Column Mapping

The following sensors are extracted from the raw 24-column MHEALTH `.log` files:

| Sensor | MHEALTH columns (0-indexed) | Channels | Body part | Tremor policy |
|--------|-----------------------------|----------|-----------|---------------|
| `Acc_chest` | 0, 1, 2 | x, y, z | chest | Tremor-free (AWGN/Rotation aug) |
| `ECG` | 3, 4 | Lead I, Lead II | chest | Tremor-free (AWGN aug only) |
| `Acc_ankle` | 5, 6, 7 | x, y, z | ankle | Tremor (ankle-scaled) |
| `Gyro_ankle` | 8, 9, 10 | x, y, z | ankle | Tremor (ankle-scaled) |
| `Mag_ankle` | 11, 12, 13 | x, y, z | ankle | Rotation-based tremor |
| `Acc_arm` | 14, 15, 16 | x, y, z | arm | Tremor |
| `Gyro_arm` | 17, 18, 19 | x, y, z | arm | Tremor |
| `Mag_arm` | 20, 21, 22 | x, y, z | arm | Rotation-based tremor |

Column 23 (0-indexed) = activity label in the raw file.

---

## Tremor Model

Tremor is simulated using a **Van der Pol oscillator** (in `Tremor.py`) that produces
realistic quasi-periodic tremor waveforms. Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TREMOR_MU` | 1.0 | Oscillator nonlinearity (amplitude stability) |
| `TREMOR_SIGMA` | 0.5 | Stochastic forcing (inter-cycle variability) |
| `TREMOR_DT` | 0.001 | ODE integration step (s) |

### Severity scores and RMS ranges (interval sampling mode)

| Score | Label | Acc RMS range (m/s²) | k_g factor | Gyro RMS range (approx, deg/s) |
|-------|-------|----------------------|------------|-------------------------------|
| 0 | No tremor | 0 | — | 0 |
| 1 | Mild | 0.05 – 0.10 | 10.5 | 0.5 – 1.1 |
| 2 | Mild-moderate | 0.25 – 0.50 | 21.3 | 5.3 – 10.7 |
| 3 | Moderate-severe | 1.10 – 2.00 | 13.7 | 15.1 – 27.4 |
| 4 | Severe | 4.65 – 5.10 | 12.9 | 60.0 – 65.8 |

### RMS computation (interval mode)

```
RMS_acc_base  ~ Uniform(SCORE_RMS_RANGE[score])
activity_beta  = BETA_ACTIVITY_BY_SCORE[score][activity]
RMS_acc        = RMS_acc_base × activity_beta   (saved as tremor_acc_rms)
RMS_gyro       = k_g(RMS_acc) × RMS_acc         (saved as tremor_gyro_rms)
freq_hz        ~ Uniform(3.0, 7.0)              (saved as tremor_freq)
```

**Jitter** adds ±15% window-to-window variability when `USE_TREMOR_JITTER = True`.

### Activity-dependent tremor modulation (`BETA_ACTIVITY_BY_SCORE`)

Tremor is strongest at rest (scores 1.0–1.2) and reduced during active movement.
Example factors for score 3:

| Activity | Beta |
|----------|------|
| Standing still | 0.78 |
| Sitting | 1.15 |
| Lying down | 1.08 |
| Walking | 0.78 |
| Running | 1.00 |
| Cycling | 0.70 |

### Ankle tremor scaling

Ankle tremor is a fraction of arm tremor (lower limbs less affected):

| Score | Ankle/Arm ratio |
|-------|-----------------|
| 0 | 0.00 |
| 1 | 0.05 (5%) |
| 2 | 0.12 (12%) |
| 3 | 0.22 (22%) |
| 4 | 0.35 (35%) |

### Magnetometer tremor (rotation-based)

`Mag_arm` and `Mag_ankle` do **not** receive independent additive noise.
Instead, the gyroscope tremor drives cumulative rotation perturbations
applied to the original magnetometer vector, preserving the Earth field norm:

```
mag_rotated(t) = R(∫ω_tremor dt) × mag_original(t)
```

---

## Augmentation Strategy per Sensor

### Tremor-bearing sensors (Acc_ankle, Gyro_ankle, Acc_arm, Gyro_arm)
- Additive tremor noise from the pre-computed cache
- Tremor labels filled from cache metadata

### Rotation-based sensors (Mag_arm, Mag_ankle)
- Gyro tremor is integrated to cumulative rotation angles
- Applied as rotation matrix to the original magnetometer signal
- Tremor labels same as their respective body-part gyro sensor

### Tremor-free sensors (Acc_chest, ECG)
- No tremor signal is added
- All tremor labels set to 0
- Alternative augmentation based on global ARM severity score:
  - Score 0 (clean): original signal, no modification
  - Score 1–2 (mild): AWGN with noise RMS = 15% of signal RMS
  - Score 3–4 (severe): random 3D rotation ≤10° (Acc_chest) or AWGN (ECG)

---

## Subject Splits

The pipeline is designed for **subject-based train/val/test splits** to prevent
data leakage between subjects. The standard split used throughout the full pipeline is:

```python
TEST_SUBJECTS  = [5, 10]
VAL_SUBJECTS   = [2, 7]
TRAIN_SUBJECTS = [1, 3, 4, 6, 8, 9]
```

These splits must be applied consistently at every stage:
1. **Data generation** (`DataGenerator_Tremor.py`) — generates data for all 10 subjects
2. **Feature extraction** (`Feature extraction CNNs/Training/`) — training uses only `TRAIN_SUBJECTS`
3. **Embedding extraction** (`Feature extraction CNNs/Extraction/extract_all_sensors.py`) — splits applied
4. **Feature fusion / ablation** (`Feature fusion activity/Tremor/`) — final evaluation on `TEST_SUBJECTS`

The `base_window_idx` field in each file is the global window index from Step 1
(across all subjects). Use `subject_id` to reconstruct the correct split.

---

## Reproducibility and Seeds

| Seed variable | Default | Controls |
|---------------|---------|----------|
| `SEED` | 0 | `augment_window` Gaussian noise in HAR branch |
| `TREMOR_SEED` | 42 | All tremor parameter sampling and Van der Pol oscillator initialization |

To generate independent tremor realizations (e.g., for robustness testing),
change `TREMOR_SEED` while keeping `SEED` fixed. The window segmentation
and subject splits remain identical across seeds.

---

## Running the generator

```bash
cd /home/linacr/projects/Code
python tremor_simulation/DataGenerator_Tremor.py
```

With `GENERATE_TREMOR = True` and the default settings, this produces three variant
folders under `data/Tremor_datagenerator_files/`:
- `s4_w4_fs50_tremor_clean`
- `s4_w4_fs50_tremor_mild_mod`
- `s4_w4_fs50_tremor_mod_severe`

Each folder contains 10 sensor files × 2 formats (`.npz` + `.txt`) + 2 tremor-branch
files × 2 formats + `info.txt` + `tremor_parkinson_params.txt`.
