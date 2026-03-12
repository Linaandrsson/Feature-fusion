# RMS Calculations

This directory contains all RMS (Root Mean Square) calculation scripts and outputs for the project.

## Structure

```
RMS_calculations/
├── calculate_rms.py                 # RMS calculation for raw IMU data
├── calculate_rms_tremor.py          # RMS calculation for tremor datasets (HAR branch)
├── calculate_rms_tremorbranch.py    # RMS calculation for tremor-branch datasets (non-normalized)
├── rms_example.py                   # Example script for raw IMU data
├── rms_tremor_example.py            # Example script for tremor datasets
├── rms_tremorbranch_example.py      # Example script for tremor-branch datasets
├── rms_results_IMU_test/            # Output: RMS results for raw IMU data
├── rms_results_tremor/              # Output: RMS results for tremor datasets
└── rms_results_tremorbranch/        # Output: RMS results for tremor-branch datasets
```

## Scripts

### 1. Raw IMU Data RMS (`calculate_rms.py`)
- Calculates RMS for raw IMU sensor data from CSV files
- Supports Accelerometer, Gyroscope, and Magnetometer
- Interactive CLI or programmatic usage via `rms_example.py`
- **Output**: `rms_results_IMU_test/`

### 2. Tremor Dataset RMS (`calculate_rms_tremor.py`)
- Calculates RMS for tremor-labeled datasets (HAR branch)
- Uses z-score normalized data
- Supports all 8 sensors (Acc_chest, ECG, Acc_ankle, etc.)
- **Dataset**: `s2_w2_fs50_tremor_*`
- **Output**: `rms_results_tremor/`

### 3. Tremor-Branch RMS (`calculate_rms_tremorbranch.py`)
- Calculates RMS for non-normalized tremor-branch datasets
- Only arm IMU sensors: `Acc_arm_tremorbranch`, `Gyro_arm_tremorbranch`
- Preserves amplitude information for tremor severity estimation
- **Dataset**: `s2_w2_fs30_tremor_*` (default) or `s2_w2_fs50_tremor_*`
- **Output**: `rms_results_tremorbranch/`

## Usage

### Interactive Mode
```bash
# Raw IMU data
python calculate_rms.py

# Tremor datasets (HAR branch)
python calculate_rms_tremor.py

# Tremor-branch (non-normalized)
python calculate_rms_tremorbranch.py
```

### Programmatic Mode
```bash
# Raw IMU data
python rms_example.py

# Tremor datasets
python rms_tremor_example.py

# Tremor-branch
python rms_tremorbranch_example.py
```

## Output Format

All scripts generate:
1. **Console output**: Formatted RMS results
2. **TXT files**: Detailed RMS results with metadata

### Example Output File
```
================================================================================
RMS ANALYSIS RESULTS - TREMOR-BRANCH DATASET (Non-Normalized)
================================================================================
Variant: SEVERE (s2_w2_fs30_tremor_mod_severe)
Activity: Walking (index 3)
Window: 0
Subject: 5
Tremor frequency: 5.20 Hz
Tremor score: 4.0000

ACCELEROMETER ARM (TREMOR BRANCH) (m/s²)
--------------------------------------------------------------------------------
  X-axis RMS:           2.5643 m/s²
  Y-axis RMS:           1.8932 m/s²
  Z-axis RMS:           3.2187 m/s²
  Magnitude RMS:        4.4521 m/s²
```

## Notes

- **HAR branch** (tremor datasets): Z-score normalized, used for activity recognition
- **Tremor branch** (tremor-branch datasets): Non-normalized, preserves amplitude for tremor severity estimation
- All output directories are automatically created when scripts run
