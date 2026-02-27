"""
Data Generator for Tremor-Labeled Datasets
===========================================

This script generates datasets with Parkinson's tremor using two sampling methods:

1. SUBJECT-based (default): Each subject has fixed tremor characteristics
   - Uses A_SUBJECT dict for baseline severity
   - Uses FREQ_TREMOR dict for subject-specific frequency
   - Modulated by body_part, activity, and jitter
   - Realistic for patient-specific studies

2. INTERVAL-based: Samples tremor parameters independently per window
   - Samples from severity ranges (SCORE_RMS_RANGE)
   - Random frequency per window (FREQ_RANGE_HZ)
   - High variability, suited for augmentation studies

Each window is labeled with:
  - tremor_freq: Tremor frequency (Hz)
  - tremor_acc_rms: Accelerometer RMS (m/s²)
  - tremor_gyro_rms: Gyroscope RMS (deg/s)
  - tremor_score: Severity score (0-4)
  - sampling_method: "subject" or "interval"

Can generate:
  1. CLEAN data: No tremor (tremor labels = 0)
  2. TREMOR data: Parkinson-specific tremor with per-window labels

Based on DataGenerator_v3.py but uses precompute_tremor_cache_with_parkinson_model()
"""

import numpy as np
from pathlib import Path
import random
from scipy.signal import resample
import json
import time
from typing import List, Dict, Optional, Tuple

# Import Parkinson tremor functions
from Noise_simulation.Tremor import (
    precompute_tremor_cache_with_parkinson_model,
    write_tremor_parkinson_params_file,
    TREMOR_MU, TREMOR_SIGMA, TREMOR_DT,
    TREMOR_INTERMITTENT, TREMOR_ON_PROB, TREMOR_MIN_ON_SEC, TREMOR_MAX_ON_SEC
)
import Noise_simulation.tremor_parkinson_config as pk_config

# ============================================================
# CONFIG - Basic parameters
# ============================================================
ORIGINAL_FS = 50        # original sampling rate in the dataset (Hz)
FS = 50                 # target sampling rate (Hz)
WINDOW_SEC = 2.0        # window length in seconds
STRIDE_SEC = 2.0        # stride in seconds
AUG_SIZE = 1            # number of augmented copies per window
NOISE_LEVEL = 0.01      # augmentation noise level (NOT tremor)
NUM_SUBJECTS = 10

# Dataset location
script_dir = Path(__file__).parent.resolve()
DATA_PATH = Path(script_dir / "data" / "MHEALTHDATASET")
OUT_BASE = Path(script_dir / "data" / "Tremor_datagenerator_files")

# Seeds
SEED = 0                        # Controls augmentation noise
TREMOR_SEED = 42                # Controls tremor generation (change for different realizations)

# ============================================================
# CONFIG - Tremor Generation
# ============================================================
# Set GENERATE_TREMOR = False for clean dataset (tremor labels = 0)
# Set GENERATE_TREMOR = True for Parkinson tremor with per-window labels
GENERATE_TREMOR = True

# Tremor parameters (from Tremor.py defaults)
TREMOR_MU = 1.0
TREMOR_SIGMA = 0.5
TREMOR_DT = 0.001
TREMOR_INTERMITTENT = False
TREMOR_ON_PROB = 0.5
TREMOR_MIN_ON_SEC = 2.0
TREMOR_MAX_ON_SEC = 8.0

# Jitter: adds window-to-window variability
USE_TREMOR_JITTER = True
TREMOR_JITTER_STD = 0.15  # 15% variability (from pk_config.JITTER_STD)

# ============================================================
# CONFIG - Tremor Sampling Method
# ============================================================
# Choose tremor parameter generation method:
#   "subject": Subject-based (default)
#              - Each subject has fixed baseline tremor severity (A_SUBJECT)
#              - Fixed frequency per subject (FREQ_TREMOR)
#              - Modulated by body_part, activity, and jitter
#              - Realistic for patient-specific studies
#
#   "interval": Per-window sampling from severity ranges
#              - Samples tremor parameters independently per window
#              - More variability across windows
#              - Suited for augmentation studies
#
# Import these from tremor_parkinson_config to avoid duplication:
# - pk_config.DEFAULT_SAMPLING_METHOD
# - pk_config.DEFAULT_AUGMENT_MODE

# ============================================================
# SENSOR COLUMN MAP (0-indexed)
# ============================================================
SENSORS = {
    "Acc_chest":  [0, 1, 2],
    "ECG":        [3, 4],
    "Acc_ankle":  [5, 6, 7],
    "Gyro_ankle": [8, 9, 10],
    "Mag_ankle":  [11, 12, 13],
    "Acc_arm":    [14, 15, 16],
    "Gyro_arm":   [17, 18, 19],
    "Mag_arm":    [20, 21, 22],
}

LABEL_COL = 23


# ============================================================
# Helpers
# ============================================================

def resample_window(win: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample window from original_fs to target_fs."""
    if original_fs == target_fs:
        return win

    L_original, C = win.shape
    L_target = int(round(L_original * target_fs / original_fs))

    resampled = np.zeros((L_target, C))
    for c in range(C):
        resampled[:, c] = resample(win[:, c], L_target)

    return resampled


def zscore_window(win: np.ndarray) -> np.ndarray:
    """Normalize per channel within window."""
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def augment_window(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small augmentation noise to window."""
    noise = rng.normal(0, NOISE_LEVEL, size=win.shape)
    return win + noise


def load_subject_data_with_retry(file_path: Path, max_retries: int = 5, delay: float = 2.0) -> np.ndarray:
    """Load subject data with retry logic for OneDrive timeout issues."""
    for attempt in range(max_retries):
        try:
            data = np.loadtxt(file_path)
            
            if data.ndim == 1 or (data.ndim > 1 and data.shape[1] < 24):
                print(f"\n{'='*60}")
                print(f"WARNING: {file_path.name} appears to be empty or incomplete!")
                print(f"This is likely a OneDrive sync issue.")
                print(f"Please check that OneDrive has fully synced the file.")
                print(f"{'='*60}\n")
            
            return data
        except (TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"  Timeout loading {file_path.name}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  Failed to load {file_path.name} after {max_retries} attempts")
                raise


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    """
    X_c_l: (N, C, L)
    Return (N, C*L) in channel-block format.
    """
    N, C, L = X_c_l.shape
    blocks = [X_c_l[:, c, :] for c in range(C)]
    return np.concatenate(blocks, axis=1)


def extract_body_part(sensor_name: str) -> str:
    """Extract body part from sensor name."""
    if 'ankle' in sensor_name:
        return 'ankle'
    elif 'arm' in sensor_name:
        return 'arm'
    else:
        return 'chest'


def get_sensor_type(sensor_name: str) -> str:
    """Get sensor type: 'acc', 'gyro', 'mag', or 'ecg'."""
    if 'Acc' in sensor_name:
        return 'acc'
    elif 'Gyro' in sensor_name:
        return 'gyro'
    elif 'Mag' in sensor_name:
        return 'mag'
    elif 'ECG' in sensor_name:
        return 'ecg'
    return 'unknown'


# ============================================================
# Main Processing
# ============================================================

def generate_tremor_dataset(variant_name: str):
    """
    Generate dataset with tremor labels.
    
    Args:
        variant_name: Name for this dataset variant (e.g., "s1_w2_tremor_clean" or "s1_w2_tremor_subj5")
    """
    
    print("=" * 80)
    print(f"TREMOR-LABELED DATASET GENERATION: {variant_name}")
    print("=" * 80)
    print(f"Generate tremor: {GENERATE_TREMOR}")
    print(f"Original FS: {ORIGINAL_FS} Hz, Target FS: {FS} Hz")
    print(f"Window: {WINDOW_SEC}s, Stride: {STRIDE_SEC}s")
    print(f"Augmentation: {AUG_SIZE} copies per window")
    print(f"Augmentation seed: {SEED}, Tremor seed: {TREMOR_SEED}")
    print("=" * 80)
    
    # Create output directory
    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    window_len = int(round(WINDOW_SEC * ORIGINAL_FS))
    stride = int(round(STRIDE_SEC * ORIGINAL_FS))
    
    # -------------------------------
    # Step 1: Collect window specs from all subjects
    # -------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: Collecting window specifications...")
    print("=" * 80)
    
    window_specs = []  # List of (subj_idx, label, win_start, win_end)
    
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        if not fname.exists():
            print(f"  [Subject {subj_idx:2d}] File not found: {fname}")
            continue
        
        data = load_subject_data_with_retry(fname)
        
        if data.shape[1] < 24:
            print(f"  [Subject {subj_idx:2d}] Skipping - insufficient columns")
            continue
        
        labels = data[:, LABEL_COL].astype(int)
        labels_used = labels[labels >= 1]
        
        # Create windows
        i = 0
        n_windows = 0
        while i + window_len <= len(data):
            lab_win = labels[i:i + window_len]
            
            if np.all(lab_win >= 1):
                lab_val = int(lab_win[0])
                window_specs.append((subj_idx, lab_val, i, i + window_len))
                n_windows += 1
            
            i += stride
        
        print(f"  [Subject {subj_idx:2d}] {n_windows:4d} windows")
    
    print(f"\n✓ Total windows: {len(window_specs)}")
    
    # -------------------------------
    # Step 2: Pre-compute tremor cache (or create dummy cache for clean data)
    # -------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Pre-computing tremor cache...")
    print("=" * 80)
    
    def load_subject_data_func(subj_idx: int) -> np.ndarray:
        """Data loader for tremor cache."""
        fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        return load_subject_data_with_retry(fname)
    
    if GENERATE_TREMOR:
        tremor_cache = precompute_tremor_cache_with_parkinson_model(
            window_specs=window_specs,
            sensor_column_mapping=SENSORS,
            data_loader_func=load_subject_data_func,
            fs=ORIGINAL_FS,
            tremor_mu=TREMOR_MU,
            tremor_sigma=TREMOR_SIGMA,
            tremor_dt=TREMOR_DT,
            tremor_intermittent=TREMOR_INTERMITTENT,
            tremor_on_prob=TREMOR_ON_PROB,
            tremor_min_on_sec=TREMOR_MIN_ON_SEC,
            tremor_max_on_sec=TREMOR_MAX_ON_SEC,
            scenario_seed=TREMOR_SEED,
            use_jitter=USE_TREMOR_JITTER,
            jitter_std=TREMOR_JITTER_STD if USE_TREMOR_JITTER else 0.0,
            sampling_method=pk_config.DEFAULT_SAMPLING_METHOD,
            augment_mode=pk_config.DEFAULT_AUGMENT_MODE,
        )
    else:
        # Create dummy cache for clean data
        print("  Generating CLEAN dataset (no tremor)")
        tremor_cache = {}
        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            L = win_end - win_start
            for body_part in ['ankle', 'arm', 'chest']:
                tremor_cache[(w_idx, body_part)] = {
                    'acc_noise': np.zeros((L, 3), dtype=np.float32),
                    'gyro_noise': np.zeros((L, 3), dtype=np.float32),
                    'mag_noise': np.zeros((L, 3), dtype=np.float32),
                    'meta': {
                        'subject_id': subj_idx,
                        'body_part': body_part,
                        'activity': label,
                        'acc_target_rms': 0.0,
                        'gyro_target_rms': 0.0,
                        'freq_hz': 0.0,
                        'severity': 'clean',
                    }
                }
        print(f"  ✓ Created clean cache with {len(tremor_cache)} entries")
    
    # -------------------------------
    # Step 3: Process each sensor and save
    # -------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Processing sensors and applying tremor...")
    print("=" * 80)
    
    aug_rng = np.random.default_rng(SEED)
    target_window_len = int(round(WINDOW_SEC * FS))
    
    for sensor_name, cols in SENSORS.items():
        print(f"\n  Processing: {sensor_name}")
        
        X_list = []
        y_list = []
        subj_list = []
        base_idx_list = []
        
        # Tremor label arrays
        tremor_freq_list = []
        tremor_acc_rms_list = []
        tremor_gyro_rms_list = []
        tremor_score_list = []
        
        body_part = extract_body_part(sensor_name)
        sensor_type = get_sensor_type(sensor_name)
        
        # Load all subject data
        subject_data_cache = {}
        for subj_idx in range(1, NUM_SUBJECTS + 1):
            fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
            if fname.exists():
                subject_data_cache[subj_idx] = load_subject_data_with_retry(fname)
        
        # Process each window
        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subject_data_cache:
                continue
            
            data = subject_data_cache[subj_idx]
            
            # Extract window data
            win_raw = data[win_start:win_end, cols].copy()
            
            # Skip ECG for tremor (only IMU sensors)
            if sensor_type in ['acc', 'gyro', 'mag']:
                cache_entry = tremor_cache.get((w_idx, body_part))
                if cache_entry is not None:
                    noise_key = f'{sensor_type}_noise'
                    tremor_noise = cache_entry[noise_key]
                    win_raw += tremor_noise
            
            # Resample
            win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
            
            # Z-score normalize
            win = zscore_window(win_resampled)
            
            # Get tremor labels for this window
            cache_entry = tremor_cache.get((w_idx, body_part))
            if cache_entry is not None and GENERATE_TREMOR:
                meta = cache_entry['meta']
                tremor_freq = meta.get('freq_hz', 0.0)
                tremor_acc_rms = meta.get('acc_target_rms', 0.0)
                tremor_gyro_rms = meta.get('gyro_target_rms', 0.0)
                tremor_score = pk_config.get_tremor_score(tremor_acc_rms)
            else:
                # Clean data
                tremor_freq = 0.0
                tremor_acc_rms = 0.0
                tremor_gyro_rms = 0.0
                tremor_score = 0
            
            # Apply augmentation (creates AUG_SIZE copies)
            for a in range(AUG_SIZE):
                win_aug = augment_window(win.copy(), aug_rng)
                
                X_list.append(win_aug.T.astype(np.float32))  # (C, L)
                y_list.append(label - 1)                      # 0..11
                subj_list.append(subj_idx)
                base_idx_list.append(w_idx)
                
                # Same tremor labels for all augmented copies
                tremor_freq_list.append(tremor_freq)
                tremor_acc_rms_list.append(tremor_acc_rms)
                tremor_gyro_rms_list.append(tremor_gyro_rms)
                tremor_score_list.append(tremor_score)
        
        # Stack arrays
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        subject_ids = np.array(subj_list, dtype=np.int64)
        base_window_idx = np.array(base_idx_list, dtype=np.int64)
        
        tremor_freq = np.array(tremor_freq_list, dtype=np.float32)
        tremor_acc_rms = np.array(tremor_acc_rms_list, dtype=np.float32)
        tremor_gyro_rms = np.array(tremor_gyro_rms_list, dtype=np.float32)
        tremor_score = np.array(tremor_score_list, dtype=np.int8)
        
        # Save NPZ with tremor labels
        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X=X,
            y=y,
            subject_id=subject_ids,
            base_window_idx=base_window_idx,
            tremor_freq=tremor_freq,
            tremor_acc_rms=tremor_acc_rms,
            tremor_gyro_rms=tremor_gyro_rms,
            tremor_score=tremor_score,
            fs=FS,
            window_len=target_window_len,
            stride=int(round(STRIDE_SEC * FS)),
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )
        
        # Save TXT with all labels
        txt_out = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)
        
        # Stack all labels as columns
        all_labels = np.column_stack([
            (y + 1),              # Activity label (1-indexed)
            subject_ids,          # Subject ID
            base_window_idx,      # Base window index
            tremor_freq,          # Tremor frequency (Hz)
            tremor_acc_rms,       # Acc RMS (m/s²)
            tremor_gyro_rms,      # Gyro RMS (deg/s)
            tremor_score          # Severity score (0-4)
        ])
        
        sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")
        
        print(f"    ✓ Saved: {npz_out.name}, {txt_out.name} | X={X.shape}, tremor_labels={tremor_freq.shape}")
    
    # -------------------------------
    # Step 4: Write variant info.txt
    # -------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: Writing configuration files...")
    print("=" * 80)
    
    info_txt = variant_dir / "info.txt"
    overlap = 1.0 - (STRIDE_SEC / WINDOW_SEC) if WINDOW_SEC > 0 else 0.0
    stride_samples = int(round(FS * STRIDE_SEC))
    
    info_lines = [
        "=" * 80,
        "TREMOR-LABELED DATASET CONFIGURATION",
        "=" * 80,
        "",
        "Basic Parameters:",
        f"  - Original FS: {ORIGINAL_FS} Hz",
        f"  - Target FS: {FS} Hz",
        f"  - Window size: {WINDOW_SEC} s ({target_window_len} samples)",
        f"  - Stride: {STRIDE_SEC} s ({stride_samples} samples)",
        f"  - Overlap: {overlap:.2f}",
        f"  - Augmentation copies (AUG_SIZE): {AUG_SIZE}",
        f"  - Augmentation noise level: {NOISE_LEVEL}",
        f"  - Augmentation seed: {SEED}",
        "",
        "=" * 80,
        "Tremor Configuration:",
        "=" * 80,
        f"  - Generate tremor: {GENERATE_TREMOR}",
    ]
    
    if GENERATE_TREMOR:
        info_lines.extend([
            f"  - Tremor seed: {TREMOR_SEED}",
            f"  - Use jitter: {USE_TREMOR_JITTER}",
            f"  - Jitter std: {TREMOR_JITTER_STD}",
            "",
            "Tremor Labels (per window):",
            "  - tremor_freq: Tremor frequency (Hz) [3.5-7.0]",
            "  - tremor_acc_rms: Accelerometer RMS (m/s²)",
            "  - tremor_gyro_rms: Gyroscope RMS (deg/s)",
            "  - tremor_score: Severity score [0-4]",
            "    * Score 0: No tremor (< 0.07 m/s²)",
            "    * Score 1: Mild (0.07-0.15 m/s²)",
            "    * Score 2: Mild-Moderate (0.15-0.7 m/s²)",
            "    * Score 3: Moderate-Severe (0.7-2.5 m/s²)",
            "    * Score 4: Severe (2.5-6.0 m/s²)",
            "",
            "Model: RMS_acc = A_subject[subj] × C[body_part] × beta[activity] × (1 + jitter)",
            "       RMS_gyro = k_g(RMS_acc) × RMS_acc",
            "       Frequency = subject-specific (FREQ_TREMOR dict)",
        ])
    else:
        info_lines.extend([
            "  - This is a CLEAN dataset (no tremor)",
            "  - All tremor labels are set to 0",
        ])
    
    info_lines.extend([
        "",
        "=" * 80,
        "Dataset Format:",
        "=" * 80,
        "  NPZ files contain:",
        "    - X: (N, C, L) sensor data",
        "    - y: (N,) activity labels (0-11)",
        "    - subject_id: (N,) subject IDs (1-10)",
        "    - base_window_idx: (N,) base window index",
        "    - tremor_freq: (N,) tremor frequency",
        "    - tremor_acc_rms: (N,) acc RMS",
        "    - tremor_gyro_rms: (N,) gyro RMS",
        "    - tremor_score: (N,) severity score",
        "",
        "  TXT files: (N, C×L+7) CSV format with columns:",
        f"    - Columns 1-{target_window_len * 3}: Sensor data (flattened C×L)",
        f"    - Column {target_window_len * 3 + 1}: Activity label (1-12)",
        f"    - Column {target_window_len * 3 + 2}: Subject ID (1-10)",
        f"    - Column {target_window_len * 3 + 3}: Base window index",
        f"    - Column {target_window_len * 3 + 4}: Tremor frequency (Hz)",
        f"    - Column {target_window_len * 3 + 5}: Tremor acc RMS (m/s²)",
        f"    - Column {target_window_len * 3 + 6}: Tremor gyro RMS (deg/s)",
        f"    - Column {target_window_len * 3 + 7}: Tremor severity score (0-4)",
        "",
        "=" * 80,
    ])
    
    with open(info_txt, 'w') as f:
        f.write('\n'.join(info_lines))
    
    print(f"  ✓ Saved: {info_txt.name}")
    
    # Write Parkinson parameters file if tremor was generated
    if GENERATE_TREMOR:
        params_file = variant_dir / "tremor_parkinson_params.txt"
        write_tremor_parkinson_params_file(
            output_path=params_file,
            mu=TREMOR_MU,
            sigma=TREMOR_SIGMA,
            dt=TREMOR_DT,
            intermittent=TREMOR_INTERMITTENT,
            on_prob=TREMOR_ON_PROB,
            min_on_sec=TREMOR_MIN_ON_SEC,
            max_on_sec=TREMOR_MAX_ON_SEC,
            use_jitter=USE_TREMOR_JITTER,
            jitter_std=TREMOR_JITTER_STD,
            scenario_seed=TREMOR_SEED
        )
        print(f"  ✓ Saved: {params_file.name}")
    
    print("\n" + "=" * 80)
    print(f"✓ DATASET GENERATION COMPLETE: {variant_name}")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Example usage:
    # Set GENERATE_TREMOR at the top of the file (line 61) before running!
    
    # For clean dataset: Set GENERATE_TREMOR = False, then run:
    if GENERATE_TREMOR:
        generate_tremor_dataset("s2_w2_tremor_parkinson")
    else:
        generate_tremor_dataset("s2_w2_tremor_clean")
