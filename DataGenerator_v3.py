import numpy as np
from pathlib import Path
import random
from scipy.signal import resample
import json
import time
from typing import List, Dict, Optional, Tuple

# Import noise simulation functions and default parameters
from Noise_simulation.AWGN import apply_awgn_rms_ratio, AWGN_RMS_RATIO
from Noise_simulation.Dropout import apply_dropout, DROPOUT_VALUE
from Noise_simulation.WeakSignal import apply_weak_signal, WEAK_SIGNAL_FACTOR
from Noise_simulation.Tremor import (
    simulate_and_add_tremor_imu,
    precompute_tremor_cache_with_relative_rms,
    TREMOR_MU, TREMOR_SIGMA, TREMOR_DT,
    TREMOR_INTERMITTENT, TREMOR_ON_PROB, TREMOR_MIN_ON_SEC, TREMOR_MAX_ON_SEC
)

# ============================================================
# CONFIG - Basic parameters
# ============================================================
ORIGINAL_FS = 50        # original sampling rate in the dataset (Hz)
FS = 30                 # target sampling rate (Hz) - set same as ORIGINAL_FS to skip resampling
WINDOW_SEC = 2.0        # window length in seconds
STRIDE_SEC = 2.0        # stride in seconds
AUG_SIZE = 2            # number of augmented copies per window
NOISE_LEVEL = 0.01      # augmentation noise level (NOT scenario noise)
NUM_SUBJECTS = 10

# Dataset location (folder containing mHealth_subject*.log)
DATA_PATH = Path(
    "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/MHEALTHDATASET"
)

# Output base folder
OUT_BASE = Path(
    "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files"
)

# Seeds for reproducibility
SEED = 0                        # Controls augmentation noise
NOISE_SCENARIO_SEED = 123       # Controls scenario noise (change this to get different tremor realizations)

# ============================================================
# CONFIG - Scenario Noise (applied to RAW signal BEFORE preprocessing)
# ============================================================
# Set SCENARIO_NOISE_TYPE to None for clean dataset (no scenario noise)
# Options: None, "AWGN", "DROPOUT", "WEAK_SIGNAL", "TREMOR"
SCENARIO_NOISE_TYPE = None # or None for clean dataset

# Noise parameters - defaults are imported from Noise_simulation/ modules
# Uncomment and modify any parameter below to override the defaults:

# AWGN_RMS_RATIO = 0.3          # Override default (0.2) - noise RMS as fraction of signal RMS
# DROPOUT_VALUE = 0.0           # Override default (0.0)  
# WEAK_SIGNAL_FACTOR = 0.3      # Override default (0.2)

# TREMOR: Relative RMS configuration (tremor scaled to window signal RMS)
# Format: target_tremor_rms = ALPHA * window_rms (clipped to [MIN, MAX])
TREMOR_ACC_ALPHA = 0.05       # 5% of window RMS
TREMOR_GYRO_ALPHA = 0.05      # 5% of window RMS
TREMOR_MAG_ALPHA = 0.02       # 2% of window RMS (magnetometers typically quieter)

# Clipping ranges to avoid extreme cases (in raw signal units)
TREMOR_ACC_RMS_MIN = 0.05     # Minimum tremor RMS for accelerometer
TREMOR_ACC_RMS_MAX = 2.0      # Maximum tremor RMS for accelerometer
TREMOR_GYRO_RMS_MIN = 0.03    # Minimum tremor RMS for gyroscope
TREMOR_GYRO_RMS_MAX = 1.5     # Maximum tremor RMS for gyroscope
TREMOR_MAG_RMS_MIN = 0.01     # Minimum tremor RMS for magnetometer
TREMOR_MAG_RMS_MAX = 0.5      # Maximum tremor RMS for magnetometer

# Note: When SCENARIO_NOISE_TYPE is set, ALL sensors receive noise in ALL windows
#       For TREMOR: Always applies to all IMU sensors (Acc/Gyro/Mag on ankle/arm/chest)
#       To get different tremor scenarios, change NOISE_SCENARIO_SEED above
#       Tremor strength is now RELATIVE to window signal RMS (more realistic)

# ============================================================
# SENSOR COLUMN MAP (0-indexed)
# From README column layout: chest acc [0,1,2], ECG [3,4], etc.
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

LABEL_COL = 23  # label column in file (0-indexed), values: 0..12, we use 1..12


# ============================================================
# Helpers
# ============================================================
def format_stride(s: float) -> str:
    """Format stride for folder naming."""
    if float(s).is_integer():
        return str(int(s))
    return str(s).rstrip("0").rstrip(".")


def sensor_to_short_code(sensor_name: str) -> str:
    """
    Convert sensor name to short code.
    Examples: Acc_ankle -> AAn, Gyro_arm -> GAr, ECG -> E
    """
    parts = sensor_name.split('_')
    
    # Sensor type mapping
    type_map = {
        'Acc': 'A',
        'Gyro': 'G',
        'Mag': 'M',
        'ECG': 'E'
    }
    
    # Location mapping
    loc_map = {
        'ankle': 'An',
        'arm': 'Ar',
        'chest': 'C'
    }
    
    sensor_type = parts[0]
    code = type_map.get(sensor_type, sensor_type[0])
    
    # Add location if present
    if len(parts) > 1:
        location = parts[1]
        code += loc_map.get(location, location[:2])
    
    return code


def make_scenario_tag() -> str:
    """Create tag for scenario noise configuration."""
    if SCENARIO_NOISE_TYPE is None:
        return ""
    
    # Base tag with noise type
    tag = f"{SCENARIO_NOISE_TYPE}"
    
    # Add parameter
    if SCENARIO_NOISE_TYPE == "AWGN":
        tag += f"_rms{AWGN_RMS_RATIO}".replace(".", "p")
    elif SCENARIO_NOISE_TYPE == "DROPOUT":
        tag += "_drop"
    elif SCENARIO_NOISE_TYPE == "WEAK_SIGNAL":
        tag += f"_w{WEAK_SIGNAL_FACTOR}".replace(".", "p")
    elif SCENARIO_NOISE_TYPE == "TREMOR":
        tag += f"_mu{TREMOR_MU}_s{TREMOR_SIGMA}".replace(".", "p")
        if TREMOR_INTERMITTENT:
            tag += "_int"
    
    return tag


def make_out_dir() -> Tuple[Path, Path]:
    """
    Create output directory with hierarchical structure.
    Returns: (parent_dir, variant_dir)
    
    Structure:
        parent_dir/     # s{STRIDE}_w{WINDOW}_aug{AUG}
            variant_dir/  # fs{FS}_clean, fs{FS}_AWGN_s0p3_AAnAC_n1, etc.
                [sensor files]
            splits.npz    # Shared across all variants (same windows regardless of FS)
            base_info.txt # Base configuration info
    
    Note: FS is NOT in parent name because windows are defined on ORIGINAL_FS,
          then resampled. So fs10 and fs50 variants share the same base windows.
    """
    stride_str = format_stride(STRIDE_SEC)
    win_str = format_stride(WINDOW_SEC)
    
    # Parent directory name (base configuration - no FS!)
    parent_name = f"s{stride_str}_w{win_str}_aug{AUG_SIZE}"
    parent_dir = OUT_BASE / parent_name
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Variant subdirectory name (includes FS + noise config)
    fs_prefix = f"fs{FS}"
    if SCENARIO_NOISE_TYPE is None:
        variant_name = f"{fs_prefix}_clean"
    else:
        variant_name = f"{fs_prefix}_{make_scenario_tag()}"
    
    variant_dir = parent_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    return parent_dir, variant_dir

def resample_window(win: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    """
    Resample window from original sampling rate to target sampling rate.
    win: shape (L, C)
    Returns: shape (L_new, C)
    """
    if original_fs == target_fs:
        return win

    L_original, C = win.shape
    L_target = int(round(L_original * target_fs / original_fs))

    resampled = np.zeros((L_target, C))
    for c in range(C):
        resampled[:, c] = resample(win[:, c], L_target)

    return resampled


def zscore_window(win: np.ndarray) -> np.ndarray:
    """
    Normalize per channel within window.
    win: shape (L, C)
    """
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def load_subject_data_with_retry(file_path: Path, max_retries: int = 5, delay: float = 2.0) -> np.ndarray:
    """
    Load subject data with retry logic for OneDrive timeout issues.
    
    Args:
        file_path: Path to subject data file
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Loaded data as numpy array
    """
    for attempt in range(max_retries):
        try:
            data = np.loadtxt(file_path)
            
            # Check if file appears to be incomplete/corrupt (OneDrive sync issue)
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
    Return (N, C*L) in the format:
      [ch1_1..ch1_L | ch2_1..ch2_L | ... | chC_1..chC_L]
    """
    N, C, L = X_c_l.shape
    blocks = [X_c_l[:, c, :] for c in range(C)]  # each (N, L)
    return np.concatenate(blocks, axis=1)        # (N, C*L)


# ============================================================
# Scenario Noise Functions
# ============================================================
# Note: Noise functions are now imported from Noise_simulation/
# - AWGN: Additive White Gaussian Noise
# - Dropout: Complete sensor failure (constant value)
# - WeakSignal: Attenuated signal strength
# - Tremor: Stochastic van der Pol oscillator-based tremor
#           Always applies to ALL IMU sensors (Acc/Gyro/Mag on all body parts)
#           Change NOISE_SCENARIO_SEED to get different tremor scenarios

# Global cache for tremor realizations: key = (window_idx, body_part)
# value = dict with keys: 'acc_noise', 'gyro_noise', 'mag_noise' (each (L,3)), 'meta'
_TREMOR_CACHE = {}


def extract_body_part(sensor_name: str) -> str:
    """Extract body part from sensor name (e.g., 'Acc_arm' -> 'arm', 'ECG' -> 'chest')."""
    if 'ankle' in sensor_name:
        return 'ankle'
    elif 'arm' in sensor_name:
        return 'arm'
    else:
        return 'chest'


def get_sensor_type(sensor_name: str) -> str:
    """Extract sensor type from sensor name (e.g., 'Acc_arm' -> 'Acc', 'Gyro_ankle' -> 'Gyro')."""
    return sensor_name.split('_')[0]


def apply_scenario_noise(
    win: np.ndarray,
    noise_type: str,
    window_idx: int,
    sensor_name: str
) -> np.ndarray:
    """
    Apply scenario noise based on type.
    Uses deterministic seed per (window, sensor) to avoid loop-order dependency.
    win: shape (L, C)
    
    Note: Noise functions are imported from Noise_simulation/
    For AWGN/TREMOR: Applied to RAW signal BEFORE resampling and z-score.
    For TREMOR: uses cache to ensure Acc/Gyro/Mag share same tremor per body part.
    """
    # Create deterministic seed per (window_idx, sensor_name)
    # Use sensor index from SENSORS dict for deterministic offset
    sensor_names = list(SENSORS.keys())
    sensor_offset = sensor_names.index(sensor_name) if sensor_name in sensor_names else 0
    seed = (NOISE_SCENARIO_SEED + window_idx * 9176 + sensor_offset * 1000) & 0xffffffff
    local_rng = np.random.RandomState(seed)
    
    if noise_type == "AWGN":
        return apply_awgn_rms_ratio(win, AWGN_RMS_RATIO, local_rng)
    elif noise_type == "DROPOUT":
        return apply_dropout(win, DROPOUT_VALUE)
    elif noise_type == "WEAK_SIGNAL":
        return apply_weak_signal(win, WEAK_SIGNAL_FACTOR)
    elif noise_type == "TREMOR":
        # For tremor: ensure consistency across Acc/Gyro/Mag for same body part
        # Cache is pre-populated in main() with window-relative RMS
        body_part = extract_body_part(sensor_name)
        sensor_type = get_sensor_type(sensor_name)
        
        # Get tremor from pre-computed cache
        cache_key = (window_idx, body_part)
        if cache_key not in _TREMOR_CACHE:
            raise RuntimeError(f"Tremor cache missing for {cache_key}. "
                             "Ensure precompute_tremor_cache() was called before sensor processing.")
        
        # Get appropriate noise based on sensor type
        tremor_data = _TREMOR_CACHE[cache_key]
        if sensor_type == 'Acc':
            noise = tremor_data['acc_noise']
        elif sensor_type == 'Gyro':
            noise = tremor_data['gyro_noise']
        elif sensor_type == 'Mag':
            noise = tremor_data['mag_noise']
        else:
            # ECG or other sensors: no tremor applied
            return win
        
        # Ensure noise shape matches window shape
        if noise.shape != win.shape:
            raise ValueError(f"Tremor noise shape {noise.shape} doesn't match window {win.shape}")
        
        return win + noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def augment_window(win: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Apply original augmentation noise (scaled by max_abs).
    win: shape (L, C)
    """
    if AUG_SIZE > 1:
        scale = NOISE_LEVEL * np.max(np.abs(win), axis=0, keepdims=True)
        noise = scale * rng.randn(*win.shape)
        return win + noise
    else:
        return win


# ============================================================
# Main generation
# ============================================================
def main():
    # Seed for augmentation noise
    random.seed(SEED)
    np.random.seed(SEED)
    aug_rng = np.random.RandomState(SEED)
    
    # Separate RNG for scenario noise
    scenario_rng = np.random.RandomState(NOISE_SCENARIO_SEED)

    # Window specs are based on ORIGINAL sampling rate
    window_len_original = int(round(ORIGINAL_FS * WINDOW_SEC))
    stride_original = int(round(ORIGINAL_FS * STRIDE_SEC))
    stride_original = max(stride_original, 1)

    # Target window length after resampling
    window_len = int(round(FS * WINDOW_SEC))
    stride = int(round(FS * STRIDE_SEC))
    stride = max(stride, 1)

    # Output folders (hierarchical structure)
    parent_dir, variant_dir = make_out_dir()

    print(f"=" * 80)
    print(f"Data Generation Configuration")
    print(f"=" * 80)
    print(f"FS: {ORIGINAL_FS}Hz -> {FS}Hz")
    print(f"Window: {WINDOW_SEC}s ({window_len} samples), Stride: {STRIDE_SEC}s ({stride} samples)")
    print(f"AUG_SIZE: {AUG_SIZE} (augmentation copies per window)")
    print(f"Augmentation NOISE_LEVEL: {NOISE_LEVEL}")
    print(f"Augmentation SEED: {SEED}")
    print(f"-" * 80)
    
    if SCENARIO_NOISE_TYPE is not None:
        print(f"Scenario Noise: {SCENARIO_NOISE_TYPE}")
        if SCENARIO_NOISE_TYPE == "AWGN":
            print(f"  - AWGN RMS ratio: {AWGN_RMS_RATIO}")
        elif SCENARIO_NOISE_TYPE == "DROPOUT":
            print(f"  - Dropout value: {DROPOUT_VALUE}")
        elif SCENARIO_NOISE_TYPE == "WEAK_SIGNAL":
            print(f"  - Weak signal factor: {WEAK_SIGNAL_FACTOR}")
        elif SCENARIO_NOISE_TYPE == "TREMOR":
            print(f"  - Tremor mu: {TREMOR_MU}, sigma: {TREMOR_SIGMA}")
            print(f"  - Tremor alpha (relative RMS):")
            print(f"      Acc: {TREMOR_ACC_ALPHA} (range: [{TREMOR_ACC_RMS_MIN}, {TREMOR_ACC_RMS_MAX}])")
            print(f"      Gyro: {TREMOR_GYRO_ALPHA} (range: [{TREMOR_GYRO_RMS_MIN}, {TREMOR_GYRO_RMS_MAX}])")
            print(f"      Mag: {TREMOR_MAG_ALPHA} (range: [{TREMOR_MAG_RMS_MIN}, {TREMOR_MAG_RMS_MAX}])")
            print(f"  - Note: Target RMS = alpha * window_RMS (per window)")
            print(f"  - Intermittent: {TREMOR_INTERMITTENT}")
            if TREMOR_INTERMITTENT:
                print(f"  - On prob: {TREMOR_ON_PROB}, duration: {TREMOR_MIN_ON_SEC}-{TREMOR_MAX_ON_SEC}s")
        print(f"  - All sensors corrupted in all windows")
        print(f"  - Scenario SEED: {NOISE_SCENARIO_SEED}")
    else:
        print(f"Scenario Noise: DISABLED (clean dataset)")
    
    print(f"=" * 80)
    print(f"Parent directory: {parent_dir}")
    print(f"Variant directory: {variant_dir}")
    print(f"=" * 80)

    # -------------------------------
    # Step 1: Build global list of base windows (canonical ordering)
    # -------------------------------
    window_specs = []
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        data = load_subject_data_with_retry(file_path)
        labels = data[:, LABEL_COL].astype(int)

        for label in range(1, 13):
            idx = np.where(labels == label)[0]
            if idx.size == 0:
                continue

            start_i, end_i = int(idx[0]), int(idx[-1]) + 1
            seg_len = end_i - start_i
            if seg_len < window_len_original:
                continue

            for offset in range(0, seg_len - window_len_original + 1, stride_original):
                win_start = start_i + offset
                win_end = win_start + window_len_original
                window_specs.append((subj_idx, label, win_start, win_end))

    if len(window_specs) == 0:
        raise RuntimeError("No windows generated. Check WINDOW_SEC/STRIDE_SEC and dataset path.")

    print(f"\nTotal base windows (before augmentation): {len(window_specs)}")

    # -------------------------------
    # Step 2: Precompute which sensors are corrupted for each window
    # -------------------------------
    corruption_log = []  # List of dicts: {window_idx, subject_id, label, corrupted_sensors}
    
    if SCENARIO_NOISE_TYPE is not None:
        # All sensors get noise in all windows
        all_sensors = list(SENSORS.keys())
        for w_idx, (subj_idx, label, _, _) in enumerate(window_specs):
            corruption_log.append({
                "window_idx": w_idx,
                "subject_id": subj_idx,
                "label": label,
                "corrupted_sensors": all_sensors
            })
    else:
        # No corruption (clean dataset)
        for w_idx, (subj_idx, label, _, _) in enumerate(window_specs):
            corruption_log.append({
                "window_idx": w_idx,
                "subject_id": subj_idx,
                "label": label,
                "corrupted_sensors": []
            })
    
    # Save corruption log to variant directory
    corruption_log_path = variant_dir / "corruption_log.jsonl"
    with open(corruption_log_path, 'w') as f:
        for entry in corruption_log:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved corruption log: {corruption_log_path.name}")
    
    # Compute and save corruption summary
    if SCENARIO_NOISE_TYPE is not None:
        all_sensors = list(SENSORS.keys())
        sensor_counts = {sensor: len(window_specs) for sensor in all_sensors}
        
        summary_path = variant_dir / "corruption_summary.json"
        summary = {
            "total_windows": len(window_specs),
            "corruption_counts": sensor_counts,
            "corruption_percentages": {
                sensor: 100.0
                for sensor in all_sensors
            },
            "note": "All sensors are corrupted in all windows"
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved corruption summary: {summary_path.name}")
        print(f"\nCorruption: ALL sensors in ALL windows (100%)")
    
    # -------------------------------
    # Step 2.5: Pre-compute tremor cache (if using TREMOR)
    # -------------------------------
    if SCENARIO_NOISE_TYPE == "TREMOR":
        # Call tremor pre-computation from Tremor.py module
        def subject_data_loader(subj_idx: int) -> np.ndarray:
            """Wrapper for loading subject data."""
            file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
            return load_subject_data_with_retry(file_path)
        
        _TREMOR_CACHE.update(
            precompute_tremor_cache_with_relative_rms(
                window_specs=window_specs,
                sensor_column_mapping=SENSORS,
                data_loader_func=subject_data_loader,
                fs=ORIGINAL_FS,
                tremor_mu=TREMOR_MU,
                tremor_sigma=TREMOR_SIGMA,
                tremor_dt=TREMOR_DT,
                tremor_acc_alpha=TREMOR_ACC_ALPHA,
                tremor_gyro_alpha=TREMOR_GYRO_ALPHA,
                tremor_mag_alpha=TREMOR_MAG_ALPHA,
                tremor_acc_rms_min=TREMOR_ACC_RMS_MIN,
                tremor_acc_rms_max=TREMOR_ACC_RMS_MAX,
                tremor_gyro_rms_min=TREMOR_GYRO_RMS_MIN,
                tremor_gyro_rms_max=TREMOR_GYRO_RMS_MAX,
                tremor_mag_rms_min=TREMOR_MAG_RMS_MIN,
                tremor_mag_rms_max=TREMOR_MAG_RMS_MAX,
                tremor_intermittent=TREMOR_INTERMITTENT,
                tremor_on_prob=TREMOR_ON_PROB,
                tremor_min_on_sec=TREMOR_MIN_ON_SEC,
                tremor_max_on_sec=TREMOR_MAX_ON_SEC,
                scenario_seed=NOISE_SCENARIO_SEED,
            )
        )
    
    # -------------------------------
    # Step 3: Process each sensor
    # -------------------------------
    for sensor_name, cols in SENSORS.items():
        X_list = []
        y_list = []
        subj_list = []
        base_idx_list = []  # Track which base window each sample comes from

        subj_cache = {}

        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subj_cache:
                file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
                subj_cache[subj_idx] = load_subject_data_with_retry(file_path)

            data = subj_cache[subj_idx]
            win = data[win_start:win_end, cols]  # (L_original, C) - raw signal

            # Apply scenario noise to RAW signal (BEFORE preprocessing)
            # This simulates hardware-level corruption (e.g., sensor measurement noise)
            corrupted_sensors = corruption_log[w_idx]["corrupted_sensors"]
            if sensor_name in corrupted_sensors:
                win = apply_scenario_noise(win, SCENARIO_NOISE_TYPE, w_idx, sensor_name)

            # Resample (if needed)
            win = resample_window(win, ORIGINAL_FS, FS)  # (L_target, C)

            # Normalize
            win = zscore_window(win)

            # Apply augmentation (creates AUG_SIZE copies)
            for a in range(AUG_SIZE):
                win_aug = augment_window(win.copy(), aug_rng)
                
                X_list.append(win_aug.T.astype(np.float32))  # (C, L)
                y_list.append(label - 1)                      # 0..11
                subj_list.append(subj_idx)
                base_idx_list.append(w_idx)  # Track base window index

        X = np.stack(X_list, axis=0)               # (N, C, L)
        y = np.array(y_list, dtype=np.int64)       # (N,)
        subject_ids = np.array(subj_list, dtype=np.int64)
        base_window_idx = np.array(base_idx_list, dtype=np.int64)  # (N,)

        # Save NPZ
        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X=X,
            y=y,
            subject_id=subject_ids,
            base_window_idx=base_window_idx,  # For lookup in corruption_log
            fs=FS,
            window_len=window_len,
            stride=stride,
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )

        # Save TXT
        txt_out = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)
        labels_col = (y + 1).reshape(-1, 1)
        sensor_txt = np.hstack([flat_rows, labels_col]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"[{sensor_name}] Saved: {txt_out.name}, {npz_out.name} | X={X.shape}, y={y.shape}")

    # -------------------------------
    # Step 4: Write variant info.txt
    # -------------------------------
    info_txt = variant_dir / "info.txt"
    overlap = 1.0 - (STRIDE_SEC / WINDOW_SEC) if WINDOW_SEC > 0 else 0.0
    stride_samples = int(round(FS * STRIDE_SEC))

    info_lines = [
        "=" * 60,
        "DATA GENERATION CONFIGURATION",
        "=" * 60,
        "",
        "Basic Parameters:",
        f"  - Original FS: {ORIGINAL_FS} Hz",
        f"  - Target FS: {FS} Hz",
        f"  - Window size: {WINDOW_SEC} s ({window_len} samples)",
        f"  - Stride: {STRIDE_SEC} s ({stride_samples} samples)",
        f"  - Overlap: {overlap:.2f}",
        f"  - Augmentation copies (AUG_SIZE): {AUG_SIZE}",
        f"  - Augmentation noise level: {NOISE_LEVEL}",
        f"  - Augmentation seed: {SEED}",
        "",
        "Scenario Noise Configuration:",
    ]
    
    if SCENARIO_NOISE_TYPE is not None:
        info_lines.extend([
            f"  - Type: {SCENARIO_NOISE_TYPE}",
        ])
        if SCENARIO_NOISE_TYPE == "AWGN":
            info_lines.append(f"  - AWGN RMS ratio: {AWGN_RMS_RATIO}")
        elif SCENARIO_NOISE_TYPE == "DROPOUT":
            info_lines.append(f"  - Dropout value: {DROPOUT_VALUE}")
        elif SCENARIO_NOISE_TYPE == "WEAK_SIGNAL":
            info_lines.append(f"  - Weak signal factor: {WEAK_SIGNAL_FACTOR}")
        elif SCENARIO_NOISE_TYPE == "TREMOR":
            info_lines.extend([
                f"  - Tremor mu: {TREMOR_MU}, sigma: {TREMOR_SIGMA}, dt: {TREMOR_DT}",
                f"  - Tremor alpha (relative RMS):",
                f"      Acc: {TREMOR_ACC_ALPHA} (clipped to [{TREMOR_ACC_RMS_MIN}, {TREMOR_ACC_RMS_MAX}])",
                f"      Gyro: {TREMOR_GYRO_ALPHA} (clipped to [{TREMOR_GYRO_RMS_MIN}, {TREMOR_GYRO_RMS_MAX}])",
                f"      Mag: {TREMOR_MAG_ALPHA} (clipped to [{TREMOR_MAG_RMS_MIN}, {TREMOR_MAG_RMS_MAX}])",
                f"  - Note: Tremor RMS = alpha * window_signal_RMS (per window, per sensor)",
                f"  - Intermittent: {TREMOR_INTERMITTENT}",
            ])
            if TREMOR_INTERMITTENT:
                info_lines.append(f"  - On prob: {TREMOR_ON_PROB}, duration: {TREMOR_MIN_ON_SEC}-{TREMOR_MAX_ON_SEC}s")
        info_lines.extend([
            f"  - All sensors corrupted in all windows",
            f"  - Scenario seed: {NOISE_SCENARIO_SEED}",
        ])
    else:
        info_lines.append("  - DISABLED (clean dataset, no scenario noise)")
    
    info_lines.extend([
        "",
        "=" * 60,
        "Notes:",
        "  - This variant is stored in hierarchical structure",
        f"  - Parent dir: {parent_dir.name}",
        f"  - Variant dir: {variant_dir.name}",
        "  - Scenario noise is applied BEFORE augmentation",
        "  - Augmentation noise is applied separately to each copy",
        "  - Same window index across sensor files = same time window",
        "  - corruption_log.jsonl contains per-BASE-window corruption info",
        "  - corruption_summary.json contains overall statistics",
        "  - NPZ files contain 'base_window_idx' to map samples to corruption_log",
        "  - With AUG_SIZE>1: use base_window_idx[i] to look up corruption info",
        "  - RNG per (window_idx, sensor_name) ensures reproducibility",
        "=" * 60,
    ])
    
    info_txt.write_text("\n".join(info_lines), encoding="utf-8")
    print(f"\nWrote variant {info_txt.name}")
    
    # -------------------------------
    # Step 4b: Write tremor_params.txt if TREMOR noise
    # -------------------------------
    if SCENARIO_NOISE_TYPE == "TREMOR":
        from Noise_simulation.Tremor import write_tremor_params_file
        tremor_params_txt = variant_dir / "tremor_params.txt"
        write_tremor_params_file(
            tremor_params_txt,
            mu=TREMOR_MU,
            sigma=TREMOR_SIGMA,
            dt=TREMOR_DT,
            acc_alpha=TREMOR_ACC_ALPHA,
            gyro_alpha=TREMOR_GYRO_ALPHA,
            mag_alpha=TREMOR_MAG_ALPHA,
            acc_rms_min=TREMOR_ACC_RMS_MIN,
            acc_rms_max=TREMOR_ACC_RMS_MAX,
            gyro_rms_min=TREMOR_GYRO_RMS_MIN,
            gyro_rms_max=TREMOR_GYRO_RMS_MAX,
            mag_rms_min=TREMOR_MAG_RMS_MIN,
            mag_rms_max=TREMOR_MAG_RMS_MAX,
            intermittent=TREMOR_INTERMITTENT,
            on_prob=TREMOR_ON_PROB,
            min_on_sec=TREMOR_MIN_ON_SEC,
            max_on_sec=TREMOR_MAX_ON_SEC,
            scenario_seed=NOISE_SCENARIO_SEED,
        )
        print(f"Wrote tremor parameters: {tremor_params_txt.name}")
    
    # -------------------------------
    # Step 5: Write base_info.txt in parent directory (shared info)
    # -------------------------------
    base_info_txt = parent_dir / "base_info.txt"
    base_info_lines = [
        "=" * 60,
        "BASE CONFIGURATION (Shared across all variants)",
        "=" * 60,
        "",
        "This directory contains multiple variants of the same base configuration.",
        "All variants share the same:",
        "  - Window size and stride",
        "  - Sampling rate",
        "  - Augmentation size",
        "  - Number of base windows",
        "  - Window alignment across sensors",
        "",
        "Base Parameters:",
        f"  - Original FS: {ORIGINAL_FS} Hz",
        f"  - Target FS: {FS} Hz",
        f"  - Window size: {WINDOW_SEC} s ({window_len} samples)",
        f"  - Stride: {STRIDE_SEC} s ({stride_samples} samples)",
        f"  - Overlap: {overlap:.2f}",
        f"  - Augmentation copies (AUG_SIZE): {AUG_SIZE}",
        f"  - Total base windows: {len(window_specs)}",
        f"  - Total samples per sensor: {len(window_specs) * AUG_SIZE}",
        "",
        "Available Variants:",
    ]
    
    # List all variant subdirectories
    variants = [d.name for d in parent_dir.iterdir() if d.is_dir()]
    for variant in sorted(variants):
        base_info_lines.append(f"  - {variant}/")
    
    base_info_lines.extend([
        "",
        "Shared Files:",
        "  - splits.npz: Train/validation/test split indices (to be generated)",
        "  - base_info.txt: This file",
        "",
        "=" * 60,
        "Usage:",
        "  1. All variants can share the same train/val/test splits",
        "  2. Use base_window_idx to map samples to corruption info",
        "  3. Compare model performance across different noise scenarios",
        "=" * 60,
    ])
    
    base_info_txt.write_text("\n".join(base_info_lines), encoding="utf-8")
    print(f"Wrote base {base_info_txt.name}")
    
    print(f"\n{'=' * 80}")
    print(f"Generation complete!")
    print(f"  Parent: {parent_dir}")
    print(f"  Variant: {variant_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
