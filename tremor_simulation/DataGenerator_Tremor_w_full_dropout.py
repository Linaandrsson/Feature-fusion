"""
Data Generator for Tremor-Labeled Datasets — with Full Signal Dropout
=======================================================================

Identical to DataGenerator_Tremor_w_partial_dropout.py except that the
global corruption applied to every sensor window is **full dropout**
instead of partial dropout.

Full dropout: every sample in the window is replaced with zero,

    y(t) = 0.

This models complete sensor failure scenarios such as hardware failure,
battery depletion or complete communication loss.

Sampling methods and tremor generation are unchanged from the AWGN variant.

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

import sys
from pathlib import Path

# Ensure Code/ root is on sys.path so tremor_simulation package is found
# regardless of whether this script is run directly or imported as a module.
_CODE_ROOT = Path(__file__).parent.parent.resolve()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np
import random
from scipy.signal import resample
import json
import time
from typing import List, Dict, Optional, Tuple

# Import Parkinson tremor functions
from tremor_simulation.Tremor import (
    precompute_tremor_cache_with_parkinson_model,
    write_tremor_parkinson_params_file,
    apply_tremor_rotation_to_magnetometer,
    TREMOR_MU, TREMOR_SIGMA, TREMOR_DT,
    TREMOR_INTERMITTENT, TREMOR_ON_PROB, TREMOR_MIN_ON_SEC, TREMOR_MAX_ON_SEC
)
import tremor_simulation.tremor_parkinson_config as pk_config

# ============================================================
# CONFIG - Basic parameters
# ============================================================
ORIGINAL_FS = 50        # original sampling rate in the dataset (Hz)
FS = 50                 # target sampling rate (Hz)
WINDOW_SEC = 4.0        # window length in seconds
STRIDE_SEC = 4.0        # stride in seconds
AUG_SIZE = 1            # number of augmented copies per window
NOISE_LEVEL = 0.01      # augmentation noise level (NOT tremor)
NUM_SUBJECTS = 10

# Dataset location
script_dir = Path(__file__).parent.resolve()
_code_dir = script_dir.parent  # Code/ root (one level up from tremor_simulation/)
DATA_PATH = Path(_code_dir / "data" / "MHEALTHDATASET")
OUT_BASE = Path(_code_dir / "data" / "Tremor_datagenerator_files")

# Seeds
SEED = 0                        # Controls augmentation noise
TREMOR_SEED = 42                # Controls tremor generation (change for different realizations)

# ============================================================
# CONFIG - Diagnostics
# ============================================================
# Set to True to run sanity checks on generated tremor
RUN_SANITY_CHECKS = True
NUM_DIAGNOSTIC_WINDOWS = 10      # Number of windows to inspect for diagnostics

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


def generate_rotation_matrix(rng: np.random.Generator, max_angle_deg: float = 15.0) -> np.ndarray:
    """
    Generate a random 3D rotation matrix from small random orientation perturbations.

    Args:
        rng: Random number generator
        max_angle_deg: Maximum rotation angle in degrees (default 15°)

    Returns:
        R: 3x3 rotation matrix

    Note:
        Rotation simulates realistic sensor orientation variations (e.g., slight
        misalignment during attachment) while preserving physical motion structure.
    """
    max_angle_rad = np.deg2rad(max_angle_deg)
    angles = rng.uniform(-max_angle_rad, max_angle_rad, size=3)

    theta_x, theta_y, theta_z = angles

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R


def apply_rotation_augmentation(win_raw: np.ndarray, rng: np.random.Generator, max_angle_deg: float = 15.0) -> np.ndarray:
    """
    Apply rotation-based data augmentation to 3-axis sensor data.

    Args:
        win_raw: Window of shape (L, 3) with raw sensor readings (BEFORE z-score)
        rng: Random number generator
        max_angle_deg: Maximum rotation angle in degrees

    Returns:
        Rotated window of shape (L, 3)
    """
    R = generate_rotation_matrix(rng, max_angle_deg)
    return win_raw @ R.T


def apply_awgn_raw(win_raw: np.ndarray, rng: np.random.Generator, rms_ratio: float = 0.2) -> np.ndarray:
    """
    Apply AWGN to raw sensor signal with noise level relative to signal RMS.
    Used for tremor-free sensor augmentation strategy.

    Args:
        win_raw: Raw input window of shape (L, C) BEFORE preprocessing
        rng: Random number generator
        rms_ratio: Noise RMS as a fraction of signal RMS per channel

    Returns:
        Corrupted window with same shape (L, C)
    """
    L, C = win_raw.shape
    noise = np.zeros_like(win_raw)

    for c in range(C):
        signal_rms = np.sqrt(np.mean(win_raw[:, c]**2))
        if signal_rms < 1e-8:
            signal_rms = 1.0
        noise_std = rms_ratio * signal_rms
        noise[:, c] = rng.normal(0, noise_std, size=L)

    return win_raw + noise


def apply_full_dropout(win: np.ndarray) -> np.ndarray:
    """
    Apply full signal dropout by replacing all samples with zero.

    Models complete sensor failure (hardware failure, battery depletion,
    communication loss). Applied after resampling, before z-score normalization.

    Args:
        win: Input window of shape (L, C)

    Returns:
        Zero array of same shape (L, C)
    """
    return np.zeros_like(win)


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


def save_tremor_branch_sensor(
    variant_dir: Path,
    sensor_name: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    base_window_idx: np.ndarray,
    tremor_freq: np.ndarray,
    tremor_acc_rms: np.ndarray,
    tremor_gyro_rms: np.ndarray,
    tremor_score: np.ndarray,
    sensor_cols: List[int],
    fs: int,
    window_len: int,
    stride: int
) -> None:
    """
    Save tremor-branch sensor data (non-normalized, for tremor severity estimation).

    Pipeline for tremor-branch data:
      1. Extract raw window
      2. Apply tremor/augmentation (same as HAR)
      3. Resample to target FS (same as HAR)
      4. Apply full dropout (same as HAR)
      5. SKIP z-score normalization (DIFFERENT from HAR)
      6. SKIP augment_window noise (DIFFERENT from HAR)
    """
    tremor_branch_name = f"{sensor_name}_tremorbranch"

    npz_out = variant_dir / f"{tremor_branch_name}.npz"
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
        fs=fs,
        window_len=window_len,
        stride=stride,
        sensor_name=tremor_branch_name,
        sensor_cols=np.array(sensor_cols, dtype=np.int64),
    )

    txt_out = variant_dir / f"{tremor_branch_name}.txt"
    flat_rows = flatten_channel_blocks(X)

    all_labels = np.column_stack([
        (y + 1),
        subject_ids,
        base_window_idx,
        tremor_freq,
        tremor_acc_rms,
        tremor_gyro_rms,
        tremor_score
    ])

    sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
    np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

    print(f"    ✓ Saved: {npz_out.name}, {txt_out.name} | X={X.shape}, tremor_labels={tremor_freq.shape}")


# ============================================================
# Diagnostic Utilities
# ============================================================

def run_tremor_sanity_checks(tremor_cache: dict, window_specs: list,
                             data_loader_func, num_windows: int = 5):
    """
    Diagnostic utility to verify tremor generation pipeline.

    Checks:
    - Rotation-based magnetometer tremor (Mag_arm, Mag_ankle)
    - Additive tremor for acc/gyro
    - Tremor-free sensor policy (Acc_chest, ECG)
    - Ankle tremor scaling
    """
    print("\n" + "="*80)
    print("TREMOR PIPELINE SANITY CHECKS")
    print("="*80)

    sample_indices = np.linspace(0, len(window_specs)-1, num_windows, dtype=int)

    mag_ankle_stats = {
        'gyro_rms': [],
        'mag_perturbation_rms': [],
        'mag_original_norm': [],
        'mag_rotated_norm': []
    }

    print("\n" + "-"*80)
    print("CHECKING SAMPLE WINDOWS")
    print("-"*80)

    for idx in sample_indices:
        w_idx = int(idx)
        subj_idx, label, win_start, win_end = window_specs[w_idx]

        print(f"\nWindow {w_idx}: Subject {subj_idx}, Activity {label}")

        for body_part in ['arm', 'ankle', 'chest']:
            cache_entry = tremor_cache.get((w_idx, body_part))
            if cache_entry is None:
                continue

            meta = cache_entry['meta']
            acc_rms = meta.get('acc_target_rms', 0.0)
            gyro_rms = meta.get('gyro_target_rms', 0.0)
            freq = meta.get('freq_hz', 0.0)

            acc_noise = cache_entry['acc_noise']
            gyro_noise = cache_entry['gyro_noise']
            mag_noise = cache_entry['mag_noise']

            acc_noise_rms = np.sqrt(np.mean(acc_noise**2))
            gyro_noise_rms = np.sqrt(np.mean(gyro_noise**2))
            mag_noise_rms = np.sqrt(np.mean(mag_noise**2))

            print(f"  {body_part.upper()}: acc_target={acc_rms:.4f}, gyro_target={gyro_rms:.4f}, "
                  f"freq={freq:.2f} Hz")
            print(f"    Noise RMS: acc={acc_noise_rms:.4f}, gyro={gyro_noise_rms:.4f}, "
                  f"mag={mag_noise_rms:.4f}")

    print("\n" + "-"*80)
    print("MAG_ANKLE ROTATION-BASED TREMOR ANALYSIS")
    print("-"*80)

    data = data_loader_func(window_specs[sample_indices[0]][0])

    for idx in sample_indices:
        w_idx = int(idx)
        subj_idx, label, win_start, win_end = window_specs[w_idx]

        if subj_idx != window_specs[sample_indices[0]][0]:
            data = data_loader_func(subj_idx)

        ankle_cache = tremor_cache.get((w_idx, 'ankle'))
        if ankle_cache is None:
            continue

        mag_cols = SENSORS['Mag_ankle']
        mag_original = data[win_start:win_end, mag_cols].copy()
        gyro_tremor = ankle_cache['gyro_noise']

        mag_rotated = apply_tremor_rotation_to_magnetometer(
            mag_signal=mag_original,
            gyro_tremor=gyro_tremor,
            fs=ORIGINAL_FS
        )

        mag_perturbation = mag_rotated - mag_original

        gyro_rms = np.sqrt(np.mean(gyro_tremor**2))
        perturbation_rms = np.sqrt(np.mean(mag_perturbation**2))
        original_norm_mean = np.mean(np.linalg.norm(mag_original, axis=1))
        rotated_norm_mean = np.mean(np.linalg.norm(mag_rotated, axis=1))

        mag_ankle_stats['gyro_rms'].append(gyro_rms)
        mag_ankle_stats['mag_perturbation_rms'].append(perturbation_rms)
        mag_ankle_stats['mag_original_norm'].append(original_norm_mean)
        mag_ankle_stats['mag_rotated_norm'].append(rotated_norm_mean)

        print(f"\nWindow {w_idx}:")
        print(f"  Gyro tremor RMS: {gyro_rms:.4f} deg/s")
        print(f"  Mag perturbation RMS: {perturbation_rms:.4f}")
        print(f"  Mag vector norm: original={original_norm_mean:.2f}, "
              f"rotated={rotated_norm_mean:.2f} (should be ~equal)")
        print(f"  Perturbation / Gyro ratio: {perturbation_rms/gyro_rms if gyro_rms > 0 else 0:.4f}")

    print("\n" + "-"*80)
    print("MAG_ANKLE SUMMARY STATISTICS")
    print("-"*80)
    print(f"Gyro tremor RMS: mean={np.mean(mag_ankle_stats['gyro_rms']):.4f}, "
          f"std={np.std(mag_ankle_stats['gyro_rms']):.4f}")
    print(f"Mag perturbation RMS: mean={np.mean(mag_ankle_stats['mag_perturbation_rms']):.4f}, "
          f"std={np.std(mag_ankle_stats['mag_perturbation_rms']):.4f}")
    print(f"Mag vector norm preservation:")
    print(f"  Original: mean={np.mean(mag_ankle_stats['mag_original_norm']):.2f}")
    print(f"  Rotated:  mean={np.mean(mag_ankle_stats['mag_rotated_norm']):.2f}")
    print(f"  Difference: {abs(np.mean(mag_ankle_stats['mag_original_norm']) - np.mean(mag_ankle_stats['mag_rotated_norm'])):.4f} (should be ~0)")

    print("\n" + "-"*80)
    print("SENSOR POLICY COMPLIANCE CHECK")
    print("-"*80)

    print(f"\nTremor-free sensors: {pk_config.TREMOR_FREE_SENSORS}")
    for sensor in pk_config.TREMOR_FREE_SENSORS:
        body_part = extract_body_part(sensor)
        cache_entry = tremor_cache.get((sample_indices[0], body_part))
        if cache_entry:
            meta = cache_entry['meta']
            print(f"  {sensor}: acc_rms={meta.get('acc_target_rms', 0.0):.4f} "
                  f"(chest always has values, but sensor gets no tremor)")

    print(f"\nRotation-based mag sensors: {pk_config.ROTATION_BASED_MAG_SENSORS}")
    print("  These use gyro-driven rotation, NOT additive mag_noise")
    for sensor in pk_config.ROTATION_BASED_MAG_SENSORS:
        body_part = extract_body_part(sensor)
        cache_entry = tremor_cache.get((sample_indices[0], body_part))
        if cache_entry:
            mag_noise_rms = np.sqrt(np.mean(cache_entry['mag_noise']**2))
            print(f"  {sensor}: mag_noise RMS={mag_noise_rms:.6f} (should be ~0)")

    print(f"\nAnkle tremor scaling (interval mode):")
    print(f"  ANKLE_RATIO_BY_SCORE: {pk_config.ANKLE_RATIO_BY_SCORE}")

    arm_acc_rms = []
    ankle_acc_rms = []
    for idx in sample_indices[:3]:
        w_idx = int(idx)
        arm_cache = tremor_cache.get((w_idx, 'arm'))
        ankle_cache = tremor_cache.get((w_idx, 'ankle'))
        if arm_cache and ankle_cache:
            arm_rms = arm_cache['meta'].get('acc_target_rms', 0.0)
            ankle_rms = ankle_cache['meta'].get('acc_target_rms', 0.0)
            arm_acc_rms.append(arm_rms)
            ankle_acc_rms.append(ankle_rms)
            ratio = ankle_rms / arm_rms if arm_rms > 0 else 0
            score = pk_config.get_tremor_score(arm_rms)
            expected_ratio = pk_config.ANKLE_RATIO_BY_SCORE.get(score, 0)
            print(f"  Window {w_idx}: arm={arm_rms:.4f}, ankle={ankle_rms:.4f}, "
                  f"ratio={ratio:.3f}, expected={expected_ratio:.3f}, score={score}")

    print("\n" + "="*80)
    print("SANITY CHECKS COMPLETE")
    print("="*80)


# ============================================================
# Main Processing
# ============================================================

def generate_tremor_dataset(variant_name: str, augment_mode: str = None):
    """
    Generate dataset with tremor labels and full signal dropout corruption.

    Args:
        variant_name: Name for this dataset variant (e.g., "s4_w4_fs50_tremor_clean_fullDrop")
        augment_mode: Tremor augmentation mode ("clean", "mild_mod", "mod_severe")
                     If None, uses pk_config.DEFAULT_AUGMENT_MODE
    """

    if augment_mode is None:
        augment_mode = pk_config.DEFAULT_AUGMENT_MODE

    print("=" * 80)
    print(f"TREMOR-LABELED DATASET GENERATION: {variant_name}")
    print("=" * 80)
    print(f"Generate tremor: {GENERATE_TREMOR}")
    print(f"Original FS: {ORIGINAL_FS} Hz, Target FS: {FS} Hz")
    print(f"Window: {WINDOW_SEC}s, Stride: {STRIDE_SEC}s")
    print(f"Augmentation: {AUG_SIZE} copies per window")
    print(f"Augmentation seed: {SEED}, Tremor seed: {TREMOR_SEED}")
    print(f"Corruption: FULL DROPOUT (y(t) = 0 for all t)")
    print("=" * 80)

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
            augment_mode=augment_mode,
        )
    else:
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
    # Optional: Run sanity checks on tremor cache
    # -------------------------------
    if RUN_SANITY_CHECKS and GENERATE_TREMOR:
        run_tremor_sanity_checks(
            tremor_cache=tremor_cache,
            window_specs=window_specs,
            data_loader_func=load_subject_data_func,
            num_windows=NUM_DIAGNOSTIC_WINDOWS
        )

    # -------------------------------
    # Step 3: Process each sensor and save
    # -------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Processing sensors and applying tremor + full dropout...")
    print("=" * 80)

    aug_rng = np.random.default_rng(SEED)
    target_window_len = int(round(WINDOW_SEC * FS))

    for sensor_name, cols in SENSORS.items():
        print(f"\n  Processing: {sensor_name}")

        X_list = []
        y_list = []
        subj_list = []
        base_idx_list = []

        tremor_freq_list = []
        tremor_acc_rms_list = []
        tremor_gyro_rms_list = []
        tremor_score_list = []

        sensor_type = get_sensor_type(sensor_name)
        signal_body_part = extract_body_part(sensor_name)

        if sensor_name in pk_config.TREMOR_FREE_SENSORS:
            severity_body_part = 'arm'
        else:
            severity_body_part = signal_body_part

        subject_data_cache = {}
        for subj_idx in range(1, NUM_SUBJECTS + 1):
            fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
            if fname.exists():
                subject_data_cache[subj_idx] = load_subject_data_with_retry(fname)

        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subject_data_cache:
                continue

            data = subject_data_cache[subj_idx]

            # Extract window data
            win_raw = data[win_start:win_end, cols].copy()

            # Get severity from appropriate cache for augmentation strategy
            severity_cache_entry = tremor_cache.get((w_idx, severity_body_part))
            severity_meta = severity_cache_entry.get('meta', {}) if severity_cache_entry else {}
            tremor_acc_rms_target = severity_meta.get('acc_target_rms', 0.0)
            if GENERATE_TREMOR:
                tremor_score_val = int(
                    severity_meta.get(
                        'sampled_score',
                        severity_meta.get('score', pk_config.get_tremor_score(tremor_acc_rms_target))
                    )
                )
            else:
                tremor_score_val = 0

            apply_awgn_aug = False
            apply_rotation_aug = False

            if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                if tremor_score_val == 0:
                    pass
                elif tremor_score_val in [1, 2]:
                    apply_awgn_aug = True
                else:
                    if sensor_type == 'acc':
                        apply_rotation_aug = True
                    else:
                        apply_awgn_aug = True
            else:
                signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
                if signal_cache_entry is not None:
                    if sensor_name in pk_config.ROTATION_BASED_MAG_SENSORS:
                        gyro_tremor = signal_cache_entry['gyro_noise']
                        win_raw = apply_tremor_rotation_to_magnetometer(
                            mag_signal=win_raw,
                            gyro_tremor=gyro_tremor,
                            fs=ORIGINAL_FS
                        )
                    elif sensor_type in ['acc', 'gyro', 'mag']:
                        noise_key = f'{sensor_type}_noise'
                        tremor_noise = signal_cache_entry[noise_key]
                        win_raw += tremor_noise

            # Apply AWGN augmentation if applicable (before resampling)
            if apply_awgn_aug:
                win_raw = apply_awgn_raw(win_raw, aug_rng, rms_ratio=0.15)

            # Apply rotation augmentation if applicable (before resampling)
            if apply_rotation_aug:
                win_raw = apply_rotation_augmentation(win_raw, aug_rng, max_angle_deg=10.0)

            # Resample
            win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)

            # Apply global full dropout corruption (after resampling, before z-score)
            win_resampled = apply_full_dropout(win_resampled)

            # Z-score normalize
            # Note: zscore of all-zero signal returns zeros (std guard sets std=1)
            win = zscore_window(win_resampled)

            # Set tremor labels
            if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                tremor_freq = 0.0
                tremor_acc_rms = 0.0
                tremor_gyro_rms = 0.0
                tremor_score = 0
            else:
                signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
                if signal_cache_entry is not None and GENERATE_TREMOR:
                    meta = signal_cache_entry['meta']
                    tremor_freq = meta.get('freq_hz', 0.0)
                    tremor_acc_rms = meta.get('acc_target_rms', 0.0)
                    tremor_gyro_rms = meta.get('gyro_target_rms', 0.0)
                    tremor_score = int(
                        meta.get(
                            'sampled_score',
                            meta.get('score', pk_config.get_tremor_score(tremor_acc_rms))
                        )
                    )
                else:
                    tremor_freq = 0.0
                    tremor_acc_rms = 0.0
                    tremor_gyro_rms = 0.0
                    tremor_score = 0

            for a in range(AUG_SIZE):
                X_list.append(win.T.astype(np.float32))  # (C, L) — pure zeros, no augmentation noise
                y_list.append(label - 1)                      # 0..11
                subj_list.append(subj_idx)
                base_idx_list.append(w_idx)

                tremor_freq_list.append(tremor_freq)
                tremor_acc_rms_list.append(tremor_acc_rms)
                tremor_gyro_rms_list.append(tremor_gyro_rms)
                tremor_score_list.append(tremor_score)

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

        all_labels = np.column_stack([
            (y + 1),
            subject_ids,
            base_window_idx,
            tremor_freq,
            tremor_acc_rms,
            tremor_gyro_rms,
            tremor_score
        ])

        sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"    ✓ Saved: {npz_out.name}, {txt_out.name} | X={X.shape}, tremor_labels={tremor_freq.shape}")

    # -------------------------------
    # Step 3.5: Generate Tremor-Branch Datasets (Acc_arm, Gyro_arm)
    # -------------------------------
    print("\n" + "=" * 80)
    print("STEP 3.5: Generating Tremor-Branch Datasets (for tremor severity estimation)...")
    print("=" * 80)
    print("Note: Tremor-branch datasets preserve amplitude information (no z-score normalization)")
    print("      Full dropout is applied; all amplitude information is therefore zero.")
    print("")

    TREMOR_BRANCH_SENSORS = {
        "Acc_arm": SENSORS["Acc_arm"],
        "Gyro_arm": SENSORS["Gyro_arm"],
    }

    aug_rng_tremor = np.random.default_rng(SEED)

    for sensor_name, cols in TREMOR_BRANCH_SENSORS.items():
        print(f"\n  Processing tremor-branch: {sensor_name}")

        X_tremor_list = []
        y_tremor_list = []
        subj_tremor_list = []
        base_idx_tremor_list = []

        tremor_freq_tremor_list = []
        tremor_acc_rms_tremor_list = []
        tremor_gyro_rms_tremor_list = []
        tremor_score_tremor_list = []

        sensor_type = get_sensor_type(sensor_name)
        signal_body_part = extract_body_part(sensor_name)
        severity_body_part = signal_body_part

        subject_data_cache = {}
        for subj_idx in range(1, NUM_SUBJECTS + 1):
            fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
            if fname.exists():
                subject_data_cache[subj_idx] = load_subject_data_with_retry(fname)

        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subject_data_cache:
                continue

            data = subject_data_cache[subj_idx]

            win_raw = data[win_start:win_end, cols].copy()

            severity_cache_entry = tremor_cache.get((w_idx, severity_body_part))
            severity_meta = severity_cache_entry.get('meta', {}) if severity_cache_entry else {}
            tremor_acc_rms_target = severity_meta.get('acc_target_rms', 0.0)
            if GENERATE_TREMOR:
                tremor_score_val = int(
                    severity_meta.get(
                        'sampled_score',
                        severity_meta.get('score', pk_config.get_tremor_score(tremor_acc_rms_target))
                    )
                )
            else:
                tremor_score_val = 0

            signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
            if signal_cache_entry is not None:
                if sensor_type in ['acc', 'gyro']:
                    noise_key = f'{sensor_type}_noise'
                    tremor_noise = signal_cache_entry[noise_key]
                    win_raw += tremor_noise

            # Resample
            win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)

            # Apply full dropout (same as HAR branch)
            win_tremor = apply_full_dropout(win_resampled)
            # CRITICAL DIFFERENCE: No z-score normalization (preserves amplitude)
            # All values are already zero due to full dropout

            signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
            if signal_cache_entry is not None and GENERATE_TREMOR:
                meta = signal_cache_entry['meta']
                tremor_freq = meta.get('freq_hz', 0.0)
                tremor_acc_rms = meta.get('acc_target_rms', 0.0)
                tremor_gyro_rms = meta.get('gyro_target_rms', 0.0)
                tremor_score = int(
                    meta.get(
                        'sampled_score',
                        meta.get('score', pk_config.get_tremor_score(tremor_acc_rms))
                    )
                )
            else:
                tremor_freq = 0.0
                tremor_acc_rms = 0.0
                tremor_gyro_rms = 0.0
                tremor_score = 0

            X_tremor_list.append(win_tremor.T.astype(np.float32))
            y_tremor_list.append(label - 1)
            subj_tremor_list.append(subj_idx)
            base_idx_tremor_list.append(w_idx)

            tremor_freq_tremor_list.append(tremor_freq)
            tremor_acc_rms_tremor_list.append(tremor_acc_rms)
            tremor_gyro_rms_tremor_list.append(tremor_gyro_rms)
            tremor_score_tremor_list.append(tremor_score)

        X_tremor = np.stack(X_tremor_list, axis=0)
        y_tremor = np.array(y_tremor_list, dtype=np.int64)
        subject_ids_tremor = np.array(subj_tremor_list, dtype=np.int64)
        base_window_idx_tremor = np.array(base_idx_tremor_list, dtype=np.int64)

        tremor_freq_tremor = np.array(tremor_freq_tremor_list, dtype=np.float32)
        tremor_acc_rms_tremor = np.array(tremor_acc_rms_tremor_list, dtype=np.float32)
        tremor_gyro_rms_tremor = np.array(tremor_gyro_rms_tremor_list, dtype=np.float32)
        tremor_score_tremor = np.array(tremor_score_tremor_list, dtype=np.int8)

        save_tremor_branch_sensor(
            variant_dir=variant_dir,
            sensor_name=sensor_name,
            X=X_tremor,
            y=y_tremor,
            subject_ids=subject_ids_tremor,
            base_window_idx=base_window_idx_tremor,
            tremor_freq=tremor_freq_tremor,
            tremor_acc_rms=tremor_acc_rms_tremor,
            tremor_gyro_rms=tremor_gyro_rms_tremor,
            tremor_score=tremor_score_tremor,
            sensor_cols=cols,
            fs=FS,
            window_len=target_window_len,
            stride=int(round(STRIDE_SEC * FS))
        )

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
        "Corruption: FULL SIGNAL DROPOUT",
        "=" * 80,
        "  All sensor samples are replaced with zero: y(t) = 0.",
        "  Applied after resampling, before per-window z-score normalization.",
        "  Models complete sensor failure (hardware failure, battery depletion,",
        "  or complete communication loss).",
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
            "=" * 80,
            "Augmentation Strategy (tremor-free sensors only):",
            "=" * 80,
            f"Tremor-free sensors (policy: {pk_config.TREMOR_FREE_SENSORS}):",
            "  - Score 0 (Clean): Original signal (no augmentation)",
            "  - Score 1-2 (Mild): AWGN (RMS ratio 15%)",
            "  - Score 3-4 (Severe): Rotation matrix (3-axis) or AWGN (2-axis)",
            "  NOTE: All augmentation is applied BEFORE full dropout, so the",
            "        final signal is zero regardless of augmentation mode.",
            "",
            f"Rotation-based magnetometer sensors (policy: {pk_config.ROTATION_BASED_MAG_SENSORS}):",
            "  - Magnetometer tremor modeled via gyro-driven cumulative rotations.",
            "  NOTE: Rotation is applied BEFORE full dropout.",
            "",
            "Note: Tremor policies are defined in tremor_parkinson_config.py",
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
        "    - Sensor data (flattened C×L channels):",
        f"      * 3-axis sensors (Acc, Gyro, Mag): Columns 1-{target_window_len * 3} (C=3, L={target_window_len})",
        f"      * ECG sensor: Columns 1-{target_window_len * 2} (C=2, L={target_window_len})",
        "    - Label columns (always last 7 columns):",
        "      * Activity label (1-12)",
        "      * Subject ID (1-10)",
        "      * Base window index",
        "      * Tremor frequency (Hz)",
        "      * Tremor acc RMS (m/s²)",
        "      * Tremor gyro RMS (deg/s)",
        "      * Tremor severity score (0-4)",
        "",
        "=" * 80,
    ])

    with open(info_txt, 'w') as f:
        f.write('\n'.join(info_lines))

    print(f"  ✓ Saved: {info_txt.name}")

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
    fs_str = f"fs{int(FS)}"
    base_name = f"s{int(STRIDE_SEC)}_w{int(WINDOW_SEC)}_{fs_str}_tremor"
    dropout_suffix = "_fullDrop"

    if GENERATE_TREMOR:
        augment_modes = ["clean", "mild_mod", "mod_severe"]
        total_variants = len(augment_modes)

        print("\n" + "="*80)
        print(f"GENERATING {total_variants} TREMOR VARIANTS (full dropout)")
        print("="*80)
        print(f"Base name: {base_name}_[mode]{dropout_suffix}")
        print(f"Variants: {', '.join(augment_modes)}")
        print("="*80 + "\n")

        for idx, mode in enumerate(augment_modes, 1):
            print("\n" + "#"*80)
            print(f"#  VARIANT {idx}/{total_variants}: {mode.upper()}")
            print("#"*80 + "\n")

            variant_name = f"{base_name}_{mode}{dropout_suffix}"
            generate_tremor_dataset(variant_name, augment_mode=mode)

            print("\n" + "#"*80)
            print(f"#  COMPLETED VARIANT {idx}/{total_variants}: {mode.upper()}")
            print("#"*80 + "\n")

        print("\n" + "="*80)
        print(f"ALL {total_variants} TREMOR VARIANTS GENERATED SUCCESSFULLY")
        print("="*80)
        print("Generated variants:")
        for mode in augment_modes:
            print(f"  {base_name}_{mode}{dropout_suffix}")
        print("="*80 + "\n")

    else:
        variant_name = f"{base_name}_clean{dropout_suffix}"
        generate_tremor_dataset(variant_name, augment_mode="clean")
