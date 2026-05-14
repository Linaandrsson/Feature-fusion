"""
Data Generator for Corrupted Datasets
======================================

Generates test-condition datasets with hardware / signal corruption applied to
clean (tremor-free) source signals.  Each corruption type is written to its own
folder so models trained on any condition can be evaluated on each type.

Corruption variants generated
------------------------------
  s4_w4_fs50_corrupt_awgn         : Additive White Gaussian Noise (SNR ~6 dB)
  s4_w4_fs50_corrupt_dropout      : Complete sensor dropout (all samples → 0)
  s4_w4_fs50_corrupt_weak_signal  : Signal attenuation to 20 % of original amplitude
  s4_w4_fs50_corrupt_timeshift    : Signal delayed by TIME_SHIFT_SEC (1 s) —
                                    windows where the shift exceeds the segment
                                    boundary are dropped
  s4_w4_fs50_corrupt_rotation     : Fixed 3-D rotation (CORRUPT_ROTATION_DEG=45 °)
                                    applied to all 3-axis sensors; ECG gets AWGN
                                    as a physically-consistent fallback

Output base
-----------
  /data/Tremor_datagenerator_files/s4_w4_fs50_corrupt_{type}/

NOTE ON ROTATION vs AUGMENTATION
  DataGenerator_CleanAug.py uses 15 ° rotation as *training augmentation*.
  Here, 45 ° represents a poorly-attached / mis-oriented sensor during *test*.
  A single fixed rotation matrix (determined by CORRUPT_ROTATION_SEED) is used
  for all windows so the corruption models one consistent sensor misplacement.

NOTE ON TIMESHIFT
  Windows whose shifted boundary would fall outside the activity segment are
  silently dropped.  The resulting variant has slightly fewer windows than the
  others.  Subject-based splits are generated consistently regardless.
"""

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).parent.parent.resolve()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np
import random
from scipy.signal import resample
import json
import time
from typing import List, Tuple

# ============================================================
# CONFIG — Basic parameters
# ============================================================
ORIGINAL_FS = 50        # Hz — sampling rate of the MHEALTH log files
FS          = 50        # Hz — target output rate (same → no resampling needed)
WINDOW_SEC  = 4.0       # window length in seconds
STRIDE_SEC  = 4.0       # stride in seconds (non-overlapping)
AUG_SIZE    = 1         # no training augmentation — corruption is the only transform
NUM_SUBJECTS = 10

# ---- Paths ----
script_dir = Path(__file__).parent.resolve()
_code_dir  = script_dir.parent          # …/Code/
DATA_PATH  = _code_dir / "data" / "MHEALTHDATASET"
OUT_BASE   = _code_dir / "data" / "Tremor_datagenerator_files"

# ---- Reproducibility ----
SEED                  = 0    # controls any per-window randomness (AWGN, rotation)
CORRUPT_ROTATION_SEED = 7    # fixed seed for the single rotation matrix

# ============================================================
# CONFIG — Corruption parameters
# ============================================================

# AWGN: noticeable but not destructive
# Noise RMS = CORRUPT_AWGN_RMS_RATIO * per-channel signal RMS
# Examples: 0.316 ≈ 10 dB, 0.1 ≈ 20 dB, 0.032 ≈ 30 dB, 0.01 ≈ 40 dB
CORRUPT_AWGN_RMS_RATIO = 0.0   # ~20 dB SNR

# Dropout: complete sensor failure
CORRUPT_DROPOUT_VALUE = 0.3

# Weak signal: attenuated but not almost removed
CORRUPT_WEAK_SIGNAL_FACTOR = 0.50   # signal kept at 50% amplitude

# Time-shift: realistic synchronization error
CORRUPT_TIME_SHIFT_SEC = 0.25       # 250 ms = 12–13 samples at 50 Hz

# Rotation: realistic sensor misalignment
CORRUPT_ROTATION_DEG = 20.0         # moderate misplacement

# ECG should not be rotated; use mild noise fallback only if needed
CORRUPT_ROTATION_ECG_RMS_RATIO = 0.15

CORRUPT_PARTIAL_DROPOUT_RATIO = 0.10  # 10 % of window length

# ============================================================
# Subject-based train / val / test splits — must match CNN training
# ============================================================
TEST_SUBJECTS  = {5, 10}
VAL_SUBJECTS   = {2, 7}
TRAIN_SUBJECTS = {1, 3, 4, 6, 8, 9}

# ============================================================
# SENSOR COLUMN MAP (0-indexed, from mHealth_subject*.log)
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

LABEL_COL = 23   # activity label column (values 0–12; we use 1–12)


# ============================================================
# Helpers
# ============================================================

def resample_window(win: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample (L, C) window from original_fs to target_fs."""
    if original_fs == target_fs:
        return win
    L_original, C = win.shape
    L_target = int(round(L_original * target_fs / original_fs))
    resampled = np.zeros((L_target, C))
    for c in range(C):
        resampled[:, c] = resample(win[:, c], L_target)
    return resampled


def zscore_window(win: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation of an (L, C) window."""
    mean = win.mean(axis=0, keepdims=True)
    std  = win.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    """(N, C, L) → (N, C×L) in channel-block order."""
    N, C, L = X_c_l.shape
    return np.concatenate([X_c_l[:, c, :] for c in range(C)], axis=1)


def load_subject_data_with_retry(file_path: Path, max_retries: int = 5, delay: float = 2.0) -> np.ndarray:
    """Load a subject log file with retry logic (NFS / OneDrive tolerance)."""
    for attempt in range(max_retries):
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1 or (data.ndim > 1 and data.shape[1] < 24):
                print(f"WARNING: {file_path.name} appears incomplete (OneDrive sync?).")
            return data
        except (TimeoutError, OSError):
            if attempt < max_retries - 1:
                print(f"  Timeout loading {file_path.name}, retry {attempt+1}/{max_retries} in {delay}s ...")
                time.sleep(delay)
            else:
                print(f"  Failed to load {file_path.name} after {max_retries} attempts.")
                raise


# ============================================================
# Corruption functions (operate on RAW signal BEFORE resampling / z-score)
# ============================================================

def corrupt_awgn(win_raw: np.ndarray, rng: np.random.Generator,
                 rms_ratio: float = CORRUPT_AWGN_RMS_RATIO) -> np.ndarray:
    """Add Gaussian noise with RMS = rms_ratio x per-channel signal RMS."""
    L, C = win_raw.shape
    out = win_raw.copy()
    for c in range(C):
        sig_rms = np.sqrt(np.mean(win_raw[:, c] ** 2))
        if sig_rms < 1e-8:
            sig_rms = 1.0
        out[:, c] += rng.normal(0, rms_ratio * sig_rms, size=L)
    return out


def corrupt_dropout(win_raw: np.ndarray,
                    value: float = CORRUPT_DROPOUT_VALUE) -> np.ndarray:
    """Complete sensor dropout -- all samples set to a constant value."""
    return np.full_like(win_raw, fill_value=value)


def corrupt_weak_signal(win_raw: np.ndarray,
                        factor: float = CORRUPT_WEAK_SIGNAL_FACTOR) -> np.ndarray:
    """Attenuate signal amplitude by multiplying by factor (< 1)."""
    return win_raw * factor


def _make_fixed_rotation_matrix(seed: int, angle_deg: float) -> np.ndarray:
    """
    Build a single fixed 3x3 rotation matrix from a seeded RNG.
    The same matrix is reused for every window in the rotation variant,
    modelling one consistent sensor mis-attachment angle.
    """
    rng = np.random.default_rng(seed)
    max_rad = np.deg2rad(angle_deg)
    theta_x, theta_y, theta_z = rng.uniform(-max_rad, max_rad, size=3)

    Rx = np.array([
        [1, 0,               0             ],
        [0,  np.cos(theta_x), -np.sin(theta_x)],
        [0,  np.sin(theta_x),  np.cos(theta_x)],
    ])
    Ry = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [ 0,               1, 0              ],
        [-np.sin(theta_y), 0, np.cos(theta_y)],
    ])
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0,                0,               1],
    ])
    return Rz @ Ry @ Rx   # combined rotation: Rz . Ry . Rx


def corrupt_partial_dropout(win: np.ndarray, rng: np.random.Generator,
                            ratio: float = CORRUPT_PARTIAL_DROPOUT_RATIO) -> np.ndarray:
    """
    Zero out a single contiguous temporal segment of `ratio` x window length.

    The masked segment length is round(ratio * L) samples.  Its start position
    is drawn uniformly at random so the entire segment fits within [0, L).
    Applied AFTER z-score so the masked region is exactly zero.
    """
    L = win.shape[0]
    mask_len = max(1, int(round(ratio * L)))
    max_start = max(0, L - mask_len)
    start = int(rng.integers(0, max_start + 1))
    out = win.copy()
    out[start:start + mask_len, :] = 0.0
    return out


def corrupt_rotation(win_raw: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a fixed rotation matrix R to a 3-axis (L, 3) window."""
    return win_raw @ R.T   # (L, 3) @ (3, 3)^T = (L, 3)


# ============================================================
# Window-spec generation
# ============================================================

def build_window_specs(corruption_mode: str) -> List[Tuple[int, int, int, int, int]]:
    """
    Walk all subjects and collect (subj_idx, label, win_start, win_end, clip_end) tuples.

    Uses the same global stride walk as DataGenerator_Tremor.py / DataGenerator_CleanAug.py:
    windows at i = 0, stride, 2*stride, ... where ALL labels[i:i+window_len] >= 1.
    This ensures corrupt variants drop exactly the same windows as the tremor variants.

    For 'timeshift': win_start/win_end are shifted forward by CORRUPT_TIME_SHIFT_SEC.
    Windows where the shifted end falls outside the data bounds or any sample in the
    shifted range has label 0 are dropped (not zero-padded).

    For all other modes: clip_end == win_end always.
    """
    window_len = int(round(ORIGINAL_FS * WINDOW_SEC))
    stride     = int(round(ORIGINAL_FS * STRIDE_SEC))
    stride     = max(stride, 1)

    shift_samples = 0
    if corruption_mode == "timeshift":
        shift_samples = int(round(ORIGINAL_FS * CORRUPT_TIME_SHIFT_SEC))

    specs = []
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        data   = load_subject_data_with_retry(file_path)
        labels = data[:, LABEL_COL].astype(int)
        n      = len(labels)

        i = 0
        while i + window_len <= n:
            lab_win = labels[i:i + window_len]
            if np.all(lab_win >= 1):
                lab_val   = int(lab_win[0])
                win_start = i + shift_samples
                win_end   = win_start + window_len
                if shift_samples > 0:
                    # Drop if shifted window goes out of data or into null-label territory
                    if win_end > n or np.any(labels[win_start:win_end] < 1):
                        i += stride
                        continue
                specs.append((subj_idx, lab_val, win_start, win_end, win_end))
            i += stride

    return specs


# ============================================================
# Split generation
# ============================================================

def write_splits(variant_dir: Path, window_specs: List[Tuple]) -> None:
    """
    Write train_idx.txt / val_idx.txt / test_idx.txt into variant_dir/splits/.
    Each file contains the (0-based) sample indices for the corresponding split.
    With AUG_SIZE=1 the sample index equals the window index.
    """
    splits_dir = variant_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = [], [], []
    for i, (subj_idx, *_) in enumerate(window_specs):
        if subj_idx in TEST_SUBJECTS:
            test_idx.append(i)
        elif subj_idx in VAL_SUBJECTS:
            val_idx.append(i)
        else:
            train_idx.append(i)

    for name, indices in [("train_idx", train_idx),
                          ("val_idx",   val_idx),
                          ("test_idx",  test_idx)]:
        (splits_dir / f"{name}.txt").write_text(
            "\n".join(str(i) for i in indices) + "\n"
        )

    print(f"  Splits: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")


# ============================================================
# Core dataset generation
# ============================================================

def generate_corrupt_dataset(variant_name: str, corruption_mode: str) -> None:
    """
    Generate a corruption dataset variant and write it to OUT_BASE / variant_name.

    Parameters
    ----------
    variant_name : str
        Output folder name, e.g. "s4_w4_fs50_corrupt_awgn".
    corruption_mode : str
        One of: "awgn", "dropout", "partial_dropout", "weak_signal", "timeshift", "rotation".
    """
    assert corruption_mode in ("awgn", "dropout", "partial_dropout", "weak_signal", "timeshift", "rotation"), \
        f"Unknown corruption_mode: {corruption_mode!r}"

    print("=" * 80)
    print(f"CORRUPTION DATASET GENERATION: {variant_name}")
    print(f"  Mode : {corruption_mode}")
    print("=" * 80)

    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    window_len_out = int(round(FS * WINDOW_SEC))   # samples in output (after resample)
    stride_out     = int(round(FS * STRIDE_SEC))

    # ------------------------------------------------------------------
    # Step 1 -- build window specifications
    # ------------------------------------------------------------------
    print("Building window specifications ...")
    window_specs = build_window_specs(corruption_mode)
    if not window_specs:
        raise RuntimeError("No windows generated -- check dataset path and parameters.")
    print(f"  Total windows: {len(window_specs)}")

    # ------------------------------------------------------------------
    # Step 1b -- pre-build rotation matrix (rotation mode only)
    # ------------------------------------------------------------------
    R_fixed = None
    if corruption_mode == "rotation":
        R_fixed = _make_fixed_rotation_matrix(CORRUPT_ROTATION_SEED, CORRUPT_ROTATION_DEG)
        print(f"  Fixed rotation matrix (seed={CORRUPT_ROTATION_SEED}, "
              f"angle<={CORRUPT_ROTATION_DEG} deg):")
        print(f"    {R_fixed}")

    # ------------------------------------------------------------------
    # Step 2 -- write subject-based splits
    # ------------------------------------------------------------------
    write_splits(variant_dir, window_specs)

    # ------------------------------------------------------------------
    # Step 3 -- process each sensor
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)   # shared RNG across sensors for AWGN

    for sensor_name, cols in SENSORS.items():
        X_list, y_list, subj_list, base_idx_list = [], [], [], []

        subj_cache: dict = {}

        for w_idx, (subj_idx, label, win_start, win_end, clip_end) in enumerate(window_specs):
            if subj_idx not in subj_cache:
                subj_cache[subj_idx] = load_subject_data_with_retry(
                    DATA_PATH / f"mHealth_subject{subj_idx}.log"
                )
            data = subj_cache[subj_idx]

            # Extract RAW window
            win = data[win_start:win_end, cols].copy()

            # ---- Apply corruption to raw signal (BEFORE resample + z-score) ----
            # weak_signal and partial_dropout are intentionally skipped here —
            # both are applied after z-score (see below).
            if corruption_mode == "awgn":
                win = corrupt_awgn(win, rng)

            elif corruption_mode == "dropout":
                win = corrupt_dropout(win)

            elif corruption_mode == "timeshift":
                pass  # window already extracted at the shifted position

            elif corruption_mode == "rotation":
                if len(cols) == 3:
                    # 3-axis sensor: apply fixed rotation matrix
                    win = corrupt_rotation(win, R_fixed)
                else:
                    # 2-axis sensor (ECG): rotation undefined -- apply AWGN instead
                    win = corrupt_awgn(win, rng, rms_ratio=CORRUPT_ROTATION_ECG_RMS_RATIO)

            # ---- Preprocessing ----
            win = resample_window(win, ORIGINAL_FS, FS)
            win = zscore_window(win)

            # ---- Apply corruption AFTER z-score ----
            # weak_signal must come here: z-score is scale-invariant so attenuating
            # before it has zero effect.  After z-score each channel has unit variance;
            # multiplying by CORRUPT_WEAK_SIGNAL_FACTOR reduces it to factor^2 variance,
            # which the CNN will observe as a genuinely weaker signal.
            # partial_dropout must come here so the masked segment is exactly zero
            # (applying before z-score would shift the masked region away from zero).
            if corruption_mode == "weak_signal":
                win = corrupt_weak_signal(win)

            elif corruption_mode == "partial_dropout":
                win = corrupt_partial_dropout(win, rng)

            X_list.append(win.T.astype(np.float32))   # (C, L)
            y_list.append(label - 1)                   # 0-based
            subj_list.append(subj_idx)
            base_idx_list.append(w_idx)

        X            = np.stack(X_list,    axis=0)              # (N, C, L)
        y            = np.array(y_list,    dtype=np.int64)
        subject_ids  = np.array(subj_list, dtype=np.int64)
        base_win_idx = np.array(base_idx_list, dtype=np.int64)
        N            = len(y)
        zeros_f32    = np.zeros(N, dtype=np.float32)

        # ---- Save NPZ ----
        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X=X,
            y=y,
            subject_id=subject_ids,
            base_window_idx=base_win_idx,
            tremor_freq=zeros_f32,
            tremor_acc_rms=zeros_f32,
            tremor_gyro_rms=zeros_f32,
            tremor_score=zeros_f32,
            fs=FS,
            window_len=window_len_out,
            stride=stride_out,
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )

        # ---- Save TXT (CSV, tremor-compatible column layout) ----
        txt_out   = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)                        # (N, C*L)
        zeros_col = np.zeros((N, 1), dtype=np.float32)
        sensor_txt = np.hstack([
            flat_rows,
            (y + 1).reshape(-1, 1),                                  # col -7: activity 1-based
            subject_ids.reshape(-1, 1),                               # col -6: subject id
            base_win_idx.reshape(-1, 1),                              # col -5: base window idx
            zeros_col,  zeros_col,  zeros_col,  zeros_col,            # cols -4..-1: tremor fields
        ]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"  [{sensor_name}] X={X.shape}  saved {npz_out.name}, {txt_out.name}")

    # ------------------------------------------------------------------
    # Step 4 -- write info.txt
    # ------------------------------------------------------------------
    _write_info(variant_dir, corruption_mode, len(window_specs),
                window_len_out, stride_out)

    print(f"\n  DONE: {variant_name}")
    print("=" * 80 + "\n")


# ============================================================
# Info file
# ============================================================

def _write_info(variant_dir: Path, corruption_mode: str,
                n_windows: int, window_len: int, stride: int) -> None:
    lines = [
        "=" * 60,
        "CORRUPTION DATASET CONFIGURATION",
        "=" * 60,
        "",
        "Basic parameters:",
        f"  Original FS  : {ORIGINAL_FS} Hz",
        f"  Target FS    : {FS} Hz",
        f"  Window       : {WINDOW_SEC} s  ({window_len} samples)",
        f"  Stride       : {STRIDE_SEC} s  ({stride} samples)",
        f"  AUG_SIZE     : {AUG_SIZE}  (no training augmentation)",
        f"  Seed         : {SEED}",
        "",
        "Corruption configuration:",
        f"  Mode         : {corruption_mode}",
    ]

    if corruption_mode == "awgn":
        lines += [f"  RMS ratio    : {CORRUPT_AWGN_RMS_RATIO}  "
                  f"(noise RMS = {CORRUPT_AWGN_RMS_RATIO*100:.0f}% of per-channel signal RMS)",
                  f"  Applied      : BEFORE resampling and z-score"]
    elif corruption_mode == "dropout":
        lines += [f"  Fill value   : {CORRUPT_DROPOUT_VALUE}  (all samples replaced)",
                  f"  Applied      : BEFORE resampling and z-score"]
    elif corruption_mode == "weak_signal":
        lines += [f"  Factor       : {CORRUPT_WEAK_SIGNAL_FACTOR}  "
                  f"(z-scored signal multiplied by factor -> variance = factor^2)",
                  f"  Applied      : AFTER z-score (scale-invariant; pre-z-score would have no effect)"]
    elif corruption_mode == "timeshift":
        lines += [
            f"  Shift        : {CORRUPT_TIME_SHIFT_SEC} s  "
            f"({int(round(ORIGINAL_FS*CORRUPT_TIME_SHIFT_SEC))} samples at {ORIGINAL_FS} Hz)",
            f"  Note         : windows where the shifted end falls outside data or null-label territory are dropped",
        ]
    elif corruption_mode == "partial_dropout":
        mask_len = max(1, int(round(CORRUPT_PARTIAL_DROPOUT_RATIO * int(round(FS * WINDOW_SEC)))))
        lines += [
            f"  Ratio        : {CORRUPT_PARTIAL_DROPOUT_RATIO}  "
            f"({mask_len} contiguous samples zeroed, start position random)",
            f"  Applied      : AFTER z-score (masked segment is exactly zero)",
        ]
    elif corruption_mode == "rotation":
        lines += [
            f"  Angle        : <= {CORRUPT_ROTATION_DEG} deg  (fixed matrix, all windows)",
            f"  Seed         : {CORRUPT_ROTATION_SEED}",
            f"  ECG fallback : AWGN (rms_ratio={CORRUPT_ROTATION_ECG_RMS_RATIO})",
        ]

    lines += [
        "",
        f"  Total windows: {n_windows}",
        "",
        "Subject splits:",
        f"  Train subjects : {sorted(TRAIN_SUBJECTS)}",
        f"  Val subjects   : {sorted(VAL_SUBJECTS)}",
        f"  Test subjects  : {sorted(TEST_SUBJECTS)}",
        "",
        "File format (NPZ):",
        "  X              : (N, C, L)  z-scored sensor data",
        "  y              : (N,)       activity labels  0-based (0-11)",
        "  subject_id     : (N,)       subject IDs 1-10",
        "  base_window_idx: (N,)       window index in this variant",
        "  tremor_*       : (N,)       all zeros (no tremor in source signal)",
        "",
        "File format (TXT / CSV):",
        "  columns: [sensor_data C*L | activity | subject | base_win_idx |",
        "            tremor_freq | tremor_acc_rms | tremor_gyro_rms | tremor_score]",
        "",
        "Corruption applied BEFORE resampling and z-score normalisation.",
        "=" * 60,
    ]

    (variant_dir / "info.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote info.txt")


# ============================================================
# Combination corruption dataset generation
# ============================================================

def generate_corrupt_dataset_combo(variant_name: str, corruption_modes: list) -> None:
    """
    Generate a corruption dataset that applies multiple corruptions simultaneously.

    Corruption order:
      1. timeshift  — handled at window-extraction level (shifted win_start/win_end)
      2. rotation   — applied pre-z-score to 3-axis sensors (ECG → AWGN fallback)
      3. awgn       — applied pre-z-score
      4. dropout    — applied pre-z-score
      5. weak_signal— applied POST-z-score (scale-invariant; must come after)

    Parameters
    ----------
    variant_name : str
        Output folder name, e.g. "s4_w4_fs50_corrupt_awgn_weak".
    corruption_modes : list of str
        Any subset of: "awgn", "dropout", "partial_dropout", "weak_signal", "timeshift", "rotation".
    """
    modes = set(corruption_modes)
    valid = {"awgn", "dropout", "partial_dropout", "weak_signal", "timeshift", "rotation"}
    assert modes.issubset(valid), f"Unknown modes: {modes - valid}"
    assert len(modes) > 1, "Use generate_corrupt_dataset() for single-mode variants."

    print("=" * 80)
    print(f"COMBO CORRUPTION DATASET: {variant_name}")
    print(f"  Modes : {sorted(modes)}")
    print("=" * 80)

    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    window_len_out = int(round(FS * WINDOW_SEC))
    stride_out     = int(round(FS * STRIDE_SEC))

    # Step 1 -- window specs (apply timeshift at extraction if needed)
    print("Building window specifications ...")
    window_specs = build_window_specs("timeshift" if "timeshift" in modes else "awgn")
    if not window_specs:
        raise RuntimeError("No windows generated.")
    print(f"  Total windows: {len(window_specs)}")

    # Step 1b -- rotation matrix
    R_fixed = None
    if "rotation" in modes:
        R_fixed = _make_fixed_rotation_matrix(CORRUPT_ROTATION_SEED, CORRUPT_ROTATION_DEG)
        print(f"  Fixed rotation matrix (seed={CORRUPT_ROTATION_SEED}, angle<={CORRUPT_ROTATION_DEG} deg)")

    # Step 2 -- splits
    write_splits(variant_dir, window_specs)

    # Step 3 -- process sensors
    rng = np.random.default_rng(SEED)

    for sensor_name, cols in SENSORS.items():
        X_list, y_list, subj_list, base_idx_list = [], [], [], []
        subj_cache: dict = {}

        for w_idx, (subj_idx, label, win_start, win_end, clip_end) in enumerate(window_specs):
            if subj_idx not in subj_cache:
                subj_cache[subj_idx] = load_subject_data_with_retry(
                    DATA_PATH / f"mHealth_subject{subj_idx}.log"
                )
            win = subj_cache[subj_idx][win_start:win_end, cols].copy()

            # --- Pre-z-score corruptions (order: rotation -> awgn -> dropout) ---
            # Track whether this sensor has already received AWGN via rotation fallback
            # so we don't apply it a second time in the awgn block.
            ecg_awgn_applied = False
            if "rotation" in modes:
                if len(cols) == 3:
                    win = corrupt_rotation(win, R_fixed)
                else:
                    # ECG: rotation undefined, use AWGN fallback
                    win = corrupt_awgn(win, rng, rms_ratio=CORRUPT_ROTATION_ECG_RMS_RATIO)
                    ecg_awgn_applied = True

            if "awgn" in modes and not ecg_awgn_applied:
                win = corrupt_awgn(win, rng)

            if "dropout" in modes:
                win = corrupt_dropout(win)

            # timeshift: window already extracted at shifted position — nothing to do here

            # --- Preprocessing ---
            win = resample_window(win, ORIGINAL_FS, FS)
            win = zscore_window(win)

            # --- Post-z-score corruptions ---
            if "weak_signal" in modes:
                win = corrupt_weak_signal(win)

            if "partial_dropout" in modes:
                win = corrupt_partial_dropout(win, rng)

            X_list.append(win.T.astype(np.float32))
            y_list.append(label - 1)
            subj_list.append(subj_idx)
            base_idx_list.append(w_idx)

        X            = np.stack(X_list,    axis=0)
        y            = np.array(y_list,    dtype=np.int64)
        subject_ids  = np.array(subj_list, dtype=np.int64)
        base_win_idx = np.array(base_idx_list, dtype=np.int64)
        N            = len(y)
        zeros_f32    = np.zeros(N, dtype=np.float32)

        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out, X=X, y=y, subject_id=subject_ids,
            base_window_idx=base_win_idx,
            tremor_freq=zeros_f32, tremor_acc_rms=zeros_f32,
            tremor_gyro_rms=zeros_f32, tremor_score=zeros_f32,
            fs=FS, window_len=window_len_out, stride=stride_out,
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )

        txt_out   = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)
        zeros_col = np.zeros((N, 1), dtype=np.float32)
        sensor_txt = np.hstack([
            flat_rows,
            (y + 1).reshape(-1, 1), subject_ids.reshape(-1, 1),
            base_win_idx.reshape(-1, 1),
            zeros_col, zeros_col, zeros_col, zeros_col,
        ]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"  [{sensor_name}] X={X.shape}  saved {npz_out.name}, {txt_out.name}")

    # Step 4 -- info.txt
    _write_info_combo(variant_dir, sorted(modes), len(window_specs),
                      window_len_out, stride_out)

    print(f"\n  DONE: {variant_name}")
    print("=" * 80 + "\n")


def _write_info_combo(variant_dir: Path, modes: list,
                      n_windows: int, window_len: int, stride: int) -> None:
    mode_descriptions = {
        "awgn":            f"AWGN (rms_ratio={CORRUPT_AWGN_RMS_RATIO}, pre-z-score)",
        "dropout":         f"Dropout (fill={CORRUPT_DROPOUT_VALUE}, pre-z-score)",
        "partial_dropout": f"Partial dropout (ratio={CORRUPT_PARTIAL_DROPOUT_RATIO}, "
                           f"contiguous segment, POST-z-score)",
        "weak_signal":     f"Weak signal (factor={CORRUPT_WEAK_SIGNAL_FACTOR}, POST-z-score)",
        "timeshift":       f"Time-shift ({CORRUPT_TIME_SHIFT_SEC} s = "
                           f"{int(round(ORIGINAL_FS*CORRUPT_TIME_SHIFT_SEC))} samples, window extraction)",
        "rotation":        f"Rotation (<={CORRUPT_ROTATION_DEG} deg fixed matrix, seed={CORRUPT_ROTATION_SEED}, "
                           f"pre-z-score; ECG->AWGN)",
    }
    lines = [
        "=" * 60,
        "COMBO CORRUPTION DATASET CONFIGURATION",
        "=" * 60,
        "",
        "Basic parameters:",
        f"  Original FS  : {ORIGINAL_FS} Hz",
        f"  Target FS    : {FS} Hz",
        f"  Window       : {WINDOW_SEC} s  ({window_len} samples)",
        f"  Stride       : {STRIDE_SEC} s  ({stride} samples)",
        f"  AUG_SIZE     : {AUG_SIZE}  (no training augmentation)",
        f"  Seed         : {SEED}",
        "",
        "Corruption modes applied (in order):",
    ]
    for m in ["timeshift", "rotation", "awgn", "dropout", "partial_dropout", "weak_signal"]:
        if m in modes:
            lines.append(f"  + {mode_descriptions[m]}")
    lines += [
        "",
        f"  Total windows: {n_windows}",
        "",
        "Subject splits:",
        f"  Train subjects : {sorted(TRAIN_SUBJECTS)}",
        f"  Val subjects   : {sorted(VAL_SUBJECTS)}",
        f"  Test subjects  : {sorted(TEST_SUBJECTS)}",
        "=" * 60,
    ]
    (variant_dir / "info.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote info.txt")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    fs_str    = f"fs{int(FS)}"
    base_name = f"s{int(STRIDE_SEC)}_w{int(WINDOW_SEC)}_{fs_str}_corrupt"
    _snr_db   = round(-20 * np.log10(CORRUPT_AWGN_RMS_RATIO))   # derived from ratio
    snr_tag   = f"{_snr_db}db"   # e.g. "20db"

    # ---- Single-mode variants ----
    single_variants = [
        ("awgn",            f"awgn_{snr_tag}",  "AWGN -- additive electronic noise"),
        ("dropout",         "dropout",          "Dropout -- complete sensor failure (zeros)"),
        ("partial_dropout", "partial_dropout",  "Partial dropout -- random contiguous 10 % segment zeroed"),
        ("weak_signal",     "weak_signal",      "Weak signal -- attenuated amplitude"),
        ("timeshift",       "timeshift",        "Time-shift -- signal delayed"),
        ("rotation",        "rotation",         "Rotation -- sensor axis misalignment"),
    ]

    # ---- Combination variants ----
    # Format: (folder_suffix, [modes], description)
    combo_variants = [
        (f"all_no_dropout",
         ["awgn", "weak_signal", "timeshift", "rotation"],
         "All corruptions except dropout (AWGN + weak signal + time-shift + rotation)"),
        (f"awgn_{snr_tag}_weak",
         ["awgn", "weak_signal"],
         "AWGN + weak signal (noisy AND attenuated)"),
        (f"awgn_{snr_tag}_timeshift",
         ["awgn", "timeshift"],
         "AWGN + time-shift (noisy AND delayed)"),
    ]

    total = len(single_variants) + len(combo_variants)
    print("\n" + "=" * 80)
    print(f"GENERATING {total} CORRUPTION VARIANTS ({len(single_variants)} single + {len(combo_variants)} combo)")
    print("=" * 80)
    print(f"Base prefix : {base_name}_[type]")
    print(f"Output dir  : {OUT_BASE}")
    print("=" * 80 + "\n")

    run_idx = 0
    for mode, suffix, description in single_variants:
        run_idx += 1
        print(f"\n{'#'*80}")
        print(f"#  VARIANT {run_idx}/{total}: {description}")
        print(f"{'#'*80}\n")

        variant_name = f"{base_name}_{suffix}"
        generate_corrupt_dataset(variant_name, corruption_mode=mode)

        print(f"\n{'#'*80}")
        print(f"#  COMPLETED {run_idx}/{total}: {variant_name}")
        print(f"{'#'*80}\n")

    for suffix, modes, description in combo_variants:
        run_idx += 1
        print(f"\n{'#'*80}")
        print(f"#  VARIANT {run_idx}/{total}: {description}")
        print(f"{'#'*80}\n")

        variant_name = f"{base_name}_{suffix}"
        generate_corrupt_dataset_combo(variant_name, corruption_modes=modes)

        print(f"\n{'#'*80}")
        print(f"#  COMPLETED {run_idx}/{total}: {variant_name}")
        print(f"{'#'*80}\n")

    print("\n" + "=" * 80)
    print(f"ALL {total} CORRUPTION VARIANTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print("Generated variants:")
    for _, suffix, _ in single_variants:
        print(f"  {base_name}_{suffix}")
    for suffix, _, _ in combo_variants:
        print(f"  {base_name}_{suffix}")
    print("=" * 80 + "\n")
