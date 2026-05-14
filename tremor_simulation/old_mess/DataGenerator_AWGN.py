"""
DataGenerator_AWGN.py
=====================

Generates a clean-data dataset with Additive White Gaussian Noise (AWGN)
applied independently to each sensor channel, before resampling and z-score.

Pipeline per window:
    x_raw  →  x_raw + n  →  resample  →  z-score  →  save

Noise model:
    n ~ N(0, σ²)
    σ = ALPHA * std(x_raw_channel)   ← std (AC component), NOT full RMS

Using std instead of RMS removes the DC offset (e.g. gravity on accelerometers
~9.8 m/s²) from the noise reference, so the effective SNR after z-score is
consistent across all sensor types (accelerometer, gyroscope, magnetometer, ECG).

ALPHA controls corruption strength and is embedded in the output folder name.

Output folder:
    data/Tremor_datagenerator_files/s4_w4_fs50_awgn_a{ALPHA_TAG}/

File format is identical to DataGenerator_Tremor.py (clean variant):
    NPZ: X (N,C,L), y, subject_id, base_window_idx,
         tremor_freq, tremor_acc_rms, tremor_gyro_rms, tremor_score (all zeros)
    TXT: flattened sensor data + 7 label columns (activity 1-based, subject,
         base_win_idx, tremor_freq, tremor_acc_rms, tremor_gyro_rms, tremor_score)

Subject-based splits (written to variant_dir/splits/):
    TEST    : subjects [5, 10]
    VAL     : subjects [2, 7]
    TRAIN   : subjects [1, 3, 4, 6, 8, 9]
"""

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).parent.parent.resolve()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np
from scipy.signal import resample
import time
from typing import List, Tuple

# ============================================================
# CONFIG
# ============================================================
ORIGINAL_FS  = 50       # Hz — sampling rate of the MHEALTH log files
FS           = 50       # Hz — target output rate (same → no resampling needed)
WINDOW_SEC   = 4.0      # window length in seconds
STRIDE_SEC   = 4.0      # stride in seconds (non-overlapping)
NUM_SUBJECTS = 10

# ---- AWGN strength ----
# σ = ALPHA * RMS(x_raw_channel)
# σ = ALPHA * std(channel)  — std-based so DC offsets don't inflate noise.
# Approximate SNR (in z-scored domain, which CNN sees):
#   0.316  ≈ 10 dB SNR
#   0.100  ≈ 20 dB SNR
#   0.032  ≈ 30 dB SNR
#   0.010  ≈ 40 dB SNR
ALPHA = 0.0    # no noise → folder: s4_w4_fs50_awgn_a000 (sanity check)

# ---- Reproducibility ----
SEED = 0

# ---- Paths ----
script_dir = Path(__file__).parent.resolve()
_code_dir  = script_dir.parent
DATA_PATH  = _code_dir / "data" / "MHEALTHDATASET"
OUT_BASE   = _code_dir / "data" / "Tremor_datagenerator_files"

# ============================================================
# Subject-based splits — must match CNN training splits
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

LABEL_COL = 23


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


def apply_awgn(win_raw: np.ndarray, rng: np.random.Generator,
               alpha: float = ALPHA) -> np.ndarray:
    """
    Apply AWGN to a raw (L, C) window.

    For each channel c:
        σ_c = alpha * std(win_raw[:, c])
        n_c ~ N(0, σ_c²)
        out[:, c] = win_raw[:, c] + n_c

    std (AC component) is used instead of RMS so that DC offsets (e.g.
    gravity on accelerometers ~9.8 m/s²) do not inflate the noise level.
    After z-score normalisation the DC is removed, so the effective SNR
    matches alpha regardless of sensor type.

    Applied BEFORE resampling and z-score.
    """
    L, C = win_raw.shape
    out = win_raw.copy()
    for c in range(C):
        sig_std = win_raw[:, c].std()
        if sig_std < 1e-8:
            sig_std = 1.0
        out[:, c] += rng.normal(0.0, alpha * sig_std, size=L)
    return out


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    """(N, C, L) → (N, C×L) in channel-block order."""
    N, C, L = X_c_l.shape
    return np.concatenate([X_c_l[:, c, :] for c in range(C)], axis=1)


def load_subject_data_with_retry(file_path: Path,
                                  max_retries: int = 5,
                                  delay: float = 2.0) -> np.ndarray:
    """Load a subject log file with retry logic (NFS / OneDrive tolerance)."""
    for attempt in range(max_retries):
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1 or (data.ndim > 1 and data.shape[1] < 24):
                print(f"WARNING: {file_path.name} appears incomplete (OneDrive sync?).")
            return data
        except (TimeoutError, OSError):
            if attempt < max_retries - 1:
                print(f"  Timeout loading {file_path.name}, "
                      f"retry {attempt+1}/{max_retries} in {delay}s ...")
                time.sleep(delay)
            else:
                print(f"  Failed to load {file_path.name} after {max_retries} attempts.")
                raise


# ============================================================
# Window-spec generation
# ============================================================

def build_window_specs() -> List[Tuple[int, int, int, int]]:
    """
    Walk all subjects and collect (subj_idx, label, win_start, win_end) tuples.

    Uses the same stride walk as DataGenerator_Tremor.py:
    windows at i = 0, stride, 2*stride, ... where ALL labels[i:i+window_len] >= 1.
    """
    window_len = int(round(ORIGINAL_FS * WINDOW_SEC))
    stride     = int(round(ORIGINAL_FS * STRIDE_SEC))
    stride     = max(stride, 1)

    specs = []
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        if not file_path.exists():
            print(f"  [Subject {subj_idx:2d}] File not found, skipping.")
            continue

        data   = load_subject_data_with_retry(file_path)
        labels = data[:, LABEL_COL].astype(int)
        n      = len(labels)
        n_win  = 0

        i = 0
        while i + window_len <= n:
            lab_win = labels[i:i + window_len]
            if np.all(lab_win >= 1):
                specs.append((subj_idx, int(lab_win[0]), i, i + window_len))
                n_win += 1
            i += stride

        print(f"  [Subject {subj_idx:2d}] {n_win:4d} windows")

    return specs


# ============================================================
# Split generation
# ============================================================

def write_splits(variant_dir: Path,
                 window_specs: List[Tuple]) -> None:
    """Write train_idx.txt / val_idx.txt / test_idx.txt into variant_dir/splits/."""
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
# Main generation
# ============================================================

def generate_awgn_dataset(variant_name: str, alpha: float = ALPHA) -> None:
    """
    Generate AWGN-corrupted dataset and write it to OUT_BASE / variant_name.

    Parameters
    ----------
    variant_name : str
        Output folder, e.g. "s4_w4_fs50_awgn_a010".
    alpha : float
        Noise strength: σ = alpha * RMS(x_raw_channel).
    """
    print("=" * 80)
    print(f"AWGN DATASET GENERATION: {variant_name}")
    print(f"  Alpha (σ/RMS ratio) : {alpha}")
    snr_approx = -20 * np.log10(alpha) if alpha > 0 else float("inf")
    print(f"  Approximate SNR     : {snr_approx:.1f} dB")
    print(f"  Seed                : {SEED}")
    print("=" * 80)

    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    window_len_out = int(round(FS * WINDOW_SEC))
    stride_out     = int(round(FS * STRIDE_SEC))

    # ------------------------------------------------------------------
    # Step 1 — build window specs
    # ------------------------------------------------------------------
    print("\nSTEP 1: Collecting window specifications...")
    window_specs = build_window_specs()
    if not window_specs:
        raise RuntimeError("No windows generated — check DATA_PATH and parameters.")
    print(f"\n  Total windows: {len(window_specs)}")

    # ------------------------------------------------------------------
    # Step 2 — write subject-based splits
    # ------------------------------------------------------------------
    print("\nSTEP 2: Writing subject-based splits...")
    write_splits(variant_dir, window_specs)

    # ------------------------------------------------------------------
    # Step 3 — process each sensor
    # ------------------------------------------------------------------
    print("\nSTEP 3: Processing sensors...")

    rng = np.random.default_rng(SEED)
    N_total = len(window_specs)
    zeros_f32 = np.zeros(N_total, dtype=np.float32)

    for sensor_name, cols in SENSORS.items():
        print(f"\n  Processing: {sensor_name}")

        X_list        = []
        y_list        = []
        subj_list     = []
        base_idx_list = []

        subj_cache: dict = {}

        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subj_cache:
                subj_cache[subj_idx] = load_subject_data_with_retry(
                    DATA_PATH / f"mHealth_subject{subj_idx}.log"
                )
            data = subj_cache[subj_idx]

            # 1. Extract raw window
            win = data[win_start:win_end, cols].copy()

            # 2. Apply AWGN on raw signal (BEFORE resampling and z-score)
            win = apply_awgn(win, rng, alpha=alpha)

            # 3. Resample (no-op when ORIGINAL_FS == FS)
            win = resample_window(win, ORIGINAL_FS, FS)

            # 4. Z-score normalise per channel
            win = zscore_window(win)

            X_list.append(win.T.astype(np.float32))   # (C, L)
            y_list.append(label - 1)                   # 0-based
            subj_list.append(subj_idx)
            base_idx_list.append(w_idx)

        X            = np.stack(X_list,    axis=0)              # (N, C, L)
        y            = np.array(y_list,    dtype=np.int64)
        subject_ids  = np.array(subj_list, dtype=np.int64)
        base_win_idx = np.array(base_idx_list, dtype=np.int64)
        N            = len(y)
        zeros_f32_n  = np.zeros(N, dtype=np.float32)

        # ---- Save NPZ ----
        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X=X,
            y=y,
            subject_id=subject_ids,
            base_window_idx=base_win_idx,
            tremor_freq=zeros_f32_n,
            tremor_acc_rms=zeros_f32_n,
            tremor_gyro_rms=zeros_f32_n,
            tremor_score=zeros_f32_n,
            fs=FS,
            window_len=window_len_out,
            stride=stride_out,
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )

        # ---- Save TXT (same column layout as DataGenerator_Tremor.py) ----
        txt_out   = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)                       # (N, C*L)
        zeros_col = np.zeros((N, 1), dtype=np.float32)
        sensor_txt = np.hstack([
            flat_rows,
            (y + 1).reshape(-1, 1),      # col -7: activity 1-based
            subject_ids.reshape(-1, 1),   # col -6: subject id
            base_win_idx.reshape(-1, 1),  # col -5: base window idx
            zeros_col,                    # col -4: tremor_freq
            zeros_col,                    # col -3: tremor_acc_rms
            zeros_col,                    # col -2: tremor_gyro_rms
            zeros_col,                    # col -1: tremor_score
        ]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"    ✓ Saved: {npz_out.name}, {txt_out.name} | X={X.shape}")

    # ------------------------------------------------------------------
    # Step 4 — write info.txt
    # ------------------------------------------------------------------
    print("\nSTEP 4: Writing info.txt...")
    _write_info(variant_dir, alpha, snr_approx, len(window_specs),
                window_len_out, stride_out)

    print(f"\n  DONE: {variant_name}")
    print("=" * 80 + "\n")


# ============================================================
# Info file
# ============================================================

def _write_info(variant_dir: Path, alpha: float, snr_db: float,
                n_windows: int, window_len: int, stride: int) -> None:
    lines = [
        "=" * 60,
        "AWGN DATASET CONFIGURATION",
        "=" * 60,
        "",
        "Basic parameters:",
        f"  Original FS    : {ORIGINAL_FS} Hz",
        f"  Target FS      : {FS} Hz",
        f"  Window         : {WINDOW_SEC} s  ({window_len} samples)",
        f"  Stride         : {STRIDE_SEC} s  ({stride} samples)",
        f"  Seed           : {SEED}",
        "",
        "AWGN configuration:",
        f"  Alpha (σ/RMS)  : {alpha}",
        f"  Approx. SNR    : {snr_db:.1f} dB",
        "  Noise model    : n ~ N(0, (alpha * RMS(x_raw))^2)  per channel",
        "  Applied        : BEFORE resampling and z-score normalisation",
        "",
        f"  Total windows  : {n_windows}",
        "",
        "Subject splits:",
        f"  Train subjects : {sorted(TRAIN_SUBJECTS)}",
        f"  Val subjects   : {sorted(VAL_SUBJECTS)}",
        f"  Test subjects  : {sorted(TEST_SUBJECTS)}",
        "",
        "File format (NPZ):",
        "  X              : (N, C, L)  z-scored AWGN-corrupted sensor data",
        "  y              : (N,)       activity labels  0-based (0-11)",
        "  subject_id     : (N,)       subject IDs 1-10",
        "  base_window_idx: (N,)       window index",
        "  tremor_*       : (N,)       all zeros (no tremor in source signal)",
        "",
        "File format (TXT / CSV):",
        "  columns: [sensor_data C*L | activity | subject | base_win_idx |",
        "            tremor_freq | tremor_acc_rms | tremor_gyro_rms | tremor_score]",
        "=" * 60,
    ]
    (variant_dir / "info.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  Wrote info.txt")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Build folder name from alpha using per-mille (α × 1000), zero-padded to 3 digits:
    #   ALPHA=0.1   → alpha_tag="100"  → folder: s4_w4_fs50_awgn_a100
    #   ALPHA=0.316 → alpha_tag="316"  → folder: s4_w4_fs50_awgn_a316
    #   ALPHA=0.032 → alpha_tag="032"  → folder: s4_w4_fs50_awgn_a032
    #   ALPHA=0.01  → alpha_tag="010"  → folder: s4_w4_fs50_awgn_a010
    alpha_tag = str(round(ALPHA * 1000)).zfill(3)

    fs_str    = f"fs{int(FS)}"
    base_name = f"s{int(STRIDE_SEC)}_w{int(WINDOW_SEC)}_{fs_str}"
    variant_name = f"{base_name}_awgn_a{alpha_tag}"

    print("\n" + "=" * 80)
    print(f"GENERATING AWGN DATASET")
    print(f"  ALPHA        = {ALPHA}  (σ = ALPHA × per-channel RMS)")
    print(f"  Approx. SNR  = {-20 * np.log10(ALPHA):.1f} dB")
    print(f"  Output folder: {variant_name}")
    print(f"  Output base  : {OUT_BASE}")
    print("=" * 80 + "\n")

    generate_awgn_dataset(variant_name, alpha=ALPHA)

    print("\n" + "=" * 80)
    print(f"COMPLETE: {variant_name}")
    print("=" * 80)
