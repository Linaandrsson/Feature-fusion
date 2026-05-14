"""
DataGenerator_CorruptAWGN.py
==============================

Corruption test dataset generator for AWGN (Additive White Gaussian Noise).

Simulates sensor noise observed in real-world deployments (thermal noise,
vibration artefacts, MEMS quantisation noise, electrical interference, etc.)
by injecting per-channel AWGN scaled relative to the local signal RMS:

    n_c ~ N(0, sigma_c^2)   where  sigma_c = AWGN_ALPHA * RMS(x_c)

The noise is added in the raw (physical) signal domain, BEFORE resampling and
z-score normalisation, so the effective SNR is:

    SNR_dB  =  20 * log10(1 / AWGN_ALPHA)

    AWGN_ALPHA = 0.10  →  SNR ≈ 20 dB  (light noise)
    AWGN_ALPHA = 0.20  →  SNR ≈ 14 dB
    AWGN_ALPHA = 0.50  →  SNR ≈  6 dB  (moderate noise)
    AWGN_ALPHA = 1.00  →  SNR ≈  0 dB  (noise power = signal power)

Output folder name format:
    s{STRIDE}_w{WINDOW}_fs{FS}_corrupt_awgn_a{int(round(AWGN_ALPHA * 100)):03d}
    e.g.  s4_w4_fs50_corrupt_awgn_a020   (AWGN_ALPHA = 0.20)
          s4_w4_fs50_corrupt_awgn_a100   (AWGN_ALPHA = 1.00)

The output directory structure and file formats are identical to the tremor
datasets produced by DataGenerator_Tremor.py:
  - One .npz and one .txt file per sensor
  - Tremor labels set to 0 (this is a sensor-corruption scenario, not tremor)
  - Same subject-based splits (TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9])
  - splits/ subfolder with train_idx.txt, val_idx.txt, test_idx.txt
  - info.txt summarising generation parameters

This means the extraction pipeline (extract_all_sensors.py + extract_features.py)
and the ablation scripts work on this data without any modification.

Usage:
    python tremor_simulation/DataGenerator_CorruptAWGN.py

To generate multiple alpha levels, change AWGN_ALPHA and re-run. Each run
produces a new, independent output folder.
"""

import sys
from pathlib import Path

# Ensure Code/ root is on sys.path
_CODE_ROOT = Path(__file__).parent.parent.resolve()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np
import time
from scipy.signal import resample
from typing import List, Tuple


# ============================================================
# CONFIG — must match the training/tremor pipeline
# ============================================================
ORIGINAL_FS  = 50       # Raw MHEALTH sampling rate (Hz)
FS           = 50       # Target output sampling rate (Hz)
WINDOW_SEC   = 4.0      # Window length (s)
STRIDE_SEC   = 4.0      # Stride (s) — 0 % overlap at default
AUG_SIZE     = 1        # Augmentation copies per window (1 = no duplication)
NOISE_LEVEL  = 0.01     # Std of tiny additive Gaussian noise after z-score
                        # (matches DataGenerator_Tremor.py; negligible vs AWGN)
NUM_SUBJECTS = 10

# ── AWGN strength ─────────────────────────────────────────────
# sigma_per_channel = AWGN_ALPHA * RMS(raw_channel)
# Change this value and re-run to produce a new corruption level.
AWGN_ALPHA = 0.20       # 0.20 → ~14 dB SNR

SEED        = 0         # Controls augment_window noise (kept consistent with
                        # DataGenerator_Tremor.py)
AWGN_SEED   = 99        # Controls AWGN injection (independent of tremor seed)

# ── Paths ──────────────────────────────────────────────────────
_script_dir = Path(__file__).parent.resolve()
_code_dir   = _script_dir.parent            # Code/ root
DATA_PATH   = _code_dir / "data" / "MHEALTHDATASET"
OUT_BASE    = _code_dir / "data" / "Tremor_datagenerator_files"

# ── Subject splits (must be consistent across the whole pipeline) ──
TEST_SUBJECTS  = [5, 10]
VAL_SUBJECTS   = [2, 7]
TRAIN_SUBJECTS = [1, 3, 4, 6, 8, 9]

# ── Sensor column map (0-indexed in raw MHEALTH .log files) ────
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

LABEL_COL = 23   # Activity label column in raw .log file


# ============================================================
# UTILITY FUNCTIONS
# (self-contained copies of the same helpers in DataGenerator_Tremor.py)
# ============================================================

def resample_window(win: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample window from original_fs to target_fs (no-op when equal)."""
    if original_fs == target_fs:
        return win
    L_original, C = win.shape
    L_target = int(round(L_original * target_fs / original_fs))
    resampled = np.zeros((L_target, C))
    for c in range(C):
        resampled[:, c] = resample(win[:, c], L_target)
    return resampled


def zscore_window(win: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation within a window."""
    mean = win.mean(axis=0, keepdims=True)
    std  = win.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def augment_window(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add tiny additive Gaussian noise after z-score (training diversity)."""
    return win + rng.normal(0, NOISE_LEVEL, size=win.shape)


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    """(N, C, L) → (N, C*L) in channel-block order."""
    N, C, L = X_c_l.shape
    return np.concatenate([X_c_l[:, c, :] for c in range(C)], axis=1)


def apply_awgn(win_raw: np.ndarray, rng: np.random.Generator, alpha: float) -> np.ndarray:
    """
    Apply AWGN to a raw sensor window.

    Noise is scaled per channel:
        sigma_c = alpha * RMS(x_c)

    Applied BEFORE resampling and z-score normalisation so that alpha maps
    directly to physical SNR:
        SNR_dB = 20 * log10(1 / alpha)

    Args:
        win_raw : (L, C) array of raw sensor samples
        rng     : numpy random Generator
        alpha   : noise strength (fraction of per-channel RMS)

    Returns:
        (L, C) array with AWGN added
    """
    L, C = win_raw.shape
    result = win_raw.copy()
    for c in range(C):
        signal_rms = float(np.sqrt(np.mean(win_raw[:, c] ** 2)))
        if signal_rms < 1e-8:
            signal_rms = 1.0          # avoid zero sigma on flat channels
        sigma = alpha * signal_rms
        result[:, c] += rng.normal(0.0, sigma, size=L)
    return result


def load_subject_data_with_retry(
    file_path: Path, max_retries: int = 5, delay: float = 2.0
) -> np.ndarray:
    """Load a MHEALTH subject .log file with retry logic."""
    for attempt in range(max_retries):
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1 or (data.ndim > 1 and data.shape[1] < 24):
                print(f"\nWARNING: {file_path.name} appears empty/incomplete "
                      f"(possible OneDrive sync issue).\n")
            return data
        except (TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"  Timeout loading {file_path.name}, "
                      f"retry {attempt + 1}/{max_retries} in {delay}s…")
                time.sleep(delay)
            else:
                raise


# ============================================================
# SPLITS HELPER
# ============================================================

def create_splits_files(
    window_specs: List[Tuple],
    aug_size: int,
    out_dir: Path,
) -> None:
    """
    Write train_idx.txt / val_idx.txt / test_idx.txt to out_dir.

    Row index for window w_idx, augmentation copy a:
        row = w_idx * aug_size + a

    Subject assignment:
        TEST_SUBJECTS  → test_idx.txt
        VAL_SUBJECTS   → val_idx.txt
        all others     → train_idx.txt
    """
    test_rows, val_rows, train_rows = [], [], []
    for w_idx, (subj_idx, _, _, _) in enumerate(window_specs):
        for a in range(aug_size):
            row = w_idx * aug_size + a
            if subj_idx in TEST_SUBJECTS:
                test_rows.append(row)
            elif subj_idx in VAL_SUBJECTS:
                val_rows.append(row)
            else:
                train_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "train_idx.txt", np.array(train_rows, dtype=int), fmt="%d")
    np.savetxt(out_dir / "val_idx.txt",   np.array(val_rows,   dtype=int), fmt="%d")
    np.savetxt(out_dir / "test_idx.txt",  np.array(test_rows,  dtype=int), fmt="%d")
    print(f"  ✓ splits/  train={len(train_rows)}  val={len(val_rows)}  "
          f"test={len(test_rows)}")


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_awgn_dataset(alpha: float = None) -> None:
    """
    Generate AWGN corruption dataset.

    Args:
        alpha : AWGN strength (overrides AWGN_ALPHA if provided)
    """
    if alpha is None:
        alpha = AWGN_ALPHA

    alpha_int    = int(round(alpha * 100))
    fs_str       = f"fs{int(FS)}"
    variant_name = (
        f"s{int(STRIDE_SEC)}_w{int(WINDOW_SEC)}_{fs_str}"
        f"_corrupt_awgn_a{alpha_int:03d}"
    )

    snr_db = 20.0 * np.log10(1.0 / alpha) if alpha > 0 else float("inf")

    print("=" * 80)
    print(f"AWGN CORRUPTION DATASET GENERATION: {variant_name}")
    print("=" * 80)
    print(f"  AWGN_ALPHA : {alpha}  (sigma = {alpha} * RMS per channel)")
    print(f"  SNR        : ~{snr_db:.1f} dB")
    print(f"  Original FS: {ORIGINAL_FS} Hz,  Target FS: {FS} Hz")
    print(f"  Window     : {WINDOW_SEC} s,  Stride: {STRIDE_SEC} s")
    print(f"  AUG_SIZE   : {AUG_SIZE}")
    print(f"  SEED       : {SEED},  AWGN_SEED: {AWGN_SEED}")
    print(f"  Splits     : TEST={TEST_SUBJECTS}, VAL={VAL_SUBJECTS}, "
          f"TRAIN={TRAIN_SUBJECTS}")
    print("=" * 80)

    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    window_len = int(round(WINDOW_SEC * ORIGINAL_FS))
    stride     = int(round(STRIDE_SEC * ORIGINAL_FS))

    # ── Step 1: Collect window specs ──────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: Collecting window specifications …")
    print("=" * 80)

    window_specs: List[Tuple] = []

    for subj_idx in range(1, NUM_SUBJECTS + 1):
        fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        if not fname.exists():
            print(f"  [Subject {subj_idx:2d}] NOT FOUND: {fname}")
            continue

        data   = load_subject_data_with_retry(fname)
        labels = data[:, LABEL_COL].astype(int)

        i = 0
        n_windows = 0
        while i + window_len <= len(data):
            lab_win = labels[i : i + window_len]
            if np.all(lab_win >= 1):
                window_specs.append((subj_idx, int(lab_win[0]), i, i + window_len))
                n_windows += 1
            i += stride

        print(f"  [Subject {subj_idx:2d}]  {n_windows:4d} windows")

    print(f"\n  ✓ Total windows: {len(window_specs)}")

    # ── Step 2: Pre-load all subject data ─────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: Loading subject data …")
    print("=" * 80)

    subject_data_cache = {}
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        fname = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        if fname.exists():
            subject_data_cache[subj_idx] = load_subject_data_with_retry(fname)
            print(f"  [Subject {subj_idx:2d}]  loaded  {subject_data_cache[subj_idx].shape}")

    # ── Step 3: Process each sensor ───────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: Applying AWGN and saving sensor files …")
    print("=" * 80)

    aug_rng  = np.random.default_rng(SEED)
    awgn_rng = np.random.default_rng(AWGN_SEED)

    target_window_len = int(round(WINDOW_SEC * FS))

    for sensor_name, cols in SENSORS.items():
        print(f"\n  Sensor: {sensor_name}  cols={cols}")

        X_list       = []
        y_list       = []
        subj_list    = []
        base_idx_list = []

        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subject_data_cache:
                continue

            data    = subject_data_cache[subj_idx]
            win_raw = data[win_start:win_end, cols].copy()   # (L, C)

            # ── Apply AWGN (raw domain, before preprocessing) ──
            win_raw = apply_awgn(win_raw, awgn_rng, alpha)

            # ── Resample ──────────────────────────────────────
            win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)

            # ── Z-score normalise ──────────────────────────────
            win_norm = zscore_window(win_resampled)

            # ── AUG_SIZE copies with tiny Gaussian noise ───────
            for _ in range(AUG_SIZE):
                win_aug = augment_window(win_norm.copy(), aug_rng)

                X_list.append(win_aug.T.astype(np.float32))  # (C, L)
                y_list.append(label - 1)                      # 0-indexed
                subj_list.append(subj_idx)
                base_idx_list.append(w_idx)

        # Stack
        X           = np.stack(X_list, axis=0)                # (N, C, L)
        y           = np.array(y_list,       dtype=np.int64)
        subject_ids = np.array(subj_list,    dtype=np.int64)
        base_window = np.array(base_idx_list, dtype=np.int64)

        # Tremor labels — all zero (no tremor in this corruption scenario)
        N = len(y)
        tremor_freq     = np.zeros(N, dtype=np.float32)
        tremor_acc_rms  = np.zeros(N, dtype=np.float32)
        tremor_gyro_rms = np.zeros(N, dtype=np.float32)
        tremor_score    = np.zeros(N, dtype=np.int8)

        # ── Save NPZ ──────────────────────────────────────────
        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X               = X,
            y               = y,
            subject_id      = subject_ids,
            base_window_idx = base_window,
            tremor_freq     = tremor_freq,
            tremor_acc_rms  = tremor_acc_rms,
            tremor_gyro_rms = tremor_gyro_rms,
            tremor_score    = tremor_score,
            fs              = FS,
            window_len      = target_window_len,
            stride          = int(round(STRIDE_SEC * FS)),
            sensor_name     = sensor_name,
            sensor_cols     = np.array(cols, dtype=np.int64),
        )

        # ── Save TXT ──────────────────────────────────────────
        txt_out   = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)

        all_labels = np.column_stack([
            (y + 1),          # Activity (1-indexed)
            subject_ids,      # Subject ID
            base_window,      # Base window index
            tremor_freq,      # 0.0
            tremor_acc_rms,   # 0.0
            tremor_gyro_rms,  # 0.0
            tremor_score,     # 0
        ])
        sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        C_dim = X.shape[1]
        print(f"    ✓ {npz_out.name}  X={X.shape}  "
              f"[{C_dim}ch × {target_window_len}samples → "
              f"{flat_rows.shape[1]} + 7 cols]")

    # ── Step 4: Create splits/ folder ─────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: Writing splits/ …")
    print("=" * 80)

    create_splits_files(
        window_specs = window_specs,
        aug_size     = AUG_SIZE,
        out_dir      = variant_dir / "splits",
    )

    # ── Step 5: Write info.txt ────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 5: Writing info.txt …")
    print("=" * 80)

    target_window_len = int(round(WINDOW_SEC * FS))
    stride_samples    = int(round(STRIDE_SEC * FS))
    overlap           = 1.0 - (STRIDE_SEC / WINDOW_SEC) if WINDOW_SEC > 0 else 0.0

    info_lines = [
        "=" * 80,
        "AWGN CORRUPTION DATASET CONFIGURATION",
        "=" * 80,
        "",
        "Corruption type: Additive White Gaussian Noise (AWGN)",
        f"  n_c ~ N(0, sigma_c^2)  where  sigma_c = AWGN_ALPHA * RMS(x_c)",
        f"  Applied per channel, in the RAW signal domain (before preprocessing)",
        "",
        f"  AWGN_ALPHA : {alpha}",
        f"  SNR        : ~{snr_db:.1f} dB  (20*log10(1/alpha))",
        f"  AWGN seed  : {AWGN_SEED}",
        "",
        "=" * 80,
        "Basic Parameters:",
        "=" * 80,
        f"  Original FS        : {ORIGINAL_FS} Hz",
        f"  Target FS          : {FS} Hz",
        f"  Window size        : {WINDOW_SEC} s ({target_window_len} samples)",
        f"  Stride             : {STRIDE_SEC} s ({stride_samples} samples)",
        f"  Overlap            : {overlap:.2f}",
        f"  AUG_SIZE           : {AUG_SIZE}",
        f"  Augmentation noise : {NOISE_LEVEL} (tiny z-score noise after AWGN)",
        f"  Augmentation seed  : {SEED}",
        "",
        "=" * 80,
        "Tremor labels (all zero — no tremor in this corruption scenario):",
        "=" * 80,
        "  tremor_freq     = 0.0",
        "  tremor_acc_rms  = 0.0",
        "  tremor_gyro_rms = 0.0",
        "  tremor_score    = 0",
        "",
        "  The tremor label columns are present for format compatibility with the",
        "  tremor datasets. They carry no information in this dataset.",
        "",
        "=" * 80,
        "Subject splits:",
        "=" * 80,
        f"  TEST_SUBJECTS  : {TEST_SUBJECTS}",
        f"  VAL_SUBJECTS   : {VAL_SUBJECTS}",
        f"  TRAIN_SUBJECTS : {TRAIN_SUBJECTS}",
        "",
        "  Split index files: splits/train_idx.txt, val_idx.txt, test_idx.txt",
        "",
        "=" * 80,
        "File format (identical to DataGenerator_Tremor.py output):",
        "=" * 80,
        "  NPZ arrays: X (N,C,L), y (N,), subject_id (N,),",
        "              base_window_idx (N,), tremor_freq (N,),",
        "              tremor_acc_rms (N,), tremor_gyro_rms (N,), tremor_score (N,)",
        "",
        "  TXT format: (N, C*L + 7)  CSV, comma-separated",
        f"    3-axis sensors: {3*target_window_len} sensor cols + 7 label cols",
        f"    ECG sensor    : {2*target_window_len} sensor cols + 7 label cols",
        "    Label columns (last 7):",
        "      [-7] Activity label (1-indexed, 1-12)",
        "      [-6] Subject ID (1-10)",
        "      [-5] Base window index",
        "      [-4] tremor_freq   (= 0.0)",
        "      [-3] tremor_acc_rms  (= 0.0)",
        "      [-2] tremor_gyro_rms (= 0.0)",
        "      [-1] tremor_score    (= 0)",
        "",
        "=" * 80,
        "Sensor list (all sensors receive AWGN):",
        "=" * 80,
    ]

    for sname, scols in SENSORS.items():
        info_lines.append(f"  {sname:<14s}  MHEALTH cols {scols}")

    info_lines += [
        "",
        "=" * 80,
        f"Generated by: tremor_simulation/DataGenerator_CorruptAWGN.py",
        "Dataset     : MHEALTH (10 subjects, 12 activities)",
        "=" * 80,
    ]

    info_path = variant_dir / "info.txt"
    info_path.write_text("\n".join(info_lines))
    print(f"  ✓ {info_path.name}")

    print("\n" + "=" * 80)
    print(f"✓ DATASET GENERATION COMPLETE: {variant_name}")
    print(f"  Output: {variant_dir}")
    print(f"  AWGN_ALPHA = {alpha}  (SNR ~ {snr_db:.1f} dB)")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Change AWGN_ALPHA at the top of this file to generate different levels.
    # Run multiple times with different alpha values to build a robustness sweep:
    #
    #   AWGN_ALPHA = 0.10  →  s4_w4_fs50_corrupt_awgn_a010  (~20 dB SNR)
    #   AWGN_ALPHA = 0.20  →  s4_w4_fs50_corrupt_awgn_a020  (~14 dB SNR)
    #   AWGN_ALPHA = 0.50  →  s4_w4_fs50_corrupt_awgn_a050  (~6 dB SNR)
    #   AWGN_ALPHA = 1.00  →  s4_w4_fs50_corrupt_awgn_a100  (~0 dB SNR)
    #
    # Or, to generate a full sweep in one run, uncomment the loop below and
    # comment out the single generate_awgn_dataset() call.

    generate_awgn_dataset()

    # ── Sweep (uncomment to generate all levels at once) ──────
    # for alpha in [0.10, 0.20, 0.50, 1.00]:
    #     generate_awgn_dataset(alpha=alpha)
