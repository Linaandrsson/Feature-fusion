import numpy as np
from pathlib import Path
import random
from scipy.signal import resample

# ============================================================
# CONFIG (edit these)
# ============================================================
ORIGINAL_FS = 50        # original sampling rate in the dataset (Hz)
FS = 50                 # target sampling rate (Hz) - set same as ORIGINAL_FS to skip resampling
WINDOW_SEC = 1.0        # window length in seconds
STRIDE_SEC = 1.0        # stride in seconds
AUG_SIZE = 2            # original augmentation mechanism
NOISE_LEVEL = 0.01      # original augmentation noise level
NUM_SUBJECTS = 10

# Dataset location (folder containing mHealth_subject*.log)
DATA_PATH = Path(
    "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/MHEALTHDATASET"
)

# Output base folder
OUT_BASE = Path(
    "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files"
)

# Reproducibility (controls AUG noise etc.)
SEED = 0

# ============================================================
# NEW: Apply noise to a percentage of BASE windows (no extra samples)
# ============================================================
NOISE_ON_ORIG_ENABLE = True

# Only these sensors get noise injected (others remain clean).
# Window selection (which base windows get noised) is global to preserve alignment.
NOISE_ON_ORIG_SENSORS = {"Acc_chest"}  # e.g. {"Acc_ankle","Gyro_ankle"}

# Probability that a BASE window gets noised (e.g. 0.30 = 30%)
NOISE_ON_ORIG_PROB = 0.50

# Noise type:
# - If True: AWGN with sigma (interpretable since windows are z-scored -> std≈1)
# - If False: scaled noise like original AUG (scale = level * max_abs(win))
NOISE_ON_ORIG_USE_AWGN = True
NOISE_ON_ORIG_AWGN_SIGMA = 0.3
NOISE_ON_ORIG_SCALE_LEVEL = 0.01

# Separate seed so window selection stays stable regardless of other randomness
NOISE_ON_ORIG_SEED = 123

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


def format_stride(s: float) -> str:
    if float(s).is_integer():
        return str(int(s))
    return str(s).rstrip("0").rstrip(".")


def format_noise_tag(prob: float) -> str:
    """Convert probability to tag, e.g. 0.3 -> N30"""
    return f"N{int(round(prob * 100))}"


def make_out_dir(fs: int, stride_sec: float, window_sec: float, aug_size: int, noise_tag: str = "") -> Path:
    stride_str = format_stride(stride_sec)
    win_str = format_stride(window_sec)

    name = f"fs{fs}_s{stride_str}_w{win_str}_aug{aug_size}"
    if noise_tag:
        name += f"_{noise_tag}"

    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    """
    X_c_l: (N, C, L)
    Return (N, C*L) in the format:
      [ch1_1..ch1_L | ch2_1..ch2_L | ... | chC_1..chC_L]
    """
    N, C, L = X_c_l.shape
    blocks = [X_c_l[:, c, :] for c in range(C)]  # each (N, L)
    return np.concatenate(blocks, axis=1)        # (N, C*L)


def awgn(win: np.ndarray, sigma: float) -> np.ndarray:
    """
    Additive White Gaussian Noise.
    win: shape (L, C), already z-scored -> std ~ 1 per channel
    """
    return win + sigma * np.random.randn(*win.shape)


def scaled_noise_like_aug(win: np.ndarray, level: float) -> np.ndarray:
    """
    Noise like original AUG mechanism:
      scale = level * max_abs(win) per channel
      noise ~ N(0, scale^2)
    """
    scale = level * np.max(np.abs(win), axis=0, keepdims=True)  # (1, C)
    return win + scale * np.random.randn(*win.shape)


# ============================================================
# Main generation
# ============================================================
def main():
    # Seed everything for reproducibility (AUG noise etc.)
    random.seed(SEED)
    np.random.seed(SEED)

    # Window specs are based on ORIGINAL sampling rate
    window_len_original = int(round(ORIGINAL_FS * WINDOW_SEC))
    stride_original = int(round(ORIGINAL_FS * STRIDE_SEC))
    stride_original = max(stride_original, 1)

    # Target window length after resampling
    window_len = int(round(FS * WINDOW_SEC))
    stride = int(round(FS * STRIDE_SEC))
    stride = max(stride, 1)

    # Output folder tag for the "noise on originals" mechanism
    noise_tag = format_noise_tag(NOISE_ON_ORIG_PROB) if NOISE_ON_ORIG_ENABLE else ""
    out_dir = make_out_dir(FS, STRIDE_SEC, WINDOW_SEC, AUG_SIZE, noise_tag=noise_tag)

    # -------------------------------
    # Step 1: Build global list of base windows (canonical ordering)
    # Each window is defined by: (subject_id, label, start_i, end_i)
    # -------------------------------
    window_specs = []
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        data = np.loadtxt(file_path)
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

    print(f"Total base windows (before augmentation): {len(window_specs)}")
    if ORIGINAL_FS != FS:
        print(f"Resampling from {ORIGINAL_FS}Hz to {FS}Hz")
    print(f"Window_len={window_len} samples, stride={stride} samples (={STRIDE_SEC}s), fs={FS}Hz")
    print(f"AUG_SIZE (original augmentation) = {AUG_SIZE}")
    print(f"Original augmentation NOISE_LEVEL = {NOISE_LEVEL}")
    print(
        f"NOISE_ON_ORIG_ENABLE={NOISE_ON_ORIG_ENABLE}, NOISE_ON_ORIG_PROB={NOISE_ON_ORIG_PROB}, "
        f"NOISE_ON_ORIG_SENSORS={sorted(list(NOISE_ON_ORIG_SENSORS))}, "
        f"NOISE_ON_ORIG_USE_AWGN={NOISE_ON_ORIG_USE_AWGN}, "
        f"AWGN_SIGMA={NOISE_ON_ORIG_AWGN_SIGMA}, SCALE_LEVEL={NOISE_ON_ORIG_SCALE_LEVEL}"
    )

    # -------------------------------
    # Step 1b: Precompute which BASE windows will be noised (global mask)
    # IMPORTANT: This decision must be identical for all sensors to preserve alignment.
    # -------------------------------
    if NOISE_ON_ORIG_ENABLE and NOISE_ON_ORIG_PROB > 0:
        rng_no = np.random.RandomState(NOISE_ON_ORIG_SEED)
        noisy_mask = rng_no.rand(len(window_specs)) < NOISE_ON_ORIG_PROB
        num_noisy = int(noisy_mask.sum())
        print(
            f"Noise-on-originals: will noise {num_noisy}/{len(window_specs)} base windows "
            f"({100.0 * num_noisy / len(window_specs):.1f}%)"
        )
    else:
        noisy_mask = np.zeros(len(window_specs), dtype=bool)
        print("Noise-on-originals disabled (or prob=0).")

    np.save(out_dir / "noisy_mask.npy", noisy_mask)


    # -------------------------------
    # Step 2: For each sensor, extract data for all window_specs in SAME order
    # Apply:
    #   (A) Z-score per window
    #   (B) Noise-on-originals for selected windows AND selected sensors
    #   (C) Original AUG loop (unchanged)
    # -------------------------------
    for sensor_name, cols in SENSORS.items():
        X_list = []
        y_list = []
        subj_list = []

        subj_cache = {}

        for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
            if subj_idx not in subj_cache:
                file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
                subj_cache[subj_idx] = np.loadtxt(file_path)

            data = subj_cache[subj_idx]
            win = data[win_start:win_end, cols]  # (L_original, C)

            # Resample (if needed)
            win = resample_window(win, ORIGINAL_FS, FS)  # (L_target, C)

            # Normalize
            win = zscore_window(win)

            # --------
            # (B) Noise on ORIGINALS (applied to a % of base windows, but only for selected sensors)
            # NOTE: no extra samples are added; this modifies the base "win" before AUG.
            # --------
            if noisy_mask[w_idx] and (sensor_name in NOISE_ON_ORIG_SENSORS):
                if NOISE_ON_ORIG_USE_AWGN:
                    win = awgn(win, NOISE_ON_ORIG_AWGN_SIGMA)
                else:
                    win = scaled_noise_like_aug(win, NOISE_ON_ORIG_SCALE_LEVEL)

            # --------
            # (C) ORIGINAL AUGMENTATION (kept as-is)
            # NOTE: This augments on top of the (possibly noised) base win.
            # --------
            for a in range(AUG_SIZE):
                if AUG_SIZE > 1:
                    scale = NOISE_LEVEL * np.max(np.abs(win), axis=0, keepdims=True)
                    noise = scale * np.random.randn(*win.shape)
                    win_aug = win + noise
                else:
                    win_aug = win

                X_list.append(win_aug.T.astype(np.float32))  # (C, L)
                y_list.append(label - 1)                    # 0..11
                subj_list.append(subj_idx)

        X = np.stack(X_list, axis=0)               # (N, C, L)
        y = np.array(y_list, dtype=np.int64)       # (N,)
        subject_ids = np.array(subj_list, dtype=np.int64)

        # Save NPZ
        npz_out = out_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X=X,
            y=y,
            subject_id=subject_ids,
            fs=FS,
            window_len=window_len,
            stride=stride,
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )

        # Save TXT
        txt_out = out_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)
        labels_col = (y + 1).reshape(-1, 1)
        sensor_txt = np.hstack([flat_rows, labels_col]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"[{sensor_name}] Saved: {txt_out.name}, {npz_out.name} | X={X.shape}, y={y.shape}")

    # -------------------------------
    # Step 3: Write info.txt
    # -------------------------------
    info_txt = out_dir / "info.txt"
    overlap = 1.0 - (STRIDE_SEC / WINDOW_SEC) if WINDOW_SEC > 0 else 0.0
    stride_samples = int(round(FS * STRIDE_SEC))

    info_lines = [
        f"Window size = {format_stride(WINDOW_SEC)} s",
        f"Overlap = {overlap}",
        f"Stride = {format_stride(STRIDE_SEC)} s / {stride_samples} samples",
        f"Fs = {FS} Hz",
        f"Aug nr (original augmentation) = {AUG_SIZE}",
        f"Original augmentation NOISE_LEVEL = {NOISE_LEVEL}",
        "",
        f"Noise-on-originals enabled = {NOISE_ON_ORIG_ENABLE}",
        f"Noise-on-originals prob = {NOISE_ON_ORIG_PROB}",
        f"Noise-on-originals sensors = {sorted(list(NOISE_ON_ORIG_SENSORS))}",
        f"Noise-on-originals type = {'AWGN' if NOISE_ON_ORIG_USE_AWGN else 'Scaled(max_abs)'}",
        f"Noise-on-originals AWGN sigma = {NOISE_ON_ORIG_AWGN_SIGMA}",
        f"Noise-on-originals SCALE_LEVEL = {NOISE_ON_ORIG_SCALE_LEVEL}",
        "",
        "The same index across different files corresponds to the same time window of the same activity recording,",
        "including augmented samples. Noise-on-originals uses a global window mask to preserve alignment across sensors.",
    ]
    info_txt.write_text("\n".join(info_lines), encoding="utf-8")
    print(f"Wrote {info_txt.name} to: {out_dir}")


if __name__ == "__main__":
    main()
