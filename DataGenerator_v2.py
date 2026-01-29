import numpy as np
from pathlib import Path
import random

# ============================================================
# CONFIG (edit these)
# ============================================================
FS = 50                 # sampling rate
WINDOW_SEC = 1.0        # window length in seconds (e.g., 1.0)
STRIDE_SEC = 1        # stride in seconds (e.g., 0.5)
AUG_SIZE = 2            # 1 = no augmentation, 2 = original + 1 noisy, etc.
NOISE_LEVEL = 0.01      # relative noise (only used if AUG_SIZE > 1)
NUM_SUBJECTS = 10

# Dataset location (folder containing mHealth_subject*.log)
DATA_PATH = Path(
    "/Users/linaandersson/.cache/kagglehub/datasets/nirmalsankalana/mhealth-dataset-data-set/versions/1/MHEALTHDATASET"
)

# Output base folder
OUT_BASE = Path("/Users/linaandersson/Desktop/master/Code/data/Datagenerator_files")

# Reproducibility (optional but recommended)
SEED = 0

# ============================================================
# SENSOR COLUMN MAP (0-indexed)
# From README column layout: chest acc [0,1,2], ECG [3,4], etc. :contentReference[oaicite:2]{index=2}
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

LABEL_COL = 23  # label column in file (0-indexed), values: 0..12, we use 1..12 :contentReference[oaicite:3]{index=3}


# ============================================================
# Helpers
# ============================================================
def zscore_window(win: np.ndarray) -> np.ndarray:
    """
    win: shape (L, C)
    Normalize per channel within window.
    """
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def format_stride(s: float) -> str:
    # Ensures folder name matches your style: 0.5 not 0.50
    # and avoids scientific notation
    if float(s).is_integer():
        return str(int(s))
    return str(s).rstrip("0").rstrip(".")


def make_out_dir(fs: int, stride_sec: float, window_sec: float, aug_size: int) -> Path:
    stride_str = format_stride(stride_sec)
    win_str = format_stride(window_sec)
    out_dir = OUT_BASE / f"fs{fs}_s{stride_str}_w{win_str}_aug{aug_size}"
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


# ============================================================
# Main generation
# ============================================================
def main():
    # Seed everything (for reproducible augmentation order/noise)
    random.seed(SEED)
    np.random.seed(SEED)

    window_len = int(round(FS * WINDOW_SEC))
    stride = int(round(FS * STRIDE_SEC))
    stride = max(stride, 1)

    out_dir = make_out_dir(FS, STRIDE_SEC, WINDOW_SEC, AUG_SIZE)

    # -------------------------------
    # Step 1: Build a global list of windows (for alignment across sensors)
    # Each window is defined by: (subject_id, label, start_global_index)
    # This list defines the canonical ordering for ALL sensor outputs.
    # -------------------------------
    window_specs = []  # list of tuples (subj_idx, label_1to12, start_i, end_i)
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
        data = np.loadtxt(file_path)

        labels = data[:, LABEL_COL].astype(int)

        for label in range(1, 13):
            idx = np.where(labels == label)[0]
            if idx.size == 0:
                continue

            # mHealth stores each activity in contiguous blocks (as in your original script) :contentReference[oaicite:4]{index=4}
            start_i, end_i = int(idx[0]), int(idx[-1]) + 1
            seg_len = end_i - start_i
            if seg_len < window_len:
                continue

            # Sliding windows over this segment
            for offset in range(0, seg_len - window_len + 1, stride):
                win_start = start_i + offset
                win_end = win_start + window_len
                window_specs.append((subj_idx, label, win_start, win_end))

    if len(window_specs) == 0:
        raise RuntimeError("No windows generated. Check WINDOW_SEC/STRIDE_SEC and dataset path.")

    print(f"Total base windows (before augmentation): {len(window_specs)}")
    print(f"Window_len={window_len} samples, stride={stride} samples (={STRIDE_SEC}s), fs={FS}Hz, aug_size={AUG_SIZE}")

    # -------------------------------
    # Step 2: For each sensor, extract data for all window_specs in the SAME order
    # -------------------------------
    for sensor_name, cols in SENSORS.items():
        X_list = []
        y_list = []
        subj_list = []

        # We will cache per subject file load to avoid re-reading file for each window
        subj_cache = {}

        for (subj_idx, label, win_start, win_end) in window_specs:
            if subj_idx not in subj_cache:
                file_path = DATA_PATH / f"mHealth_subject{subj_idx}.log"
                subj_cache[subj_idx] = np.loadtxt(file_path)

            data = subj_cache[subj_idx]
            win = data[win_start:win_end, cols]  # (L, C)

            # normalize per window, per channel
            win = zscore_window(win)

            # augmentation loop: keep SAME number/order of augmented windows for all sensors
            for a in range(AUG_SIZE):
                if AUG_SIZE > 1:
                    scale = NOISE_LEVEL * np.max(np.abs(win), axis=0, keepdims=True)
                    noise = scale * np.random.randn(*win.shape)
                    win_aug = win + noise
                else:
                    win_aug = win

                # store as (C, L) for CNN later
                X_list.append(win_aug.T.astype(np.float32))  # (C, L)
                y_list.append(label - 1)                    # 0..11
                subj_list.append(subj_idx)                  # 1..10

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

        # Save TXT in your desired flat format:
        # [ch1_1..ch1_L | ch2_1..ch2_L | ... | label]
        txt_out = out_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)                 # (N, C*L)
        labels_col = (y + 1).reshape(-1, 1)                   # back to 1..12
        sensor_txt = np.hstack([flat_rows, labels_col]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"[{sensor_name}] Saved: {txt_out.name}, {npz_out.name} | X={X.shape}, y={y.shape}")

    # -------------------------------
    # Step 3: Write info.txt (as requested)
    # -------------------------------
    info_txt = out_dir / "info.txt"
    overlap = 1.0 - (STRIDE_SEC / WINDOW_SEC) if WINDOW_SEC > 0 else 0.0
    stride_samples = int(round(FS * STRIDE_SEC))

    info_lines = [
        f"Window size = {format_stride(WINDOW_SEC)} s",
        f"Overlap = {overlap}",
        f"Stride = {format_stride(STRIDE_SEC)} s / {stride_samples} samples",
        f"Fs = {FS} Hz",
        f"Aug nr = {AUG_SIZE}",
        "",
        "The same index across different files corresponds to the same time window of the same activity recording.",
        "",
        "Train/validation/test splits are stored in splits.npz and should be shared across all sensor files to ensure consistent indexing for correct feature-level fusion.",
        "",
    ]
    info_txt.write_text("\n".join(info_lines), encoding="utf-8")
    print(f"Wrote {info_txt.name} to: {out_dir}")

    # Also write a minimal manifest (optional)
    manifest = out_dir / "sensors_manifest.txt"
    manifest_lines = ["Sensor files generated with columns (0-indexed):"]
    for k, v in SENSORS.items():
        manifest_lines.append(f"{k}: {v}")
    manifest.write_text("\n".join(manifest_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
