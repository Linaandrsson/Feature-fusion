import numpy as np
from pathlib import Path

# -------------------------------
# Parameters
# -------------------------------
TT = 1.0                  # Window duration in seconds (e.g., 1.0, 2.0)
fs = 50                   # Sampling rate (mHealth is 50 Hz)
window_len = int(fs * TT) # Samples per window

overlap = 0.5             # 50% overlap (0.0 = no overlap)
stride = int(window_len * (1 - overlap))
stride = max(stride, 1)
print("stride: ", stride)

num_subjects = 10
aug_size = 2              # number of augmentations per window (1 = no augmentation)
noise_level = 0.01        # relative noise level (applied if aug_size > 1)

# Sensor columns (0-indexed) for LEFT-ANKLE accelerometer XYZ:
# README columns 6,7,8 -> python indices 5,6,7
sensor_cols = [3,4]

# Path to dataset folder containing mHealth_subject*.log
data_path = Path("/Users/linaandersson/.cache/kagglehub/datasets/nirmalsankalana/mhealth-dataset-data-set/versions/1/MHEALTHDATASET")

# Output paths
out_dir = Path("/Users/linaandersson/Desktop/master/Code/data")
out_dir.mkdir(parents=True, exist_ok=True)

txt_out = out_dir / "ECG.txt"
npz_out = out_dir / "ECG.npz"

# -------------------------------
# Helper: z-score normalize per channel in a window
# -------------------------------
def zscore_window(win: np.ndarray) -> np.ndarray:
    """
    win: shape (L, C)
    returns: normalized win (L, C)
    """
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # avoid divide-by-zero
    return (win - mean) / std

# -------------------------------
# Load and window the data
# -------------------------------
X_windows = []
y_windows = []
subject_ids = []

for subj_idx in range(1, num_subjects + 1):
    file_path = data_path / f"mHealth_subject{subj_idx}.log"
    data = np.loadtxt(file_path)

    labels = data[:, 23].astype(int)  # label column (1..12, 0 possible)
    sensor_xyz = data[:, sensor_cols] # shape (T, 3)

    # For each activity label 1..12, create windows
    for label in range(1, 13):
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue

        # IMPORTANT:
        # The dataset stores activities in contiguous blocks, so idx is typically contiguous.
        # We take the contiguous segment only (safe for mHealth).
        start_i, end_i = idx[0], idx[-1] + 1
        segment = sensor_xyz[start_i:end_i, :]  # shape (Tseg, 3)

        Tseg = segment.shape[0]
        if Tseg < window_len:
            continue

        # Sliding windows over this segment
        for start in range(0, Tseg - window_len + 1, stride):
            win = segment[start:start + window_len, :]  # (L, 3)

            # Normalize per window & per channel
            win = zscore_window(win)

            # Augmentation: noise injection (optional)
            for a in range(aug_size):
                if aug_size > 1:
                    # Add noise independently per channel
                    scale = noise_level * np.max(np.abs(win), axis=0, keepdims=True)
                    noise = scale * np.random.randn(*win.shape)
                    win_aug = win + noise
                else:
                    win_aug = win

                # Store in (C, L) format for CNN later
                X_windows.append(win_aug.T)          # (3, L)
                y_windows.append(label - 1)          # 0..11
                subject_ids.append(subj_idx)         # 1..10

# Convert to arrays
X = np.stack(X_windows, axis=0).astype(np.float32)  # (N, C, L)
y = np.array(y_windows, dtype=np.int64)             # (N,)
subject_ids = np.array(subject_ids, dtype=np.int64) # (N,)

print(f"Generated windows: X={X.shape}, y={y.shape}, subject_ids={subject_ids.shape}")
print(f"Window_len={window_len} samples, stride={stride} samples, overlap={overlap*100:.0f}%")

# -------------------------------
# Save outputs
# -------------------------------
# 1) Save a clean NPZ (recommended for later)
np.savez_compressed(npz_out, X=X, y=y, subject_id=subject_ids, fs=fs, window_len=window_len, stride=stride)
print(f"Saved NPZ to: {npz_out}")

# # 2) Save a TXT compatible with your current ClassificationModel.py (flat + label)
# # Format per row:
# # [x1..xL, y1..yL, z1..zL, label]
# X_flat = X.transpose(0, 2, 1)  # (N, L, C)
# # flatten as [x_block, y_block, z_block]
# x_block = X_flat[:, :, 0]
# y_block = X_flat[:, :, 1]
# z_block = X_flat[:, :, 2]
# flat_rows = np.concatenate([x_block, y_block, z_block], axis=1)  # (N, 3L)
# labels_col = (y + 1).reshape(-1, 1)  # back to 1..12 for consistency with older code
# SensorData_txt = np.hstack([flat_rows, labels_col]).astype(np.float32)

# np.savetxt(txt_out, SensorData_txt, delimiter=",", fmt="%.6f")
# print(f"Saved TXT to: {txt_out}")



# -------------------------------
# Save as TXT (generic for any number of channels)
# Each row becomes:
# [ch1_1..ch1_L | ch2_1..ch2_L | ... | chC_1..chC_L | label]
# -------------------------------

X_flat = X.transpose(0, 2, 1)  # (N, L, C)
flat_rows = X_flat.reshape(X.shape[0], -1)  # (N, L*C)

labels_col = (y + 1).reshape(-1, 1)  # back to 1..12 for readability
SensorData_txt = np.hstack([flat_rows, labels_col]).astype(np.float32)

np.savetxt(txt_out, SensorData_txt, delimiter=",", fmt="%.6f")
print(f"Saved TXT to: {txt_out}")
