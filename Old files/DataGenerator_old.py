import numpy as np
from scipy.signal import resample
from scipy.io import savemat
from pathlib import Path

# -------------------------------
# Parameters
# -------------------------------

TT = 1                      # Segment duration in seconds
fs = 50                    # Target sampling rate
originalFs = 50             # Original sampling rate
TargetLength = fs * TT     # segment length in samples

new_size = (161280, 24)     # [samples, channels]
num_subjects = 10           # Number of test subjects (people)
aug_size = 2                # Data augmentation size( augmentation is carried out by adding noise), makes 2 versions of each signal segment
sensor_cols = [5, 6, 7]     # Left-ankle acceleration (X, Y, Z) - Python 0-indexed (README columns 6,7,8)
# changed datagenerator to be able to load multiple sensor columns into one file, for x,y,z axes


# -------------------------------
# Load raw subject data
# -------------------------------
data_path = "/Users/linaandersson/.cache/kagglehub/datasets/nirmalsankalana/mhealth-dataset-data-set/versions/1/MHEALTHDATASET"
P = []

for i in range(num_subjects):
    file_path = f"{data_path}/mHealth_subject{i+1}.log"
    data = np.loadtxt(file_path)
    P.append(data)

# -------------------------------
# Signal extraction + resampling
# -------------------------------
signals = []
labels = []

for subject in P:
    for label in range(1, 13):
        idx = np.where(subject[:, 23] == label)[0] # find all rows where column 24 (index 23) == label
        # idx is an array of indices for the current label
        if len(idx) == 0:
            continue

        # Extract all 3 axes (X, Y, Z) from left-ankle accelerometer
        signal_xyz = subject[idx][:, sensor_cols]  # subject[idx] = all the rows with the current label
        # [:, sensor_cols] = all rows from subject[idx], only columns sensor_cols. Shape: (num_samples, sensor_cols)

        for _ in range(aug_size):
            # Add noise to each axis independently
            noise = 0.01 * np.max(np.abs(signal_xyz), axis=0) * np.random.randn(*signal_xyz.shape) # random noise on 1% of max value of each axis
            augmented = signal_xyz + noise # augmented signal with noise added

            # Resample each axis to TargetLength
            resampled_x = resample(augmented[:, 0], TargetLength) 
            resampled_y = resample(augmented[:, 1], TargetLength)
            resampled_z = resample(augmented[:, 2], TargetLength)

            # Normalize each axis independently
            resampled_x = (resampled_x - np.mean(resampled_x)) / np.std(resampled_x)
            resampled_y = (resampled_y - np.mean(resampled_y)) / np.std(resampled_y)
            resampled_z = (resampled_z - np.mean(resampled_z)) / np.std(resampled_z)

            # Concatenate all axes: [x1, x2, ..., x50, y1, y2, ..., y50, z1, z2, ..., z50]
            combined = np.concatenate([resampled_x, resampled_y, resampled_z])

            signals.append(combined)
            labels.append(label)

# -------------------------------
# Save as text file
# -------------------------------
signals = np.array(signals)
labels = np.array(labels).reshape(-1, 1)

SensorData = np.hstack((signals, labels))

output_path = "/Users/linaandersson/Desktop/master/Code/data/SensorData.txt"
np.savetxt(
    output_path,
    SensorData,
    delimiter=",",
    fmt="%.6f"
)

print(f"SensorData.txt generated successfully at: {output_path}")