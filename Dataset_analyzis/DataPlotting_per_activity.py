import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# -------------------------
# Config
# -------------------------
base_dir = Path("/Users/linaandersson/Desktop/master/Code/data/Datagenerator_files/fs50_s0.5_w1_aug2")
label_to_plot = 12.        # 1..12
window_len = 50            # samples per axis
fs = 50

# Mode selection:
# - "random": Pick random window with the label
# - "first": Pick first window with the label  
# - "specific": Use specific_index value
mode = "random"            # Options: "random", "first", "specific"
specific_index = 7385      # Only used if mode="specific"

# Only include sensors that actually exist in mHealth:
sensor_files = {
    "Acc_ankle": base_dir / "Acc_ankle.txt",
    "Gyro_ankle": base_dir / "Gyro_ankle.txt",
    "Mag_ankle": base_dir / "Mag_ankle.txt",
    "Acc_arm":   base_dir / "Acc_arm.txt",
    "Gyro_arm":  base_dir / "Gyro_arm.txt",
    "Mag_arm":   base_dir / "Mag_arm.txt",
    "Acc_chest": base_dir / "Acc_chest.txt",
    "ECG": base_dir / "ECG.txt",
}

# Keep only files that exist (safe guard)
sensor_files = {k: v for k, v in sensor_files.items() if v.exists()}
if len(sensor_files) == 0:
    raise FileNotFoundError("No sensor files found. Check base_dir path.")

# -------------------------
# Helper: parse one row -> x,y,z,label or lead1,lead2,label
# -------------------------
def parse_row(row, L=50, num_channels=3):
    if num_channels == 2:
        # ECG case (2 leads)
        x = row[0:L]
        y = row[L:2*L]
        label = int(round(row[2*L]))
        return x, y, None, label
    else:
        # Acc, Gyro, Mag (3 channels)
        x = row[0:L]
        y = row[L:2*L]
        z = row[2*L:3*L]
        label = int(round(row[3*L]))
        return x, y, z, label

# -------------------------
# Load all sensors
# -------------------------
data = {}
for name, fpath in sensor_files.items():
    data[name] = np.loadtxt(fpath, delimiter=",")

# -------------------------
# Choose an index based on mode
# -------------------------
if mode == "specific":
    idx = specific_index
    print(f"Using specific index: {idx}")
else:
    # Filter by label for random/first modes
    ref_name = list(data.keys())[0]
    # Determine label column based on sensor type
    if "ECG" in ref_name:
        label_col = 2 * window_len
    else:
        label_col = 3 * window_len
    labels_ref = data[ref_name][:, label_col].astype(int)
    
    idx_candidates = np.where(labels_ref == label_to_plot)[0]
    if len(idx_candidates) == 0:
        raise ValueError(f"No windows found for label {label_to_plot} in {ref_name}")
    
    if mode == "random":
        idx = np.random.choice(idx_candidates)
    else:  # mode == "first"
        idx = idx_candidates[0]
    
    print(f"Plotting label {label_to_plot} using window index {idx} (mode={mode}, ref={ref_name})")

# -------------------------
# Plot: dynamic grid based on number of sensors
# -------------------------
names = sorted(data.keys())
n_plots = len(names)

cols = 3
rows = math.ceil(n_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows), sharex=True)
axes = np.array(axes).reshape(-1)  # flatten safely

t = np.arange(window_len) / fs

for i, name in enumerate(names):
    row = data[name][idx, :]
    # Determine number of channels based on sensor type
    num_channels = 2 if "ECG" in name else 3
    x, y, z, lab = parse_row(row, window_len, num_channels)

    ax = axes[i]
    if num_channels == 2:
        # ECG: 2 leads
        ax.plot(t, x, label="lead1", color='red')
        ax.plot(t, y, label="lead2", color='darkred')
    else:
        # Acc, Gyro, Mag: 3 channels
        ax.plot(t, x, label="x")
        ax.plot(t, y, label="y")
        ax.plot(t, z, label="z")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, window_len / fs)

# Turn off unused axes
for j in range(n_plots, len(axes)):
    axes[j].axis("off")

# Legend once
axes[0].legend(loc="upper right")
fig.suptitle(f"Sensor windows for Activity Label L{label_to_plot}", fontsize=16)
plt.tight_layout()

# Save plot as PNG
output_dir = Path("/Users/linaandersson/Desktop/master/Code/Dataset_analyzis/plots_per_activity")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"plot_activity_L{label_to_plot}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.close()
