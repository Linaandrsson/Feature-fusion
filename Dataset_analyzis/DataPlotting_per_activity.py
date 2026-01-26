import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# -------------------------
# Config
# -------------------------
base_dir = Path("/Users/linaandersson/Desktop/master/Code/data/Datagenerator_files/fs50_s0.5_w1_aug2")
label_to_plot = 1          # 1..12
window_len = 50            # samples per axis
fs = 50
random_pick = True         # True=random window, False=first window

# Only include sensors that actually exist in mHealth:
sensor_files = {
    "Acc_ankle": base_dir / "Acc_ankle.txt",
    "Gyro_ankle": base_dir / "Gyro_ankle.txt",
    "Mag_ankle": base_dir / "Mag_ankle.txt",
    "Acc_arm":   base_dir / "Acc_arm.txt",
    "Gyro_arm":  base_dir / "Gyro_arm.txt",
    "Mag_arm":   base_dir / "Mag_arm.txt",
    "Acc_chest": base_dir / "Acc_chest.txt",
}

# Keep only files that exist (safe guard)
sensor_files = {k: v for k, v in sensor_files.items() if v.exists()}
if len(sensor_files) == 0:
    raise FileNotFoundError("No sensor files found. Check base_dir path.")

# -------------------------
# Helper: parse one row -> x,y,z,label
# -------------------------
def parse_row(row, L=50):
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
# Choose an index for the label using a reference file
# -------------------------
ref_name = list(data.keys())[0]
labels_ref = data[ref_name][:, 3*window_len].astype(int)

idx_candidates = np.where(labels_ref == label_to_plot)[0]
if len(idx_candidates) == 0:
    raise ValueError(f"No windows found for label {label_to_plot} in {ref_name}")

idx = np.random.choice(idx_candidates) if random_pick else idx_candidates[0]
print(f"Plotting label {label_to_plot} using window index {idx} (ref={ref_name})")

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
    x, y, z, lab = parse_row(row, window_len)

    ax = axes[i]
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
output_file = "/Users/linaandersson/Desktop/master/Code/Dataset_analyzis/" + f"plot_activity_L{label_to_plot}.png"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.close()
