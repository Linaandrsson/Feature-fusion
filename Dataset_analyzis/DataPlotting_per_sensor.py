import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Config
# -------------------------
base_dir = Path("/Users/linaandersson/Desktop/master/Code/data/Datagenerator_files/fs50_s0.5_w1_aug2")

# Choose one sensor file to visualize
sensor_file = base_dir / "Acc_ankle.txt"   # e.g. Acc_arm.txt, Acc_chest.txt, Gyro_ankle.txt, Mag_arm.txt

fs = 50
seq_len = 50  # 1 second window
labels = list(range(1, 13))  # 1..12

# How to pick windows for each activity
mode = "first"  
# "random": picks one random window,
# "first": picks the first window,
# "mean": averages multiple windows for the selected activity

n_windows_mean = 50  # how many windows to average per label (used only if mode="mean")

np.random.seed(42)

# Label names (optional, for nicer titles)
label_names = {
    1: "Standing still",
    2: "Sitting & relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends",
    7: "Frontal arm raises",
    8: "Knees bending (squat)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back",
}

# -------------------------
# Helper: parse row -> x,y,z,label
# -------------------------
def parse_xyz(row, L=50):
    x = row[0:L]
    y = row[L:2*L]
    z = row[2*L:3*L]
    lab = int(round(row[3*L]))
    return x, y, z, lab

# -------------------------
# Load data
# -------------------------
arr = np.loadtxt(sensor_file, delimiter=",")
X = arr[:, :-1]                  # (N, 150)
y = arr[:, -1].astype(int)       # (N,) labels 1..12

# time axis
t = np.arange(seq_len) / fs

# -------------------------
# Plot 12 activities
# -------------------------
fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, lab in enumerate(labels):
    ax = axes[i]
    idx = np.where(y == lab)[0]

    if len(idx) == 0:
        ax.set_title(f"L{lab}: (no data)")
        ax.axis("off")
        continue

    if mode == "first":
        chosen = idx[0:1]
        x, yv, z, _ = parse_xyz(arr[chosen[0], :], seq_len)

    elif mode == "random":
        chosen = np.random.choice(idx)
        print(f"Chosen index for L{lab}: {chosen}")
        x, yv, z, _ = parse_xyz(arr[chosen, :], seq_len)

    elif mode == "mean":
        # pick up to n_windows_mean windows for stable "average pattern"
        k = min(n_windows_mean, len(idx))
        chosen_idx = np.random.choice(idx, size=k, replace=False)

        xs, ys, zs = [], [], []
        for j in chosen_idx:
            xj, yj, zj, _ = parse_xyz(arr[j, :], seq_len)
            xs.append(xj)
            ys.append(yj)
            zs.append(zj)

        x = np.mean(xs, axis=0)
        yv = np.mean(ys, axis=0)
        z = np.mean(zs, axis=0)

    else:
        raise ValueError("mode must be 'first', 'random', or 'mean'")

    ax.plot(t, x, label="x")
    ax.plot(t, yv, label="y")
    ax.plot(t, z, label="z")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)

    title = f"L{lab}: {label_names.get(lab, '')}".strip()
    ax.set_title(title)

# One legend for the whole figure
handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_legend, loc="upper right")

fig.suptitle(f"{sensor_file.name} — one subplot per activity (mode={mode})", fontsize=16)
plt.tight_layout()

# Save plot as PNG
output_file = Path("/Users/linaandersson/Desktop/master/Code/Dataset_analyzis") / f"plot_sensor_{sensor_file.stem}.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.close()
