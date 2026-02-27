import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Config
# -------------------------
base_dir = Path("/Users/linaandersson/Desktop/master/Code/data/Datagenerator_files/fs50_s0.5_w2_aug2")

# Choose one sensor file to visualize
sensor_file = base_dir / "Acc_arm.txt"   # e.g. Acc_arm.txt, Acc_chest.txt, Gyro_ankle.txt, Mag_arm.txt

fs = 50
seq_len = 100  # 2 second window
labels = list(range(1, 13))  # 1..12

# How to pick windows for each activity
mode = "comparison"  
# "random": picks one random window,
# "first": picks the first window,
# "mean": averages multiple windows for the selected activity,
# "specific": uses a specific index from specific_index variable,
# "comparison": shows specific_index + 6 random examples of comparison_label

n_windows_mean = 50  # how many windows to average per label (used only if mode="mean")
specific_index = 7385  # Only used if mode="specific" or "comparison"
comparison_label = 7   # Only used if mode="comparison" - label to compare against (e.g., predicted label)

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
# Helper: parse row -> x,y,z,label (or x,y,None,label for ECG)
# -------------------------
def parse_xyz(row, L=50):
    # Detect number of channels based on row length
    # ECG: 2*L + 1 = 201 for L=100
    # Acc/Gyro/Mag: 3*L + 1 = 301 for L=100
    if len(row) == 2*L + 1:
        # ECG case (2 leads)
        x = row[0:L]
        y = row[L:2*L]
        z = None
        lab = int(round(row[2*L]))
    else:
        # Acc, Gyro, Mag (3 channels)
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
# time axis
t = np.arange(seq_len) / fs

# -------------------------
# Comparison mode: show misclassified + examples
# -------------------------
if mode == "comparison":
    _, _, _, true_lab = parse_xyz(arr[specific_index, :], seq_len)
    print(f"Specific index {specific_index}: true=L{true_lab}, comparing with L{comparison_label}")
    
    # Get examples of TRUE label (excluding the misclassified sample itself)
    true_idx = np.where(y == true_lab)[0]
    true_idx = true_idx[true_idx != specific_index]  # Exclude the misclassified sample
    if len(true_idx) < 3:
        print(f"Warning: Only {len(true_idx)} other samples available for L{true_lab}")
        n_true_examples = len(true_idx)
    else:
        n_true_examples = 3
    true_example_indices = np.random.choice(true_idx, size=n_true_examples, replace=False) if len(true_idx) > 0 else []
    
    # Get examples of PREDICTED label
    comparison_idx = np.where(y == comparison_label)[0]
    if len(comparison_idx) < 4:
        print(f"Warning: Only {len(comparison_idx)} samples available for L{comparison_label}")
        n_pred_examples = len(comparison_idx)
    else:
        n_pred_examples = 4
    pred_example_indices = np.random.choice(comparison_idx, size=n_pred_examples, replace=False)
    
    # Plot in 2 rows: Row 1 = misclassified + true label examples, Row 2 = predicted label examples
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    
    # Row 1, Col 0: the misclassified sample
    ax = axes[0, 0]
    x, yv, z, _ = parse_xyz(arr[specific_index, :], seq_len)
    if z is None:
        # ECG: 2 leads
        ax.plot(t, x, label="lead1", color='red')
        ax.plot(t, yv, label="lead2", color='darkred')
    else:
        # Acc, Gyro, Mag: 3 channels
        ax.plot(t, x, label="x")
        ax.plot(t, yv, label="y")
        ax.plot(t, z, label="z")
    ax.set_title(f"MISCLASSIFIED\nIndex {specific_index}\nTrue: L{true_lab}, Pred: L{comparison_label}", fontweight='bold', color='red', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 1, Cols 1-3: Examples of TRUE label
    for j in range(n_true_examples):
        ax = axes[0, j + 1]
        idx = true_example_indices[j]
        x, yv, z, _ = parse_xyz(arr[idx, :], seq_len)
        if z is None:
            # ECG: 2 leads
            ax.plot(t, x, label="lead1", color='red')
            ax.plot(t, yv, label="lead2", color='darkred')
        else:
            # Acc, Gyro, Mag: 3 channels
            ax.plot(t, x, label="x")
            ax.plot(t, yv, label="y")
            ax.plot(t, z, label="z")
        ax.set_title(f"True Label Example {j+1}\nL{true_lab} (idx {idx})", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Turn off unused subplots in first row
    for j in range(n_true_examples + 1, 4):
        axes[0, j].axis('off')
    
    # Row 2: Examples of PREDICTED label
    for j in range(n_pred_examples):
        ax = axes[1, j]
        idx = pred_example_indices[j]
        x, yv, z, _ = parse_xyz(arr[idx, :], seq_len)
        if z is None:
            # ECG: 2 leads
            ax.plot(t, x, label="lead1", color='red')
            ax.plot(t, yv, label="lead2", color='darkred')
        else:
            # Acc, Gyro, Mag: 3 channels
            ax.plot(t, x, label="x")
            ax.plot(t, yv, label="y")
            ax.plot(t, z, label="z")
        ax.set_title(f"Predicted Label Example {j+1}\nL{comparison_label} (idx {idx})", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Turn off unused subplots in second row
    for j in range(n_pred_examples, 4):
        axes[1, j].axis('off')
    
    fig.suptitle(f"{sensor_file.name} — Misclassification Analysis\nTop: True Label (L{true_lab}) | Bottom: Predicted Label (L{comparison_label})", fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("/Users/linaandersson/Desktop/master/Code/Dataset_analyzis/plots_per_sensor")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"misclass_comparison_{sensor_file.stem}_idx{specific_index}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()
    
    import sys
    sys.exit(0)

# -------------------------
# Plot 12 activities (for other modes)
# -------------------------
fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
axes = axes.flatten()

# For specific mode, get the actual label first
if mode == "specific":
    _, _, _, actual_lab = parse_xyz(arr[specific_index, :], seq_len)
    print(f"Specific index {specific_index} has label L{actual_lab}")

for i, lab in enumerate(labels):
    ax = axes[i]
    
    if mode == "specific":
        # Only plot in the subplot that matches the actual label
        if lab == actual_lab:
            x, yv, z, _ = parse_xyz(arr[specific_index, :], seq_len)
            if z is None:
                # ECG: 2 leads
                ax.plot(t, x, label="lead1", color='red')
                ax.plot(t, yv, label="lead2", color='darkred')
            else:
                # Acc, Gyro, Mag: 3 channels
                ax.plot(t, x, label="x")
                ax.plot(t, yv, label="y")
                ax.plot(t, z, label="z")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, seq_len / fs)
            ax.set_title(f"L{lab} — Index {specific_index}: {label_names.get(lab, '')}")
        else:
            ax.set_title(f"L{lab}: {label_names.get(lab, '')}")
            ax.axis("off")
        continue
    
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
        raise ValueError("mode must be 'first', 'random', 'mean', or 'specific'")

    if z is None:
        # ECG: 2 leads
        ax.plot(t, x, label="lead1", color='red')
        ax.plot(t, yv, label="lead2", color='darkred')
    else:
        # Acc, Gyro, Mag: 3 channels
        ax.plot(t, x, label="x")
        ax.plot(t, yv, label="y")
        ax.plot(t, z, label="z")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, seq_len / fs)

    title = f"L{lab}: {label_names.get(lab, '')}".strip()
    ax.set_title(title)

# One legend for the whole figure
handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_legend, loc="upper right")

fig.suptitle(f"{sensor_file.name} — one subplot per activity (mode={mode})", fontsize=16)
plt.tight_layout()

# Save plot as PNG
output_dir = Path("/Users/linaandersson/Desktop/master/Code/Dataset_analyzis/plots_per_sensor")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"plot_sensor_{sensor_file.stem}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
plt.close()
