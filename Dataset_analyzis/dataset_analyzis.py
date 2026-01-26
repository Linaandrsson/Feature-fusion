import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Activity labels
activity_names = {
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back"
}

data_dir = Path("data/MHEALTHDATASET")
files = sorted(data_dir.glob("mHealth_subject*.log"))

label_counts = {label: 0 for label in range(1, 13)}

for f in files:
    data = np.loadtxt(f)
    labels = data[:, 23].astype(int)  # kolonne 24 = label

    for label in range(1, 13):
        label_counts[label] += np.sum(labels == label)

# Lag tabell
summary_label = pd.DataFrame({
    "Label": [f"L{l}" for l in label_counts],
    "Samples": list(label_counts.values()),
})

summary_label["Seconds"] = summary_label["Samples"] / 50
summary_label["Minutes"] = summary_label["Seconds"] / 60

rows = []

for i, f in enumerate(files, start=1):
    data = np.loadtxt(f)
    labels = data[:, 23].astype(int)

    row = {"Subject": f"S{i}"}
    total = 0

    for label in range(1, 13):
        count = np.sum(labels == label)
        row[f"L{label}"] = count
        total += count

    row["Total"] = total
    rows.append(row)

summary_subject = pd.DataFrame(rows)

# Save results to file
output_dir = Path("Dataset_analyzis")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "dataset_summary.txt"

with open(output_file, "w") as f:
    f.write("=== Summary by Label ===\n")
    f.write(summary_label.to_string(index=False))
    f.write("\n\n=== Summary by Subject ===\n")
    f.write(summary_subject.to_string(index=False))
    f.write("\n")

print(f"Results saved to: {output_file}")

# Print results
print("\n=== Summary by Label ===")
print(summary_label.to_string(index=False))

print("\n=== Summary by Subject ===")
print(summary_subject.to_string(index=False))

# Create visualizations - Figure 1: Activity distribution
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Pie chart
labels_pie = [activity_names[i] for i in range(1, 13)]
sizes = summary_label["Samples"].values
colors = plt.cm.Set3(range(12))

ax1.pie(sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.set_title("Distribution of Samples per Activity", fontsize=14, fontweight='bold')

# Bar chart
ax2.bar(range(1, 13), sizes, color=colors, edgecolor='black')
ax2.set_xlabel("Activity Label", fontsize=12)
ax2.set_ylabel("Number of Samples", fontsize=12)
ax2.set_title("Samples per Activity", fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels([f"L{i}" for i in range(1, 13)])
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(sizes):
    ax2.text(i + 1, v + 500, str(v), ha='center', va='bottom', fontsize=9)

# Activity legend
ax3.axis('off')
ax3.set_title("Activity Labels", fontsize=12, fontweight='bold', pad=20)
legend_text = "\n".join([f"L{i}: {activity_names[i]}" for i in range(1, 13)])
ax3.text(0.1, 0.5, legend_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plot_file1 = output_dir / "dataset_distribution.png"
plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_file1}")
plt.show()

# Create visualizations - Figure 2: Subject overview
fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Heatmap: Samples per subject and activity
heatmap_data = []
for i in range(10):
    row_data = [summary_subject.iloc[i][f"L{j}"] for j in range(1, 13)]
    heatmap_data.append(row_data)

heatmap_data = np.array(heatmap_data)
im = ax1.imshow(heatmap_data, cmap='Blues', aspect='auto')

# Set ticks and labels
ax1.set_xticks(range(12))
ax1.set_yticks(range(10))
ax1.set_xticklabels([f"L{i}" for i in range(1, 13)])
ax1.set_yticklabels([f"S{i}" for i in range(1, 11)])
ax1.set_xlabel("Activity", fontsize=12)
ax1.set_ylabel("Subject", fontsize=12)
ax1.set_title("Heatmap: Samples per Subject and Activity", fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label("Number of Samples", rotation=270, labelpad=20)

# Add text annotations with dynamic color
threshold = (heatmap_data.max() + heatmap_data.min()) / 2
for i in range(10):
    for j in range(12):
        text_color = "white" if heatmap_data[i, j] > threshold else "black"
        text = ax1.text(j, i, int(heatmap_data[i, j]),
                       ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')

# Bar chart: Total samples per subject
totals = summary_subject["Total"].values
subject_colors = plt.cm.viridis(np.linspace(0, 1, 10))

ax2.barh(range(1, 11), totals, color=subject_colors, edgecolor='black')
ax2.set_ylabel("Subject", fontsize=12)
ax2.set_xlabel("Total Number of Samples", fontsize=12)
ax2.set_title("Total Samples per Subject", fontsize=14, fontweight='bold')
ax2.set_yticks(range(1, 11))
ax2.set_yticklabels([f"S{i}" for i in range(1, 11)])
ax2.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(totals):
    ax2.text(v + 200, i + 1, str(v), va='center', fontsize=9)

# Activity legend
ax3.axis('off')
ax3.set_title("Activity Labels", fontsize=12, fontweight='bold', pad=20)
legend_text = "\n".join([f"L{i}: {activity_names[i]}" for i in range(1, 13)])
ax3.text(0.1, 0.5, legend_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plot_file2 = output_dir / "subject_overview.png"
plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_file2}")
plt.show()
