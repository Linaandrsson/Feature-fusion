"""
Plot sitting window – arm accelerometer comparison across tremor severities.

Loads Acc_arm.npz from three dataset variants, picks the first sitting window,
and produces a 3 × 3 figure:
  Rows   → X, Y, Z axes
  Columns → Clean | Mild-Moderate | Moderate-Severe
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("data/Tremor_datagenerator_files")

DATASETS = {
    "Clean":             BASE / "s4_w4_fs50_tremor_clean"      / "Acc_arm.npz",
    "Mild–Moderate":     BASE / "s4_w4_fs50_tremor_mild_mod"   / "Acc_arm.npz",
    "Moderate–Severe":   BASE / "s4_w4_fs50_tremor_mod_severe" / "Acc_arm.npz",
}

AXIS_NAMES  = ["X", "Y", "Z"]
COLORS      = ["#2166ac", "#d6604d", "#4dac26"]   # blue, red, green per axis
SITTING_IDX = 1       # 0-indexed (MHEALTH activity 2 = Sitting and relaxing)
FS          = 50      # Hz

# ── Load data ────────────────────────────────────────────────────────────────
windows = {}
scores  = {}
for label, path in DATASETS.items():
    d    = np.load(path)
    mask = d["y"] == SITTING_IDX
    X    = d["X"][mask]           # (N_sitting, 3, 200)
    sc   = d["tremor_score"][mask] if "tremor_score" in d else np.zeros(mask.sum())
    windows[label] = X
    scores[label]  = sc

# Pick the same base window index across all datasets for a fair comparison.
# Use the first available sitting window from the clean set as the reference.
ref_d   = np.load(list(DATASETS.values())[0])
ref_mask = ref_d["y"] == SITTING_IDX
ref_idx  = ref_d["base_window_idx"][ref_mask][0]   # first sitting base_window_idx

selected = {}
for label, path in DATASETS.items():
    d    = np.load(path)
    mask = d["y"] == SITTING_IDX
    bwi  = d["base_window_idx"][mask]
    # Find the window with matching base_window_idx; fall back to index 0
    hit  = np.where(bwi == ref_idx)[0]
    i    = hit[0] if len(hit) > 0 else 0
    selected[label] = {
        "signal": d["X"][mask][i],          # (3, 200)
        "score":  d["tremor_score"][mask][i] if "tremor_score" in d else 0,
        "freq":   d["tremor_freq"][mask][i]  if "tremor_freq"  in d else 0,
        "rms":    d["tremor_acc_rms"][mask][i] if "tremor_acc_rms" in d else 0,
    }

# ── Figure ───────────────────────────────────────────────────────────────────
n_axes  = 3
n_cond  = 3
labels  = list(selected.keys())
t       = np.arange(200) / FS   # seconds (0 – 4 s)

fig, axes = plt.subplots(
    nrows=n_axes, ncols=n_cond,
    figsize=(13, 7),
    sharex=True,
    sharey="row",
)
fig.subplots_adjust(hspace=0.15, wspace=0.08)

for col, label in enumerate(labels):
    sig   = selected[label]["signal"]    # (3, 200)
    score = selected[label]["score"]
    freq  = selected[label]["freq"]
    rms   = selected[label]["rms"]

    for row, ax_name in enumerate(AXIS_NAMES):
        ax = axes[row, col]
        ax.plot(t, sig[row], color=COLORS[row], linewidth=0.9)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)

        # Column header (top row only)
        if row == 0:
            subtitle = f"Score {int(score)}"
            if freq > 0:
                subtitle += f"  |  {freq:.1f} Hz  |  RMS {rms:.3f} m/s²"
            ax.set_title(f"{label}\n{subtitle}", fontsize=9.5, pad=4)

        # Row label (left column only)
        if col == 0:
            ax.set_ylabel(f"{ax_name}-axis\n(m/s²)", fontsize=9)

        # X-axis label (bottom row only)
        if row == n_axes - 1:
            ax.set_xlabel("Time (s)", fontsize=9)

        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

fig.suptitle(
    "Arm Accelerometer – Sitting Activity\n"
    "Clean vs Mild–Moderate vs Moderate–Severe Tremor",
    fontsize=11, fontweight="bold", y=1.01,
)

out_path = Path("Dataset_analyzis/Plot_clean_vs_tremor/sitting_arm_acc_tremor_comparison.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
