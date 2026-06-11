"""
Plot CT vs PD – arm accelerometer comparison for the same activity.

Uses 'Upper Right' sensor (Cal1–Cal3 = accelerometer) from the Parkinson IMU
dataset.  Compares one control (CT001) and one PD patient (PD002) performing
the same trial (Toast 1), showing the first 10 s of data.

Layout: 3 rows (X, Y, Z) × 2 columns (CT | PD)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE       = Path("Parkinson_dataset_work/Data/Data_parkinson")
ACTIVITY   = "Calibration"
SENSOR     = "Upper Right"          # right-arm IMU
ACC_COLS   = ["Cal1", "Cal2", "Cal3"]   # accelerometer axes
GYRO_COLS  = ["Cal4", "Cal5", "Cal6"]
AXIS_NAMES = ["X", "Y", "Z"]
COLORS_CT  = ["#2166ac", "#5aabe0", "#a6cee3"]   # blue shades
COLORS_PD  = ["#d6604d", "#f4a582", "#fddbc7"]   # red shades

WINDOW_SEC = 10.0   # seconds to display

# ── Helper ────────────────────────────────────────────────────────────────────
def find_file(subj_dir: Path, name: str) -> Path:
    """Case-insensitive file lookup, skipping macOS ._* metadata files."""
    for f in subj_dir.glob("*.xls"):
        if f.name.startswith("._"):
            continue
        if f.stem.lower() == name.lower():
            return f
    raise FileNotFoundError(f"{name} not found in {subj_dir}")


def load_signal(subj: str, activity: str) -> pd.DataFrame:
    subj_dir = BASE / subj
    f = find_file(subj_dir, activity)
    xl = pd.ExcelFile(f)
    df = xl.parse(SENSOR, header=0)
    return df


# ── Load ──────────────────────────────────────────────────────────────────────
ct_df = load_signal("CT001", ACTIVITY)
pd_df = load_signal("PD002", ACTIVITY)

# Infer sampling rate from time column
fs_ct = 1.0 / ct_df["Time"].diff().median()
fs_pd = 1.0 / pd_df["Time"].diff().median()
print(f"CT001 fs ≈ {fs_ct:.1f} Hz  |  PD002 fs ≈ {fs_pd:.1f} Hz")

# Trim to WINDOW_SEC
n_ct = min(len(ct_df), int(WINDOW_SEC * fs_ct))
n_pd = min(len(pd_df), int(WINDOW_SEC * fs_pd))

ct_acc = ct_df[ACC_COLS].iloc[:n_ct].values   # (N, 3)
pd_acc = pd_df[ACC_COLS].iloc[:n_pd].values

t_ct = ct_df["Time"].iloc[:n_ct].values - ct_df["Time"].iloc[0]
t_pd = pd_df["Time"].iloc[:n_pd].values - pd_df["Time"].iloc[0]

# ── Figure – Z-axis only ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(12, 4),
    sharex=True,
)
fig.subplots_adjust(wspace=0.25)

Z_IDX = 2   # index of Z-axis in Cal1/Cal2/Cal3
COLOR_CT = COLORS_CT[Z_IDX]
COLOR_PD = COLORS_PD[Z_IDX]

subjects = [
    ("CT001  (Control)",       t_ct, ct_acc, COLOR_CT),
    ("PD002  (Parkinson's)",   t_pd, pd_acc, COLOR_PD),
]

for col, (title, t, acc, color) in enumerate(subjects):
    ax = axes[col]
    ax.plot(t, acc[:, Z_IDX], color=color, linewidth=0.85, alpha=0.95)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.45)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Z-axis (m/s²)", fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle(
    f"Arm Accelerometer – {ACTIVITY.replace('1','').strip()} Activity  "
    f"(sensor: {SENSOR})\nControl vs Parkinson's Disease  |  First {WINDOW_SEC:.0f} s",
    fontsize=11, fontweight="bold", y=1.01,
)

out_dir = Path("Dataset_analyzis/Plot_clean_vs_tremor")
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "CT_vs_PD_arm_acc_calibration_z.png"
out_pdf = out_dir / "CT_vs_PD_arm_acc_calibration_z.pdf"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved → {out_png}")
plt.show()
