"""
plot_awgn_data.py
=================
Visualise AWGN-corrupted data vs clean data for a given sensor.

Plots one row per selected activity (N_ACTIVITIES rows), showing:
  Left  : clean z-scored signal
  Right : AWGN (a100, α=0.1) z-scored signal

Output saved to:
    data/Tremor_datagenerator_files/s4_w4_fs50_awgn_a100/plots/
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SENSOR        = "Acc_ankle"      # which sensor to plot
N_PER_ACTIVITY = 1               # how many windows per activity to show
ACTIVITIES    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   # 0-indexed, 0-11
CHANNEL_LABELS = ["X", "Y", "Z"]   # axis labels (3-axis sensors)

_CODE_DIR = Path(__file__).parent.parent
CLEAN_NPZ = _CODE_DIR / "data" / "Tremor_datagenerator_files" / "s4_w4_fs50_tremor_clean" / f"{SENSOR}.npz"
AWGN_VARIANT = "s4_w4_fs50_awgn_a000"   # change to "s4_w4_fs50_awgn_a100" for α=0.1
AWGN_NPZ  = _CODE_DIR / "data" / "Tremor_datagenerator_files" / AWGN_VARIANT / f"{SENSOR}.npz"
OUT_DIR   = _CODE_DIR / "data" / "Tremor_datagenerator_files" / AWGN_VARIANT / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_NAMES = {
    0: "Standing still",
    1: "Sitting / relaxing",
    2: "Lying down",
    3: "Walking",
    4: "Climbing stairs",
    5: "Waist bends forward",
    6: "Frontal elevation arms",
    7: "Knees bending (crouching)",
    8: "Cycling",
    9: "Jogging",
    10: "Running",
    11: "Jump front & back",
}

# ── Load ──────────────────────────────────────────────────────────────────────
clean = np.load(CLEAN_NPZ)
awgn  = np.load(AWGN_NPZ)

X_clean  = clean["X"]   # (N, C, L)
X_awgn   = awgn["X"]
y        = clean["y"]   # same labels for both (verified identical)
n_chan   = X_clean.shape[1]
fs       = int(clean["fs"])
t        = np.arange(X_clean.shape[2]) / fs   # time axis in seconds

# ── Collect windows ───────────────────────────────────────────────────────────
selected_acts = [a for a in ACTIVITIES if np.any(y == a)]
win_per_act   = {}
for act in selected_acts:
    idxs = np.where(y == act)[0]
    win_per_act[act] = idxs[:N_PER_ACTIVITY]

n_rows = len(selected_acts) * N_PER_ACTIVITY

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_chan * 2,           # left columns = clean, right columns = AWGN
    figsize=(5 * n_chan * 2, 2.2 * n_rows),
    squeeze=False,
)

col_titles_done = False
row = 0
for act in selected_acts:
    for win_idx in win_per_act[act]:
        xc = X_clean[win_idx]   # (C, L)
        xa = X_awgn[win_idx]    # (C, L)

        for c in range(n_chan):
            # --- Clean ---
            ax_c = axes[row][c]
            ax_c.plot(t, xc[c], color="#2196F3", lw=0.9)
            ax_c.set_xlim(t[0], t[-1])
            if not col_titles_done:
                ax_c.set_title(f"Clean — ch {CHANNEL_LABELS[c] if c < 3 else c}", fontsize=9)
            ax_c.tick_params(labelsize=7)
            if c == 0:
                label = f"Act {act+1}\n{ACTIVITY_NAMES.get(act, '')}"
                ax_c.set_ylabel(label, fontsize=7, rotation=90, labelpad=4)

            # --- AWGN ---
            ax_a = axes[row][n_chan + c]
            ax_a.plot(t, xa[c], color="#F44336", lw=0.9, alpha=0.85)
            ax_a.set_xlim(t[0], t[-1])
            if not col_titles_done:
                ax_a.set_title(f"AWGN α=0.1 — ch {CHANNEL_LABELS[c] if c < 3 else c}", fontsize=9)
            ax_a.tick_params(labelsize=7)

        col_titles_done = True
        row += 1

# Add a vertical divider label
fig.text(0.25, 1.002, "CLEAN (z-scored)", ha="center", fontsize=11, color="#2196F3", fontweight="bold")
fig.text(0.75, 1.002, "AWGN α=0.1 (z-scored)", ha="center", fontsize=11, color="#F44336", fontweight="bold")

fig.suptitle(f"Sensor: {SENSOR} — Clean vs AWGN, all 12 activities", fontsize=12, y=1.008)
plt.tight_layout()

out_file = OUT_DIR / f"clean_vs_awgn_{SENSOR}.png"
fig.savefig(out_file, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {out_file}")

# ── Also plot overlay for a single activity to show noise magnitude ──────────
OVERLAY_ACT = 3   # Walking (0-indexed)
if np.any(y == OVERLAY_ACT):
    idxs = np.where(y == OVERLAY_ACT)[0][:4]
    fig2, axes2 = plt.subplots(1, n_chan, figsize=(5 * n_chan, 3.5))
    if n_chan == 1:
        axes2 = [axes2]
    for c in range(n_chan):
        ax = axes2[c]
        for i, wi in enumerate(idxs):
            alpha = 0.9 if i == 0 else 0.4
            ax.plot(t, X_clean[wi, c], color="#2196F3", lw=1.2 if i == 0 else 0.7,
                    alpha=alpha, label="Clean" if i == 0 else None)
            ax.plot(t, X_awgn[wi, c],  color="#F44336", lw=1.2 if i == 0 else 0.7,
                    alpha=alpha, label="AWGN α=0.1" if i == 0 else None)
        ax.set_title(f"ch {CHANNEL_LABELS[c] if c < 3 else c}", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.legend(fontsize=8)
    fig2.suptitle(
        f"{SENSOR} — {ACTIVITY_NAMES.get(OVERLAY_ACT, f'Act {OVERLAY_ACT+1}')} (4 windows overlay)",
        fontsize=11,
    )
    plt.tight_layout()
    out_file2 = OUT_DIR / f"overlay_awgn_{SENSOR}_act{OVERLAY_ACT+1}.png"
    fig2.savefig(out_file2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Saved: {out_file2}")
