"""
plot_awgn_vs_clean.py
=====================
Plots clean vs AWGN-corrupted sensor signals side-by-side.

For each sensor: shows 3 random windows with all channels overlaid,
clean (blue) vs AWGN (orange), plus a power spectral density comparison.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent / "data" / "Tremor_datagenerator_files"
CLEAN_DIR = BASE / "s4_w4_fs50_tremor_clean"
AWGN_DIR  = BASE / "s4_w4_fs50_tremor_clean_awgn_a020"
OUT_DIR   = Path(__file__).parent.parent / "Dataset_analyzis" / "Plot_clean_vs_awgn"
FS        = 50          # Hz
N_WINDOWS = 3           # windows to show per sensor
SEED      = 7
SENSORS   = ["Acc_ankle", "Acc_arm", "Acc_chest", "ECG",
             "Gyro_ankle", "Gyro_arm", "Mag_ankle", "Mag_arm"]
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CH_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]   # per channel (clean)
AWGN_COLORS = ["#ff7f0e", "#bcbd22", "#9467bd"]  # per channel (awgn)

def load_sensor(folder: Path, sensor: str):
    return np.load(folder / f"{sensor}.npz")

def channel_label(sensor: str, ch: int) -> str:
    mapping = {
        "Acc":  ["X", "Y", "Z"],
        "Gyro": ["X", "Y", "Z"],
        "Mag":  ["X", "Y", "Z"],
        "ECG":  ["Ch1", "Ch2"],
    }
    for k, v in mapping.items():
        if k in sensor:
            return v[ch] if ch < len(v) else f"Ch{ch}"
    return f"Ch{ch}"

for sensor in SENSORS:
    print(f"  {sensor} ...", end=" ", flush=True)

    c = load_sensor(CLEAN_DIR, sensor)
    a = load_sensor(AWGN_DIR,  sensor)

    X_c = c["X"]   # (N, C, L)
    X_a = a["X"]
    n_ch = X_c.shape[1]
    N    = X_c.shape[0]

    # Pick same N_WINDOWS from test subjects (5, 10) if possible
    test_mask = np.isin(c["subject_id"], [5, 10])
    test_idx  = np.where(test_mask)[0]
    if len(test_idx) < N_WINDOWS:
        test_idx = np.arange(N)
    chosen = rng.choice(test_idx, size=N_WINDOWS, replace=False)

    t = np.arange(X_c.shape[2]) / FS

    # rows = windows, cols = channels + 1 PSD per window
    n_cols = n_ch + 1
    fig, axes = plt.subplots(
        N_WINDOWS, n_cols,
        figsize=(4.5 * n_cols, 3.5 * N_WINDOWS),
        gridspec_kw={"width_ratios": [3] * n_ch + [1.5]},
    )
    if N_WINDOWS == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{sensor}  —  Clean vs AWGN α=0.20 (~14 dB SNR)\n"
        f"(z-score normalised, test subjects 5 & 10)",
        fontsize=13, fontweight="bold"
    )

    for row, w_idx in enumerate(chosen):
        xc = X_c[w_idx]   # (C, L)
        xa = X_a[w_idx]

        act  = int(c["y"][w_idx]) + 1
        subj = int(c["subject_id"][w_idx])

        # ── One subplot per channel ────────────────────────────────
        for ch in range(n_ch):
            ax = axes[row, ch]
            lbl = channel_label(sensor, ch)
            ax.plot(t, xc[ch], color=CH_COLORS[0],    lw=1.2, alpha=0.9,  label="Clean")
            ax.plot(t, xa[ch], color=AWGN_COLORS[0],  lw=1.0, alpha=0.75, linestyle="--", label="AWGN")
            ax.set_title(
                (f"W{w_idx} Subj{subj} Act{act} | {lbl}" if ch == 0
                 else lbl),
                fontsize=8
            )
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("z-score", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        # ── PSD subplot (first channel) ────────────────────────────
        ax_psd = axes[row, n_ch]
        f_c, p_c = welch(xc[0], fs=FS, nperseg=min(128, xc.shape[1]))
        f_a, p_a = welch(xa[0], fs=FS, nperseg=min(128, xa.shape[1]))
        ax_psd.semilogy(f_c, p_c, color=CH_COLORS[0],   lw=1.2, label="Clean")
        ax_psd.semilogy(f_a, p_a, color=AWGN_COLORS[0], lw=1.0,
                        linestyle="--", label="AWGN")
        ax_psd.set_xlabel("Freq (Hz)", fontsize=8)
        ax_psd.set_ylabel("PSD", fontsize=8)
        ax_psd.set_title("PSD (Ch 1)", fontsize=8)
        ax_psd.legend(fontsize=7)
        ax_psd.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = OUT_DIR / f"{sensor}_clean_vs_awgn.png"
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  →  {out_file.name}")

print(f"\nAll plots saved to: {OUT_DIR}")
