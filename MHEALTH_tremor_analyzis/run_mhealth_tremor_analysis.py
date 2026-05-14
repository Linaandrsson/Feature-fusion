"""
MHEALTH Simulated Tremor Analysis
===================================

Replicates the BioStamp activity signal profile analysis on simulated
MHEALTH tremor data, enabling direct comparison between:
  - Real PD tremor (BioStamp dataset)
  - Simulated tremor (MHEALTH dataset)

Signal used: Acc_arm_tremorbranch  (raw, non-normalized)

Outputs (written to MHEALTH_tremor_analyzis/):
  mhealth_activity_signal_profile.csv
  mhealth_activity_signal_profile_ranked_by_ratio.csv
  mhealth_activity_signal_profile.txt

Usage:
    cd into Code/ and run:
        python MHEALTH_tremor_analyzis/run_mhealth_tremor_analysis.py

    Or run from within MHEALTH_tremor_analyzis/:
        python run_mhealth_tremor_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal

# ============================================================
# Paths & Configuration
# ============================================================

_HERE = Path(__file__).parent.resolve()
_CODE_ROOT = _HERE.parent
_TREMOR_DATA_ROOT = _CODE_ROOT / "data" / "Tremor_datagenerator_files"

# Dataset variant to analyse (default: mod_severe at 50 Hz)
DATASET_VARIANT = "s2_w2_fs50_tremor_mod_severe"
SENSOR_FILE = "Acc_arm_tremorbranch.npz"   # used for the existing global output

# All tremor-branch sensor files available in the dataset.
# Keys are filenames; values are the short label used in output filenames.
SENSOR_FILES = {
    "Acc_arm_tremorbranch.npz":  "arm_acc",
    "Gyro_arm_tremorbranch.npz": "arm_gyro",
}

OUTPUT_DIR = _HERE   # outputs written alongside this script

# ============================================================
# Signal-processing constants  (matching BioStamp)
# ============================================================

TREMOR_BAND_HZ   = (3.0, 7.0)    # tremor characterisation band
ANALYSIS_BAND_HZ = (2.0, 8.0)    # broader band for dominant-frequency search

# Detection threshold on tremor-band RMS.
# MHEALTH accelerometer data is in m/s²; BioStamp was in g (1 g ≈ 9.81 m/s²).
# BioStamp used 0.05 g ≈ 0.49 m/s².  We use a slightly more sensitive value
# because the new SCORE_RMS_RANGE starts at 0.16 m/s² (score 1).
TREMOR_RMS_THRESHOLD = 0.10      # m/s²

# ============================================================
# MHEALTH activity label mapping
# ============================================================
# y in NPZ is 0-indexed  (0 = L1, 1 = L2, … 11 = L12).
# We add 1 before look-up so the keys below match MHEALTH codes.

ACTIVITY_LABELS = {
    1:  "Standing still",
    2:  "Sitting and relaxing",
    3:  "Lying down",
    4:  "Walking",
    5:  "Climbing stairs",
    6:  "Waist bends forward",
    7:  "Frontal elevation of arms",
    8:  "Knees bending (crouching)",
    9:  "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back",
}


# ============================================================
# Feature extraction helpers  (identical logic to BioStamp)
# ============================================================

def _magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt(x.astype(np.float64)**2 +
                   y.astype(np.float64)**2 +
                   z.astype(np.float64)**2)


def _bandpass_rms(sig: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    nyq = fs / 2.0
    wl = max(1e-4, min(low_hz  / nyq, 0.9999))
    wh = max(wl + 1e-4, min(high_hz / nyq, 0.9999))
    try:
        b, a = scipy_signal.butter(4, [wl, wh], btype="band")
        filtered = scipy_signal.filtfilt(b, a, sig)
        return float(np.sqrt(np.mean(filtered ** 2)))
    except Exception:
        return 0.0


def _compute_psd(sig: np.ndarray, fs: float):
    nperseg = min(len(sig), max(32, len(sig) // 4))
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg,
                                    window="hann", scaling="density")
    return freqs, psd


def _dominant_freq(freqs: np.ndarray, psd: np.ndarray,
                   low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(psd[mask])])


def compute_window_features(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             fs: float) -> dict:
    """
    Compute tremor features for one 3-axis window.

    Args:
        x, y, z : 1-D float arrays of equal length (one axis each)
        fs       : sampling frequency (Hz)

    Returns a dict with:
        mag_rms, tremor_rms, tremor_band_ratio, dom_freq_hz, tremor_present
    """
    mag = _magnitude(x, y, z)

    mag_rms = float(np.sqrt(np.mean(mag ** 2)))

    tr_rms = _bandpass_rms(mag, fs,
                           TREMOR_BAND_HZ[0], TREMOR_BAND_HZ[1])

    freqs, psd = _compute_psd(mag, fs)
    dom_f = _dominant_freq(freqs, psd,
                           ANALYSIS_BAND_HZ[0], ANALYSIS_BAND_HZ[1])

    ratio = tr_rms / mag_rms if mag_rms > 1e-12 else np.nan

    return {
        "mag_rms":           mag_rms,
        "tremor_rms":        tr_rms,
        "tremor_band_ratio": ratio,
        "dom_freq_hz":       dom_f,
        "tremor_present":    bool(tr_rms >= TREMOR_RMS_THRESHOLD),
    }


# ============================================================
# Data loading
# ============================================================

def load_tremor_branch(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load Acc_arm_tremorbranch NPZ.

    Returns:
        X          : (N, 3, L) float32  — raw (non-normalised) windows
        y_activity : (N,)  int          — 1-indexed activity labels (L1–L12)
        fs         : float              — sampling frequency
    """
    d = np.load(npz_path)
    X = d["X"].astype(np.float64)         # (N, 3, L)
    y_raw = d["y"].astype(int)            # 0-indexed
    fs = float(d["fs"])
    y_activity = y_raw + 1               # convert to 1-indexed (L1=1 … L12=12)
    return X, y_activity, fs


# ============================================================
# Per-window feature table
# ============================================================

def build_feature_table(X: np.ndarray, y_activity: np.ndarray,
                         fs: float) -> pd.DataFrame:
    """
    Compute features for every window and return a DataFrame.

    Columns: activity, activity_label, mag_rms, tremor_rms,
             tremor_band_ratio, dom_freq_hz, tremor_present
    """
    rows = []
    N = X.shape[0]
    for i in range(N):
        feat = compute_window_features(
            X[i, 0, :], X[i, 1, :], X[i, 2, :], fs
        )
        act = int(y_activity[i])
        feat["activity"]       = act
        feat["activity_label"] = ACTIVITY_LABELS.get(act, f"L{act}")
        rows.append(feat)

    return pd.DataFrame(rows)


# ============================================================
# Activity-level aggregation  (matches BioStamp activity_signal_profile)
# ============================================================

def _pct(series: pd.Series, p: int) -> float:
    s = series.dropna()
    return float(np.percentile(s, p)) if len(s) > 0 else np.nan


def activity_signal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate features per activity.

    Returns DataFrame sorted descending by tremor_band_rms_median.
    Column names match BioStamp output exactly.
    """
    rows = []
    for act, grp in df.groupby("activity"):
        n_total  = len(grp)
        n_tremor = int(grp["tremor_present"].sum())

        row = {
            "activity":                  act,
            "activity_label":            ACTIVITY_LABELS.get(act, f"L{act}"),
            "n_total":                   n_total,
            "n_tremor":                  n_tremor,
            "tremor_prevalence_percent": round(n_tremor / n_total * 100, 1)
                                         if n_total > 0 else np.nan,
            # Total (magnitude) RMS
            "total_rms_median":          _pct(grp["mag_rms"], 50),
            "total_rms_p25":             _pct(grp["mag_rms"], 25),
            "total_rms_p75":             _pct(grp["mag_rms"], 75),
            # Tremor-band RMS
            "tremor_band_rms_median":    _pct(grp["tremor_rms"], 50),
            "tremor_band_rms_p25":       _pct(grp["tremor_rms"], 25),
            "tremor_band_rms_p75":       _pct(grp["tremor_rms"], 75),
            # Tremor-band ratio
            "tremor_band_ratio_median":  _pct(grp["tremor_band_ratio"], 50),
            "tremor_band_ratio_p25":     _pct(grp["tremor_band_ratio"], 25),
            "tremor_band_ratio_p75":     _pct(grp["tremor_band_ratio"], 75),
            # Dominant frequency
            "dominant_freq_median":      _pct(grp["dom_freq_hz"], 50),
            "dominant_freq_p25":         _pct(grp["dom_freq_hz"], 25),
            "dominant_freq_p75":         _pct(grp["dom_freq_hz"], 75),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("tremor_band_rms_median",
                          ascending=False).reset_index(drop=True)
    return out


def activity_signal_profile_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate features per activity for clean (no-tremor) data.

    No threshold is applied. tremor_present / tremor_prevalence_percent
    are excluded entirely. Only RMS-based and frequency metrics are kept.
    """
    rows = []
    for act, grp in df.groupby("activity"):
        row = {
            "activity":                 act,
            "activity_label":           ACTIVITY_LABELS.get(act, f"L{act}"),
            "n_total":                  len(grp),
            # Total (magnitude) RMS
            "total_rms_median":         _pct(grp["mag_rms"], 50),
            "total_rms_p25":            _pct(grp["mag_rms"], 25),
            "total_rms_p75":            _pct(grp["mag_rms"], 75),
            # Tremor-band RMS
            "tremor_band_rms_median":   _pct(grp["tremor_rms"], 50),
            "tremor_band_rms_p25":      _pct(grp["tremor_rms"], 25),
            "tremor_band_rms_p75":      _pct(grp["tremor_rms"], 75),
            # Tremor-band ratio
            "tremor_band_ratio_median": _pct(grp["tremor_band_ratio"], 50),
            "tremor_band_ratio_p25":    _pct(grp["tremor_band_ratio"], 25),
            "tremor_band_ratio_p75":    _pct(grp["tremor_band_ratio"], 75),
            # Dominant frequency
            "dominant_freq_median":     _pct(grp["dom_freq_hz"], 50),
            "dominant_freq_p25":        _pct(grp["dom_freq_hz"], 25),
            "dominant_freq_p75":        _pct(grp["dom_freq_hz"], 75),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("tremor_band_rms_median",
                          ascending=False).reset_index(drop=True)
    return out


# ============================================================
# Plain-text report  (matches BioStamp format_signal_profile_report)
# ============================================================

def format_report(profile_df: pd.DataFrame,
                  dataset_variant: str,
                  fs: float,
                  n_windows: int,
                  window_sec: float,
                  sensor_file: str = None) -> str:

    header = (
        f"{'Activity':<30}  {'N':>5}  {'Prev%':>6}  "
        f"{'TotRMS_p50':>10}  {'TrRMS_p50':>10}  "
        f"{'Ratio_p50':>10}  {'Freq_p50':>9}"
    )
    divider = "  " + "-" * (len(header) + 2)

    def _f(v, fmt=".4f"):
        return f"{v:{fmt}}" if (v is not None and not pd.isna(v)) else "     —"

    def _row(r):
        return (
            f"  {str(r['activity_label']):<30}  {int(r['n_total']):>5}  "
            f"{_f(r['tremor_prevalence_percent'], '.1f'):>6}  "
            f"{_f(r['total_rms_median']):>10}  "
            f"{_f(r['tremor_band_rms_median']):>10}  "
            f"{_f(r['tremor_band_ratio_median']):>10}  "
            f"{_f(r['dominant_freq_median'], '.2f'):>9}"
        )

    lines = [
        "",
        "=" * 78,
        "MHEALTH SIMULATED TREMOR — ACTIVITY SIGNAL PROFILE",
        "=" * 78,
        f"  Dataset variant : {dataset_variant}",
        f"  Signal          : {Path(sensor_file or SENSOR_FILE).stem}  (raw, non-normalised)",
        f"  Sampling rate   : {fs:.1f} Hz",
        f"  Window size     : {window_sec:.1f} s  ({int(window_sec * fs)} samples, no overlap)",
        f"  Total windows   : {n_windows}",
        f"  Tremor band     : {TREMOR_BAND_HZ[0]}–{TREMOR_BAND_HZ[1]} Hz  |  "
        f"Detection threshold : {TREMOR_RMS_THRESHOLD:.3f} m/s²",
        "  tremor_band_ratio = tremor_band_rms / total_mag_rms  (per window)",
        "",
        "  1. Ranked by tremor-band RMS (descending)",
        "",
        "  " + header,
        divider,
    ]
    for _, r in profile_df.iterrows():
        lines.append(_row(r))

    ratio_ranked = profile_df.sort_values(
        "tremor_band_ratio_median", ascending=False, na_position="last"
    )
    lines += [
        "",
        "  2. Ranked by tremor-band ratio (descending)",
        "",
        "  " + header,
        divider,
    ]
    for _, r in ratio_ranked.iterrows():
        lines.append(_row(r))

    lines.append("")
    return "\n".join(lines)


def format_report_clean(profile_df: pd.DataFrame,
                        dataset_variant: str,
                        fs: float,
                        n_windows: int,
                        window_sec: float,
                        sensor_file: str = None) -> str:
    """Plain-text report for clean-signal data — no threshold, no Prev% column."""
    header = (
        f"{'Activity':<30}  {'N':>5}  "
        f"{'TotRMS_p50':>10}  {'TrRMS_p50':>10}  "
        f"{'Ratio_p50':>10}  {'Freq_p50':>9}"
    )
    divider = "  " + "-" * (len(header) + 2)

    def _f(v, fmt=".4f"):
        return f"{v:{fmt}}" if (v is not None and not pd.isna(v)) else "     —"

    def _row(r):
        return (
            f"  {str(r['activity_label']):<30}  {int(r['n_total']):>5}  "
            f"{_f(r['total_rms_median']):>10}  "
            f"{_f(r['tremor_band_rms_median']):>10}  "
            f"{_f(r['tremor_band_ratio_median']):>10}  "
            f"{_f(r['dominant_freq_median'], '.2f'):>9}"
        )

    lines = [
        "",
        "=" * 78,
        "MHEALTH CLEAN SIGNAL — ACTIVITY SIGNAL PROFILE",
        "=" * 78,
        f"  Dataset variant : {dataset_variant}",
        f"  Signal          : {Path(sensor_file or SENSOR_FILE).stem}  (raw, non-normalised)",
        f"  Sampling rate   : {fs:.1f} Hz",
        f"  Window size     : {window_sec:.1f} s  ({int(window_sec * fs)} samples, no overlap)",
        f"  Total windows   : {n_windows}",
        f"  Tremor band     : {TREMOR_BAND_HZ[0]}–{TREMOR_BAND_HZ[1]} Hz  (no detection threshold)",
        "  tremor_band_ratio = tremor_band_rms / total_mag_rms  (per window)",
        "",
        "  1. Ranked by tremor-band RMS (descending)",
        "",
        "  " + header,
        divider,
    ]
    for _, r in profile_df.iterrows():
        lines.append(_row(r))

    ratio_ranked = profile_df.sort_values(
        "tremor_band_ratio_median", ascending=False, na_position="last"
    )
    lines += [
        "",
        "  2. Ranked by tremor-band ratio (descending)",
        "",
        "  " + header,
        divider,
    ]
    for _, r in ratio_ranked.iterrows():
        lines.append(_row(r))

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    npz_path = _TREMOR_DATA_ROOT / DATASET_VARIANT / SENSOR_FILE
    if not npz_path.exists():
        sys.exit(
            f"ERROR: Data file not found:\n  {npz_path}\n"
            f"Check that DATASET_VARIANT='{DATASET_VARIANT}' is correct."
        )

    print(f"[1/4] Loading  {npz_path.relative_to(_CODE_ROOT)} ...")
    X, y_activity, fs = load_tremor_branch(npz_path)
    N, C, L = X.shape
    window_sec = L / fs
    print(f"      Shape: {X.shape}  |  fs={fs} Hz  |  window={window_sec:.1f} s  "
          f"|  activities: {sorted(set(y_activity.tolist()))}")

    print(f"[2/4] Computing features for {N} windows ...")
    df = build_feature_table(X, y_activity, fs)
    print(f"      tremor_present: {df['tremor_present'].sum()} / {N} "
          f"({df['tremor_present'].mean()*100:.1f}%)")

    print("[3/4] Aggregating per activity ...")
    profile = activity_signal_profile(df)

    print("[4/4] Saving outputs ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Primary CSV (sorted by tremor_band_rms_median desc)
    out_csv = OUTPUT_DIR / "mhealth_activity_signal_profile.csv"
    profile.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"      Saved: {out_csv.name}")

    # Ratio-ranked CSV
    ratio_ranked = profile.sort_values(
        "tremor_band_ratio_median", ascending=False, na_position="last"
    ).reset_index(drop=True)
    out_ratio = OUTPUT_DIR / "mhealth_activity_signal_profile_ranked_by_ratio.csv"
    ratio_ranked.to_csv(out_ratio, index=False, float_format="%.6f")
    print(f"      Saved: {out_ratio.name}")

    # Text report
    report = format_report(profile, DATASET_VARIANT, fs, N, window_sec)
    out_txt = OUTPUT_DIR / "mhealth_activity_signal_profile.txt"
    out_txt.write_text(report, encoding="utf-8")
    print(f"      Saved: {out_txt.name}")

    # Print report to console
    print(report)

    # ------------------------------------------------------------------
    # Per-sensor outputs
    # ------------------------------------------------------------------
    print("[5/5] Saving per-sensor outputs ...")
    for sensor_filename, sensor_label in SENSOR_FILES.items():
        s_path = _TREMOR_DATA_ROOT / DATASET_VARIANT / sensor_filename
        if not s_path.exists():
            print(f"      WARNING: {sensor_filename} not found — skipping.")
            continue

        print(f"      Loading {sensor_filename} ...")
        X_s, y_s, fs_s = load_tremor_branch(s_path)
        N_s = X_s.shape[0]
        window_sec_s = X_s.shape[2] / fs_s

        df_s = build_feature_table(X_s, y_s, fs_s)
        df_s["sensor"] = sensor_label
        profile_s = activity_signal_profile(df_s)

        # CSV
        csv_s = OUTPUT_DIR / f"mhealth_activity_signal_profile_{sensor_label}.csv"
        profile_s.to_csv(csv_s, index=False, float_format="%.6f")

        # Text report
        report_s = format_report(
            profile_s, DATASET_VARIANT, fs_s, N_s, window_sec_s,
            sensor_file=sensor_filename
        )
        txt_s = OUTPUT_DIR / f"mhealth_activity_signal_profile_{sensor_label}.txt"
        txt_s.write_text(report_s, encoding="utf-8")

        print(f"      Saved: {csv_s.name}  |  {txt_s.name}  "
              f"({N_s} windows, {df_s['tremor_present'].sum()} tremor)")

    # ------------------------------------------------------------------
    # Clean-signal per-sensor outputs  (s2_w2_fs50_tremor_clean)
    # ------------------------------------------------------------------
    CLEAN_VARIANT = "s2_w2_fs50_tremor_clean"
    print(f"\n[6/6] Saving clean-signal per-sensor outputs ({CLEAN_VARIANT}) ...")
    for sensor_filename, sensor_label in SENSOR_FILES.items():
        c_path = _TREMOR_DATA_ROOT / CLEAN_VARIANT / sensor_filename
        if not c_path.exists():
            print(f"      WARNING: {sensor_filename} not found in {CLEAN_VARIANT} — skipping.")
            continue

        print(f"      Loading {sensor_filename} (clean) ...")
        X_c, y_c, fs_c = load_tremor_branch(c_path)
        N_c = X_c.shape[0]
        window_sec_c = X_c.shape[2] / fs_c

        df_c = build_feature_table(X_c, y_c, fs_c)
        df_c["sensor"] = sensor_label
        profile_c = activity_signal_profile_clean(df_c)

        # CSV  (no tremor_present / tremor_prevalence_percent columns)
        csv_c = OUTPUT_DIR / f"mhealth_activity_signal_profile_clean_{sensor_label}.csv"
        profile_c.to_csv(csv_c, index=False, float_format="%.6f")

        # Text report  (no Prev% column, no threshold note)
        report_c = format_report_clean(
            profile_c, CLEAN_VARIANT, fs_c, N_c, window_sec_c,
            sensor_file=sensor_filename
        )
        txt_c = OUTPUT_DIR / f"mhealth_activity_signal_profile_clean_{sensor_label}.txt"
        txt_c.write_text(report_c, encoding="utf-8")

        print(f"      Saved: {csv_c.name}  |  {txt_c.name}  ({N_c} windows)")

    print("Done.")


if __name__ == "__main__":
    main()
