"""
diagnose_awgn_dataset.py

Compares clean vs AWGN-corrupted saved datasets to verify that the
AWGN corruption level is physically correct (i.e. 20 dB really is mild).

Usage:
    python diagnose_awgn_dataset.py
    python diagnose_awgn_dataset.py --awgn_variant s4_w4_fs50_corrupt_awgn_20db
    python diagnose_awgn_dataset.py --awgn_variant s4_w4_fs50_corrupt_awgn_0db

Output:
    tremor_simulation/awgn_diagnostics/
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parents[1]
DATA_BASE      = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"

CLEAN_VARIANT  = "s4_w4_fs50_tremor_clean"

SENSORS = [
    "Acc_ankle",
    "Acc_arm",
    "Acc_chest",
    "ECG",
    "Gyro_ankle",
    "Gyro_arm",
    "Mag_ankle",
    "Mag_arm",
]

# Representative sensors for side-by-side plots
PLOT_SENSORS = ["Acc_ankle", "Gyro_ankle", "Mag_ankle", "ECG"]
N_PLOT_WINDOWS = 4   # windows to plot per sensor


# ── Helpers ───────────────────────────────────────────────────────────────

def load_npz(variant: str, sensor: str) -> dict:
    path = DATA_BASE / variant / f"{sensor}.npz"
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def centered_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - np.mean(x)) ** 2)))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """SNR = 20*log10(RMS_centered_signal / RMS_noise)."""
    sig_rms = centered_rms(signal)
    nse_rms = rms(noise)
    if nse_rms < 1e-12:
        return np.inf
    if sig_rms < 1e-12:
        return -np.inf
    return 20.0 * np.log10(sig_rms / nse_rms)


def snr_per_window_channel(X_clean: np.ndarray,
                            X_awgn:  np.ndarray) -> np.ndarray:
    """
    Returns array of shape (N_windows, C) with per-window per-channel SNR.
    X: shape (N, C, L)
    """
    N, C, L = X_clean.shape
    snrs = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            sig = X_clean[i, c]
            noise = X_awgn[i, c] - X_clean[i, c]
            snrs[i, c] = snr_db(sig, noise)
    return snrs


# ── Sanity checks ─────────────────────────────────────────────────────────

def sanity_checks(clean: dict, awgn: dict, sensor: str) -> list[str]:
    issues = []

    # Shape
    if clean["X"].shape != awgn["X"].shape:
        issues.append(f"SHAPE MISMATCH: clean={clean['X'].shape} awgn={awgn['X'].shape}")

    # Labels
    if not np.array_equal(clean["y"], awgn["y"]):
        n_diff = np.sum(clean["y"] != awgn["y"])
        issues.append(f"LABEL MISMATCH: {n_diff}/{len(clean['y'])} windows differ")

    # Subject IDs
    if not np.array_equal(clean["subject_id"], awgn["subject_id"]):
        n_diff = np.sum(clean["subject_id"] != awgn["subject_id"])
        issues.append(f"SUBJECT_ID MISMATCH: {n_diff} windows differ")

    # Window indices
    if not np.array_equal(clean["base_window_idx"], awgn["base_window_idx"]):
        n_diff = np.sum(clean["base_window_idx"] != awgn["base_window_idx"])
        issues.append(f"BASE_WINDOW_IDX MISMATCH: {n_diff} windows differ")

    # Finiteness
    if not np.all(np.isfinite(awgn["X"])):
        n_bad = np.sum(~np.isfinite(awgn["X"]))
        issues.append(f"NON-FINITE values in AWGN X: {n_bad} entries")

    if not np.all(np.isfinite(clean["X"])):
        n_bad = np.sum(~np.isfinite(clean["X"]))
        issues.append(f"NON-FINITE values in clean X: {n_bad} entries")

    # Suspiciously large values
    for tag, arr in [("clean", clean["X"]), ("awgn", awgn["X"])]:
        absmax = np.abs(arr).max()
        if absmax > 100:
            issues.append(f"SUSPICIOUS large value in {tag} X: max_abs={absmax:.2f}")

    return issues


# ── Signal-level stats ────────────────────────────────────────────────────

def per_channel_stats(X_clean: np.ndarray,
                      X_awgn:  np.ndarray,
                      label: str = "") -> dict:
    """Compute stats across all windows for each channel."""
    N, C, L = X_clean.shape
    stats = {}
    for c in range(C):
        cl = X_clean[:, c, :].ravel()
        aw = X_awgn[:,  c, :].ravel()
        noise = aw - cl
        stats[c] = {
            "mean_clean":  float(np.mean(cl)),
            "std_clean":   float(np.std(cl)),
            "rms_clean":   rms(cl),
            "mean_awgn":   float(np.mean(aw)),
            "std_awgn":    float(np.std(aw)),
            "rms_awgn":    rms(aw),
            "rms_noise":   rms(noise),
            "snr_db_global": snr_db(cl, noise),
        }
    return stats


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_signal_comparison(X_clean, X_awgn, y_clean, sensor, out_dir, n_windows=4):
    """Plot clean vs AWGN side-by-side for a few windows."""
    N, C, L = X_clean.shape
    t = np.arange(L)

    # Pick windows spread across dataset (different activities if possible)
    activities = np.unique(y_clean)
    chosen_idx = []
    for act in activities:
        idxs = np.where(y_clean == act)[0]
        if len(idxs) > 0:
            chosen_idx.append(int(idxs[len(idxs) // 2]))
        if len(chosen_idx) >= n_windows:
            break
    # Pad if needed
    while len(chosen_idx) < n_windows:
        chosen_idx.append(int(np.random.randint(N)))

    fig, axes = plt.subplots(n_windows, C, figsize=(4 * C, 2.5 * n_windows))
    if n_windows == 1:
        axes = axes[np.newaxis, :]
    if C == 1:
        axes = axes[:, np.newaxis]

    for row, win_i in enumerate(chosen_idx[:n_windows]):
        for c in range(C):
            ax = axes[row, c]
            ax.plot(t, X_clean[win_i, c], label="clean",  linewidth=1, alpha=0.8)
            ax.plot(t, X_awgn[win_i,  c], label="awgn",   linewidth=0.8, alpha=0.7, linestyle="--")
            noise = X_awgn[win_i, c] - X_clean[win_i, c]
            snr = snr_db(X_clean[win_i, c], noise)
            ax.set_title(f"win={win_i} act={y_clean[win_i]} ch={c}  SNR={snr:.1f}dB",
                         fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.legend(fontsize=6)

    fig.suptitle(f"{sensor} — clean vs AWGN", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"signal_comparison_{sensor}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  [plot] {out_path.name}")


def plot_snr_histogram(snr_values: np.ndarray, sensor: str, out_dir: Path):
    finite = snr_values[np.isfinite(snr_values)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite, bins=60, edgecolor="none", alpha=0.75)
    ax.axvline(20, color="red", linestyle="--", label="target 20 dB")
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Count (windows × channels)")
    ax.set_title(f"SNR distribution — {sensor}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / f"snr_hist_{sensor}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  [plot] {out_path.name}")


def plot_noise_rms_histogram(X_clean: np.ndarray, X_awgn: np.ndarray,
                              sensor: str, out_dir: Path):
    N, C, L = X_clean.shape
    noise_rms_all = []
    clean_std_all = []
    awgn_std_all  = []
    for i in range(N):
        for c in range(C):
            noise_rms_all.append(rms(X_awgn[i, c] - X_clean[i, c]))
            clean_std_all.append(float(np.std(X_clean[i, c])))
            awgn_std_all.append(float(np.std(X_awgn[i, c])))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(noise_rms_all, bins=60, edgecolor="none", alpha=0.75, color="orange")
    axes[0].set_title("RMS(noise) per window/channel")
    axes[0].set_xlabel("RMS(awgn - clean)")

    axes[1].hist(clean_std_all, bins=60, edgecolor="none", alpha=0.75, label="clean", color="steelblue")
    axes[1].hist(awgn_std_all,  bins=60, edgecolor="none", alpha=0.55, label="awgn",  color="tomato")
    axes[1].set_title("std per window/channel")
    axes[1].set_xlabel("std")
    axes[1].legend(fontsize=8)

    # Ratio noise/signal
    ratio = np.array(noise_rms_all) / (np.array(clean_std_all) + 1e-12)
    axes[2].hist(ratio, bins=60, edgecolor="none", alpha=0.75, color="purple")
    axes[2].set_title("noise_rms / clean_std ratio")
    axes[2].axvline(0.1, color="red", linestyle="--", label="expected 20dB (0.10)")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{sensor} — noise / signal distributions", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"noise_dist_{sensor}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  [plot] {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--awgn_variant", type=str,
                        default="s4_w4_fs50_corrupt_awgn_20db")
    parser.add_argument("--clean_variant", type=str,
                        default=CLEAN_VARIANT)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "awgn_diagnostics"
    out_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("AWGN DATASET DIAGNOSTIC")
    print("=" * 70)
    print(f"  Clean  : {args.clean_variant}")
    print(f"  AWGN   : {args.awgn_variant}")
    print(f"  Output : {out_dir}")
    print("=" * 70)

    all_sensor_stats = {}
    all_sensor_snrs  = {}
    global_issues    = []

    for sensor in SENSORS:
        print(f"\n── {sensor} ─────────────────────────────────────────────")

        try:
            clean = load_npz(args.clean_variant, sensor)
            awgn  = load_npz(args.awgn_variant,  sensor)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            global_issues.append(str(e))
            continue

        # ── Sanity checks ──────────────────────────────────────────────
        issues = sanity_checks(clean, awgn, sensor)
        if issues:
            for iss in issues:
                print(f"  ⚠ {iss}")
                global_issues.append(f"{sensor}: {iss}")
        else:
            print(f"  ✓ Sanity checks passed")

        X_clean = clean["X"]   # (N, C, L)
        X_awgn  = awgn["X"]
        N, C, L = X_clean.shape
        print(f"  Shape: {X_clean.shape}  (N={N}, C={C}, L={L})")

        # ── Per-channel global stats ───────────────────────────────────
        stats = per_channel_stats(X_clean, X_awgn, sensor)
        all_sensor_stats[sensor] = stats
        print(f"  {'ch':>3}  {'mean_cl':>8} {'std_cl':>8} {'rms_cl':>8}"
              f"  {'mean_aw':>8} {'std_aw':>8} {'rms_aw':>8}"
              f"  {'rms_noise':>9} {'SNR_dB':>8}")
        for c, s in stats.items():
            print(f"  {c:>3}  {s['mean_clean']:>8.4f} {s['std_clean']:>8.4f} {s['rms_clean']:>8.4f}"
                  f"  {s['mean_awgn']:>8.4f} {s['std_awgn']:>8.4f} {s['rms_awgn']:>8.4f}"
                  f"  {s['rms_noise']:>9.4f} {s['snr_db_global']:>8.2f}")

        # ── Per-window per-channel SNR ─────────────────────────────────
        snrs = snr_per_window_channel(X_clean, X_awgn)  # (N, C)
        all_sensor_snrs[sensor] = snrs
        finite_snrs = snrs[np.isfinite(snrs)]
        print(f"  SNR (dB) over all windows/channels:")
        print(f"    mean={np.mean(finite_snrs):.2f}  median={np.median(finite_snrs):.2f}"
              f"  min={np.min(finite_snrs):.2f}  max={np.max(finite_snrs):.2f}")
        for thresh in [10, 5, 0]:
            frac = np.mean(finite_snrs < thresh)
            print(f"    frac SNR < {thresh:2d} dB: {frac:.3f}  ({int(frac*len(finite_snrs))} / {len(finite_snrs)})")

        # ── Plots ──────────────────────────────────────────────────────
        if sensor in PLOT_SENSORS:
            plot_signal_comparison(X_clean, X_awgn, clean["y"],
                                   sensor, out_dir, n_windows=N_PLOT_WINDOWS)
            plot_snr_histogram(snrs.ravel(), sensor, out_dir)
            plot_noise_rms_histogram(X_clean, X_awgn, sensor, out_dir)

    # ── Global summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Sensor':<14}  {'Mean SNR':>9}  {'Median SNR':>10}  "
          f"{'frac<10dB':>10}  {'frac<5dB':>9}  {'frac<0dB':>8}")
    print("-" * 70)
    for sensor, snrs in all_sensor_snrs.items():
        finite = snrs[np.isfinite(snrs)]
        if len(finite) == 0:
            print(f"{sensor:<14}  {'N/A':>9}")
            continue
        print(f"{sensor:<14}  {np.mean(finite):>9.2f}  {np.median(finite):>10.2f}  "
              f"{np.mean(finite<10):>10.3f}  {np.mean(finite<5):>9.3f}  "
              f"{np.mean(finite<0):>8.3f}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if global_issues:
        print("\n⚠ ISSUES FOUND:")
        for iss in global_issues:
            print(f"  - {iss}")
    else:
        print("\n✓ No sanity check failures.")

    # Overall SNR estimate
    all_snrs = np.concatenate([s.ravel() for s in all_sensor_snrs.values()])
    finite_all = all_snrs[np.isfinite(all_snrs)]
    mean_snr   = float(np.mean(finite_all)) if len(finite_all) > 0 else np.nan
    target_snr = 20.0  # expected

    print(f"\n  Target AWGN SNR    : {target_snr:.1f} dB")
    print(f"  Observed mean SNR  : {mean_snr:.2f} dB  (over all sensors/windows/channels)")
    print(f"  Observed median SNR: {float(np.median(finite_all)):.2f} dB")

    if mean_snr < target_snr - 3:
        print(f"\n  ❌ AWGN is STRONGER than expected — mean SNR {mean_snr:.1f} dB << {target_snr:.1f} dB target.")
        print(f"     Likely cause: sig_rms is underestimated (DC bias / gravity issue) OR")
        print(f"     the corruption is applied to raw data but z-score removes the signal")
        print(f"     variance, leaving relative noise higher than intended.")
    elif mean_snr > target_snr + 3:
        print(f"\n  ❌ AWGN is WEAKER than expected — mean SNR {mean_snr:.1f} dB >> {target_snr:.1f} dB.")
    else:
        print(f"\n  ✓ Observed SNR ≈ {mean_snr:.1f} dB — within ±3 dB of target {target_snr:.1f} dB.")

    frac_low = float(np.mean(finite_all < 10))
    if frac_low > 0.05:
        print(f"  ⚠ {frac_low:.1%} of window/channel pairs have estimated SNR < 10 dB.")

    print(f"\n  Diagnostic plots saved to: {out_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
