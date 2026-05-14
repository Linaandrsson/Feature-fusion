"""
DataGenerator_AWGN_v2.py
========================

Adds Additive White Gaussian Noise (AWGN) on top of already-generated
tremor_clean data (z-scored, with simulated Parkinson's tremor).

Input:
    data/Tremor_datagenerator_files/s4_w4_fs50_tremor_clean/{sensor}.npz
    (X already z-scored, shape (N, C, L))

Noise model (applied in z-scored domain):
    σ_c = ALPHA * std(X[i, c, :])   ← std ≈ 1 since already z-scored
    X_noisy[i, c, :] = X[i, c, :] + N(0, σ_c²)

Output:
    data/Tremor_datagenerator_files/s4_w4_fs50_tremor_clean_awgn_a{TAG}/

Naming:
    ALPHA=0.0   → a000  (sanity check — identical to tremor_clean)
    ALPHA=0.1   → a100  (~20 dB SNR in z-scored domain)
    ALPHA=0.316 → a316  (~10 dB SNR)

All metadata (y, subject_id, tremor_*, splits) copied verbatim from source.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

# ============================================================
# CONFIG
# ============================================================
ALPHA = 0.1    # noise strength: σ = ALPHA * std(channel)
SEED  = 0

SOURCE_VARIANT = "s4_w4_fs50_tremor_clean"

_CODE_DIR = Path(__file__).parent.parent
DATA_BASE = _CODE_DIR / "data" / "Tremor_datagenerator_files"
SOURCE_DIR = DATA_BASE / SOURCE_VARIANT

SENSORS = ["Acc_ankle", "Acc_arm", "Acc_chest", "ECG",
           "Gyro_ankle", "Gyro_arm", "Mag_ankle", "Mag_arm"]

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    alpha_tag    = str(round(ALPHA * 1000)).zfill(3)   # 0.1 → "100"
    variant_name = f"s4_w4_fs50_tremor_clean_awgn_a{alpha_tag}"
    out_dir      = DATA_BASE / variant_name

    print(f"{'='*60}")
    print(f"AWGN ON TREMOR_CLEAN  →  {variant_name}")
    print(f"  ALPHA = {ALPHA}  (tag = a{alpha_tag})")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    for sensor in SENSORS:
        src_npz = SOURCE_DIR / f"{sensor}.npz"
        if not src_npz.exists():
            print(f"  [SKIP] {sensor} — source NPZ not found: {src_npz}")
            continue

        data = np.load(src_npz)
        X    = data["X"].copy().astype(np.float32)   # (N, C, L)
        N, C, L = X.shape

        if ALPHA > 0:
            for i in range(N):
                for c in range(C):
                    sig_std = X[i, c, :].std()
                    if sig_std < 1e-8:
                        sig_std = 1.0
                    X[i, c, :] += rng.normal(0.0, ALPHA * sig_std, size=L).astype(np.float32)

        # Save NPZ — identical metadata, only X replaced
        out_npz = out_dir / f"{sensor}.npz"
        np.savez(
            out_npz,
            X              = X,
            y              = data["y"],
            subject_id     = data["subject_id"],
            base_window_idx= data["base_window_idx"],
            tremor_freq    = data["tremor_freq"],
            tremor_acc_rms = data["tremor_acc_rms"],
            tremor_gyro_rms= data["tremor_gyro_rms"],
            tremor_score   = data["tremor_score"],
            fs             = data["fs"],
            window_len     = data["window_len"],
            stride         = data["stride"],
            sensor_name    = data["sensor_name"],
            sensor_cols    = data["sensor_cols"],
        )

        # Save TXT (CSV) expected by extract_features.py
        # Format: [X flattened (C*L cols)] | [y+1 | subject_id | base_idx | tremor_freq | tremor_acc_rms | tremor_gyro_rms | tremor_score]
        X_flat = X.reshape(N, C * L)          # (N, C*L)
        meta = np.column_stack([
            data["y"].astype(np.float32) + 1,  # 0-indexed → 1-indexed (1..12)
            data["subject_id"].astype(np.float32),
            data["base_window_idx"].astype(np.float32),
            data["tremor_freq"].astype(np.float32),
            data["tremor_acc_rms"].astype(np.float32),
            data["tremor_gyro_rms"].astype(np.float32),
            data["tremor_score"].astype(np.float32),
        ])
        txt_data = np.concatenate([X_flat, meta], axis=1)  # (N, C*L+7)
        np.savetxt(out_dir / f"{sensor}.txt", txt_data, delimiter=",", fmt="%.6f")

        print(f"  ✓ {sensor}: {N} windows, AWGN α={ALPHA}")

    # Copy splits verbatim (same window order as source)
    splits_dst = out_dir / "splits"
    splits_dst.mkdir(exist_ok=True)
    splits_src = SOURCE_DIR / "splits"
    for split_file in ["train_idx.txt", "val_idx.txt", "test_idx.txt"]:
        shutil.copy2(splits_src / split_file, splits_dst / split_file)
    print(f"  ✓ Splits copied from {splits_src}")

    # Write info.txt
    info_lines = [
        f"variant         : {variant_name}",
        f"source          : {SOURCE_VARIANT}",
        f"alpha           : {ALPHA}",
        f"alpha_tag       : a{alpha_tag}",
        f"noise_domain    : z-scored (post z-score AWGN)",
        f"noise_model     : N(0, (alpha * std(channel))^2) per window per channel",
        f"seed            : {SEED}",
        f"sensors         : {', '.join(SENSORS)}",
    ]
    (out_dir / "info.txt").write_text("\n".join(info_lines), encoding="utf-8")
    print(f"  ✓ info.txt written")

    print(f"\nDONE → {out_dir}")
