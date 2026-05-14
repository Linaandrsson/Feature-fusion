"""
Extract CNN embeddings for ALL 8 corrupt variants, all 8 sensors each.

Runs without modifying config_extraction.py — passes variant_name via
environment variable, which extract_features.py reads if present.

Usage:
    python extract_all_corrupt_variants.py

To skip already-done variants, set SKIP_EXISTING=1:
    SKIP_EXISTING=1 python extract_all_corrupt_variants.py
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Variants to process ────────────────────────────────────────────────────
# CORRUPT_VARIANTS = [
#     # non-AWGN single variants
#     "s4_w4_fs50_corrupt_dropout",
#     "s4_w4_fs50_corrupt_partial_dropout",
#     "s4_w4_fs50_corrupt_weak_signal",
#     "s4_w4_fs50_corrupt_timeshift",
#     "s4_w4_fs50_corrupt_rotation",
#     "s4_w4_fs50_corrupt_all_no_dropout",
#     # AWGN single variants (with SNR dB suffix)
#     "s4_w4_fs50_corrupt_awgn_0db",
#     "s4_w4_fs50_corrupt_awgn_3db",
#     "s4_w4_fs50_corrupt_awgn_6db",
#     "s4_w4_fs50_corrupt_awgn_10db",
#     "s4_w4_fs50_corrupt_awgn_14db",
#     "s4_w4_fs50_corrupt_awgn_15db",
#     "s4_w4_fs50_corrupt_awgn_20db",
#     # AWGN combo variants
#     "s4_w4_fs50_corrupt_awgn_0db_weak",
#     "s4_w4_fs50_corrupt_awgn_0db_timeshift",
#     "s4_w4_fs50_corrupt_awgn_3db_weak",
#     "s4_w4_fs50_corrupt_awgn_3db_timeshift",
#     "s4_w4_fs50_corrupt_awgn_6db_weak",
#     "s4_w4_fs50_corrupt_awgn_6db_timeshift",
#     "s4_w4_fs50_corrupt_awgn_10db_weak",
#     "s4_w4_fs50_corrupt_awgn_10db_timeshift",
#     "s4_w4_fs50_corrupt_awgn_15db_weak",
#     "s4_w4_fs50_corrupt_awgn_15db_timeshift",
#     "s4_w4_fs50_corrupt_awgn_20db_weak",
#     "s4_w4_fs50_corrupt_awgn_20db_timeshift",
# ]

CORRUPT_VARIANTS = ["s4_w4_fs50_corrupt_awgn_20db"]

# ── Model config ────────────────────────────────────────────────────────────
# For clean model:   DATASET_CONFIG="s4_w4_clean",  MODEL_VARIANT="fs50_tremor_clean",  OUTPUT_SUBDIR=None (→ ExtractedFeatures_clean)
# For mixed model:   DATASET_CONFIG="s4_w4_mixed",  MODEL_VARIANT="fs50_tremor_mixed_all3",  OUTPUT_SUBDIR="ExtractedFeatures_mixed"
#mixed:
# DATASET_CONFIG = "s4_w4_mixed"
# MODEL_VARIANT  = "fs50_tremor_mixed_all3"
# OUTPUT_SUBDIR  = "ExtractedFeatures_mixed"   # set to None to use config_extraction default

#clean:
DATASET_CONFIG = "s4_w4_clean"
MODEL_VARIANT  = "fs50_tremor_clean"
OUTPUT_SUBDIR  = "ExtractedFeatures_clean"

SENSORS = [
    "Acc_ankle", "Acc_arm", "Acc_chest", "ECG",
    "Gyro_ankle", "Gyro_arm", "Mag_ankle", "Mag_arm",
]

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir     = Path(__file__).parent.resolve()
workspace_root = script_dir.parents[1]
data_base      = workspace_root / "data" / "Tremor_datagenerator_files"
extract_script = script_dir / "extract_features.py"

skip_existing = os.environ.get("SKIP_EXISTING", "0") == "1"


def variant_done(variant_name: str) -> bool:
    """Return True if all 8 sensor embeddings already exist for this variant."""
    subdir = OUTPUT_SUBDIR or "ExtractedFeatures_clean"
    out_dir = data_base / variant_name / subdir
    return all((out_dir / f"{s}_embeddings.npz").exists() for s in SENSORS)


def run_variant(variant_name: str):
    """Extract all sensors for one variant."""
    print(f"\n{'#'*60}")
    print(f"  Variant: {variant_name}")
    print(f"{'#'*60}")

    if skip_existing and variant_done(variant_name):
        print("  → Already complete, skipping.\n")
        return True, []

    env = os.environ.copy()
    env["EXTRACTION_VARIANT_NAME"]   = variant_name
    env["EXTRACTION_DATASET_CONFIG"] = DATASET_CONFIG
    env["EXTRACTION_MODEL_VARIANT"]  = MODEL_VARIANT
    if OUTPUT_SUBDIR:
        env["EXTRACTION_OUTPUT_SUBDIR"] = OUTPUT_SUBDIR

    success, failed = [], []
    for i, sensor in enumerate(SENSORS, 1):
        print(f"  [{i}/{len(SENSORS)}] {sensor}", end=" ... ", flush=True)
        result = subprocess.run(
            [sys.executable, str(extract_script), "--sensor", sensor],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓")
            success.append(sensor)
        else:
            print("✗")
            # Print last few lines of stderr for quick diagnosis
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
            failed.append(sensor)

    return len(failed) == 0, failed


def main():
    print(f"\n{'='*60}")
    print(f"Batch corrupt-variant extraction")
    print(f"  {len(CORRUPT_VARIANTS)} variants × {len(SENSORS)} sensors")
    print(f"  Models: {DATASET_CONFIG}/{MODEL_VARIANT}")
    print(f"  Output subdir: {OUTPUT_SUBDIR or '(config default)'}")
    print(f"  Skip existing: {skip_existing}")
    print(f"{'='*60}")

    all_results = {}
    for variant in CORRUPT_VARIANTS:
        ok, failed = run_variant(variant)
        all_results[variant] = (ok, failed)

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for variant, (ok, failed) in all_results.items():
        status = "✓" if ok else f"✗ (failed: {', '.join(failed)})"
        print(f"  {status}  {variant}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
