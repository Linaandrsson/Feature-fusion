"""
extract_corrupt_variant.py
===========================

Extracts CNN embeddings for a single corrupt/test data variant across all
8 sensors, by passing the variant and model choice via the environment
variables that extract_features.py already supports:

    EXTRACTION_VARIANT_NAME    – data folder in Tremor_datagenerator_files/
    EXTRACTION_DATASET_CONFIG  – model family  (s4_w4_clean | s4_w4_mixed)
    EXTRACTION_MODEL_VARIANT   – checkpoint folder (fs50_tremor_clean | ...)
    EXTRACTION_OUTPUT_SUBDIR   – embedding folder inside variant dir

This approach requires no changes to extract_features.py or config_extraction.py.

Usage:
    python extract_corrupt_variant.py

Edit the three CONFIG variables below, then run. Re-run with different
VARIANT to extract embeddings for another data folder.
"""

import subprocess
import sys
import os
from pathlib import Path

# ============================================================
# CONFIG — edit these three lines for each extraction run
# ============================================================

# Which data folder to extract FROM (must exist in Tremor_datagenerator_files/)
VARIANT = "s4_w4_fs50_tremor_clean_awgn"

# Which model to use: "clean" or "mixed"
MODEL = "clean"   # → s4_w4_clean / fs50_tremor_clean / ExtractedFeatures_clean
# MODEL = "mixed" # → s4_w4_mixed / fs50_tremor_mixed_all3 / ExtractedFeatures_mixed

# ============================================================
# Derived — do not change
# ============================================================
_MODEL_LOOKUP = {
    "clean": ("s4_w4_clean", "fs50_tremor_clean",       "ExtractedFeatures_clean"),
    "mixed": ("s4_w4_mixed", "fs50_tremor_mixed_all3",  "ExtractedFeatures_mixed"),
}

_script_dir     = Path(__file__).parent.resolve()
_workspace_root = _script_dir.parents[1]          # Code/
_data_base      = _workspace_root / "data" / "Tremor_datagenerator_files"
_extract_script = _script_dir / "extract_features.py"

SENSORS = [
    "Acc_ankle", "Acc_arm", "Acc_chest", "ECG",
    "Gyro_ankle", "Gyro_arm", "Mag_ankle", "Mag_arm",
]


def main():
    if MODEL not in _MODEL_LOOKUP:
        print(f"Unknown MODEL '{MODEL}'. Choose 'clean' or 'mixed'.")
        sys.exit(1)

    dataset_config, model_variant, output_subdir = _MODEL_LOOKUP[MODEL]

    # Verify data folder exists
    variant_path = _data_base / VARIANT
    if not variant_path.exists():
        print(f"Data folder not found: {variant_path}")
        print("Generate it first (e.g. python tremor_simulation/DataGenerator_CorruptAWGN.py)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Corrupt variant extraction")
    print(f"  Variant : {VARIANT}")
    print(f"  Model   : {dataset_config}/{model_variant}")
    print(f"  Output  : {VARIANT}/{output_subdir}/")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["EXTRACTION_VARIANT_NAME"]   = VARIANT
    env["EXTRACTION_DATASET_CONFIG"] = dataset_config
    env["EXTRACTION_MODEL_VARIANT"]  = model_variant
    env["EXTRACTION_OUTPUT_SUBDIR"]  = output_subdir

    success, failed = [], []
    for i, sensor in enumerate(SENSORS, 1):
        print(f"[{i}/{len(SENSORS)}] {sensor}", end=" ... ", flush=True)
        result = subprocess.run(
            [sys.executable, str(_extract_script), "--sensor", sensor],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓")
            success.append(sensor)
        else:
            print("✗")
            for line in (result.stderr or result.stdout).strip().splitlines()[-5:]:
                print(f"    {line}")
            failed.append(sensor)

    print(f"\n{'='*60}")
    print(f"Done: {len(success)}/{len(SENSORS)} sensors succeeded")
    if failed:
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        out_dir = variant_path / output_subdir
        print(f"Embeddings saved to: {out_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
