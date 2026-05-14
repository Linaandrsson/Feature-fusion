"""
check_splits_consistency.py

Sanity check that verifies subject-based train/val/test splits are consistent
across all three training scenarios (clean, clean_aug, mixed) and across all
8 sensors within each scenario.

Checks:
  1. Each NPZ file contains subject IDs that match the expected split.
  2. Train/val/test subject sets are identical across all scenarios and sensors.
  3. No subject leakage between splits within any NPZ file.
  4. The subject sets in NPZ embeddings match the fusion ablation script constants.

Usage:
    python check_splits_consistency.py
"""

from pathlib import Path
import numpy as np
import sys

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — must match fusion_concat_ablation_study_tremor.py
# ═══════════════════════════════════════════════════════════════

EXPECTED_TEST  = {5, 10}
EXPECTED_VAL   = {2, 7}
EXPECTED_TRAIN = {1, 3, 4, 6, 8, 9}

DATA_DIR = Path(__file__).parents[2] / "data" / "Tremor_datagenerator_files"
EMBEDDINGS_FOLDER = "Activity_ExtractedFeatures"

SCENARIOS = {
    "clean":     ["s4_w4_fs50_tremor_clean"],
    "clean_aug": ["s4_w4_fs50_tremor_clean",
                  "s4_w4_fs50_tremor_clean_awgn",
                  "s4_w4_fs50_tremor_clean_rotation"],
    "mixed":     ["s4_w4_fs50_tremor_clean",
                  "s4_w4_fs50_tremor_mild_mod",
                  "s4_w4_fs50_tremor_mod_severe"],
}

SENSORS = [
    "Acc_ankle", "Acc_arm", "Acc_chest",
    "ECG",
    "Gyro_ankle", "Gyro_arm",
    "Mag_ankle", "Mag_arm",
]

# ═══════════════════════════════════════════════════════════════

def check_npz(npz_path: Path, scenario: str, variant: str, sensor: str) -> list[str]:
    """Load one NPZ and return list of error strings (empty = OK)."""
    errors = []
    label = f"{scenario}/{variant}/{sensor}"

    if not npz_path.exists():
        errors.append(f"MISSING  {label} — {npz_path}")
        return errors

    data = np.load(npz_path)

    required_keys = [
        "train_embeddings", "val_embeddings", "test_embeddings",
        "train_subjects",   "val_subjects",   "test_subjects",
    ]
    for k in required_keys:
        if k not in data:
            errors.append(f"MISSING KEY '{k}'  {label}")
    if errors:
        return errors

    train_subj = set(data["train_subjects"].astype(int).tolist())
    val_subj   = set(data["val_subjects"].astype(int).tolist())
    test_subj  = set(data["test_subjects"].astype(int).tolist())

    # Check expected subjects
    if train_subj != EXPECTED_TRAIN:
        errors.append(f"TRAIN SUBJECTS  {label}: got {sorted(train_subj)}, expected {sorted(EXPECTED_TRAIN)}")
    if val_subj != EXPECTED_VAL:
        errors.append(f"VAL SUBJECTS    {label}: got {sorted(val_subj)}, expected {sorted(EXPECTED_VAL)}")
    if test_subj != EXPECTED_TEST:
        errors.append(f"TEST SUBJECTS   {label}: got {sorted(test_subj)}, expected {sorted(EXPECTED_TEST)}")

    # Check no leakage between splits
    if train_subj & val_subj:
        errors.append(f"LEAKAGE train∩val  {label}: {sorted(train_subj & val_subj)}")
    if train_subj & test_subj:
        errors.append(f"LEAKAGE train∩test {label}: {sorted(train_subj & test_subj)}")
    if val_subj & test_subj:
        errors.append(f"LEAKAGE val∩test   {label}: {sorted(val_subj & test_subj)}")

    return errors


def main():
    print("=" * 65)
    print("  SPLIT CONSISTENCY SANITY CHECK")
    print(f"  Expected TRAIN subjects : {sorted(EXPECTED_TRAIN)}")
    print(f"  Expected VAL   subjects : {sorted(EXPECTED_VAL)}")
    print(f"  Expected TEST  subjects : {sorted(EXPECTED_TEST)}")
    print("=" * 65)

    all_errors = []
    total_checked = 0

    for scenario, variants in SCENARIOS.items():
        print(f"\n[{scenario}]")
        for variant in variants:
            variant_dir = DATA_DIR / variant / EMBEDDINGS_FOLDER
            if not variant_dir.exists():
                msg = f"  MISSING variant folder: {variant_dir}"
                print(msg)
                all_errors.append(msg)
                continue

            for sensor in SENSORS:
                npz_path = variant_dir / f"{sensor}_embeddings.npz"
                errors = check_npz(npz_path, scenario, variant, sensor)
                total_checked += 1

                if errors:
                    for e in errors:
                        print(f"  ✗ {e}")
                    all_errors.extend(errors)
                else:
                    print(f"  ✓ {variant}/{sensor}")

    # ── Cross-scenario consistency: same subjects present in all ──
    print("\n[Cross-scenario subject consistency]")
    reference = None
    reference_label = None
    cross_errors = []

    for scenario, variants in SCENARIOS.items():
        for variant in variants:
            for sensor in SENSORS:
                npz_path = DATA_DIR / variant / EMBEDDINGS_FOLDER / f"{sensor}_embeddings.npz"
                if not npz_path.exists():
                    continue
                data = np.load(npz_path)
                if "train_subjects" not in data:
                    continue

                key = (
                    frozenset(data["train_subjects"].astype(int).tolist()),
                    frozenset(data["val_subjects"].astype(int).tolist()),
                    frozenset(data["test_subjects"].astype(int).tolist()),
                )
                if reference is None:
                    reference = key
                    reference_label = f"{scenario}/{variant}/{sensor}"
                elif key != reference:
                    msg = (f"  ✗ MISMATCH vs {reference_label}: "
                           f"{scenario}/{variant}/{sensor}")
                    print(msg)
                    cross_errors.append(msg)

    if not cross_errors:
        print("  ✓ All NPZ files share identical subject splits")

    all_errors.extend(cross_errors)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_errors:
        print(f"  FAILED — {len(all_errors)} error(s) found across {total_checked} files checked")
        print("=" * 65)
        sys.exit(1)
    else:
        print(f"  PASSED — all {total_checked} NPZ files verified ✓")
        print("=" * 65)
        sys.exit(0)


if __name__ == "__main__":
    main()
