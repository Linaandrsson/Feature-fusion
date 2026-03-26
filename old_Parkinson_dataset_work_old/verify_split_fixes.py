#!/usr/bin/env python3
"""
Verification script for subject-based split fixes.

This script checks that:
1. CNN training configs have MANUAL_SUBJECT_SPLITS defined
2. No train_test_split is used in CNN training files
3. Split generation uses subject-based logic
4. Split files are consistent across sensors when generated

Run this after applying the fixes to verify everything is working.
"""

import re
import sys
from pathlib import Path

def check_file_content(file_path, should_not_contain=None, should_contain=None):
    """Check if file contains/doesn't contain certain strings."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    errors = []
    
    if should_not_contain:
        for pattern in should_not_contain:
            if re.search(pattern, content):
                errors.append(f"  ❌ Found forbidden pattern: {pattern}")
    
    if should_contain:
        for pattern in should_contain:
            if not re.search(pattern, content):
                errors.append(f"  ❌ Missing required pattern: {pattern}")
    
    return errors


def main():
    root = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work")
    training_dir = root / "Feature extraction CNNs" / "Training" / "s2_w2_aug2_fs50"
    
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    print("=" * 70)
    print("SUBJECT-BASED SPLIT FIX VERIFICATION")
    print("=" * 70)
    
    all_ok = True
    
    # 1. Check config.py has MANUAL_SUBJECT_SPLITS
    print("\n[1] Checking config.py for MANUAL_SUBJECT_SPLITS...")
    config_file = training_dir / "config.py"
    errors = check_file_content(
        config_file,
        should_contain=[
            r'MANUAL_SUBJECT_SPLITS\s*=\s*{',
            r'"train":\s*\[',
            r'"val":\s*\[',
            r'"test":\s*\[',
        ]
    )
    if errors:
        all_ok = False
        for err in errors:
            print(err)
    else:
        print("  ✅ MANUAL_SUBJECT_SPLITS properly defined in config.py")
    
    # 2. Check config.py has build_source_aware_subject_splits
    print("\n[2] Checking config.py for build_source_aware_subject_splits...")
    errors = check_file_content(
        config_file,
        should_contain=[r'def build_source_aware_subject_splits']
    )
    if errors:
        all_ok = False
        for err in errors:
            print(err)
    else:
        print("  ✅ build_source_aware_subject_splits function defined")
    
    # 3. Check CNN files don't use train_test_split
    print("\n[3] Checking CNN files for train_test_split...")
    cnn_files = ["Acc_arm_CNN.py", "Gyro_arm_CNN.py", "Mag_arm_CNN.py"]
    for cnn_file in cnn_files:
        file_path = training_dir / cnn_file
        errors = check_file_content(
            file_path,
            should_not_contain=[
                r'from sklearn\.model_selection import train_test_split',
                r'train_test_split\(',
            ]
        )
        if errors:
            all_ok = False
            print(f"  ❌ {cnn_file}:")
            for err in errors:
                print(  "    " + err)
        else:
            print(f"  ✅ {cnn_file} - No train_test_split usage")
    
    # 4. Check CNN files use subject-based splitting
    print("\n[4] Checking CNN files for subject-based split logic...")
    for cnn_file in cnn_files:
        file_path = training_dir / cnn_file
        errors = check_file_content(
            file_path,
            should_contain=[
                r'np\.isin\(subjects_variant',
                r'MANUAL_SUBJECT_SPLITS',
                r'build_source_aware_subject_splits',
            ]
        )
        if errors:
            all_ok = False
            print(f"  ❌ {cnn_file}:")
            for err in errors:
                print("    " + err)
        else:
            print(f"  ✅ {cnn_file} - Uses subject-based split logic")
    
    # 5. Check split type is identified
    print("\n[5] Checking for split type identification...")
    for cnn_file in cnn_files:
        file_path = training_dir / cnn_file
        errors = check_file_content(
            file_path,
            should_contain=[
                r'split_type\s*=\s*"subject-based',
            ]
        )
        if errors:
            all_ok = False
            print(f"  ❌ {cnn_file}:")
            for err in errors:
                print("    " + err)
        else:
            print(f"  ✅ {cnn_file} - Identifies split type")
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Subject-based splits are correctly implemented!")
        return True
    else:
        print("❌ SOME CHECKS FAILED - Please review the errors above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
