#!/usr/bin/env python3
"""Check if subject 1 data exists in tremor datasets."""

import numpy as np
from pathlib import Path

data_root = Path("data/Tremor_datagenerator_files")

datasets = [
    "s2_w2_fs50_tremor_clean",
    "s2_w2_fs50_tremor_mild_mod",
    "s2_w2_fs50_tremor_mod_severe"
]

print("=" * 80)
print("CHECKING FOR SUBJECT 1 DATA IN TREMOR DATASETS")
print("=" * 80)

for dataset_name in datasets:
    print(f"\n{dataset_name}:")
    print("-" * 80)
    
    acc_file = data_root / dataset_name / "Acc_arm.npz"
    
    if not acc_file.exists():
        print(f"  ✗ File not found: {acc_file}")
        continue
    
    data = np.load(acc_file)
    
    if 'y_subject' in data:
        subjects = data['y_subject']
        unique_subjects = sorted(np.unique(subjects).astype(int))
        
        print(f"  Subjects in dataset: {unique_subjects}")
        
        if 1 in unique_subjects:
            subject1_count = np.sum(subjects == 1)
            print(f"  ✓ Subject 1 IS present ({subject1_count} samples)")
        else:
            print(f"  ✗ Subject 1 is NOT in this dataset")
            
        # Show distribution
        print(f"\n  Samples per subject:")
        for s in unique_subjects:
            count = np.sum(subjects == s)
            print(f"    Subject {s:2d}: {count:5d} samples")
    else:
        print(f"  ✗ No 'y_subject' key in file")

print("\n" + "=" * 80)
