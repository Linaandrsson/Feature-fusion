import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
# Specify the PARENT directory (base configuration)
# The script will look for a variant subdirectory to read data from,
# and save splits.npz to the parent directory (shared across all variants)

PARENT_DIR = "/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files/s2_w2_aug2"

# Optionally specify which variant to use for reading sample info
# If None, will use the first variant found
VARIANT_SUBDIR = "fs50_clean"  # e.g., "clean", "AWGN_s0p3_AAnAC_n1", etc.

# Split ratios
TEST_SIZE = 0.30      # 30% for validation + test
VAL_TEST_SPLIT = 0.50 # Split the 30% equally: 15% val, 15% test
RANDOM_STATE = 42

# ============================================================
# MAIN
# ============================================================
parent_path = Path(PARENT_DIR)

# Find variant directory to read from
if VARIANT_SUBDIR:
    variant_path = parent_path / VARIANT_SUBDIR
    if not variant_path.exists():
        raise ValueError(f"Specified variant '{VARIANT_SUBDIR}' not found in {parent_path}")
else:
    # Use first variant found
    variants = [d for d in parent_path.iterdir() if d.is_dir()]
    if not variants:
        raise ValueError(f"No variant subdirectories found in {parent_path}")
    variant_path = variants[0]
    print(f"Using variant: {variant_path.name}")

# Load reference data from variant
ref_file = variant_path / "Acc_ankle.npz"
if not ref_file.exists():
    raise FileNotFoundError(f"Reference file not found: {ref_file}")

ref = np.load(ref_file)
y = ref["y"]                    # shape (N,)
subj = ref["subject_id"]        # shape (N,)
base_window_idx = ref["base_window_idx"]  # shape (N,)

N = len(y)
all_idx = np.arange(N)

print(f"Loaded data from: {variant_path.name}")
print(f"Total samples: {N}")
print(f"Unique base windows: {len(np.unique(base_window_idx))}")

# Train vs temp split
train_idx, temp_idx = train_test_split(
    all_idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Val vs test split (from temp)
y_temp = y[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=VAL_TEST_SPLIT, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print(f"Split ratios: train={len(train_idx)/N:.1%}, val={len(val_idx)/N:.1%}, test={len(test_idx)/N:.1%}")

# Save splits to PARENT directory (shared across all variants)
out_path = parent_path / "splits.npz"

np.savez(
    out_path,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

print(f"\n✓ Saved splits to: {out_path}")
print(f"  This file is SHARED across all variants in {parent_path.name}/")

# Also save as text files
np.savetxt(parent_path / "train_idx.txt", train_idx, fmt="%d")
np.savetxt(parent_path / "val_idx.txt", val_idx, fmt="%d")
np.savetxt(parent_path / "test_idx.txt", test_idx, fmt="%d")

print(f"✓ Also saved train_idx.txt, val_idx.txt, test_idx.txt")

# Print summary
print(f"\n{'='*60}")
print("Summary:")
print(f"  Parent directory: {parent_path.name}/")
print(f"  Reference variant: {variant_path.name}/")
print(f"  Splits file: splits.npz (in parent, shared by all variants)")
print(f"  Total samples: {N}")
print(f"  Train: {len(train_idx)} ({len(train_idx)/N:.1%})")
print(f"  Val:   {len(val_idx)} ({len(val_idx)/N:.1%})")
print(f"  Test:  {len(test_idx)} ({len(test_idx)/N:.1%})")
print(f"{'='*60}")

