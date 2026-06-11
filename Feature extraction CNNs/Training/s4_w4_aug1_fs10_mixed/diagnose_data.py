"""
Data Diagnostics Script
=======================
Checks data loading, labels, splits, and class distribution.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import parent_dir, variant_dirs, seq_len, load_combined_sensor_data

print("=" * 80)
print("DATA DIAGNOSTICS")
print("=" * 80)

# Test with Gyro_arm
sensor_name = "Gyro_arm"
num_channels = 3

print(f"\n1. Loading {sensor_name} data...")
data = load_combined_sensor_data(f"{sensor_name}.txt")
print(f"   Combined data shape: {data.shape}")

# Extract components
X = data[:, :-7]
y_activity = data[:, -7].astype(int)
y_subject = data[:, -6].astype(int)
base_idx = data[:, -5].astype(int)
tremor_freq = data[:, -4]
tremor_acc_rms = data[:, -3]
tremor_gyro_rms = data[:, -2]
tremor_score = data[:, -1].astype(int)

print(f"\n2. Data components:")
print(f"   X (sensor):     {X.shape}")
print(f"   y_activity:     {y_activity.shape} | range: [{y_activity.min()}, {y_activity.max()}]")
print(f"   y_subject:      {y_subject.shape} | unique: {sorted(np.unique(y_subject))}")
print(f"   base_idx:       {base_idx.shape} | range: [{base_idx.min()}, {base_idx.max()}]")
print(f"   tremor_score:   {tremor_score.shape} | unique: {sorted(np.unique(tremor_score))}")

print(f"\n3. Activity distribution (BEFORE subtracting 1):")
unique_acts, counts = np.unique(y_activity, return_counts=True)
for act, count in zip(unique_acts, counts):
    print(f"   Activity {act:2d}: {count:5d} samples ({count/len(y_activity)*100:.1f}%)")

print(f"\n4. Subject distribution:")
unique_subs, counts = np.unique(y_subject, return_counts=True)
for sub, count in zip(unique_subs, counts):
    print(f"   Subject {sub:2d}: {count:5d} samples ({count/len(y_subject)*100:.1f}%)")

# Check splits
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]

train_idx = np.where(~np.isin(y_subject, TEST_SUBJECTS + VAL_SUBJECTS))[0]
val_idx = np.where(np.isin(y_subject, VAL_SUBJECTS))[0]
test_idx = np.where(np.isin(y_subject, TEST_SUBJECTS))[0]

print(f"\n5. Subject-based splits:")
print(f"   Train: {len(train_idx)} samples - subjects: {sorted(np.unique(y_subject[train_idx]))}")
print(f"   Val:   {len(val_idx)} samples - subjects: {sorted(np.unique(y_subject[val_idx]))}")
print(f"   Test:  {len(test_idx)} samples - subjects: {sorted(np.unique(y_subject[test_idx]))}")

# Check if subjects are disjoint
train_subs = set(y_subject[train_idx])
val_subs = set(y_subject[val_idx])
test_subs = set(y_subject[test_idx])

print(f"\n6. Checking for subject overlap:")
print(f"   Train ∩ Val:  {train_subs & val_subs} (should be empty)")
print(f"   Train ∩ Test: {train_subs & test_subs} (should be empty)")
print(f"   Val ∩ Test:   {val_subs & test_subs} (should be empty)")

# Activity distribution in each split
print(f"\n7. Activity distribution in splits (AFTER subtracting 1):")
y_activity_0indexed = y_activity - 1

print("   TRAIN:")
unique_acts, counts = np.unique(y_activity_0indexed[train_idx], return_counts=True)
for act, count in zip(unique_acts, counts):
    print(f"      Activity {act:2d}: {count:5d} samples ({count/len(train_idx)*100:.1f}%)")

print("   VAL:")
unique_acts, counts = np.unique(y_activity_0indexed[val_idx], return_counts=True)
for act, count in zip(unique_acts, counts):
    print(f"      Activity {act:2d}: {count:5d} samples ({count/len(val_idx)*100:.1f}%)")

print("   TEST:")
unique_acts, counts = np.unique(y_activity_0indexed[test_idx], return_counts=True)
for act, count in zip(unique_acts, counts):
    print(f"      Activity {act:2d}: {count:5d} samples ({count/len(test_idx)*100:.1f}%)")

# Check for missing classes in splits
all_activities = set(range(12))
train_activities = set(y_activity_0indexed[train_idx])
val_activities = set(y_activity_0indexed[val_idx])
test_activities = set(y_activity_0indexed[test_idx])

print(f"\n8. Class coverage:")
print(f"   Missing in train: {all_activities - train_activities}")
print(f"   Missing in val:   {all_activities - val_activities}")
print(f"   Missing in test:  {all_activities - test_activities}")

# Check data shape after reshaping
X_reshaped = X.reshape(-1, num_channels, seq_len).astype(np.float32)
print(f"\n9. Reshaping check:")
print(f"   Original X:  {X.shape}")
print(f"   Reshaped X:  {X_reshaped.shape}")
print(f"   Expected:    ({len(data)}, {num_channels}, {seq_len})")

# Check for NaN or inf
print(f"\n10. Data quality:")
print(f"   NaN values in X:       {np.isnan(X).sum()}")
print(f"   Inf values in X:       {np.isinf(X).sum()}")
print(f"   X min/max:             [{X.min():.2f}, {X.max():.2f}]")
print(f"   X mean/std:            {X.mean():.2f} ± {X.std():.2f}")

# Sample a few data points to check
print(f"\n11. Sample data inspection (first 3 samples):")
for i in range(min(3, len(data))):
    print(f"   Sample {i}:")
    print(f"      Activity: {y_activity[i]} (0-indexed: {y_activity_0indexed[i]})")
    print(f"      Subject:  {y_subject[i]}")
    print(f"      Base idx: {base_idx[i]}")
    print(f"      Tremor:   {tremor_score[i]}")
    print(f"      X shape:  {X_reshaped[i].shape}")
    print(f"      X range:  [{X_reshaped[i].min():.2f}, {X_reshaped[i].max():.2f}]")

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
