#!/usr/bin/env python3
"""
Check if embeddings exhibit pattern inversion:
- Positive correlation → same signal
- Negative correlation → inverted signal
"""

import numpy as np
from scipy import stats
from pathlib import Path

workspace = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work")

# Load training data
train_dir = workspace / "data" / "Tremor_datagenerator_files" / "s2_w2_fs50_tremor_clean" / "Tremor_ExtractedFeatures"
train_gyro = np.load(train_dir / "Gyro_arm_embeddings.npz")['train_embeddings'][:100]

# Load Subject 1 data
s1_dir = workspace / "Tremor_head" / "feature_extractors" / "Extraction" / "subject1_embeddings"
s1_data = np.load(s1_dir / "subject1_embeddings_clean.npz", allow_pickle=True)
s1_gyro = s1_data['embeddings'][:100, 128:256]  # Extract Gyro_arm component

print("="*80)
print("CORRELATION ANALYSIS: Training vs Subject 1 Embeddings")
print("="*80)

print(f"\nTesting if Subject 1 Gyro_arm embeddings are INVERTED (negatively correlated)")
print(f"Training Gyro_arm shape: {train_gyro.shape}")
print(f"Subject 1 Gyro_arm shape: {s1_gyro.shape}")

# Calculate correlation
# Method 1: Flatten and correlate first samples
train_flat = train_gyro[0].flatten()
s1_flat = s1_gyro[0].flatten()
corr, pval = stats.pearsonr(train_flat, s1_flat)
print(f"\nPearson correlation (sample 0): {corr:.4f} (p={pval:.4e})")

if abs(corr) > 0.7:
    if corr > 0:
        print("  → POSITIVE correlation: embeddings are SIMILAR (same signal)")
    else:
        print("  → NEGATIVE correlation: embeddings are INVERTED (opposite signal)")
else:
    print("  → LOW correlation: different distributions/features")

# Method 2: Check sign patterns
print(f"\nSign pattern analysis (first 10 values):")
print(f"  Training Gyro[0:10]: {train_gyro[0, 0:10]}")
print(f"  Subject1 Gyro[0:10]: {s1_gyro[0, 0:10]}")

signs_train = np.sign(train_gyro[0])
signs_s1 = np.sign(s1_gyro[0])
matching_signs = np.mean(signs_train == signs_s1)
print(f"\n  Matching sign patterns: {matching_signs:.1%}")
if matching_signs < 0.5:
    print("  → Signs are often OPPOSITE (suggests inversion)")
else:
    print("  → Signs are mostly same")

# Method 3: Model perspective - what does the network see?
print(f"\n" + "="*80)
print("MODEL PERSPECTIVE")
print("="*80)

train_magnitude = np.linalg.norm(train_gyro, axis=1)
s1_magnitude = np.linalg.norm(s1_gyro, axis=1)

print(f"Training Gyro_arm magnitude: mean={train_magnitude.mean():.3f}, std={train_magnitude.std():.3f}")
print(f"Subject1 Gyro_arm magnitude: mean={s1_magnitude.mean():.3f}, std={s1_magnitude.std():.3f}")

print(f"\nSUMMARY:")
print(f"- Subject 1 embeddings have {s1_magnitude.mean()/train_magnitude.mean():.1f}x larger magnitude")
print(f"- Model trained on normalized embeddings")
print(f"- Receiving non-normalized Subject 1 embeddings")
print(f"- This amplifies logits, potentially causing decision boundary flip")
