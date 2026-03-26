#!/usr/bin/env python3
"""Check embedding magnitude per sensor in Subject 1 vs training"""

import numpy as np
from pathlib import Path

workspace = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work")
tremor_head = workspace / "Tremor_head"
feature_extractors_dir = tremor_head / "feature_extractors"

# Training embeddings per sensor (clean variant)
training_embeddings_dir = workspace / "data" / "Tremor_datagenerator_files" / "s2_w2_fs50_tremor_clean" / "Tremor_ExtractedFeatures"

sensors = ["Acc_arm", "Gyro_arm", "Mag_arm"]
subject1_embeddings_dir = feature_extractors_dir / "Extraction" / "subject1_embeddings"

print("="*80)
print("ANALYZING EMBEDDING MAGNITUDES PER SENSOR")
print("="*80)

print("\n" + "="*80)
print("TRAINING DATA (clean variant)")
print("="*80)

train_embeddings_by_sensor = {}
for sensor in sensors:
    file = training_embeddings_dir / f"{sensor}_embeddings.npz"
    if file.exists():
        data = np.load(file)
        embeds = data['train_embeddings']
        train_embeddings_by_sensor[sensor] = embeds
        print(f"\n{sensor}:")
        print(f"  Shape: {embeds.shape}")
        print(f"  Stats: min={embeds.min():.3f}, max={embeds.max():.3f}, " +
              f"mean={embeds.mean():.3f}, std={embeds.std():.3f}")
        print(f"  Magnitude (L2): min={np.linalg.norm(embeds, axis=1).min():.3f}, " +
              f"max={np.linalg.norm(embeds, axis=1).max():.3f}, " +
              f"mean={np.linalg.norm(embeds, axis=1).mean():.3f}")

print("\n" + "="*80)
print("SUBJECT 1 DATA - Testing where embedding structure comes from")
print("="*80)

# Load full Subject 1 embeddings
s1_file = subject1_embeddings_dir / "subject1_embeddings_clean.npz"
if s1_file.exists():
    s1_data = np.load(s1_file, allow_pickle=True)
    full_embed = s1_data['embeddings']  # 384-dim concatenated
    print(f"\nSubject 1 clean embeddings (CONCATENATED 384-dim):")
    print(f"  Shape: {full_embed.shape}")
    print(f"  Stats: min={full_embed.min():.3f}, max={full_embed.max():.3f}, " +
          f"mean={full_embed.mean():.3f}, std={full_embed.std():.3f}")
    
    # Split into sensor components (assuming Acc first 128, Gyro next 128, Mag last 128)
    print(f"\n  Split by sensor (assuming order: Acc, Gyro, Mag):")
    for i, sensor in enumerate(sensors):
        start = i * 128
        end = (i + 1) * 128
        sensor_embed = full_embed[:, start:end]
        print(f"\n  {sensor} (indices {start}:{end}):")
        print(f"    Stats: min={sensor_embed.min():.3f}, max={sensor_embed.max():.3f}, " +
              f"mean={sensor_embed.mean():.3f}, std={sensor_embed.std():.3f}")
        print(f"    Magnitude (L2): min={np.linalg.norm(sensor_embed, axis=1).min():.3f}, " +
              f"max={np.linalg.norm(sensor_embed, axis=1).max():.3f}, " +
              f"mean={np.linalg.norm(sensor_embed, axis=1).mean():.3f}")

print("\n" + "="*80)
print("COMPARISON: Subject 1 vs Training")
print("="*80)

print("\nTraining Acc_arm mean magnitude:", f"{np.linalg.norm(train_embeddings_by_sensor['Acc_arm'], axis=1).mean():.3f}")
print("Subject 1 Acc_arm mean magnitude (if first 128):", f"{np.linalg.norm(full_embed[:, 0:128], axis=1).mean():.3f}")

print("\n→ If Subject 1 has larger magnitudes, embeddings may not be normalized the same way!")
