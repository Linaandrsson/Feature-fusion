#!/usr/bin/env python3
"""Check if training embeddings were extracted with same norm_stats as Subject 1"""

import torch
from pathlib import Path

workspace = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work")
models_dir = workspace / "Tremor_head" / "feature_extractors" / "models" / "s2_w2_tremor" / "fs50_mixed3"

print("="*80)
print("CHECKING FEATURE EXTRACTOR CHECKPOINTS FOR NORM_STATS")
print("="*80)

for sensor in ["Acc_arm", "Gyro_arm", "Mag_arm"]:
    model_path = models_dir / f"feature_extractor_{sensor}.pth"
    
    print(f"\n{sensor}:")
    print(f"  Path: {model_path}")
    print(f"  Exists: {model_path.exists()}")
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        
        if "norm_stats" in checkpoint:
            norm_stats = checkpoint["norm_stats"]
            print(f"  ✓ norm_stats found!")
            print(f"    Keys: {list(norm_stats.keys())}")
            
            for key in norm_stats.keys():
                val = norm_stats[key]
                print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                print(f"           min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")
        else:
            print(f"  ⚠️  norm_stats NOT found in checkpoint!")
            print(f"      This means data extraction may NOT have applied normalization!")
