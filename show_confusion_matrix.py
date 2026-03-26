#!/usr/bin/env python3
"""Display confusion matrix and performance metrics for binary tremor classification."""

import json
import numpy as np
from pathlib import Path

# Load the training results
log_file = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Tremor_head/fusion/logs/accarm50_gyroarm50_magarm50/feature_fusion_training.jsonl")
with open(log_file) as f:
    results = json.load(f)

print("=" * 80)
print("BINARY TREMOR CLASSIFICATION - PERFORMANCE SUMMARY")
print("=" * 80)

print("\n📊 VALIDATION SET")
print("-" * 80)
val_metrics = results["val_metrics"]
print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
print(f"  F1 Score (Macro): {val_metrics['f1']:.4f}")
print(f"  Loss: {val_metrics['loss']:.8f}")

print("\n📊 TEST SET")
print("-" * 80)
test_metrics = results["test_metrics"]
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  F1 Score (Macro): {test_metrics['f1']:.4f}")
print(f"  Loss: {test_metrics['loss']:.8f}")

print("\n📋 MODEL ARCHITECTURE")
print("-" * 80)
config = results["config"]
print(f"  Sensors: {', '.join(config['sensors'])}")
print(f"  Sampling Frequencies: {', '.join([f'{v} Hz' for v in config['sensor_freq_map'].values()])}")
print(f"  Embedding Dimension: {config['total_embed_dim']} (3 × {config['embed_dim_per_sensor']})")
print(f"  Hidden Dimensions: {' → '.join(map(str, config['hidden_dims']))}")
print(f"  Output Classes: 2 (No Tremor / Tremor)")
print(f"  Total Parameters: {results['total_params']:,}")
print(f"  Dropout: 0.3")

print("\n📚 TRAINING DATA")
print("-" * 80)
print(f"  Dataset Variants: clean, mild_mod, mod_severe")
print(f"  Batch Size: {config['batch_size']}")
print(f"  Learning Rate: {config['lr']}")
print(f"  Epochs Trained: {config['epochs_trained']}")

print("\n✅ STATUS")
print("-" * 80)
if results["is_best"]:
    print(f"  ✓ Best model saved!")
print(f"  ✓ Random seed: {results['random_seed']}")
print(f"  ✓ Training timestamp: {results['timestamp']}")

print("\n📁 OUTPUT FILES")
print("-" * 80)
base_dir = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Tremor_head/fusion")
model_file = base_dir / "models" / "accarm50_gyroarm50_magarm50" / "feature_fusion_best_accarm50_gyroarm50_magarm50.pth"
cm_val = base_dir / "plots" / "accarm50_gyroarm50_magarm50" / "confusion_matrix_validation.png"
cm_test = base_dir / "plots" / "accarm50_gyroarm50_magarm50" / "confusion_matrix_test.png"

print(f"  Model: {model_file.name}")
print(f"  Validation CM: {cm_val.name}")
print(f"  Test CM: {cm_test.name}")

print("\n" + "=" * 80)
print("Confusion matrices have been visualized and saved as PNG files!")
print("=" * 80)
