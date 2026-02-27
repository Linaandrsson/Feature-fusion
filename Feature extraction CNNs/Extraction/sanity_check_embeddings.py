"""
Sanity check to verify that embeddings from different sensors and variants have the same order.

This script checks:
1. All sensors within each variant have the same number of samples
2. Labels are identical across all sensors (same order)
3. Labels are identical across all variants (same windows)
4. Train/val/test splits have consistent sizes

Usage:
    python sanity_check_embeddings.py
    
Configure dataset_config below to check different parent directories.
"""

import numpy as np
from pathlib import Path

# ========================= CONFIGURE THIS ===========================
dataset_config = "s1_w1_aug2"  # Parent directory to check
workspace_root = Path("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code")
# ====================================================================

parent_dir = workspace_root / "data" / "Datagenerator_files" / dataset_config

SENSORS = {
    "Acc_ankle": {"num_channels": 3},
    "Acc_arm": {"num_channels": 3},
    "Acc_chest": {"num_channels": 3},
    "ECG": {"num_channels": 2},
    "Gyro_ankle": {"num_channels": 3},
    "Gyro_arm": {"num_channels": 3},
    "Mag_ankle": {"num_channels": 3},
    "Mag_arm": {"num_channels": 3},
}


def find_extracted_features_dirs():
    """Find all directories containing ExtractedFeatures."""
    extracted_dirs = []
    
    for variant_dir in parent_dir.iterdir():
        if not variant_dir.is_dir():
            continue
        
        extracted_path = variant_dir / "ExtractedFeatures"
        if extracted_path.exists() and extracted_path.is_dir():
            extracted_dirs.append(extracted_path)
    
    return sorted(extracted_dirs)


def load_embeddings(extracted_dir: Path, sensor_name: str, split: str):
    """Load embeddings for a sensor and split."""
    file_path = extracted_dir / f"{sensor_name}_{split}.npz"
    if not file_path.exists():
        return None, None
    
    try:
        data = np.load(file_path)
        return data['embeddings'], data['labels']
    except Exception as e:
        print(f"    Error loading {file_path.name}: {e}")
        return None, None


def check_variant(extracted_dir: Path):
    """Check alignment within a single variant."""
    variant_name = extracted_dir.parent.name
    splits = ['train', 'val', 'test']
    variant_good = True
    variant_labels = {}  # Store labels for cross-variant comparison
    has_any_embeddings = False
    
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"{'='*60}")
    
    for split in splits:
        sensor_labels = {}
        sensor_shapes = {}
        
        for sensor_name in SENSORS.keys():
            embeddings, labels = load_embeddings(extracted_dir, sensor_name, split)
            if embeddings is None:
                continue
            
            has_any_embeddings = True
            sensor_labels[sensor_name] = labels
            sensor_shapes[sensor_name] = embeddings.shape
        
        if not sensor_labels:
            print(f"  [{split.upper()}] No embeddings found (not extracted yet)")
            continue
        
        print(f"\n  [{split.upper()}]")
        
        # Show shapes
        for sensor, shape in sensor_shapes.items():
            print(f"    {sensor:15s} | samples: {len(sensor_labels[sensor]):5d} | shape: {shape}")
        
        # Check same number of samples
        sample_counts = [len(labels) for labels in sensor_labels.values()]
        if len(set(sample_counts)) > 1:
            print(f"    ✗ MISMATCH: Different number of samples across sensors!")
            variant_good = False
            continue
        else:
            print(f"    ✓ All sensors have {sample_counts[0]} samples")
        
        # Check labels identical
        reference_sensor = list(sensor_labels.keys())[0]
        reference_labels = sensor_labels[reference_sensor]
        
        mismatches = []
        for sensor_name, labels in sensor_labels.items():
            if sensor_name == reference_sensor:
                continue
            
            if not np.array_equal(labels, reference_labels):
                mismatches.append(sensor_name)
                variant_good = False
        
        if mismatches:
            print(f"    ✗ Label mismatch in sensors: {', '.join(mismatches)}")
        else:
            print(f"    ✓ All labels match (same order)")
            
            # Store for cross-variant comparison
            variant_labels[split] = reference_labels
    
    # If variant has no embeddings at all, don't count it as failed
    if not has_any_embeddings:
        variant_good = None  # Mark as "not applicable" rather than failed
    
    return variant_good, variant_labels


def check_cross_variant(all_variant_labels):
    """Check that labels are identical across all variants."""
    print(f"\n{'='*60}")
    print(f"Cross-Variant Consistency Check")
    print(f"{'='*60}")
    
    splits = ['train', 'val', 'test']
    all_good = True
    
    for split in splits:
        print(f"\n  [{split.upper()}]")
        
        # Get all variants that have this split
        variants_with_split = {
            variant: labels[split] 
            for variant, labels in all_variant_labels.items() 
            if split in labels
        }
        
        if len(variants_with_split) < 2:
            print(f"    - Only {len(variants_with_split)} variant(s) found, skipping comparison")
            continue
        
        # Check all have same length
        lengths = {variant: len(labels) for variant, labels in variants_with_split.items()}
        if len(set(lengths.values())) > 1:
            print(f"    ✗ MISMATCH: Different number of samples across variants!")
            for variant, length in lengths.items():
                print(f"      {variant}: {length} samples")
            all_good = False
            continue
        else:
            print(f"    ✓ All variants have {list(lengths.values())[0]} samples")
        
        # Compare labels
        reference_variant = list(variants_with_split.keys())[0]
        reference_labels = variants_with_split[reference_variant]
        
        mismatches = []
        for variant, labels in variants_with_split.items():
            if variant == reference_variant:
                continue
            
            if not np.array_equal(labels, reference_labels):
                mismatches.append(variant)
                diff_count = np.sum(labels != reference_labels)
                print(f"    ✗ {variant} differs from {reference_variant} at {diff_count} positions")
                all_good = False
        
        if not mismatches:
            print(f"    ✓ All variants have identical labels (same window order)")
    
    return all_good


def main():
    """Main sanity check function."""
    
    print(f"\n{'='*60}")
    print(f"Sanity Check: Embedding Alignment")
    print(f"{'='*60}")
    print(f"Parent dir: {parent_dir}")
    print(f"Checking all variants with ExtractedFeatures folders...")
    print(f"{'='*60}")
    
    # Find all ExtractedFeatures directories
    extracted_dirs = find_extracted_features_dirs()
    
    if not extracted_dirs:
        print(f"\n✗ No ExtractedFeatures directories found in {parent_dir}")
        return
    
    print(f"\nFound {len(extracted_dirs)} variant(s) with extracted features:")
    for dir in extracted_dirs:
        print(f"  - {dir.parent.name}")
    
    # Check each variant
    all_good = True
    all_variant_labels = {}
    variants_checked = 0
    variants_failed = 0
    variants_not_extracted = 0
    
    for extracted_dir in extracted_dirs:
        variant_name = extracted_dir.parent.name
        variant_good, variant_labels = check_variant(extracted_dir)
        
        if variant_good is None:
            # Variant has no embeddings (not extracted yet)
            variants_not_extracted += 1
        elif not variant_good:
            # Variant has embeddings but failed checks
            variants_failed += 1
            all_good = False
        else:
            # Variant passed checks
            variants_checked += 1
        
        if variant_labels:
            all_variant_labels[variant_name] = variant_labels
    
    # Check cross-variant consistency
    if len(all_variant_labels) > 1:
        cross_good = check_cross_variant(all_variant_labels)
        if not cross_good:
            all_good = False
    
    # Final summary
    print(f"\n{'='*60}")
    if all_good:
        print("✓ SANITY CHECK PASSED")
        print("  All embeddings are properly aligned!")
        print("  - Within each variant: sensors have matching labels")
        print("  - Across variants: all have same windows in same order")
    else:
        print("✗ SANITY CHECK FAILED")
        print("  Some embeddings have alignment issues!")
        print("  Check the details above for specific problems.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
