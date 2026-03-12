"""
Extract Tremor Embeddings
==========================

Extracts 128-dim embeddings from trained Acc_arm and Gyro_arm CNNs.

For each dataset variant (clean, mild_mod, mod_severe):
  1. Loads trained models from Tremor_head/feature_extractors/Models/
  2. Loads raw sensor data
  3. Extracts embeddings using model.extract_features()
  4. Saves to: data/Tremor_datagenerator_files/{variant}/Tremor_ExtractedFeatures/

Output format (NPZ files):
  - {sensor}_embeddings.npz containing:
    - train_embeddings: (N_train, 128)
    - val_embeddings: (N_val, 128)
    - test_embeddings: (N_test, 128)
    - train_labels: (N_train,) tremor scores
    - val_labels: (N_val,)
    - test_labels: (N_test,)
    - train_activities: (N_train,)
    - val_activities: (N_val,)
    - test_activities: (N_test,)

Usage:
    python extract_tremor_embeddings.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - CHANGE THESE FOR DIFFERENT SAMPLING FREQUENCIES
# ═══════════════════════════════════════════════════════════════

# ======================== SELECT SAMPLING FREQUENCY ========================
# Uncomment ONE of the following configurations:

# --- Option 1: 50Hz sampling (fs50) ---
# SAMPLING_FREQ = "fs50"
# TRAINING_SCRIPTS_DIR = "s2_w2_tremor_fs50"
# SEQ_LEN = 100  # 50Hz * 2s = 100 samples

# --- Option 2: 30Hz sampling (fs30) ---
SAMPLING_FREQ = "fs50"
TRAINING_SCRIPTS_DIR = "s2_w2_tremor_fs50"
SEQ_LEN = 100  # 50Hz * 2s = 100 samples

# ===========================================================================

# Auto-generated paths based on configuration
WORKSPACE_ROOT = Path("/Volumes/NO NAME/Master Lina/Code")
DATA_PATH = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"
MODEL_DIR = WORKSPACE_ROOT / "Tremor_head" / "feature_extractors" / "Models" / "s2_w2_tremor" / f"{SAMPLING_FREQ}_mixed"

# Dataset variants (auto-generated based on sampling frequency)
DATASET_VARIANTS = [
    f"s2_w2_{SAMPLING_FREQ}_tremor_clean",
    f"s2_w2_{SAMPLING_FREQ}_tremor_mild_mod",
    f"s2_w2_{SAMPLING_FREQ}_tremor_mod_severe"
]

# Sensors and model paths
SENSORS = {
    "Acc_arm": MODEL_DIR / "feature_extractor_Acc_arm.pth",
    "Gyro_arm": MODEL_DIR / "feature_extractor_Gyro_arm.pth"
}

# Import model architectures from the correct training directory
sys.path.append(str(WORKSPACE_ROOT / "Tremor_head" / "feature_extractors" / "Training" / TRAINING_SCRIPTS_DIR))
from train_Acc_arm import AccArmFeatureExtractor
from train_Gyro_arm import GyroArmFeatureExtractor

# ═══════════════════════════════════════════════════════════════
# OTHER CONFIGURATION (usually no need to change)
# ═══════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split configuration (must match training)
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]

# Architecture parameters
num_channels = 3
batch_size = 64


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_tremor_data(sensor_name: str, dataset_path: Path) -> Dict[str, np.ndarray]:
    """Load tremor dataset from TXT file."""
    file_path = dataset_path / f"{sensor_name}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"  Loading {sensor_name}...")
    data = np.loadtxt(str(file_path), delimiter=",")
    
    # Determine number of channels
    if "ECG" in sensor_name:
        n_channels = 2
        window_length = SEQ_LEN  # Use configured sequence length
    else:
        n_channels = 3
        window_length = SEQ_LEN  # Use configured sequence length
    
    # Extract data
    sensor_data = data[:, :n_channels * window_length]
    
    # Reshape to (N, C, L)
    N = data.shape[0]
    X = sensor_data.reshape(N, n_channels, window_length)
    
    # Extract labels
    col_offset = n_channels * window_length
    y_activity = data[:, col_offset].astype(np.int64) - 1  # 0-indexed
    y_subject = data[:, col_offset + 1].astype(np.int64)   # 1-indexed
    y_tremor_score = data[:, col_offset + 6].astype(np.int64)
    
    return {
        "X": X.astype(np.float32),
        "y_activity": y_activity,
        "y_subject": y_subject,
        "y_tremor_score": y_tremor_score,
    }


def split_data_by_subject(
    data: Dict[str, np.ndarray],
    val_subjects: List[int],
    test_subjects: List[int]
) -> Tuple[Dict, Dict, Dict]:
    """Split data by subject IDs."""
    subjects = data["y_subject"]
    
    train_mask = ~np.isin(subjects, val_subjects + test_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    train_data = {k: v[train_mask] for k, v in data.items()}
    val_data = {k: v[val_mask] for k, v in data.items()}
    test_data = {k: v[test_mask] for k, v in data.items()}
    
    return train_data, val_data, test_data


def normalize_data(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize data using precomputed statistics."""
    return (X - stats["mean"]) / stats["std"]


# ═══════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_embeddings(model, X, activities, norm_stats, device):
    """
    Extract embeddings from model.
    
    Args:
        model: Trained feature extractor
        X: (N, C, L) raw sensor data
        activities: (N,) activity labels
        norm_stats: Normalization statistics
        device: torch device
    
    Returns:
        embeddings: (N, 128) feature embeddings
    """
    model.eval()
    
    # Normalize data
    X_norm = normalize_data(X, norm_stats)
    
    # Create dataset and loader (no shuffling to preserve order)
    dataset = TensorDataset(
        torch.from_numpy(X_norm),
        torch.from_numpy(activities)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract embeddings
    all_embeddings = []
    with torch.no_grad():
        for X_batch, act_batch in loader:
            X_batch = X_batch.to(device)
            act_batch = act_batch.to(device)
            
            embeddings = model.extract_features(X_batch, act_batch)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def extract_sensor_embeddings(sensor_name: str, model_path: Path, variant_path: Path, output_dir: Path):
    """
    Extract embeddings for one sensor from one dataset variant.
    
    Args:
        sensor_name: Sensor name (Acc_arm or Gyro_arm)
        model_path: Path to trained model checkpoint
        variant_path: Path to dataset variant
        output_dir: Output directory for embeddings
    """
    print(f"\n  Extracting {sensor_name} embeddings...")
    
    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    norm_stats = checkpoint['norm_stats']
    
    # Create model
    if sensor_name == "Acc_arm":
        model = AccArmFeatureExtractor().to(device)
    elif sensor_name == "Gyro_arm":
        model = GyroArmFeatureExtractor().to(device)
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    data = load_tremor_data(sensor_name, variant_path)
    
    # Split by subject
    train_data, val_data, test_data = split_data_by_subject(data, VAL_SUBJECTS, TEST_SUBJECTS)
    
    # DEBUG: Print subject splits
    print(f"    DEBUG - Train subjects: {np.unique(train_data['y_subject'])}")
    print(f"    DEBUG - Val subjects: {np.unique(val_data['y_subject'])}")
    print(f"    DEBUG - Test subjects: {np.unique(test_data['y_subject'])}")
    
    # Extract embeddings for each split
    train_embeddings = extract_embeddings(model, train_data["X"], train_data["y_activity"], norm_stats, device)
    val_embeddings = extract_embeddings(model, val_data["X"], val_data["y_activity"], norm_stats, device)
    test_embeddings = extract_embeddings(model, test_data["X"], test_data["y_activity"], norm_stats, device)
    
    print(f"    Train: {train_embeddings.shape}")
    print(f"    Val:   {val_embeddings.shape}")
    print(f"    Test:  {test_embeddings.shape}")
    
    # Save embeddings
    output_file = output_dir / f"{sensor_name}_embeddings.npz"
    np.savez_compressed(
        output_file,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        train_labels=train_data["y_tremor_score"],
        val_labels=val_data["y_tremor_score"],
        test_labels=test_data["y_tremor_score"],
        train_activities=train_data["y_activity"],
        val_activities=val_data["y_activity"],
        test_activities=test_data["y_activity"],
        train_subjects=train_data["y_subject"],
        val_subjects=val_data["y_subject"],
        test_subjects=test_data["y_subject"]
    )
    
    print(f"    Saved to: {output_file}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Ensure valid working directory for numpy
    import os
    os.chdir(WORKSPACE_ROOT)
    
    print("=" * 80)
    print("EXTRACTING TREMOR EMBEDDINGS")
    print("=" * 80)
    
    # Display current configuration
    print("\nConfiguration:")
    print(f"  Sampling Frequency: {SAMPLING_FREQ.upper()}")
    print(f"  Sequence Length: {SEQ_LEN} samples")
    print(f"  Model Directory: {MODEL_DIR}")
    print(f"  Dataset Variants: {len(DATASET_VARIANTS)} variants")
    for variant in DATASET_VARIANTS:
        print(f"    - {variant}")
    
    # Verify trained models exist
    print("\nVerifying trained models...")
    print()
    for sensor, model_path in SENSORS.items():
        if not model_path.exists():
            print(f"ERROR: Model not found for {sensor}: {model_path}")
            print(f"Please train the model first using the training scripts in:")
            print(f"  Tremor_head/feature_extractors/Training/s2_w2_tremor_fs50/")
            return
        print(f"  ✓ {sensor}: {model_path}")
    
    # Extract embeddings for each variant
    for variant in DATASET_VARIANTS:
        print(f"\n{'=' * 80}")
        print(f"Processing variant: {variant}")
        print('=' * 80)
        
        variant_path = DATA_PATH / variant
        if not variant_path.exists():
            print(f"  Warning: Variant not found: {variant_path}")
            continue
        
        # Create output directory
        output_dir = variant_path / "Tremor_ExtractedFeatures"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract embeddings for each sensor
        for sensor_name, model_path in SENSORS.items():
            try:
                extract_sensor_embeddings(sensor_name, model_path, variant_path, output_dir)
            except Exception as e:
                print(f"  ERROR extracting {sensor_name}: {e}")
                continue
        
        print(f"\n  All embeddings saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nEmbeddings extracted for: {SAMPLING_FREQ.upper()}")
    print(f"Embeddings are ready for feature-level fusion!")
    print(f"\nLocation pattern:")
    print(f"  data/Tremor_datagenerator_files/s2_w2_{SAMPLING_FREQ}_tremor_{{variant}}/Tremor_ExtractedFeatures/")
    print(f"\nTo extract embeddings for a different sampling frequency:")
    print(f"  1. Edit the CONFIGURATION section at the top of this file")
    print(f"  2. Uncomment the desired configuration (fs30 or fs50)")
    print(f"  3. Comment out the other configuration")
    print(f"  4. Run this script again")


if __name__ == "__main__":
    main()
