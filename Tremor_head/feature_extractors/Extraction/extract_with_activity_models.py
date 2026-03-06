"""
Extract Tremor Embeddings Using Activity-Trained Models
=========================================================

Tests cross-domain transfer learning by extracting embeddings from tremor data
using models trained on activity classification.

Purpose:
  - Investigate if a single feature extractor can work for both pipelines
  - Compare performance: activity-trained vs tremor-trained embeddings

For each tremor dataset variant (clean, parkinson_mild, parkinson_severe):
  1. Loads activity-trained models from: Feature extraction CNNs/Models/s2_w2_aug2/fs50_clean/
  2. Loads tremor data with tremor_score labels
  3. Extracts embeddings using activity models
  4. Saves to: data/Tremor_datagenerator_files/{variant}/ExtractedFeatures_ActivityModels/

Output format (NPZ files):
  - {sensor}_embeddings.npz containing:
    - train_embeddings: (N_train, 128)
    - val_embeddings: (N_val, 128)
    - test_embeddings: (N_test, 128)
    - train_labels: (N_train,) tremor scores 0-4
    - val_labels: (N_val,)
    - test_labels: (N_test,)
    - train_activities: (N_train,) activity labels
    - val_activities: (N_val,)
    - test_activities: (N_test,)
    - train_subjects: (N_train,) subject IDs
    - val_subjects: (N_val,)
    - test_subjects: (N_test,)

Usage:
    python extract_with_activity_models.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
WORKSPACE_ROOT = Path("/Volumes/NO NAME/Master Lina/Code")
TREMOR_DATA_PATH = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"
ACTIVITY_MODEL_DIR = WORKSPACE_ROOT / "Feature extraction CNNs" / "Models" / "s2_w2_aug2" / "fs50_clean"

# Dataset variants to extract embeddings for
DATASET_VARIANTS = [
    "s2_w2_tremor_clean", 
    "s2_w2_tremor_parkinson_mild", 
    "s2_w2_tremor_parkinson_severe"
]

# Sensors and their activity-trained models
SENSORS = {
    "Acc_arm": ACTIVITY_MODEL_DIR / "feature_extractor_Acc_arm.pth",
    "Gyro_arm": ACTIVITY_MODEL_DIR / "feature_extractor_Gyro_arm.pth"
}

# Split configuration (must match training)
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]

# Architecture parameters
num_channels = 3
seq_len = 100
batch_size = 64


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (must match training)
# ═══════════════════════════════════════════════════════════════

class IMUCNN(nn.Module):
    """
    CNN architecture for activity classification (used for feature extraction).
    
    Supports two architectures:
    - Architecture A (128→256): Acc_arm, Acc_ankle, Mag_arm, Mag_ankle
    - Architecture B (256→128): Gyro_arm, Gyro_ankle, ECG_chest, Acc_chest
    """
    def __init__(self, num_classes, seq_len, num_channels, conv1_out=128, conv2_out=256, dropout1=0.3, dropout2=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, conv1_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout1),

            nn.Conv1d(conv1_out, conv2_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout2),
        )

        self.flattened_dim = (seq_len // 4) * conv2_out

        # Embedding layer (feature representation)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)

        # Classification head
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        """Extract 128-dim embeddings (before classification head)."""
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc_embed(x)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        return self.fc_cls(z)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_tremor_data(sensor_name: str, dataset_path: Path):
    """Load tremor dataset from TXT file."""
    file_path = dataset_path / f"{sensor_name}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"  Loading {sensor_name}...")
    data = np.loadtxt(file_path, delimiter=",")
    
    # Data format (tremor-compatible):
    # [...sensor data...] | [-7] activity | [-6] subject | [-5] base_idx |
    # [-4] tremor_freq | [-3] tremor_acc_rms | [-2] tremor_gyro_rms | [-1] tremor_score
    X = data[:, :-7]  # Sensor data
    y_activity = data[:, -7].astype(np.int64) - 1  # 0-indexed activity
    y_subject = data[:, -6].astype(np.int64)   # 1-indexed subject
    y_tremor_score = data[:, -1].astype(np.int64)  # Tremor score 0-4
    
    # Reshape to (N, C, L)
    N = data.shape[0]
    X = X.reshape(N, num_channels, seq_len).astype(np.float32)
    
    return {
        "X": X,
        "y_activity": y_activity,
        "y_subject": y_subject,
        "y_tremor_score": y_tremor_score,
    }


def split_data_by_subject(data, val_subjects, test_subjects):
    """Split data by subject IDs."""
    subjects = data["y_subject"]
    
    train_mask = ~np.isin(subjects, val_subjects + test_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    train_data = {k: v[train_mask] for k, v in data.items()}
    val_data = {k: v[val_mask] for k, v in data.items()}
    test_data = {k: v[test_mask] for k, v in data.items()}
    
    return train_data, val_data, test_data


# ═══════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_embeddings(model, X, device):
    """Extract embeddings from model."""
    model.eval()
    
    # Create dataset and loader (no shuffling to preserve order)
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract embeddings
    all_embeddings = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            embeddings = model.extract_features(X_batch)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def extract_sensor_embeddings(sensor_name: str, model_path: Path, variant_path: Path, output_dir: Path):
    """
    Extract embeddings for one sensor from one dataset variant using activity-trained model.
    
    Args:
        sensor_name: Sensor name (Acc_arm or Gyro_arm)
        model_path: Path to activity-trained model checkpoint
        variant_path: Path to tremor dataset variant
        output_dir: Output directory for embeddings
    """
    print(f"\n{'='*70}")
    print(f"Extracting {sensor_name} embeddings using activity-trained model")
    print(f"{'='*70}")
    
    # Load activity-trained model
    if not model_path.exists():
        raise FileNotFoundError(f"Activity model not found: {model_path}")
    
    print(f"  Loading activity-trained model: {model_path.name}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model parameters
    num_classes = checkpoint['num_classes']
    seq_len_ckpt = checkpoint['seq_len']
    num_channels_ckpt = checkpoint['num_channels']
    
    if seq_len_ckpt != seq_len or num_channels_ckpt != num_channels:
        raise ValueError(
            f"Model mismatch: expected seq_len={seq_len}, num_channels={num_channels}, "
            f"got seq_len={seq_len_ckpt}, num_channels={num_channels_ckpt}"
        )
    
    # Infer architecture from checkpoint (read conv layer dimensions)
    state_dict = checkpoint['model_state_dict']
    conv1_out = state_dict['features.0.weight'].shape[0]  # Output channels of first conv
    conv2_out = state_dict['features.5.weight'].shape[0]  # Output channels of second conv
    
    print(f"  Detected architecture: Conv1d({num_channels}→{conv1_out}), Conv1d({conv1_out}→{conv2_out})")
    
    # Create model and load weights
    model = IMUCNN(num_classes, seq_len, num_channels, conv1_out=conv1_out, conv2_out=conv2_out).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✓ Model loaded (num_classes={num_classes}, embedding_dim=128)")
    
    # Load tremor data
    print(f"  Loading tremor data from: {variant_path.name}")
    data = load_tremor_data(sensor_name, variant_path)
    
    # Split data by subject
    train_data, val_data, test_data = split_data_by_subject(data, VAL_SUBJECTS, TEST_SUBJECTS)
    
    print(f"  Split sizes: Train={len(train_data['X'])}, Val={len(val_data['X'])}, Test={len(test_data['X'])}")
    
    # Extract embeddings
    print(f"  Extracting embeddings...")
    train_embeddings = extract_embeddings(model, train_data['X'], device)
    val_embeddings = extract_embeddings(model, val_data['X'], device)
    test_embeddings = extract_embeddings(model, test_data['X'], device)
    
    print(f"  ✓ Train embeddings: {train_embeddings.shape}")
    print(f"  ✓ Val embeddings:   {val_embeddings.shape}")
    print(f"  ✓ Test embeddings:  {test_embeddings.shape}")
    
    # Save embeddings in tremor-fusion compatible format
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{sensor_name}_embeddings.npz"
    
    np.savez_compressed(
        output_file,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        train_labels=train_data['y_tremor_score'],  # Tremor scores for classification
        val_labels=val_data['y_tremor_score'],
        test_labels=test_data['y_tremor_score'],
        train_activities=train_data['y_activity'],  # Keep for reference
        val_activities=val_data['y_activity'],
        test_activities=test_data['y_activity'],
        train_subjects=train_data['y_subject'],
        val_subjects=val_data['y_subject'],
        test_subjects=test_data['y_subject'],
    )
    
    print(f"  ✓ Saved to: {output_file}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    """Extract embeddings for all sensors and all dataset variants."""
    
    print("\n" + "="*70)
    print("TREMOR EMBEDDING EXTRACTION WITH ACTIVITY-TRAINED MODELS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Activity models from: {ACTIVITY_MODEL_DIR}")
    print(f"Tremor data from: {TREMOR_DATA_PATH}")
    print(f"Sensors: {list(SENSORS.keys())}")
    print(f"Dataset variants: {DATASET_VARIANTS}")
    print("="*70 + "\n")
    
    # Process each dataset variant
    for variant_name in DATASET_VARIANTS:
        variant_path = TREMOR_DATA_PATH / variant_name
        
        if not variant_path.exists():
            print(f"⚠️  Warning: Dataset variant not found: {variant_path}")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# Processing variant: {variant_name}")
        print(f"{'#'*70}")
        
        # Output directory (separate from tremor-trained embeddings)
        output_dir = variant_path / "ExtractedFeatures_ActivityModels"
        
        # Process each sensor
        for sensor_name, model_path in SENSORS.items():
            try:
                extract_sensor_embeddings(sensor_name, model_path, variant_path, output_dir)
            except Exception as e:
                print(f"⚠️  Error extracting {sensor_name} from {variant_name}: {e}")
                continue
    
    print("\n" + "="*70)
    print("✓ EXTRACTION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Update tremor fusion script to load from 'ExtractedFeatures_ActivityModels/'")
    print("  2. Run: python Tremor_head/fusion/training/tremor_classification_feature_fusion.py")
    print("  3. Compare performance: activity-trained vs tremor-trained embeddings")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
