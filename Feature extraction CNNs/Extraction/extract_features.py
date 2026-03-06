"""
Generic feature extraction script for single sensor.

Usage:
    python extract_features.py --sensor Acc_ankle
    
This script:
1. Loads a trained model from Models/
2. Loads sensor data from specified variant
3. Extracts embeddings for train/val/test splits
4. Saves features to variant_dir/ExtractedFeatures/
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse
import sys

# Import configuration
from config_extraction import (
    parent_dir, variant_dir, models_dir, seq_len, batch_size,
    get_model_path, get_data_path, output_dir, SENSORS
)


class IMUCNN(nn.Module):
    """CNN model for IMU feature extraction (must match training architecture)."""
    def __init__(self, num_classes: int, seq_len: int, num_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.flattened_dim = (seq_len // 4) * 128
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc_embed(x)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        return self.fc_cls(z)


def load_data(sensor_name: str, num_channels: int):
    """Load sensor data and splits."""
    # Load sensor data
    data_path = get_data_path(sensor_name)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    data = np.loadtxt(data_path, delimiter=",")
    
    # Data format (tremor-compatible): 
    # [...sensor data...] | [-7] activity | [-6] subject | [-5] base_idx | 
    # [-4] tremor_freq | [-3] tremor_acc_rms | [-2] tremor_gyro_rms | [-1] tremor_score
    X = data[:, :-7]  # Sensor data only (all columns except last 7)
    y_activity = data[:, -7].astype(int) - 1  # Activity label 1..12 -> 0..11
    y_subject = data[:, -6].astype(int)  # Subject ID
    base_idx = data[:, -5].astype(int)  # Base window index
    tremor_freq = data[:, -4].astype(np.float32)  # Tremor frequency
    tremor_acc_rms = data[:, -3].astype(np.float32)  # Tremor ACC RMS
    tremor_gyro_rms = data[:, -2].astype(np.float32)  # Tremor Gyro RMS
    tremor_score = data[:, -1].astype(int)  # Tremor score 0-4
    
    X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
    
    # Load splits from parent directory
    train_idx = np.loadtxt(parent_dir / "train_idx.txt", dtype=int)
    val_idx = np.loadtxt(parent_dir / "val_idx.txt", dtype=int)
    test_idx = np.loadtxt(parent_dir / "test_idx.txt", dtype=int)
    
    # Split all data
    X_train, y_activity_train = X[train_idx], y_activity[train_idx]
    X_val, y_activity_val = X[val_idx], y_activity[val_idx]
    X_test, y_activity_test = X[test_idx], y_activity[test_idx]
    
    # Also split metadata
    meta_train = {
        "activity": y_activity_train,
        "subject": y_subject[train_idx],
        "base_idx": base_idx[train_idx],
        "tremor_freq": tremor_freq[train_idx],
        "tremor_acc_rms": tremor_acc_rms[train_idx],
        "tremor_gyro_rms": tremor_gyro_rms[train_idx],
        "tremor_score": tremor_score[train_idx],
    }
    meta_val = {
        "activity": y_activity_val,
        "subject": y_subject[val_idx],
        "base_idx": base_idx[val_idx],
        "tremor_freq": tremor_freq[val_idx],
        "tremor_acc_rms": tremor_acc_rms[val_idx],
        "tremor_gyro_rms": tremor_gyro_rms[val_idx],
        "tremor_score": tremor_score[val_idx],
    }
    meta_test = {
        "activity": y_activity_test,
        "subject": y_subject[test_idx],
        "base_idx": base_idx[test_idx],
        "tremor_freq": tremor_freq[test_idx],
        "tremor_acc_rms": tremor_acc_rms[test_idx],
        "tremor_gyro_rms": tremor_gyro_rms[test_idx],
        "tremor_score": tremor_score[test_idx],
    }
    
    return (X_train, X_val, X_test), (meta_train, meta_val, meta_test)


def extract_embeddings(model, loader, device):
    """Extract embeddings from data loader."""
    model.eval()
    feats = []
    labs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            z = model.extract_features(xb)
            feats.append(z.cpu().numpy())
            labs.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)


def main(sensor_name: str):
    """Extract features for a single sensor."""
    
    if sensor_name not in SENSORS:
        raise ValueError(f"Unknown sensor: {sensor_name}. Available: {list(SENSORS.keys())}")
    
    num_channels = SENSORS[sensor_name]["num_channels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Extracting features for: {sensor_name}")
    print(f"{'='*60}")
    print(f"Parent dir (splits): {parent_dir}")
    print(f"Variant dir (data):  {variant_dir}")
    print(f"Models dir:          {models_dir}")
    print(f"Output dir:          {output_dir}")
    
    # Load data
    print(f"\n[1/4] Loading data...")
    (X_train, X_val, X_test), (meta_train, meta_val, meta_test) = load_data(sensor_name, num_channels)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create data loaders (no shuffling for alignment)
    # Use activity labels for training (even though we're extracting features for tremor later)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(meta_train["activity"])),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(meta_val["activity"])),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(meta_test["activity"])),
        batch_size=batch_size, shuffle=False
    )
    
    num_classes = len(np.unique(meta_train["activity"]))
    
    # Load model
    print(f"\n[2/4] Loading model...")
    model_path = get_model_path(sensor_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = IMUCNN(num_classes=num_classes, seq_len=seq_len, num_channels=num_channels).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded from: {model_path}")
    
    # Extract features
    print(f"\n[3/4] Extracting features...")
    Z_train, y_train_check = extract_embeddings(model, train_loader, device)
    Z_val, y_val_check = extract_embeddings(model, val_loader, device)
    Z_test, y_test_check = extract_embeddings(model, test_loader, device)
    
    print(f"  Train embeddings: {Z_train.shape}")
    print(f"  Val embeddings:   {Z_val.shape}")
    print(f"  Test embeddings:  {Z_test.shape}")
    
    # Save features with tremor-compatible format
    print(f"\n[4/4] Saving features...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in tremor-fusion compatible format (matching extract_tremor_embeddings.py output)
    np.savez_compressed(
        output_dir / f"{sensor_name}_embeddings.npz",
        train_embeddings=Z_train,
        val_embeddings=Z_val,
        test_embeddings=Z_test,
        train_labels=meta_train["tremor_score"],  # For tremor classification
        val_labels=meta_val["tremor_score"],
        test_labels=meta_test["tremor_score"],
        train_activities=meta_train["activity"],  # Keep activity for reference
        val_activities=meta_val["activity"],
        test_activities=meta_test["activity"],
        train_subjects=meta_train["subject"],
        val_subjects=meta_val["subject"],
        test_subjects=meta_test["subject"],
    )
    
    print(f"  ✓ Saved to: {output_dir / f'{sensor_name}_embeddings.npz'}")
    print(f"\n{'='*60}")
    print(f"✓ Feature extraction complete for {sensor_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for a sensor")
    parser.add_argument("--sensor", type=str, required=True, 
                       help=f"Sensor name. Options: {list(SENSORS.keys())}")
    args = parser.parse_args()
    
    main(args.sensor)
