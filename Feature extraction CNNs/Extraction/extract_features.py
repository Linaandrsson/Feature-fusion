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
    X = data[:, :-1]
    y = data[:, -1].astype(int) - 1  # 1..12 -> 0..11
    X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
    
    # Load splits from parent directory
    train_idx = np.loadtxt(parent_dir / "train_idx.txt", dtype=int)
    val_idx = np.loadtxt(parent_dir / "val_idx.txt", dtype=int)
    test_idx = np.loadtxt(parent_dir / "test_idx.txt", dtype=int)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


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
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(sensor_name, num_channels)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create data loaders (no shuffling for alignment)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size, shuffle=False
    )
    
    num_classes = len(np.unique(y_train))
    
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
    
    # Save features
    print(f"\n[4/4] Saving features...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_dir / f"{sensor_name}_train.npz",
        embeddings=Z_train,
        labels=y_train_check
    )
    np.savez_compressed(
        output_dir / f"{sensor_name}_val.npz",
        embeddings=Z_val,
        labels=y_val_check
    )
    np.savez_compressed(
        output_dir / f"{sensor_name}_test.npz",
        embeddings=Z_test,
        labels=y_test_check
    )
    
    print(f"  ✓ Saved to: {output_dir}")
    print(f"\n{'='*60}")
    print(f"✓ Feature extraction complete for {sensor_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for a sensor")
    parser.add_argument("--sensor", type=str, required=True, 
                       help=f"Sensor name. Options: {list(SENSORS.keys())}")
    args = parser.parse_args()
    
    main(args.sensor)
