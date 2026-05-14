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
# If called from extract_all_corrupt_variants.py, env vars override config_extraction.py
import os as _os
import importlib as _importlib

_cfg = _importlib.import_module("config_extraction")

_variant_name    = _os.environ.get("EXTRACTION_VARIANT_NAME")
_dataset_config  = _os.environ.get("EXTRACTION_DATASET_CONFIG")
_model_variant   = _os.environ.get("EXTRACTION_MODEL_VARIANT")
_output_subdir   = _os.environ.get("EXTRACTION_OUTPUT_SUBDIR")  # overrides output folder name

if _variant_name:
    _workspace_root = Path(_cfg.__file__).parents[2]
    _data_base      = _workspace_root / "data" / "Tremor_datagenerator_files"
    variant_dir     = _data_base / _variant_name
    _variant_splits = variant_dir / "splits"
    _training_splits = _workspace_root / "Feature extraction CNNs" / "Training" / "s4_w4_aug1_fs50_clean" / "splits"
    parent_dir      = _variant_splits if _variant_splits.exists() else _training_splits
    _dc             = _dataset_config or _cfg.dataset_config
    _mv             = _model_variant  or _cfg.model_variant
    models_dir      = _workspace_root / "Feature extraction CNNs" / "Models" / _dc / _mv
    output_dir      = variant_dir / (_output_subdir or _cfg.output_dir.name)

    def get_model_path(sensor_name):
        return models_dir / f"feature_extractor_{sensor_name}.pth"

    def get_data_path(sensor_name):
        return variant_dir / f"{sensor_name}.txt"
else:
    parent_dir    = _cfg.parent_dir
    variant_dir   = _cfg.variant_dir
    models_dir    = _cfg.models_dir
    output_dir    = _cfg.output_dir
    get_model_path = _cfg.get_model_path
    get_data_path  = _cfg.get_data_path

seq_len    = _cfg.seq_len
batch_size = _cfg.batch_size
SENSORS    = _cfg.SENSORS


class IMUCNN(nn.Module):
    """CNN model for IMU feature extraction.

    Architecture is inferred from the checkpoint via build_model_from_ckpt().
    ch1 / ch2 are the out-channels of the 1st and 2nd Conv1d blocks.
    has_bn_embed adds a BatchNorm after fc_embed (used by the ECG model).
    """
    def __init__(self, num_classes: int, seq_len: int, num_channels: int,
                 ch1: int = 128, ch2: int = 128, has_bn_embed: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, ch1, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.flattened_dim = (seq_len // 4) * ch2
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)
        self.has_bn_embed = has_bn_embed
        if has_bn_embed:
            self.bn_embed = nn.BatchNorm1d(128)
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc_embed(x)
        if self.has_bn_embed:
            z = self.bn_embed(z)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        return self.fc_cls(z)


def build_model_from_ckpt(ckpt: dict, num_channels: int, seq_len: int, num_classes: int) -> "IMUCNN":
    """Instantiate IMUCNN with the architecture inferred from a checkpoint's state_dict."""
    sd = ckpt["model_state_dict"]
    ch1 = sd["features.0.weight"].shape[0]
    ch2 = sd["features.5.weight"].shape[0]
    has_bn_embed = "bn_embed.weight" in sd
    return IMUCNN(num_classes=num_classes, seq_len=seq_len, num_channels=num_channels,
                  ch1=ch1, ch2=ch2, has_bn_embed=has_bn_embed)


def load_data(sensor_name: str, num_channels: int):
    """Load sensor data and split by subject ID (identical to training script logic)."""
    # Subject-based splits — must match training config
    TEST_SUBJECTS = [5, 10]
    VAL_SUBJECTS  = [2, 7]

    data_path = get_data_path(sensor_name)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = np.loadtxt(data_path, delimiter=",")

    # Data format (tremor-compatible):
    # [...sensor data...] | [-7] activity | [-6] subject | [-5] base_idx |
    # [-4] tremor_freq | [-3] tremor_acc_rms | [-2] tremor_gyro_rms | [-1] tremor_score
    X = data[:, :-7]  # Sensor data only (all columns except last 7)
    y_activity  = data[:, -7].astype(int) - 1   # Activity label 1..12 -> 0..11
    y_subject   = data[:, -6].astype(int)        # Subject ID
    base_idx    = data[:, -5].astype(int)
    tremor_freq     = data[:, -4].astype(np.float32)
    tremor_acc_rms  = data[:, -3].astype(np.float32)
    tremor_gyro_rms = data[:, -2].astype(np.float32)
    tremor_score    = data[:, -1].astype(int)

    X = X.reshape(-1, num_channels, seq_len).astype(np.float32)

    # Subject-based splitting (identical to Acc_ankle_CNN.py in Training/)
    train_idx = np.where(~np.isin(y_subject, TEST_SUBJECTS + VAL_SUBJECTS))[0]
    val_idx   = np.where( np.isin(y_subject, VAL_SUBJECTS))[0]
    test_idx  = np.where( np.isin(y_subject, TEST_SUBJECTS))[0]

    print(f"  Subject-based splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"    train subjects: {sorted(set(y_subject[train_idx]))}")
    print(f"    val   subjects: {sorted(set(y_subject[val_idx]))}")
    print(f"    test  subjects: {sorted(set(y_subject[test_idx]))}")

    def _meta(idx):
        return {
            "activity":      y_activity[idx],
            "subject":       y_subject[idx],
            "base_idx":      base_idx[idx],
            "tremor_freq":   tremor_freq[idx],
            "tremor_acc_rms":  tremor_acc_rms[idx],
            "tremor_gyro_rms": tremor_gyro_rms[idx],
            "tremor_score":  tremor_score[idx],
        }

    return (X[train_idx], X[val_idx], X[test_idx]), (_meta(train_idx), _meta(val_idx), _meta(test_idx))


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
    print(f"Variant dir (data):  {variant_dir}")
    print(f"Models dir:          {models_dir}")
    print(f"Output dir:          {output_dir}")
    print(f"Splitting:           subject-based (TEST=[5,10], VAL=[2,7])")
    
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
    
    ckpt = torch.load(model_path, map_location=device)
    model = build_model_from_ckpt(ckpt, num_channels=num_channels, seq_len=seq_len, num_classes=num_classes).to(device)
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
