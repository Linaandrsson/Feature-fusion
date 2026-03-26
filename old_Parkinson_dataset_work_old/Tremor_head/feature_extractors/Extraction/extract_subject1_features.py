#!/usr/bin/env python3
"""
Extract Features from Subject 1 Tremor Dataset

Goal:
- Load Subject 1 tremor data variants (clean, mild_mod, mod_severe)
- Use trained feature extractors (Acc_arm, Gyro_arm, Mag_arm) to extract embeddings
- Save embeddings for downstream fusion classifier
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, List


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Configuration & Paths
# ============================================================

# Auto-detect paths
script_dir = Path(__file__).parent.resolve()  # Extraction/
feature_extractors_dir = script_dir.parent  # feature_extractors/
tremor_head_dir = feature_extractors_dir.parent  # Tremor_head/
WORKSPACE_ROOT = tremor_head_dir.parent  # Parkinson_dataset_work/

DATA_ROOT = WORKSPACE_ROOT / "Data"
SUBJECT1_DATA = DATA_ROOT / "Tremor_datagenerator_files_subject1"
MODELS_DIR = feature_extractors_dir / "models" / "s2_w2_tremor" / "fs50_mixed3"
OUTPUT_DIR = script_dir / "subject1_embeddings"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENSORS = ["Acc_arm", "Gyro_arm", "Mag_arm"]
TREMOR_VARIANTS = ["clean", "mild_mod", "mod_severe"]

print("=" * 80)
print("SUBJECT 1 FEATURE EXTRACTION")
print("=" * 80)
print(f"Data directory: {SUBJECT1_DATA}")
print(f"Models directory: {MODELS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Device: {DEVICE}")
print("=" * 80)


# ============================================================
# Dataset Class
# ============================================================

class TremorDataset(Dataset):
    """Load tremor dataset from NPZ file."""
    
    def __init__(self, npz_file: Path):
        """Load data from NPZ file."""
        data = np.load(npz_file, allow_pickle=True)
        self.X = data["X"].astype(np.float32)  # (N, C, L)
        self.y = data["y"].astype(np.int64)    # Activity labels
        self.subject_id = data["subject_id"].astype(np.int64)
        self.tremor_score = data["tremor_score"].astype(np.int8)
        self.base_window_idx = data["base_window_idx"].astype(np.int64)
        
        print(f"    Loaded {npz_file.name}: X.shape={self.X.shape}, y.shape={self.y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            "X": torch.from_numpy(self.X[idx]),
            "y": self.y[idx],
            "subject_id": self.subject_id[idx],
            "tremor_score": self.tremor_score[idx],
            "base_window_idx": self.base_window_idx[idx],
        }


# ============================================================
# Feature Extractor Models
# ============================================================

# We need to define or import the feature extractor models
# For now, we'll load them from the checkpoint files


def load_feature_extractor(sensor_name: str, model_path: Path) -> Tuple[nn.Module, Dict[str, np.ndarray]]:
    """Load a trained feature extractor model and its normalization statistics.
    
    Returns:
        (model, norm_stats) where norm_stats contains 'mean' and 'std' for normalization
    
    Raises:
        KeyError: If norm_stats is missing from checkpoint
    """
    
    # Import the model class dynamically
    sys.path.insert(0, str(feature_extractors_dir / "Training" / "s2_w2_tremor_fs50"))
    
    if sensor_name == "Acc_arm":
        from train_Acc_arm import AccArmFeatureExtractor as ModelClass
    elif sensor_name == "Gyro_arm":
        from train_Gyro_arm import GyroArmFeatureExtractor as ModelClass
    elif sensor_name == "Mag_arm":
        from train_Mag_arm import MagArmFeatureExtractor as ModelClass
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")
    
    model = ModelClass().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load and verify normalization stats
    if "norm_stats" not in checkpoint:
        raise KeyError(
            f"ERROR: norm_stats not found in checkpoint for {sensor_name}!\n"
            f"Path: {model_path}\n"
            f"This checkpoint was not saved with normalization statistics."
        )
    
    norm_stats = checkpoint["norm_stats"]
    
    # Verify norm_stats has required keys
    if "mean" not in norm_stats or "std" not in norm_stats:
        raise KeyError(
            f"ERROR: norm_stats incomplete for {sensor_name}!\n"
            f"Expected keys: ['mean', 'std']\n"
            f"Got keys: {list(norm_stats.keys())}"
        )
    
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    print(f"    ✓ Loaded (mean shape: {mean.shape}, std shape: {std.shape})")
    
    return model, norm_stats


# ============================================================
# Normalization Helper
# ============================================================

def normalize_data(X: np.ndarray, norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply normalization using saved statistics.
    
    Args:
        X: Data array of shape (N, C, L)
        norm_stats: Dict with 'mean' and 'std' (each of shape (1, C, 1))
    
    Returns:
        Normalized data: (X - mean) / std
    """
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    return (X - mean) / std


# ============================================================
# Extract Features
# ============================================================

def extract_features_for_variant(variant_name: str) -> None:
    """Extract features for one tremor variant."""
    
    print(f"\n{'=' * 80}")
    print(f"Extracting features: {variant_name}")
    print(f"{'=' * 80}")
    
    variant_dir = SUBJECT1_DATA / f"s2_w2_fs50_tremor_{variant_name}"
    
    if not variant_dir.exists():
        print(f"ERROR: Directory not found: {variant_dir}")
        return
    
    # Load datasets for each sensor
    print("\nLoading data...")
    datasets = {}
    for sensor in SENSORS:
        sensor_file = variant_dir / f"{sensor}.npz"
        if not sensor_file.exists():
            print(f"  ⚠ {sensor} not found: {sensor_file}")
            continue
        
        print(f"  {sensor}:")
        datasets[sensor] = TremorDataset(sensor_file)
    
    if not datasets:
        print(f"  No datasets loaded for {variant_name}")
        return
    
    # Load feature extractors and normalization stats
    print("\nLoading feature extractors and normalization stats...")
    feature_extractors = {}
    norm_stats_dict = {}
    
    for sensor in SENSORS:
        model_path = MODELS_DIR / f"feature_extractor_{sensor}.pth"
        if not model_path.exists():
            print(f"  ⚠ Model not found: {model_path}")
            continue
        
        print(f"  {sensor}:")
        try:
            model, norm_stats = load_feature_extractor(sensor, model_path)
            feature_extractors[sensor] = model
            norm_stats_dict[sensor] = norm_stats
        except KeyError as e:
            print(f"    ERROR: {e}")
            raise
    
    if not feature_extractors:
        print(f"  No feature extractors loaded")
        return
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings_by_sensor = {}
    labels_dict = {}
    
    for sensor in SENSORS:
        if sensor not in datasets or sensor not in feature_extractors:
            continue
        
        print(f"  {sensor}...", end=" ", flush=True)
        
        dataset = datasets[sensor]
        model = feature_extractors[sensor]
        norm_stats = norm_stats_dict[sensor]
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in loader:
                X_batch = batch["X"].numpy()  # Get as numpy first
                y_batch = batch["y"]
                
                # Apply normalization using the saved statistics
                X_batch_normalized = normalize_data(X_batch, norm_stats)
                
                # Convert to tensor and move to device
                X_batch_tensor = torch.from_numpy(X_batch_normalized).to(DEVICE)
                
                # Extract embedding using the model's extract_features method
                embedding = model.extract_features(X_batch_tensor, y_batch.to(DEVICE))
                embeddings.append(embedding.cpu().numpy())
                labels.append(y_batch.numpy())
        
        embeddings_array = np.concatenate(embeddings, axis=0)
        embeddings_by_sensor[sensor] = embeddings_array
        labels_dict[sensor] = np.concatenate(labels, axis=0)
        
        print(f"✓ {embeddings_array.shape}")
    
    # Concatenate embeddings from all sensors
    print("\nConcatenating embeddings...")
    if len(embeddings_by_sensor) != 3:
        print(f"  ⚠ Not all sensors available (have {len(embeddings_by_sensor)}/3)")
    
    all_embeddings = []
    for sensor in SENSORS:
        if sensor in embeddings_by_sensor:
            all_embeddings.append(embeddings_by_sensor[sensor])
    
    if not all_embeddings:
        print("  ERROR: No embeddings to concatenate")
        return
    
    concatenated = np.concatenate(all_embeddings, axis=1)
    print(f"  Concatenated shape: {concatenated.shape}")
    
    # Get metadata
    metadata_sensor = SENSORS[0]  # Use first available sensor for metadata
    if metadata_sensor not in datasets:
        metadata_sensor = list(datasets.keys())[0]
    
    sample_batch = next(iter(DataLoader(datasets[metadata_sensor], batch_size=len(datasets[metadata_sensor]))))
    y = sample_batch["y"].numpy()
    subject_id = sample_batch["subject_id"].numpy()
    tremor_score = sample_batch["tremor_score"].numpy()
    base_window_idx = sample_batch["base_window_idx"].numpy()
    
    # Save embeddings
    print("\nSaving embeddings...")
    output_file = OUTPUT_DIR / f"subject1_embeddings_{variant_name}.npz"
    
    np.savez_compressed(
        output_file,
        embeddings=concatenated,
        y=y,
        subject_id=subject_id,
        tremor_score=tremor_score,
        base_window_idx=base_window_idx,
        embedding_dim=concatenated.shape[1],
        sensors=np.array(SENSORS),
    )
    
    print(f"  ✓ Saved to {output_file.name}")
    print(f"    Shape: {concatenated.shape}")
    print(f"    Label distribution: {np.bincount(y)}")
    print(f"    Tremor score distribution: {np.bincount(tremor_score)}")



# ============================================================
# Main
# ============================================================

def main() -> None:
    """Extract features for all Subject 1 variants."""
    
    for variant in TREMOR_VARIANTS:
        extract_features_for_variant(variant)
    
    print("\n" + "=" * 80)
    print("Feature extraction complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
