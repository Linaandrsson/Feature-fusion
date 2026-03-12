"""
tremor_featureFusion_tremorEmb_fs_study.py

Tremor Classification: Sampling Frequency Combination Study
===========================================================

Systematically tests ALL possible sampling frequency combinations for Acc_arm and Gyro_arm.

For each combination:
  - Loads pre-extracted embeddings from corresponding fs variants
  - Merges clean, mild_mod, and mod_severe tremor types
  - Trains simple MLP fusion classifier
  - Logs results (test F1, accuracy, etc.) to JSONL

Usage:
  - Set FS_CANDIDATES: List of frequencies to test (e.g., [10, 20, 30, 50])
  - Set MAX_COMBOS: Limit number of combinations (None = test all)
  - Run script to test all combinations

Results are logged to: Tremor_head/fusion/logs/tremor_fs_study.jsonl
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import random
import hashlib
from typing import List, Dict, Tuple
from datetime import datetime
from itertools import product
from sklearn.metrics import accuracy_score, f1_score


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - FS COMBINATION STUDY
# ═══════════════════════════════════════════════════════════════

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Data paths
# -------------------------------
WORKSPACE_ROOT = Path("/Volumes/NO NAME/Master Lina/Code")
DATA_PATH = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"

# -------------------------------
# FS Combination Study Configuration
# -------------------------------
# Fixed sensor set (always Acc_arm and Gyro_arm for tremor)
FIXED_SENSORS = ["Acc_arm", "Gyro_arm"]

# Sampling frequency candidates to test
FS_CANDIDATES = [10, 20, 30, 40, 50]  # List of FS values to test for each sensor

# Combination limits
MAX_COMBOS = None  # None = test all combinations, int = limit to first N

# Skip already completed combinations (based on fs_combo_hash in log file)
SKIP_EXISTING = True

# Tremor severity types to merge
TREMOR_TYPES = ["tremor_clean", "tremor_mild_mod", "tremor_mod_severe"]

# -------------------------------
# Training hyperparameters
# -------------------------------
batch_size = 64
epochs = 150
lr = 1e-3
patience = 50
min_delta = 1e-5

# -------------------------------
# Model architecture
# -------------------------------
embed_dim_per_sensor = 128  # Each sensor provides 128-dim embedding
hidden_dims = [128, 64]  # MLP hidden layers
dropout = 0.3
num_classes = 5  # Tremor scores: 0-4

# -------------------------------
# Logging
# -------------------------------
LOG_DIR = WORKSPACE_ROOT / "Tremor_head" / "fusion" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "tremor_fs_study.jsonl"

# -------------------------------
# Global embedding cache (in-memory)
# -------------------------------
_EMBEDDING_CACHE = {}


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def generate_fs_combo_hash(sensor_fs: List[int]) -> str:
    """Generate deterministic hash for a specific FS combination."""
    fs_str = "_".join(str(fs) for fs in sensor_fs)
    hash_obj = hashlib.md5(fs_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def load_existing_combinations(log_file: Path, experiment_id: str) -> set:
    """Load all FS combo hashes for a specific experiment from existing log file."""
    if not log_file.exists():
        return set()
    
    combo_hashes = set()
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get('experiment_id') == experiment_id:
                        combo_hashes.add(data.get('fs_combo_hash', ''))
                except json.JSONDecodeError:
                    continue
    
    return combo_hashes


def load_embeddings(variant_name: str, sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Load pre-extracted embeddings for one sensor from one dataset variant.
    Uses global cache to avoid redundant loading.
    
    Returns:
        Dictionary with train/val/test embeddings, labels, activities, subjects
    """
    cache_key = (variant_name, sensor_name)
    
    # Check cache first
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    
    # Load from disk
    embeddings_path = DATA_PATH / variant_name / "Tremor_ExtractedFeatures" / f"{sensor_name}_embeddings.npz"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            f"Please run extract_tremor_embeddings.py first!"
        )
    
    data = np.load(embeddings_path)
    
    result = {
        "train_embeddings": data["train_embeddings"].astype(np.float32),
        "val_embeddings": data["val_embeddings"].astype(np.float32),
        "test_embeddings": data["test_embeddings"].astype(np.float32),
        "train_labels": data["train_labels"].astype(np.int64),
        "val_labels": data["val_labels"].astype(np.int64),
        "test_labels": data["test_labels"].astype(np.int64),
    }
    
    # Cache it
    _EMBEDDING_CACHE[cache_key] = result
    
    return result


def verify_label_alignment(embeddings_dict: Dict, split: str) -> None:
    """Verify that labels are identical across all sensors and tremor types."""
    reference_labels = None
    reference_key = None
    
    for key, data in embeddings_dict.items():
        labels = data[f"{split}_labels"]
        if reference_labels is None:
            reference_labels = labels
            reference_key = key
        else:
            if not np.array_equal(labels, reference_labels):
                raise ValueError(
                    f"Label mismatch in {split} split!\n"
                    f"  Reference: {reference_key}\n"
                    f"  Current: {key}\n"
                )


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class TremorEmbeddingDataset(Dataset):
    """
    Dataset for tremor fusion using pre-extracted embeddings.
    Concatenates embeddings from multiple sensors along feature dimension.
    """
    
    def __init__(self, embeddings_list: List[np.ndarray], labels: np.ndarray):
        """
        Args:
            embeddings_list: List of embedding arrays, one per sensor
            labels: Tremor severity labels (0-4)
        """
        # Concatenate embeddings along feature dimension
        self.embeddings = np.concatenate(embeddings_list, axis=1).astype(np.float32)
        self.labels = labels.astype(np.int64)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class FeatureFusionClassifier(nn.Module):
    """
    Simple MLP classifier for concatenated sensor embeddings.
    
    Architecture:
      - Input: Concatenated embeddings (256-dim for 2 sensors)
      - Hidden layers with dropout
      - Output: 5-class classification (tremor scores 0-4)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, D) concatenated sensor embeddings
        
        Returns:
            (B, num_classes) logits
        """
        return self.classifier(embeddings)


# ═══════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1


def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def run_one_experiment(
    sensors: List[str],
    sensor_fs: List[int],
    experiment_id: str,
    run_timestamp: str
) -> Dict:
    """
    Run one complete training experiment for a given FS configuration.
    
    Args:
        sensors: List of sensor names (e.g., ["Acc_arm", "Gyro_arm"])
        sensor_fs: List of sampling rates (one per sensor)
        experiment_id: Unique experiment identifier
        run_timestamp: Timestamp for this run
    
    Returns:
        Dictionary with results (accuracy, F1, etc.)
    """
    
    # Create sensor-to-fs mapping
    fs_map = {sensors[i]: sensor_fs[i] for i in range(len(sensors))}
    
    # Generate variant names for each sensor and tremor type
    sensor_variants = {}
    for sensor, freq in fs_map.items():
        sensor_variants[sensor] = [
            f"s2_w2_fs{freq}_{tremor_type}" for tremor_type in TREMOR_TYPES
        ]
    
    print(f"\nVariant mapping:")
    for sensor, variants in sensor_variants.items():
        print(f"  {sensor} (fs{fs_map[sensor]}): {variants}")
    
    # Load embeddings for all combinations
    embeddings_dict = {}
    for sensor, variants in sensor_variants.items():
        for variant in variants:
            try:
                data = load_embeddings(variant, sensor)
                embeddings_dict[(sensor, variant)] = data
            except FileNotFoundError as e:
                print(f"  ✗ Missing: {sensor} - {variant}")
                raise
    
    print(f"  Loaded {len(embeddings_dict)} embedding files from cache")
    
    # Merge tremor types for each sensor
    # For each split, concatenate all 3 tremor types along sample axis
    merged_data = {}
    
    for sensor, variants in sensor_variants.items():
        for split in ["train", "val", "test"]:
            # Collect embeddings and labels from all tremor types
            split_embeddings = []
            split_labels = []
            
            for variant in variants:
                data = embeddings_dict[(sensor, variant)]
                split_embeddings.append(data[f"{split}_embeddings"])
                split_labels.append(data[f"{split}_labels"])
            
            # Concatenate along sample axis
            merged_embeddings = np.concatenate(split_embeddings, axis=0)
            merged_labels = np.concatenate(split_labels, axis=0)
            
            merged_data[(sensor, split)] = {
                "embeddings": merged_embeddings,
                "labels": merged_labels
            }
    
    # Verify all splits have same number of samples and labels match
    for split in ["train", "val", "test"]:
        n_samples = None
        reference_labels = None
        
        for sensor in sensors:
            data = merged_data[(sensor, split)]
            
            if n_samples is None:
                n_samples = len(data["labels"])
                reference_labels = data["labels"]
            else:
                if len(data["labels"]) != n_samples:
                    raise ValueError(f"Sample count mismatch in {split} split for {sensor}")
                if not np.array_equal(data["labels"], reference_labels):
                    raise ValueError(f"Label mismatch in {split} split for {sensor}")
    
    # Get dataset info
    embed_dim = embed_dim_per_sensor * len(sensors)
    
    # Prepare data for datasets: list of embeddings per sensor
    train_embeddings = [merged_data[(sensor, "train")]["embeddings"] for sensor in sensors]
    val_embeddings = [merged_data[(sensor, "val")]["embeddings"] for sensor in sensors]
    test_embeddings = [merged_data[(sensor, "test")]["embeddings"] for sensor in sensors]
    
    train_labels = merged_data[(sensors[0], "train")]["labels"]
    val_labels = merged_data[(sensors[0], "val")]["labels"]
    test_labels = merged_data[(sensors[0], "test")]["labels"]
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_labels)} samples")
    print(f"  Val:   {len(val_labels)} samples")
    print(f"  Test:  {len(test_labels)} samples")
    print(f"  Embedding dim: {embed_dim}")
    
    # Create datasets
    train_dataset = TremorEmbeddingDataset(train_embeddings, train_labels)
    val_dataset = TremorEmbeddingDataset(val_embeddings, val_labels)
    test_dataset = TremorEmbeddingDataset(test_embeddings, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = FeatureFusionClassifier(
        input_dim=embed_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    
    print(f"\nTraining:")
    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.3f} F1={train_f1:.3f} | "
                  f"Val: Loss={val_loss:.4f} Acc={val_acc:.3f} F1={val_f1:.3f}")
        
        # Early stopping based on validation F1
        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    # Test evaluation
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, device)
    
    # Generate combo hash
    fs_combo_hash = generate_fs_combo_hash(sensor_fs)
    
    # Return results
    results = {
        "experiment_id": experiment_id,
        "fs_combo_hash": fs_combo_hash,
        "run_timestamp": run_timestamp,
        "sensors": sensors,
        "fs_map": fs_map,
        "val_accuracy": float(best_val_acc),
        "val_f1": float(best_val_f1),
        "val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_loss": float(test_loss),
        "seed": SEED,
        "tremor_types_merged": TREMOR_TYPES,
    }
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_only = datetime.now().strftime("%H-%M-%S")
    
    # Generate descriptive experiment ID for this run (shared by all combinations)
    num_sensors = len(FIXED_SENSORS)
    fs_str = "_".join(str(fs) for fs in FS_CANDIDATES)
    experiment_id = f"tremor_fs_study_k{num_sensors}_fs{fs_str}_{time_only}"
    
    print("="*70)
    print("TREMOR CLASSIFICATION: SAMPLING FREQUENCY COMBINATION STUDY")
    print("="*70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Run timestamp: {run_timestamp}")
    print(f"\nConfiguration:")
    print(f"  Fixed sensors: {FIXED_SENSORS}")
    print(f"  FS candidates: {FS_CANDIDATES}")
    print(f"  Tremor types: {TREMOR_TYPES}")
    print(f"  Dataset path: {DATA_PATH}")
    
    # Generate all FS combinations
    all_fs_combinations = list(product(FS_CANDIDATES, repeat=num_sensors))
    total_combinations = len(all_fs_combinations)
    
    print(f"\nTotal combinations: {total_combinations}")
    print(f"  ({num_sensors} sensors × {len(FS_CANDIDATES)} frequencies = {len(FS_CANDIDATES)}^{num_sensors})")
    
    # Apply MAX_COMBOS limit
    if MAX_COMBOS is not None and MAX_COMBOS < total_combinations:
        all_fs_combinations = all_fs_combinations[:MAX_COMBOS]
        print(f"  Limited to first {MAX_COMBOS} combinations")
    
    # Load existing combinations if skipping
    existing_combos = set()
    if SKIP_EXISTING:
        existing_combos = load_existing_combinations(LOG_FILE, experiment_id)
        if existing_combos:
            print(f"\nFound {len(existing_combos)} existing combinations for this experiment")
    
    # Run experiments
    print(f"\n{'='*70}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*70}\n")
    
    completed = 0
    skipped = 0
    
    for combo_idx, fs_assignment in enumerate(all_fs_combinations, 1):
        sensor_fs = list(fs_assignment)
        
        # Generate combo hash
        combo_hash = generate_fs_combo_hash(sensor_fs)
        
        # Check if already exists
        if SKIP_EXISTING and combo_hash in existing_combos:
            skipped += 1
            print(f"[{combo_idx}/{len(all_fs_combinations)}] Skipping {combo_hash} (already completed)")
            continue
        
        # Print header
        print(f"\n{'*'*70}")
        print(f"Combination {combo_idx}/{len(all_fs_combinations)}")
        print(f"{'*'*70}")
        fs_map = {FIXED_SENSORS[i]: sensor_fs[i] for i in range(num_sensors)}
        print(f"FS map: {fs_map}")
        print(f"Combo hash: {combo_hash}")
        
        try:
            # Run experiment
            result = run_one_experiment(
                sensors=FIXED_SENSORS,
                sensor_fs=sensor_fs,
                experiment_id=experiment_id,
                run_timestamp=run_timestamp
            )
            
            # Print summary
            print(f"\nResults:")
            print(f"  Val F1:   {result['val_f1']:.4f} (Acc: {result['val_accuracy']:.4f})")
            print(f"  Test F1:  {result['test_f1']:.4f} (Acc: {result['test_accuracy']:.4f})")
            
            # Log to file
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            print(f"\n✓ Logged to {LOG_FILE}")
            completed += 1
            
        except Exception as e:
            print(f"\n✗ Error in combination {combo_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*70}")
    print("STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\nCompleted: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(all_fs_combinations)}")
    print(f"\nResults logged to: {LOG_FILE}")
    print(f"Cache size: {len(_EMBEDDING_CACHE)} embeddings loaded")


if __name__ == "__main__":
    main()
