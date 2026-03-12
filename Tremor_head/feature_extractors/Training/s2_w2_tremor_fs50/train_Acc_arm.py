"""
Tremor Feature Extractor Training: Acc_arm CNN
================================================

Trains a CNN feature extractor for Acc_arm sensor on tremor dataset.
The model learns to classify tremor scores (0-4) and can extract 128-dim embeddings.

Architecture (same as data fusion model):
  - Conv1d(3, 128, kernel_size=5) + BatchNorm + ReLU + MaxPool
  - Conv1d(128, 256, kernel_size=5) + BatchNorm + ReLU + MaxPool
  - Flatten + Activity embedding concatenation
  - fc_embed: Linear(combined_dim, 128)  # 128-dim feature embedding
  - fc_cls: Linear(128, 5)  # 5 tremor severity classes

Dataset:
  - Merged from: clean + parkinson_mild + parkinson_severe
  - Split by subject (same as main tremor classification)

Usage:
    python train_acc_arm_extractor.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Import configuration
from config import *

# Sensor name (using tremorbranch - non-normalized for tremor severity estimation)
# SENSOR = "Acc_arm_tremorbranch"
SENSOR = "Acc_arm"

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Reproducibility - Random seed for each run, but logged for reproducibility
SEED = int(time.time() * 1000) % 100000  # Random seed based on timestamp
print(f"Using random seed: {SEED}")
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

# Paths from config
DATA_PATH = data_path
DATASET_VARIANTS = variant_names
MODEL_PATH = model_dir / f"feature_extractor_{SENSOR}.pth"
LOG_FILE = model_dir / f"{SENSOR}_training.jsonl"

# Ensure model directory exists
model_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_tremor_data(sensor_name: str, dataset_path: Path) -> Dict[str, np.ndarray]:
    """Load tremor dataset from TXT file."""
    file_path = dataset_path / f"{sensor_name}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"Loading {sensor_name} from {file_path.name}...")
    data = np.loadtxt(str(file_path), delimiter=",")
    
    # Determine number of channels
    if "ECG" in sensor_name:
        n_channels = 2
        window_length = seq_len  # Use seq_len from config (100 for fs50)
    else:
        n_channels = 3
        window_length = seq_len  # Use seq_len from config (100 for fs50)
    
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


def merge_dataset_variants(sensor_name: str, dataset_variants: List[str], base_path: Path) -> Dict[str, np.ndarray]:
    """Load and concatenate data from multiple dataset variants."""
    all_variant_data = []
    
    for variant in dataset_variants:
        print(f"\nLoading variant: {variant}")
        dataset_path = base_path / variant
        variant_data = load_tremor_data(sensor_name, dataset_path)
        all_variant_data.append(variant_data)
        print(f"  Loaded {variant_data['X'].shape[0]} samples")
    
    # Concatenate all variants
    merged = {}
    for key in all_variant_data[0].keys():
        merged[key] = np.concatenate([data[key] for data in all_variant_data], axis=0)
    
    print(f"\nMerged {len(dataset_variants)} dataset variants:")
    print(f"  Total samples: {merged['X'].shape[0]}")
    print(f"  X shape: {merged['X'].shape}")
    print(f"  Score distribution: {np.bincount(merged['y_tremor_score'], minlength=5)}")
    
    return merged


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
    
    train_subjects = np.unique(train_data["y_subject"])
    print(f"\nTrain subjects: {sorted(train_subjects)}")
    print(f"Val subjects: {sorted(val_subjects)}")
    print(f"Test subjects: {sorted(test_subjects)}")
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data['X'])} samples")
    print(f"  Val:   {len(val_data['X'])} samples")
    print(f"  Test:  {len(test_data['X'])} samples")
    
    return train_data, val_data, test_data


def compute_normalization_stats(X_train: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute mean/std for normalization from training data."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    return {"mean": mean, "std": std}


def normalize_data(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize data using precomputed statistics."""
    return (X - stats["mean"]) / stats["std"]


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class TremorDataset(Dataset):
    """Dataset for tremor severity classification."""
    
    def __init__(self, data: Dict[str, np.ndarray], norm_stats: Dict[str, np.ndarray]):
        self.X = normalize_data(data["X"], norm_stats)
        self.y_activity = data["y_activity"]
        self.y_score = data["y_tremor_score"]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])  # (C, L)
        activity = torch.tensor(self.y_activity[idx], dtype=torch.long)
        score = torch.tensor(self.y_score[idx], dtype=torch.long)
        
        return X, activity, score


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class AccArmFeatureExtractor(nn.Module):
    """
    CNN feature extractor for Acc_arm sensor.
    
    Architecture identical to TremorClassificationCNN but for single sensor.
    Extracts 128-dim embeddings from fc_embed layer.
    """
    
    def __init__(self):
        super().__init__()
        
        # Activity embedding
        self.activity_embed = nn.Embedding(num_activities, activity_embed_dim)
        
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        # Flattened dimension: (seq_len // 4) * 128 = 25 * 128 = 3200
        self.flattened_dim = (seq_len // 4) * 128
        
        # Combined dimension: CNN features + activity embedding
        combined_dim = self.flattened_dim + activity_embed_dim
        
        # Embedding layer (feature representation)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(combined_dim, embed_dim)
        
        # Classification head
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(embed_dim, num_classes)
    
    def extract_features(self, X, activity):
        """Extract 128-dim embeddings (used for feature fusion)."""
        # CNN feature extraction
        x = self.features(X)  # (B, 256, 25)
        cnn_flat = self.flatten(x)  # (B, 6400)
        
        # Activity embedding
        act_emb = self.activity_embed(activity)  # (B, 16)
        
        # Combine
        combined = torch.cat([cnn_flat, act_emb], dim=1)  # (B, 6416)
        
        # Feature embedding
        z = self.fc_embed(combined)  # (B, 128)
        
        return z
    
    def forward(self, X, activity):
        """Forward pass for training (classification)."""
        z = self.extract_features(X, activity)
        z = self.drop_cls(z)
        logits = self.fc_cls(z)
        return logits


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for X, activity, score in loader:
        X = X.to(device)
        activity = activity.to(device)
        score = score.to(device)
        
        optimizer.zero_grad()
        logits = model(X, activity)
        loss = criterion(logits, score)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(score.cpu().numpy())
    
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
        for X, activity, score in loader:
            X = X.to(device)
            activity = activity.to(device)
            score = score.to(device)
            
            logits = model(X, activity)
            loss = criterion(logits, score)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(score.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════
# MODEL VERSIONING
# ═══════════════════════════════════════════════════════════════

def save_model(model_state, optimizer_state, norm_stats, epoch, val_loss, val_acc):
    """Save model (overwrites existing model)."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'norm_stats': norm_stats
    }, MODEL_PATH)
    
    print(f"  Model saved: {MODEL_PATH}")
    return MODEL_PATH


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Ensure valid working directory for numpy
    import os
    os.chdir(DATA_PATH.parent.parent)
    
    print("=" * 80)
    print("TRAINING ACC_ARM FEATURE EXTRACTOR FOR TREMOR CLASSIFICATION")
    print("=" * 80)
    
    # Load and merge datasets
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    data = merge_dataset_variants(SENSOR, DATASET_VARIANTS, DATA_PATH)
    
    # Split by subject
    train_data, val_data, test_data = split_data_by_subject(data, VAL_SUBJECTS, TEST_SUBJECTS)
    
    # Compute normalization stats
    norm_stats = compute_normalization_stats(train_data["X"])
    
    # Create datasets
    train_dataset = TremorDataset(train_data, norm_stats)
    val_dataset = TremorDataset(val_data, norm_stats)
    test_dataset = TremorDataset(test_data, norm_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model = AccArmFeatureExtractor().to(device)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")
        
        # Early stopping based on validation F1-score
        if val_f1 > best_val_f1 + min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model state in memory
            best_model_state = model.state_dict().copy()
            best_optimizer_state = optimizer.state_dict().copy()
            print(f"  → Best model so far (F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for testing
    if best_model_state is None:
        print("ERROR: No model was saved during training!")
        return
    
    model.load_state_dict(best_model_state)
    
    # Test evaluation
    print("\n" + "=" * 80)
    print("TESTING")
    print("=" * 80)
    
    test_loss, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE ACCURACY TO HISTORY (like Feature extraction CNNs)
    # ═══════════════════════════════════════════════════════════════
    model_key = f"{Path(__file__).parent.name}/{SENSOR}"
    history_file = Path(__file__).parent.parent.parent / "accuracy_history.json"
    
    # Load and update accuracy history
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {}
    
    if model_key not in history:
        history[model_key] = []
    
    history[model_key].append(test_f1)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, sort_keys=True)
    
    print(f"\nF1-score added to history: {history_file}")
    
    # ═══════════════════════════════════════════════════════════════
    # CHECK IF THIS IS A NEW BEST ACCURACY
    # ═══════════════════════════════════════════════════════════════
    best_acc_file = Path(__file__).parent.parent.parent / "best_accuracies.json"
    
    # Load existing best accuracies
    if best_acc_file.exists():
        with open(best_acc_file, 'r') as f:
            best_accs = json.load(f)
    else:
        best_accs = {}
    
    # Handle old format (just accuracy) vs new format (dict with f1 and accuracy)
    if model_key in best_accs:
        if isinstance(best_accs[model_key], dict):
            previous_best = best_accs[model_key].get("f1", 0.0)
        else:
            # Old format: just a number (was accuracy)
            previous_best = best_accs[model_key] if isinstance(best_accs[model_key], (int, float)) else 0.0
    else:
        previous_best = 0.0
    
    print(f"\nTest F1: {test_f1:.4f} (Accuracy: {test_acc:.4f})")
    print(f"Previous Best F1: {previous_best:.4f}")
    
    if test_f1 <= previous_best:
        print(f"❌ No improvement in F1-score. Skipping model save.")
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE (no new best)")
        print("=" * 80)
        return
    
    print(f"✅ New best F1-score! Saving model...")
    
    # Get model architecture as string
    model_architecture = str(model)
    
    # Update best accuracies file (now based on F1)
    best_accs[model_key] = {
        "f1": test_f1,
        "accuracy": test_acc,
        "f1": test_f1,
        "test_loss": test_loss,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "val_f1": best_val_f1,
        "architecture": model_architecture,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "random_seed": SEED,
        "timestamp": datetime.now().isoformat()
    }
    with open(best_acc_file, 'w') as f:
        json.dump(best_accs, f, indent=2, sort_keys=True)
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE MODEL (only if new best)
    # ═══════════════════════════════════════════════════════════════
    save_model(best_model_state, best_optimizer_state, norm_stats, best_epoch, best_val_loss, best_val_acc)
    print(f"Model saved: {MODEL_PATH}")
    
    # Save metadata to JSONL log
    results = {
        "sensor": SENSOR,
        "dataset_config": dataset_config,
        "model_variant": model_variant,
        "dataset_variants": DATASET_VARIANTS,
        "test_subjects": TEST_SUBJECTS,
        "val_subjects": VAL_SUBJECTS,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs_trained": best_epoch + 1,
        "model_path": str(MODEL_PATH),
        "model_architecture": model_architecture,
        "random_seed": SEED,
        "is_new_best": True,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(results) + '\n')
    
    print(f"Metadata logged to: {LOG_FILE}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE (new best!)")
    print("=" * 80)


if __name__ == "__main__":
    main()

