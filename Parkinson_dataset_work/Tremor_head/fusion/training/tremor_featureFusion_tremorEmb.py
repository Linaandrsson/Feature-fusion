"""
Tremor Classification: Feature-Level Fusion
============================================

Uses pre-extracted embeddings from Acc_arm, Gyro_arm, and Mag_arm CNNs for binary tremor classification.

Architecture:
  - Input: Concatenated embeddings (Acc_arm: 128-dim + Gyro_arm: 128-dim + Mag_arm: 128-dim = 384-dim)
  - Simple MLP classifier with dropout
  - Output: Binary tremor classification (0=no tremor, 1=tremor present)

Data Flow:
  1. Train feature extractors: train_Acc_arm.py, train_Gyro_arm.py, train_Mag_arm.py
  2. Extract embeddings: extract_tremor_embeddings.py
  3. Train fusion classifier: THIS FILE

Usage:
    python tremor_featureFusion_tremorEmb.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# -------------------------------
# ⚙️  SAMPLING FREQUENCY CONFIGURATION
# -------------------------------
# Specify sampling frequency for each sensor (in Hz)
# Order corresponds to SENSORS list below

# Sensors to fuse
SENSORS = ["Acc_arm", "Gyro_arm", "Mag_arm"]

# Sampling frequencies per sensor (must match length of SENSORS)
SAMPLING_FREQ = [50, 50, 50]  # All sensors: 50 Hz (FS50)

# Create sensor-to-frequency mapping
sensor_freq_map = dict(zip(SENSORS, SAMPLING_FREQ))

print(f"\n{'='*80}")
print(f"TREMOR FUSION CONFIGURATION")
print(f"{'='*80}")
for sensor, freq in sensor_freq_map.items():
    print(f"{sensor}: {freq} Hz ({2*freq} samples per window)")
print(f"{'='*80}\n")

# Reproducibility
SEED = random.randint(10000, 99999)  # Random seed for each run
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

print(f"Using random seed: {SEED}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Auto-detect workspace root based on script location
# This script is at: .../Parkinson_dataset_work/Tremor_head/fusion/training/tremor_featureFusion_tremorEmb.py
script_dir = Path(__file__).parent.resolve()  # training/
fusion_dir = script_dir.parent  # fusion/
tremor_head_dir = fusion_dir.parent  # Tremor_head/
WORKSPACE_ROOT = tremor_head_dir.parent  # Parkinson_dataset_work/

# Data paths
DATA_PATH = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"

# Auto-generate dataset variants per sensor based on their sampling frequency
TREMOR_TYPES = ["tremor_clean", "tremor_mild_mod", "tremor_mod_severe"]

# Create sensor-specific dataset variants
sensor_variants = {}
for sensor, freq in sensor_freq_map.items():
    sensor_variants[sensor] = [
        f"s2_w2_fs{freq}_{tremor_type}" for tremor_type in TREMOR_TYPES
    ]

# Training hyperparameters
batch_size = 64
epochs = 150
lr = 1e-3
patience = 50
min_delta = 1e-5

# Model architecture
embed_dim_per_sensor = 128  # Each sensor provides 128-dim embedding
total_embed_dim = len(SENSORS) * embed_dim_per_sensor  # 384-dim (3 sensors)
hidden_dims = [256, 128, 64]  # MLP hidden layers (adjusted for 384-dim input)
dropout = 0.3
num_classes = 2  # Binary tremor classification: 0=no tremor, 1=tremor

# Generate combined frequency string for paths (e.g., "acc30_gyro50")
freq_str = "_".join([f"{sensor.lower().replace('_', '')}{freq}" for sensor, freq in sensor_freq_map.items()])

# Logging
LOG_DIR = tremor_head_dir / "fusion" / "logs" / freq_str
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = tremor_head_dir / "fusion" / "plots" / freq_str
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feature_fusion_training.jsonl"

# Model paths
MODEL_DIR = tremor_head_dir / "fusion" / "models" / freq_str
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BEST_PATH = MODEL_DIR / f"feature_fusion_best_{freq_str}.pth"

# History files
BEST_ACCURACIES_FILE = Path(__file__).parent / "best_accuracies.json"
ACCURACY_HISTORY_FILE = LOG_DIR / f"accuracy_history_{freq_str}.json"

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_embeddings(variant_name: str, sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Load pre-extracted embeddings for one sensor from one dataset variant.
    Converts tremor severity scores (0-5) to binary classification (0=no tremor, 1=tremor).
    
    Returns:
        Dictionary with train/val/test embeddings, labels, activities, subjects
    """
    embeddings_path = DATA_PATH / variant_name / "Tremor_ExtractedFeatures" / f"{sensor_name}_embeddings.npz"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            f"Please run extract_tremor_embeddings.py first!"
        )
    
    print(f"  Loading {sensor_name} from {embeddings_path.name}")
    print(f"     Full path: {embeddings_path}")
    data = np.load(embeddings_path)
    print(f"     Keys in file: {list(data.keys())}")
    
    # ═════════════════════════════════════════════════════════════════
    # DEBUG: Show raw labels before binarization
    # ═════════════════════════════════════════════════════════════════
    print(f"\n     [DEBUG] Raw labels before binarization:")
    print(f"       train_labels (first 20): {data['train_labels'][:20]}")
    print(f"       val_labels (first 20):   {data['val_labels'][:20]}")
    print(f"       test_labels (first 20):  {data['test_labels'][:20]}")
    print(f"       train_labels unique: {np.unique(data['train_labels'])}")
    print(f"       val_labels unique:   {np.unique(data['val_labels'])}")
    print(f"       test_labels unique:  {np.unique(data['test_labels'])}")
    
    # Convert tremor severity scores (0-5) to binary (0=no tremor, 1=tremor present)
    train_labels = (data["train_labels"] > 0).astype(np.int64)
    print(f"\n     [DEBUG] Binarized train labels (0=no tremor, 1=tremor):")
    print(f"       data train_binary (first 20): {data['train_labels'][:20]}")
    print(f" train labels: {train_labels[:20]}")
    
    val_labels = (data["val_labels"] > 0).astype(np.int64)
    test_labels = (data["test_labels"] > 0).astype(np.int64)
    
    # ═════════════════════════════════════════════════════════════════
    # DEBUG: Show binary labels after binarization
    # ═════════════════════════════════════════════════════════════════
    print(f"\n     [DEBUG] Binary labels after binarization (>0):")
    print(f"       train_binary (first 20): {train_labels[:20]}")
    print(f"       val_binary (first 20):   {val_labels[:20]}")
    print(f"       test_binary (first 20):  {test_labels[:20]}")
    print(f"       train_binary distribution: {np.bincount(train_labels)}")
    print(f"       val_binary distribution:   {np.bincount(val_labels)}")
    print(f"       test_binary distribution:  {np.bincount(test_labels)}")
    
    return {
        "train_embeddings": data["train_embeddings"],
        "val_embeddings": data["val_embeddings"],
        "test_embeddings": data["test_embeddings"],
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_activities": data["train_activities"],
        "val_activities": data["val_activities"],
        "test_activities": data["test_activities"],
        "train_subjects": data["train_subjects"],
        "val_subjects": data["val_subjects"],
        "test_subjects": data["test_subjects"],
    }


def merge_sensor_embeddings(sensor_variants_dict: Dict[str, List[str]], tremor_type_idx: int) -> Dict[str, np.ndarray]:
    """
    Load and concatenate embeddings from multiple sensors, each from their own variant.
    
    Args:
        sensor_variants_dict: Dict mapping sensor name -> list of variant names
        tremor_type_idx: Index into the tremor type list (clean=0, mild_mod=1, mod_severe=2)
    
    Returns:
        Dictionary with concatenated train/val/test embeddings
    """
    all_sensor_data = []
    for sensor, variants in sensor_variants_dict.items():
        variant_name = variants[tremor_type_idx]
        sensor_data = load_embeddings(variant_name, sensor)
        all_sensor_data.append(sensor_data)
    
    # Verify labels match across sensors
    for key in ["train_labels", "val_labels", "test_labels", "train_activities", "val_activities", "test_activities"]:
        ref = all_sensor_data[0][key]
        sensor_list = list(sensor_variants_dict.keys())
        for i, data in enumerate(all_sensor_data[1:], 1):
            if not np.array_equal(data[key], ref):
                raise ValueError(f"Label mismatch ({key}) between {sensor_list[0]} and {sensor_list[i]}")
    
    # Concatenate embeddings along feature dimension
    merged = {
        "train_embeddings": np.concatenate([d["train_embeddings"] for d in all_sensor_data], axis=1),
        "val_embeddings": np.concatenate([d["val_embeddings"] for d in all_sensor_data], axis=1),
        "test_embeddings": np.concatenate([d["test_embeddings"] for d in all_sensor_data], axis=1),
        "train_labels": all_sensor_data[0]["train_labels"],
        "val_labels": all_sensor_data[0]["val_labels"],
        "test_labels": all_sensor_data[0]["test_labels"],
        "train_activities": all_sensor_data[0]["train_activities"],
        "val_activities": all_sensor_data[0]["val_activities"],
        "test_activities": all_sensor_data[0]["test_activities"],
        "train_subjects": all_sensor_data[0]["train_subjects"],
        "val_subjects": all_sensor_data[0]["val_subjects"],
        "test_subjects": all_sensor_data[0]["test_subjects"],
    }
    
    print(f"  Merged embeddings shape: {merged['train_embeddings'].shape}")
    
    return merged


def merge_dataset_variants(sensor_variants_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Load and concatenate embeddings from multiple dataset variants (tremor types).
    Each sensor loads from its own frequency-specific variants.
    
    Returns:
        Dictionary with all merged data
    """
    all_variant_data = []
    
    # Get number of tremor types from first sensor's variant list
    num_tremor_types = len(next(iter(sensor_variants_dict.values())))
    
    for tremor_idx in range(num_tremor_types):
        # Get variant names for display
        variant_names = [variants[tremor_idx] for variants in sensor_variants_dict.values()]
        print(f"\nLoading tremor type {tremor_idx + 1}/{num_tremor_types}:")
        for sensor, variant_name in zip(sensor_variants_dict.keys(), variant_names):
            print(f"  {sensor}: {variant_name}")
        
        variant_data = merge_sensor_embeddings(sensor_variants_dict, tremor_idx)
        all_variant_data.append(variant_data)
        print(f"  Samples: train={len(variant_data['train_embeddings'])}, "
              f"val={len(variant_data['val_embeddings'])}, "
              f"test={len(variant_data['test_embeddings'])}")
        
        # ═════════════════════════════════════════════════════════════════
        # DEBUG: Show label distribution per variant
        # ═════════════════════════════════════════════════════════════════
        print(f"  [DEBUG] Label distribution for variant {tremor_idx}:")
        print(f"    Train: {np.bincount(variant_data['train_labels'])}")
        print(f"    Val:   {np.bincount(variant_data['val_labels'])}")
        print(f"    Test:  {np.bincount(variant_data['test_labels'])}")
        
        # Sanity check: verify variant-label consistency
        tremor_type_name = ["clean", "mild_mod", "mod_severe"][tremor_idx]
        if tremor_type_name == "clean":
            # clean should have all labels == 0
            if not np.all(variant_data['train_labels'] == 0):
                print(f"    ⚠️  WARNING: clean variant has non-zero labels in train!")
            if not np.all(variant_data['val_labels'] == 0):
                print(f"    ⚠️  WARNING: clean variant has non-zero labels in val!")
            if not np.all(variant_data['test_labels'] == 0):
                print(f"    ⚠️  WARNING: clean variant has non-zero labels in test!")
        else:
            # mild_mod and mod_severe should have all labels == 1
            if not np.all(variant_data['train_labels'] == 1):
                print(f"    ⚠️  WARNING: {tremor_type_name} variant has non-one labels in train!")
            if not np.all(variant_data['val_labels'] == 1):
                print(f"    ⚠️  WARNING: {tremor_type_name} variant has non-one labels in val!")
            if not np.all(variant_data['test_labels'] == 1):
                print(f"    ⚠️  WARNING: {tremor_type_name} variant has non-one labels in test!")
    
    # Concatenate all variants along sample axis
    merged = {}
    for key in all_variant_data[0].keys():
        merged[key] = np.concatenate([data[key] for data in all_variant_data], axis=0)
    
    print(f"\nMerged {num_tremor_types} tremor types:")
    print(f"  Train samples: {len(merged['train_embeddings'])}")
    print(f"  Val samples: {len(merged['val_embeddings'])}")
    print(f"  Test samples: {len(merged['test_embeddings'])}")
    print(f"  Embedding dim: {merged['train_embeddings'].shape[1]}")
    # Show binary label distribution
    train_label_counts = np.bincount(merged['train_labels'].astype(int), minlength=2)
    print(f"  Label distribution (train): No Tremor: {train_label_counts[0]}, Tremor: {train_label_counts[1]}")
    
    # ═════════════════════════════════════════════════════════════════
    # DEBUG: Merged label distribution
    # ═════════════════════════════════════════════════════════════════
    print(f"\n  [DEBUG] Merged label distributions:")
    print(f"    Train: {np.bincount(merged['train_labels'])}")
    print(f"    Val:   {np.bincount(merged['val_labels'])}")
    print(f"    Test:  {np.bincount(merged['test_labels'])}")
    
    return merged


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class EmbeddingDataset(Dataset):
    """
    Dataset for feature-level fusion using pre-extracted embeddings.
    
    Returns:
        - embeddings: (D,) concatenated sensor embeddings
        - label: (,) binary tremor classification (0=no tremor, 1=tremor)
    """
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
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
      - Input: Concatenated embeddings (384-dim)
      - Hidden layers with dropout
      - Output: 2-class classification (0=no tremor, 1=tremor)
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
# TRAINING
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
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
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
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, f1, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, title, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Tremor', 'Tremor'],
                   yticklabels=['No Tremor', 'Tremor'])
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.xticks(range(2), ['No Tremor', 'Tremor'])
        plt.yticks(range(2), ['No Tremor', 'Tremor'])
    
    plt.xlabel('Predicted Score')
    plt.ylabel('True Score')
    plt.title(f'{title} Confusion Matrix')
    plt.tight_layout()
    
    save_path = save_dir / f'confusion_matrix_{title.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Set working directory to workspace root (Parkinson_dataset_work)
    os.chdir(WORKSPACE_ROOT)
    
    print("=" * 80)
    print("TREMOR CLASSIFICATION: FEATURE-LEVEL FUSION")
    print("=" * 80)
    
    # Load embeddings
    print("\n" + "=" * 80)
    print("LOADING EMBEDDINGS")
    print("=" * 80)
    
    data = merge_dataset_variants(sensor_variants)
    
    # Create datasets
    train_dataset = EmbeddingDataset(data["train_embeddings"], data["train_labels"])
    val_dataset = EmbeddingDataset(data["val_embeddings"], data["val_labels"])
    test_dataset = EmbeddingDataset(data["test_embeddings"], data["test_labels"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ═════════════════════════════════════════════════════════════════
    # DEBUG: Show dataset properties before training
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("DATASET SANITY CHECKS")
    print("=" * 80)
    
    print("\nTraining dataset:")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Label distribution: {np.bincount(data['train_labels'])}")
    
    print("\nValidation dataset:")
    print(f"  Samples: {len(val_dataset)}")
    print(f"  Label distribution: {np.bincount(data['val_labels'])}")
    
    print("\nTest dataset:")
    print(f"  Samples: {len(test_dataset)}")
    print(f"  Label distribution: {np.bincount(data['test_labels'])}")
    
    # Get first batch to inspect
    first_batch = next(iter(train_loader))
    print(f"\nFirst batch from training loader:")
    print(f"  Embeddings shape: {first_batch[0].shape}")
    print(f"  Labels: {first_batch[1][:10].numpy()}")
    print(f"  First 10 labels (should have 0s and 1s): {first_batch[1][:10].numpy()}")
    
    # Get samples from each class
    train_labels = data['train_labels']
    label_0_idx = np.where(train_labels == 0)[0]
    label_1_idx = np.where(train_labels == 1)[0]
    print(f"\n  Class 0 (no tremor) samples: {len(label_0_idx)}")
    print(f"  Class 1 (tremor) samples: {len(label_1_idx)}")
    if len(label_0_idx) > 0:
        print(f"    First class 0 indices: {label_0_idx[:5]}")
    if len(label_1_idx) > 0:
        print(f"    First class 1 indices: {label_1_idx[:5]}")
    
    # Create model
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    
    model = FeatureFusionClassifier(
        input_dim=total_embed_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    print(model)
    
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
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    best_optimizer_state = None
    
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
        
        # ═════════════════════════════════════════════════════════════════
        # DEBUG: Show model outputs on epoch 0 only
        # ═════════════════════════════════════════════════════════════════
        if epoch == 0:
            print("\n" + "=" * 80)
            print("EPOCH 0 DEBUG: Model output inspection")
            print("=" * 80)
            
            # Get first batch
            first_batch = next(iter(train_loader))
            X_batch = first_batch[0].to(device)
            y_batch = first_batch[1].to(device)
            
            with torch.no_grad():
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            # Convert to numpy for analysis
            y_np = y_batch.cpu().numpy()
            preds_np = preds.cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            print(f"\nFirst 10 samples from training loader:")
            print(f"  True labels:  {y_np[:10]}")
            print(f"  Predictions:  {preds_np[:10]}")
            print(f"  Logits (class 0, class 1):")
            for i in range(10):
                lg = logits_np[i]
                print(f"    Sample {i}: [{lg[0]:7.3f}, {lg[1]:7.3f}]")
            print(f"  Probabilities (P(class 0), P(class 1)):")
            for i in range(10):
                pb = probs[i].cpu().numpy()
                print(f"    Sample {i}: [{pb[0]:.4f}, {pb[1]:.4f}]")
            
            # ═════════════════════════════════════════════════════════
            # NEW DIAGNOSTICS: Batch-level statistics
            # ═════════════════════════════════════════════════════════
            print(f"\n  [BATCH STATISTICS]")
            print(f"    Batch label distribution: {np.bincount(y_np.astype(int), minlength=2)}")
            print(f"    Batch prediction distribution: {np.bincount(preds_np.astype(int), minlength=2)}")
            print(f"    Fraction of predictions == 1: {(preds_np == 1).mean():.4f}")
            
            print(f"\n  [LOGIT STATISTICS]")
            print(f"    Logits for class 0: min={logits_np[:, 0].min():.4f}, max={logits_np[:, 0].max():.4f}, mean={logits_np[:, 0].mean():.4f}")
            print(f"    Logits for class 1: min={logits_np[:, 1].min():.4f}, max={logits_np[:, 1].max():.4f}, mean={logits_np[:, 1].mean():.4f}")
            
            print(f"\n  [INTERPRETATION]")
            print(f"    If pred matches true label (0->0 or 1->1): model learning correctly")
            print(f"    If pred is inverted (0->1 or 1->0): possible label inversion issue")
            print(f"    If fraction==1 is high: model may be defaulting to class 1 at initialization")
        
        # Print progress
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
    
    # Load best model
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    if best_model_state is None:
        print("ERROR: No model was saved during training!")
        return
    
    model.load_state_dict(best_model_state)
    
    # Evaluate on validation and test sets
    val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
    test_loss, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    
    print("Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print()
    
    print("Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print()
    
    # Classification report
    print("Test Classification Report:")
    print(classification_report(test_labels, test_preds,
                                labels=[0, 1],
                                target_names=['No Tremor', 'Tremor'],
                                zero_division=0))
    
    # Confusion matrices
    val_cm = confusion_matrix(val_labels, val_preds)
    test_cm = confusion_matrix(test_labels, test_preds)
    
    print("Test Confusion Matrix:")
    print(test_cm)
    print()
    
    # Plot confusion matrices
    print("Saving plots...")
    plot_confusion_matrix(val_cm, "Validation", PLOT_DIR)
    plot_confusion_matrix(test_cm, "Test", PLOT_DIR)
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE ACCURACY TO HISTORY
    # ═══════════════════════════════════════════════════════════════
    # Model key includes sensor-specific sampling frequency (e.g., accarm30_gyroarm50)
    model_key = f"feature_fusion_tremorEmb_{freq_str}"
    
    # Load and update accuracy history
    if ACCURACY_HISTORY_FILE.exists():
        with open(ACCURACY_HISTORY_FILE, 'r') as f:
            history_data = json.load(f)
    else:
        history_data = {}
    
    if model_key not in history_data:
        history_data[model_key] = []
    
    history_data[model_key].append(best_val_f1)
    
    with open(ACCURACY_HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=2, sort_keys=True)
    
    print(f"\nValidation F1-score added to history: {ACCURACY_HISTORY_FILE}")
    
    # ═══════════════════════════════════════════════════════════════
    # CHECK IF THIS IS A NEW BEST ACCURACY
    # ═══════════════════════════════════════════════════════════════
    
    # Load existing best accuracies
    if BEST_ACCURACIES_FILE.exists():
        with open(BEST_ACCURACIES_FILE, 'r') as f:
            best_accs = json.load(f)
    else:
        best_accs = {}
    
    # Get previous best validation F1-score
    if model_key in best_accs:
        if isinstance(best_accs[model_key], dict):
            previous_best = best_accs[model_key].get("f1", 0.0)
        else:
            # Legacy: old format stored accuracy
            previous_best = best_accs[model_key] if isinstance(best_accs[model_key], (int, float)) else 0.0
    else:
        previous_best = 0.0
    
    print(f"\nBest validation F1 this run: {best_val_f1:.4f} (Validation Accuracy: {best_val_acc:.4f})")
    print(f"Previous Best validation F1: {previous_best:.4f}")
    print(f"\nTest metrics (for reference, not used for model selection):")
    print(f"  Test F1: {test_f1:.4f} (Test Accuracy: {test_acc:.4f})")
    print()
    
    is_best = best_val_f1 > previous_best
    
    if not is_best:
        print(f"❌ No improvement in validation F1-score. Skipping best model save.")
    else:
        print(f"✅ New best validation F1-score! Saving best model...")
        
        # Update best accuracies file (based on VALIDATION F1, not test F1)
        best_accs[model_key] = {
            "f1": best_val_f1,
            "accuracy": best_val_acc,
            "val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "architecture": str(model),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "random_seed": SEED,
            "timestamp": datetime.now().isoformat()
        }
        with open(BEST_ACCURACIES_FILE, 'w') as f:
            json.dump(best_accs, f, indent=2, sort_keys=True)
        
        # Save best model
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'val_loss': best_val_loss,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'random_seed': SEED,
            'sensor_freq_map': sensor_freq_map,
            'config': {
                'embed_dim_per_sensor': embed_dim_per_sensor,
                'total_embed_dim': total_embed_dim,
                'hidden_dims': hidden_dims,
                'num_classes': num_classes,
                'dropout': dropout,
            }
        }, MODEL_BEST_PATH)
        
        print(f"Best model saved: {MODEL_BEST_PATH}")
    
    # ═══════════════════════════════════════════════════════════════
    # LOG RESULTS TO JSONL
    # ═══════════════════════════════════════════════════════════════
    # Save results
    results = {
        "model_type": "feature_fusion",
        "config": {
            "sensors": SENSORS,
            "sensor_freq_map": sensor_freq_map,
            "dataset_variants": sensor_variants,
            "embed_dim_per_sensor": embed_dim_per_sensor,
            "total_embed_dim": total_embed_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "epochs_trained": best_epoch + 1,
        },
        "val_metrics": {
            "accuracy": float(val_acc),
            "f1": float(val_f1),
            "loss": float(val_loss)
        },
        "test_metrics": {
            "accuracy": float(test_acc),
            "f1": float(test_f1),
            "loss": float(test_loss)
        },
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "random_seed": SEED,
        "is_best": is_best,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(results) + '\n')
    
    print(f"\nResults logged to: {LOG_FILE}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE" + (" (new best!)" if is_best else ""))
    print("=" * 80)


if __name__ == "__main__":
    main()
