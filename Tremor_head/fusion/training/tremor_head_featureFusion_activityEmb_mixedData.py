"""
Tremor Classification: Feature-Level Fusion
============================================

Uses pre-extracted embeddings from Acc_arm and Gyro_arm CNNs for classification.

Architecture:
  - Input: Concatenated embeddings (Acc_arm: 128-dim + Gyro_arm: 128-dim = 256-dim)
  - Simple MLP classifier with dropout
  - Output: Tremor severity score (0-4)

Data Flow:
  1. Train feature extractors: train_acc_arm_extractor.py and train_gyro_arm_extractor.py
  2. Extract embeddings: extract_tremor_embeddings.py
  3. Train fusion classifier: THIS FILE

Usage:
    python tremor_classification_feature_fusion.py
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

# Data paths
DATA_PATH = Path("/Volumes/NO NAME/Master Lina/Code/data/Tremor_datagenerator_files")
DATASET_VARIANTS = ["s2_w2_tremor_clean", "s2_w2_tremor_parkinson_mild", "s2_w2_tremor_parkinson_severe"]

# Sensors to fuse
SENSORS = ["Acc_arm", "Gyro_arm"]

# Training hyperparameters
batch_size = 64
epochs = 150
lr = 1e-3
patience = 50
min_delta = 1e-5

# Model architecture
embed_dim_per_sensor = 128  # Each sensor provides 128-dim embedding
total_embed_dim = len(SENSORS) * embed_dim_per_sensor  # 256-dim
hidden_dims = [128, 64]  # MLP hidden layers
dropout = 0.3
num_classes = 5  # Tremor scores: 0-4

# Logging
LOG_DIR = Path("Tremor_head/fusion")
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR = LOG_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "feature_fusion_training_activity_emb_mixed.jsonl"

# Model paths
MODEL_DIR = LOG_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_BEST_PATH = MODEL_DIR / "feature_fusion_best_activity_emb_mixed.pth"

# History files
ACCURACY_HISTORY_FILE = LOG_DIR / "accuracy_history_activity_emb_mixed.json"
BEST_ACCURACIES_FILE = LOG_DIR / "best_accuracies_activity_emb_mixed.json"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_embeddings(variant_name: str, sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Load pre-extracted embeddings for one sensor from one dataset variant.
    
    Returns:
        Dictionary with train/val/test embeddings, labels, activities, subjects
    """
    #embeddings_path = DATA_PATH / variant_name / "ExtractedFeatures" / f"{sensor_name}_embeddings.npz"
    embeddings_path = DATA_PATH / variant_name / "ExtractedFeatures_ActivityEmb_mixedSet" / f"{sensor_name}_embeddings.npz"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            f"Please run extract_tremor_embeddings.py first!"
        )
    
    print(f"  Loading {sensor_name} from {embeddings_path.name}")
    print(f"     Full path: {embeddings_path}")
    data = np.load(embeddings_path)
    print(f"     Keys in file: {list(data.keys())}")
    
    return {
        "train_embeddings": data["train_embeddings"],
        "val_embeddings": data["val_embeddings"],
        "test_embeddings": data["test_embeddings"],
        "train_labels": data["train_labels"],
        "val_labels": data["val_labels"],
        "test_labels": data["test_labels"],
        "train_activities": data["train_activities"],
        "val_activities": data["val_activities"],
        "test_activities": data["test_activities"],
        "train_subjects": data["train_subjects"],
        "val_subjects": data["val_subjects"],
        "test_subjects": data["test_subjects"],
    }


def merge_sensor_embeddings(variant_name: str, sensor_list: List[str]) -> Dict[str, np.ndarray]:
    """
    Load and concatenate embeddings from multiple sensors for one variant.
    
    Returns:
        Dictionary with concatenated train/val/test embeddings
    """
    all_sensor_data = [load_embeddings(variant_name, sensor) for sensor in sensor_list]
    
    # Verify labels match across sensors
    for key in ["train_labels", "val_labels", "test_labels", "train_activities", "val_activities", "test_activities"]:
        ref = all_sensor_data[0][key]
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


def merge_dataset_variants(sensor_list: List[str], variant_list: List[str]) -> Dict[str, np.ndarray]:
    """
    Load and concatenate embeddings from multiple dataset variants.
    
    Returns:
        Dictionary with all merged data
    """
    all_variant_data = []
    
    for variant in variant_list:
        print(f"\nLoading variant: {variant}")
        variant_data = merge_sensor_embeddings(variant, sensor_list)
        all_variant_data.append(variant_data)
        print(f"  Samples: train={len(variant_data['train_embeddings'])}, "
              f"val={len(variant_data['val_embeddings'])}, "
              f"test={len(variant_data['test_embeddings'])}")
    
    # Concatenate all variants along sample axis
    merged = {}
    for key in all_variant_data[0].keys():
        merged[key] = np.concatenate([data[key] for data in all_variant_data], axis=0)
    
    print(f"\nMerged {len(variant_list)} dataset variants:")
    print(f"  Train samples: {len(merged['train_embeddings'])}")
    print(f"  Val samples: {len(merged['val_embeddings'])}")
    print(f"  Test samples: {len(merged['test_embeddings'])}")
    print(f"  Embedding dim: {merged['train_embeddings'].shape[1]}")
    print(f"  Score distribution (train): {np.bincount(merged['train_labels'].astype(int), minlength=5)}")
    
    return merged


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class EmbeddingDataset(Dataset):
    """
    Dataset for feature-level fusion using pre-extracted embeddings.
    
    Returns:
        - embeddings: (D,) concatenated sensor embeddings
        - label: (,) tremor score (0-4)
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
      - Input: Concatenated embeddings (256-dim)
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
    
    return avg_loss, acc, f1, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, title, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'S{i}' for i in range(5)],
                   yticklabels=[f'S{i}' for i in range(5)])
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        for i in range(5):
            for j in range(5):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.xticks(range(5), [f'S{i}' for i in range(5)])
        plt.yticks(range(5), [f'S{i}' for i in range(5)])
    
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
    print("=" * 80)
    print("TREMOR CLASSIFICATION: FEATURE-LEVEL FUSION")
    print("=" * 80)
    
    # Load embeddings
    print("\n" + "=" * 80)
    print("LOADING EMBEDDINGS")
    print("=" * 80)
    
    data = merge_dataset_variants(SENSORS, DATASET_VARIANTS)
    
    # Create datasets
    train_dataset = EmbeddingDataset(data["train_embeddings"], data["train_labels"])
    val_dataset = EmbeddingDataset(data["val_embeddings"], data["val_labels"])
    test_dataset = EmbeddingDataset(data["test_embeddings"], data["test_labels"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model state in memory
            best_model_state = model.state_dict().copy()
            best_optimizer_state = optimizer.state_dict().copy()
            print(f"  → Best model so far")
            
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
                                labels=list(range(5)),
                                target_names=[f'Score {i}' for i in range(5)],
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
    model_key = "feature_fusion"
    
    # Load and update accuracy history
    if ACCURACY_HISTORY_FILE.exists():
        with open(ACCURACY_HISTORY_FILE, 'r') as f:
            history_data = json.load(f)
    else:
        history_data = {}
    
    if model_key not in history_data:
        history_data[model_key] = []
    
    history_data[model_key].append(test_acc)
    
    with open(ACCURACY_HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=2, sort_keys=True)
    
    print(f"\nAccuracy added to history: {ACCURACY_HISTORY_FILE}")
    
    # ═══════════════════════════════════════════════════════════════
    # CHECK IF THIS IS A NEW BEST ACCURACY
    # ═══════════════════════════════════════════════════════════════
    
    # Load existing best accuracies
    if BEST_ACCURACIES_FILE.exists():
        with open(BEST_ACCURACIES_FILE, 'r') as f:
            best_accs = json.load(f)
    else:
        best_accs = {}
    
    # Get previous best
    if model_key in best_accs:
        if isinstance(best_accs[model_key], dict):
            previous_best = best_accs[model_key].get("accuracy", 0.0)
        else:
            previous_best = best_accs[model_key]
    else:
        previous_best = 0.0
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Previous Best: {previous_best:.4f}")
    
    is_best = test_acc > previous_best
    
    if not is_best:
        print(f"❌ No improvement. Skipping best model save.")
    else:
        print(f"✅ New best accuracy! Saving best model...")
        
        # Update best accuracies file
        best_accs[model_key] = {
            "accuracy": test_acc,
            "f1": test_f1,
            "test_loss": test_loss,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
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
            "dataset_variants": DATASET_VARIANTS,
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
