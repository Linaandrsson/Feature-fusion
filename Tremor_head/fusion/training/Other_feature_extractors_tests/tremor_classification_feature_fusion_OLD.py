"""
Tremor Classification Model: Predict tremor severity score (0-4)
=================================================================

Model that takes:
  - Input: Raw sensor window data + activity label
  - Output: Tremor severity score (0-4)
    0 = No tremor (< 0.07 m/s²)
    1 = Mild (0.07-0.15 m/s²)
    2 = Mild-Moderate (0.15-0.7 m/s²)
    3 = Moderate-Severe (0.7-2.5 m/s²)
    4 = Severe (2.5-6.0 m/s²)

Architecture:
  - 1D-CNN backbone for temporal feature extraction
  - Activity embedding concatenated in latent layer
  - Classification head with softmax output
  - Cross-entropy loss (with optional class weighting)

Usage:
  1. Configure sensors and data paths in CONFIGURATION section
  2. Run to train and evaluate
  3. Results saved to tremor_logs/
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# Optional import for prettier plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
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
# Data paths and sensors
# -------------------------------
# Path to tremor dataset
DATA_PATH = Path("/Volumes/NO NAME/Master Lina/Code/data/Tremor_datagenerator_files")

# Dataset variants to load and merge:
# Loading both clean and parkinson datasets gives:
#   - More training data (augmentation)
#   - Better class balance (clean provides score=0, parkinson provides score 0-4)
#   - Realistic mix of tremor and non-tremor samples
DATASET_VARIANTS = ["s2_w2_tremor_clean", "s2_w2_tremor_parkinson_mild", "s2_w2_tremor_parkinson_severe"]

# Sensors to use (can select multiple)
# Available: Acc_ankle, Acc_arm, Acc_chest, Gyro_ankle, Gyro_arm, Mag_ankle, Mag_arm
# Data-level fusion: Acc_arm (3 channels) + Gyro_arm (3 channels) = 6 channels total
SENSORS = ["Acc_arm", "Gyro_arm"]  # Data-level fusion (concat)

# -------------------------------
# Dataset splits
# -------------------------------
# Split by subject for generalization testing
SPLIT_BY_SUBJECT = True  # False = random split (for sanity check only)
TEST_SUBJECTS = [5,10]  # Held-out subjects for testing
VAL_SUBJECTS = [2,7]    # Validation subjects
# Train subjects: [1, 2, 3, 4, 5, 6]

# Alternative: Random split (use when SPLIT_BY_SUBJECT = False)
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# -------------------------------
# Normalization strategy
# -------------------------------
# Options:
#   "standard": Normalize with mean/std from all training data
#   "none": No normalization (sanity check - uses absolute amplitude)
#   "clean_only": Normalize with mean/std from clean samples only (tremor preserved as deviation)
NORMALIZATION_MODE = "clean_only"  # "standard", "none", or "clean_only"

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
# CNN backbone (data-level fusion: 6 input channels from Acc_arm + Gyro_arm)
cnn_filters = [32, 64]  # 2 conv layers for feature extraction
cnn_kernel_size = 5
cnn_pool_size = 2
cnn_dropout = 0.3

# Activity embedding
activity_embed_dim = 16  # Small embedding for activity (1-12)
num_activities = 12

# Classification head
head_hidden_dims = [128, 64]  # Hidden layers before output
head_dropout = 0.3
num_classes = 5  # Scores 0-4

# -------------------------------
# Loss function configuration
# -------------------------------
# Use class weights to handle imbalance
USE_CLASS_WEIGHTS = True  # Compute from training data

# Alternative: Focal Loss for hard examples
USE_FOCAL_LOSS = False  # Set True to use focal loss instead of CE
FOCAL_ALPHA = None  # Will be computed from class frequencies if None
FOCAL_GAMMA = 2.0   # Focusing parameter

# -------------------------------
# Logging and plotting
# -------------------------------
SAVE_PLOTS = True
LOG_DIR = Path("tremor_logs")
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR = LOG_DIR / "classification_plots"
PLOT_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "tremor_classification_experiments.jsonl"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (same as regression)
# ═══════════════════════════════════════════════════════════════

def load_tremor_data(sensor_name: str, dataset_path: Path) -> Dict[str, np.ndarray]:
    """
    Load tremor dataset from TXT file.
    
    Returns:
        - X: (N, C, L) sensor windows
        - y_activity: (N,) activity labels (1-12)
        - y_subject: (N,) subject IDs (1-10)
        - y_tremor_freq: (N,) tremor frequency (Hz)
        - y_tremor_acc_rms: (N,) tremor acc RMS (m/s²)
        - y_tremor_gyro_rms: (N,) tremor gyro RMS (deg/s)
        - y_tremor_score: (N,) tremor score (0-4)
    """
    file_path = dataset_path / f"{sensor_name}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"Loading {sensor_name} from {file_path.name}...")
    data = np.loadtxt(file_path, delimiter=",")
    
    # Determine number of channels based on sensor type
    if "ECG" in sensor_name:
        n_channels = 2
        window_length = 100
    else:
        n_channels = 3
        window_length = 100
    
    # Extract data
    sensor_data = data[:, :n_channels * window_length]
    
    # Reshape to (N, C, L) - channel-major format
    N = data.shape[0]
    X = sensor_data.reshape(N, n_channels, window_length)
    
    # Extract labels (1-indexed in file)
    col_offset = n_channels * window_length
    y_activity = data[:, col_offset].astype(np.int64) - 1  # Convert to 0-indexed
    y_subject = data[:, col_offset + 1].astype(np.int64)   # Keep 1-indexed
    y_window_idx = data[:, col_offset + 2].astype(np.int64)
    y_tremor_freq = data[:, col_offset + 3].astype(np.float32)
    y_tremor_acc_rms = data[:, col_offset + 4].astype(np.float32)
    y_tremor_gyro_rms = data[:, col_offset + 5].astype(np.float32)
    y_tremor_score = data[:, col_offset + 6].astype(np.int64)
    
    return {
        "X": X.astype(np.float32),
        "y_activity": y_activity,
        "y_subject": y_subject,
        "y_window_idx": y_window_idx,
        "y_tremor_freq": y_tremor_freq,
        "y_tremor_acc_rms": y_tremor_acc_rms,
        "y_tremor_gyro_rms": y_tremor_gyro_rms,
        "y_tremor_score": y_tremor_score,
    }


def merge_sensor_data(sensor_list: List[str], dataset_path: Path) -> Dict[str, np.ndarray]:
    """
    Load and concatenate multiple sensors.
    
    Returns merged data with X shape (N, C_total, L).
    """
    all_data = [load_tremor_data(s, dataset_path) for s in sensor_list]
    
    # Verify labels match across sensors
    ref_labels = all_data[0]["y_activity"]
    # Verify labels match across sensors
    for key in ["y_activity", "y_subject", "y_window_idx", "y_tremor_score"]:
        ref = all_data[0][key]
        for i, d in enumerate(all_data[1:], 1):
            if not np.array_equal(d[key], ref):
                raise ValueError(f"Label mismatch ({key}) between {sensor_list[0]} and {sensor_list[i]}")


    # Concatenate sensor data along channel axis
    X_merged = np.concatenate([data["X"] for data in all_data], axis=1)
    
    # Return merged X with labels from first sensor (all identical)
    result = all_data[0].copy()
    result["X"] = X_merged
    
    print(f"Merged {len(sensor_list)} sensors: X shape = {X_merged.shape}")
    return result


def merge_dataset_variants(sensor_list: List[str], dataset_variants: List[str], base_path: Path) -> Dict[str, np.ndarray]:
    """
    Load and concatenate data from multiple dataset variants.
    
    This allows combining clean and parkinson datasets for augmentation.
    
    Args:
        sensor_list: List of sensor names to load
        dataset_variants: List of dataset variant names (e.g., ["s2_w2_tremor_clean", "s2_w2_tremor_parkinson"])
        base_path: Base path to datasets
    
    Returns:
        Merged data with all samples from all variants
    """
    all_variant_data = []
    
    for variant in dataset_variants:
        print(f"\nLoading variant: {variant}")
        dataset_path = base_path / variant
        variant_data = merge_sensor_data(sensor_list, dataset_path)
        all_variant_data.append(variant_data)
        print(f"  Loaded {variant_data['X'].shape[0]} samples")
    
    # Concatenate all variants along sample axis
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
    """
    Split data by subject IDs.
    """
    subjects = data["y_subject"]
    
    train_mask = ~np.isin(subjects, val_subjects + test_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    train_data = {k: v[train_mask] for k, v in data.items()}
    val_data = {k: v[val_mask] for k, v in data.items()}
    test_data = {k: v[test_mask] for k, v in data.items()}
    
    print(f"Split by subject:")
    print(f"  Train: {train_mask.sum()} samples (subjects: {np.unique(train_data['y_subject'])})")
    print(f"  Val:   {val_mask.sum()} samples (subjects: {val_subjects})")
    print(f"  Test:  {test_mask.sum()} samples (subjects: {test_subjects})")
    
    return train_data, val_data, test_data


def split_data_random(
    data: Dict[str, np.ndarray],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Random split (for sanity check only).
    """
    N = data["X"].shape[0]
    indices = np.arange(N)
    
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    val_size = int(N * val_frac)
    test_size = int(N * test_frac)
    
    val_idx = indices[:val_size]
    test_idx = indices[val_size:val_size + test_size]
    train_idx = indices[val_size + test_size:]
    
    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}
    
    print(f"Random split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    return train_data, val_data, test_data


def compute_normalization_stats(
    train_data: Dict[str, np.ndarray],
    mode: str = "standard"
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std per channel from training data.
    
    Args:
        train_data: Training data dictionary
        mode: Normalization mode
            - "standard": Use all training data
            - "clean_only": Use only clean samples (score=0)
            - "none": Return identity transform (no normalization)
    
    Returns:
        Dict with "mean" and "std" arrays
    """
    X_train = train_data["X"]  # (N, C, L)
    
    if mode == "none":
        # Identity transform (no normalization)
        mean = np.zeros((1, X_train.shape[1], 1), dtype=np.float32)
        std = np.ones((1, X_train.shape[1], 1), dtype=np.float32)
        print("  Using NO normalization (identity transform)")
        return {"mean": mean, "std": std}
    
    elif mode == "clean_only":
        # Use only clean samples (score=0) for computing stats
        clean_mask = train_data["y_tremor_score"] == 0
        X_clean = X_train[clean_mask]
        
        if len(X_clean) == 0:
            print("  Warning: No clean samples found! Falling back to standard normalization.")
            X_stats = X_train
        else:
            X_stats = X_clean
            print(f"  Using clean samples only for normalization: {len(X_clean)}/{len(X_train)} samples")
    
    else:  # "standard"
        X_stats = X_train
        print(f"  Using all training samples for normalization: {len(X_train)} samples")
    
    # Compute per-channel statistics
    mean = X_stats.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std = X_stats.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    
    return {"mean": mean, "std": std}


def normalize_data(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalize data using precomputed statistics.
    """
    return (X - stats["mean"]) / stats["std"]


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Compute class weights inversely proportional to frequency.
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    
    # Inverse frequency
    weights = total / (num_classes * class_counts + 1e-6)
    
    return torch.FloatTensor(weights)


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class TremorClassificationDataset(Dataset):
    """
    Dataset for tremor severity classification.
    
    Returns:
        - X: (C, L) sensor window (normalized)
        - activity: (,) activity label (0-11)
        - score: (,) tremor score (0-4)
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        norm_stats: Dict[str, np.ndarray]
    ):
        self.X = normalize_data(data["X"], norm_stats)  # (N, C, L)
        self.y_activity = data["y_activity"]  # (N,)
        self.y_score = data["y_tremor_score"]  # (N,)
        
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

class TremorClassificationCNN(nn.Module):
    """
    1D-CNN for tremor severity classification (0-4).
    
    Architecture (based on IMUCNN from feature extraction):
      - 2-layer 1D CNN for temporal feature extraction (data-level fusion)
      - Activity embedding concatenated to CNN features
      - Embedding layer + classification head
    """
    
    def __init__(self):
        super().__init__()
        
        num_channels = 6  # Acc_arm (3) + Gyro_arm (3)
        seq_len = 100
        num_classes = 5
        
        # Activity embedding (12 activities -> 16-dim embedding)
        self.activity_embed = nn.Embedding(12, 16)
        
        # CNN feature extractor (same as IMUCNN)
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        # Flattened dimension: (seq_len // 4) * 256 = 25 * 256 = 6400
        self.flattened_dim = (seq_len // 4) * 256
        
        # Combined dimension: CNN features + activity embedding
        # 6400 + 16 = 6416
        combined_dim = self.flattened_dim + 16
        
        # Embedding layer (feature representation)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(combined_dim, 128)
        
        # Classification head
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)
    
    def forward(self, X, activity):
        """
        Args:
            X: (B, C, L) sensor data - (B, 6, 100)
            activity: (B,) activity labels
        
        Returns:
            (B, 5) logits for 5 tremor severity classes
        """
        # CNN feature extraction
        x = self.features(X)         # (B, 6, 100) -> (B, 256, 25)
        
        # Flatten CNN output
        cnn_flat = self.flatten(x)   # (B, 6400)
        
        # Activity embedding
        act_emb = self.activity_embed(activity)  # (B, 16)
        
        # Combine CNN features with activity embedding
        combined = torch.cat([cnn_flat, act_emb], dim=1)  # (B, 6416)
        
        # Feature embedding layer
        z = self.fc_embed(combined)  # (B, 128)
        
        # Classification head
        z = self.drop_cls(z)         # (B, 128)
        logits = self.fc_cls(z)      # (B, 5)
        
        return logits


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits
            targets: (B,) class indices
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(logits.device)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ═══════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X, activity, score in loader:
        X = X.to(device)
        activity = activity.to(device)
        score = score.to(device)
        
        optimizer.zero_grad()
        logits = model(X, activity)
        loss = criterion(logits, score)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """
    Evaluate model and compute metrics.
    
    Returns:
        - loss: average loss
        - accuracy: classification accuracy
        - f1_macro: macro F1 score
        - predictions: (N,) array
        - targets: (N,) array
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, activity, score in loader:
            X = X.to(device)
            activity = activity.to(device)
            score = score.to(device)
            
            logits = model(X, activity)
            loss = criterion(logits, score)
            
            preds = logits.argmax(dim=1)
            
            total_loss += loss.item() * X.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(score.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Compute metrics
    accuracy = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
    
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, accuracy, f1_macro, preds, targets


def plot_confusion_matrix(cm, split_name, plot_dir):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['0', '1', '2', '3', '4'],
                    yticklabels=['0', '1', '2', '3', '4'])
    else:
        # Fallback to matplotlib imshow
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), 
                        ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.xticks(range(5), ['0', '1', '2', '3', '4'])
        plt.yticks(range(5), ['0', '1', '2', '3', '4'])
    
    plt.xlabel('Predicted Score')
    plt.ylabel('True Score')
    plt.title(f'{split_name}: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plot_dir / f"{split_name.lower()}_confusion_matrix.png", dpi=150)
    plt.close()


def plot_class_distribution(train_labels, val_labels, test_labels, plot_dir):
    """
    Plot class distribution across splits.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, labels, name in zip(axes, 
                                  [train_labels, val_labels, test_labels],
                                  ['Train', 'Val', 'Test']):
        counts = np.bincount(labels, minlength=5)
        ax.bar(range(5), counts)
        ax.set_xlabel('Tremor Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}: Class Distribution')
        ax.set_xticks(range(5))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "class_distribution.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("TREMOR CLASSIFICATION MODEL TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Datasets: {DATASET_VARIANTS}")
    print(f"Sensors: {SENSORS}")
    print(f"Split by subject: {SPLIT_BY_SUBJECT}")
    print(f"Use class weights: {USE_CLASS_WEIGHTS}")
    print(f"Use focal loss: {USE_FOCAL_LOSS}")
    print()
    
    # Load and merge data from all dataset variants
    data = merge_dataset_variants(SENSORS, DATASET_VARIANTS, DATA_PATH)
    
    # Split data
    if SPLIT_BY_SUBJECT:
        train_data, val_data, test_data = split_data_by_subject(
            data, VAL_SUBJECTS, TEST_SUBJECTS
        )
    else:
        train_data, val_data, test_data = split_data_random(
            data, VAL_FRACTION, TEST_FRACTION, SEED
        )
    
    # Print class distribution
    print("Class distribution:")
    print(f"  Train: {np.bincount(train_data['y_tremor_score'], minlength=5)}")
    print(f"  Val:   {np.bincount(val_data['y_tremor_score'], minlength=5)}")
    print(f"  Test:  {np.bincount(test_data['y_tremor_score'], minlength=5)}")
    print()
    
    # Compute normalization stats from training data
    print(f"\nNormalization mode: {NORMALIZATION_MODE}")
    norm_stats = compute_normalization_stats(train_data, mode=NORMALIZATION_MODE)
    print(f"  Mean shape: {norm_stats['mean'].shape}")
    print(f"  Std shape: {norm_stats['std'].shape}")
    print()
    
    # Create datasets
    train_dataset = TremorClassificationDataset(train_data, norm_stats)
    val_dataset = TremorClassificationDataset(val_data, norm_stats)
    test_dataset = TremorClassificationDataset(test_data, norm_stats)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Datasets created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print()
    
    # Create model
    model = TremorClassificationCNN().to(device)
    
    print(f"Model architecture:")
    print(f"  Input: (6, 100) - Acc_arm + Gyro_arm, data-level fusion")
    print(f"  CNN: Conv1(6->128) + Conv2(128->256) [same as IMUCNN]")
    print(f"  Activity embedding: 12 -> 16 dims")
    print(f"  Feature embedding: FC(6416->128)")
    print(f"  Classifier: FC(128->5) with Dropout(0.5)")
    print(f"  Output: 5 classes (tremor scores 0-4)")
    print()
    
    # Loss and optimizer
    if USE_FOCAL_LOSS:
        if FOCAL_ALPHA is None:
            # Compute alpha from class frequencies
            alpha = compute_class_weights(train_data['y_tremor_score'], num_classes)
        else:
            alpha = torch.FloatTensor(FOCAL_ALPHA)
        criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
        print(f"Using Focal Loss (gamma={FOCAL_GAMMA})")
        print(f"  Alpha: {alpha.numpy()}")
    else:
        if USE_CLASS_WEIGHTS:
            class_weights = compute_class_weights(train_data['y_tremor_score'], num_classes)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"Using weighted Cross-Entropy")
            print(f"  Weights: {class_weights.numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using standard Cross-Entropy")
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Plot class distribution
    if SAVE_PLOTS:
        plot_class_distribution(
            train_data['y_tremor_score'],
            val_data['y_tremor_score'],
            test_data['y_tremor_score'],
            PLOT_DIR
        )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }
    
    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), LOG_DIR / "best_tremor_classification_model.pth")
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Val loss:   {val_loss:.6f}")
            print(f"  Val acc:    {val_acc:.4f}")
            print(f"  Val F1:     {val_f1:.4f}")
            print()
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(LOG_DIR / "best_tremor_classification_model.pth"))
    
    # Final evaluation
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    val_loss, val_acc, val_f1, val_preds, val_targets = evaluate(
        model, val_loader, criterion, device
    )
    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    
    print("Validation metrics:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Macro F1: {val_f1:.4f}")
    print()
    
    print("Test metrics:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")
    print()
    
    # Detailed classification report
    print("Test Classification Report:")
    print(classification_report(test_targets, test_preds, 
                                labels=list(range(5)),
                                target_names=[f'Score {i}' for i in range(5)],
                                zero_division=0))
    
    # Confusion matrices
    val_cm = confusion_matrix(val_targets, val_preds)
    test_cm = confusion_matrix(test_targets, test_preds)
    
    print("Test Confusion Matrix:")
    print(test_cm)
    print()
    
    # Plot confusion matrices
    if SAVE_PLOTS:
        plot_confusion_matrix(val_cm, "Validation", PLOT_DIR)
        plot_confusion_matrix(test_cm, "Test", PLOT_DIR)
        print(f"Plots saved to {PLOT_DIR}")
    
    # Save results
    results = {
        "config": {
            "datasets": DATASET_VARIANTS,
            "sensors": SENSORS,
            "normalization_mode": NORMALIZATION_MODE,
            "split_by_subject": SPLIT_BY_SUBJECT,
            "test_subjects": TEST_SUBJECTS if SPLIT_BY_SUBJECT else None,
            "val_subjects": VAL_SUBJECTS if SPLIT_BY_SUBJECT else None,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "use_focal_loss": USE_FOCAL_LOSS,
            "epochs_trained": epoch + 1,
            "batch_size": batch_size,
            "lr": lr,
        },
        "val_metrics": {
            "accuracy": val_acc,
            "macro_f1": val_f1,
            "loss": val_loss
        },
        "test_metrics": {
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "loss": test_loss
        },
        "best_val_loss": best_val_loss,
        "test_confusion_matrix": test_cm.tolist(),
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    print(f"Results logged to {LOG_FILE}")
    print("Training complete!")


if __name__ == "__main__":
    main()
