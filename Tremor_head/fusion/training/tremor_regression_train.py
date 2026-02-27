"""
Tremor Regression Model: Predict (RMS_acc, RMS_gyro, freq)
===========================================================

Model that takes:
  - Input: Raw sensor window data + activity label
  - Output: (tremor_acc_rms, tremor_gyro_rms, tremor_freq)

Architecture:
  - 1D-CNN backbone for temporal feature extraction
  - Activity embedding concatenated in latent layer
  - 3-head regression output
  - Multi-task loss with Huber loss per target

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
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error


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
DATASET_VARIANTS = ["s2_w2_tremor_clean", "s2_w2_tremor_parkinson"]

# Sensors to use (can select multiple)
# Available: Acc_ankle, Acc_arm, Acc_chest, Gyro_ankle, Gyro_arm, Mag_ankle, Mag_arm
SENSORS = ["Acc_arm", "Gyro_arm"]  # Both acc and gyro for realism

# -------------------------------
# Dataset splits
# -------------------------------
# Split by subject for generalization testing
SPLIT_BY_SUBJECT = True  # False = random split (for sanity check only)
TEST_SUBJECTS = [9, 10]  # Held-out subjects for testing
VAL_SUBJECTS = [7, 8]    # Validation subjects
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
patience = 25
min_delta = 1e-5

# -------------------------------
# Model architecture
# -------------------------------
# CNN backbone
cnn_filters = [32, 64, 64]  # Number of filters per conv layer
cnn_kernel_size = 5
cnn_pool_size = 2
cnn_dropout = 0.3

# Activity embedding
activity_embed_dim = 16  # Small embedding for activity (1-12)
num_activities = 12

# Regression heads
head_hidden_dim = 128
head_dropout = 0.2

# -------------------------------
# Loss function weights
# -------------------------------
# Huber loss per target
# freq often dominates, so reduce its weight
LAMBDA_ACC = 1.0   # Weight for RMS_acc loss
LAMBDA_GYRO = 1.0  # Weight for RMS_gyro loss
LAMBDA_FREQ = 0.2  # Weight for freq loss (lower to prevent overfocus)

# Huber delta (robust to outliers)
HUBER_DELTA = 1.0

# -------------------------------
# Target transformations
# -------------------------------
# Use log1p on RMS values for stability
USE_LOG_TRANSFORM = True  # log(1 + RMS) for acc and gyro

# -------------------------------
# Logging and plotting
# -------------------------------
SAVE_PLOTS = True
LOG_DIR = Path("tremor_logs")
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR = LOG_DIR / "regression_plots"
PLOT_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "tremor_regression_experiments.jsonl"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
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
    for i, data in enumerate(all_data[1:], 1):
        if not np.array_equal(data["y_activity"], ref_labels):
            raise ValueError(f"Label mismatch between {sensor_list[0]} and {sensor_list[i]}")
    
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


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class TremorRegressionDataset(Dataset):
    """
    Dataset for tremor regression.
    
    Returns:
        - X: (C, L) sensor window (normalized)
        - activity: (,) activity label (0-11)
        - targets: (3,) [tremor_acc_rms, tremor_gyro_rms, tremor_freq]
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        norm_stats: Dict[str, np.ndarray],
        use_log_transform: bool = True
    ):
        self.X = normalize_data(data["X"], norm_stats)  # (N, C, L)
        self.y_activity = data["y_activity"]  # (N,)
        self.y_acc_rms = data["y_tremor_acc_rms"]  # (N,)
        self.y_gyro_rms = data["y_tremor_gyro_rms"]  # (N,)
        self.y_freq = data["y_tremor_freq"]  # (N,)
        self.use_log_transform = use_log_transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])  # (C, L)
        activity = torch.tensor(self.y_activity[idx], dtype=torch.long)
        
        # Get targets
        acc_rms = self.y_acc_rms[idx]
        gyro_rms = self.y_gyro_rms[idx]
        freq = self.y_freq[idx]
        
        # Apply log transform to RMS values if enabled
        if self.use_log_transform:
            acc_rms = np.log1p(acc_rms)
            gyro_rms = np.log1p(gyro_rms)
        
        targets = torch.tensor([acc_rms, gyro_rms, freq], dtype=torch.float32)
        
        return X, activity, targets


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class TremorRegressionCNN(nn.Module):
    """
    1D-CNN for tremor regression.
    
    Architecture:
      - Multi-layer 1D CNN for temporal feature extraction
      - Activity embedding concatenated to latent features
      - 3 separate regression heads for (acc_rms, gyro_rms, freq)
    """
    
    def __init__(
        self,
        input_channels: int,
        input_length: int,
        num_activities: int = 12,
        activity_embed_dim: int = 16,
        cnn_filters: List[int] = [32, 64, 64],
        cnn_kernel_size: int = 5,
        cnn_pool_size: int = 2,
        cnn_dropout: float = 0.3,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_length = input_length
        
        # Activity embedding
        self.activity_embed = nn.Embedding(num_activities, activity_embed_dim)
        
        # CNN backbone
        layers = []
        in_channels = input_channels
        
        for out_channels in cnn_filters:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(cnn_pool_size),
                nn.Dropout(cnn_dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate output size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_length)
            cnn_output = self.cnn(dummy_input)
            cnn_flat_size = cnn_output.numel()
        
        # Combine CNN features with activity embedding
        combined_size = cnn_flat_size + activity_embed_dim
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_size, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout)
        )
        
        # Three separate regression heads
        self.head_acc = nn.Linear(head_hidden_dim, 1)
        self.head_gyro = nn.Linear(head_hidden_dim, 1)
        self.head_freq = nn.Linear(head_hidden_dim, 1)
    
    def forward(self, X, activity):
        """
        Args:
            X: (B, C, L) sensor data
            activity: (B,) activity labels
        
        Returns:
            (B, 3) predictions [acc_rms, gyro_rms, freq]
        """
        # CNN feature extraction
        cnn_out = self.cnn(X)  # (B, C', L')
        cnn_flat = cnn_out.flatten(1)  # (B, C'*L')
        
        # Activity embedding
        act_emb = self.activity_embed(activity)  # (B, embed_dim)
        
        # Combine features
        combined = torch.cat([cnn_flat, act_emb], dim=1)  # (B, combined_size)
        
        # Shared layer
        shared_features = self.shared_fc(combined)  # (B, head_hidden_dim)
        
        # Three heads
        acc_pred = self.head_acc(shared_features).squeeze(-1)  # (B,)
        gyro_pred = self.head_gyro(shared_features).squeeze(-1)  # (B,)
        freq_pred = self.head_freq(shared_features).squeeze(-1)  # (B,)
        
        return torch.stack([acc_pred, gyro_pred, freq_pred], dim=1)  # (B, 3)


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

class WeightedHuberLoss(nn.Module):
    """
    Multi-task Huber loss with per-target weighting.
    """
    
    def __init__(
        self,
        weights: List[float] = [1.0, 1.0, 0.2],
        delta: float = 1.0
    ):
        super().__init__()
        self.weights = torch.tensor(weights)
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 3) predictions
            target: (B, 3) targets
        
        Returns:
            scalar loss
        """
        # Compute Huber loss per target
        losses = self.huber(pred, target)  # (B, 3)
        
        # Apply weights
        weights = self.weights.to(pred.device)
        weighted_losses = losses * weights  # (B, 3)
        
        # Return mean
        return weighted_losses.mean()


# ═══════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X, activity, targets in loader:
        X = X.to(device)
        activity = activity.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        pred = model(X, activity)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, use_log_transform=True):
    """
    Evaluate model and compute metrics.
    
    Returns:
        - loss: average loss
        - metrics: dict with MAE, R², Spearman per target
        - predictions: (N, 3) array
        - targets: (N, 3) array
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, activity, targets in loader:
            X = X.to(device)
            activity = activity.to(device)
            targets = targets.to(device)
            
            pred = model(X, activity)
            loss = criterion(pred, targets)
            
            total_loss += loss.item() * X.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)  # (N, 3)
    targets = np.concatenate(all_targets, axis=0)  # (N, 3)
    
    # Inverse transform if log was used
    if use_log_transform:
        preds[:, 0] = np.expm1(preds[:, 0])  # acc
        preds[:, 1] = np.expm1(preds[:, 1])  # gyro
        targets[:, 0] = np.expm1(targets[:, 0])
        targets[:, 1] = np.expm1(targets[:, 1])
    
    # Compute metrics per target
    metrics = {}
    target_names = ["acc_rms", "gyro_rms", "freq"]
    
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        r2 = r2_score(targets[:, i], preds[:, i])
        
        # Spearman correlation (handles non-linear relationships)
        spearman, _ = spearmanr(targets[:, i], preds[:, i])
        
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_r2"] = r2
        metrics[f"{name}_spearman"] = spearman
    
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, metrics, preds, targets


def plot_predictions(preds, targets, split_name, plot_dir):
    """
    Plot predictions vs targets for all three outputs.
    """
    target_names = ["Acc RMS (m/s²)", "Gyro RMS (deg/s)", "Freq (Hz)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(targets[:, i], preds[:, i], alpha=0.4, s=10)
        ax.plot([targets[:, i].min(), targets[:, i].max()],
                [targets[:, i].min(), targets[:, i].max()],
                'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{split_name}: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"{split_name.lower()}_predictions.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("TREMOR REGRESSION MODEL TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Datasets: {DATASET_VARIANTS}")
    print(f"Sensors: {SENSORS}")
    print(f"Split by subject: {SPLIT_BY_SUBJECT}")
    print(f"Use log transform: {USE_LOG_TRANSFORM}")
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
    
    # Compute normalization stats from training data
    print(f"\nNormalization mode: {NORMALIZATION_MODE}")
    norm_stats = compute_normalization_stats(train_data, mode=NORMALIZATION_MODE)
    print(f"  Mean shape: {norm_stats['mean'].shape}")
    print(f"  Std shape: {norm_stats['std'].shape}")
    print()
    
    # Create datasets
    train_dataset = TremorRegressionDataset(train_data, norm_stats, USE_LOG_TRANSFORM)
    val_dataset = TremorRegressionDataset(val_data, norm_stats, USE_LOG_TRANSFORM)
    test_dataset = TremorRegressionDataset(test_data, norm_stats, USE_LOG_TRANSFORM)
    
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
    input_channels = data["X"].shape[1]  # Total channels from all sensors
    input_length = data["X"].shape[2]    # Window length
    
    model = TremorRegressionCNN(
        input_channels=input_channels,
        input_length=input_length,
        num_activities=num_activities,
        activity_embed_dim=activity_embed_dim,
        cnn_filters=cnn_filters,
        cnn_kernel_size=cnn_kernel_size,
        cnn_pool_size=cnn_pool_size,
        cnn_dropout=cnn_dropout,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  Input: ({input_channels}, {input_length})")
    print(f"  CNN filters: {cnn_filters}")
    print(f"  Activity embedding: {activity_embed_dim}")
    print(f"  Output: 3 regression heads (acc_rms, gyro_rms, freq)")
    print()
    
    # Loss and optimizer
    criterion = WeightedHuberLoss(
        weights=[LAMBDA_ACC, LAMBDA_GYRO, LAMBDA_FREQ],
        delta=HUBER_DELTA
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Loss weights: acc={LAMBDA_ACC}, gyro={LAMBDA_GYRO}, freq={LAMBDA_FREQ}")
    print()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": []
    }
    
    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, USE_LOG_TRANSFORM)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), LOG_DIR / "best_tremor_regression_model.pth")
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Val loss:   {val_loss:.6f}")
            print(f"  Val metrics:")
            for k, v in val_metrics.items():
                print(f"    {k}: {v:.4f}")
            print()
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(LOG_DIR / "best_tremor_regression_model.pth"))
    
    # Final evaluation
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    val_loss, val_metrics, val_preds, val_targets = evaluate(
        model, val_loader, criterion, device, USE_LOG_TRANSFORM
    )
    test_loss, test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, USE_LOG_TRANSFORM
    )
    
    print("Validation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    print()
    
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print()
    
    # Plot predictions
    if SAVE_PLOTS:
        plot_predictions(val_preds, val_targets, "Validation", PLOT_DIR)
        plot_predictions(test_preds, test_targets, "Test", PLOT_DIR)
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
            "use_log_transform": USE_LOG_TRANSFORM,
            "epochs_trained": epoch + 1,
            "batch_size": batch_size,
            "lr": lr,
            "loss_weights": {"acc": LAMBDA_ACC, "gyro": LAMBDA_GYRO, "freq": LAMBDA_FREQ},
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    print(f"Results logged to {LOG_FILE}")
    print("Training complete!")


if __name__ == "__main__":
    main()
