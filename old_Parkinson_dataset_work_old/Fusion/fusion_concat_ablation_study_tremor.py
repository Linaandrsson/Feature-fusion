"""
fusion_concat_ablation_study_tremor.py

Tremor Sensor Ablation Study Framework

Features:
- Automatic sensor ablation studies with k-sensor subsets
- Tests all possible combinations of k sensors from available sensor list  
- Uses tremor-generated data with configurable variants
- Uses strict subject-based split with explicit holdout subjects
- Comprehensive logging of all combinations and their performance
- Automatic ranking from best to worst accuracy

Usage:
- Specify available sensors (all use same FS for tremor data)
- Set ABLATION_K to desired subset size (or list of sizes)
- Run to test all combinations automatically
- Results are logged and ranked by performance
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
import os
import random
import hashlib
from datetime import datetime
from itertools import combinations
from typing import List, Dict, Tuple, Optional


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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Data paths and sensors
# -------------------------------
# Base directory where tremor data lives (inside Parkinson_dataset_work)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
base_data_dir = PROJECT_ROOT / "Data" / "Tremor_datagenerator_files"

# Tremor variants (combined as augmentations)
tremor_variants = ["s2_w2_fs50_tremor_clean", "s2_w2_fs50_tremor_mild_mod", "s2_w2_fs50_tremor_mod_severe"]
# Optional tag for report folder name.
# Example: "mixed", "clean_only", "tremor_only"
# If empty, folder stays "ablation_reports".
ABLATION_REPORT_TAG = "mixed"
embeddings_folder_name = "Activity_ExtractedFeatures"  # Folder name where CNN embeddings are stored

# Available sensors (ablation will test subsets of these)
ALL_SENSORS = ["Acc_arm", "Gyro_arm", "Mag_arm"]

# Subject-based split configuration (must match feature extractor policy)
HOLDOUT_SUBJECTS = [1]  # Never used in train/val/test for fusion model development
TRAIN_SUBJECTS = [2, 3, 4, 5, 10, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29]
VAL_SUBJECTS = [6, 8, 13, 15, 25]
TEST_SUBJECTS = [7, 9, 18, 19, 27]

# -------------------------------
# Sensor Ablation Configuration
# -------------------------------
ABLATION_K = [len(ALL_SENSORS)]  # List of subset sizes to test (e.g., [2, 3] tests all 2-sensor and 3-sensor combos)
                     # Set to [len(ALL_SENSORS)] to test full sensor set only

print(f"\nUsing tremor variants as augmentations:")
for variant in tremor_variants:
    print(f"  - {variant}")
print("\nUsing strict subject-based split for fusion:")
print(f"  HOLDOUT={HOLDOUT_SUBJECTS}")
print(f"  TRAIN={TRAIN_SUBJECTS}")
print(f"  VAL={VAL_SUBJECTS}")
print(f"  TEST={TEST_SUBJECTS}")

# -------------------------------
# Training hyperparameters
# -------------------------------
batch_size = 128
epochs = 200
lr = 1e-3
patience = 30
min_delta = 1e-4

# -------------------------------
# Fusion model configuration
# -------------------------------
USE_GATING = False  # True = gated fusion, False = concat baseline

# Gating parameters (only used if USE_GATING=True)
gate_type = "sigmoid"  # "sigmoid" or "softmax"
gate_hidden = 64  # 0 = linear gate, >0 = MLP gate
gate_dropout = 0.1
use_layernorm = True
alpha_floor = 0.05

# Classifier head
head_hidden_dims = (64, 64)
head_dropout = 0.4

# -------------------------------
# Logging
# -------------------------------
LOG_DIR = SCRIPT_DIR / "tremor_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "tremor_fusion_ablation.jsonl"

_tag = ABLATION_REPORT_TAG.strip()
if _tag:
    # Keep folder names shell/file-system friendly.
    _safe_tag = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in _tag)
    ABLATION_REPORT_DIR = LOG_DIR / f"ablation_reports_{_safe_tag}"
else:
    ABLATION_REPORT_DIR = LOG_DIR / "ablation_reports"

ABLATION_REPORT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

def load_combined_tremor_embeddings(sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Load and combine embeddings from all configured tremor variants for a sensor.
    Performs strict subject-based splitting for fusion.
    
    Each variant folder contains:
      Activity_ExtractedFeatures/{sensor}_embeddings.npz with:
        - train_embeddings, val_embeddings, test_embeddings (currently have leaky splits)
        - train_activities, val_activities, test_activities
        - train_subjects, val_subjects, test_subjects
        - train_labels, val_labels, test_labels (tremor scores - not used for classification)
    
    Source files may contain split leakage from extraction stage.
    To avoid leakage, this function concatenates all rows from each variant,
    then re-splits by subject IDs using TRAIN_SUBJECTS/VAL_SUBJECTS/TEST_SUBJECTS.
    HOLDOUT_SUBJECTS are explicitly excluded.
    
    Returns dict with combined (concatenated) data:
      Z_train, y_train, Z_val, y_val, Z_test, y_test
    """
    all_embeddings = []
    all_activities = []
    all_subjects = []
    
    for variant_name in tremor_variants:
        variant_dir = base_data_dir / variant_name
        feat_dir = variant_dir / embeddings_folder_name
        npz_path = feat_dir / f"{sensor_name}_embeddings.npz"
        
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {npz_path}\n"
                f"Make sure CNNs have been trained and embeddings extracted for {sensor_name}"
            )
        
        data = np.load(npz_path)
        
        # Concatenate all rows from this variant, then re-split by subject.
        variant_embeddings = np.concatenate(
            [data["train_embeddings"], data["val_embeddings"], data["test_embeddings"]], axis=0
        ).astype(np.float32)
        variant_activities = np.concatenate(
            [data["train_activities"], data["val_activities"], data["test_activities"]], axis=0
        ).astype(np.int64)
        variant_subjects = np.concatenate(
            [data["train_subjects"], data["val_subjects"], data["test_subjects"]], axis=0
        ).astype(np.int64)

        all_embeddings.append(variant_embeddings)
        all_activities.append(variant_activities)
        all_subjects.append(variant_subjects)

    Z_all = np.concatenate(all_embeddings, axis=0)
    y_all = np.concatenate(all_activities, axis=0)
    subj_all = np.concatenate(all_subjects, axis=0)

    holdout_mask = np.isin(subj_all, HOLDOUT_SUBJECTS)
    train_mask = np.isin(subj_all, TRAIN_SUBJECTS)
    val_mask = np.isin(subj_all, VAL_SUBJECTS)
    test_mask = np.isin(subj_all, TEST_SUBJECTS)

    # Safety checks: disjoint subject sets and no holdout leakage.
    if set(TRAIN_SUBJECTS) & set(VAL_SUBJECTS):
        raise ValueError("TRAIN_SUBJECTS and VAL_SUBJECTS overlap")
    if set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS):
        raise ValueError("TRAIN_SUBJECTS and TEST_SUBJECTS overlap")
    if set(VAL_SUBJECTS) & set(TEST_SUBJECTS):
        raise ValueError("VAL_SUBJECTS and TEST_SUBJECTS overlap")
    if (set(HOLDOUT_SUBJECTS) & set(TRAIN_SUBJECTS)) or (set(HOLDOUT_SUBJECTS) & set(VAL_SUBJECTS)) or (set(HOLDOUT_SUBJECTS) & set(TEST_SUBJECTS)):
        raise ValueError("HOLDOUT_SUBJECTS overlap with TRAIN/VAL/TEST")

    Z_train = Z_all[train_mask]
    y_train = y_all[train_mask]
    Z_val = Z_all[val_mask]
    y_val = y_all[val_mask]
    Z_test = Z_all[test_mask]
    y_test = y_all[test_mask]

    train_subjects = sorted(np.unique(subj_all[train_mask]).astype(int).tolist())
    val_subjects = sorted(np.unique(subj_all[val_mask]).astype(int).tolist())
    test_subjects = sorted(np.unique(subj_all[test_mask]).astype(int).tolist())
    holdout_subjects = sorted(np.unique(subj_all[holdout_mask]).astype(int).tolist())

    if train_subjects != sorted(TRAIN_SUBJECTS):
        raise ValueError(f"Train subjects mismatch: {train_subjects} != {sorted(TRAIN_SUBJECTS)}")
    if val_subjects != sorted(VAL_SUBJECTS):
        raise ValueError(f"Val subjects mismatch: {val_subjects} != {sorted(VAL_SUBJECTS)}")
    if test_subjects != sorted(TEST_SUBJECTS):
        raise ValueError(f"Test subjects mismatch: {test_subjects} != {sorted(TEST_SUBJECTS)}")

    print(f"  {sensor_name}: subject-based split applied across {len(tremor_variants)} variants ✓")
    print(f"    Train: {len(Z_train)} samples from subjects {train_subjects}")
    print(f"    Val:   {len(Z_val)} samples from subjects {val_subjects}")
    print(f"    Test:  {len(Z_test)} samples from subjects {test_subjects}")
    print(f"    Holdout (excluded): subjects {holdout_subjects}")
    
    return {
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_val": Z_val,
        "y_val": y_val,
        "Z_test": Z_test,
        "y_test": y_test,
    }


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class MultiSensorDataset(Dataset):
    """Simple dataset for multi-sensor embeddings."""
    
    def __init__(self, Z_list: List[torch.Tensor], y: torch.Tensor):
        self.Z_list = Z_list
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return tuple([Z[idx] for Z in self.Z_list] + [self.y[idx]])


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════

class PaperHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dims=(256, 256), dropout_p: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class SensorGate(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        else:
            self.net = nn.Linear(embed_dim, 1)

    def forward(self, z):
        return self.net(z)


class GatedConcatFusionModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_sensors: int,
        num_classes: int,
        head_hidden_dims=(128, 128),
        head_dropout=0.3,
        gate_type: str = "sigmoid",
        gate_hidden: int = 0,
        gate_dropout: float = 0.0,
        use_layernorm: bool = True,
        alpha_floor: float = 0.0
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.embed_dim = embed_dim
        self.gate_type = gate_type.lower()
        assert self.gate_type in ("sigmoid", "softmax")
        self.alpha_floor = alpha_floor

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_sensors)]) if use_layernorm else None
        self.gate_in_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else None
        self.gates = nn.ModuleList([SensorGate(embed_dim, hidden=gate_hidden) for _ in range(num_sensors)])

        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes,
                              hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        Z_proc = []
        for i, z in enumerate(Z_list):
            if self.norms is not None:
                z = self.norms[i](z)
            if self.gate_in_dropout is not None:
                z = self.gate_in_dropout(z)
            Z_proc.append(z)

        scores = [gate(z) for gate, z in zip(self.gates, Z_proc)]
        scores = torch.cat(scores, dim=1)

        if self.gate_type == "softmax":
            alphas = torch.softmax(scores, dim=1)
        else:
            alphas = torch.sigmoid(scores)

        if self.alpha_floor > 0:
            alphas = self.alpha_floor + (1.0 - self.alpha_floor) * alphas

        gated_blocks = [alphas[:, i:i+1] * Z_proc[i] for i in range(self.num_sensors)]
        z_cat = torch.cat(gated_blocks, dim=1)
        logits = self.head(z_cat)
        return logits, alphas


class ConcatFusionModel(nn.Module):
    def __init__(self, embed_dim: int, num_sensors: int, num_classes: int,
                 head_hidden_dims=(256, 256, 128), head_dropout=0.3):
        super().__init__()
        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes,
                              hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        z = torch.cat(Z_list, dim=1)
        logits = self.head(z)
        return logits, None


# ═══════════════════════════════════════════════════════════════
# TRAINING PIPELINE FOR SENSOR COMBINATIONS
# ═══════════════════════════════════════════════════════════════

def train_sensor_combination(
    sensors: List[str],
    embeddings_dict: Dict[str, Dict[str, np.ndarray]],
    embed_dim: int,
    num_classes: int,
    all_sensors: List[str],
    experiment_id: str
) -> Dict:
    """
    Train and evaluate a specific sensor combination using tremor data.
    
    Args:
        sensors: List of sensor names in this combination
        embeddings_dict: Pre-loaded embeddings for all sensors {sensor_name: {Z_train, y_train, ...}}
        embed_dim: Embedding dimension
        num_classes: Number of classes
        all_sensors: List of all available sensors (for ablation tracking)
        experiment_id: Unique ID for this experiment
    
    Returns:
        Dictionary with results (test_acc, val_acc, val_loss, etc.)
    """
    num_sensors = len(sensors)
    
    # Calculate ablated sensors (those NOT in this combination)
    ablated_sensors = sorted(list(set(all_sensors) - set(sensors)))
    
    print(f"\n{'='*70}")
    print(f"Training combination: {sensors}")
    if ablated_sensors:
        print(f"Ablated sensors: {ablated_sensors}")
    print(f"{'='*70}")
    
    # ---- Step 1: Verify embeddings ----
    print("\n[1/5] Verifying embeddings for this combination...")
    
    for sensor in sensors:
        if sensor not in embeddings_dict:
            raise ValueError(f"Missing embeddings for {sensor}")
    
    # Get basic info
    n_train = len(embeddings_dict[sensors[0]]["y_train"])
    n_val = len(embeddings_dict[sensors[0]]["y_val"])
    n_test = len(embeddings_dict[sensors[0]]["y_test"])
    
    print(f"  ✓ All embeddings available for {num_sensors} sensors")
    print(f"\nDataset info:")
    print(f"  Sensors: {num_sensors}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")
    
    # ---- Step 2: Create datasets ----
    print("\n[2/5] Creating datasets...")
    
    # Prepare tensors for each split
    def prepare_tensors(split):
        Z_key = f"Z_{split}"
        y_key = f"y_{split}"
        
        Z_list = [torch.from_numpy(embeddings_dict[s][Z_key]) for s in sensors]
        y = torch.from_numpy(embeddings_dict[sensors[0]][y_key])
        
        return Z_list, y
    
    Z_train_list, y_train = prepare_tensors("train")
    Z_val_list, y_val = prepare_tensors("val")
    Z_test_list, y_test = prepare_tensors("test")
    
    # Create simple tensor datasets
    train_dataset = MultiSensorDataset(Z_train_list, y_train)
    val_dataset = MultiSensorDataset(Z_val_list, y_val)
    test_dataset = MultiSensorDataset(Z_test_list, y_test)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ---- Step 3: Create model ----
    print("\n[3/5] Creating fusion model...")
    
    if USE_GATING:
        model = GatedConcatFusionModel(
            embed_dim=embed_dim,
            num_sensors=num_sensors,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
            gate_type=gate_type,
            gate_hidden=gate_hidden,
            gate_dropout=gate_dropout,
            use_layernorm=use_layernorm,
            alpha_floor=alpha_floor
        ).to(device)
        print(f"  Mode: GATED (gate_type={gate_type}, gate_hidden={gate_hidden})")
    else:
        model = ConcatFusionModel(
            embed_dim=embed_dim,
            num_sensors=num_sensors,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout
        ).to(device)
        print(f"  Mode: CONCAT BASELINE")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n{model}")
    
    # ---- Step 4: Training loop ----
    print("\n[4/5] Training...")
    
    best_val_f1_macro = -float("inf")
    best_state = None
    no_improve = 0
    
    def unpack_batch(batch):
        *Z, y = batch
        Z = [z.to(device) for z in Z]
        y = y.to(device)
        return Z, y
    
    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            Z, y_batch = unpack_batch(batch)
            
            optimizer.zero_grad()
            logits, _ = model(Z)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * y_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        
        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                Z, y_batch = unpack_batch(batch)
                logits, _ = model(Z)
                loss = criterion(logits, y_batch)
                
                val_loss_sum += loss.item() * y_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        
        # Compute macro F1 on validation set
        val_y_all = []
        val_pred_all = []
        with torch.no_grad():
            for batch in val_loader:
                Z, y_batch = unpack_batch(batch)
                logits, _ = model(Z)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_y_all.extend(y_batch.cpu().numpy())
                val_pred_all.extend(preds)
        val_f1_macro = f1_score(val_y_all, val_pred_all, average='macro')
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"Val F1 (macro) {val_f1_macro:.4f}")
        
        # ---- Early stopping (based on macro F1) ----
        if val_f1_macro > best_val_f1_macro + min_delta:
            best_val_f1_macro = val_f1_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best val F1 (macro): {best_val_f1_macro:.4f}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    # ---- Step 5: Test evaluation ----
    print("\n[5/5] Evaluating on test set...")
    
    model.eval()
    y_pred = []
    test_loss_total = 0.0
    gate_log_test = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            Z, y_batch = unpack_batch(batch)
            logits, alphas = model(Z)
            
            # Compute loss
            loss = criterion(logits, y_batch)
            test_loss_total += loss.item() * len(y_batch)
            
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
            if USE_GATING and alphas is not None:
                for i in range(num_sensors):
                    gate_log_test[sensors[i]].append(alphas[:, i].cpu().numpy())
    
    # Get true labels
    y_test_np = embeddings_dict[sensors[0]]["y_test"]
    test_loss = test_loss_total / len(y_test_np)
    test_acc = accuracy_score(y_test_np, y_pred)
    test_f1_macro = f1_score(y_test_np, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test_np, y_pred, average='weighted')
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1_macro:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    
    # Compute and save confusion matrix with two subplots
    cm = confusion_matrix(y_test_np, y_pred)
    
    # Create figure with 2 subplots: absolute counts and percentages (larger size to fit text)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    # Left plot: Absolute counts
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title("Absolute Counts", fontsize=14, fontweight='bold')
    
    # Reduce text size in left plot
    for labels in [axes[0].texts]:
        for label in labels:
            if hasattr(label, 'set_fontsize'):
                label.set_fontsize(8)
    for text in axes[0].texts:
        text.set_fontsize(8)
    
    # Right plot: Normalized by true label (recall per class - percentage of samples per true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=np.arange(num_classes))
    disp2.plot(ax=axes[1], cmap='Blues', values_format='.1%')
    axes[1].set_title("Recall per True Label (% of samples)", fontsize=14, fontweight='bold')
    
    # Reduce text size in right plot
    for text in axes[1].texts:
        text.set_fontsize(8)
    
    # Overall title
    fig.suptitle(f"Confusion Matrix - Sensors: {sensors}\nTest Acc: {test_acc:.4f}, Test F1 (macro): {test_f1_macro:.4f}", 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save confusion matrix to ablation report folder
    cm_filename = ABLATION_REPORT_DIR / f"cm_{experiment_id}.png"
    plt.savefig(cm_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved to {cm_filename}")
    
    # Compute gate statistics
    gate_mean = {}
    gate_std = {}
    if USE_GATING:
        for s in sensors:
            vals = np.concatenate(gate_log_test[s], axis=0)
            gate_mean[s] = float(vals.mean())
            gate_std[s] = float(vals.std())
        
        print("\nGating weights (test set):")
        for s in sensors:
            print(f"  {s}: {gate_mean[s]:.3f} ± {gate_std[s]:.3f}")
    
    # Return results for ablation tracking
    results = {
        "experiment_id": experiment_id,
        "sensors": sensors,
        "ablated_sensors": ablated_sensors,
        "num_sensors": len(sensors),
        "fusion_mode": "gated" if USE_GATING else "concat",
        "val_accuracy": float(val_acc),
        "val_f1_macro": float(best_val_f1_macro),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "confusion_matrix_file": str(cm_filename.name),
        "gate_mean": gate_mean if USE_GATING else None,
        "gate_std": gate_std if USE_GATING else None,
        "seed": SEED,
    }
    
    # Log run results
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(results) + "\n")
        print(f"  ✓ Logged to {LOG_FILE}")
    except Exception as e:
        print(f"  ✗ Warning: Could not write to {LOG_FILE}: {e}")
    
    print("\n" + "="*70)
    print(f"Combination complete: {sensors}")
    print(f"Test F1 (macro): {test_f1_macro:.4f} | Test Loss: {test_loss:.4f}")
    print("="*70)
    
    return results


def main():
    """
    Main ablation study: Test all k-sensor combinations using tremor data.
    """
    print("="*70)
    print("Tremor Sensor Ablation Study")
    print("="*70)
    print(f"\nTremor variants (combined as augmentations):")
    for variant in tremor_variants:
        print(f"  - {variant}")
    print("\nSubject split policy:")
    print(f"  HOLDOUT={HOLDOUT_SUBJECTS}")
    print(f"  TRAIN={TRAIN_SUBJECTS}")
    print(f"  VAL={VAL_SUBJECTS}")
    print(f"  TEST={TEST_SUBJECTS}")
    
    # Timestamp for this run (readable format: MMDD_HHMM)
    run_timestamp = datetime.now().strftime("%m%d_%H%M")
    ablation_k_list = ABLATION_K if isinstance(ABLATION_K, list) else [ABLATION_K]
    
    # ---- Step 1: Load embeddings from all sensors ----
    print("\n[1/3] Loading embeddings from all sensors...")
    print(f"Sensors: {ALL_SENSORS}")
    
    embeddings_dict = {}
    for sensor in ALL_SENSORS:
        try:
            data = load_combined_tremor_embeddings(sensor)
            embeddings_dict[sensor] = data
            print(f"  ✓ {sensor}: train={data['Z_train'].shape}, val={data['Z_val'].shape}, test={data['Z_test'].shape}")
        except FileNotFoundError as e:
            print(f"  ✗ {sensor}: {e}")
            raise
    
    # ---- Step 2: Verify label alignment ----
    print("\n[2/3] Verifying label alignment...")
    
    # Verify all sensors have same labels
    reference_sensor = ALL_SENSORS[0]
    for split in ["train", "val", "test"]:
        y_key = f"y_{split}"
        reference_labels = embeddings_dict[reference_sensor][y_key]
        
        for sensor in ALL_SENSORS[1:]:
            if not np.array_equal(embeddings_dict[sensor][y_key], reference_labels):
                raise ValueError(f"Label mismatch: {reference_sensor} vs {sensor} in {split} split")
        
        print(f"  ✓ {split}: {len(reference_labels)} samples, labels aligned across all sensors")
    
    # Get dataset info
    embed_dim = embeddings_dict[ALL_SENSORS[0]]["Z_train"].shape[1]
    ref = embeddings_dict[ALL_SENSORS[0]]
    num_classes = len(np.unique(np.concatenate([ref["y_train"], ref["y_val"], ref["y_test"]], axis=0)))
    
    print(f"\nDataset info:")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Augmentation factor: {len(tremor_variants)}x")
    
    # ---- Step 3: Run ablation study ----
    print("\n[3/3] Running ablation study...")
    all_results = []
    
    for k in ablation_k_list:
        print(f"\n{'='*70}")
        print(f"TESTING ALL {k}-SENSOR COMBINATIONS")
        print(f"{'='*70}")
        
        # Create experiment ID for this k-value
        experiment_id = f"tremor_ablation_k{k}_{run_timestamp}"
        print(f"\nExperiment ID: {experiment_id}")
        
        # Generate all k-combinations
        sensor_combinations = list(combinations(range(len(ALL_SENSORS)), k))
        total_combos = len(sensor_combinations)
        
        print(f"\nTotal combinations to test: {total_combos}")
        
        for combo_idx, sensor_indices in enumerate(sensor_combinations, 1):
            # Get sensor names for this combination
            combo_sensors = [ALL_SENSORS[i] for i in sensor_indices]
            
            print(f"\n{'*'*70}")
            print(f"Combination {combo_idx}/{total_combos} (k={k})")
            print(f"{'*'*70}")
            
            try:
                # Train this combination
                result = train_sensor_combination(
                    sensors=combo_sensors,
                    embeddings_dict=embeddings_dict,
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    all_sensors=ALL_SENSORS,
                    experiment_id=experiment_id
                )
                
                result["combination_id"] = combo_idx
                result["k"] = k
                all_results.append(result)
                
            except Exception as e:
                print(f"\n✗ ERROR in combination {combo_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # ---- Final Summary ----
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    
    if not all_results:
        print("\n✗ No results collected!")
        return
    
    # Sort by test F1 macro (descending)
    all_results_sorted = sorted(all_results, key=lambda x: x["test_f1_macro"], reverse=True)
    
    # Calculate dynamic column widths based on content
    max_sensors_len = max(len(", ".join(r["sensors"])) for r in all_results_sorted)
    max_ablated_len = max(len(", ".join(r.get("ablated_sensors", []))) for r in all_results_sorted)
    
    # Add some padding and set minimum widths
    rank_width = 6
    sensors_width = max(max_sensors_len + 2, 10)  # Min 10 chars
    ablated_width = max(max_ablated_len + 2, 10)  # Min 10 chars
    test_f1_width = 10
    test_loss_width = 11
    val_acc_width = 10
    
    total_width = rank_width + sensors_width + ablated_width + test_f1_width + test_loss_width + val_acc_width + 5  # +5 for spacing
    
    # Print summary
    print(f"\nTested {len(all_results)} sensor combinations")
    print(f"\nRESULTS RANKED BY TEST F1 (MACRO):")
    print(f"{'='*total_width}")
    print(f"{'Rank':<{rank_width}} {'Sensors':<{sensors_width}} {'Ablated':<{ablated_width}} {'Test F1':<{test_f1_width}} {'Test Loss':<{test_loss_width}} {'Val Acc':<{val_acc_width}}")
    print(f"{'-'*total_width}")
    
    for rank, result in enumerate(all_results_sorted, 1):
        sensors_str = ", ".join(result["sensors"])
        ablated_str = ", ".join(result.get("ablated_sensors", []))
        print(f"{rank:<{rank_width}} {sensors_str:<{sensors_width}} {ablated_str:<{ablated_width}} {result['test_f1_macro']:.4f}      "
              f"{result['test_loss']:.4f}       {result['val_accuracy']:.4f}")
    
    # Print best combination
    best = all_results_sorted[0]
    print(f"\n{'='*70}")
    print("BEST COMBINATION:")
    print(f"{'='*70}")
    print(f"  Sensors: {', '.join(best['sensors'])}")
    if best.get('ablated_sensors'):
        print(f"  Ablated: {', '.join(best['ablated_sensors'])}")
    print(f"  Test F1 (macro): {best['test_f1_macro']:.4f}")
    print(f"  Test F1 (weighted): {best['test_f1_weighted']:.4f}")
    print(f"  Test Loss: {best['test_loss']:.4f}")
    print(f"  Val Loss: {best['val_loss']:.4f}")
    print(f"{'='*70}")
    
    # Save summary report
    report_file = ABLATION_REPORT_DIR / f"tremor_ablation_k{ablation_k_list}_report_{run_timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("TREMOR SENSOR ABLATION STUDY RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {run_timestamp}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"K values tested: {ablation_k_list}\n")
        f.write(f"Total combinations: {len(all_results)}\n\n")
        
        f.write("Tremor variants used:\n")
        for variant in tremor_variants:
            f.write(f"  - {variant}\n")
        f.write("\nSplits: using train/val/test from extracted embedding files\n\n")
        f.write(f"Holdout subjects (excluded): {HOLDOUT_SUBJECTS}\n")
        f.write(f"Train subjects: {TRAIN_SUBJECTS}\n")
        f.write(f"Val subjects: {VAL_SUBJECTS}\n")
        f.write(f"Test subjects: {TEST_SUBJECTS}\n\n")
        
        f.write("="*total_width + "\n")
        f.write("RESULTS RANKED BY TEST F1 (MACRO)\n")
        f.write("="*total_width + "\n")
        f.write(f"{'Rank':<{rank_width}} {'Sensors':<{sensors_width}} {'Ablated':<{ablated_width}} {'Test F1':<{test_f1_width}} {'Test Loss':<{test_loss_width}} {'Val Acc':<{val_acc_width}}\n")
        f.write("-"*total_width + "\n")
        
        for rank, result in enumerate(all_results_sorted, 1):
            sensors_str = ", ".join(result["sensors"])
            ablated_str = ", ".join(result.get("ablated_sensors", []))
            f.write(f"{rank:<{rank_width}} {sensors_str:<{sensors_width}} {ablated_str:<{ablated_width}} {result['test_f1_macro']:.4f}      "
                    f"{result['test_loss']:.4f}       {result['val_accuracy']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("BEST COMBINATION\n")
        f.write("="*70 + "\n")
        f.write(f"  Sensors: {', '.join(best['sensors'])}\n")
        if best.get('ablated_sensors'):
            f.write(f"  Ablated: {', '.join(best['ablated_sensors'])}\n")
        f.write(f"  Test F1 (macro): {best['test_f1_macro']:.4f}\n")
        f.write(f"  Test F1 (weighted): {best['test_f1_weighted']:.4f}\n")
        f.write(f"  Test Loss: {best['test_loss']:.4f}\n")
        f.write(f"  Val Loss: {best['val_loss']:.4f}\n")
    
    print(f"\n✓ Summary report saved to: {report_file}")


if __name__ == "__main__":
    main()
