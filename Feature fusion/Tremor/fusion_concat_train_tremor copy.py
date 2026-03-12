"""
fusion_concat_train_tremor.py

Tremor Fusion Training with Subject-Based Splitting

Features:
- Loads embeddings from 3 tremor variants (clean, mild_mod, mod_severe)
- Combines them as augmentations (concatenates train/val/test splits)
- Uses subject-based splitting: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]
- Supports gated or concat fusion
- Compatible with tremor-generated CNN embeddings

Usage:
- Run after CNN feature extraction is complete for all 3 tremor variants
- Embeddings are loaded from Activity_ExtractedFeatures folders
- All 3 variants are combined to increase training data (3x augmentation)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import random
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# -------------------------------
# ⚙️  SAMPLING FREQUENCY CONFIGURATION
# -------------------------------
# Choose which sampling frequency to use (uncomment ONE):

# Option 1: 50 Hz (100 samples per window)
SAMPLING_FREQ = "fs50"
SEQ_LEN = 50*2

# Option 2: 30 Hz (60 samples per window)
# SAMPLING_FREQ = "fs30"
# SEQ_LEN = 60

print(f"\n{'='*70}")
print(f"FUSION TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Sampling Frequency: {SAMPLING_FREQ.upper()}")
print(f"Sequence Length: {SEQ_LEN} samples")
print(f"{'='*70}\n")

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
# Base directory where tremor data lives
WORKSPACE_ROOT = Path("/Volumes/NO NAME/Master Lina/Code")
base_data_dir = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"

# Auto-generate tremor variants based on sampling frequency
tremor_variants = [
    f"s2_w2_{SAMPLING_FREQ}_tremor_clean",
    f"s2_w2_{SAMPLING_FREQ}_tremor_mild_mod",
    f"s2_w2_{SAMPLING_FREQ}_tremor_mod_severe"
]

# Sensors to use - only the ones with trained models and extracted embeddings
# For tremor head, we only trained Acc_arm and Gyro_arm
sensors = ["Acc_arm", "Gyro_arm"]  
embeddings_folder_name = "Tremor_ExtractedFeatures"  # Folder name where CNN embeddings are stored

# -------------------------------
# Tremor augmentation configuration
# -------------------------------
# All 3 tremor variants will be combined (concatenated) as augmentations
# Subject-based splitting: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]
# This matches the CNN training splits

print(f"\nUsing tremor variants as augmentations:")
for variant in tremor_variants:
    print(f"  - {variant}")
print(f"\nSubject-based splits: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]")

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
# Logging, plotting, and model saving
# -------------------------------
SAVE_PLOTS = True
PLOT_DIR = WORKSPACE_ROOT / "gating_plots" / SAMPLING_FREQ
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = WORKSPACE_ROOT / "fusion_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"tremor_fusion_{SAMPLING_FREQ}_experiments.jsonl"

# Model saving directory
MODEL_DIR = WORKSPACE_ROOT / "Tremor_head" / "fusion" / "models" / SAMPLING_FREQ
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

def load_combined_tremor_embeddings(sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Load and combine embeddings from all 3 tremor variants for a sensor.
    
    Each variant folder contains:
      Activity_ExtractedFeatures/{sensor}_embeddings.npz with:
        - train_embeddings, train_activities, train_subjects (subjects 1,3,4,6,8,9)
        - val_embeddings, val_activities, val_subjects (subjects 2,7)
        - test_embeddings, test_activities, test_subjects (subjects 5,10)
        - train_labels, val_labels, test_labels (tremor scores - not used for classification)
    
    Returns dict with combined (concatenated) data:
      Z_train, y_train, Z_val, y_val, Z_test, y_test
    """
    Z_train_all = []
    Z_val_all = []
    Z_test_all = []
    y_train_all = []
    y_val_all = []
    y_test_all = []
    
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
        
        # Concatenate embeddings from this variant
        Z_train_all.append(data["train_embeddings"].astype(np.float32))
        Z_val_all.append(data["val_embeddings"].astype(np.float32))
        Z_test_all.append(data["test_embeddings"].astype(np.float32))
        
        # Use ACTIVITIES (not tremor labels) for classification
        if len(y_train_all) == 0:
            # First variant - save activity labels
            y_train_all = data["train_activities"].astype(np.int64)
            y_val_all = data["val_activities"].astype(np.int64)
            y_test_all = data["test_activities"].astype(np.int64)
        else:
            # Verify activity labels match across variants (should be same for same subjects)
            if not np.array_equal(data["train_activities"], y_train_all):
                raise ValueError(f"Train activity mismatch in {variant_name} for {sensor_name}")
            if not np.array_equal(data["val_activities"], y_val_all):
                raise ValueError(f"Val activity mismatch in {variant_name} for {sensor_name}")
            if not np.array_equal(data["test_activities"], y_test_all):
                raise ValueError(f"Test activity mismatch in {variant_name} for {sensor_name}")
    
    # Concatenate embeddings from all variants (augmentation)
    Z_train = np.concatenate(Z_train_all, axis=0)
    Z_val = np.concatenate(Z_val_all, axis=0)
    Z_test = np.concatenate(Z_test_all, axis=0)
    
    # Repeat activity labels for each variant
    y_train = np.tile(y_train_all, len(tremor_variants))
    y_val = np.tile(y_val_all, len(tremor_variants))
    y_test = np.tile(y_test_all, len(tremor_variants))
    
    return {
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_val": Z_val,
        "y_val": y_val,
        "Z_test": Z_test,
        "y_test": y_test,
    }


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
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    # Set working directory to workspace root
    os.chdir(WORKSPACE_ROOT)
    
    print("="*70)
    print("Tremor Fusion Training - Subject-Based Splitting")
    print("="*70)
    print(f"\nTremor variants (combined as augmentations):")
    for variant in tremor_variants:
        print(f"  - {variant}")
    print(f"\nSubject splits: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]")
    
    # ---- Step 1: Load embeddings from all sensors ----
    print("\n[1/5] Loading embeddings from all sensors...")
    print(f"Sensors: {sensors}")
    
    embeddings_dict = {}
    for sensor in sensors:
        try:
            data = load_combined_tremor_embeddings(sensor)
            embeddings_dict[sensor] = data
            print(f"  ✓ {sensor}: train={data['Z_train'].shape}, val={data['Z_val'].shape}, test={data['Z_test'].shape}")
        except FileNotFoundError as e:
            print(f"  ✗ {sensor}: {e}")
            raise
    
    # ---- Step 2: Verify label alignment ----
    print("\n[2/5] Verifying label alignment...")
    
    # Verify all sensors have same labels
    reference_sensor = sensors[0]
    for split in ["train", "val", "test"]:
        y_key = f"y_{split}"
        reference_labels = embeddings_dict[reference_sensor][y_key]
        
        for sensor in sensors[1:]:
            if not np.array_equal(embeddings_dict[sensor][y_key], reference_labels):
                raise ValueError(f"Label mismatch: {reference_sensor} vs {sensor} in {split} split")
        
        print(f"  ✓ {split}: {len(reference_labels)} samples, labels aligned across all sensors")
    
    # Get dataset info
    embed_dim = embeddings_dict[sensors[0]]["Z_train"].shape[1]
    num_classes = len(np.unique(embeddings_dict[sensors[0]]["y_train"]))
    num_sensors = len(sensors)
    
    n_train = len(embeddings_dict[sensors[0]]["y_train"])
    n_val = len(embeddings_dict[sensors[0]]["y_val"])
    n_test = len(embeddings_dict[sensors[0]]["y_test"])
    
    print(f"\nDataset info:")
    print(f"  Sensors: {num_sensors}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")
    print(f"  Augmentation factor: {len(tremor_variants)}x")
    
    # ---- Step 3: Create simple TensorDatasets ----
    print("\n[3/5] Creating datasets...")
    
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
    class MultiSensorDataset(Dataset):
        def __init__(self, Z_list, y):
            self.Z_list = Z_list
            self.y = y
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return tuple([Z[idx] for Z in self.Z_list] + [self.y[idx]])
    
    train_dataset = MultiSensorDataset(Z_train_list, y_train)
    val_dataset = MultiSensorDataset(Z_val_list, y_val)
    test_dataset = MultiSensorDataset(Z_test_list, y_test)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ---- Step 4: Create model ----
    print("\n[4/5] Creating fusion model...")
    
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
    
    # ---- Step 5: Training loop ----
    print("\n[5/5] Training...")
    
    best_val_loss = float("inf")
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
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f}")
        
        # ---- Early stopping ----
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    # ---- Save best model ----
    print("\nSaving best model...")
    fusion_mode = "gated" if USE_GATING else "concat"
    model_filename = f"fusion_{fusion_mode}_{SAMPLING_FREQ}_tremor.pth"
    model_path = MODEL_DIR / model_filename
    
    torch.save({
        'model_state_dict': best_state if best_state is not None else model.state_dict(),
        'embed_dim': embed_dim,
        'num_sensors': num_sensors,
        'num_classes': num_classes,
        'sensors': sensors,
        'sampling_freq': SAMPLING_FREQ,
        'seq_len': SEQ_LEN,
        'fusion_mode': fusion_mode,
        'best_val_loss': best_val_loss,
        'config': {
            'head_hidden_dims': head_hidden_dims,
            'head_dropout': head_dropout,
            'gate_type': gate_type if USE_GATING else None,
            'gate_hidden': gate_hidden if USE_GATING else None,
            'gate_dropout': gate_dropout if USE_GATING else None,
            'use_layernorm': use_layernorm if USE_GATING else None,
            'alpha_floor': alpha_floor if USE_GATING else None,
        }
    }, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # ---- Test evaluation ----
    print("\nEvaluating on test set...")
    
    model.eval()
    y_pred = []
    gate_log_test = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            Z, y_batch = unpack_batch(batch)
            logits, alphas = model(Z)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
            if USE_GATING and alphas is not None:
                for i in range(num_sensors):
                    gate_log_test[sensors[i]].append(alphas[:, i].cpu().numpy())
    
    # Get true labels
    test_acc = accuracy_score(y_test.numpy(), y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # ---- Logging and plotting ----
    print("\nLogging results...")
    
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
    
    # Log run results
    run_log = {
        "experiment": "tremor_fusion_subject_split",
        "sampling_freq": SAMPLING_FREQ,
        "seq_len": SEQ_LEN,
        "sensors": sensors,
        "tremor_variants": tremor_variants,
        "num_variants": len(tremor_variants),
        "fusion_mode": "gated" if USE_GATING else "concat",
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "gate_mean": gate_mean if USE_GATING else None,
        "gate_std": gate_std if USE_GATING else None,
        "seed": SEED,
        "best_val_loss": float(best_val_loss),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "model_saved": str(model_path),
    }
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(run_log) + "\n")
        print(f"  ✓ Logged to {LOG_FILE}")
    except Exception as e:
        print(f"  ✗ Warning: Could not write to {LOG_FILE}: {e}")
    
    # Plot gating weights
    if USE_GATING and SAVE_PLOTS:
        print("\nPlotting gating weights...")
        
        # Concatenate gate logs
        for k in list(gate_log_test.keys()):
            gate_log_test[k] = np.concatenate(gate_log_test[k], axis=0)
        
        # Mean bar plot
        means = {k: float(v.mean()) for k, v in gate_log_test.items()}
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(means)), list(means.values()))
        plt.xticks(range(len(means)), list(means.keys()), rotation=45, ha="right")
        plt.ylabel("Average gating weight")
        plt.title(f"Test Set - Average Sensor Gating Weights ({SAMPLING_FREQ.upper()})")
        plt.tight_layout()
        plot_file = PLOT_DIR / f"test_gates_mean_{fusion_mode}.png"
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Saved {plot_file}")
        plt.close()
        
        # Distribution plot
        plt.figure(figsize=(10, 4))
        for k, v in gate_log_test.items():
            plt.hist(v, bins=60, alpha=0.4, label=k)
        plt.xlabel("Gating weight")
        plt.ylabel("Count")
        plt.title(f"Test Set - Gating Weight Distributions ({SAMPLING_FREQ.upper()})")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plot_file = PLOT_DIR / f"test_gates_hist_{fusion_mode}.png"
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Saved {plot_file}")
        plt.close()
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
