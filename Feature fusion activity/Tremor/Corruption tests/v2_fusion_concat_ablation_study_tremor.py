"""
fusion_concat_ablation_study_tremor.py

Tremor Sensor Ablation Study Framework

Features:
- Automatic sensor ablation studies with k-sensor subsets
- Tests all possible combinations of k sensors from available sensor list  
- Uses tremor-generated data with 3 augmentation variants
- Subject-based splitting: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pathlib import Path
from collections import defaultdict
import csv
import json
import os
import random
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on servers without display
import matplotlib.pyplot as plt
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
SEED = 39
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
# parents[0]=Corruption tests, parents[1]=Tremor, parents[2]=Feature fusion activity, parents[3]=Code (workspace root)
base_data_dir = Path(__file__).parents[3] / "data" / "Tremor_datagenerator_files"

# tremor_variants — TRAINING DATA for the fusion classifier head.
#
# These variants are used as the train/val split. Each variant listed here is loaded
# and concatenated as a data augmentation — more variants = larger/richer training set.
# The test set always comes from corrupt_test_variant (see below), NOT from these.
#
# Available options (all at 4 s window / 4 s stride / fs=50):
#   "s4_w4_fs50_tremor_clean"          → clean MHEALTH signal with simulated tremor (no sensor corruption)
#   "s4_w4_fs50_tremor_mild_mod"       → mild-to-moderate tremor severity
#   "s4_w4_fs50_tremor_mod_severe"     → moderate-to-severe tremor severity
#   "s4_w4_fs50_tremor_clean_awgn"     → clean tremor + AWGN sensor noise augmentation
#   "s4_w4_fs50_tremor_clean_rotation" → clean tremor + random axis rotation augmentation
#
# NOTE: mixing s4_w4 and s2_w2 variants gives inconsistent window sizes and will crash.
#
# Examples:
#   tremor_variants = ["s4_w4_fs50_tremor_clean"]                              # clean only (current)
#   tremor_variants = ["s4_w4_fs50_tremor_clean", "s4_w4_fs50_tremor_mild_mod", "s4_w4_fs50_tremor_mod_severe"]
#   tremor_variants = ["s4_w4_fs50_tremor_clean", "s4_w4_fs50_tremor_clean_awgn", "s4_w4_fs50_tremor_clean_rotation"]
# Optional tag for report folder name.
# Example: "mixed", "clean_only", "tremor_only"
# If empty, folder stays "ablation_reports".

#mixed:
tremor_variants = ["s4_w4_fs50_tremor_clean", "s4_w4_fs50_tremor_mild_mod", "s4_w4_fs50_tremor_mod_severe"]  # mixed 
ABLATION_REPORT_TAG = "mixed_k4_mixed_test_accAnkle_fullDropout"
embeddings_folder_name = "ExtractedFeatures_mixed"  # Folder name where CNN embeddings are stored (train/val)

#clean:
# tremor_variants = ["s4_w4_fs50_tremor_clean"]  # clean 
# ABLATION_REPORT_TAG = "clean_k4_magArm_fullDropout"
# embeddings_folder_name = "ExtractedFeatures_clean"  # Folder name where CNN embeddings are stored (train/val)


# Corrupt variant(s) used ONLY for test evaluation (train/val still use tremor_variants above).
# Can be a single string or a list of strings. If a list, equal weight is given to each variant.
# corrupt_test_mix (below) overrides this per sensor.
# corrupt_test_variant = [
#     "s4_w4_fs50_tremor_clean",
#     "s4_w4_fs50_tremor_mild_mod",
#     "s4_w4_fs50_tremor_mod_severe",
# ]  # mixed tremor test set

# clean single-variant examples:
# corrupt_test_variant = "s4_w4_fs50_tremor_clean_awgn_a000"
# corrupt_test_variant = ["s4_w4_fs50_tremor_clean_awgn_a000"]

# fallback used when a sensor is NOT listed in corrupt_test_mix below
# (currently unused — all sensors are covered by corrupt_test_mix)
# corrupt_test_variant = "s4_w4_fs50_tremor_clean"

corrupt_test_variant = ["s4_w4_fs50_tremor_clean", "s4_w4_fs50_tremor_mild_mod", "s4_w4_fs50_tremor_mod_severe"]  # mixed tremor test set

# fallback for sensors not in corrupt_test_mix
# Per-sensor test distribution — overrides corrupt_test_variant for listed sensors.
# Format: {sensor_name: {variant_name: probability}}  — probabilities must sum to 1.0 per sensor.
#
# All variants use the folder name defined by embeddings_folder_name above
# ("ExtractedFeatures_clean" for clean-trained CNNs, "ExtractedFeatures_mixed" for mixed-trained CNNs).
#
# Examples:
#   "Acc_ankle": {"s4_w4_fs50_tremor_clean": 1.0}                                    → 100 % clean
#   "Acc_arm":   {"s4_w4_fs50_corrupt_awgn": 0.8, "s4_w4_fs50_corrupt_dropout": 0.2} → mixed
#

# Available sensors (ablation will test subsets of these)
ALL_SENSORS = ["Acc_ankle", "Acc_arm", "Mag_ankle", "Mag_arm"]

corrupt_test_mix: Dict[str, Dict[str, float]] = {
    "Acc_ankle":  {"s4_w4_fs50_tremor_clean_fullDrop": 1},
}
# Leave the dict empty {} to use corrupt_test_variant (100 %) for all sensors.
#corrupt_test_mix: Dict[str, Dict[str, float]] = {}
# corrupt_test_mix: Dict[str, Dict[str, float]] = {
#     "Acc_ankle":  {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "Acc_arm":    {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "Acc_chest":  {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "ECG":        {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "Gyro_ankle": {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "Gyro_arm":   {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "Mag_ankle":  {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
#     "Mag_arm":    {"s4_w4_fs50_tremor_clean": 0.50, "s4_w4_fs50_corrupt_awgn":     0.40, "s4_w4_fs50_corrupt_dropout": 0.10},
# }


# corrupt_test_mix: Dict[str, Dict[str, float]] = {
#     "Acc_ankle":  {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "Acc_arm":    {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "Gyro_ankle": {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "Gyro_arm":   {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "Mag_ankle":  {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "Mag_arm":    {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "Acc_chest":  {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
#     "ECG":        {"s4_w4_fs50_tremor_clean_awgn_a000": 0.25, "s4_w4_fs50_tremor_clean_awgn_a018": 0.25, "s4_w4_fs50_tremor_clean_dropout_p010": 0.25, "s4_w4_fs50_tremor_clean_orient_r045": 0.25},
# }

# Subject-based splits (to prevent data leakage)
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]
# TRAIN_SUBJECTS = [1, 3, 4, 6, 8, 9] (implicitly, all others)

# -------------------------------
# Sensor Ablation Configuration
# -------------------------------
ABLATION_K = [4]  # List of subset sizes to test (e.g., [2, 3] tests all 2-sensor and 3-sensor combos)
                     # [8] = full 8-sensor set only; expand to [1,2,...,8] for full ablation sweep

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
# Logging
# -------------------------------
# Use script's directory as base for logs
SCRIPT_DIR = Path(__file__).parent

LOG_DIR = SCRIPT_DIR / "corruption_logs"
LOG_DIR.mkdir(exist_ok=True)

_tag = ABLATION_REPORT_TAG.strip()
if _tag:
    # Keep folder names shell/file-system friendly.
    _safe_tag = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in _tag)
    ABLATION_REPORT_DIR = LOG_DIR / f"ablation_reports_{_safe_tag}"
    ABLATION_JSON_DIR   = LOG_DIR / f"ablation_json_files_{_safe_tag}"
else:
    ABLATION_REPORT_DIR = LOG_DIR / "ablation_reports"
    ABLATION_JSON_DIR   = LOG_DIR / "ablation_json_files"

ABLATION_REPORT_DIR.mkdir(exist_ok=True)
ABLATION_JSON_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

def load_combined_tremor_embeddings(sensor_name: str) -> Dict[str, np.ndarray]:
    """
    Load and combine embeddings from all 3 tremor variants for a sensor.
    Uses SUBJECT-BASED splits to prevent data leakage.
    
    Each variant folder contains:
      {embeddings_folder_name}/{sensor}_embeddings.npz with:
        - train_embeddings, val_embeddings, test_embeddings (currently have leaky splits)
        - train_activities, val_activities, test_activities
        - train_subjects, val_subjects, test_subjects
        - train_labels, val_labels, test_labels (tremor scores - not used for classification)
    
    We concatenate ALL data (train+val+test), then re-split by SUBJECT ID:
      TEST: subjects [5, 10]
      VAL: subjects [2, 7]  
      TRAIN: subjects [1, 3, 4, 6, 8, 9]
    
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
        
        # Concatenate ALL data (train+val+test) from this variant
        variant_embeddings = np.concatenate([
            data["train_embeddings"],
            data["val_embeddings"],
            data["test_embeddings"]
        ], axis=0).astype(np.float32)
        
        variant_activities = np.concatenate([
            data["train_activities"],
            data["val_activities"],
            data["test_activities"]
        ], axis=0).astype(np.int64)
        
        variant_subjects = np.concatenate([
            data["train_subjects"],
            data["val_subjects"],
            data["test_subjects"]
        ], axis=0).astype(np.int64)
        
        all_embeddings.append(variant_embeddings)
        all_activities.append(variant_activities)
        all_subjects.append(variant_subjects)
    
    # Concatenate across variants (augmentation)
    Z_all = np.concatenate(all_embeddings, axis=0)
    y_all = np.concatenate(all_activities, axis=0)
    subj_all = np.concatenate(all_subjects, axis=0)
    
    # Now do SUBJECT-BASED splitting
    train_mask = ~np.isin(subj_all, TEST_SUBJECTS + VAL_SUBJECTS)
    val_mask = np.isin(subj_all, VAL_SUBJECTS)
    test_mask = np.isin(subj_all, TEST_SUBJECTS)
    
    Z_train = Z_all[train_mask]
    y_train = y_all[train_mask]
    Z_val = Z_all[val_mask]
    y_val = y_all[val_mask]
    Z_test = Z_all[test_mask]
    y_test = y_all[test_mask]
    
    # Verify splits are correct
    train_subjects = np.unique(subj_all[train_mask])
    val_subjects = np.unique(subj_all[val_mask])
    test_subjects = np.unique(subj_all[test_mask])
    
    expected_train = sorted([1, 3, 4, 6, 8, 9])
    expected_val = sorted(VAL_SUBJECTS)
    expected_test = sorted(TEST_SUBJECTS)
    
    assert list(train_subjects) == expected_train, f"Train subjects mismatch: {train_subjects} != {expected_train}"
    assert list(val_subjects) == expected_val, f"Val subjects mismatch: {val_subjects} != {expected_val}"
    assert list(test_subjects) == expected_test, f"Test subjects mismatch: {test_subjects} != {expected_test}"
    
    print(f"  {sensor_name}: Subject-based split verified ✓")
    print(f"    Train: {len(Z_train)} samples from subjects {list(train_subjects)}")
    print(f"    Val:   {len(Z_val)} samples from subjects {list(val_subjects)}")
    print(f"    Test:  {len(Z_test)} samples from subjects {list(test_subjects)}")
    
    return {
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_val": Z_val,
        "y_val": y_val,
        "Z_test": Z_test,
        "y_test": y_test,
    }


def _get_embeddings_folder(variant_name: str) -> str:
    """Return embedding subfolder name (determined by embeddings_folder_name config)."""
    return embeddings_folder_name


def _load_test_windows_from_variant(variant_name: str, sensor_name: str):
    """Load and subject-filter test embeddings from a single variant. Returns (Z_test, y_test, subj)."""
    folder   = _get_embeddings_folder(variant_name)
    npz_path = base_data_dir / variant_name / folder / f"{sensor_name}_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {npz_path}\n"
            f"Run the appropriate extraction script first."
        )
    data     = np.load(npz_path)
    Z_all    = np.concatenate([data["train_embeddings"], data["val_embeddings"], data["test_embeddings"]], axis=0).astype(np.float32)
    y_all    = np.concatenate([data["train_activities"],  data["val_activities"],  data["test_activities"]],  axis=0).astype(np.int64)
    subj_all = np.concatenate([data["train_subjects"],    data["val_subjects"],    data["test_subjects"]],    axis=0).astype(np.int64)
    mask     = np.isin(subj_all, TEST_SUBJECTS)
    return Z_all[mask], y_all[mask], subj_all[mask]  # per-window subject array


def load_mixed_corrupt_test_embeddings(sensor_name: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Load test-split embeddings for one sensor with optional per-window variant mixing.

    Resolves the mix from corrupt_test_mix[sensor_name] if defined,
    otherwise falls back to corrupt_test_variant (equal weight if a list).
    Per-window variant selection is drawn from `rng` (seeded for reproducibility).

    If variants have different window counts for the same subject, they are aligned
    subject-wise: for each subject only the first min(count_across_variants) windows
    are kept. Surplus windows are skipped and reported via the returned 'skipped' key.
    """
    _ctv = corrupt_test_variant
    if isinstance(_ctv, str):
        _fallback: Dict[str, float] = {_ctv: 1.0}
    else:
        _fallback = {v: 1.0 / len(_ctv) for v in _ctv}
    mix: Dict[str, float] = (corrupt_test_mix.get(sensor_name) or {}) or _fallback

    total_p = sum(mix.values())
    if abs(total_p - 1.0) > 1e-5:
        raise ValueError(
            f"corrupt_test_mix probabilities for {sensor_name} sum to {total_p:.4f}, must be 1.0"
        )

    # Load per-window (Z, y, subj) arrays for every variant in the mix
    variant_Z:    Dict[str, np.ndarray] = {}
    variant_y:    Dict[str, np.ndarray] = {}
    variant_subj: Dict[str, np.ndarray] = {}

    for variant_name in mix:
        Z, y, subj = _load_test_windows_from_variant(variant_name, sensor_name)
        unique_subj = np.unique(subj)
        assert list(sorted(unique_subj.tolist())) == sorted(TEST_SUBJECTS), \
            f"Test subjects mismatch in {variant_name}/{sensor_name}: {unique_subj}"
        variant_Z[variant_name]    = Z
        variant_y[variant_name]    = y
        variant_subj[variant_name] = subj

    # Subject-wise alignment: for each subject keep only min(count_across_variants) windows
    aligned_Z:          Dict[str, List[np.ndarray]] = {v: [] for v in mix}
    aligned_y_parts:    List[np.ndarray]            = []
    aligned_subj_parts: List[np.ndarray]            = []
    skipped_per_variant: Dict[str, int]             = {v: 0 for v in mix}
    intra_skipped = 0

    for subj_id in sorted(TEST_SUBJECTS):
        subj_idx = {v: np.where(variant_subj[v] == subj_id)[0] for v in mix}
        counts   = {v: len(idx) for v, idx in subj_idx.items()}
        n_keep   = min(counts.values())
        n_max    = max(counts.values())

        if n_max > n_keep:
            for v, cnt in counts.items():
                skipped_per_variant[v] += cnt - n_keep
            intra_skipped += n_max - n_keep

        # Verify label alignment for the kept windows
        ref_v   = next(iter(mix))
        ref_idx = subj_idx[ref_v][:n_keep]
        ref_y_s = variant_y[ref_v][ref_idx]

        for v in mix:
            idx_keep = subj_idx[v][:n_keep]
            y_v = variant_y[v][idx_keep]
            if not np.array_equal(y_v, ref_y_s):
                raise ValueError(
                    f"Activity label mismatch for {sensor_name}/subject {subj_id} in '{v}' — "
                    f"variants must be window-aligned."
                )
            aligned_Z[v].append(variant_Z[v][idx_keep])

        aligned_y_parts.append(ref_y_s)
        aligned_subj_parts.append(np.full(n_keep, subj_id, dtype=np.int64))

    # Concatenate across subjects
    for v in mix:
        aligned_Z[v] = np.concatenate(aligned_Z[v], axis=0)
    ref_y     = np.concatenate(aligned_y_parts,    axis=0)
    subj_test = np.concatenate(aligned_subj_parts, axis=0)
    n_test    = len(ref_y)

    if intra_skipped > 0:
        skip_str = ", ".join(f"{v}: -{cnt}" for v, cnt in skipped_per_variant.items() if cnt > 0)
        print(f"  {sensor_name} [intra-variant alignment: skipped {intra_skipped} windows ({skip_str})]")

    # Single variant — no sampling needed
    if len(mix) == 1:
        only = next(iter(mix))
        print(f"  {sensor_name} [test]: {n_test} windows from '{only}' (100 %) ✓")
        return {"Z_test": aligned_Z[only], "y_test": ref_y, "subj_test": subj_test, "skipped": intra_skipped}

    # Per-window sampling
    variant_names = list(mix.keys())
    probs         = np.array([mix[v] for v in variant_names])
    chosen        = rng.choice(len(variant_names), size=n_test, p=probs)

    embed_dim = aligned_Z[variant_names[0]].shape[1]
    Z_mixed   = np.empty((n_test, embed_dim), dtype=np.float32)
    for i, vi in enumerate(chosen):
        Z_mixed[i] = aligned_Z[variant_names[vi]][i]

    counts_chosen = {v: int((chosen == i).sum()) for i, v in enumerate(variant_names)}
    mix_str = ", ".join(f"{v}: {c}/{n_test} ({c/n_test*100:.0f}%)" for v, c in counts_chosen.items())
    print(f"  {sensor_name} [test mix]: {mix_str} ✓")
    return {"Z_test": Z_mixed, "y_test": ref_y, "subj_test": subj_test, "skipped": intra_skipped}


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
    experiment_id: str,
    json_file: Path = None,
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
    print(f"  Train/Val: {tremor_variants}")
    print(f"  Test:      {corrupt_test_variant}  (corruption robustness)")
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    
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
    
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    
    def unpack_batch(batch):
        *Z, y = batch
        Z = [z.to(device, non_blocking=True) for z in Z]
        y = y.to(device, non_blocking=True)
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
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                Z, y_batch = unpack_batch(batch)
                logits, _ = model(Z)
                loss = criterion(logits, y_batch)
                
                val_loss_sum += loss.item() * y_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} macro_f1 {val_f1_macro:.4f}")
        
        # ---- Early stopping ----
        if val_f1_macro > best_val_macro_f1 + min_delta:
            best_val_macro_f1 = val_f1_macro
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best val macro F1: {best_val_macro_f1:.4f}")
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
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1_macro:.4f}")
    
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
        "val_f1_macro": float(best_val_macro_f1),
        "val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "y_pred": list(int(v) for v in y_pred),
        "y_true": list(int(v) for v in y_test_np),
        "gate_mean": gate_mean if USE_GATING else None,
        "gate_std": gate_std if USE_GATING else None,
        "seed": SEED,
        "embeddings_folder": embeddings_folder_name,
        "tremor_variants": tremor_variants,
        "corrupt_test_variant": corrupt_test_variant,
        "corrupt_test_mix": {
            s: (corrupt_test_mix.get(s) or (
                {corrupt_test_variant: 1.0} if isinstance(corrupt_test_variant, str)
                else {v: 1.0 / len(corrupt_test_variant) for v in corrupt_test_variant}
            ))
            for s in sensors
        },
    }
    
    # Log run results
    if json_file is not None:
        try:
            with open(json_file, "a") as f:
                f.write(json.dumps(results) + "\n")
        except Exception as e:
            print(f"  ✗ Warning: Could not write to {json_file}: {e}")
    
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
    print(f"\nTremor variants (train/val augmentations):")
    for variant in tremor_variants:
        print(f"  - {variant}")
    print(f"\nCorrupt test variant: {corrupt_test_variant}")
    print(f"Subject splits: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]")

    # Timestamp for this run — includes seed so files are unique across runs
    _ts = datetime.now().strftime("%m%d_%H%M")
    run_timestamp = f"{_ts}_s{SEED}"
    ablation_k_list = ABLATION_K if isinstance(ABLATION_K, list) else [ABLATION_K]

    # ---- Step 1: Load train/val embeddings from all sensors ----
    print("\n[1/3] Loading train/val embeddings (tremor variants)...")
    print(f"Sensors: {ALL_SENSORS}")

    embeddings_dict = {}
    for sensor in ALL_SENSORS:
        try:
            data = load_combined_tremor_embeddings(sensor)
            embeddings_dict[sensor] = data
            print(f"  ✓ {sensor}: train={data['Z_train'].shape}, val={data['Z_val'].shape}")
        except FileNotFoundError as e:
            print(f"  ✗ {sensor}: {e}")
            raise

    # ---- Replace test split with (optionally mixed) corrupt variant embeddings ----
    # Use a seeded RNG so per-window sampling is reproducible across runs with the same SEED.
    test_sample_rng = np.random.default_rng(SEED)
    print(f"\n  Loading test embeddings (per-sensor corrupt mix, seed={SEED})...")
    intra_skipped_per_sensor: Dict[str, int] = {}
    for sensor in ALL_SENSORS:
        corrupt_data = load_mixed_corrupt_test_embeddings(sensor, test_sample_rng)
        embeddings_dict[sensor]["Z_test"]    = corrupt_data["Z_test"]
        embeddings_dict[sensor]["y_test"]    = corrupt_data["y_test"]
        embeddings_dict[sensor]["subj_test"] = corrupt_data["subj_test"]
        intra_skipped_per_sensor[sensor]     = corrupt_data["skipped"]

    # ---- Cross-sensor alignment: trim all sensors to global min per subject ----
    # When different sensors use different mixes, their test window counts may differ.
    # We trim every sensor to the minimum count per subject so the fusion model receives
    # one consistent row of embeddings per window index.
    global_min_per_subj = {
        sid: min(int((embeddings_dict[s]["subj_test"] == sid).sum()) for s in ALL_SENSORS)
        for sid in sorted(TEST_SUBJECTS)
    }
    cross_skipped_per_sensor: Dict[str, int] = {s: 0 for s in ALL_SENSORS}
    for s in ALL_SENSORS:
        subj_arr = embeddings_dict[s]["subj_test"]
        n_before = len(subj_arr)
        keep_idx = np.concatenate([
            np.where(subj_arr == sid)[0][: global_min_per_subj[sid]]
            for sid in sorted(TEST_SUBJECTS)
        ])
        n_after = len(keep_idx)
        if n_after < n_before:
            cross_skipped_per_sensor[s] = n_before - n_after
            embeddings_dict[s]["Z_test"]    = embeddings_dict[s]["Z_test"][keep_idx]
            embeddings_dict[s]["y_test"]    = embeddings_dict[s]["y_test"][keep_idx]
            embeddings_dict[s]["subj_test"] = subj_arr[keep_idx]
    cross_total = sum(cross_skipped_per_sensor.values())
    if cross_total > 0:
        print(f"  Cross-sensor trim: {cross_total} windows removed to align sensor counts")
        for s in ALL_SENSORS:
            if cross_skipped_per_sensor[s] > 0:
                print(f"    {s}: -{cross_skipped_per_sensor[s]}")

    # ---- Step 2: Verify label alignment ----
    print("\n[2/3] Verifying label alignment...")

    reference_sensor = ALL_SENSORS[0]
    # Train/val: aligned across all sensors (from same tremor variants)
    for split in ["train", "val"]:
        y_key = f"y_{split}"
        reference_labels = embeddings_dict[reference_sensor][y_key]
        for sensor in ALL_SENSORS[1:]:
            if not np.array_equal(embeddings_dict[sensor][y_key], reference_labels):
                raise ValueError(f"Label mismatch: {reference_sensor} vs {sensor} in {split} split")
        print(f"  ✓ {split}: {len(reference_labels)} samples, labels aligned across all sensors")
    # Test: verify corrupt test labels are aligned
    ref_test = embeddings_dict[reference_sensor]["y_test"]
    for sensor in ALL_SENSORS[1:]:
        if not np.array_equal(embeddings_dict[sensor]["y_test"], ref_test):
            raise ValueError(f"Corrupt test label mismatch: {reference_sensor} vs {sensor}")
    print(f"  ✓ test (corrupt): {len(ref_test)} samples, labels aligned across all sensors")

    # Get dataset info
    embed_dim = embeddings_dict[ALL_SENSORS[0]]["Z_train"].shape[1]
    num_classes = len(np.unique(embeddings_dict[ALL_SENSORS[0]]["y_train"]))

    print(f"\nDataset info:")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Train/val augmentation factor: {len(tremor_variants)}x")
    print(f"  Corrupt test variant: {corrupt_test_variant}")
    
    # ---- Step 3: Run ablation study ----
    print("\n[3/3] Running ablation study...")
    all_results = []
    
    for k in ablation_k_list:
        print(f"\n{'='*70}")
        print(f"TESTING ALL {k}-SENSOR COMBINATIONS")
        print(f"{'='*70}")
        
        # Create experiment ID and per-run jsonl file for this k-value
        experiment_id = f"tremor_ablation_k{k}_{run_timestamp}"
        json_file = ABLATION_JSON_DIR / f"tremor_ablation_k{k}_{run_timestamp}.jsonl"
        print(f"\nExperiment ID: {experiment_id}")
        print(f"JSON log:      {json_file}")
        
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
                    experiment_id=experiment_id,
                    json_file=json_file,
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
    test_acc_width = 11
    test_loss_width = 11
    val_acc_width = 10
    
    total_width = rank_width + sensors_width + ablated_width + test_f1_width + test_acc_width + test_loss_width + val_acc_width + 5  # +5 for spacing
    
    # Print summary
    print(f"\nTested {len(all_results)} sensor combinations")
    print(f"\nRESULTS RANKED BY TEST F1 (MACRO):")
    print(f"{'='*total_width}")
    print(f"{'Rank':<{rank_width}} {'Sensors':<{sensors_width}} {'Ablated':<{ablated_width}} {'Test F1':<{test_f1_width}} {'Test Acc':<{test_acc_width}} {'Test Loss':<{test_loss_width}} {'Val Acc':<{val_acc_width}}")
    print(f"{'-'*total_width}")
    
    for rank, result in enumerate(all_results_sorted, 1):
        sensors_str = ", ".join(result["sensors"])
        ablated_str = ", ".join(result.get("ablated_sensors", []))
        print(f"{rank:<{rank_width}} {sensors_str:<{sensors_width}} {ablated_str:<{ablated_width}} {result['test_f1_macro']:.4f}      "
              f"{result['test_accuracy']:.4f}     {result['test_loss']:.4f}       {result['val_accuracy']:.4f}")
    
    # Print best combination
    best = all_results_sorted[0]
    print(f"\n{'='*70}")
    print("BEST COMBINATION:")
    print(f"{'='*70}")
    print(f"  Sensors: {', '.join(best['sensors'])}")
    if best.get('ablated_sensors'):
        print(f"  Ablated: {', '.join(best['ablated_sensors'])}")
    print(f"  Test F1 (macro): {best['test_f1_macro']:.4f}")
    print(f"  Test Accuracy:   {best['test_accuracy']:.4f}")
    print(f"  Val F1 (macro):  {best['val_f1_macro']:.4f}")
    print(f"  Test Loss: {best['test_loss']:.4f}")
    print(f"  Val Loss:  {best['val_loss']:.4f}")
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
        
        f.write("Train/val tremor variants:\n")
        for variant in tremor_variants:
            f.write(f"  - {variant}\n")
        f.write(f"Embeddings folder (CNN extractor): {embeddings_folder_name}\n")
        f.write(f"Corrupt test variant (fallback): {corrupt_test_variant}\n")
        f.write(f"\nPer-sensor test mix:\n")
        for s in ALL_SENSORS:
            _ctv = corrupt_test_variant
            _fb = {_ctv: 1.0} if isinstance(_ctv, str) else {v: 1.0/len(_ctv) for v in _ctv}
            mix_s = corrupt_test_mix.get(s) or _fb
            mix_str = ", ".join(f"{v}: {p:.2f}" for v, p in mix_s.items())
            f.write(f"  {s:<15}: {mix_str}\n")
        f.write(f"\nSubject splits: TEST=[5,10], VAL=[2,7], TRAIN=[1,3,4,6,8,9]\n\n")
        
        f.write("="*total_width + "\n")
        f.write("RESULTS RANKED BY TEST F1 (MACRO)\n")
        f.write("="*total_width + "\n")
        f.write(f"{'Rank':<{rank_width}} {'Sensors':<{sensors_width}} {'Ablated':<{ablated_width}} {'Test F1':<{test_f1_width}} {'Test Acc':<{test_acc_width}} {'Test Loss':<{test_loss_width}} {'Val Acc':<{val_acc_width}}\n")
        f.write("-"*total_width + "\n")
        
        for rank, result in enumerate(all_results_sorted, 1):
            sensors_str = ", ".join(result["sensors"])
            ablated_str = ", ".join(result.get("ablated_sensors", []))
            f.write(f"{rank:<{rank_width}} {sensors_str:<{sensors_width}} {ablated_str:<{ablated_width}} {result['test_f1_macro']:.4f}      "
                    f"{result['test_accuracy']:.4f}     {result['test_loss']:.4f}       {result['val_accuracy']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("BEST COMBINATION\n")
        f.write("="*70 + "\n")
        f.write(f"  Sensors: {', '.join(best['sensors'])}\n")
        if best.get('ablated_sensors'):
            f.write(f"  Ablated: {', '.join(best['ablated_sensors'])}\n")
        f.write(f"  Test F1 (macro): {best['test_f1_macro']:.4f}\n")
        f.write(f"  Test Accuracy:   {best['test_accuracy']:.4f}\n")
        f.write(f"  Val F1 (macro):  {best['val_f1_macro']:.4f}\n")
        f.write(f"  Test Loss: {best['test_loss']:.4f}\n")
        f.write(f"  Val Loss:  {best['val_loss']:.4f}\n")
    
    print(f"\n✓ Summary report saved to: {report_file}")

    # ---- Confusion matrix for best combination ----
    if "y_pred" in best and "y_true" in best:
        y_true_cm = np.array(best["y_true"])
        y_pred_cm = np.array(best["y_pred"])
        labels = sorted(np.unique(np.concatenate([y_true_cm, y_pred_cm])).tolist())
        # MHEALTH activities are 0-indexed internally; display as 1-indexed
        display_labels = [str(l + 1) for l in labels]
        cm = confusion_matrix(y_true_cm, y_pred_cm, labels=labels)

        # Print to console
        print("\nConfusion matrix (rows=true, cols=predicted, labels=activity 1-indexed):")
        header = "     " + " ".join(f"{lbl:>4}" for lbl in display_labels)
        print(header)
        for i, row_label in enumerate(display_labels):
            row_str = " ".join(f"{cm[i, j]:>4}" for j in range(len(labels)))
            print(f"{row_label:>4} {row_str}")

        # Save plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(display_labels, fontsize=9)
        ax.set_yticklabels(display_labels, fontsize=9)
        ax.set_xlabel("Predicted activity", fontsize=11)
        ax.set_ylabel("True activity", fontsize=11)
        ax.set_title(
            f"Confusion matrix — {ABLATION_REPORT_TAG}\n"
            f"Test F1={best['test_f1_macro']:.4f}  "
            f"sensors={len(best['sensors'])}",
            fontsize=11,
        )
        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7)
        plt.tight_layout()
        cm_file = ABLATION_REPORT_DIR / f"confusion_matrix_{run_timestamp}.png"
        fig.savefig(cm_file, dpi=150)
        plt.close(fig)
        print(f"✓ Confusion matrix saved to: {cm_file}")

    # Save CSV for easy plotting (box plots: x=num_sensors, y=accuracy/f1)
    csv_file = ABLATION_REPORT_DIR / f"tremor_ablation_k{ablation_k_list}_results_{run_timestamp}.csv"
    csv_fields = ["timestamp", "seed", "corrupt_test_variant", "k", "sensors", "num_sensors",
                  "test_accuracy", "test_f1_macro", "test_loss",
                  "val_accuracy", "val_f1_macro", "val_loss"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                "timestamp":            run_timestamp,
                "seed":                 SEED,
                "corrupt_test_variant": corrupt_test_variant,
                "k":                    result["num_sensors"],
                "sensors":              "|".join(result["sensors"]),
                "num_sensors":          result["num_sensors"],
                "test_accuracy":        result["test_accuracy"],
                "test_f1_macro":        result["test_f1_macro"],
                "test_loss":            result["test_loss"],
                "val_accuracy":         result["val_accuracy"],
                "val_f1_macro":         result["val_f1_macro"],
                "val_loss":             result["val_loss"],
            })
    print(f"✓ CSV results saved to:   {csv_file}")

    # ---- Skip summary (always shown at the very end) ----
    total_intra   = sum(intra_skipped_per_sensor.values())
    total_cross   = sum(cross_skipped_per_sensor.values())
    total_skipped = total_intra + total_cross
    print(f"\n{'='*70}")
    if total_skipped == 0:
        print("✓ No windows skipped — all variants and sensors fully aligned.")
    else:
        print("SKIPPED WINDOWS SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Sensor':<15}  {'Intra-variant':>14}  {'Cross-sensor':>13}  {'Total':>7}")
        print(f"  {'-'*53}")
        for s in ALL_SENSORS:
            intra = intra_skipped_per_sensor.get(s, 0)
            cross = cross_skipped_per_sensor.get(s, 0)
            tot   = intra + cross
            marker = " ←" if tot > 0 else ""
            print(f"  {s:<15}  {intra:>14}  {cross:>13}  {tot:>7}{marker}")
        print(f"  {'-'*53}")
        print(f"  {'TOTAL':<15}  {total_intra:>14}  {total_cross:>13}  {total_skipped:>7}")
        print(f"  Intra-variant: windows dropped to align variant window counts within a sensor's mix")
        print(f"  Cross-sensor:  windows dropped to align the final test set size across all sensors")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tremor sensor ablation study")
    parser.add_argument("--ablation_k", type=int, default=None,
                        help="Override ABLATION_K with a single integer (e.g. --ablation_k 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override SEED for reproducibility (e.g. --seed 42)")
    parser.add_argument("--corrupt_test_variant", type=str, default=None,
                        help="Override corrupt_test_variant (e.g. --corrupt_test_variant s4_w4_fs50_corrupt_dropout)")
    args = parser.parse_args()
    if args.ablation_k is not None:
        ABLATION_K = [args.ablation_k]
    if args.seed is not None:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    if args.corrupt_test_variant is not None:
        corrupt_test_variant = args.corrupt_test_variant
    main()
