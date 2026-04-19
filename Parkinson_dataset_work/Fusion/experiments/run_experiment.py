"""
run_experiment.py

Single-run fusion experiment with configurable sensor set and per-sensor
sampling frequencies. Identical model architecture and hyperparameters to
fusion_concat_ablation_study_tremor.py.

Features:
  - User-configurable sensor list with per-sensor sampling frequency (fs)
  - Explicit list of data variant folders (TREMOR_VARIANTS) — same pattern
    as tremor_variants in fusion_concat_ablation_study_tremor.py
  - Optional EXPERIMENT_TAG to organise results by data type
    (e.g. "clean", "mixed", "pd_only", "pd_mixed")
  - Per-config run log  →  run_logs/<tag>/{config_id}_runs.jsonl
  - Global best-accuracy tracker  →  best_accuracy_tracker_<tag>.json
  - Model saved to models/<tag>/{config_id}_best.pth only when a new best

Usage:
  1. Edit SENSORS_CONFIG  (sensor name → sampling frequency)
  2. Edit TREMOR_VARIANTS (list of folder names to load embeddings from)
  3. Set EXPERIMENT_TAG to describe this data configuration
  4. Set SEED as desired
  5. Run the script
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════
# USER CONFIGURATION  ← Edit this section
# ═══════════════════════════════════════════════════════════════

# Map each sensor to its sampling frequency (Hz).
# Available sensor names:
#   Acc_LL, Acc_LR, Acc_UL, Acc_UR, Acc_head
#   Gyro_LL, Gyro_LR, Gyro_UL, Gyro_UR, Gyro_head
#   Mag_LL, Mag_LR, Mag_UL, Mag_UR, Mag_head
# The fs value must match the folder names listed in TREMOR_VARIANTS below.
SENSORS_CONFIG: Dict[str, int] = {
    "Acc_LL":   50,
    "Acc_LR":   50,
    "Mag_LL":   50,
    "Mag_LR":   50,
    "Mag_head": 50,
}

# Exact folder names inside  .../Data/Tremor_datagenerator_files/
# that embeddings should be loaded from and combined (used as augmentations).
#
# Examples:
#   Clean only:
#     ["s2_w2_fs50_tremor_clean"]
#   Mixed (clean + simulated tremor):
#     ["s2_w2_fs50_tremor_clean", "s2_w2_fs50_tremor_mild_mod", "s2_w2_fs50_tremor_mod_severe"]
#   Real PD patients only:
#     ["s2_w2_fs50_tremor_parkinson"]
#   Mixed PD + simulated:
#     ["s2_w2_fs50_tremor_parkinson", "s2_w2_fs50_tremor_clean",
#      "s2_w2_fs50_tremor_mild_mod", "s2_w2_fs50_tremor_mod_severe"]
TREMOR_VARIANTS: List[str] = [
    #"s2_w2_fs50_tremor_clean",
    "s2_w2_fs50_tremor_mild_mod",
    "s2_w2_fs50_tremor_mod_severe",
]

# Short tag that labels this data configuration.
# Used in output folder names and the best-tracker filename so that
# results from different data setups stay separated.
# Examples: "clean", "mixed", "pd_only", "pd_mixed"
EXPERIMENT_TAG: str = "mild_severe"

SEED: int = 41


# ═══════════════════════════════════════════════════════════════
# FIXED CONFIGURATION  (must stay identical to ablation script)
# ═══════════════════════════════════════════════════════════════

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory layout:  .../Parkinson_dataset_work/Fusion/experiments/  ← SCRIPT_DIR
SCRIPT_DIR   = Path(__file__).parent.resolve()          # .../experiments/
FUSION_DIR   = SCRIPT_DIR.parent                        # .../Fusion/
PROJECT_ROOT = FUSION_DIR.parent                        # .../Parkinson_dataset_work/

base_data_dir       = PROJECT_ROOT / "Data" / "Tremor_datagenerator_files"
embeddings_folder_name = "Activity_ExtractedFeatures"

# Subject-based split (must match extraction policy)
HOLDOUT_SUBJECTS: List[int] = [1, 14, 19]
VAL_SUBJECTS:     List[int] = [4, 9, 16]
TEST_SUBJECTS:    List[int] = [2, 7, 11]
TRAIN_SUBJECTS_OVERRIDE: Optional[List[int]] = None

# Training hyperparameters
batch_size = 128
epochs     = 200
lr         = 1e-3
patience   = 30
min_delta  = 1e-4

# Fusion model configuration
USE_GATING    = False   # True = gated fusion, False = concat baseline
gate_type     = "sigmoid"
gate_hidden   = 64
gate_dropout  = 0.1
use_layernorm = True
alpha_floor   = 0.05

# Classifier head
head_hidden_dims = (64, 64)
head_dropout     = 0.5

# Output directories — organised under a sub-folder named after EXPERIMENT_TAG
# so that different data configurations never overwrite each other.
def _safe_tag(tag: str) -> str:
    """Make a tag safe for use as a directory / filename component."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in tag.strip()) or "default"

_tag_dir = _safe_tag(EXPERIMENT_TAG)

MODELS_DIR        = SCRIPT_DIR / "models"    / _tag_dir
RUN_LOGS_DIR      = SCRIPT_DIR / "run_logs"  / _tag_dir
BEST_TRACKER_FILE = SCRIPT_DIR / f"best_accuracy_tracker_{_tag_dir}.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CONFIG ID  (human-readable, unique per sensor+fs combination)
# ═══════════════════════════════════════════════════════════════

# Abbreviation tables
_TYPE_ABBREV = {"Acc": "A", "Gyro": "G", "Mag": "M"}
_POS_ABBREV  = {"LL": "LL", "LR": "LR", "UL": "UL", "UR": "UR", "head": "Hd"}


def _abbrev_sensor(sensor: str) -> str:
    """'Acc_LL' → 'ALL',  'Gyro_head' → 'GHd', etc."""
    parts = sensor.split("_", 1)
    t = _TYPE_ABBREV.get(parts[0], parts[0][:3])
    p = _POS_ABBREV.get(parts[1], parts[1]) if len(parts) > 1 else ""
    return t + p


def make_config_id(sensors_config: Dict[str, int]) -> str:
    """
    Build a short, unique config identifier from the sensor+fs mapping
    and the active EXPERIMENT_TAG.

    Examples
    --------
    All same fs, tag='mixed' → 'mixed__ALL-ALR-GHd_fs50'
    Mixed fs,    tag='clean' → 'clean__ALL50-ALR50-GHd30'
    """
    sorted_items = sorted(sensors_config.items())  # deterministic order
    fs_values = [fs for _, fs in sorted_items]
    abbrevs   = [_abbrev_sensor(s) for s, _ in sorted_items]

    if len(set(fs_values)) == 1:
        sensor_part = "-".join(abbrevs) + f"_fs{fs_values[0]}"
    else:
        sensor_part = "-".join(f"{a}{fs}" for a, fs in zip(abbrevs, fs_values))

    return f"{_tag_dir}__{sensor_part}"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_embeddings_for_sensor(sensor_name: str, fs: int) -> Dict[str, np.ndarray]:
    """
    Load and combine embeddings from all folders listed in TREMOR_VARIANTS
    for a sensor.  Applies strict subject-based re-splitting to avoid
    leakage from the extraction stage.

    Returns
    -------
    dict with keys: Z_train, y_train, Z_val, y_val, Z_test, y_test
    """
    variants = TREMOR_VARIANTS
    if not variants:
        raise ValueError("TREMOR_VARIANTS is empty — add at least one folder name.")

    all_embeddings, all_activities, all_subjects = [], [], []

    for variant_name in variants:
        npz_path = base_data_dir / variant_name / embeddings_folder_name / f"{sensor_name}_embeddings.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {npz_path}\n"
                f"Make sure CNNs have been trained and embeddings extracted for "
                f"{sensor_name} (variant: {variant_name})."
            )

        data = np.load(npz_path)

        # Concatenate all rows from this variant, then re-split by subject.
        emb  = np.concatenate([data["train_embeddings"], data["val_embeddings"],  data["test_embeddings"]], axis=0).astype(np.float32)
        acts = np.concatenate([data["train_activities"], data["val_activities"],  data["test_activities"]], axis=0).astype(np.int64)
        subj = np.concatenate([data["train_subjects"],   data["val_subjects"],    data["test_subjects"]],   axis=0).astype(np.int64)

        all_embeddings.append(emb)
        all_activities.append(acts)
        all_subjects.append(subj)

    Z_all    = np.concatenate(all_embeddings, axis=0)
    y_all    = np.concatenate(all_activities, axis=0)
    subj_all = np.concatenate(all_subjects, axis=0)

    observed_subjects = sorted(np.unique(subj_all).astype(int).tolist())
    forbidden = set(HOLDOUT_SUBJECTS) | set(VAL_SUBJECTS) | set(TEST_SUBJECTS)

    if TRAIN_SUBJECTS_OVERRIDE is None:
        train_subjects_cfg = [s for s in observed_subjects if s not in forbidden]
    else:
        train_subjects_cfg = sorted(TRAIN_SUBJECTS_OVERRIDE)

    if not train_subjects_cfg:
        raise ValueError("No train subjects available after applying HOLDOUT/VAL/TEST.")

    # Disjoint-set safety checks
    if set(train_subjects_cfg) & set(VAL_SUBJECTS):
        raise ValueError("TRAIN and VAL subjects overlap.")
    if set(train_subjects_cfg) & set(TEST_SUBJECTS):
        raise ValueError("TRAIN and TEST subjects overlap.")
    if set(VAL_SUBJECTS) & set(TEST_SUBJECTS):
        raise ValueError("VAL and TEST subjects overlap.")
    if set(HOLDOUT_SUBJECTS) & (set(train_subjects_cfg) | set(VAL_SUBJECTS) | set(TEST_SUBJECTS)):
        raise ValueError("HOLDOUT subjects overlap with TRAIN/VAL/TEST.")

    train_mask = np.isin(subj_all, train_subjects_cfg)
    val_mask   = np.isin(subj_all, VAL_SUBJECTS)
    test_mask  = np.isin(subj_all, TEST_SUBJECTS)

    Z_train, y_train = Z_all[train_mask], y_all[train_mask]
    Z_val,   y_val   = Z_all[val_mask],   y_all[val_mask]
    Z_test,  y_test  = Z_all[test_mask],  y_all[test_mask]

    actual_train = sorted(np.unique(subj_all[train_mask]).astype(int).tolist())
    actual_val   = sorted(np.unique(subj_all[val_mask]).astype(int).tolist())
    actual_test  = sorted(np.unique(subj_all[test_mask]).astype(int).tolist())

    if actual_train != sorted(train_subjects_cfg):
        raise ValueError(f"Train subjects mismatch: {actual_train} != {sorted(train_subjects_cfg)}")
    if actual_val != sorted(VAL_SUBJECTS):
        raise ValueError(f"Val subjects mismatch: {actual_val} != {sorted(VAL_SUBJECTS)}")
    if actual_test != sorted(TEST_SUBJECTS):
        raise ValueError(f"Test subjects mismatch: {actual_test} != {sorted(TEST_SUBJECTS)}")

    print(f"  {sensor_name}: subject split applied across {len(variants)} variant(s) ✓")
    print(f"    Train: {len(Z_train)} samples | subjects {actual_train}")
    print(f"    Val:   {len(Z_val)} samples   | subjects {actual_val}")
    print(f"    Test:  {len(Z_test)} samples  | subjects {actual_test}")

    return {
        "Z_train": Z_train, "y_train": y_train,
        "Z_val":   Z_val,   "y_val":   y_val,
        "Z_test":  Z_test,  "y_test":  y_test,
    }


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class MultiSensorDataset(Dataset):
    def __init__(self, Z_list: List[torch.Tensor], y: torch.Tensor):
        self.Z_list = Z_list
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple([Z[idx] for Z in self.Z_list] + [self.y[idx]])


# ═══════════════════════════════════════════════════════════════
# MODEL  (identical to fusion_concat_ablation_study_tremor.py)
# ═══════════════════════════════════════════════════════════════

class PaperHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int,
                 hidden_dims=(256, 256), dropout_p: float = 0.3):
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
                nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
        else:
            self.net = nn.Linear(embed_dim, 1)

    def forward(self, z):
        return self.net(z)


class GatedConcatFusionModel(nn.Module):
    def __init__(self, embed_dim, num_sensors, num_classes,
                 head_hidden_dims=(128, 128), head_dropout=0.3,
                 gate_type="sigmoid", gate_hidden=0, gate_dropout=0.0,
                 use_layernorm=True, alpha_floor=0.0):
        super().__init__()
        self.num_sensors = num_sensors
        self.embed_dim   = embed_dim
        self.gate_type   = gate_type.lower()
        self.alpha_floor = alpha_floor

        assert self.gate_type in ("sigmoid", "softmax")

        self.norms = (
            nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_sensors)])
            if use_layernorm else None
        )
        self.gate_in_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else None
        self.gates = nn.ModuleList([SensorGate(embed_dim, hidden=gate_hidden)
                                    for _ in range(num_sensors)])
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

        scores = torch.cat([gate(z) for gate, z in zip(self.gates, Z_proc)], dim=1)

        if self.gate_type == "softmax":
            alphas = torch.softmax(scores, dim=1)
        else:
            alphas = torch.sigmoid(scores)

        if self.alpha_floor > 0:
            alphas = self.alpha_floor + (1.0 - self.alpha_floor) * alphas

        z_cat  = torch.cat([alphas[:, i:i+1] * Z_proc[i] for i in range(self.num_sensors)], dim=1)
        logits = self.head(z_cat)
        return logits, alphas


class ConcatFusionModel(nn.Module):
    def __init__(self, embed_dim, num_sensors, num_classes,
                 head_hidden_dims=(256, 256, 128), head_dropout=0.3):
        super().__init__()
        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes,
                              hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        z      = torch.cat(Z_list, dim=1)
        logits = self.head(z)
        return logits, None


# ═══════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════

def load_best_tracker() -> Dict:
    """Load best_accuracy_tracker.json, return empty dict if missing."""
    if BEST_TRACKER_FILE.exists():
        with open(BEST_TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}


def save_best_tracker(tracker: Dict) -> None:
    with open(BEST_TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def append_run_log(config_id: str, entry: Dict) -> None:
    """Append one JSON line to run_logs/{config_id}_runs.jsonl."""
    log_path = RUN_LOGS_DIR / f"{config_id}_runs.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_experiment(sensors_config: Dict[str, int], seed: int) -> None:
    sensors    = sorted(sensors_config.keys())
    config_id  = make_config_id(sensors_config)
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_sensors = len(sensors)

    print("=" * 70)
    print(f"Experiment: {config_id}")
    print(f"Tag: {EXPERIMENT_TAG}")
    print(f"Sensors ({num_sensors}): {sensors}")
    print(f"Seed: {seed}  |  Device: {device}")
    print(f"\nData variants ({len(TREMOR_VARIANTS)}):")
    for v in TREMOR_VARIANTS:
        print(f"  - {v}")
    print("=" * 70)

    # ── 1. Load embeddings ─────────────────────────────────────
    print("\n[1/5] Loading embeddings...")
    embeddings: Dict[str, Dict] = {}
    for sensor in sensors:
        fs = sensors_config[sensor]
        embeddings[sensor] = load_embeddings_for_sensor(sensor, fs)

    # Verify label alignment across sensors
    ref_sensor = sensors[0]
    for split in ("train", "val", "test"):
        ref_y = embeddings[ref_sensor][f"y_{split}"]
        for s in sensors[1:]:
            if not np.array_equal(embeddings[s][f"y_{split}"], ref_y):
                raise ValueError(f"Label mismatch between {ref_sensor} and {s} on {split} split.")
    print("  ✓ Labels aligned across all sensors.")

    embed_dim   = embeddings[ref_sensor]["Z_train"].shape[1]
    num_classes = len(np.unique(np.concatenate(
        [embeddings[ref_sensor]["y_train"],
         embeddings[ref_sensor]["y_val"],
         embeddings[ref_sensor]["y_test"]]
    )))
    print(f"  embed_dim={embed_dim}  |  num_classes={num_classes}")

    # ── 2. Build datasets / loaders ───────────────────────────
    print("\n[2/5] Creating datasets...")

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        Z_list = [torch.from_numpy(embeddings[s][f"Z_{split}"]) for s in sensors]
        y      = torch.from_numpy(embeddings[ref_sensor][f"y_{split}"])
        return DataLoader(MultiSensorDataset(Z_list, y),
                          batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val",   shuffle=False)
    test_loader  = make_loader("test",  shuffle=False)

    n_train = len(embeddings[ref_sensor]["y_train"])
    n_val   = len(embeddings[ref_sensor]["y_val"])
    n_test  = len(embeddings[ref_sensor]["y_test"])
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")

    # ── 3. Build model ────────────────────────────────────────
    print("\n[3/5] Building model...")

    if USE_GATING:
        model = GatedConcatFusionModel(
            embed_dim=embed_dim, num_sensors=num_sensors, num_classes=num_classes,
            head_hidden_dims=head_hidden_dims, head_dropout=head_dropout,
            gate_type=gate_type, gate_hidden=gate_hidden, gate_dropout=gate_dropout,
            use_layernorm=use_layernorm, alpha_floor=alpha_floor,
        ).to(device)
        print(f"  Mode: GATED (gate_type={gate_type}, gate_hidden={gate_hidden})")
    else:
        model = ConcatFusionModel(
            embed_dim=embed_dim, num_sensors=num_sensors, num_classes=num_classes,
            head_hidden_dims=head_hidden_dims, head_dropout=head_dropout,
        ).to(device)
        print("  Mode: CONCAT BASELINE")

    print(f"\n{model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    # ── 4. Training loop ──────────────────────────────────────
    print("\n[4/5] Training...")

    best_val_f1_macro = -float("inf")
    best_state        = None
    no_improve        = 0

    def unpack(batch):
        *Z, y = batch
        return [z.to(device) for z in Z], y.to(device)

    for epoch in range(epochs):
        # Train
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for batch in train_loader:
            Z, y_b = unpack(batch)
            optimizer.zero_grad()
            logits, _ = model(Z)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * y_b.size(0)
            tr_correct  += (logits.argmax(1) == y_b).sum().item()
            tr_total    += y_b.size(0)

        tr_loss = tr_loss_sum / tr_total
        tr_acc  = tr_correct  / tr_total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        val_y_all, val_pred_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                Z, y_b = unpack(batch)
                logits, _ = model(Z)
                loss = criterion(logits, y_b)
                val_loss_sum += loss.item() * y_b.size(0)
                preds = logits.argmax(1)
                val_correct  += (preds == y_b).sum().item()
                val_total    += y_b.size(0)
                val_y_all.extend(y_b.cpu().numpy())
                val_pred_all.extend(preds.cpu().numpy())

        val_loss     = val_loss_sum / val_total
        val_acc      = val_correct  / val_total
        val_f1_macro = f1_score(val_y_all, val_pred_all, average="macro")

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"Val F1(macro) {val_f1_macro:.4f}"
        )

        # Early stopping based on val macro F1
        if val_f1_macro > best_val_f1_macro + min_delta:
            best_val_f1_macro = val_f1_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. "
                      f"Best val F1(macro): {best_val_f1_macro:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ── 5. Test evaluation ────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")

    model.eval()
    y_pred   = []
    test_loss_sum, test_total = 0.0, 0

    with torch.no_grad():
        for batch in test_loader:
            Z, y_b = unpack(batch)
            logits, _ = model(Z)
            loss = criterion(logits, y_b)
            test_loss_sum += loss.item() * y_b.size(0)
            test_total    += y_b.size(0)
            y_pred.extend(logits.argmax(1).cpu().numpy())

    y_test_np        = embeddings[ref_sensor]["y_test"]
    test_loss        = test_loss_sum / test_total
    test_acc         = accuracy_score(y_test_np, y_pred)
    test_f1_macro    = f1_score(y_test_np, y_pred, average="macro")
    test_f1_weighted = f1_score(y_test_np, y_pred, average="weighted")

    print(f"\nTest Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test F1 (macro):  {test_f1_macro:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test_np, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=np.arange(num_classes)).plot(
        ax=axes[0], cmap="Blues", values_format="d"
    )
    axes[0].set_title("Absolute Counts", fontsize=14, fontweight="bold")
    for text in axes[0].texts:
        text.set_fontsize(8)

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                           display_labels=np.arange(num_classes)).plot(
        ax=axes[1], cmap="Blues", values_format=".1%"
    )
    axes[1].set_title("Recall per True Label (% of samples)", fontsize=14, fontweight="bold")
    for text in axes[1].texts:
        text.set_fontsize(8)

    fig.suptitle(
        f"Confusion Matrix — {config_id}\n"
        f"Test Acc: {test_acc:.4f}  |  Test F1(macro): {test_f1_macro:.4f}  |  Seed: {seed}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    cm_path = RUN_LOGS_DIR / f"{config_id}_seed{seed}_cm.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {cm_path}")

    # ── Per-config run log ────────────────────────────────────
    run_entry = {
        "timestamp":        timestamp,
        "seed":             seed,
        "experiment_tag":   EXPERIMENT_TAG,
        "data_variants":    TREMOR_VARIANTS,
        "config_id":        config_id,
        "sensors":          sensors,
        "sensors_fs":       {s: sensors_config[s] for s in sensors},
        "test_accuracy":    round(float(test_acc), 6),
        "test_f1_macro":    round(float(test_f1_macro), 6),
        "test_f1_weighted": round(float(test_f1_weighted), 6),
        "test_loss":        round(float(test_loss), 6),
        "val_f1_macro":     round(float(best_val_f1_macro), 6),
        "val_loss":         round(float(val_loss), 6),
        "val_accuracy":     round(float(val_acc), 6),
    }
    append_run_log(config_id, run_entry)
    print(f"  ✓ Run logged → run_logs/{_tag_dir}/{config_id}_runs.jsonl")

    # ── Compare with best tracker & conditionally save model ──
    tracker = load_best_tracker()
    prev_best = tracker.get(config_id, {})
    prev_acc  = prev_best.get("best_test_accuracy", -1.0)

    model_filename = f"{config_id}_best.pth"
    model_path     = MODELS_DIR / model_filename

    if test_acc > prev_acc:
        # Save model
        torch.save(
            {
                "model_state_dict": best_state,
                "config_id":        config_id,
                "sensors":          sensors,
                "sensors_fs":       {s: sensors_config[s] for s in sensors},
                "embed_dim":        embed_dim,
                "num_classes":      num_classes,
                "num_sensors":      num_sensors,
                "head_hidden_dims": head_hidden_dims,
                "head_dropout":     head_dropout,
                "use_gating":       USE_GATING,
                "seed":             seed,
                "test_accuracy":    round(float(test_acc), 6),
                "test_f1_macro":    round(float(test_f1_macro), 6),
                "timestamp":        timestamp,
            },
            model_path,
        )
        print(f"\n  ★ NEW BEST for '{config_id}'!")
        print(f"    Previous best: {prev_acc:.4f}  →  New best: {test_acc:.4f}")
        print(f"  ✓ Model saved  → models/{model_filename}")

        # Update tracker
        tracker[config_id] = {
            "best_test_accuracy":    round(float(test_acc), 6),
            "best_test_f1_macro":    round(float(test_f1_macro), 6),
            "best_test_f1_weighted": round(float(test_f1_weighted), 6),
            "seed":                  seed,
            "timestamp":             timestamp,
            "model_file":            model_filename,
            "sensors":               sensors,
            "sensors_fs":            {s: sensors_config[s] for s in sensors},
        }
        save_best_tracker(tracker)
        print(f"  ✓ Best tracker updated → best_accuracy_tracker.json")
    else:
        print(f"\n  — No improvement for '{config_id}'.")
        print(f"    Current run: {test_acc:.4f}  |  Existing best: {prev_acc:.4f}")
        print(f"    Model NOT saved (existing model at models/{model_filename} is better).")

    print("\n" + "=" * 70)
    print("Experiment complete.")
    print(f"  Config ID : {config_id}")
    print(f"  Test Acc  : {test_acc:.4f}")
    print(f"  F1 (macro): {test_f1_macro:.4f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_experiment(sensors_config=SENSORS_CONFIG, seed=SEED)
