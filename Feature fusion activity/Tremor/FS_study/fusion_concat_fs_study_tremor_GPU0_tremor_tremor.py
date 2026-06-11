"""
fusion_concat_fs_study_tremor_GPU0.py

Tremor FS Combination Study — GPU 0 — Scenario: Tremor→Tremor

Tests all sampling-frequency combinations for a fixed sensor list.
Trains on mixed (all 3 tremor) embeddings, tests on mixed embeddings.

GPU isolation: GPU_ID = 0  →  device = cuda:0
Log directory: tremor_logs_gpu0/
Log file:      tremor_fusion_fs_combo_study_tremor_tremor.jsonl

Run alongside GPU1 script without any crossover:
  python fusion_concat_fs_study_tremor_GPU0.py             # Clean→Clean on cuda:0
  python fusion_concat_fs_study_tremor_GPU1.py             # Clean→Tremor on cuda:1
  python fusion_concat_fs_study_tremor_GPU0_tremor_tremor.py  # Tremor→Tremor on cuda:0

Start:
  cd '/home/linacr/projects/Code/Feature fusion activity/Tremor/FS_study'
  source /home/linacr/projects/Code/.venv/bin/activate
  python fusion_concat_fs_study_tremor_GPU0_tremor_tremor.py
"""

import json
import os
import random
import hashlib
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# -------------------------------
# GPU isolation  ← only line that differs between GPU0 and GPU1
# -------------------------------
GPU_ID = 0
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Reproducibility
# -------------------------------
SEED = [35,36,37,38,39]   # single int or list of ints
_init_seed = SEED[0] if isinstance(SEED, list) else SEED
random.seed(_init_seed)
np.random.seed(_init_seed)
torch.manual_seed(_init_seed)
torch.cuda.manual_seed_all(_init_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# -------------------------------
# Paths
# -------------------------------
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent
BASE_DATA_DIR = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"

# Window/stride prefix — must match the datagenerator files used
WINDOW_PREFIX = "s4_w4"   # "s4_w4" = 4 s window / 4 s stride

# Folder where extracted embeddings are stored for tremor activity fusion
# "ExtractedFeatures_clean"  → CNNs trained on clean data only
# "ExtractedFeatures_mixed"  → CNNs trained on mixed (tremor-augmented) data
EMBEDDINGS_FOLDER_NAME = "ExtractedFeatures_mixed"

# -------------------------------
# Sensors + FS study setup
# -------------------------------
FIXED_SENSORS = ["Acc_ankle", "Mag_ankle", "Mag_arm", "Acc_chest"]
FS_CANDIDATES = [10, 20, 30, 40, 50]
MAX_COMBOS = None
SKIP_EXISTING = True

# -----------------------------------------------------------------------
# Scenario configuration — Clean→Clean
# -----------------------------------------------------------------------
# TRAIN_TREMOR_TYPES  — variants concatenated for train + val
# TEST_TREMOR_TYPES   — variants concatenated for the test set
#
# Scenario examples:
#   Clean→Clean  : TRAIN=["tremor_clean"],          TEST=["tremor_clean"],          EMBEDDINGS_FOLDER_NAME="ExtractedFeatures_clean"
#   Clean→Tremor : TRAIN=["tremor_clean"],          TEST=[all 3],                   EMBEDDINGS_FOLDER_NAME="ExtractedFeatures_clean"
#   Tremor→Tremor: TRAIN=[all 3],                   TEST=[all 3],                   EMBEDDINGS_FOLDER_NAME="ExtractedFeatures_mixed"
# -----------------------------------------------------------------------
TRAIN_TREMOR_TYPES = ["tremor_clean", "tremor_mild_mod", "tremor_mod_severe"]   # all 3
TEST_TREMOR_TYPES  = ["tremor_clean", "tremor_mild_mod", "tremor_mod_severe"]   # all 3

# Subject-based split (same as ablation script)
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]

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
USE_GATING = False

gate_type = "sigmoid"
gate_hidden = 64
gate_dropout = 0.1
use_layernorm = True
alpha_floor = 0.05

head_hidden_dims = (64, 64)
head_dropout = 0.4

# -------------------------------
# Logging — dedicated directory for Tremor→Tremor runs
# -------------------------------
LOG_DIR = SCRIPT_DIR / "tremor_logs_tremor_tremor"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "tremor_fusion_fs_combo_study_tremor_tremor.jsonl"

# -------------------------------
# In-memory cache
# -------------------------------
_EMBEDDING_CACHE: Dict[tuple, Dict[str, np.ndarray]] = {}


# ═══════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════

def generate_fs_combo_hash(sensor_fs: List[int]) -> str:
    fs_str = "_".join(str(fs) for fs in sensor_fs)
    return hashlib.md5(fs_str.encode("utf-8")).hexdigest()[:8]


def get_variant_names_for_fs(fs: int, tremor_types: list) -> List[str]:
    return [f"{WINDOW_PREFIX}_fs{fs}_{t}" for t in tremor_types]


def get_train_variants_for_fs(fs: int) -> List[str]:
    return get_variant_names_for_fs(fs, TRAIN_TREMOR_TYPES)


def get_test_variants_for_fs(fs: int) -> List[str]:
    return get_variant_names_for_fs(fs, TEST_TREMOR_TYPES)


def load_existing_combinations(log_file: Path, experiment_id: str) -> set:
    if not log_file.exists():
        return set()

    combo_hashes = set()
    with open(log_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("experiment_id") == experiment_id:
                combo_hash = entry.get("fs_combo_hash")
                if combo_hash:
                    combo_hashes.add(combo_hash)

    return combo_hashes


def load_combined_tremor_embeddings(
    sensor_name: str,
    fs: int,
    tremor_types: List[str],
    subjects_for_test: List[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load + combine embeddings across the given tremor_types variants for one sensor/fs,
    then re-split by subject IDs to avoid leakage.

    If subjects_for_test is provided, only the test-subject rows are returned
    (used when loading a separate test-only split from TEST_TREMOR_TYPES).
    """
    cache_key = (sensor_name, fs, EMBEDDINGS_FOLDER_NAME, tuple(tremor_types), tuple(subjects_for_test or []))
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    all_embeddings = []
    all_activities = []
    all_subjects = []

    variant_names = get_variant_names_for_fs(fs, tremor_types)

    for variant_name in variant_names:
        variant_dir = BASE_DATA_DIR / variant_name
        npz_path = variant_dir / EMBEDDINGS_FOLDER_NAME / f"{sensor_name}_embeddings.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {npz_path}\n"
                f"Expected variants for fs{fs}: {variant_names}"
            )

        data = np.load(npz_path)

        required_keys = [
            "train_embeddings", "val_embeddings", "test_embeddings",
            "train_activities", "val_activities", "test_activities",
            "train_subjects", "val_subjects", "test_subjects",
        ]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(f"Missing keys in {npz_path}: {missing}")

        variant_embeddings = np.concatenate([
            data["train_embeddings"],
            data["val_embeddings"],
            data["test_embeddings"],
        ], axis=0).astype(np.float32)

        variant_activities = np.concatenate([
            data["train_activities"],
            data["val_activities"],
            data["test_activities"],
        ], axis=0).astype(np.int64)

        variant_subjects = np.concatenate([
            data["train_subjects"],
            data["val_subjects"],
            data["test_subjects"],
        ], axis=0).astype(np.int64)

        all_embeddings.append(variant_embeddings)
        all_activities.append(variant_activities)
        all_subjects.append(variant_subjects)

    # Concatenate across tremor variants
    Z_all = np.concatenate(all_embeddings, axis=0)
    y_all = np.concatenate(all_activities, axis=0)
    subj_all = np.concatenate(all_subjects, axis=0)

    # Subject-based split
    train_mask = ~np.isin(subj_all, TEST_SUBJECTS + VAL_SUBJECTS)
    val_mask = np.isin(subj_all, VAL_SUBJECTS)
    test_mask = np.isin(subj_all, TEST_SUBJECTS)

    Z_train = Z_all[train_mask]
    y_train = y_all[train_mask]
    Z_val = Z_all[val_mask]
    y_val = y_all[val_mask]
    Z_test = Z_all[test_mask]
    y_test = y_all[test_mask]

    # Validate subject split
    train_subjects = np.unique(subj_all[train_mask]).tolist()
    val_subjects_found = np.unique(subj_all[val_mask]).tolist()
    test_subjects_found = np.unique(subj_all[test_mask]).tolist()

    expected_train = [1, 3, 4, 6, 8, 9]
    expected_val = sorted(VAL_SUBJECTS)
    expected_test = sorted(TEST_SUBJECTS)

    # Only validate train/val when not a test-only load
    if subjects_for_test is None:
        if sorted(train_subjects) != expected_train:
            raise ValueError(f"Train subjects mismatch for {sensor_name} fs{fs}: {train_subjects} != {expected_train}")
        if sorted(val_subjects_found) != expected_val:
            raise ValueError(f"Val subjects mismatch for {sensor_name} fs{fs}: {val_subjects_found} != {expected_val}")
    if sorted(test_subjects_found) != expected_test:
        raise ValueError(f"Test subjects mismatch for {sensor_name} fs{fs}: {test_subjects_found} != {expected_test}")

    if subjects_for_test is not None:
        # Test-only mode: return only the test-subject rows
        result = {
            "Z_test": Z_test,
            "y_test": y_test,
        }
    else:
        result = {
            "Z_train": Z_train,
            "y_train": y_train,
            "Z_val": Z_val,
            "y_val": y_val,
            "Z_test": Z_test,
            "y_test": y_test,
        }

    _EMBEDDING_CACHE[cache_key] = result
    return result


def verify_label_alignment(embeddings_dict: Dict[str, Dict[str, np.ndarray]], sensors: List[str]) -> None:
    for split in ["train", "val", "test"]:
        y_key = f"y_{split}"
        ref_sensor = sensors[0]
        ref_labels = embeddings_dict[ref_sensor][y_key]

        for sensor in sensors[1:]:
            if not np.array_equal(embeddings_dict[sensor][y_key], ref_labels):
                raise ValueError(f"Label mismatch in {split} split: {ref_sensor} vs {sensor}")


def evaluate(model, loader, criterion, use_gating_flag: bool):
    model.eval()

    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    gate_log = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            *Z, y = batch
            Z = [z.to(device) for z in Z]
            y = y.to(device)

            logits, alphas = model(Z)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if use_gating_flag and alphas is not None:
                for sensor_idx in range(alphas.shape[1]):
                    gate_log[sensor_idx].append(alphas[:, sensor_idx].cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1_macro, f1_weighted, gate_log


# ═══════════════════════════════════════════════════════════════
# DATASET + MODELS
# ═══════════════════════════════════════════════════════════════

class MultiSensorDataset(Dataset):
    def __init__(self, Z_list: List[torch.Tensor], y: torch.Tensor):
        self.Z_list = Z_list
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple([Z[idx] for Z in self.Z_list] + [self.y[idx]])


class PaperHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dims=(256, 256), dropout_p: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = hidden_dim
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class SensorGate(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
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
        alpha_floor: float = 0.0,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.gate_type = gate_type.lower()
        if self.gate_type not in ("sigmoid", "softmax"):
            raise ValueError(f"Unsupported gate_type: {gate_type}")

        self.alpha_floor = alpha_floor
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_sensors)]) if use_layernorm else None
        self.gate_in_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else None
        self.gates = nn.ModuleList([SensorGate(embed_dim, hidden=gate_hidden) for _ in range(num_sensors)])

        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes, hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        Z_proc = []
        for index, z in enumerate(Z_list):
            if self.norms is not None:
                z = self.norms[index](z)
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

        gated_blocks = [alphas[:, i:i + 1] * Z_proc[i] for i in range(self.num_sensors)]
        z_cat = torch.cat(gated_blocks, dim=1)
        logits = self.head(z_cat)
        return logits, alphas


class ConcatFusionModel(nn.Module):
    def __init__(self, embed_dim: int, num_sensors: int, num_classes: int,
                 head_hidden_dims=(256, 256, 128), head_dropout=0.3):
        super().__init__()
        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes, hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        z_cat = torch.cat(Z_list, dim=1)
        logits = self.head(z_cat)
        return logits, None


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════════════════════════════

def run_one_experiment(
    sensors: List[str],
    sensor_fs: List[int],
    experiment_id: str,
    run_timestamp: str,
) -> Dict:
    fs_map = {sensors[i]: sensor_fs[i] for i in range(len(sensors))}

    print("\nVariant mapping (train/val):")
    for sensor in sensors:
        variants = get_train_variants_for_fs(fs_map[sensor])
        print(f"  {sensor} (fs{fs_map[sensor]}): {variants}")
    print("Variant mapping (test):")
    for sensor in sensors:
        variants = get_test_variants_for_fs(fs_map[sensor])
        print(f"  {sensor} (fs{fs_map[sensor]}): {variants}")

    # 1a) Load train/val embeddings (from TRAIN_TREMOR_TYPES)
    trainval_dict = {}
    for sensor in sensors:
        fs = fs_map[sensor]
        trainval_dict[sensor] = load_combined_tremor_embeddings(sensor, fs, TRAIN_TREMOR_TYPES)

    # 1b) Load test embeddings (from TEST_TREMOR_TYPES — may differ from train variants)
    test_same_as_train = (sorted(TRAIN_TREMOR_TYPES) == sorted(TEST_TREMOR_TYPES))
    if test_same_as_train:
        test_dict = trainval_dict   # reuse — identical variants
    else:
        test_dict = {}
        for sensor in sensors:
            fs = fs_map[sensor]
            test_dict[sensor] = load_combined_tremor_embeddings(
                sensor, fs, TEST_TREMOR_TYPES, subjects_for_test=TEST_SUBJECTS
            )

    # 2) Verify alignment
    verify_label_alignment(trainval_dict, sensors)
    if not test_same_as_train:
        # Verify test labels align across sensors
        ref = test_dict[sensors[0]]["y_test"]
        for s in sensors[1:]:
            if not np.array_equal(test_dict[s]["y_test"], ref):
                raise ValueError(f"Test label mismatch between {sensors[0]} and {s}")

    embed_dim = trainval_dict[sensors[0]]["Z_train"].shape[1]
    num_classes = len(np.unique(trainval_dict[sensors[0]]["y_train"]))

    # 3) Build datasets — train/val from trainval_dict, test from test_dict
    def prepare_split_tensors(src_dict: dict, split: str):
        Z_key = f"Z_{split}"
        y_key = f"y_{split}"
        Z_list = [torch.from_numpy(src_dict[s][Z_key]) for s in sensors]
        y = torch.from_numpy(src_dict[sensors[0]][y_key])
        return Z_list, y

    Z_train, y_train = prepare_split_tensors(trainval_dict, "train")
    Z_val,   y_val   = prepare_split_tensors(trainval_dict, "val")
    Z_test,  y_test  = prepare_split_tensors(test_dict, "test")

    train_dataset = MultiSensorDataset(Z_train, y_train)
    val_dataset = MultiSensorDataset(Z_val, y_val)
    test_dataset = MultiSensorDataset(Z_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4) Build model
    num_sensors = len(sensors)
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
            alpha_floor=alpha_floor,
        ).to(device)
    else:
        model = ConcatFusionModel(
            embed_dim=embed_dim,
            num_sensors=num_sensors,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 5) Train with early stopping on val_f1_macro (equal class weighting)
    best_state = None
    best_val_f1_macro = -1.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_f1_macro_stored = 0.0
    best_val_f1_weighted = 0.0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            *Z_batch, y_batch = batch
            Z_batch = [z.to(device) for z in Z_batch]
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(Z_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_f1_macro, val_f1_weighted, _ = evaluate(
            model, val_loader, criterion, use_gating_flag=USE_GATING
        )

        if val_f1_macro > best_val_f1_macro + min_delta:
            best_val_f1_macro = val_f1_macro
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1_macro_stored = val_f1_macro
            best_val_f1_weighted = val_f1_weighted
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # 6) Test
    test_loss, test_acc, test_f1_macro, test_f1_weighted, gate_log_test = evaluate(
        model, test_loader, criterion, use_gating_flag=USE_GATING
    )

    gate_mean = {}
    gate_std = {}
    if USE_GATING:
        for idx, sensor in enumerate(sensors):
            if idx in gate_log_test and gate_log_test[idx]:
                vals = np.concatenate(gate_log_test[idx], axis=0)
                gate_mean[sensor] = float(vals.mean())
                gate_std[sensor] = float(vals.std())

    fs_combo_hash = generate_fs_combo_hash(sensor_fs)

    result = {
        "experiment_id": experiment_id,
        "fs_combo_hash": fs_combo_hash,
        "run_timestamp": run_timestamp,
        "sensors": sensors,
        "fs_map": fs_map,
        "fusion_mode": "gated" if USE_GATING else "concat",
        "embeddings_folder": EMBEDDINGS_FOLDER_NAME,
        "window_prefix": WINDOW_PREFIX,
        "train_tremor_types": TRAIN_TREMOR_TYPES,
        "test_tremor_types": TEST_TREMOR_TYPES,
        "subject_split": {
            "train_subjects": [1, 3, 4, 6, 8, 9],
            "val_subjects": VAL_SUBJECTS,
            "test_subjects": TEST_SUBJECTS,
        },
        "val_loss": float(best_val_loss),
        "val_accuracy": float(best_val_acc),
        "val_f1": float(best_val_f1_macro),
        "val_f1_macro": float(best_val_f1_macro),
        "val_f1_weighted": float(best_val_f1_weighted),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1_macro),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "gate_mean": gate_mean if USE_GATING else None,
        "gate_std": gate_std if USE_GATING else None,
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "seed": SEED,
        "gpu_id": GPU_ID,
    }

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_only = datetime.now().strftime("%H-%M-%S")

    num_sensors = len(FIXED_SENSORS)
    fs_str = "_".join(str(fs) for fs in FS_CANDIDATES)
    experiment_id = f"tremor_fs_study_k{num_sensors}_fs{fs_str}_{time_only}"

    print("=" * 80)
    print(f"TREMOR FS COMBINATION STUDY  [GPU {GPU_ID}]  —  Tremor→Tremor")
    print("=" * 80)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Run timestamp: {run_timestamp}")
    print(f"Device:        {device}")
    print("\nConfiguration:")
    print(f"  Sensors: {FIXED_SENSORS}")
    print(f"  FS candidates: {FS_CANDIDATES}")
    print(f"  Embeddings folder: {EMBEDDINGS_FOLDER_NAME}")
    print(f"  Window prefix:     {WINDOW_PREFIX}")
    print(f"  Train variants:    {TRAIN_TREMOR_TYPES}")
    print(f"  Test variants:     {TEST_TREMOR_TYPES}")
    print(f"  Subject split: train=[1,3,4,6,8,9], val={VAL_SUBJECTS}, test={TEST_SUBJECTS}")
    print(f"  Fusion mode: {'gated' if USE_GATING else 'concat'}")
    print(f"  Seed: {SEED}")

    all_fs_combinations = list(product(FS_CANDIDATES, repeat=num_sensors))
    total_combinations = len(all_fs_combinations)

    print(f"\nTotal combinations: {total_combinations}")
    print(f"  ({num_sensors} sensors × {len(FS_CANDIDATES)} frequencies = {len(FS_CANDIDATES)}^{num_sensors})")

    if MAX_COMBOS is not None and MAX_COMBOS < total_combinations:
        all_fs_combinations = all_fs_combinations[:MAX_COMBOS]
        print(f"  Limited to first {MAX_COMBOS} combinations")

    existing_combos = set()
    if SKIP_EXISTING:
        existing_combos = load_existing_combinations(LOG_FILE, experiment_id)
        if existing_combos:
            print(f"\nFound {len(existing_combos)} existing combinations for this experiment")

    print(f"\n{'=' * 80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'=' * 80}\n")

    completed = 0
    skipped = 0
    collected_results = []

    for combo_idx, fs_assignment in enumerate(all_fs_combinations, 1):
        sensor_fs = list(fs_assignment)
        combo_hash = generate_fs_combo_hash(sensor_fs)

        if SKIP_EXISTING and combo_hash in existing_combos:
            skipped += 1
            print(f"[{combo_idx}/{len(all_fs_combinations)}] Skipping {combo_hash} (already completed)")
            continue

        fs_map = {FIXED_SENSORS[i]: sensor_fs[i] for i in range(num_sensors)}
        print(f"\n{'*' * 80}")
        print(f"Combination {combo_idx}/{len(all_fs_combinations)}")
        print(f"FS map: {fs_map}")
        print(f"Combo hash: {combo_hash}")
        print(f"{'*' * 80}")

        try:
            result = run_one_experiment(
                sensors=FIXED_SENSORS,
                sensor_fs=sensor_fs,
                experiment_id=experiment_id,
                run_timestamp=run_timestamp,
            )

            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

            print("\nResults:")
            print(f"  Val F1:   {result['val_f1']:.4f} (Acc: {result['val_accuracy']:.4f})")
            print(f"  Test F1:  {result['test_f1']:.4f} (Acc: {result['test_accuracy']:.4f})")
            print(f"  Logged to: {LOG_FILE}")

            completed += 1
            collected_results.append(result)

        except Exception as error:
            print(f"\n✗ Error in combination {combo_idx}: {error}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("STUDY COMPLETE")
    print(f"{'=' * 80}")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Total listed combinations: {len(all_fs_combinations)}")
    print(f"Cache size: {len(_EMBEDDING_CACHE)} loaded sensor/fs entries")
    print(f"Results log: {LOG_FILE}")

    if collected_results:
        ranked = sorted(collected_results, key=lambda item: item["test_f1"], reverse=True)
        best = ranked[0]
        worst = ranked[-1]

        # Dynamic column widths
        fs_width = max((len(str(r["fs_map"])) for r in ranked), default=50) + 2
        fs_width = max(fs_width, 50)
        rank_width = 6
        hash_width = 12
        test_f1_width = 10
        test_acc_width = 11
        val_f1_width = 10
        total_width = rank_width + fs_width + hash_width + test_f1_width + test_acc_width + val_f1_width + 5
        header = (f"{'Rank':<{rank_width}} {'FS map':<{fs_width}} {'Hash':<{hash_width}} "
                  f"{'Test F1':<{test_f1_width}} {'Test Acc':<{test_acc_width}} {'Val F1':<{val_f1_width}}")
        sep = "-" * total_width

        print(f"\nRESULTS RANKED BY TEST F1 (MACRO):")
        print("=" * total_width)
        print(header)
        print(sep)
        for rank_idx, r in enumerate(ranked, 1):
            fs_str = str(r["fs_map"])
            print(f"{rank_idx:<{rank_width}} {fs_str:<{fs_width}} {r['fs_combo_hash']:<{hash_width}} "
                  f"{r['test_f1']:.4f}     {r['test_accuracy']:.4f}     {r['val_f1']:.4f}")

        print(f"\n{'=' * 70}")
        print("BEST COMBINATION:")
        print(f"{'=' * 70}")
        print(f"  FS map:   {best['fs_map']}")
        print(f"  Test F1:  {best['test_f1']:.4f}")
        print(f"  Test Acc: {best['test_accuracy']:.4f}")
        print(f"  Val F1:   {best['val_f1']:.4f}")
        print(f"  Val Acc:  {best['val_accuracy']:.4f}")
        print(f"{'=' * 70}")

        # Save .txt report
        report_file = LOG_DIR / f"fs_study_report_{run_timestamp}.txt"
        with open(report_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"TREMOR FS COMBINATION STUDY RESULTS  [GPU {GPU_ID}]\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp:         {run_timestamp}\n")
            f.write(f"Seed:              {SEED}\n")
            f.write(f"Sensors:           {FIXED_SENSORS}\n")
            f.write(f"FS candidates:     {FS_CANDIDATES}\n")
            f.write(f"Embeddings folder: {EMBEDDINGS_FOLDER_NAME}\n")
            f.write(f"Train variants:    {TRAIN_TREMOR_TYPES}\n")
            f.write(f"Test variants:     {TEST_TREMOR_TYPES}\n")
            f.write(f"Subject split:     train=[1,3,4,6,8,9], val={VAL_SUBJECTS}, test={TEST_SUBJECTS}\n")
            f.write(f"Fusion mode:       {'gated' if USE_GATING else 'concat'}\n")
            f.write(f"Total completed:   {completed}\n")
            f.write(f"Log file:          {LOG_FILE}\n\n")
            f.write("=" * total_width + "\n")
            f.write("RESULTS RANKED BY TEST F1 (MACRO)\n")
            f.write("=" * total_width + "\n")
            f.write(header + "\n")
            f.write(sep + "\n")
            for rank_idx, r in enumerate(ranked, 1):
                fs_str = str(r["fs_map"])
                f.write(f"{rank_idx:<{rank_width}} {fs_str:<{fs_width}} {r['fs_combo_hash']:<{hash_width}} "
                        f"{r['test_f1']:.4f}     {r['test_accuracy']:.4f}     {r['val_f1']:.4f}\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write("BEST COMBINATION\n")
            f.write("=" * 70 + "\n")
            f.write(f"  FS map:   {best['fs_map']}\n")
            f.write(f"  Test F1:  {best['test_f1']:.4f}\n")
            f.write(f"  Test Acc: {best['test_accuracy']:.4f}\n")
            f.write(f"  Val F1:   {best['val_f1']:.4f}\n")
            f.write(f"  Val Acc:  {best['val_accuracy']:.4f}\n")
        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tremor FS combination study — GPU0 — Clean→Clean")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override SEED with a single int (e.g. --seed 42)")
    args = parser.parse_args()
    if args.seed is not None:
        SEED = args.seed

    seeds_to_run = SEED if isinstance(SEED, list) else [SEED]
    for _seed in seeds_to_run:
        SEED = _seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        _EMBEDDING_CACHE.clear()   # clear cache between seeds for clean state
        print(f"\n{'='*60}\nRunning with SEED={SEED} ({seeds_to_run.index(_seed)+1}/{len(seeds_to_run)})\n{'='*60}")
        main()
