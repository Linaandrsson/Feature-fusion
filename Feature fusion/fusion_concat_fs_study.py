"""
fusion_concat_fs_study.py

Sampling Frequency Combination Study

This script systematically tests ALL possible sampling frequency combinations
for a fixed set of sensors. For n sensors and m frequency candidates,
this generates m^n combinations.

Usage:
- Set FIXED_SENSORS: The sensor list (no ablation)
- Set FS_CANDIDATES: List of frequencies to test (e.g., [10, 20, 30, 50])
- Set MAX_COMBOS: Limit number of combinations to test (None = all)
- Run the script to test all combinations

Each combination is trained from scratch and logged to fs_combo_study.jsonl.
Results include experiment_id, fs_map, accuracies, and gating weights.
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
import hashlib
from typing import List, Dict, Tuple, Optional
from itertools import product
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - FS COMBINATION STUDY
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
# Data paths
# -------------------------------
base_data_dir = Path("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files")
dataset_config = "s1_w1_aug2"  # Dataset configuration

# -------------------------------
# FS Combination Study Configuration
# -------------------------------
# Fixed sensor set (NO ablation - always use all these sensors)
FIXED_SENSORS = ["Acc_ankle", "Mag_arm", "Mag_ankle"]

# Sampling frequency candidates to test
FS_CANDIDATES = [10, 30]  # List of FS values to test for each sensor

# Combination limits
MAX_COMBOS = None  # None = test all combinations, int = limit to first N

# Skip already completed combinations (based on experiment_id in log file)
SKIP_EXISTING = True

# -------------------------------
# Train/Test Modes
# -------------------------------
TRAIN_MODE = "clean"      # "clean" | "corrupted"
TEST_MODE  = "clean"      # "clean" | "corrupted"

# -------------------------------
# Corruption configuration for TRAINING
# -------------------------------
train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.30,
    "DROPOUT_drop/Acc_arm": 0.10,
    "WEAK_SIGNAL_w0p2/Acc_chest": 0.15,
}

# -------------------------------
# Corruption configuration for TEST
# -------------------------------
test_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.20,
    "DROPOUT_drop/Acc_arm": 0.30,
    "AWGN_s0p3/Acc_chest": 0.15,
}

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
USE_GATING = True  # True = gated fusion, False = concat baseline

# Gating parameters (only used if USE_GATING=True)
gate_type = "sigmoid"
gate_hidden = 64
gate_dropout = 0.1
use_layernorm = True
alpha_floor = 0.05

# Classifier head
head_hidden_dims = (64, 64)
head_dropout = 0.4

# -------------------------------
# Logging
# -------------------------------
LOG_DIR = Path("fusion_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "fs_combo_study.jsonl"

# Enable detailed per-sample corruption logging (warning: can be large!)
LOG_SAMPLE_CORRUPTION = False

# -------------------------------
# Global embedding cache (in-memory)
# -------------------------------
_EMBEDDING_CACHE = {}


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def generate_fs_combo_hash(sensor_fs: List[int]) -> str:
    """
    Generate deterministic hash for a specific FS combination.
    This identifies a unique FS assignment.
    """
    # Sort sensors with their fs to make it deterministic
    fs_str = "_".join(str(fs) for fs in sensor_fs)
    hash_obj = hashlib.md5(fs_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def load_existing_combinations(log_file: Path, experiment_id: str) -> set:
    """Load all FS combo hashes for a specific experiment from existing log file."""
    if not log_file.exists():
        return set()
    
    combo_hashes = set()
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get('experiment_id') == experiment_id:
                        combo_hashes.add(data.get('fs_combo_hash', ''))
                except json.JSONDecodeError:
                    continue
    
    return combo_hashes


def parse_corruption_config(
    corruption_config: Dict[str, float],
    sensors: List[str]
) -> Tuple[Dict[str, Dict[str, float]], set]:
    """
    Parse corruption config and build per-sensor probability dictionaries.
    
    Args:
        corruption_config: Dict with entries like:
            - "scenario": prob (applies to all sensors)
            - "scenario/Sensor_name": prob (applies to specific sensor)
        sensors: List of sensor names
    
    Returns:
        - sensor_probs: {sensor: {scenario: prob}}
        - all_scenarios: Set of all scenarios needed
    
    Raises:
        ValueError: If probabilities sum > 1.0 for any sensor
    """
    # Initialize: all sensors start with clean=1.0
    sensor_probs = {sensor: {} for sensor in sensors}
    all_scenarios = set(["clean"])
    
    # Parse config
    for key, prob in corruption_config.items():
        if "/" in key:
            # Sensor-specific rule: "scenario/sensor"
            scenario, sensor = key.split("/", 1)
            if sensor not in sensors:
                continue
            sensor_probs[sensor][scenario] = prob
            all_scenarios.add(scenario)
        else:
            # Global rule: applies to all sensors
            scenario = key
            for sensor in sensors:
                sensor_probs[sensor][scenario] = prob
            all_scenarios.add(scenario)
    
    # Compute clean probability as remainder and validate
    for sensor in sensors:
        non_clean_sum = sum(sensor_probs[sensor].values())
        
        if non_clean_sum > 1.0 + 1e-6:
            raise ValueError(
                f"Corruption probabilities for {sensor} sum to {non_clean_sum:.3f} > 1.0!\n"
                f"  Config: {sensor_probs[sensor]}"
            )
        
        # Set clean probability
        clean_prob = max(0.0, 1.0 - non_clean_sum)
        sensor_probs[sensor]["clean"] = clean_prob
    
    return sensor_probs, all_scenarios


def get_variant_dir(fs: int, scenario: str) -> Path:
    """Construct variant directory path."""
    variant_name = f"fs{fs}_{scenario}"
    return base_data_dir / dataset_config / variant_name


def load_embeddings(sensor_name: str, fs: int, scenario: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings for a specific sensor, FS, and scenario.
    Uses global cache to avoid redundant loading.
    
    Returns dict with keys: Z_train, y_train, Z_val, y_val, Z_test, y_test
    """
    cache_key = (sensor_name, fs, scenario)
    
    # Check cache first
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    
    # Load from disk
    variant_dir = get_variant_dir(fs, scenario)
    feat_dir = variant_dir / "ExtractedFeatures"
    
    train_path = feat_dir / f"{sensor_name}_train.npz"
    val_path = feat_dir / f"{sensor_name}_val.npz"
    test_path = feat_dir / f"{sensor_name}_test.npz"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {test_path}")
    
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)
    
    data = {
        "Z_train": train_data["embeddings"].astype(np.float32),
        "y_train": train_data["labels"].astype(np.int64),
        "Z_val": val_data["embeddings"].astype(np.float32),
        "y_val": val_data["labels"].astype(np.int64),
        "Z_test": test_data["embeddings"].astype(np.float32),
        "y_test": test_data["labels"].astype(np.int64),
    }
    
    # Cache it
    _EMBEDDING_CACHE[cache_key] = data
    
    return data


def verify_label_alignment(data_dict: Dict[str, Dict], split: str = "train") -> None:
    """Verify that labels are identical across all sensors and scenarios."""
    y_key = f"y_{split}"
    reference_labels = None
    reference_key = None
    
    for key, data in data_dict.items():
        if reference_labels is None:
            reference_labels = data[y_key]
            reference_key = key
        else:
            if not np.array_equal(data[y_key], reference_labels):
                raise ValueError(
                    f"Label mismatch in {split} split!\n"
                    f"  Reference: {reference_key}\n"
                    f"  Mismatch:  {key}"
                )


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class MultiScenarioDataset(Dataset):
    """
    Dataset that can mix clean and corrupted embeddings per sample.
    
    For each sample index:
    - Select scenario based on per-sensor probability distribution
    - Selection is deterministic given (epoch, idx, seed) for reproducibility
    - Return embeddings from selected scenario
    """
    
    def __init__(
        self,
        embeddings_dict: Dict[str, Dict[str, np.ndarray]],
        split: str,
        sensors: List[str],
        per_sensor_config: Dict[str, Dict[str, float]],
        epoch: int = 0,
        log_corruption: bool = False
    ):
        """
        Args:
            embeddings_dict: {(sensor, scenario): {"Z_train": ..., "y_train": ..., etc.}}
            split: "train", "val", or "test"
            sensors: List of sensor names
            per_sensor_config: {sensor: {scenario: prob}}
            epoch: Current epoch (for deterministic scenario selection)
            log_corruption: Whether to log corruption decisions
        """
        self.embeddings_dict = embeddings_dict
        self.split = split
        self.sensors = sensors
        self.per_sensor_config = per_sensor_config
        self.epoch = epoch
        self.log_corruption = log_corruption
        
        # Get data keys
        self.Z_key = f"Z_{split}"
        self.y_key = f"y_{split}"
        
        # Get dataset size from first sensor
        first_key = (sensors[0], "clean")
        self.labels = embeddings_dict[first_key][self.y_key]
        self.n_samples = len(self.labels)
        
        # Precompute scenarios list and cumulative probabilities for each sensor
        self.sensor_scenarios = {}
        self.sensor_cum_probs = {}
        
        for sensor in sensors:
            probs = per_sensor_config[sensor]
            scenarios = list(probs.keys())
            cum_probs = np.cumsum(list(probs.values()))
            
            self.sensor_scenarios[sensor] = scenarios
            self.sensor_cum_probs[sensor] = cum_probs
        
        # Initialize corruption log
        self.corruption_log = []
        
    def set_epoch(self, epoch: int):
        """Update epoch for deterministic sampling"""
        self.epoch = epoch
    
    def select_scenario(self, sensor: str, idx: int) -> str:
        """
        Deterministically select scenario for given sensor and sample index.
        Uses MD5 hash of (epoch, sensor, idx, SEED) for reproducibility.
        """
        hash_input = f"{self.epoch}_{sensor}_{idx}_{SEED}"
        hash_obj = hashlib.md5(hash_input.encode("utf-8"))
        hash_int = int(hash_obj.hexdigest(), 16)
        rng_val = (hash_int % 10000) / 10000.0
        
        scenarios = self.sensor_scenarios[sensor]
        cum_probs = self.sensor_cum_probs[sensor]
        
        scenario_idx = np.searchsorted(cum_probs, rng_val)
        scenario_idx = min(scenario_idx, len(scenarios) - 1)
        
        return scenarios[scenario_idx]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Return embeddings for all sensors at given index."""
        embeddings = []
        corrupted_sensors = []
        
        for sensor in self.sensors:
            scenario = self.select_scenario(sensor, idx)
            key = (sensor, scenario)
            Z = self.embeddings_dict[key][self.Z_key][idx]
            embeddings.append(torch.from_numpy(Z))
            
            if scenario != "clean":
                corrupted_sensors.append({
                    "sensor": sensor,
                    "scenario": scenario
                })
        
        if self.log_corruption and corrupted_sensors:
            self.corruption_log.append({
                "idx": int(idx),
                "label": int(self.labels[idx]),
                "corrupted_sensors": corrupted_sensors
            })
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return (*embeddings, label)
    
    def get_corruption_summary(self) -> Dict:
        """Get summary statistics about corruption."""
        sensor_corruption_counts = defaultdict(lambda: defaultdict(int))
        
        for sensor in self.sensors:
            for idx in range(self.n_samples):
                scenario = self.select_scenario(sensor, idx)
                sensor_corruption_counts[sensor][scenario] += 1
        
        summary = {}
        for sensor, counts in sensor_corruption_counts.items():
            total = sum(counts.values())
            summary[sensor] = {
                scenario: {
                    "count": count,
                    "percentage": count / total * 100
                }
                for scenario, count in counts.items()
            }
        
        return summary


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
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def run_one_experiment(
    sensors: List[str],
    sensor_fs: List[int],
    train_mode: str,
    test_mode: str,
    use_gating: bool,
    experiment_id: str,
    run_timestamp: str
) -> Dict:
    """
    Run one complete training experiment for a given FS configuration.
    
    Args:
        sensors: List of sensor names
        sensor_fs: List of sampling rates (one per sensor)
        train_mode: "clean" or "corrupted"
        test_mode: "clean" or "corrupted"
        use_gating: Whether to use gated fusion
        experiment_id: Unique experiment identifier
    
    Returns:
        Dictionary with results (accuracy, gate weights, etc.)
    """
    
    # Parse corruption configurations
    if train_mode == "corrupted":
        train_sensor_probs, train_scenarios = parse_corruption_config(
            train_corruption_config, sensors
        )
    else:
        train_sensor_probs = {sensor: {"clean": 1.0} for sensor in sensors}
        train_scenarios = {"clean"}
    
    if test_mode == "corrupted":
        test_sensor_probs, test_scenarios = parse_corruption_config(
            test_corruption_config, sensors
        )
    else:
        test_sensor_probs = {sensor: {"clean": 1.0} for sensor in sensors}
        test_scenarios = {"clean"}
    
    # Combine all scenarios we need
    all_scenarios = train_scenarios | test_scenarios
    
    # Load embeddings for all (sensor, fs, scenario) combinations
    embeddings_dict = {}
    for sensor_idx, sensor in enumerate(sensors):
        fs = sensor_fs[sensor_idx]
        for scenario in all_scenarios:
            try:
                data = load_embeddings(sensor, fs, scenario)
                embeddings_dict[(sensor, scenario)] = data
            except FileNotFoundError as e:
                print(f"  ✗ Missing: {sensor} (fs{fs}) - {scenario}")
                raise
    
    # Verify label alignment
    verify_label_alignment(embeddings_dict, "train")
    verify_label_alignment(embeddings_dict, "val")
    verify_label_alignment(embeddings_dict, "test")
    
    # Get dataset info
    first_key = (sensors[0], "clean")
    embed_dim = embeddings_dict[first_key]["Z_train"].shape[1]
    num_classes = len(np.unique(embeddings_dict[first_key]["y_train"]))
    num_sensors = len(sensors)
    
    # Create datasets
    train_dataset = MultiScenarioDataset(
        embeddings_dict=embeddings_dict,
        split="train",
        sensors=sensors,
        per_sensor_config=train_sensor_probs,
        epoch=0,
        log_corruption=LOG_SAMPLE_CORRUPTION
    )
    
    val_sensor_probs = {sensor: {"clean": 1.0} for sensor in sensors}
    val_dataset = MultiScenarioDataset(
        embeddings_dict=embeddings_dict,
        split="val",
        sensors=sensors,
        per_sensor_config=val_sensor_probs,
        epoch=0,
        log_corruption=False
    )
    
    test_dataset = MultiScenarioDataset(
        embeddings_dict=embeddings_dict,
        split="test",
        sensors=sensors,
        per_sensor_config=test_sensor_probs,
        epoch=0,
        log_corruption=LOG_SAMPLE_CORRUPTION
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    if use_gating:
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
    else:
        model = ConcatFusionModel(
            embed_dim=embed_dim,
            num_sensors=num_sensors,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    
    def unpack_batch(batch):
        *Z, y = batch
        Z = [z.to(device) for z in Z]
        y = y.to(device)
        return Z, y
    
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        
        # Train
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
        
        # Validate
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
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    # Test evaluation
    model.eval()
    y_pred = []
    gate_log_test = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            Z, y_batch = unpack_batch(batch)
            logits, alphas = model(Z)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
            if use_gating and alphas is not None:
                for i in range(num_sensors):
                    gate_log_test[sensors[i]].append(alphas[:, i].cpu().numpy())
    
    y_test = embeddings_dict[first_key]["y_test"]
    test_acc = accuracy_score(y_test, y_pred)
    
    # Compute gate statistics
    gate_mean = {}
    gate_std = {}
    if use_gating:
        for s in sensors:
            vals = np.concatenate(gate_log_test[s], axis=0)
            gate_mean[s] = float(vals.mean())
            gate_std[s] = float(vals.std())
    
    # Build fs_map and generate combo hash
    fs_map = {sensors[i]: sensor_fs[i] for i in range(len(sensors))}
    fs_combo_hash = generate_fs_combo_hash(sensor_fs)
    
    # Return results
    results = {
        "experiment_id": experiment_id,
        "fs_combo_hash": fs_combo_hash,
        "run_timestamp": run_timestamp,
        "sensors": sensors,
        "fs_map": fs_map,
        "train_mode": train_mode,
        "test_mode": test_mode,
        "fusion_mode": "gated" if use_gating else "concat",
        "val_accuracy": float(val_acc),
        "val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "gate_mean": gate_mean if use_gating else None,
        "gate_std": gate_std if use_gating else None,
        "seed": SEED,
        "dataset_config": dataset_config,
    }
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_only = datetime.now().strftime("%H-%M-%S")
    
    # Generate descriptive experiment ID for this run (shared by all combinations)
    num_sensors = len(FIXED_SENSORS)
    fs_str = "_".join(str(fs) for fs in FS_CANDIDATES)
    experiment_id = f"fs_study_k{num_sensors}_fs{fs_str}_{time_only}"
    
    print("="*70)
    print("SAMPLING FREQUENCY COMBINATION STUDY")
    print("="*70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Run timestamp: {run_timestamp}")
    print(f"\nConfiguration:")
    print(f"  Fixed sensors: {FIXED_SENSORS}")
    print(f"  FS candidates: {FS_CANDIDATES}")
    print(f"  Train mode: {TRAIN_MODE}")
    print(f"  Test mode: {TEST_MODE}")
    print(f"  Use gating: {USE_GATING}")
    print(f"  Dataset config: {dataset_config}")
    
    # Generate all FS combinations
    all_fs_combinations = list(product(FS_CANDIDATES, repeat=num_sensors))
    total_combinations = len(all_fs_combinations)
    
    print(f"\nTotal combinations: {total_combinations}")
    print(f"  ({num_sensors} sensors × {len(FS_CANDIDATES)} frequencies = {len(FS_CANDIDATES)}^{num_sensors})")
    
    # Warning for large combination spaces
    if total_combinations > 5000:
        print(f"\n{'!'*70}")
        print("WARNING: Large number of FS combinations. This may take a very long time.")
        print(f"{'!'*70}")
    
    # Apply MAX_COMBOS limit
    if MAX_COMBOS is not None and MAX_COMBOS < total_combinations:
        all_fs_combinations = all_fs_combinations[:MAX_COMBOS]
        print(f"  Limited to first {MAX_COMBOS} combinations")
    
    # Load existing combinations if skipping
    existing_combos = set()
    if SKIP_EXISTING:
        existing_combos = load_existing_combinations(LOG_FILE, experiment_id)
        if existing_combos:
            print(f"\nFound {len(existing_combos)} existing combinations for this experiment")
    
    # Run experiments
    print(f"\n{'='*70}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*70}\n")
    
    completed = 0
    skipped = 0
    
    for combo_idx, fs_assignment in enumerate(all_fs_combinations, 1):
        sensor_fs = list(fs_assignment)
        
        # Generate combo hash
        combo_hash = generate_fs_combo_hash(sensor_fs)
        
        # Check if already exists
        if SKIP_EXISTING and combo_hash in existing_combos:
            skipped += 1
            continue
        
        # Print header
        print(f"{'*'*70}")
        print(f"Combination {combo_idx}/{len(all_fs_combinations)}")
        print(f"{'*'*70}")
        fs_map = {FIXED_SENSORS[i]: sensor_fs[i] for i in range(num_sensors)}
        print(f"FS map: {fs_map}")
        print(f"Combo hash: {combo_hash}")
        
        try:
            # Run experiment
            result = run_one_experiment(
                sensors=FIXED_SENSORS,
                sensor_fs=sensor_fs,
                train_mode=TRAIN_MODE,
                test_mode=TEST_MODE,
                use_gating=USE_GATING,
                experiment_id=experiment_id,
                run_timestamp=run_timestamp
            )
            
            # Print summary
            print(f"\nResults:")
            print(f"  Val Acc:  {result['val_accuracy']:.4f}")
            print(f"  Test Acc: {result['test_accuracy']:.4f}")
            
            if USE_GATING and result['gate_mean']:
                print(f"\nGating weights:")
                for sensor in FIXED_SENSORS:
                    print(f"  {sensor}: {result['gate_mean'][sensor]:.3f} ± {result['gate_std'][sensor]:.3f}")
            
            # Log to file
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            print(f"\n✓ Logged to {LOG_FILE}")
            completed += 1
            
        except Exception as e:
            print(f"\n✗ Error in combination {combo_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*70}")
    print("STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\nCompleted: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(all_fs_combinations)}")
    print(f"\nResults logged to: {LOG_FILE}")
    print(f"Cache size: {len(_EMBEDDING_CACHE)} embeddings loaded")


if __name__ == "__main__":
    main()
