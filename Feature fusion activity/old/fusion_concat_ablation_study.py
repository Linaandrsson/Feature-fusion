"""
fusion_concat_train_v4.py

Sensor Ablation Study Framework

New features in v4:
1. Automatic sensor ablation studies with k-sensor subsets
2. Tests all possible combinations of k sensors from available sensor list
3. Choose between clean-only or corruption-augmented training
4. Comprehensive logging of all combinations and their performance
5. Automatic ranking from best to worst accuracy
6. Backwards compatible with v3 (set ABLATION_K to full sensor count)

Usage:
- Specify available sensors and their sampling rates
- Set ABLATION_K to desired subset size (or list of sizes)
- Set TRAIN_MODE and TEST_MODE independently ("clean" or "corrupted")
- Run to test all combinations automatically
- Results are logged and ranked by performance
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
# Base directory where all scenario folders live
base_data_dir = Path("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files")

# Dataset configuration (e.g., "s1_w1_aug2")
dataset_config = "s1_w1_aug2"

# Available sensors and their sampling rates (ablation will test subsets of these)
ALL_SENSORS = ["Acc_ankle", "Acc_arm", "Acc_chest", "Mag_arm", "Mag_ankle", "Gyro_ankle", "Gyro_arm", "ECG"]  # All available sensors
ALL_SENSOR_FS = [50,50,50,50,50,50,50,50]  # FS for each sensor (fs10, fs20, fs30, fs50)

assert len(ALL_SENSORS) == len(ALL_SENSOR_FS), "ALL_SENSORS and ALL_SENSOR_FS must have same length"

# -------------------------------
# Sensor Ablation Configuration
# -------------------------------
ABLATION_K = [3]  # List of subset sizes to test (e.g., [2, 3] tests all 2-sensor and 3-sensor combos)
                     # Set to [len(ALL_SENSORS)] to test full sensor set only

# -------------------------------
# Train/Test Modes
# -------------------------------
TRAIN_MODE = "clean"      # "clean" | "corrupted"
TEST_MODE  = "clean"      # "clean" | "corrupted"

# -------------------------------
# Corruption configuration for TRAINING
# -------------------------------
# Define corruption per sensor using "scenario/sensor" syntax
# Default: All sensors use clean data (probability 1.0)

# Realistic scenario: Daily patient monitoring with wearable sensors
train_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.30,           # Ankle: Impact noise from walking, 30%
    "DROPOUT_drop/Acc_arm": 0.10,          # Arm: Patient adjusts sensor, 10%
    "WEAK_SIGNAL_w0p2/Acc_chest": 0.15,    # Chest: Weak signal, 15%
    # Note: Remaining probability (clean) is calculated automatically
}

# -------------------------------
# Corruption configuration for TEST
# -------------------------------
# Same format as train_corruption_config
# Default: clean data only for all sensors
#
# Stress test: Simulate a "bad day" with multiple sensor issues
test_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.20,           # Heavy impact noise, poor placement, 20%
    "DROPOUT_drop/Acc_arm": 0.30,          # Sensor loosened, frequent dropouts, 30%
    "AWGN_s0p3/Acc_chest": 0.15,           # Even chest sensor affected slightly, 15%
}
# Result: Testing robustness under worst-case realistic conditions

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
gate_type = "sigmoid"  # "sigmoid" or "softmax"
gate_hidden = 64  # 0 = linear gate, >0 = MLP gate
gate_dropout = 0.1
use_layernorm = True
alpha_floor = 0.05

# Classifier head
head_hidden_dims = (64, 64)
head_dropout = 0.4

# -------------------------------
# Logging and plotting
# -------------------------------
SAVE_PLOTS = True
PLOT_DIR = Path("ablation_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("fusion_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "fusion_v4_ablation.jsonl"

# Corruption metadata logging
CORRUPTION_LOG_FILE = LOG_DIR / "corruption_metadata.jsonl"

# Enable detailed per-sample corruption logging (warning: can be large!)
LOG_SAMPLE_CORRUPTION = False  # Set to True for full per-sample logs


# ═══════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

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
            # Strip .npz extension if present (for backward compatibility)
            sensor = sensor.replace(".npz", "")
            if sensor not in sensors:
                print(f"Warning: Sensor '{sensor}' in config not in sensor list. Skipping.")
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
        
        if non_clean_sum > 1.0 + 1e-6:  # Small tolerance for float errors
            raise ValueError(
                f"Corruption probabilities for {sensor} sum to {non_clean_sum:.3f} > 1.0!\n"
                f"  Config: {sensor_probs[sensor]}"
            )
        
        # Set clean probability
        clean_prob = max(0.0, 1.0 - non_clean_sum)
        sensor_probs[sensor]["clean"] = clean_prob
        
        if clean_prob < 1e-6:
            print(f"Warning: {sensor} has no clean samples (prob={clean_prob:.3f})")
    
    return sensor_probs, all_scenarios


def get_variant_dir(fs: int, scenario: str) -> Path:
    """
    Construct variant directory path.
    E.g., fs30_clean, fs30_AWGN_s0p3, etc.
    """
    variant_name = f"fs{fs}_{scenario}"
    return base_data_dir / dataset_config / variant_name


def load_embeddings(sensor_name: str, fs: int, scenario: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings for a specific sensor, FS, and scenario.
    Returns dict with keys: Z_train, y_train, Z_val, y_val, Z_test, y_test
    
    Loads from separate split files: {sensor}_train.npz, {sensor}_val.npz, {sensor}_test.npz
    """
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
    
    return {
        "Z_train": train_data["embeddings"].astype(np.float32),
        "y_train": train_data["labels"].astype(np.int64),
        "Z_val": val_data["embeddings"].astype(np.float32),
        "y_val": val_data["labels"].astype(np.int64),
        "Z_test": test_data["embeddings"].astype(np.float32),
        "y_test": test_data["labels"].astype(np.int64),
    }


def verify_label_alignment(data_dict: Dict[str, Dict], split: str = "train") -> None:
    """
    Verify that labels are identical across all sensors and scenarios.
    Raises ValueError if mismatch detected.
    """
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
                    f"  Mismatch:  {key}\n"
                    f"  Shapes: {reference_labels.shape} vs {data[y_key].shape}\n"
                    f"  First 10: {reference_labels[:10]} vs {data[y_key][:10]}"
                )
    
    print(f"✓ Label alignment verified for {split} split ({len(data_dict)} variants)")


# ═══════════════════════════════════════════════════════════════
# DATASET CLASSES
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
        embeddings_dict: Dict[str, Dict[str, np.ndarray]],  # {(sensor, scenario): data}
        split: str,  # "train", "val", or "test"
        sensors: List[str],
        corruption_probs: Optional[Dict[str, float]],  # Deprecated, kept for compatibility
        per_sensor_config: Dict[str, Dict[str, float]],  # {sensor: {scenario: prob}}
        epoch: int = 0,
        log_corruption: bool = False
    ):
        """
        Args:
            embeddings_dict: Nested dict with structure:
                {(sensor, scenario): {"Z_train": ..., "y_train": ..., etc.}}
            split: Which split to use ("train", "val", "test")
            sensors: List of sensor names
            corruption_probs: Deprecated (use per_sensor_config)
            per_sensor_config: Per-sensor corruption probabilities {sensor: {scenario: prob}}
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
        Uses stable hash (md5) of (epoch, sensor, idx, SEED) for reproducibility.
        """
        # Create deterministic random value in [0, 1) using stable hash
        hash_input = f"{self.epoch}_{sensor}_{idx}_{SEED}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        # Convert first 8 bytes to integer for stable random value
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        rng_val = (hash_int % 100000) / 100000.0
        
        # Select scenario based on cumulative probabilities
        scenarios = self.sensor_scenarios[sensor]
        cum_probs = self.sensor_cum_probs[sensor]
        
        scenario_idx = np.searchsorted(cum_probs, rng_val)
        scenario_idx = min(scenario_idx, len(scenarios) - 1)
        
        return scenarios[scenario_idx]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Return embeddings for all sensors at given index.
        Each sensor's embedding may come from different scenario.
        """
        embeddings = []
        corrupted_sensors = []
        
        for sensor in self.sensors:
            # Select scenario for this sensor
            scenario = self.select_scenario(sensor, idx)
            
            # Get embedding
            key = (sensor, scenario)
            Z = self.embeddings_dict[key][self.Z_key][idx]
            embeddings.append(torch.from_numpy(Z))
            
            # Track corruption
            if scenario != "clean":
                corrupted_sensors.append({
                    "sensor": sensor,
                    "scenario": scenario
                })
        
        # Log corruption if enabled
        if self.log_corruption and corrupted_sensors:
            self.corruption_log.append({
                "idx": int(idx),
                "label": int(self.labels[idx]),
                "corrupted_sensors": corrupted_sensors
            })
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return (*embeddings, label)
    
    def get_corruption_summary(self) -> Dict:
        """
        Get summary statistics about corruption.
        """
        sensor_corruption_counts = defaultdict(lambda: defaultdict(int))
        
        for sensor in self.sensors:
            for idx in range(self.n_samples):
                scenario = self.select_scenario(sensor, idx)
                sensor_corruption_counts[sensor][scenario] += 1
        
        # Convert to regular dict and compute percentages
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
# MODEL DEFINITIONS (from v2, unchanged)
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

def train_sensor_combination(
    sensors: List[str],
    sensor_fs: List[int],
    embeddings_dict: Dict,
    embed_dim: int,
    num_classes: int,
    train_mode: str,
    test_mode: str,
    all_sensors: List[str],
    experiment_id: str
) -> Dict:
    """
    Train and evaluate a specific sensor combination.
    
    Args:
        sensors: List of sensor names in this combination
        sensor_fs: List of sampling rates for each sensor
        embeddings_dict: Pre-loaded embeddings for all sensors
        embed_dim: Embedding dimension
        num_classes: Number of classes
        train_mode: "clean" or "corrupted" for training data
        test_mode: "clean" or "corrupted" for test data
        all_sensors: List of all available sensors (for ablation tracking)
    
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
    print(f"Sampling rates: {sensor_fs}")
    print(f"Train mode: {train_mode.upper()}")
    print(f"Test mode: {test_mode.upper()}")
    print(f"{'='*70}")
    
    # ---- Step 0: Parse corruption configurations ----
    print("\n[0/7] Configuring data sources...")
    
    # Training configuration
    if train_mode == "corrupted":
        # Parse training corruption config
        train_sensor_probs, train_scenarios = parse_corruption_config(
            train_corruption_config, sensors
        )
    else:
        # Clean data only for training
        train_sensor_probs = {sensor: {"clean": 1.0} for sensor in sensors}
        train_scenarios = {"clean"}
    
    # Test configuration
    if test_mode == "corrupted":
        # Parse test corruption config
        test_sensor_probs, test_scenarios = parse_corruption_config(
            test_corruption_config, sensors
        )
    else:
        # Clean data only for testing
        test_sensor_probs = {sensor: {"clean": 1.0} for sensor in sensors}
        test_scenarios = {"clean"}
    
    # Print configuration summary
    if train_mode == "corrupted":
        print("\nTraining corruption configuration:")
        for sensor in sensors:
            probs = train_sensor_probs[sensor]
            non_clean = {k: v for k, v in probs.items() if k != "clean" and v > 0}
            if non_clean:
                print(f"  {sensor}:")
                for scenario, prob in non_clean.items():
                    print(f"    {scenario}: {prob:.1%}")
                print(f"    clean: {probs['clean']:.1%}")
            else:
                print(f"  {sensor}: 100% clean")
    else:
        print("\nTraining: CLEAN DATA ONLY")
    
    if test_mode == "corrupted":
        print("\nTest corruption configuration:")
        for sensor in sensors:
            probs = test_sensor_probs[sensor]
            non_clean = {k: v for k, v in probs.items() if k != "clean" and v > 0}
            if non_clean:
                print(f"  {sensor}:")
                for scenario, prob in non_clean.items():
                    print(f"    {scenario}: {prob:.1%}")
                print(f"    clean: {probs['clean']:.1%}")
            else:
                print(f"  {sensor}: 100% clean")
    else:
        print("Test: CLEAN DATA ONLY")
    
    # ---- Step 1: Use embeddings already loaded ----
    print("\n[1/6] Using pre-loaded embeddings for this combination...")
    
    # Verify we have all needed embeddings for this sensor combination
    needed_pairs = set()
    for sensor in sensors:
        for scenario in train_sensor_probs[sensor].keys():
            if train_sensor_probs[sensor][scenario] > 0:
                needed_pairs.add((sensor, scenario))
    for sensor in sensors:
        for scenario in test_sensor_probs[sensor].keys():
            if test_sensor_probs[sensor][scenario] > 0:
                needed_pairs.add((sensor, scenario))
    
    # Check all needed pairs exist
    for sensor, scenario in needed_pairs:
        if (sensor, scenario) not in embeddings_dict:
            raise ValueError(f"Missing embeddings for {sensor} - {scenario}")
    
    print(f"  ✓ All embeddings available for {len(sensors)} sensors")
    
    # Get basic info
    first_key = (sensors[0], "clean")
    n_train = len(embeddings_dict[first_key]["y_train"])
    n_val = len(embeddings_dict[first_key]["y_val"])
    n_test = len(embeddings_dict[first_key]["y_test"])
    
    print(f"\nDataset info:")
    print(f"  Sensors: {num_sensors}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")
    
    # ---- Step 2: Create datasets ----
    print("\n[2/6] Creating datasets...")
    
    # Training dataset with corruption
    train_dataset = MultiScenarioDataset(
        embeddings_dict=embeddings_dict,
        split="train",
        sensors=sensors,
        corruption_probs=None,  # Not used anymore
        per_sensor_config=train_sensor_probs,
        epoch=0,
        log_corruption=LOG_SAMPLE_CORRUPTION
    )
    
    # Validation dataset (clean only)
    val_sensor_probs = {sensor: {"clean": 1.0} for sensor in sensors}
    val_dataset = MultiScenarioDataset(
        embeddings_dict=embeddings_dict,
        split="val",
        sensors=sensors,
        corruption_probs=None,
        per_sensor_config=val_sensor_probs,
        epoch=0,
        log_corruption=False
    )
    
    # Test dataset
    test_dataset = MultiScenarioDataset(
        embeddings_dict=embeddings_dict,
        split="test",
        sensors=sensors,
        corruption_probs=None,
        per_sensor_config=test_sensor_probs,
        epoch=0,
        log_corruption=LOG_SAMPLE_CORRUPTION
    )
    
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples (clean only)")
    print(f"  Test dataset: {len(test_dataset)} samples")
    
    # Print corruption summary
    print("\nTraining corruption summary:")
    train_summary = train_dataset.get_corruption_summary()
    for sensor, scenarios in train_summary.items():
        print(f"  {sensor}:")
        for scenario, stats in scenarios.items():
            print(f"    {scenario}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ---- Step 3: Create model ----
    print("\n[3/6] Creating fusion model...")
    
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
    print("\n[4/6] Training...")
    
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    
    def unpack_batch(batch):
        *Z, y = batch
        Z = [z.to(device) for z in Z]
        y = y.to(device)
        return Z, y
    
    for epoch in range(epochs):
        # Update epoch in dataset for deterministic corruption
        train_dataset.set_epoch(epoch)
        
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
    
    # ---- Step 5: Test evaluation ----
    print("\n[5/6] Evaluating on test set...")
    
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
    y_test = embeddings_dict[first_key]["y_test"]
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Test corruption summary (show if any sensor has non-clean data)
    has_corruption = any(
        any(prob > 0 for scenario, prob in test_sensor_probs[s].items() if scenario != "clean")
        for s in sensors
    )
    if has_corruption:
        print("\nTest corruption summary:")
        test_summary = test_dataset.get_corruption_summary()
        for sensor, scenarios in test_summary.items():
            print(f"  {sensor}:")
            for scenario, stats in scenarios.items():
                print(f"    {scenario}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    # ---- Step 6: Logging results ----
    print("\n[6/6] Logging results...")
    
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
        "sensor_fs": sensor_fs,
        "num_sensors": len(sensors),
        "train_mode": train_mode,
        "test_mode": test_mode,
        "fusion_mode": "gated" if USE_GATING else "concat",
        "val_accuracy": float(val_acc),
        "val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "gate_mean": gate_mean if USE_GATING else None,
        "gate_std": gate_std if USE_GATING else None,
        "seed": SEED,
    }
    
    # Log run results
    run_log = {
        **results,
        "train_sensor_probs": train_sensor_probs if train_mode == "corrupted" else None,
        "test_sensor_probs": test_sensor_probs if test_mode == "corrupted" else None,
    }
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(run_log) + "\n")
        print(f"  ✓ Logged to {LOG_FILE}")
    except Exception as e:
        print(f"  ✗ Warning: Could not write to {LOG_FILE}: {e}")
    
    # Log corruption metadata
    corruption_metadata = {
        "train_summary": train_dataset.get_corruption_summary(),
        "test_summary": test_dataset.get_corruption_summary(),
        "train_corruption_config": train_corruption_config,
        "test_corruption_config": test_corruption_config,
        "train_sensor_probs": train_sensor_probs,
        "test_sensor_probs": test_sensor_probs,
        "sensors": sensors,
        "sensor_fs": sensor_fs,
    }
    
    try:
        with open(CORRUPTION_LOG_FILE, "a") as f:
            f.write(json.dumps(corruption_metadata) + "\n")
        print(f"  ✓ Logged corruption metadata to {CORRUPTION_LOG_FILE}")
    except Exception as e:
        print(f"  ✗ Warning: Could not write corruption metadata: {e}")
    
    # Plot gating weights
    if USE_GATING and SAVE_PLOTS:
        print("\nPlotting gating weights...")
        
        # Concatenate gate logs
        for k in list(gate_log_test.keys()):
            gate_log_test[k] = np.concatenate(gate_log_test[k], axis=0)
        
        # Create unique filename based on sensor combination
        sensor_id = "_".join(sensors)
        
        # Mean bar plot
        means = {k: float(v.mean()) for k, v in gate_log_test.items()}
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(means)), list(means.values()))
        plt.xticks(range(len(means)), list(means.keys()), rotation=45, ha="right")
        plt.ylabel("Average gating weight")
        plt.title(f"Test Set - Average Sensor Gating Weights\n{', '.join(sensors)}")
        plt.tight_layout()
        plot_file = PLOT_DIR / f"{sensor_id}_gates_mean.png"
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Saved {plot_file}")
        plt.close()
        
        # Distribution plot
        plt.figure(figsize=(10, 4))
        for k, v in gate_log_test.items():
            plt.hist(v, bins=60, alpha=0.4, label=k)
        plt.xlabel("Gating weight")
        plt.ylabel("Count")
        plt.title(f"Test Set - Gating Weight Distributions\n{', '.join(sensors)}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plot_file = PLOT_DIR / f"{sensor_id}_gates_hist.png"
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Saved {plot_file}")
        plt.close()
    
    print("\n" + "="*70)
    print(f"Combination complete: {sensors}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*70)
    
    return results


def main():
    """
    Main ablation study loop.
    Tests all k-sensor combinations and ranks results.
    """
    # Generate experiment timestamp (once per run)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("="*70)
    print("SENSOR ABLATION FRAMEWORK (v4)")
    print("="*70)
    print(f"\nRun timestamp: {run_timestamp}")
    print(f"\nConfiguration:")
    print(f"  Available sensors: {ALL_SENSORS}")
    print(f"  Sensor FS: {ALL_SENSOR_FS}")
    print(f"  Subset sizes (k): {ABLATION_K}")
    print(f"  Train mode: {TRAIN_MODE}")
    print(f"  Test mode: {TEST_MODE}")
    print(f"  Dataset config: {dataset_config}")
    
    # Validate ablation_k
    if isinstance(ABLATION_K, int):
        ablation_k_list = [ABLATION_K]
    else:
        ablation_k_list = ABLATION_K
    
    for k in ablation_k_list:
        if k > len(ALL_SENSORS):
            raise ValueError(f"ABLATION_K={k} > number of available sensors ({len(ALL_SENSORS)})")
        if k < 1:
            raise ValueError(f"ABLATION_K={k} must be >= 1")
    
    # ---- Load ALL embeddings once ----
    print(f"\n{'='*70}")
    print("PRE-LOADING ALL EMBEDDINGS")
    print(f"{'='*70}")
    
    # Determine which embeddings we need
    train_scenarios = set()
    test_scenarios = set()
    
    if TRAIN_MODE == "corrupted":
        # Parse train configs to know which scenarios we need
        temp_sensor_probs_train, train_scenarios = parse_corruption_config(
            train_corruption_config, ALL_SENSORS
        )
    else:
        train_scenarios = {"clean"}
    
    if TEST_MODE == "corrupted":
        # Parse test configs to know which scenarios we need
        temp_sensor_probs_test, test_scenarios = parse_corruption_config(
            test_corruption_config, ALL_SENSORS
        )
    else:
        test_scenarios = {"clean"}
    
    # Combine all scenarios we need
    all_scenarios = train_scenarios | test_scenarios
    
    # Load all (sensor, scenario) pairs
    print(f"\nLoading embeddings for {len(ALL_SENSORS)} sensors, {len(all_scenarios)} scenarios...")
    embeddings_dict = {}
    for sensor_idx, sensor in enumerate(ALL_SENSORS):
        fs = ALL_SENSOR_FS[sensor_idx]
        for scenario in all_scenarios:
            try:
                data = load_embeddings(sensor, fs, scenario)
                embeddings_dict[(sensor, scenario)] = data
                print(f"  ✓ {sensor} (fs{fs}) - {scenario}")
            except FileNotFoundError as e:
                print(f"  ✗ Missing: {sensor} (fs{fs}) - {scenario}")
                if scenario == "clean":
                    raise  # Clean is required
                print(f"     Skipping this scenario for {sensor}")
    
    # Verify label alignment
    print("\nVerifying label alignment across all data...")
    verify_label_alignment(embeddings_dict, "train")
    verify_label_alignment(embeddings_dict, "val")
    verify_label_alignment(embeddings_dict, "test")
    print("  ✓ All labels aligned")
    
    # Get dataset info
    first_key = (ALL_SENSORS[0], "clean")
    embed_dim = embeddings_dict[first_key]["Z_train"].shape[1]
    num_classes = len(np.unique(embeddings_dict[first_key]["y_train"]))
    
    print(f"\nDataset info:")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Num classes: {num_classes}")
    
    # ---- Run ablation study ----
    all_results = []
    
    for k in ablation_k_list:
        print(f"\n{'='*70}")
        print(f"TESTING ALL {k}-SENSOR COMBINATIONS")
        print(f"{'='*70}")
        
        # Create experiment ID for this k-value
        experiment_id = f"ablation_k{k}_{run_timestamp}"
        print(f"\nExperiment ID: {experiment_id}")
        
        # Generate all k-combinations
        sensor_combinations = list(combinations(range(len(ALL_SENSORS)), k))
        total_combos = len(sensor_combinations)
        
        print(f"\nTotal combinations to test: {total_combos}")
        
        for combo_idx, sensor_indices in enumerate(sensor_combinations, 1):
            # Get sensor names and fs for this combination
            combo_sensors = [ALL_SENSORS[i] for i in sensor_indices]
            combo_fs = [ALL_SENSOR_FS[i] for i in sensor_indices]
            
            print(f"\n{'*'*70}")
            print(f"Combination {combo_idx}/{total_combos} (k={k})")
            print(f"{'*'*70}")
            
            try:
                # Train this combination
                result = train_sensor_combination(
                    sensors=combo_sensors,
                    sensor_fs=combo_fs,
                    embeddings_dict=embeddings_dict,
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    train_mode=TRAIN_MODE,
                    test_mode=TEST_MODE,
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
    
    # Sort by test accuracy (descending)
    all_results_sorted = sorted(all_results, key=lambda x: x["test_accuracy"], reverse=True)
    
    # Print summary
    print(f"\nTested {len(all_results)} sensor combinations")
    print(f"\nRESULTS RANKED BY TEST ACCURACY:")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Sensors':<30} {'Test Acc':<10} {'Val Acc':<10} {'Val Loss':<10}")
    print(f"{'-'*70}")
    
    for rank, result in enumerate(all_results_sorted, 1):
        sensors_str = ", ".join(result["sensors"])
        if len(sensors_str) > 28:
            sensors_str = sensors_str[:25] + "..."
        print(f"{rank:<6} {sensors_str:<30} {result['test_accuracy']:.4f}    "
              f"{result['val_accuracy']:.4f}    {result['val_loss']:.4f}")
    
    # Print best combination
    best = all_results_sorted[0]
    print(f"\n{'='*70}")
    print("BEST COMBINATION:")
    print(f"{'='*70}")
    print(f"  Sensors: {', '.join(best['sensors'])}")
    if best.get('ablated_sensors'):
        print(f"  Ablated: {', '.join(best['ablated_sensors'])}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    print(f"  Val Accuracy: {best['val_accuracy']:.4f}")
    print(f"  Val Loss: {best['val_loss']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
