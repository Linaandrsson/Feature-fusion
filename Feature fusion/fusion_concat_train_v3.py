"""
fusion_concat_train_v3.py

Multi-FS, Multi-Scenario Fusion Training Pipeline

New features in v3:
1. Per-sensor FS specification (sensor_fs)
2. Training with controlled corruption mixing (clean + noise scenarios)
3. Val is clean-only (default)
4. Test with controlled corruption
5. Corruption metadata logging
6. Full backwards compatibility with v2

Usage:
- Specify sensors and their sampling rates
- Define corruption scenarios and probabilities for training
- Validation uses clean embeddings only
- Test can use clean or controlled corruption
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

# Sensors to use and their sampling rates
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest"]  # List of sensor names to include
sensor_fs = [30, 20, 30]  # FS for each sensor (fs10, fs20, fs30, fs50)

assert len(sensors) == len(sensor_fs), "sensors and sensor_fs must have same length"

# -------------------------------
# Corruption configuration for TRAINING
# -------------------------------
# Define corruption per sensor using "scenario/sensor" syntax
# Default: All sensors use clean data (probability 1.0)

# Realistic scenario: Daily patient monitoring with wearable sensors
train_corruption_config = {
    "AWGN_s0p3": 0.30,           # Ankle: Impact noise from walking, 20%
    "DROPOUT_drop": 0.10,          # Arm: Patient adjusts sensor, 10%
    "WEAK_SIGNAL_w0p2": 0.15,      # Acc_chest
    # Acc_chest: 100% clean (most stable placement, strapped to chest)
}
# Result: ankle 80% clean, arm 90% clean, mag_arm 75% clean, chest 100% clean

# -------------------------------
# Corruption configuration for TEST
# -------------------------------
# Same format as train_corruption_config
# Default: clean data only for all sensors
#
# Stress test: Simulate a "bad day" with multiple sensor issues
test_corruption_config = {
    "AWGN_s0p3/Acc_ankle": 0.40,           # Heavy impact noise, poor placement, 40%
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
# Logging and plotting
# -------------------------------
SAVE_PLOTS = True
PLOT_DIR = Path("gating_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("fusion_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "fusion_v3_experiments.jsonl"

# Corruption metadata logging
CORRUPTION_LOG_FILE = LOG_DIR / "corruption_metadata.jsonl"

# Enable detailed per-sample corruption logging (warning: can be large!)
LOG_SAMPLE_CORRUPTION = True  # Set to True for full per-sample logs


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
        Uses hash of (epoch, sensor, idx, SEED) for reproducibility.
        """
        # Create deterministic random value in [0, 1)
        hash_input = f"{self.epoch}_{sensor}_{idx}_{SEED}"
        hash_val = hash(hash_input)
        rng_val = (hash_val % 10000) / 10000.0
        
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

def main():
    print("="*70)
    print("Multi-FS Multi-Scenario Fusion Training (v3)")
    print("="*70)
    
    # ---- Step 0: Parse corruption configurations ----
    print("\n[0/7] Parsing corruption configurations...")
    
    # Parse training corruption config
    train_sensor_probs, train_scenarios = parse_corruption_config(
        train_corruption_config, sensors
    )
    
    # Parse test corruption config
    test_sensor_probs, test_scenarios = parse_corruption_config(
        test_corruption_config, sensors
    )
    
    # Print configuration summary
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
    
    # ---- Step 1: Load all required embeddings ----
    print("\n[1/7] Loading embeddings...")
    
    # Determine which (sensor, scenario) pairs we actually need
    needed_pairs = set()
    
    # Add from training config
    for sensor in sensors:
        for scenario in train_sensor_probs[sensor].keys():
            if train_sensor_probs[sensor][scenario] > 0:
                needed_pairs.add((sensor, scenario))
    
    # Add from test config
    for sensor in sensors:
        for scenario in test_sensor_probs[sensor].keys():
            if test_sensor_probs[sensor][scenario] > 0:
                needed_pairs.add((sensor, scenario))
    
    print(f"Sensors: {sensors}")
    print(f"Sensor FS: {sensor_fs}")
    print(f"Loading {len(needed_pairs)} sensor-scenario combinations...")
    
    # Load embeddings only for needed pairs
    embeddings_dict = {}
    for sensor, scenario in sorted(needed_pairs):
        fs = sensor_fs[sensors.index(sensor)]
        try:
            data = load_embeddings(sensor, fs, scenario)
            embeddings_dict[(sensor, scenario)] = data
            print(f"  ✓ Loaded {sensor} (fs{fs}) - {scenario}")
        except FileNotFoundError as e:
            print(f"  ✗ Missing: {sensor} (fs{fs}) - {scenario}")
            raise
    
    # ---- Step 2: Verify alignment ----
    print("\n[2/7] Verifying label alignment...")
    verify_label_alignment(embeddings_dict, "train")
    verify_label_alignment(embeddings_dict, "val")
    verify_label_alignment(embeddings_dict, "test")
    
    # Get basic info
    first_key = (sensors[0], "clean")
    embed_dim = embeddings_dict[first_key]["Z_train"].shape[1]
    num_classes = len(np.unique(embeddings_dict[first_key]["y_train"]))
    num_sensors = len(sensors)
    
    n_train = len(embeddings_dict[first_key]["y_train"])
    n_val = len(embeddings_dict[first_key]["y_val"])
    n_test = len(embeddings_dict[first_key]["y_test"])
    
    print(f"\nDataset info:")
    print(f"  Sensors: {num_sensors}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")
    
    # ---- Step 3: Create datasets ----
    print("\n[3/7] Creating datasets...")
    
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
    
    # ---- Step 4: Create model ----
    print("\n[4/7] Creating fusion model...")
    
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
    print("\n[5/7] Training...")
    
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
    
    # ---- Step 6: Test evaluation ----
    print("\n[6/7] Evaluating on test set...")
    
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
    
    # ---- Step 7: Logging and plotting ----
    print("\n[7/7] Logging results...")
    
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
        "experiment": "fusion_v3",
        "sensors": sensors,
        "sensor_fs": sensor_fs,
        "train_corruption_config": train_corruption_config,
        "test_corruption_config": test_corruption_config,
        "train_sensor_probs": train_sensor_probs,
        "test_sensor_probs": test_sensor_probs,
        "fusion_mode": "gated" if USE_GATING else "concat",
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "gate_mean": gate_mean if USE_GATING else None,
        "gate_std": gate_std if USE_GATING else None,
        "seed": SEED,
        "best_val_loss": float(best_val_loss),
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
        
        # Mean bar plot
        means = {k: float(v.mean()) for k, v in gate_log_test.items()}
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(means)), list(means.values()))
        plt.xticks(range(len(means)), list(means.keys()), rotation=45, ha="right")
        plt.ylabel("Average gating weight")
        plt.title("Test Set - Average Sensor Gating Weights")
        plt.tight_layout()
        plot_file = PLOT_DIR / "test_gates_mean.png"
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Saved {plot_file}")
        plt.close()
        
        # Distribution plot
        plt.figure(figsize=(10, 4))
        for k, v in gate_log_test.items():
            plt.hist(v, bins=60, alpha=0.4, label=k)
        plt.xlabel("Gating weight")
        plt.ylabel("Count")
        plt.title("Test Set - Gating Weight Distributions")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plot_file = PLOT_DIR / "test_gates_hist.png"
        plt.savefig(plot_file, dpi=200)
        print(f"  ✓ Saved {plot_file}")
        plt.close()
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
