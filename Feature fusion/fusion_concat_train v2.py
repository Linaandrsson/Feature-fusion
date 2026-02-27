"""
fusion_concat_train.py  (FULL REPLACEMENT)

Feature-level fusion training with OPTIONAL dynamic gating weights.

What this script does:
- Loads per-sensor embeddings from *_embeddings.npz files (Z_train/Z_val/Z_test, y_*)
- Builds a fusion classifier:
    Option A (baseline): concat([z1,z2,...]) -> MLP head
    Option B (gating):  compute per-sample sensor weights alpha via small gate nets,
                        fuse via weighted sum -> MLP head
- Trains with early stopping
- Evaluates on test set
- Logs and plots gating weights (if gating enabled)

You do NOT need to change any CNN feature extractors for this.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import random
import time
from datetime import datetime


# -------------------------------
# Reproducibility - Random seed for each run
# -------------------------------
SEED = int(time.time() * 1000) % 100000  # Random seed based on timestamp
print(f"Using random seed: {SEED}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For PyTorch 2.x (optional but good)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# -------------------------------
# Config
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder where your embeddings live (npz files)
#feat_dir = Path("//Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/Feature extraction CNNs/Feature extractors/fs50_s1_w1_aug2/ExtractedFeatures")

feat_dir = Path("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files/fs50_s1_w1_aug2_N50/ExtractedFeatures")
# Sensors to fuse (names must match your npz files)
sensors = ["Acc_ankle", "Acc_arm", "Acc_chest"]

chest_idx = sensors.index("Acc_chest")


# Training
batch_size = 128
epochs = 200
lr = 1e-3
patience = 30
min_delta = 1e-4


gate_type="sigmoid"
gate_hidden=0
gate_dropout=0.1
alpha_floor=0.05


# Fusion mode
USE_GATING = True  # True = dynamic gating weights, False = concat baseline

# Classifier head dims
head_hidden_dims = (64, 64)
head_dropout = 0.4

# Logging/plots
SAVE_PLOTS = True
PLOT_DIR = Path("gating_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


LOG_DIR = Path("fusion_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "gating_experiments.jsonl"


# -------------------------------
# Helpers: data loading
# -------------------------------
def load_sensor_npz(sensor_name: str):
    p = feat_dir / f"{sensor_name}_embeddings.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing embedding file: {p}")
    d = np.load(p)
    return d["Z_train"], d["y_train"], d["Z_val"], d["y_val"], d["Z_test"], d["y_test"]


# -------------------------------
# Load per-sensor embeddings (KEEP SEPARATE, do not concat yet)
# -------------------------------
Zt_list, Zv_list, Zte_list = [], [], []
y_train = y_val = y_test = None

for s in sensors:
    Z_train, y_tr, Z_val, y_v, Z_test, y_te = load_sensor_npz(s)
    Zt_list.append(Z_train.astype(np.float32))
    Zv_list.append(Z_val.astype(np.float32))
    Zte_list.append(Z_test.astype(np.float32))

    # Labels should match across sensors (same split ordering), keep first
    if y_train is None:
        y_train, y_val, y_test = y_tr.astype(int), y_v.astype(int), y_te.astype(int)

# Sanity checks
num_sensors = len(sensors)
embed_dim = Zt_list[0].shape[1]
for i, Z in enumerate(Zt_list):
    assert Z.shape[1] == embed_dim, f"Embedding dim mismatch for {sensors[i]}: {Z.shape[1]} vs {embed_dim}"

num_classes = len(np.unique(y_train))
print("Sensors:", sensors)
print("Num sensors:", num_sensors)
print("Embedding dim per sensor:", embed_dim)
print("Num classes:", num_classes)
print("Train/Val/Test shapes (per sensor):", Zt_list[0].shape, Zv_list[0].shape, Zte_list[0].shape)

# -------------------------------
# Build DataLoaders
# We store per-sensor tensors separately in the dataset.
# Each batch from loader will be: (z1, z2, ..., zS, y)
# -------------------------------
def make_loader(Z_list, y, shuffle: bool):
    tensors = [torch.tensor(Z, dtype=torch.float32) for Z in Z_list]
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(*tensors, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(Zt_list, y_train, shuffle=True)
val_loader   = make_loader(Zv_list, y_val, shuffle=False)
test_loader  = make_loader(Zte_list, y_test, shuffle=False)

# -------------------------------
# Models
# -------------------------------
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
    """
    Small gating network: embedding -> scalar score.
    Slightly stronger than pure linear (optional), but still tiny.
    """
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
        return self.net(z)  # (B,1)


class GatedConcatFusionModel(nn.Module):
    """
    Gating WITHOUT bottleneck:
    - Compute per-sensor scalar weights alpha_s per sample
    - Gate each embedding block: alpha_s * z_s
    - Concatenate gated blocks -> MLP head (same input dim as baseline concat)

    Options:
      gate_type="sigmoid": independent gating per sensor (recommended)
      gate_type="softmax": competition across sensors (sum=1)
    """
    def __init__(
        self,
        embed_dim: int,
        num_sensors: int,
        num_classes: int,
        head_hidden_dims=(128, 128),
        head_dropout=0.3,
        gate_type: str = "sigmoid",   # "sigmoid" or "softmax"
        gate_hidden: int = 0,         # 0 = linear gate, >0 = tiny MLP gate
        gate_dropout: float = 0.0,    # dropout on embeddings before gate (optional)
        use_layernorm: bool = True,   # LayerNorm per sensor embedding
        alpha_floor: float = 0.0      # e.g. 0.05 to avoid gating to ~0
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.embed_dim = embed_dim
        self.gate_type = gate_type.lower()
        assert self.gate_type in ("sigmoid", "softmax"), "gate_type must be 'sigmoid' or 'softmax'"
        self.alpha_floor = alpha_floor

        # Optional per-sensor LayerNorm to align scales across sensors
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_sensors)]) if use_layernorm else None

        # Optional dropout before gate
        self.gate_in_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else None

        # Gates
        self.gates = nn.ModuleList([SensorGate(embed_dim, hidden=gate_hidden) for _ in range(num_sensors)])

        # Head sees full concat dim (same as baseline)
        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes,
                              hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        # Z_list: list length S, each (B, D)
        Z_proc = []
        for i, z in enumerate(Z_list):
            if self.norms is not None:
                z = self.norms[i](z)
            if self.gate_in_dropout is not None:
                z = self.gate_in_dropout(z)
            Z_proc.append(z)

        scores = [gate(z) for gate, z in zip(self.gates, Z_proc)]  # list of (B,1)
        scores = torch.cat(scores, dim=1)                          # (B,S)

        if self.gate_type == "softmax":
            alphas = torch.softmax(scores, dim=1)                  # (B,S), sum=1
        else:
            alphas = torch.sigmoid(scores)                         # (B,S), independent

        # Optional: prevent alphas from collapsing to 0 (helps stability)
        if self.alpha_floor > 0:
            alphas = self.alpha_floor + (1.0 - self.alpha_floor) * alphas

        # Gate each block, then concat (no bottleneck)
        gated_blocks = [alphas[:, i:i+1] * Z_proc[i] for i in range(self.num_sensors)]  # each (B,D)
        z_cat = torch.cat(gated_blocks, dim=1)  # (B, S*D)
        logits = self.head(z_cat)
        return logits, alphas



class ConcatFusionModel(nn.Module):
    """
    Baseline: concat embeddings then classifier head.
    """
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


# Instantiate model
if USE_GATING:
    model = GatedConcatFusionModel(
        embed_dim=embed_dim,
        num_sensors=num_sensors,
        num_classes=num_classes,
        head_hidden_dims=head_hidden_dims,
        head_dropout=head_dropout,

        gate_type="sigmoid",     # <-- start med "sigmoid" (ofte best)
        gate_hidden=64,          # <-- liten MLP gate (prøv 0 eller 64)
        gate_dropout=0.1,        # <-- litt regularisering på gate
        use_layernorm=True,      # <-- stabiliserer sensorskala
        alpha_floor=0.05         # <-- hindrer gate i å bli helt 0
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

print("\nFusion mode:", "GATED" if USE_GATING else "CONCAT")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Get model architecture as string for logging
model_architecture = str(model)

# -------------------------------
# Training + Early stopping
# -------------------------------
best_val_loss = float("inf")
best_state = None
no_improve = 0

def unpack_batch(batch):
    # batch = (z1, z2, ..., zS, y)
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
        Z, yb = unpack_batch(batch)

        optimizer.zero_grad()
        logits, _ = model(Z)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * yb.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ---- Validate ----
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    # log gating weights on val (if enabled)
    gate_log = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            Z, yb = unpack_batch(batch)
            logits, alphas = model(Z)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)

            if USE_GATING and alphas is not None:
                for i in range(num_sensors):
                    gate_log[f"{sensors[i]}"].append(alphas[:, i].cpu().numpy())

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

# -------------------------------
# Evaluation on test set
# -------------------------------
model.eval()
y_pred = []
gate_log_test = defaultdict(list)

with torch.no_grad():
    for batch in test_loader:
        Z, _yb = unpack_batch(batch)

        # --- Ablation: remove Acc_chest contribution ---
        Z[chest_idx] = torch.zeros_like(Z[chest_idx])

        logits, alphas = model(Z)
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

        if USE_GATING and alphas is not None:
            for i in range(num_sensors):
                gate_log_test[f"{sensors[i]}"].append(alphas[:, i].cpu().numpy())

test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -------------------------------
# Gating weights logging (test set)
# -------------------------------


gate_mean = {}
gate_std = {}

if USE_GATING:
    for s in sensors:
        vals = np.concatenate(gate_log_test[s], axis=0)
        gate_mean[s] = float(vals.mean())
        gate_std[s] = float(vals.std())

run_log = {
    "val_accuracy": float(val_acc),
    "test_accuracy": float(test_acc),
    "gate_mean": gate_mean if USE_GATING else None,
    "gate_std": gate_std if USE_GATING else None,
    "seed": SEED,
}

try:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(run_log) + "\n")
    print(f"Logged run to {LOG_FILE}")
except (TimeoutError, OSError) as e:
    print(f"Warning: Could not write to {LOG_FILE}: {e}")
    print(f"Run results: {run_log}")


# -------------------------------
# Plot gating weights (test set)
# -------------------------------
def plot_gates(gate_log_dict, title_prefix: str, out_dir: Path):
    # Concatenate lists
    for k in list(gate_log_dict.keys()):
        gate_log_dict[k] = np.concatenate(gate_log_dict[k], axis=0)

    # Means bar plot
    means = {k: float(v.mean()) for k, v in gate_log_dict.items()}
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(means)), list(means.values()))
    plt.xticks(range(len(means)), list(means.keys()), rotation=45, ha="right")
    plt.ylabel("Average gating weight")
    plt.title(f"{title_prefix} - Average sensor gating weights")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_gates_mean.png", dpi=200)
   #plt.show()

    # Distribution plot
    plt.figure(figsize=(10, 4))
    for k, v in gate_log_dict.items():
        plt.hist(v, bins=60, alpha=0.4, label=k)
    plt.xlabel("Gating weight")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} - Gating weight distributions")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_gates_hist.png", dpi=200)
    #plt.show()

if USE_GATING:
    plot_gates(gate_log_test, "Test", PLOT_DIR)

# -------------------------------
# Save best accuracy history (optional)
# -------------------------------
# This keeps the same spirit as your other scripts, but avoids hardcoded absolute paths.
history_file = Path("accuracy_history.json")
best_acc_file = Path("best_accuracies.json")
model_key = f"fusion_{'gated' if USE_GATING else 'concat'}_" + "_".join(sensors)

# Load history
if history_file.exists():
    with open(history_file, "r") as f:
        history = json.load(f)
else:
    history = {}

history.setdefault(model_key, [])
history[model_key].append(float(test_acc))

with open(history_file, "w") as f:
    json.dump(history, f, indent=2, sort_keys=True)

# Best accuracies
if best_acc_file.exists():
    with open(best_acc_file, "r") as f:
        best_accs = json.load(f)
else:
    best_accs = {}

# Handle old format (just accuracy) vs new format (dict with accuracy and architecture)
if model_key in best_accs:
    if isinstance(best_accs[model_key], dict):
        previous_best = best_accs[model_key].get("accuracy", 0.0)
    else:
        # Old format: just a number
        previous_best = best_accs[model_key]
else:
    previous_best = 0.0

print(f"Previous Best: {previous_best:.4f}")

if test_acc > previous_best:
    # Update best accuracies file with accuracy AND architecture
    best_accs[model_key] = {
        "accuracy": float(test_acc),
        "architecture": model_architecture,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "fusion_mode": "gated" if USE_GATING else "concat",
        "random_seed": SEED,
        "timestamp": datetime.now().isoformat()
    }
    with open(best_acc_file, "w") as f:
        json.dump(best_accs, f, indent=2, sort_keys=True)
    print("✅ New best accuracy saved.")
else:
    print("❌ No improvement over previous best.")

print("\nDone.")
