import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from config import parent_dir, variant_dirs, seq_len, models_output_dir, load_combined_sensor_data, embeddings_base_dir, embeddings_folder_name

# Get script directory for saving plots
script_dir = Path(__file__).parent

# -------------------------------
# Config
# -------------------------------
sensor_name = "Mag_ankle"  # Sensor file name without extension
num_channels = 3

batch_size = 64
epochs = 200
lr = 1e-3

# Early stopping
patience = 10
min_delta = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Parent directory: {parent_dir}")
print(f"Loading combined data from {len(variant_dirs)} variant directories...")

# -------------------------------
# Load combined data from all 3 tremor augmentation folders
# -------------------------------
data = load_combined_sensor_data(f"{sensor_name}.txt")

# Data format (tremor-compatible): 
# [...sensor data...] | [-7] activity | [-6] subject | [-5] base_idx | 
# [-4] tremor_freq | [-3] tremor_acc_rms | [-2] tremor_gyro_rms | [-1] tremor_score
X = data[:, :-7]  # Sensor data only (all columns except last 7)
y = data[:, -7].astype(int) - 1  # Activity label 1..12 -> 0..11

expected_features = num_channels * seq_len
assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
assert y.min() >= 0 and y.max() <= 11, f"Labels out of range: min={y.min()}, max={y.max()}"

X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
num_classes = len(np.unique(y))

# -------------------------------
# Generate train/val/test splits for combined data
# -------------------------------
from sklearn.model_selection import train_test_split

N = X.shape[0]
indices = np.arange(N)

# 70% train, 15% val, 15% test
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

print(f"\nGenerated splits for {N} samples:")
print(f"  Train: {len(train_idx)} ({len(train_idx)/N*100:.1f}%)")
print(f"  Val:   {len(val_idx)} ({len(val_idx)/N*100:.1f}%)")
print(f"  Test:  {len(test_idx)} ({len(test_idx)/N*100:.1f}%)")

# Save splits for consistency across sensors
split_save_dir = script_dir / "splits"
split_save_dir.mkdir(exist_ok=True)

if not (split_save_dir / "train_idx.txt").exists():
    np.savetxt(split_save_dir / "train_idx.txt", train_idx, fmt='%d')
    np.savetxt(split_save_dir / "val_idx.txt", val_idx, fmt='%d')
    np.savetxt(split_save_dir / "test_idx.txt", test_idx, fmt='%d')
    print(f"Saved split indices to: {split_save_dir}")
else:
    # Load existing splits to ensure consistency across sensors
    train_idx = np.loadtxt(split_save_dir / "train_idx.txt", dtype=int)
    val_idx = np.loadtxt(split_save_dir / "val_idx.txt", dtype=int)
    test_idx = np.loadtxt(split_save_dir / "test_idx.txt", dtype=int)
    print(f"Loaded existing split indices from: {split_save_dir}")

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print("Split sizes:", len(train_idx), len(val_idx), len(test_idx))

# Torch tensors (explicit dtypes, like your first script)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size, shuffle=False)

# ---- Non-shuffled loaders for embedding extraction (important for fusion alignment) ----
train_loader_feat = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
val_loader_feat   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=batch_size, shuffle=False)
test_loader_feat  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size, shuffle=False)


# -------------------------------
# CNN Feature Extractor Model
# -------------------------------
class IMUCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, num_channels: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # seq_len=50 -> 50//2=25 -> 25//2=12 -> 12 * 128 = 1536
        self.flattened_dim = (seq_len // 4) * 128

        # Embedding head (128-dim)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)

        # Classification head
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        """Return embedding BEFORE dropout (batch, 128)."""
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc_embed(x)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        logits = self.fc_cls(z)
        return logits

model = IMUCNN(num_classes=num_classes, seq_len=seq_len, num_channels=num_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# Training + Early stopping (with Train+Val loss+acc print)
# -------------------------------
best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(epochs):
    # ---- Train ----
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += xb.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ---- Validate ----
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:3d}/{epochs} | "
          f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

    # ---- Early stopping check (on val loss) ----
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

# -------------------------------
# Save feature extractor model
# -------------------------------
save_path = models_output_dir / f"feature_extractor_{sensor_name}.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": num_classes,
    "seq_len": seq_len,
    "num_channels": num_channels,
    "embedding_dim": model.fc_embed.out_features
}, save_path)

print(f"Feature extractor saved to:\n{save_path.resolve()}")

# -------------------------------
# Evaluation on test set
# -------------------------------
model.eval()
y_pred = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

test_acc = accuracy_score(y_test, y_pred)

# -------------------------------
# Save accuracy to history
# -------------------------------
model_key = f"{Path(__file__).parent.name}/{sensor_name}"
history_file = Path(__file__).parent.parent.parent.parent / "accuracy_history.json"

# Load and update accuracy history
if history_file.exists():
    with open(history_file, 'r') as f:
        history = json.load(f)
else:
    history = {}

if model_key not in history:
    history[model_key] = []

history[model_key].append(test_acc)

with open(history_file, 'w') as f:
    json.dump(history, f, indent=2, sort_keys=True)

# -------------------------------
# Check if this is a new best accuracy
# -------------------------------
best_acc_file = Path(__file__).parent.parent.parent.parent / "best_accuracies.json"

# Load existing best accuracies
if best_acc_file.exists():
    with open(best_acc_file, 'r') as f:
        best_accs = json.load(f)
else:
    best_accs = {}

previous_best = best_accs.get(model_key, 0.0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Previous Best: {previous_best:.4f}")

if test_acc <= previous_best:
    print(f"❌ No improvement. Skipping save operations.")
    sys.exit(0)

print(f"✅ New best accuracy! Saving model artifacts...")

# Update best accuracies file
best_accs[model_key] = test_acc
with open(best_acc_file, 'w') as f:
    json.dump(best_accs, f, indent=2, sort_keys=True)

# -------------------------------
# Save best model (same folder as this script)
# -------------------------------
save_path = models_output_dir / f"feature_extractor_{sensor_name}.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": num_classes,
    "seq_len": seq_len,
    "num_channels": num_channels,
    "embedding_dim": model.fc_embed.out_features
}, save_path)

print(f"Feature extractor saved to:\n{save_path.resolve()}")

# -------------------------------
# Find and save misclassified samples
# -------------------------------
misclassified_indices = []
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
    if true_label != pred_label:
        misclassified_indices.append({
            'test_index': int(test_idx[i]),
            'true_label': int(true_label + 1),
            'predicted_label': int(pred_label + 1)
        })

if misclassified_indices:
    import pandas as pd
    df_misclass = pd.DataFrame(misclassified_indices)
    
    misclass_dir = script_dir / "Misclassified"
    misclass_dir.mkdir(parents=True, exist_ok=True)
    
    misclass_file = misclass_dir / f"{sensor_name}_misclassified.csv"
    df_misclass.to_csv(misclass_file, index=False)
    
    print(f"\nMisclassified samples: {len(misclassified_indices)}/{len(y_test)}")
    print(f"Saved to: {misclass_file}")
    
    # Show summary of most common misclassifications
    print("\nMost common misclassifications:")
    misclass_pairs = df_misclass.groupby(['true_label', 'predicted_label']).size()
    print(misclass_pairs.sort_values(ascending=False).head(10))

# -------------------------------
# Confusion Matrices (Counts + Normalized)
# -------------------------------
labels_display = list(range(1, 13))

# Create output directory for confusion matrices
current_folder = Path(__file__).parent.name  # e.g., "fs50_s0.5_w2_aug2"
cm_output_dir = Path("/Users/linaandersson/Desktop/master/Confusion_Matrixes") / current_folder
cm_output_dir.mkdir(parents=True, exist_ok=True)

# Confusion Matrix (Counts)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_display)
disp.plot(cmap="viridis")
plt.title("Confusion Matrix (Counts)")
cm_file = cm_output_dir / f"{sensor_name}_matrix.png"
plt.savefig(cm_file, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_file}")
plt.show()

# Confusion Matrix (Normalized)
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels_display)
disp_norm.plot(cmap="viridis", values_format=".2f")
plt.title("Confusion Matrix (Normalized per Class)")
cm_norm_file = cm_output_dir / f"{sensor_name}_norm_matrix.png"
plt.savefig(cm_norm_file, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix saved to: {cm_norm_file}")
plt.show()

# -------------------------------
# Extract embeddings (features) and save to NPZ
# -------------------------------
model.eval()
def extract_embeddings(loader):
    feats = []
    labs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            z = model.extract_features(xb)         # (batch, 32)
            feats.append(z.cpu().numpy())
            labs.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)

# Save embeddings for each variant in their respective folders
print(f"\n📂 Extracting and saving embeddings for each variant...")

for variant_dir in variant_dirs:
    variant_name = variant_dir.name
    
    # Load data from this specific variant only
    variant_file = variant_dir / f"{sensor_name}.txt"
    if not variant_file.exists():
        print(f"   ⚠️  Skipping {variant_name}: file not found")
        continue
    
    # Convert to string to avoid numpy path resolution issues
    variant_data = np.loadtxt(str(variant_file.resolve()), delimiter=",")
    X_variant = variant_data[:, :-7]  # Sensor data
    activities_variant = variant_data[:, -7].astype(int) - 1  # Activity (0-11)
    subjects_variant = variant_data[:, -6].astype(int)  # Subject ID
    tremor_labels_variant = variant_data[:, -1]  # Tremor score (last column)
    
    X_variant = X_variant.reshape(-1, num_channels, seq_len).astype(np.float32)
    
    # Create DataLoader (no shuffle to maintain order)
    X_variant_t = torch.tensor(X_variant, dtype=torch.float32)
    y_variant_t = torch.tensor(activities_variant, dtype=torch.long)
    variant_loader = DataLoader(TensorDataset(X_variant_t, y_variant_t), batch_size=batch_size, shuffle=False)
    
    # Extract ALL embeddings from this variant
    Z_variant_all, _ = extract_embeddings(variant_loader)
    
    # Generate or load train/val/test splits FOR THIS VARIANT
    variant_split_dir = variant_dir / "splits"
    variant_split_dir.mkdir(exist_ok=True)
    
    N_variant = len(Z_variant_all)
    
    if not (variant_split_dir / "train_idx.txt").exists():
        # Generate splits for this variant (70/15/15 ratio)
        indices_variant = np.arange(N_variant)
        train_idx_var, temp_idx_var = train_test_split(
            indices_variant, test_size=0.3, random_state=42, stratify=activities_variant
        )
        val_idx_var, test_idx_var = train_test_split(
            temp_idx_var, test_size=0.5, random_state=42, stratify=activities_variant[temp_idx_var]
        )
        
        np.savetxt(variant_split_dir / "train_idx.txt", train_idx_var, fmt='%d')
        np.savetxt(variant_split_dir / "val_idx.txt", val_idx_var, fmt='%d')
        np.savetxt(variant_split_dir / "test_idx.txt", test_idx_var, fmt='%d')
        print(f"   Generated splits for {variant_name}: {len(train_idx_var)}/{len(val_idx_var)}/{len(test_idx_var)}")
    else:
        # Load existing splits to ensure consistency across sensors
        train_idx_var = np.loadtxt(variant_split_dir / "train_idx.txt", dtype=int)
        val_idx_var = np.loadtxt(variant_split_dir / "val_idx.txt", dtype=int)
        test_idx_var = np.loadtxt(variant_split_dir / "test_idx.txt", dtype=int)
        print(f"   Loaded splits for {variant_name}: {len(train_idx_var)}/{len(val_idx_var)}/{len(test_idx_var)}")
    
    # Apply splits to this variant's data
    Z_variant_train = Z_variant_all[train_idx_var]
    Z_variant_val = Z_variant_all[val_idx_var]
    Z_variant_test = Z_variant_all[test_idx_var]
    
    activities_train = activities_variant[train_idx_var]
    activities_val = activities_variant[val_idx_var]
    activities_test = activities_variant[test_idx_var]
    
    subjects_train = subjects_variant[train_idx_var]
    subjects_val = subjects_variant[val_idx_var]
    subjects_test = subjects_variant[test_idx_var]
    
    tremor_train = tremor_labels_variant[train_idx_var]
    tremor_val = tremor_labels_variant[val_idx_var]
    tremor_test = tremor_labels_variant[test_idx_var]
    
    # Save in variant's embeddings folder with fusion-compatible format (using config settings)
    # Get corresponding variant directory in embeddings_base_dir
    embeddings_variant_dir = embeddings_base_dir / variant_name
    variant_feat_dir = embeddings_variant_dir / embeddings_folder_name
    variant_feat_dir.mkdir(parents=True, exist_ok=True)
    variant_feat_path = variant_feat_dir / f"{sensor_name}_embeddings.npz"
    
    np.savez_compressed(
        variant_feat_path,
        # Embeddings
        train_embeddings=Z_variant_train,
        val_embeddings=Z_variant_val,
        test_embeddings=Z_variant_test,
        # Tremor labels (for fusion classification)
        train_labels=tremor_train,
        val_labels=tremor_val,
        test_labels=tremor_test,
        # Activity labels
        train_activities=activities_train,
        val_activities=activities_val,
        test_activities=activities_test,
        # Subject IDs
        train_subjects=subjects_train,
        val_subjects=subjects_val,
        test_subjects=subjects_test,
    )
    
    print(f"   ✅ {variant_name}: train={Z_variant_train.shape}, val={Z_variant_val.shape}, test={Z_variant_test.shape} → {variant_feat_path}")