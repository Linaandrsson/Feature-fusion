import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from datetime import datetime
from config import (
    parent_dir,
    variant_dirs,
    seq_len,
    models_output_dir,
    load_combined_sensor_data,
    embeddings_base_dir,
    embeddings_folder_name,
    num_activity_classes,
    activity_label_min,
    build_source_aware_subject_splits,
    split_file_prefix,
    confusion_matrices_dir,
    MANUAL_SUBJECT_SPLITS,
)

# Get script directory for saving plots
script_dir = Path(__file__).parent

# -------------------------------
# Config
# -------------------------------
sensor_name = "Acc_arm"  # Sensor file name without extension
num_channels = 3

batch_size = 64
epochs = 200
lr = 1e-3

# Early stopping
patience = 10
min_delta = 1e-4

# Set random seed for reproducibility
random_seed = np.random.randint(0, 100000)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Parent directory: {parent_dir}")
print(f"Loading combined data from {len(variant_dirs)} variant directories...")

# -------------------------------
# Load combined data from all configured variant folders
# -------------------------------
data = load_combined_sensor_data(f"{sensor_name}.txt")

# Data format (tremor-compatible): 
# [...sensor data...] | [-7] activity | [-6] subject | [-5] base_idx | 
# [-4] tremor_freq | [-3] tremor_acc_rms | [-2] tremor_gyro_rms | [-1] tremor_score
X = data[:, :-7]  # Sensor data only (all columns except last 7)
y = data[:, -7].astype(int) - activity_label_min  # Activity label 1..16 -> 0..15

expected_features = num_channels * seq_len
assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
assert y.min() >= 0 and y.max() <= (num_activity_classes - 1), f"Labels out of range: min={y.min()}, max={y.max()}"

X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
num_classes = num_activity_classes
print(f"Detected classes in data: {sorted(np.unique(y).tolist())} / expected 0..{num_activity_classes - 1}")

# -------------------------------
# Generate train/val/test splits (SUBJECT-BASED, source-aware)
# -------------------------------
# Extract subject IDs from combined data
subjects = data[:, -6].astype(int)  # Subject ID is in column -6

# Use manual subject splits if defined, otherwise auto-generate
if MANUAL_SUBJECT_SPLITS is not None:
    print("Using manual subject splits from config...")
    train_subjects = MANUAL_SUBJECT_SPLITS.get("train", [])
    val_subjects = MANUAL_SUBJECT_SPLITS.get("val", [])
    test_subjects = MANUAL_SUBJECT_SPLITS.get("test", [])
else:
    # Build deterministic source-aware held-out subject sets.
    train_subjects, val_subjects, test_subjects = build_source_aware_subject_splits(subjects, seed=42)

# Create splits based on subjects
train_idx = np.where(np.isin(subjects, train_subjects))[0]
val_idx = np.where(np.isin(subjects, val_subjects))[0]
test_idx = np.where(np.isin(subjects, test_subjects))[0]

N = X.shape[0]
print(f"\nSubject-based splits for {N} samples:")
print(f"  Train: {len(train_idx)} ({len(train_idx)/N*100:.1f}%) - Subjects: {sorted(np.unique(subjects[train_idx]))}")
print(f"  Val:   {len(val_idx)} ({len(val_idx)/N*100:.1f}%) - Subjects: {val_subjects}")
print(f"  Test:  {len(test_idx)} ({len(test_idx)/N*100:.1f}%) - Subjects: {test_subjects}")

# Save splits for consistency across sensors
split_save_dir = script_dir / "splits"
split_save_dir.mkdir(exist_ok=True)

train_idx_file = split_save_dir / f"{split_file_prefix}_train_idx.txt"
val_idx_file = split_save_dir / f"{split_file_prefix}_val_idx.txt"
test_idx_file = split_save_dir / f"{split_file_prefix}_test_idx.txt"

if not train_idx_file.exists():
    np.savetxt(train_idx_file, train_idx, fmt='%d')
    np.savetxt(val_idx_file, val_idx, fmt='%d')
    np.savetxt(test_idx_file, test_idx, fmt='%d')
    print(f"Saved split indices to: {split_save_dir}")
else:
    # Load existing splits to ensure consistency across sensors
    train_idx = np.loadtxt(train_idx_file, dtype=int)
    val_idx = np.loadtxt(val_idx_file, dtype=int)
    test_idx = np.loadtxt(test_idx_file, dtype=int)
    print(f"Loaded existing split indices from: {split_save_dir} ({split_file_prefix})")

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print("Split sizes:", len(train_idx), len(val_idx), len(test_idx))

# Torch tensors (explicit dtypes)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

# ---- Non-shuffled loaders for embedding extraction (important for fusion alignment) ----
train_loader_feat = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
val_loader_feat   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=batch_size, shuffle=False)
test_loader_feat  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size, shuffle=False)


# -------------------------------
# CNN Model (DO NOT CHANGE MODEL)
# -------------------------------
class IMUCNN(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        self.flattened_dim = (seq_len // 4) * 256

        # Embedding layer (feature representation)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)

        # Classification head
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc_embed(x)
        return z

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        return self.fc_cls(z)

model = IMUCNN(num_classes, seq_len, num_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# Training + Early stopping (based on macro F1)
# -------------------------------
best_val_macro_f1 = 0.0
best_val_weighted_f1 = 0.0
best_val_loss = float('inf')
best_val_acc = 0.0
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
    val_preds_all = []
    val_labels_all = []

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
            
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(yb.cpu().numpy())

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total
    val_macro_f1 = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)
    val_weighted_f1 = f1_score(val_labels_all, val_preds_all, average='weighted', zero_division=0)

    print(f"Epoch {epoch+1:3d}/{epochs} | "
          f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"Val loss {val_loss:.4f} acc {val_acc:.4f} "
          f"f1_macro {val_macro_f1:.4f} f1_weighted {val_weighted_f1:.4f}")

    # ---- Early stopping check (on val macro F1) ----
    if val_macro_f1 > best_val_macro_f1 + min_delta:
        best_val_macro_f1 = val_macro_f1
        best_val_weighted_f1 = val_weighted_f1
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Best val macro F1: {best_val_macro_f1:.4f}")
            break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

# -------------------------------
# Evaluation on test set (same as ankle)
# -------------------------------
model.eval()
y_pred = []
test_loss_sum = 0.0
test_total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        
        test_loss_sum += loss.item() * xb.size(0)
        test_total += xb.size(0)
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

test_acc = accuracy_score(y_test, y_pred)
test_macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
test_weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
test_loss = test_loss_sum / test_total

# -------------------------------
# Calculate model parameters
# -------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------
# Check if this is a new best score (macro F1 selection metric)
# -------------------------------
model_key = f"{Path(__file__).parent.name}/{sensor_name}"
best_models_file = script_dir / "best_models.json"

# Load existing best models
if best_models_file.exists():
    with open(best_models_file, 'r') as f:
        best_models = json.load(f)
else:
    best_models = {}

previous_best_macro_f1 = best_models.get(model_key, {}).get("macro_f1", -1.0)
print(f"\nTest Macro F1: {test_macro_f1:.4f} (Weighted F1: {test_weighted_f1:.4f}, Accuracy: {test_acc:.4f})")
if previous_best_macro_f1 < 0:
    print("Previous Best Macro F1: not set")
else:
    print(f"Previous Best Macro F1: {previous_best_macro_f1:.4f}")

if test_macro_f1 <= previous_best_macro_f1:
    print(f"❌ No improvement. Skipping save operations.")
    sys.exit(0)

print(f"✅ New best Macro F1! Saving model artifacts...")

# Update best models file with all metrics
best_models[model_key] = {
    "selection_metric": "macro_f1",
    "macro_f1": test_macro_f1,
    "weighted_f1": test_weighted_f1,
    "f1": test_macro_f1,
    "accuracy": test_acc,
    "test_loss": test_loss,
    "val_macro_f1": best_val_macro_f1,
    "val_weighted_f1": best_val_weighted_f1,
    "val_f1": best_val_macro_f1,
    "val_acc": best_val_acc,
    "val_loss": best_val_loss,
    "random_seed": random_seed,
    "total_params": total_params,
    "trainable_params": trainable_params,
    "architecture": str(model),
    "timestamp": datetime.now().isoformat()
}

with open(best_models_file, 'w') as f:
    json.dump(best_models, f, indent=2, sort_keys=True)

print(f"Saved best model info to: {best_models_file}")

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
labels_display = list(range(activity_label_min, activity_label_min + num_classes))

# Create output directory for confusion matrices
cm_output_dir = confusion_matrices_dir
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
    activities_variant = variant_data[:, -7].astype(int) - activity_label_min  # Activity (0-15)
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
        # Generate SUBJECT-BASED splits for this variant using MANUAL_SUBJECT_SPLITS from config
        # This ensures consistent subject-level splits across all variants and sensors
        if MANUAL_SUBJECT_SPLITS is not None:
            # Use manual subject splits (subject-based, not random)
            train_mask = np.isin(subjects_variant, MANUAL_SUBJECT_SPLITS["train"])
            val_mask = np.isin(subjects_variant, MANUAL_SUBJECT_SPLITS["val"])
            test_mask = np.isin(subjects_variant, MANUAL_SUBJECT_SPLITS["test"])
            
            train_idx_var = np.where(train_mask)[0]
            val_idx_var = np.where(val_mask)[0]
            test_idx_var = np.where(test_mask)[0]
            
            split_type = "subject-based (manual)"
        else:
            # Fallback: use source-aware random splits if manual splits not defined
            train_idx_var, val_idx_var, test_idx_var = build_source_aware_subject_splits(
                subjects_variant, seed=42
            )
            split_type = "source-aware subject-based"
        
        np.savetxt(variant_split_dir / "train_idx.txt", train_idx_var, fmt='%d')
        np.savetxt(variant_split_dir / "val_idx.txt", val_idx_var, fmt='%d')
        np.savetxt(variant_split_dir / "test_idx.txt", test_idx_var, fmt='%d')
        print(f"   Generated {split_type} splits for {variant_name}: {len(train_idx_var)}/{len(val_idx_var)}/{len(test_idx_var)}")
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
