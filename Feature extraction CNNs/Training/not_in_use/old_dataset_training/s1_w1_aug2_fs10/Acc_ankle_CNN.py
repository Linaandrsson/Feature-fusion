import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from config import parent_dir, variant_dir, seq_len, models_output_dir

# Get script directory for saving plots
script_dir = Path(__file__).parent

# -------------------------------
# Config
# -------------------------------
file_path = variant_dir / "Acc_ankle.txt"
split_dir = parent_dir  # Splits are in parent directory

num_channels = 3

batch_size = 64
epochs = 200
lr = 1e-3

# Early stopping
patience = 10
min_delta = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Parent directory (splits): {parent_dir}")
print(f"Variant directory (data): {variant_dir}")

# -------------------------------
# Load data
# -------------------------------
if not file_path.exists():
    raise FileNotFoundError(f"Sensor data not found: {file_path}")

data = np.loadtxt(file_path, delimiter=",")

X = data[:, :-1]
y = data[:, -1].astype(int) - 1  # 1..12 -> 0..11

expected_features = num_channels * seq_len
assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
assert y.min() >= 0 and y.max() <= 11, f"Labels out of range: min={y.min()}, max={y.max()}"

X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
num_classes = len(np.unique(y))

# -------------------------------
# Load fixed split indices
# -------------------------------
train_idx = np.loadtxt(split_dir / "train_idx.txt", dtype=int)
val_idx   = np.loadtxt(split_dir / "val_idx.txt", dtype=int)
test_idx  = np.loadtxt(split_dir / "test_idx.txt", dtype=int)

N = X.shape[0]
assert train_idx.max() < N and val_idx.max() < N and test_idx.max() < N, "Split indices out of range!"
assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test overlap!"
assert len(set(val_idx) & set(test_idx)) == 0, "Val/Test overlap!"

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print("Split sizes:", len(train_idx), len(val_idx), len(test_idx))

# Torch tensors
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
# CNN Model (same architecture as yours)
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
            #nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        self.flattened_dim = (seq_len // 2) * 128 # 1536 numbers per sample

        # Embedding head (128-dim)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)

        # Classification head (kept for training/evaluation)
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        # Return embedding BEFORE dropout (stable for inference)
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc_embed(x)  
        return z

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        return self.fc_cls(z)


model = IMUCNN(num_classes=num_classes, seq_len=seq_len, num_channels=num_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# Training + Early stopping
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
# Save best model to Models/ directory
# -------------------------------
sensor_name = Path(file_path).stem  # e.g. "Acc_ankle"

save_path = models_output_dir / f"feature_extractor_{sensor_name}.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": num_classes,
    "seq_len": seq_len,
    "num_channels": num_channels,
    "embedding_dim": model.fc_embed.out_features
}, save_path)

print(f"Best model saved to:\n{save_path.resolve()}")



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
model_key = f"{Path(__file__).parent.name}/{Path(file_path).stem}"
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
# Save best model to Models/ directory
# -------------------------------
sensor_name = Path(file_path).stem  # "Acc_ankle"

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

# Get sensor name from file path
sensor_name = Path(file_path).stem  # e.g., "Acc_arm"

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
# NEW: Extract embeddings (features) and save to NPZ
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

train_Z, train_y = extract_embeddings(train_loader_feat)
val_Z,   val_y   = extract_embeddings(val_loader_feat)
test_Z,  test_y  = extract_embeddings(test_loader_feat)

# feat_dir = Path("ExtractedFeatures")
# feat_dir.mkdir(parents=True, exist_ok=True)
# feat_path = feat_dir / f"{sensor_name}_embeddings.npz"

feat_dir = variant_dir / "ExtractedFeatures"
feat_dir.mkdir(parents=True, exist_ok=True)
feat_path = feat_dir / f"{sensor_name}_embeddings.npz"  


np.savez_compressed(
    feat_path,
    Z_train=train_Z, y_train=train_y,
    Z_val=val_Z,     y_val=val_y,
    Z_test=test_Z,   y_test=test_y
)

print(f"Saved embeddings to {feat_path}")
print("Embedding shapes:",
      train_Z.shape, val_Z.shape, test_Z.shape)
