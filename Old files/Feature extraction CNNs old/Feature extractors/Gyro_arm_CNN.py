import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from pathlib import Path

# -------------------------------
# Config
# -------------------------------
file_path = "data/Datagenerator_files/fs50_s0.5_w1_aug2/Gyro_arm.txt"
split_dir = Path("data/Datagenerator_files/fs50_s0.5_w1_aug2")

seq_len = 50
num_channels = 3

batch_size = 64
epochs = 200
lr = 1e-3

# Early stopping
patience = 10
min_delta = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load data
# -------------------------------
data = np.loadtxt(file_path, delimiter=",")

X = data[:, :-1]
y = data[:, -1].astype(int) - 1  # 1..12 -> 0..11

X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
num_classes = len(np.unique(y))

# -------------------------------
# Load fixed split indices
# -------------------------------
train_idx = np.loadtxt(split_dir / "train_idx.txt", dtype=int)
val_idx   = np.loadtxt(split_dir / "val_idx.txt", dtype=int)
test_idx  = np.loadtxt(split_dir / "test_idx.txt", dtype=int)

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print("Split sizes:", len(train_idx), len(val_idx), len(test_idx))

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=batch_size, shuffle=False
)

# -------------------------------
# CNN Feature Extractor Model
# -------------------------------
class IMUCNN(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        self.flattened_dim = (seq_len // 4) * 128

        # Embedding layer (feature representation)
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 32)

        # Classification head
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(32, num_classes)

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
# Training + Early stopping
# -------------------------------
best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_idx)

    print(f"Epoch {epoch+1:3d}/{epochs} | Val loss {val_loss:.4f}")

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_state)
model.to(device)

# -------------------------------
# Save feature extractor model
# -------------------------------
script_dir = Path(__file__).parent
sensor_name = Path(file_path).stem  # Gyro_arm
save_path = script_dir / f"feature_extractor_{sensor_name}.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "seq_len": seq_len,
    "num_channels": num_channels,
    "embedding_dim": 32
}, save_path)

print(f"Feature extractor saved to:\n{save_path}")

# -------------------------------
# Test accuracy
# -------------------------------
model.eval()
y_pred = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        y_pred.extend(torch.argmax(model(xb), dim=1).cpu().numpy())

print("Test accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# Extract embeddings
# -------------------------------
def extract_embeddings(loader):
    Z, Y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            Z.append(model.extract_features(xb).cpu().numpy())
            Y.append(yb.numpy())
    return np.concatenate(Z), np.concatenate(Y)

train_Z, train_y = extract_embeddings(train_loader)
val_Z, val_y = extract_embeddings(val_loader)
test_Z, test_y = extract_embeddings(test_loader)

feat_dir = script_dir / "ExtractedFeatures"
feat_dir.mkdir(exist_ok=True)

np.savez_compressed(
    feat_dir / f"{sensor_name}_embeddings.npz",
    Z_train=train_Z, y_train=train_y,
    Z_val=val_Z, y_val=val_y,
    Z_test=test_Z, y_test=test_y
)

print("Saved embeddings:", train_Z.shape, val_Z.shape, test_Z.shape)
