import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from config import base_dir, seq_len, get_ckpt_path

# --- CONFIG ---
sensor_name = "Gyro_arm"
num_channels = 3
batch_size = 256

# Where the trained clean model checkpoint is saved
ckpt_path = get_ckpt_path(sensor_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load data (same format as your training script) ---
file_path = base_dir / f"{sensor_name}.txt"
data = np.loadtxt(file_path, delimiter=",")

X = data[:, :-1]
y = data[:, -1].astype(int) - 1  # 1..12 -> 0..11

X = X.reshape(-1, num_channels, seq_len).astype(np.float32)

# --- Load splits (recommended for alignment) ---
train_idx = np.loadtxt(base_dir / "train_idx.txt", dtype=int)
val_idx   = np.loadtxt(base_dir / "val_idx.txt", dtype=int)
test_idx  = np.loadtxt(base_dir / "test_idx.txt", dtype=int)

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val),   torch.tensor(y_val)),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),  batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(y_train))

# --- Define the SAME model class as in Acc_ankle_CNN.py ---
class IMUCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, num_channels: int):
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
            nn.Dropout(0.3),
        )
        self.flattened_dim = (seq_len // 4) * 128
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)
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

# --- Load checkpoint ---
model = IMUCNN(num_classes=num_classes, seq_len=seq_len, num_channels=num_channels).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Extract embeddings ---
def extract_embeddings(loader):
    feats = []
    labs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            z = model.extract_features(xb)
            feats.append(z.cpu().numpy())
            labs.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)

Z_train, y_train2 = extract_embeddings(train_loader)
Z_val, y_val2     = extract_embeddings(val_loader)
Z_test, y_test2   = extract_embeddings(test_loader)

# --- Save in the same folder as this script ---
out_dir = Path(__file__).parent / "ExtractedFeatures"
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"{sensor_name}_embeddings.npz"
np.savez_compressed(out_path,
    Z_train=Z_train, y_train=y_train2,
    Z_val=Z_val,     y_val=y_val2,
    Z_test=Z_test,   y_test=y_test2
)

print("Saved:", out_path)
print("Shapes:", Z_train.shape, Z_val.shape, Z_test.shape)
