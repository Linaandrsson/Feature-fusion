"""
Quick Training Test
===================
Test if model can learn on a small subset of data.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import parent_dir, variant_dirs, seq_len, load_combined_sensor_data

print("=" * 80)
print("QUICK TRAINING TEST")
print("=" * 80)

# Load data
sensor_name = "Gyro_arm"
num_channels = 3
data = load_combined_sensor_data(f"{sensor_name}.txt")

X = data[:, :-7]
y = data[:, -7].astype(int) - 1
subjects = data[:, -6].astype(int)

X = X.reshape(-1, num_channels, seq_len).astype(np.float32)
num_classes = 12

# Create splits
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]
train_idx = np.where(~np.isin(subjects, TEST_SUBJECTS + VAL_SUBJECTS))[0]
val_idx = np.where(np.isin(subjects, VAL_SUBJECTS))[0]

# Take small subset for quick test
train_idx_small = train_idx[:500]
val_idx_small = val_idx[:100]

X_train = torch.tensor(X[train_idx_small], dtype=torch.float32)
y_train = torch.tensor(y[train_idx_small], dtype=torch.long)
X_val = torch.tensor(X[val_idx_small], dtype=torch.float32)
y_val = torch.tensor(y[val_idx_small], dtype=torch.long)

print(f"\nTrain: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Classes: {num_classes}")
print(f"Input shape: {X_train[0].shape}")

# Check label distribution
print(f"\nTrain label distribution:")
unique, counts = np.unique(y_train.numpy(), return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} samples")

# Simple CNN (from your files)
class IMUCNN(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
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
        self.flattened_dim = (seq_len // 4) * 128
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc_embed(x)
        x = self.drop_cls(x)
        return self.fc_cls(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IMUCNN(num_classes, seq_len, num_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nDevice: {device}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Quick training
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

print(f"\n{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
print("-" * 70)

for epoch in range(20):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += xb.size(0)
    
    train_loss /= train_total
    train_acc = train_correct / train_total
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            
            val_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    print(f"{epoch+1:<10} {train_loss:<15.4f} {train_acc:<15.3%} {val_loss:<15.4f} {val_acc:<15.3%}")

print("\n" + "=" * 80)
print("OBSERVATIONS:")
print("=" * 80)
print("1. If train loss decreases: model CAN learn")
print("2. If train acc increases: model IS learning")
print("3. If val acc is low: subject-independent generalization is hard (expected)")
print("4. If BOTH train & val stay low: problem with model/data")
print("=" * 80)
