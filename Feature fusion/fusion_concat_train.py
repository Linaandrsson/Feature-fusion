import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Choose which sensors to fuse (names must match your npz files) ----
#feat_dir = Path("/Users/linaandersson/Desktop/master/Code/Feature extraction CNNs/Feature extractors/ExtractedFeatures")  # <- set this to the folder where your embeddings live
feat_dir = Path("Feature extraction CNNs/Feature extractors/fs50_s0.5_w1_aug2/ExtractedFeatures")  # <- set this to the folder where your embeddings live
#sensors = ["Acc_ankle", "Mag_ankle", "Gyro_ankle", "Acc_arm", "Gyro_arm", "Mag_arm"]  # example fusion set
#sensors = ["Mag_ankle", "Acc_ankle", "Acc_arm", "Gyro_arm", "Gyro_ankle", "Mag_arm", "ECG"]
sensors = ["Mag_ankle", "Acc_ankle", "Acc_chest"]

def load_sensor_npz(sensor_name: str):
    p = feat_dir / f"{sensor_name}_embeddings.npz"
    d = np.load(p)
    return d["Z_train"], d["y_train"], d["Z_val"], d["y_val"], d["Z_test"], d["y_test"]

# ---- Load and concat embeddings ----
Zt_list, Zv_list, Zte_list = [], [], []
y_train = y_val = y_test = None

for s in sensors:
    Z_train, y_tr, Z_val, y_v, Z_test, y_te = load_sensor_npz(s)
    Zt_list.append(Z_train)
    Zv_list.append(Z_val)
    Zte_list.append(Z_test)

    # labels should match across sensors (same splits), so we just keep the first
    if y_train is None:
        y_train, y_val, y_test = y_tr, y_v, y_te

Z_train_fused = np.concatenate(Zt_list, axis=1)
Z_val_fused   = np.concatenate(Zv_list, axis=1)
Z_test_fused  = np.concatenate(Zte_list, axis=1)

print("Fused shapes:", Z_train_fused.shape, Z_val_fused.shape, Z_test_fused.shape)

# ---- DataLoaders ----
batch_size = 128
train_loader = DataLoader(
    TensorDataset(torch.tensor(Z_train_fused, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(Z_val_fused, dtype=torch.float32),
                  torch.tensor(y_val, dtype=torch.long)),
    batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(Z_test_fused, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.long)),
    batch_size=batch_size, shuffle=False
)



class OriginalHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        z = self.drop(z)
        return self.fc(z)

class PaperHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dims=(512, 256, 128),
        dropout_p: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dims[2], num_classes)
        )

    def forward(self, z):
        return self.net(z)


# ---- Fusion classifier head (small MLP) ----
# in_dim = Z_train_fused.shape[1]
# num_classes = len(np.unique(y_train))

# model = nn.Sequential(
#     nn.Linear(in_dim, 32),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(32, num_classes)

# ).to(device)

in_dim = Z_train_fused.shape[1]
num_classes = len(np.unique(y_train))

#model = OriginalHead(in_dim=in_dim, num_classes=num_classes, dropout_p=0.5).to(device)

model = PaperHead(
    in_dim=in_dim,
    num_classes=num_classes,
    hidden_dims=(256, 256, 128),
    dropout_p=0.3
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 200
patience = 10
min_delta = 1e-4

best_val_loss = float("inf")
best_state = None
no_improve = 0

for epoch in range(epochs):
    # -------------------
    # Train
    # -------------------
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

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

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    # -------------------
    # Logging
    # -------------------
    print(
        f"Epoch {epoch+1:3d}/{epochs} | "
        f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
    )

    # -------------------
    # Early stopping
    # -------------------
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Restore best model
model.load_state_dict(best_state)
model.to(device)
model.eval()

# test accuracy
y_pred = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        y_pred.extend(torch.argmax(model(xb), dim=1).cpu().numpy())

test_acc = accuracy_score(y_test, y_pred)
print("Fusion test accuracy:", test_acc)

# -------------------------------
# Confusion Matrices (Counts + Normalized)
# -------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels_display = list(range(1, 13))  # activities 1..12

# Counts
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels_display)
disp.plot(cmap="viridis")
plt.title(f"Fusion Confusion Matrix (Counts)\nSensors: {', '.join(sensors)}")
plt.show()

# Normalized per class
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                   display_labels=labels_display)
disp_norm.plot(cmap="viridis", values_format=".2f")
plt.title(f"Fusion Confusion Matrix (Normalized)\nSensors: {', '.join(sensors)}")
plt.show()
