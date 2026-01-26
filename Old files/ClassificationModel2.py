import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# Config
# -------------------------------
file_path = "data/Datagenerator_files/fs50_s0.5_w1_aug2/ankle_acc.txt"
seq_len = 50                        # 1s @ 50Hz
num_channels = 3                    # x,y,z
test_size = 0.30
batch_size = 64
epochs = 20
lr = 1e-3
random_state = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load data
# -------------------------------
data = np.loadtxt(file_path, delimiter=",")

X = data[:, :-1]
y = data[:, -1].astype(int) - 1  # 1..12 -> 0..11

# Sanity checks
expected_features = num_channels * seq_len
assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
assert y.min() >= 0 and y.max() <= 11, f"Labels out of range: min={y.min()}, max={y.max()}"

# Reshape to (N, C, L) for Conv1d
X = X.reshape(-1, num_channels, seq_len).astype(np.float32)

num_classes = len(np.unique(y))

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=random_state
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=batch_size,
    shuffle=False
)

# -------------------------------
# CNN Model
# -------------------------------
class IMUCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L -> L/2

            nn.Conv1d(32, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),  # L -> L/4
        )

        # After two MaxPool(2): L becomes floor(floor(L/2)/2) = floor(L/4)
        flattened_dim = (seq_len // 4) * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = IMUCNN(num_classes=num_classes, seq_len=seq_len).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# Training
# -------------------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")

# -------------------------------
# Evaluation
# -------------------------------
model.eval()
y_pred = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Confusion Matrix (Normalized per true label)
# -------------------------------
cm_norm = confusion_matrix(
    y_test,
    y_pred,
    normalize="true"   # normaliserer hver rad
)

disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
disp_norm.plot(cmap="viridis", values_format=".2f")
plt.title("Confusion Matrix (Normalized per Class)")
plt.show()

