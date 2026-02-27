import torch
import torch.nn as nn

# ---- Must match the trained architecture exactly ----
class IMUCNN(nn.Module):
    def __init__(self, num_classes: int, num_channels: int, seq_len: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        flattened_dim = (seq_len // 4) * 8  # with 2x MaxPool(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "Baseline models/baseline_ankle_acc_cnn.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    seq_len = checkpoint["seq_len"]
    num_channels = checkpoint["num_channels"]

    model = IMUCNN(
        num_classes=num_classes,
        num_channels=num_channels,
        seq_len=seq_len
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✅ Baseline model loaded successfully!")
    print(f"num_classes={num_classes}, num_channels={num_channels}, seq_len={seq_len}")
    print(model)

if __name__ == "__main__":
    main()
