"""Extract embeddings for one sensor from one tremor variant."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config_extraction import (
    HOLDOUT_SUBJECTS,
    SENSORS,
    TEST_SUBJECTS,
    VAL_SUBJECTS,
    batch_size,
    get_data_path,
    get_model_path,
    get_output_dir,
    get_variant_dir,
    models_dir,
    overwrite_split_files,
    seq_len,
    split_mode,
)


class IMUCNN(nn.Module):
    """CNN architecture matching the training scripts in s2_w2_aug2_fs50."""

    def __init__(self, num_classes: int, seq_len: int, num_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.flattened_dim = (seq_len // 4) * 256
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(self.flattened_dim, 128)
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.fc_embed(x)

    def forward(self, x):
        z = self.extract_features(x)
        z = self.drop_cls(z)
        return self.fc_cls(z)


def _to_1d_int(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(int)


def _load_idx_file(idx_file: Path) -> np.ndarray:
    """Load one index file; return empty array if file is intentionally empty."""
    if not idx_file.exists() or idx_file.stat().st_size == 0:
        return np.array([], dtype=int)
    return _to_1d_int(np.loadtxt(idx_file, dtype=int))


def _load_or_create_split_indices(subjects: np.ndarray, variant_dir: Path):
    split_dir = variant_dir / "splits"
    train_file = split_dir / "train_idx.txt"
    val_file = split_dir / "val_idx.txt"
    test_file = split_dir / "test_idx.txt"
    holdout_file = split_dir / "holdout_idx.txt"

    if split_mode == "existing_files":
        if not (train_file.exists() and val_file.exists() and test_file.exists()):
            raise FileNotFoundError(
                f"split_mode='existing_files' but split files missing in {split_dir}"
            )
        train_idx = _load_idx_file(train_file)
        val_idx = _load_idx_file(val_file)
        test_idx = _load_idx_file(test_file)
        holdout_idx = _load_idx_file(holdout_file)
        return train_idx, val_idx, test_idx, holdout_idx, "existing_files"

    if split_mode != "subject_based":
        raise ValueError(f"Unknown split_mode: {split_mode}")

    test_mask = np.isin(subjects, TEST_SUBJECTS)
    val_mask = np.isin(subjects, VAL_SUBJECTS)
    holdout_mask = np.isin(subjects, HOLDOUT_SUBJECTS)
    train_mask = ~(test_mask | val_mask | holdout_mask)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    holdout_idx = np.where(holdout_mask)[0]

    # Parkinson-only variant does not include CT subject IDs from train/val/test config.
    # Keep these samples as inference/test to avoid creating a synthetic train split.
    if len(val_idx) == 0 and len(test_idx) == 0:
        train_idx = np.array([], dtype=int)
        val_idx = np.array([], dtype=int)
        test_idx = np.arange(len(subjects), dtype=int)
        holdout_idx = np.array([], dtype=int)

    if overwrite_split_files or not (train_file.exists() and val_file.exists() and test_file.exists()):
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(train_file, train_idx, fmt="%d")
        np.savetxt(val_file, val_idx, fmt="%d")
        np.savetxt(test_file, test_idx, fmt="%d")
        np.savetxt(holdout_file, holdout_idx, fmt="%d")

    return train_idx, val_idx, test_idx, holdout_idx, "subject_based"


def load_data(variant_name: str, sensor_name: str, num_channels: int):
    """Load sensor data and apply split indices for one variant."""
    data_path = get_data_path(variant_name, sensor_name)
    variant_dir = get_variant_dir(variant_name)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = np.loadtxt(str(data_path), delimiter=",")
    X = data[:, :-7]
    y_activity = data[:, -7].astype(int) - 1
    y_subject = data[:, -6].astype(int)
    base_idx = data[:, -5].astype(int)
    tremor_freq = data[:, -4].astype(np.float32)
    tremor_acc_rms = data[:, -3].astype(np.float32)
    tremor_gyro_rms = data[:, -2].astype(np.float32)
    tremor_score = data[:, -1].astype(int)

    X = X.reshape(-1, num_channels, seq_len).astype(np.float32)

    train_idx, val_idx, test_idx, holdout_idx, split_source = _load_or_create_split_indices(
        y_subject, variant_dir
    )

    meta_all = {
        "activity": y_activity,
        "subject": y_subject,
        "base_idx": base_idx,
        "tremor_freq": tremor_freq,
        "tremor_acc_rms": tremor_acc_rms,
        "tremor_gyro_rms": tremor_gyro_rms,
        "tremor_score": tremor_score,
    }

    def select(idx: np.ndarray):
        return X[idx], {k: v[idx] for k, v in meta_all.items()}

    X_train, meta_train = select(train_idx)
    X_val, meta_val = select(val_idx)
    X_test, meta_test = select(test_idx)
    X_holdout, meta_holdout = select(holdout_idx)

    return (
        (X_train, X_val, X_test, X_holdout),
        (meta_train, meta_val, meta_test, meta_holdout),
        split_source,
    )


def extract_embeddings(model, X_split: np.ndarray, y_split: np.ndarray, device: torch.device):
    """Extract embeddings for one split; returns empty arrays for empty splits."""
    if len(X_split) == 0:
        return np.empty((0, 128), dtype=np.float32), np.empty((0,), dtype=np.int64)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_split), torch.tensor(y_split)),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    feats = []
    labs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            z = model.extract_features(xb)
            feats.append(z.cpu().numpy())
            labs.append(yb.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)


def main(sensor_name: str, variant_name: str):
    """Extract features for one sensor in one variant."""
    if sensor_name not in SENSORS:
        raise ValueError(f"Unknown sensor: {sensor_name}. Available: {list(SENSORS.keys())}")

    num_channels = SENSORS[sensor_name]["num_channels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant_dir = get_variant_dir(variant_name)
    output_dir = get_output_dir(variant_name)

    print(f"\n{'=' * 70}")
    print(f"Extracting features for sensor={sensor_name}, variant={variant_name}")
    print(f"{'=' * 70}")
    print(f"Variant dir: {variant_dir}")
    print(f"Models dir:  {models_dir}")
    print(f"Output dir:  {output_dir}")

    print("\n[1/4] Loading data and split indices...")
    (X_train, X_val, X_test, X_holdout), (meta_train, meta_val, meta_test, meta_holdout), split_source = (
        load_data(variant_name, sensor_name, num_channels)
    )
    print(f"  Split source: {split_source}")
    print(
        f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Holdout: {len(X_holdout)}"
    )

    print("\n[2/4] Loading model checkpoint...")
    model_path = get_model_path(sensor_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        num_classes = int(ckpt.get("num_classes", 12))
    else:
        state_dict = ckpt
        num_classes = 12

    model = IMUCNN(num_classes=num_classes, seq_len=seq_len, num_channels=num_channels).to(device)
    model.load_state_dict(state_dict)
    print(f"  Loaded from: {model_path}")

    print("\n[3/4] Extracting embeddings...")
    Z_train, _ = extract_embeddings(model, X_train, meta_train["activity"], device)
    Z_val, _ = extract_embeddings(model, X_val, meta_val["activity"], device)
    Z_test, _ = extract_embeddings(model, X_test, meta_test["activity"], device)
    Z_holdout, _ = extract_embeddings(model, X_holdout, meta_holdout["activity"], device)
    print(f"  Train embeddings: {Z_train.shape}")
    print(f"  Val embeddings:   {Z_val.shape}")
    print(f"  Test embeddings:  {Z_test.shape}")
    print(f"  Holdout embeddings: {Z_holdout.shape}")

    print("\n[4/4] Saving embeddings...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{sensor_name}_embeddings.npz"
    np.savez_compressed(
        output_file,
        train_embeddings=Z_train,
        val_embeddings=Z_val,
        test_embeddings=Z_test,
        holdout_embeddings=Z_holdout,
        train_labels=meta_train["tremor_score"],
        val_labels=meta_val["tremor_score"],
        test_labels=meta_test["tremor_score"],
        holdout_labels=meta_holdout["tremor_score"],
        train_activities=meta_train["activity"],
        val_activities=meta_val["activity"],
        test_activities=meta_test["activity"],
        holdout_activities=meta_holdout["activity"],
        train_subjects=meta_train["subject"],
        val_subjects=meta_val["subject"],
        test_subjects=meta_test["subject"],
        holdout_subjects=meta_holdout["subject"],
        train_base_idx=meta_train["base_idx"],
        val_base_idx=meta_val["base_idx"],
        test_base_idx=meta_test["base_idx"],
        holdout_base_idx=meta_holdout["base_idx"],
    )

    print(f"  Saved: {output_file}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings for one sensor and one variant")
    parser.add_argument("--sensor", type=str, required=True, help=f"One of: {list(SENSORS.keys())}")
    parser.add_argument("--variant", type=str, required=True, help="Variant directory name")
    args = parser.parse_args()

    main(args.sensor, args.variant)
