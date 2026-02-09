from pathlib import Path

# ============================================================
# EXTRACTION CONFIGURATION
# ============================================================

# ========================= CHANGE THESE ===========================
dataset_config = "s1_w1_aug2"      # Base config (stride_window_augmentation)
variant_name = "fs50_WEAK_SIGNAL_w0p2"  # Which data variant to extract features FROM
model_variant = "fs50_clean"       # Which variant models were trained ON
seq_len = 50                       # Must match model training AND data FS!
# ==================================================================

# Build paths using parameters
workspace_root = Path("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code")
parent_dir = workspace_root / "data" / "Datagenerator_files" / dataset_config
variant_dir = parent_dir / variant_name
models_dir = workspace_root / "Feature extraction CNNs" / "Models" / dataset_config / model_variant

# Which sensors to process
SENSORS = {
    "Acc_ankle": {"num_channels": 3},
    "Acc_arm": {"num_channels": 3},
    "Acc_chest": {"num_channels": 3},
    "ECG": {"num_channels": 2},
    "Gyro_ankle": {"num_channels": 3},
    "Gyro_arm": {"num_channels": 3},
    "Mag_ankle": {"num_channels": 3},
    "Mag_arm": {"num_channels": 3},
}

# Extraction settings
batch_size = 256
device = "cuda"  # or "cpu"

# Output directory (features saved inside variant directory)
output_dir = variant_dir / "ExtractedFeatures"

def get_model_path(sensor_name: str) -> Path:
    """Get path to trained model checkpoint for a sensor."""
    return models_dir / f"feature_extractor_{sensor_name}.pth"

def get_data_path(sensor_name: str) -> Path:
    """Get path to sensor data file."""
    return variant_dir / f"{sensor_name}.txt"
