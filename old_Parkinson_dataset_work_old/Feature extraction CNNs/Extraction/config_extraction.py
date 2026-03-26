from pathlib import Path

# ============================================================
# EXTRACTION CONFIGURATION
# ============================================================

# ========================= CHANGE THESE ===========================
dataset_config = "s2_w2"
variant_name = "s2_w2_fs50_tremor_clean" # Which data variant to extract features FROM. "s2_w2_fs50_tremor_parkinson", "s2_w2_fs50_tremor_mod_severe", "s2_w2_fs50_tremor_mild_mod", "s2_w2_fs50_tremor_clean"
model_variant = "fs50_tremor_mixed_all3"  # Where trained arm models are stored
seq_len = 100  # Must match model training AND data FS
split_file_prefix = ""  # "" -> use train_idx.txt in variant/splits
# ==================================================================

# Build paths using parameters
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent
parent_dir = workspace_root / "Data" / "Tremor_datagenerator_files"
variant_dir = parent_dir / variant_name
models_dir = workspace_root / "Feature extraction CNNs" / "Models" / dataset_config / model_variant
split_dir = variant_dir / "splits"

# Which sensors to process
SENSORS = {
    "Acc_arm": {"num_channels": 3},
    "Gyro_arm": {"num_channels": 3},
    "Mag_arm": {"num_channels": 3},
}

# Extraction settings
batch_size = 256
device = "cuda"  # or "cpu"

# Output directory (features saved inside variant directory)
output_dir = variant_dir / "Activity_ExtractedFeatures"

def get_model_path(sensor_name: str) -> Path:
    """Get path to trained model checkpoint for a sensor."""
    return models_dir / f"feature_extractor_{sensor_name}.pth"

def get_data_path(sensor_name: str) -> Path:
    """Get path to sensor data file."""
    return variant_dir / f"{sensor_name}.txt"


def get_split_path(split_name: str) -> Path:
    """Get split-file path for train/val/test indices."""
    if split_file_prefix:
        return split_dir / f"{split_file_prefix}_{split_name}_idx.txt"
    return split_dir / f"{split_name}_idx.txt"
