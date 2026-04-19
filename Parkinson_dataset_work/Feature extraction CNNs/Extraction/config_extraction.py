from pathlib import Path

# ============================================================
# EXTRACTION CONFIGURATION
# ============================================================

# Resolve paths relative to this file so scripts work from any cwd.
script_dir = Path(__file__).parent.resolve()
feature_extraction_root = script_dir.parent
project_root = feature_extraction_root.parent

# ========================= CHANGE THESE ===========================
dataset_config = "s2_w2"
model_variant = "fs50_tremor_mixed_all3"
seq_len = 100

# Extract embeddings for all four tremor variants.
variant_names = [
    "s2_w2_fs50_tremor_clean",
    "s2_w2_fs50_tremor_mild_mod",
    "s2_w2_fs50_tremor_mod_severe",
    "s2_w2_fs50_tremor_parkinson",
]

# Subject-based split definition from training config.
TEST_SUBJECTS = [2, 7, 11]
VAL_SUBJECTS = [4, 9, 16]
HOLDOUT_SUBJECTS = [1, 14, 19]

# Split handling:
# - "subject_based": build splits from TEST_SUBJECTS / VAL_SUBJECTS / HOLDOUT_SUBJECTS
# - "existing_files": load variant_dir/splits/*.txt as-is
split_mode = "subject_based"

# If split_mode is "subject_based", write split files to each variant and overwrite old ones.
overwrite_split_files = True
# ==================================================================

# Data/model paths for Parkinson_dataset_work layout.
data_root = project_root / "Data" / "Tremor_datagenerator_files"
models_dir = feature_extraction_root / "Models" / dataset_config / model_variant

# Output directory inside each variant.
embeddings_folder_name = "Activity_ExtractedFeatures"

# Which sensors to process (matching trained models in Models/s2_w2/fs50_tremor_mixed_all3).
SENSORS = {
    "Acc_LL": {"num_channels": 3},
    "Acc_LR": {"num_channels": 3},
    "Acc_UL": {"num_channels": 3},
    "Acc_UR": {"num_channels": 3},
    "Acc_head": {"num_channels": 3},
    "Gyro_LL": {"num_channels": 3},
    "Gyro_LR": {"num_channels": 3},
    "Gyro_UL": {"num_channels": 3},
    "Gyro_UR": {"num_channels": 3},
    "Gyro_head": {"num_channels": 3},
    "Mag_LL": {"num_channels": 3},
    "Mag_LR": {"num_channels": 3},
    "Mag_UL": {"num_channels": 3},
    "Mag_UR": {"num_channels": 3},
    "Mag_head": {"num_channels": 3},
}

# Extraction settings
batch_size = 256


def get_model_path(sensor_name: str) -> Path:
    """Get path to trained model checkpoint for a sensor."""
    return models_dir / f"feature_extractor_{sensor_name}.pth"


def get_variant_dir(variant_name: str) -> Path:
    """Get path to one input variant directory."""
    return data_root / variant_name


def get_data_path(variant_name: str, sensor_name: str) -> Path:
    """Get path to one sensor data file for a variant."""
    return get_variant_dir(variant_name) / f"{sensor_name}.txt"


def get_output_dir(variant_name: str) -> Path:
    """Get output directory for one variant."""
    return get_variant_dir(variant_name) / embeddings_folder_name
