from pathlib import Path

# ============================================================
# EXTRACTION CONFIGURATION
# ============================================================

# ========================= CHANGE THESE ===========================
FS             = 50                    # Sampling frequency: 10, 20, 30, 40, 50
#TREMOR_VARIANT = "tremor_mild_mod"     # "tremor_clean", "tremor_mild_mod", "tremor_mod_severe"
#TREMOR_VARIANT = "tremor_clean"
TREMOR_VARIANT = "tremor_mod_severe"
MODEL_TYPE     = "mixed"               # "mixed" or "clean"
# ==================================================================

dataset_config = "s4_w4"
variant_name   = f"s4_w4_fs{FS}_{TREMOR_VARIANT}"
model_variant  = f"fs{FS}_tremor_mixed_all3" if MODEL_TYPE == "mixed" else f"fs{FS}_tremor_clean"
seq_len        = 4 * FS

# Build paths using parameters
workspace_root = Path(__file__).parents[2]  # Feature extraction CNNs -> Code
_data_base     = workspace_root / "data" / "Tremor_datagenerator_files"
variant_dir    = _data_base / variant_name
_fs_tag        = model_variant.split("_")[0]   # e.g. "fs10_tremor_clean" → "fs10"
_model_type    = "mixed" if "mixed" in model_variant else "clean"
_training_dir  = workspace_root / "Feature extraction CNNs" / "Training" / f"s4_w4_aug1_{_fs_tag}_{_model_type}"
_variant_splits = variant_dir / "splits"
parent_dir     = _training_dir / "splits" if not _variant_splits.exists() else _variant_splits  # extract_features.py loads train/val/test_idx.txt from here
models_dir     = workspace_root / "Feature extraction CNNs" / "Models" / dataset_config / model_variant

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
output_dir = variant_dir / ("ExtractedFeatures_mixed" if MODEL_TYPE == "mixed" else "ExtractedFeatures_clean")

def get_model_path(sensor_name: str) -> Path:
    """Get path to trained model checkpoint for a sensor."""
    return models_dir / f"feature_extractor_{sensor_name}.pth"

def get_data_path(sensor_name: str) -> Path:
    """Get path to sensor data file."""
    return variant_dir / f"{sensor_name}.txt"
