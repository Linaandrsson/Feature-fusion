from pathlib import Path

# ============================================================
# EXTRACTION CONFIGURATION
# ============================================================

# ========================= CHANGE THESE ===========================
# dataset_config = "s4_w4_clean"     # Base config (matches models_output_dir in training config)
# variant_name = "s4_w4_fs50_tremor_clean"   # Which data variant to extract features FROM
# model_variant = "fs50_tremor_mixed_all3"   # Which variant models were trained ON

#mixed:
dataset_config = "s4_w4"
#variant_name   = "s4_w4_fs50_tremor_clean_awgn_a000"   # bytt for hver korrupt variant
#variant_name   = "s4_w4_fs50_tremor_mild_mod_awgn_a000"
#variant_name   = "s4_w4_fs50_tremor_mod_severe_awgn_a000"
#variant_name = "s4_w4_fs50_tremor_clean_orient_r045"
#variant_name = "s4_w4_fs50_tremor_clean_dropout_p010"

#variant_name = "s4_w4_fs50_tremor_clean_fullDrop"
#variant_name = "s4_w4_fs50_tremor_mild_mod_fullDrop"
variant_name = "s4_w4_fs50_tremor_mod_severe_fullDrop"
model_variant  = "fs50_tremor_mixed_all3"

#clean
# dataset_config = "s4_w4"
# # #variant_name   = "s4_w4_fs50_tremor_clean_awgn_a100"   # bytt for hver korrupt variant
# # #variant_name   = "s4_w4_fs50_tremor_mild_mod_awgn_a100"
# # variant_name   = "s4_w4_fs50_tremor_mod_severe"
# # #variant_name = "s4_w4_fs50_tremor_mild_mod"
# #variant_name = "s4_w4_fs50_tremor_clean_orient_r045"
# #variant_name = "s4_w4_fs50_tremor_clean_fullDrop"
# #variant_name = "s4_w4_fs50_tremor_mild_mod_fullDrop"
# variant_name = "s4_w4_fs50_tremor_mod_severe_fullDrop"
# model_variant  = "fs50_tremor_clean"

seq_len = 200                       # Must match model training AND data FS! (50 Hz × 4s)
# ==================================================================

# Build paths using parameters
workspace_root = Path(__file__).parents[2]  # Feature extraction CNNs -> Code
_data_base     = workspace_root / "data" / "Tremor_datagenerator_files"
variant_dir    = _data_base / variant_name
_training_dir  = workspace_root / "Feature extraction CNNs" / "Training" / "s4_w4_aug1_fs50_clean"
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
output_dir = variant_dir / "ExtractedFeatures_mixed" #cleanwhen exrtcated with clean feature extractor, mixed when the other

def get_model_path(sensor_name: str) -> Path:
    """Get path to trained model checkpoint for a sensor."""
    return models_dir / f"feature_extractor_{sensor_name}.pth"

def get_data_path(sensor_name: str) -> Path:
    """Get path to sensor data file."""
    return variant_dir / f"{sensor_name}.txt"
