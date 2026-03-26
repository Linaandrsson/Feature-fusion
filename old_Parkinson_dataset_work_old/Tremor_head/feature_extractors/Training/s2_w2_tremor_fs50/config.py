# Configuration for s2_w2_tremor_fs50 feature extractor training
from pathlib import Path

# ============================================================
# DATA CONFIGURATION
# ============================================================
# Get absolute paths (works regardless of where script is run from)
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent.parent  # Go up to workspace root

# ========================= CHANGE THESE ===========================
# Dataset configuration name (stride_window_tremor)
dataset_config = "s2_w2_tremor"

# Dataset variants to merge (clean + tremor)
variant_names = ["s2_w2_fs50_tremor_clean", "s2_w2_fs50_tremor_mild_mod", "s2_w2_fs50_tremor_mod_severe"]

# Model variant name (sampling frequency + dataset mix)
model_variant = "fs50_mixed3"  # fs50 = 50Hz, mixed = clean + mild_mod + mod_severe

# Sequence length   
seq_len = 100  # 50Hz * 2s window = 100 samples
# ====================================================================

# Build paths
data_path = workspace_root / "data" / "Tremor_datagenerator_files"
model_dir = script_dir.parent.parent / "Models" / dataset_config / model_variant

# ========================= SUBJECT SPLIT CONFIG ===========================
# Consistent with Feature extraction CNNs/Training/s2_w2_aug2_fs50
# Subject 1 is held out for later evaluation
MANUAL_SUBJECT_SPLITS = {
    "train": [2, 3, 4, 5, 10, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29],
    "val": [6, 8, 13, 15, 25],
    "test": [7, 9, 18, 19, 27]
}

# Training configuration (for backward compatibility with code that uses these)
TEST_SUBJECTS = MANUAL_SUBJECT_SPLITS["test"]
VAL_SUBJECTS = MANUAL_SUBJECT_SPLITS["val"]

# Model hyperparameters
batch_size = 64
epochs = 150
lr = 1e-3
patience = 10
min_delta = 1e-5

# Architecture
num_channels = 3  # IMU sensors: x, y, z
num_activities = 16  # Activities 0-15 in tremor dataset
activity_embed_dim = 16
embed_dim = 128  # Feature embedding dimension
num_classes = 2  # Binary tremor classification: 0 = no tremor, 1 = tremor present
