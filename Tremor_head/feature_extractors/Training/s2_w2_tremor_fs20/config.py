# Configuration for s2_w2_tremor_fs30 feature extractor training
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
variant_names = ["s2_w2_fs20_tremor_clean", "s2_w2_fs20_tremor_mild_mod", "s2_w2_fs20_tremor_mod_severe"]

# Model variant name (sampling frequency + dataset mix)
model_variant = "fs20_mixed"  # fs20 = 20Hz, mixed = clean + mild_mod + mod_severe

# Sequence length
seq_len = 2*20  # 20Hz * 2s window = 40 samples
# ====================================================================

# Build paths
data_path = workspace_root / "data" / "Tremor_datagenerator_files"
model_dir = script_dir.parent.parent / "Models" / dataset_config / model_variant

# Training configuration
TEST_SUBJECTS = [5, 10]
VAL_SUBJECTS = [2, 7]

# Model hyperparameters
batch_size = 64
epochs = 150
lr = 1e-3
patience = 10
min_delta = 1e-5

# Architecture
num_channels = 3  # IMU sensors: x, y, z
num_activities = 12
activity_embed_dim = 16
embed_dim = 128  # Feature embedding dimension
num_classes = 5  # Tremor scores: 0-4
