# Configuration for s1_w1_aug2_fs50 feature extractor training
from pathlib import Path

# ============================================================
# DATA CONFIGURATION
# ============================================================
# Get absolute paths (works regardless of where script is run from)
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent  # Go up to workspace root (Training -> Feature extraction CNNs -> Code)

# ========================= CHANGE THESE ===========================
# Dataset configuration name (stride_window_augmentation)
dataset_config = "s1_w1_aug2"

# CHANGE THIS TO RIGHT CLEAN FOLDER 
variant_name = "fs50_clean" 

# CHANGE THIS TO RIGHT SEQ LEN 
sampling_rate = 50                # Must match data FS!
window_sec = 1.0                  # Must match data window length in seconds!
seq_len = int(sampling_rate * window_sec)
# ====================================================================

# Use dataset_config parameter to build paths
parent_dir = workspace_root / "data" / "Datagenerator_files" / dataset_config

variant_dir = parent_dir / variant_name

# For backward compatibility
base_dir = str(variant_dir)



# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
# Where to save trained models (inside Feature extraction CNNs folder)
feature_extraction_root = script_dir.parent.parent  # Training/s1_w1_aug2_fs50 -> Training -> Feature extraction CNNs
models_output_dir = feature_extraction_root / "Models" / dataset_config / variant_name
models_output_dir.mkdir(parents=True, exist_ok=True)
