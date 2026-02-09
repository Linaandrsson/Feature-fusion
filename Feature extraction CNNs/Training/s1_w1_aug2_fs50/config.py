# Configuration for s1_w1_aug2_fs50 feature extractor training
from pathlib import Path

# ============================================================
# DATA CONFIGURATION
# ============================================================
# Get absolute paths (works regardless of where script is run from)
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent  # Go up to workspace root (Training -> Feature extraction CNNs -> Code)

# Parent directory contains splits.npz (shared across all variants)
parent_dir = workspace_root / "data" / "Datagenerator_files" / "s1_w1_aug2"

# Variant subdirectory contains sensor data to TRAIN on
variant_name = "fs50_clean"  # Change this to train on different variant (e.g., "fs50_AWGN_s0p3_AAnAC_n1")
variant_dir = parent_dir / variant_name

# For backward compatibility
base_dir = str(variant_dir)

# Model parameters
seq_len = 50

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
# Where to save trained models (inside Feature extraction CNNs folder)
feature_extraction_root = script_dir.parent.parent  # Training/s1_w1_aug2_fs50 -> Training -> Feature extraction CNNs
models_output_dir = feature_extraction_root / "Models" / "s1_w1_aug2" / variant_name
models_output_dir.mkdir(parents=True, exist_ok=True)
