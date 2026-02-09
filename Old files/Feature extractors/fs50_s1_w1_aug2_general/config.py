from pathlib import Path

# ============================================================
# HIERARCHICAL STRUCTURE CONFIGURATION
# ============================================================
# Parent directory contains splits.npz (shared across all variants)
parent_dir = Path("/Users/linaandersson/Library/CloudStorage/OneDrive-NTNU/master/Code/data/Datagenerator_files/s1_w1_aug2")

# Variant subdirectory contains sensor data (e.g., fs50_clean, fs50_AWGN_s0p3_AAnAC_n1)
variant_name = "fs50_clean"  # Change this to use different variant
variant_dir = parent_dir / variant_name

# For backward compatibility
base_dir = variant_dir  # Deprecated, use variant_dir directly

seq_len = 50

# Output directory for extracted features (inside variant folder)
out_dir = variant_dir / "ExtractedFeatures"

# Directory containing CLEAN trained feature extractors
clean_models_dir = Path("Feature extraction CNNs/Feature extractors/fs50_s1_w1_aug2")

def get_ckpt_path(sensor_name: str) -> Path:
    return clean_models_dir / f"feature_extractor_{sensor_name}.pth"
