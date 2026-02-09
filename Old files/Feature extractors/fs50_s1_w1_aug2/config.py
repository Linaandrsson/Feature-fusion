# Configuration for fs50_s1_w1_aug2 feature extractors
from pathlib import Path

# ============================================================
# HIERARCHICAL STRUCTURE CONFIGURATION
# ============================================================
# Parent directory contains splits.npz (shared across all variants)
parent_dir = Path("data/Datagenerator_files/s1_w1_aug2")

# Variant subdirectory contains sensor data (e.g., fs50_clean, fs50_AWGN_s0p3_AAnAC_n1)
variant_name = "fs50_clean"  # Change this to use different variant
variant_dir = parent_dir / variant_name

# For backward compatibility
base_dir = str(variant_dir)  # Deprecated, use variant_dir directly

seq_len = 50
