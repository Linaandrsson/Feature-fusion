# Configuration for s1_w1_aug2_fs50 feature extractor training
from pathlib import Path

# ============================================================
# DATA CONFIGURATION
# ============================================================
# Get absolute paths (works regardless of where script is run from)
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent  # Go up to workspace root (Training -> Feature extraction CNNs -> Code)

# ========================= CHANGE THESE ===========================
# Dataset configuration - Using Tremor data with all 3 augmentations combined
dataset_config = "s2_w2"

# Variant names to combine (all 3 Parkinson tremor augmentations)
# Format: s{STRIDE}_w{WINDOW}_fs{FS}_tremor_{AUGMENT_MODE}
variant_names = [
    "s2_w2_fs40_tremor_clean",       # No tremor (100% score=0)
    "s2_w2_fs40_tremor_mild_mod",    # Mild tremor (50% score=1, 50% score=2)
    "s2_w2_fs40_tremor_mod_severe"   # Severe tremor (50% score=3, 50% score=4)
]

# Sequence length
seq_len = 40*2  # fs*s 

output_variant_name = "fs40_tremor_mixed_all3"  # Combined output from all 3 augmentations
# ====================================================================

# Use Tremor_datagenerator_files instead of regular Datagenerator_files
parent_dir = workspace_root / "data" / "Tremor_datagenerator_files"

# Create list of all variant directories
variant_dirs = [parent_dir / variant_name for variant_name in variant_names]

# For single-variant backward compatibility (points to first dir, but should use variant_dirs list instead)
variant_dir = variant_dirs[0]
base_dir = str(variant_dir)



# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
# Where to save trained models (inside Feature extraction CNNs folder)
feature_extraction_root = script_dir.parent.parent  # Training/s2_w2_aug2_fs40_mixed_TODO -> Training -> Feature extraction CNNs
models_output_dir = feature_extraction_root / "Models" / dataset_config / output_variant_name
models_output_dir.mkdir(parents=True, exist_ok=True)

# Where to save embeddings/extracted features
# Default: Save to each variant's folder in Tremor_datagenerator_files
embeddings_base_dir = parent_dir  # Same as data location (Tremor_datagenerator_files)
embeddings_folder_name = "Activity_ExtractedFeatures"  # Folder name within each variant directory

# To save embeddings elsewhere, uncomment and modify:
# embeddings_base_dir = workspace_root / "path" / "to" / "custom" / "location"
# embeddings_folder_name = "CustomEmbeddingsFolder"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
import numpy as np

def load_combined_sensor_data(sensor_filename: str):
    """
    Load and combine sensor data from all 3 tremor augmentation folders.
    
    Args:
        sensor_filename: Name of sensor file (e.g., "Acc_ankle.txt")
    
    Returns:
        Combined numpy array with data from all 3 variants concatenated
    """
    all_data = []
    
    for variant_dir in variant_dirs:
        file_path = variant_dir / sensor_filename
        if not file_path.exists():
            raise FileNotFoundError(f"Sensor data not found: {file_path}")
        
        # Convert to string to avoid numpy path resolution issues
        data = np.loadtxt(str(file_path.resolve()), delimiter=",")
        all_data.append(data)
        print(f"Loaded {data.shape[0]} samples from {variant_dir.name}/{sensor_filename}")
    
    combined_data = np.concatenate(all_data, axis=0)
    print(f"Total combined samples: {combined_data.shape[0]} (from {len(variant_dirs)} variants)")
    
    return combined_data
