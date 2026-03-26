# Configuration for s2_w2_aug2_fs50 feature extractor training
from pathlib import Path

# ============================================================
# DATA CONFIGURATION
# ============================================================
# Get absolute paths (works regardless of where script is run from)
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent  # Go up to workspace root (Training -> Feature extraction CNNs -> Code)

# ========================= CHANGE THESE ===========================
# Dataset configuration - Using Tremor data with all 4 variants combined
dataset_config = "s2_w2"

# Variant names to combine (clean + mild_mod + mod_severe + parkinson)
# Format: s{STRIDE}_w{WINDOW}_fs{FS}_tremor_{AUGMENT_MODE}
variant_names = [
    "s2_w2_fs50_tremor_clean",       # No tremor (100% score=0)
    "s2_w2_fs50_tremor_mild_mod",    # Mild tremor (50% score=1, 50% score=2)
    "s2_w2_fs50_tremor_mod_severe",  # Severe tremor (50% score=3, 50% score=4)
    #"s2_w2_fs50_tremor_parkinson",   # Real PD windows (score=5 sentinel)
]

# HAR labels are 1..16 in this merged setup
num_activity_classes = 16
activity_label_min = 1
activity_label_max = activity_label_min + num_activity_classes - 1

# Prefix for split files so old 12-class split files are not reused
split_file_prefix = "arm16_all4_sourceaware"

# Sequence length
seq_len = 100  # fs*s (50 Hz × 2s = 100 samples)

# Output variant name (used for saving models and embeddings)
output_variant_name = "fs50_tremor_mixed_all4"  # Combined output from all 4 variants

# ========================= SUBJECT SPLIT CONFIG ===========================
# Optionally specify exact train/val/test subjects (override auto-generation)
# Set to None to auto-generate based on source-aware random split
# Set to dict with keys "train", "val", "test" to use manual specification
MANUAL_SUBJECT_SPLITS = None  # Example: {"train": [1,2,3,4,5,...], "val": [6,7,...], "test": [8,9,...]}

MANUAL_SUBJECT_SPLITS = {
    "train": [2, 3, 4, 5, 10, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29],  # mHealth: 2,3,4,5,10 + CT: 11,12,14,16,17,20,21,22,23,24,26,28,29
    "val": [6, 8, 13, 15, 25],  # mHealth: 6,8 + CT: 13,15,25
    "test": [7, 9, 18, 19, 27]  # mHealth: 7,9 + CT: 18,19,27
}
# Subject 1 is held out for later evaluation
# Subject 1 is held out for later evaluation
# ====================================================================

# Use Tremor_datagenerator_files instead of regular Datagenerator_files
parent_dir = workspace_root / "Data" / "Tremor_datagenerator_files"

# Create list of all variant directories
variant_dirs = [parent_dir / variant_name for variant_name in variant_names]

# For single-variant backward compatibility (points to first dir, but should use variant_dirs list instead)
variant_dir = variant_dirs[0]
base_dir = str(variant_dir)



# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
# Where to save trained models (inside Feature extraction CNNs folder)
feature_extraction_root = script_dir.parent.parent  # Training/s2_w2_aug2_fs50_mixed_TODO -> Training -> Feature extraction CNNs
models_output_dir = feature_extraction_root / "Models" / dataset_config / output_variant_name
models_output_dir.mkdir(parents=True, exist_ok=True)

# Where to save embeddings/extracted features
# Default: Save to each variant's folder in Tremor_datagenerator_files
embeddings_base_dir = parent_dir  # Same as data location (Tremor_datagenerator_files)
embeddings_folder_name = "Activity_ExtractedFeatures"  # Folder name within each variant directory

# Where to save confusion matrices
confusion_matrices_dir = workspace_root / "Confusion_Matrixes" / script_dir.name
confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

# To save embeddings elsewhere, uncomment and modify:
# embeddings_base_dir = workspace_root / "path" / "to" / "custom" / "location"
# embeddings_folder_name = "CustomEmbeddingsFolder"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
import numpy as np


def _subject_source(subject_id: int) -> str:
    """
    Infer source from merged subject-id ranges used by the generator.

    1..10   -> mhealth
    11..29  -> ct
    30..44  -> pd
    """
    sid = int(subject_id)
    if sid <= 10:
        return "mhealth"
    if sid <= 29:
        return "ct"
    return "pd"


def build_source_aware_subject_splits(
    subject_ids: np.ndarray,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Build deterministic subject-level train/val/test splits while ensuring
    each source group contributes to val and test when possible.
    """
    unique_subjects = sorted({int(s) for s in np.asarray(subject_ids).reshape(-1)})

    by_source: dict[str, list[int]] = {"mhealth": [], "ct": [], "pd": []}
    for sid in unique_subjects:
        by_source[_subject_source(sid)].append(sid)

    rng = np.random.default_rng(seed)
    train_subjects: list[int] = []
    val_subjects: list[int] = []
    test_subjects: list[int] = []

    for source in ["mhealth", "ct", "pd"]:
        ids = np.array(sorted(by_source[source]), dtype=int)
        n = len(ids)
        if n == 0:
            continue
        if n == 1:
            train_subjects.append(int(ids[0]))
            continue
        if n == 2:
            perm = ids[rng.permutation(n)]
            test_subjects.append(int(perm[0]))
            train_subjects.append(int(perm[1]))
            continue

        perm = ids[rng.permutation(n)]

        n_test = max(1, int(np.floor(n * test_ratio)))
        n_val = max(1, int(np.floor(n * val_ratio)))

        # Keep at least one train subject in each source group.
        while n_test + n_val >= n:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        test_subjects.extend([int(x) for x in perm[:n_test]])
        val_subjects.extend([int(x) for x in perm[n_test : n_test + n_val]])
        train_subjects.extend([int(x) for x in perm[n_test + n_val :]])

    return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

def load_combined_sensor_data(sensor_filename: str):
    """
    Load and combine sensor data from all configured variant folders.
    
    Args:
        sensor_filename: Name of sensor file (e.g., "Acc_ankle.txt")
    
    Returns:
        Combined numpy array with data from all variants concatenated
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
