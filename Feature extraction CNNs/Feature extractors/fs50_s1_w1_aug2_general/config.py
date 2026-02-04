from pathlib import Path

# Dataset you want to extract embeddings from (can point to noisy)
base_dir = Path("data/Datagenerator_files/fs50_s1_w1_aug2_N50")
seq_len = 50

out_dir = base_dir / "ExtractedFeatures"

# Directory containing CLEAN trained feature extractors
clean_models_dir = Path("Feature extraction CNNs/Feature extractors/fs50_s1_w1_aug2")

def get_ckpt_path(sensor_name: str) -> Path:
    return clean_models_dir / f"feature_extractor_{sensor_name}.pth"
