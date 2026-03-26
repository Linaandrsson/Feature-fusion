#!/usr/bin/env python3
"""
Subject 1 Evaluation: Load raw data, extract features, and run inference.

Since Subject 1 was held out from training/extraction, we need to:
1. Load Subject 1's raw tremor data (all three variants)
2. Extract features using trained feature extractors
3. Run inference with fusion classifier
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Auto-detect paths
script_dir = Path(__file__).parent.resolve()  # fusion/
tremor_head_dir = script_dir.parent  # Tremor_head/
WORKSPACE_ROOT = tremor_head_dir.parent  # Parkinson_dataset_work/

DATA_ROOT = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"
MODELS_DIR = tremor_head_dir / "fusion" / "models" / "accarm50_gyroarm50_magarm50"
FEATURE_EXTRACTORS_DIR = tremor_head_dir / "feature_extractors"
TRAINING_SCRIPTS_DIR = FEATURE_EXTRACTORS_DIR / "Training" / "s2_w2_tremor_fs50"

# Add training scripts to path for imports
sys.path.insert(0, str(TRAINING_SCRIPTS_DIR))

MODEL_PATH = MODELS_DIR / "feature_fusion_best_accarm50_gyroarm50_magarm50.pth"

SENSORS = ["Acc_arm", "Gyro_arm", "Mag_arm"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config (must match training)
EMBED_DIM_PER_SENSOR = 128
TOTAL_EMBED_DIM = len(SENSORS) * EMBED_DIM_PER_SENSOR  # 384
HIDDEN_DIMS = [256, 128, 64]
NUM_CLASSES = 2  # Binary: 0=no tremor, 1=tremor
DROPOUT = 0.3

# Data config
SEQ_LEN = 100  # 50Hz * 2s window
NUM_ACTIVITIES = 16
ACTIVITY_EMBED_DIM = 16

print("=" * 80)
print("SUBJECT 1 EVALUATION - FEATURE EXTRACTION + INFERENCE")
print("=" * 80)
print(f"\nWorkspace: {WORKSPACE_ROOT}")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# CLASS DEFINITIONS
# ═══════════════════════════════════════════════════════════════

class FeatureFusionClassifier(nn.Module):
    """Binary tremor classifier using fused sensor embeddings."""
    
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, embeddings):
        """Forward pass."""
        return self.classifier(embeddings)


class RawDataDataset(Dataset):
    """Dataset for raw tremor data with windowing."""
    
    def __init__(self, X: np.ndarray, y_tremor: np.ndarray, y_activity: np.ndarray, 
                 y_subject: np.ndarray, seq_len: int = 100):
        self.X = X.astype(np.float32)
        self.y_tremor = y_tremor.astype(np.int64)
        self.y_activity = y_activity.astype(np.int64)
        self.y_subject = y_subject.astype(np.int64)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        tremor = torch.tensor(self.y_tremor[idx], dtype=torch.long)
        activity = torch.tensor(self.y_activity[idx], dtype=torch.long)
        subject = torch.tensor(self.y_subject[idx], dtype=torch.long)
        
        return x, tremor, activity, subject


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def load_feature_extractors():
    """Load trained feature extractors for all three sensors."""
    print("\n" + "=" * 80)
    print("LOADING TRAINED FEATURE EXTRACTORS")
    print("=" * 80)
    
    # Import model classes with delayed imports to avoid import issues
    feature_extractors = {}
    
    for sensor in SENSORS:
        print(f"\n  {sensor}:")
        
        if sensor == "Acc_arm":
            from train_Acc_arm import AccArmFeatureExtractor as ModelClass
        elif sensor == "Gyro_arm":
            from train_Gyro_arm import GyroArmFeatureExtractor as ModelClass
        elif sensor == "Mag_arm":
            from train_Mag_arm import MagArmFeatureExtractor as ModelClass
        else:
            raise ValueError(f"Unknown sensor: {sensor}")
        
        # Build model path
        model_path = FEATURE_EXTRACTORS_DIR / "Models" / "s2_w2_tremor" / "fs50_mixed3" / f"feature_extractor_{sensor}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = ModelClass().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"    ✓ Loaded from {model_path.name}")
        feature_extractors[sensor] = model
    
    return feature_extractors


def extract_features_for_subject1(feature_extractors: dict):
    """Load Subject 1 raw data and extract features using trained extractors."""
    
    print("\n" + "=" * 80)
    print("LOADING SUBJECT 1 RAW DATA")
    print("=" * 80)
    
    tremor_types = ["tremor_clean", "tremor_mild_mod", "tremor_mod_severe"]
    all_embeddings = []
    all_labels = []
    all_subjects = []
    
    for tremor_type in tremor_types:
        print(f"\nTremor type: {tremor_type}")
        print("-" * 80)
        
        variant_name = f"s2_w2_fs50_{tremor_type}"
        variant_dir = DATA_ROOT / variant_name
        
        # Load raw data for all sensors
        sensor_data = {}
        for sensor in SENSORS:
            data_file = variant_dir / f"{sensor}.npz"
            if not data_file.exists():
                print(f"  ⚠ {sensor} data not found: {data_file}")
                continue
            
            data = np.load(data_file)
            sensor_data[sensor] = {
                'X': data['X'],
                'subject_id': data['subject_id'],
                'tremor_score': data['tremor_score'],
                'activity': data['y']  # Activity labels stored as 'y'
            }
            print(f"  ✓ {sensor}: {data['X'].shape}")
        
        if not sensor_data:
            print(f"  ⚠ No sensor data loaded for {tremor_type}")
            continue
        
        # Get Subject 1 mask from first sensor
        first_sensor = list(sensor_data.keys())[0]
        subjects = sensor_data[first_sensor]['subject_id']
        s1_mask = subjects == 1
        s1_count = np.sum(s1_mask)
        
        if s1_count == 0:
            print(f"  ⚠ No Subject 1 data in {tremor_type}")
            continue
        
        print(f"  ✓ Found {s1_count} Subject 1 samples")
        
        # Extract features using trained extractors
        s1_embeddings = []
        
        for sensor in SENSORS:
            if sensor not in sensor_data:
                continue
            
            X = sensor_data[sensor]['X'][s1_mask]
            
            # Create dataset and loader
            dataset = RawDataDataset(
                X=X,
                y_tremor=sensor_data[sensor]['tremor_score'][s1_mask],
                y_activity=sensor_data[sensor]['activity'][s1_mask],
                y_subject=sensor_data[sensor]['subject_id'][s1_mask],
                seq_len=SEQ_LEN
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            
            # Extract embeddings
            embeddings = []
            with torch.no_grad():
                for x_batch, tremor_batch, activity_batch, subject_batch in loader:
                    x_batch = x_batch.to(DEVICE)
                    activity_batch = activity_batch.to(DEVICE)
                    
                    # Get embedding from feature extractor using extract_features method
                    emb = feature_extractors[sensor].extract_features(x_batch, activity_batch)
                    embeddings.append(emb.cpu().numpy())
            
            s1_embeddings.append(np.concatenate(embeddings, axis=0))
        
        # Concatenate sensor embeddings
        fused = np.concatenate(s1_embeddings, axis=1)
        s1_embeddings.append(fused)
        
        # Get labels
        s1_labels = (sensor_data[first_sensor]['tremor_score'][s1_mask] > 0).astype(np.int64)
        s1_subjects = sensor_data[first_sensor]['subject_id'][s1_mask]
        
        all_embeddings.append(fused)
        all_labels.append(s1_labels)
        all_subjects.append(s1_subjects)
        
        print(f"  Binary label distribution: No tremor={np.sum(s1_labels==0)}, Tremor={np.sum(s1_labels==1)}")
    
    # Concatenate all tremor types
    if not all_embeddings:
        print("\n⚠ No Subject 1 data found!")
        return None
    
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    final_subjects = np.concatenate(all_subjects, axis=0)
    
    print("\n" + "-" * 80)
    print(f"Total Subject 1 samples: {len(final_embeddings)}")
    print(f"Embedding dimension: {final_embeddings.shape[1]}")
    print(f"Label distribution: No tremor={np.sum(final_labels==0)}, Tremor={np.sum(final_labels==1)}")
    
    return final_embeddings, final_labels, final_subjects


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def run_inference(embeddings: np.ndarray, labels: np.ndarray):
    """Run inference with fusion classifier."""
    
    print("\n" + "=" * 80)
    print("LOADING FUSION CLASSIFIER")
    print("=" * 80)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = FeatureFusionClassifier(
        input_dim=TOTAL_EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Create dataset and loader
    class EmbeddingDataset(Dataset):
        def __init__(self, emb, lab):
            self.emb = emb.astype(np.float32)
            self.lab = lab.astype(np.int64)
        
        def __len__(self):
            return len(self.emb)
        
        def __getitem__(self, idx):
            return torch.from_numpy(self.emb[idx]), torch.tensor(self.lab[idx], dtype=torch.long)
    
    dataset = EmbeddingDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Inference
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for emb_batch, _ in loader:
            emb_batch = emb_batch.to(DEVICE)
            logits = model(emb_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    accuracy = accuracy_score(labels, all_preds)
    f1_macro = f1_score(labels, all_preds, average='macro', zero_division=0)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                No Tremor  Tremor")
    print(f"Actual No Tremor {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"Actual Tremor    {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        labels, all_preds,
        labels=[0, 1],
        target_names=['No Tremor', 'Tremor'],
        zero_division=0
    ))
    
    # Confidence
    print(f"\nModel Confidence:")
    print(f"  Mean prob (tremor class): {np.mean(all_probs[:, 1]):.4f}")
    print(f"  Min confidence: {np.min(np.max(all_probs, axis=1)):.4f}")
    print(f"  Max confidence: {np.max(np.max(all_probs, axis=1)):.4f}")
    
    # Save plot if available
    if HAS_PLOTTING:
        plot_dir = Path(__file__).parent / "subject1_evaluation"
        plot_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Tremor', 'Tremor'],
                   yticklabels=['No Tremor', 'Tremor'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Subject 1 - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(plot_dir / "confusion_matrix_subject1.png", dpi=150)
        print(f"\n✓ Plot saved to {plot_dir}/confusion_matrix_subject1.png")
        plt.close()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return {
        "accuracy": accuracy,
        "f1": f1_macro,
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": labels,
        "confusion_matrix": cm
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # Load feature extractors
        feature_extractors = load_feature_extractors()
        
        # Extract features for Subject 1
        result = extract_features_for_subject1(feature_extractors)
        
        if result is not None:
            embeddings, labels, subjects = result
            
            # Run inference
            results = run_inference(embeddings, labels)
        else:
            print("\n❌ Could not proceed - no Subject 1 data")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
