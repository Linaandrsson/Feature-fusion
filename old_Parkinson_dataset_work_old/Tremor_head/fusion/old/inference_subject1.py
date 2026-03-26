#!/usr/bin/env python3
"""
Inference Script: Load trained fusion model and evaluate on Subject 1 data.

This script loads the best trained binary tremor fusion classifier and
runs inference on Subject 1 data (held out from training).
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
    roc_auc_score,
    roc_curve
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
# Script is at: .../Parkinson_dataset_work/Tremor_head/fusion/inference_subject1.py
script_dir = Path(__file__).parent.resolve()  # .../Tremor_head/fusion
tremor_head_dir = script_dir.parent  # .../Tremor_head
WORKSPACE_ROOT = tremor_head_dir.parent  # .../Parkinson_dataset_work

TREMOR_HEAD_DIR = tremor_head_dir
MODELS_DIR = TREMOR_HEAD_DIR / "fusion" / "models" / "accarm50_gyroarm50_magarm50"
MODEL_PATH = MODELS_DIR / "feature_fusion_best_accarm50_gyroarm50_magarm50.pth"

DATA_ROOT = WORKSPACE_ROOT / "data" / "Tremor_datagenerator_files"
EXTRACTED_FEATURES_DIR = "Tremor_ExtractedFeatures"

SENSORS = ["Acc_arm", "Gyro_arm", "Mag_arm"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config (must match training)
EMBED_DIM_PER_SENSOR = 128
TOTAL_EMBED_DIM = len(SENSORS) * EMBED_DIM_PER_SENSOR  # 384
HIDDEN_DIMS = [256, 128, 64]
NUM_CLASSES = 2  # Binary: 0=no tremor, 1=tremor
DROPOUT = 0.3

print("=" * 80)
print("SUBJECT 1 EVALUATION - BINARY TREMOR CLASSIFICATION")
print("=" * 80)
print(f"\nWorkspace: {WORKSPACE_ROOT}")
print(f"Model: {MODEL_PATH}")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
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


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, subjects: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.subjects = subjects.astype(np.int64)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subject = torch.tensor(self.subjects[idx], dtype=torch.long)
        
        return embedding, label, subject


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_subject1_embeddings():
    """
    Load pre-extracted embeddings for all three sensors for Subject 1.
    Subject 1 data is from all three dataset variants (clean, mild_mod, mod_severe).
    """
    
    tremor_types = ["tremor_clean", "tremor_mild_mod", "tremor_mod_severe"]
    all_embeddings = []
    all_labels = []
    all_subjects = []
    
    print("\n" + "=" * 80)
    print("LOADING SUBJECT 1 EMBEDDINGS")
    print("=" * 80)
    
    for tremor_type in tremor_types:
        print(f"\nTremor type: {tremor_type}")
        print("-" * 80)
        
        variant_name = f"s2_w2_fs50_{tremor_type}"
        variant_embeddings = []
        variant_labels = None
        variant_subjects = None
        
        # Load embeddings from all three sensors
        for sensor in SENSORS:
            emb_path = DATA_ROOT / variant_name / EXTRACTED_FEATURES_DIR / f"{sensor}_embeddings.npz"
            
            if not emb_path.exists():
                raise FileNotFoundError(f"Embeddings not found: {emb_path}")
            
            data = np.load(emb_path)
            
            # Combine train, val, test
            all_emb = np.concatenate([
                data['train_embeddings'],
                data['val_embeddings'],
                data['test_embeddings']
            ], axis=0)
            
            all_lab = np.concatenate([
                data['train_labels'],
                data['val_labels'],
                data['test_labels']
            ], axis=0)
            
            all_subj = np.concatenate([
                data['train_subjects'],
                data['val_subjects'],
                data['test_subjects']
            ], axis=0)
            
            variant_embeddings.append(all_emb)
            if variant_labels is None:
                variant_labels = all_lab
                variant_subjects = all_subj
            
            print(f"  {sensor}: {all_emb.shape}")
        
        # Filter for subject 1
        subject1_mask = variant_subjects == 1
        num_s1 = np.sum(subject1_mask)
        
        if num_s1 == 0:
            print(f"  ⚠ No Subject 1 data in {tremor_type}")
            continue
        
        print(f"  ✓ Found {num_s1} Subject 1 samples in {tremor_type}")
        
        # Concatenate sensor embeddings and filter
        fused_emb = np.concatenate(variant_embeddings, axis=1)
        s1_embeddings = fused_emb[subject1_mask]
        s1_labels = variant_labels[subject1_mask]
        s1_subjects = variant_subjects[subject1_mask]
        
        # Convert tremor severity (0-5) to binary (0=no tremor, 1=tremor)
        s1_labels_binary = (s1_labels > 0).astype(np.int64)
        
        all_embeddings.append(s1_embeddings)
        all_labels.append(s1_labels_binary)
        all_subjects.append(s1_subjects)
        
        print(f"  Binary label distribution: No tremor={np.sum(s1_labels_binary==0)}, Tremor={np.sum(s1_labels_binary==1)}")
    
    # Concatenate all tremor types
    if not all_embeddings:
        print("\n⚠ No Subject 1 data found in any dataset variant!")
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

def evaluate_subject1():
    """Load model and evaluate on Subject 1 data."""
    
    # Load embeddings
    result = load_subject1_embeddings()
    if result is None:
        print("\n⚠ Cannot proceed - no Subject 1 data available")
        return
    
    embeddings, labels, subjects = result
    
    # Create dataset and loader
    dataset = EmbeddingDataset(embeddings, labels, subjects)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = FeatureFusionClassifier(
        input_dim=TOTAL_EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Random seed: {checkpoint['random_seed']}")
    print(f"  Val F1: {checkpoint['config'].get('val_f1', 'N/A')}")
    
    # Inference
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE ON SUBJECT 1")
    print("=" * 80)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings_batch, labels_batch, subjects_batch in loader:
            embeddings_batch = embeddings_batch.to(DEVICE)
            
            logits = model(embeddings_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                No Tremor  Tremor")
    print(f"Actual No Tremor {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"Actual Tremor    {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=['No Tremor', 'Tremor'],
        zero_division=0
    ))
    
    # ROC-AUC if we have both classes
    if len(np.unique(all_labels)) == 2:
        try:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")
    
    # Probability statistics
    print(f"\nPrediction Confidence:")
    print(f"  Mean confidence (tremor class): {np.mean(all_probs[:, 1]):.4f}")
    print(f"  Min confidence: {np.min(np.max(all_probs, axis=1)):.4f}")
    print(f"  Max confidence: {np.max(np.max(all_probs, axis=1)):.4f}")
    
    # Save results if plotting available
    if HAS_PLOTTING:
        plot_dir = Path(__file__).parent / "subject1_evaluation"
        plot_dir.mkdir(exist_ok=True)
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Tremor', 'Tremor'],
                   yticklabels=['No Tremor', 'Tremor'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Subject 1 - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(plot_dir / "confusion_matrix_subject1.png", dpi=150)
        print(f"\n✓ Confusion matrix plot saved to {plot_dir}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    try:
        results = evaluate_subject1()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
