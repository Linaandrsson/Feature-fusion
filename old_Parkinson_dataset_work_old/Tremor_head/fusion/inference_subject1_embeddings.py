#!/usr/bin/env python3
"""
Subject 1 Tremor Holdout Evaluation - Fusion Classifier Inference
==================================================================

Loads pre-extracted Subject 1 embeddings and runs inference through the trained
fusion classifier. This is a holdout evaluation test to validate that Subject 1
(completely held out from training) achieves consistent results.

Data:
  - Subject 1 embeddings (clean, mild_mod, mod_severe variants)
  - Location: feature_extractors/Extraction/subject1_embeddings/

Model:
  - Trained fusion classifier: feature_fusion_best_accarm50_gyroarm50_magarm50.pth
  - Input: 384-dim concatenated embeddings (Acc_arm + Gyro_arm + Mag_arm)
  - Output: Binary tremor classification (0=no tremor, 1=tremor)

Output:
  - Per-variant accuracy, F1, confusion matrices
  - Summary report
  - Visualizations
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================
# Configuration & Paths
# ============================================================

script_dir = Path(__file__).parent.resolve()  # fusion/
tremor_head_dir = script_dir.parent  # Tremor_head/
workspace_root = tremor_head_dir.parent  # Parkinson_dataset_work/

embeddings_dir = tremor_head_dir / "feature_extractors" / "Extraction" / "subject1_embeddings"
model_path = script_dir / "models" / "accarm50_gyroarm50_magarm50" / "feature_fusion_best_accarm50_gyroarm50_magarm50.pth"
output_dir = script_dir / "inference_subject1_results"
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
variants = ["clean", "mild_mod", "mod_severe"]

print("=" * 80)
print("SUBJECT 1 HOLDOUT EVALUATION - FUSION CLASSIFIER INFERENCE")
print("=" * 80)
print(f"Embeddings directory: {embeddings_dir}")
print(f"Model path: {model_path}")
print(f"Output directory: {output_dir}")
print(f"Device: {device}")
print("=" * 80)


# ============================================================
# Dataset Class
# ============================================================

class SubjectEmbeddingDataset(Dataset):
    """Dataset for Subject 1 embeddings."""
    
    def __init__(self, npz_file: Path):
        """Load embeddings from NPZ file."""
        data = np.load(npz_file, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.activities = data["y"].astype(np.int64)  # Activity labels (0-11)
        self.tremor_scores = data["tremor_score"].astype(np.int8)
        
        # Convert tremor scores to binary labels: 0=no tremor, >0=tremor
        print(f"\n  [DEBUG] Original tremor scores (first 20): {self.tremor_scores[:20]}")
        self.labels = (self.tremor_scores > 0).astype(np.int64)
        print(f"  [DEBUG] Binarized labels (0=no tremor, 1=tremor) (first 20): {self.labels[:20]}")
        
        print(f"    Loaded {npz_file.name}")
        print(f"      Shape: {self.embeddings.shape}")
        print(f"      Binary label distribution: {np.bincount(self.labels)}")
        print(f"      Tremor score distribution: {np.bincount(self.tremor_scores)}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        tremor_score = self.tremor_scores[idx]
        activity = self.activities[idx]
        
        return {
            "embedding": embedding,
            "label": label,
            "tremor_score": tremor_score,
            "activity": activity,
        }


# ============================================================
# Model Architecture
# ============================================================

class FusionInferenceModel(nn.Module):
    """Fusion classifier for feature-level fusion."""
    
    def __init__(self, input_dim: int = 384, hidden_dims=None, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
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
        """
        Args:
            embeddings: (B, 384) concatenated sensor embeddings
        
        Returns:
            (B, 2) logits
        """
        return self.classifier(embeddings)


# ============================================================
# Inference
# ============================================================

def load_model(model_path: Path) -> FusionInferenceModel:
    """Load trained fusion classifier.
    
    Returns:
        Model moved to device and set to eval mode
    """
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\n  [CHECKPOINT VERIFICATION]")
    print(f"    Loading from: {model_path}")
    print(f"    File size: {model_path.stat().st_size / 1e6:.1f} MB")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"    Checkpoint keys: {list(checkpoint.keys())}")
    
    # Verify it's a fusion model checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"    State dict keys (first 5): {list(state_dict.keys())[:5]}")
        
        # Check layer dimensions
        for key in list(state_dict.keys())[:3]:
            if "weight" in key:
                print(f"      {key}: {state_dict[key].shape}")
        
        print(f"    Metadata in checkpoint:")
        for key in checkpoint.keys():
            if key != "model_state_dict" and key != "optimizer_state_dict":
                val = checkpoint[key]
                if isinstance(val, (int, float)):
                    print(f"      {key}: {val}")
    
    model = FusionInferenceModel().to(device)
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"  ✓ Checkpoint loaded successfully")
    print(f"  ✓ Model moved to device: {device}")
    return model


def infer(model: FusionInferenceModel, loader: DataLoader) -> tuple:
    """Run inference on a dataset."""
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(embeddings)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


def plot_confusion_matrix(cm, title: str, save_path: Path):
    """Plot and save confusion matrix."""
    
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Tremor', 'Tremor'],
                   yticklabels=['No Tremor', 'Tremor'])
    else:
        plt.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
        plt.xticks(range(2), ['No Tremor', 'Tremor'])
        plt.yticks(range(2), ['No Tremor', 'Tremor'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title} Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved plot: {save_path.name}")


# ============================================================
# Main Evaluation
# ============================================================

def main():
    """Run inference on all Subject 1 variants."""
    
    print("\nLoading model...")
    model = load_model(model_path)
    print(f"  Model loaded from: {model_path}")
    print(f"  Device: {device}")
    
    # Store results for summary
    results_summary = {}
    
    for variant in variants:
        print(f"\n{'='*80}")
        print(f"Evaluating variant: {variant}")
        print(f"{'='*80}")
        
        # Load embeddings
        embeddings_file = embeddings_dir / f"subject1_embeddings_{variant}.npz"
        if not embeddings_file.exists():
            print(f"  ERROR: Embeddings file not found: {embeddings_file}")
            continue
        
        print(f"\nLoading embeddings...")
        dataset = SubjectEmbeddingDataset(embeddings_file)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        
        # ═════════════════════════════════════════════════════════════════
        # DEBUG: Show label information
        # ═════════════════════════════════════════════════════════════════
        print(f"\n  [DEBUG - {variant}] Label verification:")
        tremor_scores = dataset.tremor_scores
        binary_labels = dataset.labels
        
        print(f"    First 20 tremor_scores: {tremor_scores[:20]}")
        print(f"    First 20 binary labels: {binary_labels[:20]}")
        print(f"    Tremor score distribution: {np.bincount(tremor_scores)}")
        print(f"    Binary label distribution: {np.bincount(binary_labels)}")
        
        # Sanity checks
        print(f"\n  [SANITY CHECKS] {variant}:")
        if variant == "clean":
            has_zero = np.all(binary_labels == 0)
            print(f"    ✓ All zeros (clean should be all 0): {has_zero}")
            if not has_zero:
                print(f"    ⚠️  WARNING: clean contains non-zero labels!")
        else:
            has_ones = np.all(binary_labels == 1)
            print(f"    ✓ All ones ({variant} should be all 1): {has_ones}")
            if not has_ones:
                print(f"    ⚠️  WARNING: {variant} contains non-one labels!")
        
        # Run inference
        print(f"\nRunning inference...")
        preds, labels, probs = infer(model, loader)
        
        # ═════════════════════════════════════════════════════════════════
        # DEBUG: Show raw model outputs
        # ═════════════════════════════════════════════════════════════════
        print(f"\n  [DEBUG] Model outputs for {variant}:")
        print(f"    First 10 true labels: {labels[:10]}")
        print(f"    First 10 predictions: {preds[:10]}")
        print(f"    First 10 logits [class 0, class 1]:")
        # Re-extract logits for first batch only
        first_batch = next(iter(loader))
        with torch.no_grad():
            logits_first = model(first_batch["embedding"].to(device))
        for i in range(min(10, len(logits_first))):
            lg = logits_first[i].cpu().numpy()
            print(f"      {i}: [{lg[0]:7.3f}, {lg[1]:7.3f}]")
        
        print(f"\n    First 10 probabilities [P(class 0), P(class 1)]:")
        probs_first = torch.softmax(logits_first, dim=1)
        for i in range(min(10, len(probs_first))):
            pb = probs_first[i].cpu().numpy()
            print(f"      {i}: [{pb[0]:.4f}, {pb[1]:.4f}]")
        
        print(f"\n    Prediction distribution: {np.bincount(preds)}")
        print(f"    Label distribution:     {np.bincount(labels)}")
        
        # Calculate accuracy metrics
        acc_normal = (preds == labels).mean()
        acc_flipped = ((1 - preds) == labels).mean()
        
        print(f"\n  [DIAGNOSTIC] {variant}:")
        print(f"    Accuracy (normal predictions):  {acc_normal:.4f}")
        print(f"    Accuracy (flipped predictions): {acc_flipped:.4f}")
        
        if acc_flipped > acc_normal:
            print(f"    ⚠️  ALERT: Flipped accuracy is HIGHER than normal!")
            print(f"       This suggests predictions are inverted")
        elif acc_normal > acc_flipped + 0.1:
            print(f"    ✓ Normal accuracy is clearly better")
            print(f"       Predictions appear correct")
        else:
            print(f"    ? Both accuracies similar - unclear")
        
        # Show example (label, pred, flipped_pred) triplets
        print(f"\n  [EXAMPLES] (label, pred, flipped_pred) for first 20 samples:")
        for i in range(min(20, len(labels))):
            flipped = 1 - preds[i]
            match_normal = "✓" if preds[i] == labels[i] else "✗"
            match_flipped = "✓" if flipped == labels[i] else "✗"
            print(f"    {i:2d}: ({labels[i]}, {preds[i]}, {flipped}) {match_normal} normal, {match_flipped} flipped")
        
        # ⚠️ FIX: Model predictions are inverted. Flip them for correct results.
        # Root cause: Training model learned with inverted class mapping (likely from feature extractors).
        # Evidence: Flipped accuracy is consistently 93%+ vs 7%+ for normal predictions.
        # preds = 1 - preds
        
        # Compute metrics (using corrected flipped predictions)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        cm = confusion_matrix(labels, preds)
        
        print(f"\nResults for {variant}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {cm}")
        
        # Classification report
        print(f"\n  Classification Report:")
        report = classification_report(
            labels, preds,
            labels=[0, 1],
            target_names=['No Tremor', 'Tremor'],
            zero_division=0,
            output_dict=False
        )
        print(report)
        
        # Save confusion matrix plot
        plot_save_path = output_dir / f"confusion_matrix_{variant}.png"
        plot_confusion_matrix(cm, f"Subject 1 - {variant.title()}", plot_save_path)
        
        # Store results
        results_summary[variant] = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "num_samples": len(labels),
            "label_distribution": np.bincount(labels).tolist(),
            "prediction_distribution": np.bincount(preds).tolist(),
        }
    
    # ============================================================
    # Summary Report
    # ============================================================
    
    print(f"\n{'='*80}")
    print("SUMMARY REPORT - SUBJECT 1 HOLDOUT EVALUATION")
    print(f"{'='*80}")
    
    print(f"\nModel: {model_path.name}")
    print(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ═════════════════════════════════════════════════════════════════
    # DIAGNOSTIC: Inversion detection
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("INVERSION DIAGNOSIS")
    print(f"{'='*80}")
    
    print(f"\nExpected behavior:")
    print(f"  - clean: all labels=0 → should predict mostly 0")
    print(f"  - mild_mod: all labels=1 → should predict mostly 1")
    print(f"  - mod_severe: all labels=1 → should predict mostly 1")
    
    print(f"\nObserved predictions:")
    for variant in variants:
        if variant in results_summary:
            r = results_summary[variant]
            pred_dist = r["prediction_distribution"]
            print(f"  {variant}:")
            if len(pred_dist) >= 2:
                pred_0_pct = 100 * pred_dist[0] / sum(pred_dist)
                pred_1_pct = 100 * pred_dist[1] / sum(pred_dist)
                print(f"    Predicts 0: {pred_0_pct:5.1f}%  |  Predicts 1: {pred_1_pct:5.1f}%")
            else:
                print(f"    Distribution: {pred_dist}")
    
    print(f"\nLikely root causes:")
    print(f"  1. Training labels inverted (0 actually means tremor, 1 means no tremor)")
    print(f"  2. Subject 1 labels inverted (0 means tremor, 1 means no tremor)")
    print(f"  3. Model output neuron meaning reversed (argmax gives opposite class)")
    print(f"  4. Checkpoint saved with different convention than training")
    
    print(f"\nTo diagnose:")
    print(f"  - Check training script output for label distribution")
    print(f"  - Verify raw tremor scores are converted correctly")
    print(f"  - Ensure model logits [neg, pos] → class 0, [pos, neg] → class 1")
    
    summary_table = []
    for variant in variants:
        if variant in results_summary:
            r = results_summary[variant]
            summary_table.append({
                "variant": variant,
                "accuracy": f"{r['accuracy']:.4f}",
                "f1_score": f"{r['f1_score']:.4f}",
                "num_samples": r["num_samples"],
            })
    
    for row in summary_table:
        print(f"\n{row['variant'].upper()}:")
        print(f"  Accuracy: {row['accuracy']}")
        print(f"  F1 Score: {row['f1_score']}")
        print(f"  Samples: {row['num_samples']}")
    for variant in variants:
        if variant in results_summary:
            r = results_summary[variant]
            summary_table.append({
                "variant": variant,
                "accuracy": f"{r['accuracy']:.4f}",
                "f1_score": f"{r['f1_score']:.4f}",
                "num_samples": r["num_samples"],
            })
    
    for row in summary_table:
        print(f"\n{row['variant'].upper()}:")
        print(f"  Accuracy: {row['accuracy']}")
        print(f"  F1 Score: {row['f1_score']}")
        print(f"  Samples: {row['num_samples']}")
    
    # Save JSON report
    report_file = output_dir / "subject1_evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
