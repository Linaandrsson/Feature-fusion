#!/usr/bin/env python3
"""
Debug script to trace label inversion through training vs Subject 1 inference.

Key insight: Training learns correctly, but Subject 1 inference is inverted.
This points to either:
1. Subject 1 embeddings having inverted signal
2. Subject 1 labels being mislabeled  
3. Model being saved/loaded incorrectly

This script will help identify WHERE the inversion occurs.
"""

import numpy as np
import torch
from pathlib import Path

# Paths
workspace = Path("/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work")
tremor_head = workspace / "Tremor_head"
fusion_dir = tremor_head / "fusion"
feature_extractors_dir = tremor_head / "feature_extractors"

# Training data path
training_embeddings_dir = workspace / "data" / "Tremor_datagenerator_files" / "s2_w2_fs50_tremor_clean" / "Tremor_ExtractedFeatures"

# Subject 1 embeddings
subject1_embeddings_dir = feature_extractors_dir / "Extraction" / "subject1_embeddings"

print("="*80)
print("DEBUG: LABEL INVERSION INVESTIGATION")
print("="*80)

# ============================================================
# STEP 1: Inspect Training Embeddings
# ============================================================
print("\n" + "="*80)
print("STEP 1: Training Embeddings (Acc_arm clean variant)")
print("="*80)

train_file = training_embeddings_dir / "Acc_arm_embeddings.npz"
if train_file.exists():
    train_data = np.load(train_file)
    print(f"\nTraining embeddings file: {train_file.name}")
    print(f"  Keys: {list(train_data.keys())}")
    print(f"  Train embeddings shape: {train_data['train_embeddings'].shape}")
    print(f"  Train labels range: [{train_data['train_labels'].min()}, {train_data['train_labels'].max()}]")
    print(f"  First 10 train labels: {train_data['train_labels'][:10]}")
    print(f"  Label distribution: {np.bincount(train_data['train_labels'])}")
    print(f"  Embedding stat (train): min={train_data['train_embeddings'].min():.3f}, max={train_data['train_embeddings'].max():.3f}, mean={train_data['train_embeddings'].mean():.3f}")
else:
    print(f"  ERROR: File not found: {train_file}")

# ============================================================
# STEP 2: Inspect Subject 1 Embeddings
# ============================================================
print("\n" + "="*80)
print("STEP 2: Subject 1 Embeddings")
print("="*80)

for variant in ["clean", "mild_mod", "mod_severe"]:
    subject1_file = subject1_embeddings_dir / f"subject1_embeddings_{variant}.npz"
    if subject1_file.exists():
        s1_data = np.load(subject1_file, allow_pickle=True)
        print(f"\n{variant}:")
        print(f"  Keys: {list(s1_data.keys())}")
        print(f"  Embeddings shape: {s1_data['embeddings'].shape}")
        tremor_scores = s1_data['tremor_score']
        print(f"  Tremor score range: [{tremor_scores.min()}, {tremor_scores.max()}]")
        print(f"  First 10 tremor_scores: {tremor_scores[:10]}")
        print(f"  Tremor score distribution: {np.bincount(tremor_scores)}")
        
        # Binarized labels like in inference
        binary_labels = (tremor_scores > 0).astype(np.int64)
        print(f"  Binary labels (tremor_score>0): {np.bincount(binary_labels)}")
        print(f"  Embedding stat: min={s1_data['embeddings'].min():.3f}, max={s1_data['embeddings'].max():.3f}, mean={s1_data['embeddings'].mean():.3f}")
    else:
        print(f"\n{variant}: FILE NOT FOUND")

# ============================================================
# STEP 3: Run Quick Inference and Check
# ============================================================
print("\n" + "="*80)
print("STEP 3: Load Model and Run Inference on Subject 1 Clean")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model_path = fusion_dir / "models" / "accarm50_gyroarm50_magarm50" / "feature_fusion_best_accarm50_gyroarm50_magarm50.pth"
print(f"\nLoading model from: {model_path}")
print(f"  File exists: {model_path.exists()}")
print(f"  File size: {model_path.stat().st_size / 1e6:.1f} MB" if model_path.exists() else "")

# Load checkpoint
if model_path.exists():
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    
    if "model_state_dict" in checkpoint:
        print(f"  Trained epochs: {checkpoint.get('epochs_trained', 'N/A')}")
        print(f"  Val accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # Load Subject 1 embeddings
    subject1_clean = subject1_embeddings_dir / "subject1_embeddings_clean.npz"
    if subject1_clean.exists():
        s1_data = np.load(subject1_clean, allow_pickle=True)
        embeddings = s1_data['embeddings'][:20].astype(np.float32)
        tremor_scores = s1_data['tremor_score'][:20]
        true_labels = (tremor_scores > 0).astype(np.int64)
        
        print(f"\nRunning inference on first 20 Subject 1 clean samples:")
        print(f"  Tremor scores: {tremor_scores}")
        print(f"  True binary labels: {true_labels}")
        
        # Convert to tensor
        X_batch = torch.from_numpy(embeddings).to(device)
        
        # Define model architecture (simplified)
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(384, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(64, 2)
                )
            def forward(self, x):
                return self.net(x)
        
        model = SimpleModel().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        with torch.no_grad():
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        print(f"\nModel predictions:")
        print(f"  Predictions: {preds}")
        print(f"  Match normal: {(preds == true_labels).astype(int)} (✓ = correct)")
        print(f"  Match flipped: {(preds == 1 - true_labels).astype(int)} (✓ = inverted)")
        
        print(f"\nLogits (first 5 samples):")
        for i in range(min(5, len(logits))):
            lg = logits[i].cpu().numpy()
            print(f"  Sample {i}: [class0={lg[0]:7.3f}, class1={lg[1]:7.3f}]  " +
                  f"true_label={true_labels[i]} pred={preds[i]} match={preds[i]==true_labels[i]}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
