# Tremor Feature-Level Fusion Pipeline

This directory contains scripts for training CNN feature extractors and performing feature-level fusion for tremor classification.

## Architecture Overview

**Feature-Level Fusion** learns separate feature representations for each sensor, then combines them for classification.

```
Raw Data → [Acc_arm CNN] → 128-dim embedding ┐
                                              ├→ Concatenate (256-dim) → MLP Classifier → Score (0-4)
Raw Data → [Gyro_arm CNN] → 128-dim embedding ┘
```

## Workflow

### Step 1: Train Feature Extractors

Train separate CNNs for Acc_arm and Gyro_arm sensors:

```bash
# Train Acc_arm CNN
python Tremor_head/feature_extractors/Training/train_acc_arm_extractor.py

# Train Gyro_arm CNN
python Tremor_head/feature_extractors/Training/train_gyro_arm_extractor.py
```

**CNN Architecture:**
- Conv1d(3, 128) + BatchNorm + ReLU + MaxPool
- Conv1d(128, 256) + BatchNorm + ReLU + MaxPool
- Flatten + Activity Embedding (16-dim)
- fc_embed: Linear(6416, 128) → **128-dim embeddings**
- fc_cls: Linear(128, 5) for classification

**Training Data:**
- Merged from: `s2_w2_tremor_clean`, `s2_w2_tremor_parkinson_mild`, `s2_w2_tremor_parkinson_severe`
- Split by subject: Test=[5,10], Val=[2,7]
- Task: Classify tremor scores 0-4

**Output:**
- Models saved to: `Tremor_head/feature_extractors/models/`
  - `acc_arm_v20260226_143022.pth` (versioned with timestamp)
  - `gyro_arm_v20260226_143530.pth`
  - `acc_arm_best.pth` ← **BEST model ever** (used for embeddings)
  - `gyro_arm_best.pth` ← **BEST model ever** (used for embeddings)

**Model Versioning & Selection:**
- Each training run creates a versioned model with timestamp
- Script compares test accuracy with all previous runs (from JSONL log)
- **Only if new model beats previous best:** 
  - Updates `{sensor}_best.pth` (copy of best versioned model)
  - Prints "🎉 NEW BEST MODEL!" confirmation
- **If not better than previous best:**
  - Keeps old `{sensor}_best.pth` unchanged
  - You can still find versioned model in logs folder

**Why this matters:**
- You can retrain with different hyperparameters without losing best model
- Embeddings are always extracted from best performing model
- Full history preserved via versioned models

### Step 2: Extract Embeddings

Extract 128-dim embeddings from **best trained models**:

```bash
python Tremor_head/feature_extractors/extract_tremor_embeddings.py
```

**Process:**
- Loads **BEST models only** (`acc_arm_best.pth`, `gyro_arm_best.pth`)
- Ensures embeddings are from highest-performing models
- For each dataset variant:
  - Loads raw sensor data
  - Extracts embeddings using `model.extract_features()`
  - Saves to NPZ files

**Output:**
```
data/Tremor_datagenerator_files/
  s2_w2_tremor_clean/
    ExtractedFeatures/
      Acc_arm_embeddings.npz   (train/val/test splits)
      Gyro_arm_embeddings.npz
  s2_w2_tremor_parkinson_mild/
    ExtractedFeatures/...
  s2_w2_tremor_parkinson_severe/
    ExtractedFeatures/...
```

**NPZ File Contents:**
- `train_embeddings`: (N_train, 128)
- `val_embeddings`: (N_val, 128)
- `test_embeddings`: (N_test, 128)
- `train_labels/val_labels/test_labels`: Tremor scores (0-4)
- `train_activities/val_activities/test_activities`: Activity labels
- `train_subjects/val_subjects/test_subjects`: Subject IDs

### Step 3: Train Fusion Classifier

Train MLP classifier on concatenated embeddings:

```bash
python Tremor_head/tremor_classification_feature_fusion.py
```

**MLP Architecture:**
- Input: Concatenated embeddings (256-dim)
- Hidden: [128, 64] with dropout (0.3)
- Output: 5-class classification (scores 0-4)

**Output:**
- Model: `tremor_logs/best_tremor_feature_fusion_model.pth`
- Logs: `tremor_logs/tremor_feature_fusion_experiments.jsonl`
- Plots: `tremor_logs/feature_fusion_plots/`

## Comparison: Data-Level vs Feature-Level Fusion

### Data-Level Fusion
- **Approach:** Concatenate raw sensor data before CNN
- **Input:** (Acc_arm + Gyro_arm) = 6 channels × 100 timesteps
- **Model:** Single CNN processes all channels together
- **Pros:** Simpler, learns joint patterns
- **Cons:** May not capture sensor-specific features well

### Feature-Level Fusion (This Pipeline)
- **Approach:** Extract features separately, then combine
- **Input:** Separate CNNs for each sensor
- **Model:** Two CNNs → Concatenate embeddings → MLP
- **Pros:** Captures sensor-specific features, more flexible
- **Cons:** More models to train, higher complexity

## File Structure

```
Tremor_head/feature_extractors/
  ├── train_acc_arm_extractor.py      # Train Acc_arm CNN
  ├── train_gyro_arm_extractor.py     # Train Gyro_arm CNN
  ├── extract_tremor_embeddings.py    # Extract embeddings
  └── logs/
      ├── acc_arm_best.pth            # BEST Acc_arm model (used for embeddings)
      ├── gyro_arm_best.pth           # BEST Gyro_arm model (used for embeddings)
      ├── acc_arm_v20260226_143022.pth  # Versioned models (kept for history)
      ├── gyro_arm_v20260226_143530.pth
      ├── acc_arm_training.jsonl      # Training logs
      └── gyro_arm_training.jsonl

Tremor_head/
  └── tremor_classification_feature_fusion.py  # Fusion classifier

tremor_logs/
  ├── best_tremor_feature_fusion_model.pth
  ├── tremor_feature_fusion_experiments.jsonl
  └── feature_fusion_plots/
      ├── confusion_matrix_validation.png
      └── confusion_matrix_test.png
```

## Notes

- **Best Model Tracking:** Only best performing models (by test accuracy) are used for embeddings
- **Model Versioning:** All trained models are saved with timestamps for history
- **Same splits:** All models use identical train/val/test splits (by subject)
- **Embeddings:** 128-dim per sensor (activity-aware via embedding)
- **Dataset:** Merged clean + mild + severe for class balance
- **Reproducibility:** Fixed random seed (42) for consistency

## Tips

**Experimenting with hyperparameters:**
- Train multiple times with different settings (lr, dropout, etc.)
- Each run creates a versioned model (`acc_arm_vYYYYMMDD_HHMMSS.pth`)
- Best model is automatically updated if new run beats previous best
- Embeddings always use `acc_arm_best.pth` / `gyro_arm_best.pth`

**Checking model history:**
```bash
# View all training results
cat Tremor_head/feature_extractors/models/acc_arm_training.jsonl | jq '.test_acc'

# Find best model info
cat Tremor_head/feature_extractors/models/acc_arm_training.jsonl | jq 'select(.is_best == true)'
```
