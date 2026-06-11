# CNN Feature Extractors - Training Pipeline

This directory contains CNN-based feature extractors for IMU sensor classification. Each sensor has its own dedicated CNN that learns to extract 128-dimensional embeddings for downstream fusion tasks.

---

## Table of Contents
1. [Configuration](#configuration)
2. [Directory Structure](#directory-structure)
3. [CNN Architecture](#cnn-architecture)
4. [Data Splits](#data-splits)
5. [Training Process](#training-process)
6. [Output Files](#output-files)

---

## 1. Configuration

**Config file:** [`config.py`](config.py)

### Dataset Configuration
```python
dataset_config = "s2_w2"  # 2-second windows, 2-second stride

# Combined tremor augmentations (all 3 variants)
variant_names = [
    "s2_w2_fs50_tremor_clean",         # 100% score=0 (no tremor)
    "s2_w2_fs50_tremor_mild_mod",      # 50% score=1, 50% score=2
    "s2_w2_fs50_tremor_mod_severe"     # 50% score=3, 50% score=4
]

seq_len = 100  # Sequence length (2 seconds × 50 Hz)
```

### Data Loading
- **Source:** `data/Tremor_datagenerator_files/[variant_name]/`
- **Function:** `load_combined_sensor_data(sensor_filename)` combines all 3 variants

**Data format per variant:**
```
[...sensor_data...] | activity | subject | base_idx | tremor_freq | tremor_acc_rms | tremor_gyro_rms | tremor_score
     300 columns        1         1         1            1              1                 1               1
```

### Output Configuration
```python
# Models saved to:
models_output_dir = "Feature extraction CNNs/Models/s2_w2/fs50_tremor_mixed_all3/"

# Embeddings saved to (per variant):
embeddings_base_dir = "data/Tremor_datagenerator_files/"
embeddings_folder_name = "ExtractedFeatures"
```

---

## 2. Directory Structure

```
s2_w2_aug2_fs50_mixed/
│
├── config.py                   # Shared configuration for all sensors
│
├── Acc_ankle_CNN.py            # CNN for ankle accelerometer
├── Acc_arm_CNN.py              # CNN for arm accelerometer
├── Acc_chest_CNN.py            # CNN for chest accelerometer
├── ECG_chest.py                # CNN for ECG (2 channels)
├── Gyro_ankle_CNN.py           # CNN for ankle gyroscope
├── Gyro_arm_CNN.py             # CNN for arm gyroscope
├── Mag_ankle_CNN.py            # CNN for ankle magnetometer
├── Mag_arm_CNN.py              # CNN for arm magnetometer
│
├── splits/                     # Train/val/test indices (shared across sensors)
│   ├── train_idx.txt           # 70% of combined data
│   ├── val_idx.txt             # 15% of combined data
│   └── test_idx.txt            # 15% of combined data
│
└── Misclassified/              # Analysis of misclassified samples
    ├── Acc_ankle_misclassified.csv
    ├── Acc_arm_misclassified.csv
    └── ...
```

**8 sensors total:**
- 3 Accelerometers (Acc_ankle, Acc_arm, Acc_chest)
- 2 Gyroscopes (Gyro_ankle, Gyro_arm)
- 2 Magnetometers (Mag_ankle, Mag_arm)
- 1 ECG (ECG_chest)

---

## 3. CNN Architecture

### Model: `IMUCNN`

**Input:** `(batch_size, num_channels, seq_len)`
- Most sensors: `(batch, 3, 100)` 
- ECG only: `(batch, 2, 100)`

**Architecture:**

```python
class IMUCNN(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),              # seq_len → seq_len/2
            nn.Dropout(0.4),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),              # seq_len/2 → seq_len/4
            nn.Dropout(0.3),
        )
        
        # Embedding head → 128-dimensional features
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(flattened_dim, 128)
        
        # Classification head (for training only)
        self.drop_cls = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(128, num_classes)  # 12 activities
```

**Output shapes:**
- `extract_features()`: `(batch, 128)` → **Stable embeddings for fusion**
- `forward()`: `(batch, 12)` → **Logits for activity classification**

**Parameter calculation:**
```
Conv1: in=3,  out=128, kernel=5 → ~3,840 params
Conv2: in=128, out=256, kernel=5 → ~163,840 params
FC embed: in=6,400 (256×25), out=128 → ~819,200 params
FC classifier: in=128, out=12 → ~1,548 params
---
Total: ~1M parameters per sensor CNN
```

---

## 4. Data Splits

### Split Strategy
**Method:** Stratified train/val/test split on **combined** data from all 3 variants

```python
# Combined dataset size: ~9,939 samples (3,313 × 3 variants)
Total samples from all 3 tremor augmentations

70% train (shuffle=True during training)
15% val   (shuffle=False)
15% test  (shuffle=False)
```

**Stratification:** Ensures equal activity distribution across splits

### Shared Split Indices
- **Location:** `splits/train_idx.txt`, `splits/val_idx.txt`, `splits/test_idx.txt`
- **Format:** One index per line (integer)
- **Purpose:** All 8 sensors use **identical splits** for consistency

### Per-Variant Splits (for embeddings)
After training on combined data, embeddings are extracted separately for each variant:
```
data/Tremor_datagenerator_files/
├── s2_w2_fs50_tremor_clean/
│   ├── splits/
│   │   ├── train_idx.txt    # 70% of clean variant only
│   │   ├── val_idx.txt      # 15%
│   │   └── test_idx.txt     # 15%
│   └── ExtractedFeatures/
│       └── Acc_ankle_embeddings.npz
│
├── s2_w2_fs50_tremor_mild_mod/
│   └── (same structure)
│
└── s2_w2_fs50_tremor_mod_severe/
    └── (same structure)
```

**Why?** Fusion models need tremor-specific splits to evaluate performance on each tremor severity separately.

---

## 5. Training Process

### Hyperparameters
```python
batch_size = 64
epochs = 200
lr = 1e-3           # Adam optimizer
patience = 10       # Early stopping
min_delta = 1e-4    # Minimum improvement threshold
```

### Early Stopping
Training stops if validation accuracy doesn't improve by `min_delta` for `patience` epochs.

### Training Loop
1. **Load combined data** from all 3 tremor variants
2. **Generate/load splits** (70/15/15 ratio)
3. **Train CNN** on activity classification (12 classes)
4. **Monitor validation accuracy** for early stopping
5. **Evaluate on test set** after training completes

### Best Model Selection
**Criterion:** New best if `test_accuracy > previous_best`

Location of previous best tracking:
```
/Volumes/NO NAME/Master Lina/Code/best_accuracies.json

{
  "s2_w2_aug2_fs50_mixed/Acc_ankle": 0.8732,
  "s2_w2_aug2_fs50_mixed/Acc_arm": 0.9145,
  ...
}
```

**If accuracy improves:**
- ✅ Save model to `Models/` directory
- ✅ Save embeddings for all 3 variants
- ✅ Save confusion matrices
- ✅ Save misclassified samples

**If accuracy does NOT improve:**
- ❌ Skip all save operations
- ❌ Exit early

---

## 6. Output Files

### 6.1 Trained Models
**Location:** `Feature extraction CNNs/Models/s2_w2/fs50_tremor_mixed_all3/`

**Files:**
```
feature_extractor_Acc_ankle.pth
feature_extractor_Acc_arm.pth
feature_extractor_Acc_chest.pth
feature_extractor_ECG_chest.pth
feature_extractor_Gyro_ankle.pth
feature_extractor_Gyro_arm.pth
feature_extractor_Mag_ankle.pth
feature_extractor_Mag_arm.pth
```

**Content (PyTorch checkpoint):**
```python
{
    "model_state_dict": OrderedDict(...),  # Trained weights
    "num_classes": 12,
    "seq_len": 100,
    "num_channels": 3,  # or 2 for ECG
    "embedding_dim": 128
}
```

**Loading example:**
```python
checkpoint = torch.load("feature_extractor_Acc_ankle.pth")
model = IMUCNN(
    num_classes=checkpoint["num_classes"],
    seq_len=checkpoint["seq_len"],
    num_channels=checkpoint["num_channels"]
)
model.load_state_dict(checkpoint["model_state_dict"])
```

---

### 6.2 Extracted Embeddings (128-dim features)

**Location (per variant):** `data/Tremor_datagenerator_files/[variant_name]/ExtractedFeatures/`

**Files:**
```
s2_w2_fs50_tremor_clean/ExtractedFeatures/
├── Acc_ankle_embeddings.npz
├── Acc_arm_embeddings.npz
├── Acc_chest_embeddings.npz
├── ECG_chest_embeddings.npz
├── Gyro_ankle_embeddings.npz
├── Gyro_arm_embeddings.npz
├── Mag_ankle_embeddings.npz
└── Mag_arm_embeddings.npz

(Same structure for fs50_tremor_mild_mod and fs50_tremor_mod_severe variants)
```

**NPZ content:**
```python
npz = np.load("Acc_ankle_embeddings.npz")

# Embeddings (128-dim features)
npz['train_embeddings']  # Shape: (N_train, 128)
npz['val_embeddings']    # Shape: (N_val, 128)
npz['test_embeddings']   # Shape: (N_test, 128)

# Tremor labels (for fusion classification)
npz['train_labels']      # Tremor severity scores (0-4)
npz['val_labels']
npz['test_labels']

# Activity labels (1-12)
npz['train_activities']
npz['val_activities']
npz['test_activities']

# Subject IDs (1-10)
npz['train_subjects']
npz['val_subjects']
npz['test_subjects']
```

**Usage in fusion:**
```python
# Load embeddings from all 8 sensors
acc_ankle = np.load("Acc_ankle_embeddings.npz")
acc_arm = np.load("Acc_arm_embeddings.npz")
# ... load all 8 sensors

# Concatenate for fusion
X_train = np.concatenate([
    acc_ankle['train_embeddings'],
    acc_arm['train_embeddings'],
    # ... all 8 sensors
], axis=1)  # Shape: (N_train, 1024) for 8×128

# Labels for tremor classification
y_train = acc_ankle['train_labels']  # Same across all sensors
```

---

### 6.3 Confusion Matrices

**Location:** `/Users/linaandersson/Desktop/master/Confusion_Matrixes/s2_w2_aug2_fs50_mixed/`

**Files (per sensor):**
```
Acc_ankle_matrix.png           # Counts
Acc_ankle_norm_matrix.png      # Normalized per class

Acc_arm_matrix.png
Acc_arm_norm_matrix.png

... (8 sensors × 2 matrices = 16 PNG files)
```

**Matrix dimensions:** `12×12` (activity classification)

---

### 6.4 Misclassified Samples

**Location:** `Misclassified/`

**Files:**
```
Acc_ankle_misclassified.csv
Acc_arm_misclassified.csv
...
```

**CSV format:**
```
test_index, true_label, predicted_label
      523,          4,               3
     1042,          7,               8
      ...
```

**Usage:** Analyze which activity pairs are commonly confused

---

### 6.5 Accuracy Tracking

**Workspace root files:**

**`accuracy_history.json`** (All training runs):
```json
{
  "s2_w2_aug2_fs50_mixed/Acc_ankle": [0.8532, 0.8634, 0.8732],
  "s2_w2_aug2_fs50_mixed/Acc_arm": [0.9045, 0.9112, 0.9145]
}
```

**`best_accuracies.json`** (Current best per sensor):
```json
{
  "s2_w2_aug2_fs50_mixed/Acc_ankle": 0.8732,
  "s2_w2_aug2_fs50_mixed/Acc_arm": 0.9145
}
```

---

## Running the Pipeline

### Train a Single Sensor
```bash
cd "/Volumes/NO NAME/Master Lina/Code/Feature extraction CNNs/Training/s2_w2_aug2_fs50_mixed"
python Acc_ankle_CNN.py
```

### Train All Sensors
```bash
for sensor in Acc_ankle Acc_arm Acc_chest ECG_chest Gyro_ankle Gyro_arm Mag_ankle Mag_arm; do
    python ${sensor}_CNN.py
done
```

**Expected outputs per sensor:**
1. ✅ Model checkpoint in `Models/` directory
2. ✅ Embeddings (3 NPZ files, one per tremor variant)
3. ✅ 2 confusion matrices (counts + normalized)
4. ✅ Misclassified samples CSV
5. ✅ Updated accuracy history

---

## Key Design Decisions

### 1. Why Combine All 3 Tremor Variants for Training?
- **More data** → Better generalization (9,939 samples vs 3,313)
- **Tremor-invariant features** → CNNs learn activity patterns regardless of tremor severity
- **Consistent splits** → Fair comparison across sensors

### 2. Why Separate Embeddings Per Variant?
- **Fusion evaluation** requires tremor-specific test sets
- **Ablation studies** can compare performance on clean vs mild vs severe data
- **Deployment flexibility** → Can use models on any tremor severity

### 3. Why 128-dim Embeddings?
- **Compact representation** for fusion (8 sensors × 128 = 1024 features)
- **Prevents curse of dimensionality** in fusion classifier
- **Fast inference** for real-time applications

### 4. Why Train on Activity Classification?
- **Pretext task** forces CNN to learn meaningful temporal patterns
- **Transfer learning** → Embeddings trained on activities generalize to tremor classification
- **Multi-task potential** → Can classify both activity AND tremor severity

---

## Next Steps: Feature Fusion

After extracting embeddings from all 8 sensors:

1. Navigate to `Feature fusion/` directory
2. Load embeddings using `np.load()`
3. Concatenate sensor features (8 × 128 = 1024-dim)
4. Train fusion classifier for tremor severity (5 classes: score 0-4)
5. Evaluate on test set per variant

---

**Last updated:** March 2026  
**Training config:** s2_w2 (2s windows, 50 Hz), 3 tremor variants combined
