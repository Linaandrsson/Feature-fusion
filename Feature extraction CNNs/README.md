# Feature Extraction CNNs - New Structure

This folder has been reorganized to separate training, models, and extraction. The new structure enables consistent paths across different data configurations and noise scenarios.

## 📁 Folder Structure

```
Feature extraction CNNs/
├── Training/              # CNN training scripts
│   ├── s1_w1_aug2_fs50/  # 1s stride, 1s window, 50Hz
│   ├── s0p5_w2_aug2_fs50/ # 0.5s stride, 2s window, 50Hz  
│   └── s2_w2_aug2_fs50/   # 2s stride, 2s window, 50Hz
├── Models/                # Saved trained models
│   └── {config}/
│       └── {variant}/
│           ├── feature_extractor_Acc_ankle.pth
│           ├── feature_extractor_Acc_arm.pth
│           └── ...
└── Extraction/            # Generic feature extraction scripts
    ├── config_extraction.py
    ├── extract_features.py
    ├── extract_all_sensors.py
    └── README.md
```

## 🔧 Configuration System

Each Training folder has a `config.py` that defines:

- **parent_dir**: Base data directory (contains shared splits)
- **variant_dir**: Specific data variant to train on (e.g., `fs50_clean`, `fs50_AWGN_AAn`)
- **seq_len**: Window size in samples (depends on window duration and sampling rate)
- **models_output_dir**: Where to save trained models

### Example: Training/s1_w1_aug2_fs50/config.py
```python
parent_dir = Path("../../data/Datagenerator_files/s1_w1_aug2")
variant_name = "fs50_clean"  # Change to train on different variant
variant_dir = parent_dir / variant_name
seq_len = 50  # 1 second at 50Hz
models_output_dir = Path("../../Models/s1_w1_aug2") / variant_name
```

## 🚀 Usage

### Step 1: Generate Data
```bash
# Generate clean data
cd "../.."  # Go to workspace root
python DataGenerator_v3.py
```

This creates:
```
data/Datagenerator_files/
└── s1_w1_aug2/              # Parent (shared splits)
    ├── fs50_clean/          # Variant (sensor data)
    │   ├── Acc_ankle.txt
    │   ├── Acc_arm.txt
    │   └── ...
    └── splits.npz
```

### Step 2: Train Models

Navigate to the appropriate Training folder:
```bash
cd "Feature extraction CNNs/Training/s1_w1_aug2_fs50"
```

Train a single sensor:
```bash
python Acc_ankle_CNN.py
```

Or train all sensors:
```bash
for script in *_CNN.py ECG_chest.py; do
    echo "Training $script..."
    python "$script"
done
```

Models are saved to:
```
Models/s1_w1_aug2/fs50_clean/
├── feature_extractor_Acc_ankle.pth
├── feature_extractor_Acc_arm.pth
└── ...
```

### Step 3: Extract Features

```bash
cd "../Extraction"

# Configure which models and data to use
# Edit config_extraction.py:
#   - model_config = "s1_w1_aug2"
#   - model_variant = "fs50_clean"
#   - variant_name = "fs50_clean"

# Extract features for all sensors
python extract_all_sensors.py
```

Features are saved to:
```
data/Datagenerator_files/s1_w1_aug2/fs50_clean/ExtractedFeatures/
├── Acc_ankle_features_train.npz
├── Acc_ankle_features_val.npz
├── Acc_ankle_features_test.npz
└── ...
```

## 🔄 Training on Different Variants

To train models on noisy data:

1. **Generate noisy variant**:
   ```python
   # In DataGenerator_v3.py, set:
   SCENARIO_NOISE_TYPE = "AWGN"
   SCENARIO_NOISE_POOL = ["Acc_ankle", "Acc_arm"]
   ```
   
   This creates: `data/Datagenerator_files/s1_w1_aug2/fs50_AWGN_AAn_AAr/`

2. **Update training config**:
   ```python
   # In Training/s1_w1_aug2_fs50/config.py:
   variant_name = "fs50_AWGN_AAn_AAr"
   ```

3. **Train models** (same commands as Step 2)

Models are saved to: `Models/s1_w1_aug2/fs50_AWGN_AAn_AAr/`

## 📊 Model Architecture

All CNN models follow the same architecture:

```python
Input: (batch, channels, seq_len)
  ↓
Conv1d(channels → 128) + BN + ReLU + MaxPool(2) + Dropout(0.2)
  ↓
Conv1d(128 → 128) + BN + ReLU + MaxPool(2) + Dropout(0.3)
  ↓
Flatten → Linear(flattened → 128)  # ← This is the embedding
  ↓
Dropout(0.5) → Linear(128 → num_classes)
```

**Embedding dimension**: 128 (extracted from `fc_embed` layer)

## 🔍 File Naming Convention

### Sensor Files
- `Acc_ankle.txt` - Accelerometer on ankle (3 channels)
- `Acc_arm.txt` - Accelerometer on wrist (3 channels)
- `Acc_chest.txt` - Accelerometer on chest (3 channels)
- `ECG.txt` - ECG on chest (2 leads)
- `Gyro_ankle.txt` - Gyroscope on ankle (3 channels)
- `Gyro_arm.txt` - Gyroscope on wrist (3 channels)
- `Mag_ankle.txt` - Magnetometer on ankle (3 channels)
- `Mag_arm.txt` - Magnetometer on wrist (3 channels)

### Model Checkpoints
Format: `feature_extractor_{sensor_name}.pth`

Contains:
- `model_state_dict`: Model weights
- `num_classes`: 12 (activity classes)
- `seq_len`: Window size in samples
- `num_channels`: Sensor channels

## 🧪 Verifying Configuration

Run verification script:
```bash
cd Training
python verify_configs.py
```

This checks:
- ✅ parent_dir exists and contains splits
- ✅ variant_dir exists and contains sensor data
- ✅ models_output_dir is created
- 📏 seq_len is correct for window size

## 📝 Migration from Old Structure

See [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for details on changes from the old structure.

### Key Changes:
1. **Separated concerns**: Training, Models, Extraction in different folders
2. **Hierarchical data**: parent/variant structure enables split sharing
3. **Config-based paths**: All paths imported from `config.py`
4. **Consistent model saving**: Models saved to `Models/{config}/{variant}/`

## ⚠️ Common Issues

### "parent_dir NOT found"
You need to generate data first using `DataGenerator_v3.py`.

### "variant_dir not yet created"
This is expected if you haven't generated that specific variant. Update DataGenerator config and run it.

### "splits.npz not found"
Run `SplitIndexes.py` after generating data to create splits.

### "Model checkpoint not found" (during extraction)
You need to train models first. Go to Training folder and run CNN scripts.

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Generate data | `python DataGenerator_v3.py` |
| Create splits | `python SplitIndexes.py` |
| Train single sensor | `cd Training/s1_w1_aug2_fs50 && python Acc_ankle_CNN.py` |
| Train all sensors | `for f in *_CNN.py ECG_chest.py; do python "$f"; done` |
| Extract features | `cd Extraction && python extract_all_sensors.py` |
| Verify configs | `cd Training && python verify_configs.py` |
