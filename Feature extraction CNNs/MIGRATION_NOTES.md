# Feature Extractor Training

## New Structure Overview

```
Feature extraction CNNs/
├── Training/              # Training code per configuration
│   ├── s1_w1_aug2_fs50/  # One folder per (stride, window, aug, fs)
│   ├── s0p5_w2_aug2_fs50/
│   └── ...
│
├── Models/                # Saved models (output)
│   └── s1_w1_aug2/       # Organized by base config
│       ├── fs50_clean/   # Which variant trained on
│       └── ...
│
└── Extraction/            # Feature extraction using trained models
    ├── extract_features.py
    └── config_extraction.py
```

## Changes Made

### 1. Reorganized Structure
- **Training/**: Training scripts per configuration (organized by seq_len + fs)
- **Models/**: Centralized model storage (organized by base_config / variant)
- **Extraction/**: Generic feature extraction scripts

### 2. Updated Training Scripts
- Models now save to `Models/{base_config}/{variant}/`
- Config updated to use hierarchical data structure
- Paths relative to new location

### 3. Created Generic Extraction
- `extract_features.py`: Extract for single sensor
- `extract_all_sensors.py`: Batch process all sensors
- `config_extraction.py`: Configure which data + models to use

## Migration Notes

### Old Structure (deprecated)
```
Feature extractors/
  fs50_s1_w1_aug2/
    Acc_ankle_CNN.py
    feature_extractor_Acc_ankle.pth  # Model saved here
```

### New Structure
```
Training/
  s1_w1_aug2_fs50/
    Acc_ankle_CNN.py          # Training script
    config.py                 # Which data variant to train on

Models/
  s1_w1_aug2/
    fs50_clean/
      feature_extractor_Acc_ankle.pth  # Models saved here
```

## Next Steps

1. **Train new models**: Use scripts in `Training/` folders
2. **Extract features**: Use `Extraction/extract_features.py`
3. **Update fusion code**: Point to new `Models/` structure

## Backward Compatibility

Old `Feature extractors/` folder is preserved but should not be used for new work.
