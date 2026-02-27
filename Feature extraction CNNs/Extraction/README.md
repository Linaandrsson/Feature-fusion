# Feature Extraction Guide

## Overview

This directory contains generic scripts for extracting features from sensor data using pre-trained CNN models.

## Structure

```
Extraction/
├── config_extraction.py      # Configuration (which data + which models to use)
├── extract_features.py        # Extract features for single sensor
├── extract_all_sensors.py     # Extract features for all sensors
└── README.md                  # This file
```

## Quick Start

### 1. Configure extraction

Edit `config_extraction.py`:

```python
# Which data to extract features FROM
parent_dir = Path(".../s1_w1_aug2")
variant_name = "fs50_AWGN_s0p3_AAnAC_n1"  # Change this!

# Which models to USE
model_config = "s1_w1_aug2"
model_variant = "fs50_clean"  # Models trained on clean data
```

### 2. Extract features for one sensor

```bash
python extract_features.py --sensor Acc_ankle
```

### 3. Extract features for all sensors

```bash
python extract_all_sensors.py
```

## Output

Features are saved to: `variant_dir/ExtractedFeatures/`

```
s1_w1_aug2/
  fs50_AWGN_s0p3_AAnAC_n1/
    ExtractedFeatures/
      Acc_ankle_train.npz    # embeddings, labels
      Acc_ankle_val.npz
      Acc_ankle_test.npz
      ... (all sensors)
```

## Common Use Cases

### Extract features from noisy data using clean models

```python
# config_extraction.py
variant_name = "fs50_AWGN_s0p3_AAnAC_n1"  # Noisy data
model_variant = "fs50_clean"               # Clean models
```

### Extract features from different FS using same models

```python
# config_extraction.py
parent_dir = Path(".../s1_w1_aug2")
variant_name = "fs10_clean"        # 10 Hz data
model_variant = "fs50_clean"       # Models trained on 50 Hz

# Note: This only works if window SIZE (in samples) is the same!
```

### Use models trained on noisy data

```python
# config_extraction.py
model_variant = "fs50_AWGN_s0p3_AAnAC_n1"  # Models trained on noisy
variant_name = "fs50_clean"                 # Extract from clean
```

## Notes

- Splits (train/val/test) are loaded from parent directory
- Models must match the data's seq_len (window size in samples)
- Output features maintain index alignment across sensors
- No shuffling during extraction for proper alignment

## Troubleshooting

**Model not found:**
- Check that models exist in `../Models/{model_config}/{model_variant}/`
- Ensure model files are named `feature_extractor_{sensor_name}.pth`

**Data not found:**
- Check that variant directory exists
- Ensure sensor .txt files exist in variant directory

**Dimension mismatch:**
- Model's seq_len must match data's seq_len
- Check num_channels in config matches sensor
