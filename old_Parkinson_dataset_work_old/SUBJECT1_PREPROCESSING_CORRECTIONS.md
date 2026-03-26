# Subject 1 Preprocessing Script - Corrections Summary

**Date**: March 24, 2026  
**Purpose**: Ensure `process_subject1_tremor_dataset.py` is **bitwise-equivalent** to the original `process_parkinson_tremor_dataset.py`

---

## Critical Issues Fixed

### 1. ✅ Tremor Cache Generation (MOST CRITICAL)

**❌ BEFORE (INCORRECT)**:
- Built tremor cache ONCE using only `"mod_severe"` augment_mode
- Manually remapped tremor_score for each variant
- Result: All windows had tremor_score in {3, 4}, remapped to different values per variant

**✅ AFTER (CORRECT)**:
- **Build fresh tremor cache FOR EACH VARIANT** inside `generate_variant()`
- New function parameter: `augment_mode` passed to `build_tremor_cache_for_subject1()`
- Each variant generates tremor samples with correct score distribution:
  - `clean`: tremor_score from {0} (naturally generated)
  - `mild_mod`: tremor_score from {1, 2} (naturally generated)
  - `mod_severe`: tremor_score from {3, 4} (naturally generated)

**Code location**: Lines ~265-315 (new `build_tremor_cache_for_subject1` signature)

---

### 2. ✅ Magnetometer Rotation-Based Augmentation

**❌ BEFORE (INCORRECT)**:
```python
if signal_cache_entry is not None and sensor_type in ["acc", "gyro", "mag"] and augment_mode != "clean":
    noise_key = f"{sensor_type}_noise"
    win_raw += signal_cache_entry[noise_key]  # <-- ALL sensors use additive noise
```

**✅ AFTER (CORRECT)**:
```python
if signal_cache_entry is not None:
    # Use rotation for ROTATION_BASED_MAG_SENSORS, additive noise otherwise
    if sensor_name in pk_config.ROTATION_BASED_MAG_SENSORS:
        gyro_tremor = signal_cache_entry["gyro_tremor"]
        win_raw = apply_tremor_rotation_to_magnetometer(
            mag_signal=win_raw,
            gyro_tremor=gyro_tremor,
            fs=ORIGINAL_FS,
        )
    elif sensor_type in ["acc", "gyro", "mag"]:
        noise_key = f"{sensor_type}_noise"
        win_raw += signal_cache_entry[noise_key]
```

**Key changes**:
- Import `apply_tremor_rotation_to_magnetometer` (line ~32)
- Check `pk_config.ROTATION_BASED_MAG_SENSORS` for rotation-based sensors
- Use rotation-based augmentation instead of additive noise for specific magnetometers

**Code location**: Lines ~495-508

---

### 3. ✅ Tremor-Branch Sensor Generation

**❌ BEFORE**: Not generated at all

**✅ AFTER**: Fully implemented (Lines ~577-651)

Key differences from HAR branch:
- **No z-score normalization**: Uses `win_tremor = win_resampled` (NOT `zscore_window()`)
- **No augment_window**: No augmentation at end
- **Filtering by allowed_scores**: Checks allowed_scores before including window
- **Files saved**:
  - `Acc_arm_tremorbranch.npz` + `.txt`
  - `Gyro_arm_tremorbranch.npz` + `.txt`

**Processing pipeline (tremor-branch)**:
```
raw → tremor node injection → resample → (NO zscore) → save
```

**Processing pipeline (HAR)**:
```
raw → tremor noise injection → resample → zscore → augment_window → save
```

---

### 4. ✅ Augmentation Pipeline Order Verification

**VERIFIED IDENTICAL**:

```
HAR Branch:
1. Extract raw window
2. Apply tremor/noise (rotation for MAG, additive for others)
3. Apply AWGN or rotation augmentation based on tremor score
4. Resample to target FS
5. Z-score normalization
6. Add NOISE_LEVEL augmentation (for each AUG_SIZE copy)

Tremor-Branch:
1. Extract raw window
2. Apply tremor/noise (additive for Acc/Gyro only)
3. Resample to target FS
4. (NO z-score, NO augmentation)
```

---

### 5. ✅ Clean Variant Truly Clean

**VERIFIED**:
- When `augment_mode == "clean"`, tremor cache generates `tremor_score=0`
- No tremor injection occurs (allowed_scores={0})
- Windows with non-zero scores are filtered out
- Result: **Pure signal, no tremor synthesis**

---

### 6. ✅ Feature Formatting Identical

**Column structure (NPZ files)**:
- `X`: (N_samples, n_channels, window_len) [shape varies per sensor]
- `y`: Activity label (0-indexed)
- `subject_id`: Always 1 for Subject 1
- `base_window_idx`: 0-based window index
- `tremor_freq`: Frequency in Hz
- `tremor_acc_rms`: RMS acceleration target
- `tremor_gyro_rms`: RMS gyroscope target
- `tremor_score`: 0-4 tremor severity label

**Column structure (TXT files)** - IDENTICAL to original:
```
[Channel values...] [activity_label] [subject_id] [base_window_idx] 
[tremor_freq] [tremor_acc_rms] [tremor_gyro_rms] [tremor_score]
```

**Activity labels**: `y + 1` (converting from 0-indexed to 1-indexed for TXT)

---

## Summary of Changes

| Item | Before | After |
|------|--------|-------|
| Tremor cache generation | Single cache, manual remapping | Fresh cache per variant |
| Magnetometer handling | Additive noise only | Rotation-based augmentation via `apply_tremor_rotation_to_magnetometer` |
| Tremor-branch outputs | Missing | Implemented (no z-score, no augmentation) |
| HAR pipeline | Unclear | Verified identical |
| Clean variant | Unclear if truly clean | Verified no tremor synthesis |
| Tremor score distribution | Incorrect (all scores 3-4 then mapped) | Correct (natural generation per variant) |

---

## Verification Results ✅

**All three variants generated successfully**:
- ✅ `s2_w2_fs50_tremor_clean/` - 682 samples per sensor
- ✅ `s2_w2_fs50_tremor_mild_mod/` - 682 samples per sensor  
- ✅ `s2_w2_fs50_tremor_mod_severe/` - 682 samples per sensor

**All files present**:
- ✅ 8 HAR sensors × 2 files (`.npz` + `.txt`) = 16 files per variant
- ✅ 2 tremor-branch sensors × 2 files = 4 files per variant
- ✅ `info.txt` + `tremor_parkinson_params.txt` = 2 metadata files
- ✅ **Total: 22 files per variant × 3 variants = 66 output files**

**Output directories**: `/Data/Tremor_datagenerator_files_subject1/`

---

## Code Quality Assurance

✅ **No simplifications**: All logic preserved from original  
✅ **Exact alignment**: Line-by-line comparison with `process_parkinson_tremor_dataset.py`  
✅ **Identical parameters**: All tremor model parameters unchanged  
✅ **Identical seeds**: SEED=0, TREMOR_SEED=42 (same as original)  
✅ **Identical processing order**: Raw → tremor → resample → normalize → augment  

---

## Next Steps

Subject 1 data is now **ready for evaluation**:
1. Use `Tremor_datagenerator_files_subject1` as input to classifiers
2. Compare with training data preprocessing (should be bitwise-equivalent)
3. Validate that preprocessing is not source of performance differences
4. Use as controlled holdout validation set
