# TREMOR Scenario Noise - User Guide

## Overview

The TREMOR scenario noise has been successfully integrated into `DataGenerator_v3.py`. It simulates tremor-like motion using a stochastic van der Pol oscillator, which is commonly used to model pathological tremor in medical applications.

## Key Features

✅ **Consistent across modalities**: For the same body part and window, Acc/Gyro/Mag sensors share:
   - Same underlying stochastic process (same random seed)
   - Same 3D projection direction vector
   - Same intermittent on/off envelope (if enabled)

✅ **Modality-specific mapping**:
   - Accelerometer uses state variable `x1(t)`
   - Gyroscope uses state variable `x2(t)` (90° phase-shifted)
   - Magnetometer uses state variable `x1(t)` with lower amplitude

✅ **Cache mechanism**: Ensures tremor consistency across sensors for the same (window_idx, body_part)

## Configuration Parameters

In `DataGenerator_v3.py`, set these parameters at the top of the file:

```python
# Basic setting
SCENARIO_NOISE_TYPE = "TREMOR"  # Enable tremor noise

# Van der Pol oscillator parameters
TREMOR_MU = 1.0                 # Nonlinearity parameter (typical: 0.5-2.0)
TREMOR_SIGMA = 0.5              # Stochastic noise intensity (typical: 0.1-1.0)
TREMOR_DT = 0.001               # Integration time step in seconds

# Tremor strength per modality (RMS in sensor units, applied to raw signals)
TREMOR_ACC_RMS = 0.2            # Accelerometer tremor RMS
TREMOR_GYRO_RMS = 0.2           # Gyroscope tremor RMS  
TREMOR_MAG_RMS = 0.05           # Magnetometer tremor RMS (typically lower)

# Intermittent tremor (optional)
TREMOR_INTERMITTENT = False     # Enable on/off episodes
TREMOR_ON_PROB = 0.5            # Probability of tremor being active
TREMOR_MIN_ON_SEC = 2.0         # Minimum activity duration
TREMOR_MAX_ON_SEC = 8.0         # Maximum activity duration
```

## Important Notes

**TREMOR always applies to ALL IMU sensors** (Acc/Gyro/Mag on ankle, arm, and chest). ECG is not affected.

**To create different tremor scenarios**, simply change `NOISE_SCENARIO_SEED`:
```python
NOISE_SCENARIO_SEED = 123  # Tremor scenario 1
NOISE_SCENARIO_SEED = 456  # Tremor scenario 2
NOISE_SCENARIO_SEED = 789  # Tremor scenario 3
```

Each seed will generate completely different tremor realizations while keeping all other parameters the same.

## Usage Examples

### Example 1: Basic Tremor (Default Settings)

```python
SCENARIO_NOISE_TYPE = "TREMOR"
TREMOR_MU = 1.0
TREMOR_SIGMA = 0.5
TREMOR_ACC_RMS = 0.2
TREMOR_GYRO_RMS = 0.2
TREMOR_MAG_RMS = 0.05
TREMOR_INTERMITTENT = False
```

This generates continuous tremor with moderate strength.

### Example 2: Strong Tremor

```python
SCENARIO_NOISE_TYPE = "TREMOR"
TREMOR_MU = 1.5          # Higher nonlinearity
TREMOR_SIGMA = 0.8       # More stochastic variation
TREMOR_ACC_RMS = 0.4     # Stronger amplitude
TREMOR_GYRO_RMS = 0.4
TREMOR_MAG_RMS = 0.1
TREMOR_INTERMITTENT = False
```

### Example 3: Intermittent Tremor (More Realistic)

```python
SCENARIO_NOISE_TYPE = "TREMOR"
TREMOR_MU = 1.0
TREMOR_SIGMA = 0.5
TREMOR_ACC_RMS = 0.3
TREMOR_GYRO_RMS = 0.3
TREMOR_MAG_RMS = 0.08
TREMOR_INTERMITTENT = True    # Enable intermittent episodes
TREMOR_ON_PROB = 0.6          # 60% probability of being active
TREMOR_MIN_ON_SEC = 1.0       # 1-5 second episodes
TREMOR_MAX_ON_SEC = 5.0
```

This simulates tremor that comes and goes, more realistic for pathological conditions.

### Example 4: Subtle Tremor

```python
SCENARIO_NOISE_TYPE = "TREMOR"
TREMOR_MU = 0.8          # Lower nonlinearity
TREMOR_SIGMA = 0.3       # Less variation
TREMOR_ACC_RMS = 0.1     # Weak amplitude
TREMOR_GYRO_RMS = 0.1
TREMOR_MAG_RMS = 0.03
TREMOR_INTERMITTENT = False
```

## Running the Data Generator

Simply run `DataGenerator_v3.py` as usual:

```bash
python DataGenerator_v3.py
```

The output will be saved to a directory like:
```
data/Datagenerator_files/s2_w2_aug2/fs20_TREMOR_mu1_s0p5/
```

The folder name includes the tremor parameters for easy identification.

## Output Structure

The generated dataset will include:

- **Sensor files**: `Acc_arm.npz`, `Gyro_arm.npz`, etc. with tremor-corrupted data
- **corruption_log.jsonl**: Per-window corruption information
- **corruption_summary.json**: Overall statistics
- **info.txt**: Complete configuration including tremor parameters

## Technical Details

### How Tremor is Generated

1. **Stochastic van der Pol oscillator** is simulated at high temporal resolution (dt=0.001s)
2. States `x1(t)` and `x2(t)` are resampled to match the sensor sampling rate
3. Each state is zero-meaned and scaled to the desired RMS value
4. Optional intermittent envelope modulates the amplitude over time
5. Tremor is projected onto a random 3D unit direction vector `v`
6. Final tremor: `noise = s(t) * v` where `s(t)` is the scaled state variable

### Cache Mechanism

The `_TREMOR_CACHE` dictionary stores tremor realizations by `(window_idx, body_part)`:

```python
cache_key = (window_idx, 'arm')  # Example
_TREMOR_CACHE[cache_key] = {
    'acc_noise': (L, 3) array,
    'gyro_noise': (L, 3) array,
    'mag_noise': (L, 3) array,
    'meta': {...}
}
```

When processing `Acc_arm`, `Gyro_arm`, `Mag_arm` for the same window:
- First sensor generates tremor and caches it
- Subsequent sensors reuse the cached tremor
- Ensures perfect consistency across modalities

### Body Part Extraction

Sensors are mapped to body parts:
- `Acc_ankle`, `Gyro_ankle`, `Mag_ankle` → `'ankle'`
- `Acc_arm`, `Gyro_arm`, `Mag_arm` → `'arm'`
- `Acc_chest`, `ECG` → `'chest'`

Each body part gets independent tremor realizations.

## Verification

Run the test script to verify the integration:

```bash
python test_tremor_integration.py
```

This tests:
- ✓ Tremor generation
- ✓ Reproducibility (same seed → same tremor)
- ✓ Uniqueness (different seed → different tremor)
- ✓ Direction vector consistency
- ✓ Intermittent envelope

## Comparison with Other Scenario Noises

| Noise Type | Description | When to Use |
|------------|-------------|-------------|
| **AWGN** | Additive white Gaussian noise | General sensor noise, electronic interference |
| **DROPOUT** | Complete sensor failure (constant value) | Sensor malfunction, disconnection |
| **WEAK_SIGNAL** | Attenuated signal strength | Poor contact, battery issues |
| **TREMOR** | Oscillatory motion pattern | Pathological tremor, vibration, unstable mounting |

## Tips for Parameter Selection

### For Pathological Tremor Simulation:
- Use `TREMOR_MU = 0.8-1.2` for realistic tremor frequency
- Enable `TREMOR_INTERMITTENT = True` for rest tremor
- Higher `TREMOR_SIGMA` for more irregular tremor

### For Mechanical Vibration:
- Use `TREMOR_MU = 1.5-2.0` for more regular oscillation
- Keep `TREMOR_INTERMITTENT = False`
- Lower `TREMOR_SIGMA` for more periodic motion

### For Sensor Mounting Issues:
- Use lower RMS values (`0.1-0.15`)
- Keep `TREMOR_INTERMITTENT = False`
- Focus on accelerometer/gyroscope, less on magnetometer

## Troubleshooting

**Q: Tremor seems too weak/strong?**
- Adjust `TREMOR_ACC_RMS`, `TREMOR_GYRO_RMS`, `TREMOR_MAG_RMS` values
- Note: Tremor is now applied to RAW signals BEFORE resampling and z-score
- Values are scaled relative to the generated oscillator RMS

**Q: Tremor frequency doesn't look right?**
- Try adjusting `TREMOR_MU` (controls oscillation frequency)
- Typical tremor: 4-12 Hz, use `TREMOR_MU = 0.8-1.5`

**Q: Want more/less variability?**
- Adjust `TREMOR_SIGMA` (stochastic noise intensity)
- Higher = more irregular, Lower = more periodic

**Q: Intermittent tremor not working as expected?**
- Check `TREMOR_ON_PROB` (0.5 = 50% active on average)
- Adjust `TREMOR_MIN_ON_SEC` and `TREMOR_MAX_ON_SEC` for episode duration

## Credits

Tremor generation code developed using stochastic van der Pol oscillator model.
Integration into DataGenerator_v3 completed successfully.

---

*Last updated: February 10, 2026*
