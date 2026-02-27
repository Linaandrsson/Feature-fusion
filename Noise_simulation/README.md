# Noise Simulation Modules

This directory contains modular implementations of various noise types and corruption scenarios for sensor data simulation.

## Overview

Each noise type is implemented in its own module for better organization and reusability. These modules can be imported and used in data generation scripts like `DataGenerator_v3.py`.

## Available Modules

### 1. AWGN.py - Additive White Gaussian Noise
Simulates hardware measurement noise added to raw sensor signals.

**Functions:**
- `apply_awgn_rms_ratio(win, rms_ratio, rng)` - Add noise relative to signal RMS (raw signals)
- `apply_awgn(win, sigma, rng)` - Add Gaussian noise with fixed sigma (legacy)
- `apply_awgn_seed_based(win, sigma, seed)` - Convenience function with seed

**Use cases:**
- Hardware measurement noise (sensor noise floor)
- Electronic noise in sensors
- Thermal noise
- Measurement uncertainty
- General random perturbations

**Example (raw signal):**
```python
from Noise_simulation.AWGN import apply_awgn_rms_ratio
import numpy as np

rng = np.random.RandomState(42)
raw_signal = np.random.randn(50, 3) * 100  # Raw sensor readings
noisy_signal = apply_awgn_rms_ratio(raw_signal, rms_ratio=0.2, rng=rng)
```

**Example (legacy, z-scored signal):**
```python
from Noise_simulation.AWGN import apply_awgn
import numpy as np

rng = np.random.RandomState(42)
clean_signal = np.random.randn(100, 3)  # (L, C)
noisy_signal = apply_awgn(clean_signal, sigma=0.3, rng=rng)
```

### 2. Dropout.py - Sensor Dropout/Failure
Simulates complete sensor failure by setting all values to a constant.

**Functions:**
- `apply_dropout(win, value=0.0)` - Set window to constant value

**Use cases:**
- Complete sensor failure
- Signal loss
- Connection problems
- Battery depletion
- Sensor malfunction

**Example:**
```python
from Noise_simulation.Dropout import apply_dropout

signal = np.random.randn(100, 3)
dropout_signal = apply_dropout(signal, value=0.0)
```

### 3. WeakSignal.py - Signal Attenuation
Simulates reduced signal strength by multiplying by a factor < 1.

**Functions:**
- `apply_weak_signal(win, factor)` - Attenuate signal amplitude

**Use cases:**
- Sensor degradation
- Poor contact/connection
- Low battery
- Distance effects
- Environmental interference

**Example:**
```python
from Noise_simulation.WeakSignal import apply_weak_signal

signal = np.random.randn(100, 3)
weak_signal = apply_weak_signal(signal, factor=0.2)  # 20% strength
```

### 4. Tremor.py - Physiological Tremor Simulation
Simulates realistic tremor using stochastic van der Pol oscillator.

#### NEW: Parkinson Patient Model (v2.0)
**Recommended for realistic clinical simulation!**

The new Parkinson model generates subject-specific, anatomically-aware, and activity-modulated tremor:

**Formula:**
```
RMS = A_subject[subject_id][sensor] * C[body_part] * beta[activity] * (1 + jitter)
```

Where:
- **A_subject**: Baseline tremor severity per patient (mild/moderate/severe)
- **C**: Body part sensitivity (arm > ankle > chest)
- **beta**: Activity modulation (high at rest, reduced during movement)
- **jitter**: Optional window-to-window variability (~15%)

**Configuration:** All parameters defined in `tremor_parkinson_config.py`:
- 10 subjects with varying severity (mild, moderate, severe)
- Body part scaling: arm (1.5x), ankle (0.8x), chest (0.6x)
- Activity modulation: sitting (1.5x), walking (0.9x), running (0.5x)

**Functions:**
- `precompute_tremor_cache_with_parkinson_model()` - Pre-compute tremor for all windows (NEW!)
- `write_tremor_parkinson_params_file()` - Document parameters
- `simulate_and_add_tremor_imu()` - Low-level tremor generation (legacy)
- `precompute_tremor_cache_with_relative_rms()` - Old relative RMS method (deprecated)

**Example (Parkinson model - RECOMMENDED):**
```python
from Noise_simulation.Tremor import precompute_tremor_cache_with_parkinson_model

# Pre-compute tremor for all windows
# window_specs: [(subject_id, activity_label, win_start, win_end), ...]
tremor_cache = precompute_tremor_cache_with_parkinson_model(
    window_specs=window_specs,
    sensor_column_mapping=sensor_cols,
    data_loader_func=load_subject_data,
    fs=50.0,
    scenario_seed=42,
    use_jitter=True  # Add natural variability
)

# Apply cached tremor to a specific window and body part
noise_data = tremor_cache[(window_idx, 'arm')]
acc_corrupted = acc_clean + noise_data['acc_noise']
gyro_corrupted = gyro_clean + noise_data['gyro_noise']
mag_corrupted = mag_clean + noise_data['mag_noise']
```

**Clinical realism features:**
- ✅ Subject-specific severity (e.g., Subject 1: mild, Subject 10: severe)
- ✅ Upper limbs more affected than lower limbs
- ✅ Prominent resting tremor (sitting, standing)
- ✅ Reduced tremor during active movement (running, jumping)
- ✅ Natural window-to-window variability
- ✅ Physiologically realistic 4-6 Hz oscillation

**To customize subjects:**
Edit `tremor_parkinson_config.py` and modify:
- `A_SUBJECT`: Add/modify subject tremor severities
- `C_BODY_PART`: Adjust body part sensitivity
- `BETA_ACTIVITY`: Modify activity-dependent modulation

**Use cases:**
- Parkinsonian tremor (4-6 Hz) - **PRIMARY USE CASE**
- Essential tremor (6-12 Hz) - adjust mu parameter
- Movement artifacts in clinical populations
- Testing robustness to patient-specific variability

**Legacy Example (low-level, manual RMS):**
```python
from Noise_simulation.Tremor import simulate_and_add_tremor_imu

# Simulate tremor on IMU data (manual RMS control)
X_acc_t, X_gyro_t, X_mag_t, meta = simulate_and_add_tremor_imu(
    X_acc=clean_acc,
    X_gyro=clean_gyro, 
    X_mag=clean_mag,
    fs=50,
    mu=1.0,
    sigma=0.5,
    acc_rms=0.2,
    gyro_rms=0.2,
    mag_rms=0.05
)
```

## Usage in DataGenerator_v3.py

The data generator imports these modules and applies them based on configuration:

```python
from Noise_simulation.AWGN import apply_awgn
from Noise_simulation.Dropout import apply_dropout
from Noise_simulation.WeakSignal import apply_weak_signal

# Configure noise type
SCENARIO_NOISE_TYPE = "AWGN"  # or "DROPOUT", "WEAK_SIGNAL", None

# Apply during window processing
if SCENARIO_NOISE_TYPE == "AWGN":
    win = apply_awgn(win, AWGN_SIGMA, rng)
elif SCENARIO_NOISE_TYPE == "DROPOUT":
    win = apply_dropout(win, DROPOUT_VALUE)
elif SCENARIO_NOISE_TYPE == "WEAK_SIGNAL":
    win = apply_weak_signal(win, WEAK_SIGNAL_FACTOR)
```

## Design Principles

1. **Modularity**: Each noise type is self-contained
2. **Reusability**: Functions can be used in any data processing pipeline
3. **Reproducibility**: All random operations use explicit seeds/RNGs
4. **Documentation**: Each function has detailed docstrings
5. **Extensibility**: Easy to add new noise types

## Adding New Noise Types

To add a new noise type:

1. Create a new Python file in this directory (e.g., `NewNoise.py`)
2. Implement noise function with signature: `apply_new_noise(win, params, rng)`
3. Add detailed docstring explaining:
   - What the noise simulates
   - Parameters and their effects
   - Use cases
   - Example usage
4. Import in DataGenerator or other scripts
5. Update this README

## Input/Output Format

All noise functions expect:

**Input:**
- `win`: np.ndarray of shape (L, C)
  - L = window length (number of time steps)
  - C = number of channels
  - Typically z-scored (mean~0, std~1 per channel)

**Output:**
- Corrupted window with same shape (L, C)

## Common Parameters

- `sigma`: Noise level for Gaussian noise (std deviation)
- `factor`: Multiplication factor for attenuation (< 1.0)
- `value`: Constant value for dropout (typically 0.0)
- `rng`: Random number generator (np.random.RandomState)
- `seed`: Random seed for reproducibility

## Testing

Each module includes example usage in docstrings. To test:

### Test AWGN
```python
from Noise_simulation.AWGN import apply_awgn
import numpy as np

rng = np.random.RandomState(42)
signal = np.random.randn(100, 3)
noisy = apply_awgn(signal, sigma=0.3, rng=rng)
print(f"Signal shape: {signal.shape}, Noisy shape: {noisy.shape}")
print(f"SNR: {np.var(signal) / np.var(noisy - signal):.2f}")
```

### Test Parkinson Tremor Configuration
```python
# Run configuration validation
python tremor_parkinson_config.py

# This will output:
# - Subject severity summaries (mild/moderate/severe)
# - RMS calculation examples for different activities
# - Reproducibility test with seeds
# - Activity modulation demonstration
```

## References

- **AWGN**: Standard additive noise model
- **Tremor**: Stochastic van der Pol oscillator for pathological tremor simulation
- **Dropout**: Sensor failure models in HAR systems
- **Weak Signal**: Sensor degradation in wearable devices
