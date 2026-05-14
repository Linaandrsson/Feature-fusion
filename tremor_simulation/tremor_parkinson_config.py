"""
Parkinson Tremor Configuration
===============================

This module defines realistic tremor parameters for simulating Parkinson's disease
in different subjects, based on:
- Subject-specific baseline tremor severity (A_subject) - accelerometer RMS
- Subject-specific tremor frequency (FREQ_TREMOR) - Hz in range [3.5, 7.0]
- Severity-dependent k_g factor for gyroscope RMS calculation
- Body part location sensitivity (C)
- Activity-dependent modulation (beta)

RMS Formulas:
    RMS_acc = A_subject[subject_id] * C[body_part] * beta[activity] * (1 + jitter)
    RMS_gyro = k_g(RMS_acc) * RMS_acc
    
    where k_g is determined by accelerometer RMS severity:
      - 0.05-0.10 m/s²: k_g = 10.5 (mild)
      - 0.25-0.5 m/s²:  k_g = 21.3 (mild-moderate)
      - 1.10-2.0 m/s²:  k_g = 13.7 (moderate-severe)
      - 4.65-5.10 m/s²: k_g = 12.9 (severe)

Frequency:
    Each subject has a characteristic tremor frequency in the range [3.5, 7.0] Hz
    Frequency is constant per subject and used for dataset labeling

Dataset Labeling:
    Use get_tremor_params() to get all three values for labeling windows:
    - acc_rms: Accelerometer RMS (m/s²)
    - gyro_rms: Gyroscope RMS (deg/s)
    - freq: Tremor frequency (Hz)

References:
- Resting tremor typically 4-6 Hz (van der Pol oscillator naturally produces this)
- Upper limbs typically more affected than lower limbs
- Tremor often reduced during active movement, increased at rest
- Gyroscope-to-accelerometer ratios based on empirical Parkinson's tremor data
"""

import numpy as np

# ============================================================
# Tremor Policy Configuration
# ============================================================
# Defines which sensors should NOT receive tremor in the signal.
# Tremor-free sensors represent cleaner measurement sites or sensors
# that are physiologically less affected by Parkinson's tremor.
#
# Note: These sensors can still receive alternative augmentations
# (e.g., AWGN, rotation) at the dataset generation level to avoid duplicates.

TREMOR_FREE_SENSORS = ['Acc_chest', 'ECG']

# Defines which magnetometer sensors use rotation-based tremor modeling
# (derived from gyroscope tremor) instead of independent additive noise.
#
# Physical rationale: Magnetometer tremor is a consequence of tremor-induced
# orientation changes, not independent additive noise. The gyroscope tremor
# component drives cumulative rotational perturbations applied to the 
# magnetometer vector.
#
# Both Mag_arm and Mag_ankle follow this physical model:
# - Mag_arm rotation driven by Gyro_arm tremor
# - Mag_ankle rotation driven by Gyro_ankle tremor

ROTATION_BASED_MAG_SENSORS = ['Mag_arm', 'Mag_ankle']

# ============================================================
# Ankle Tremor Scaling
# ============================================================
# Ankle tremor is scaled relative to arm tremor using severity-dependent ratios.
# Lower limbs typically exhibit less tremor than upper limbs in Parkinson's disease.
#
# Ratios define ankle tremor as a fraction of arm tremor for each severity score:
#   - Score 0 (clean): 0% (no tremor)
#   - Score 1 (mild): 5% of arm tremor
#   - Score 2 (mild-moderate): 12% of arm tremor
#   - Score 3 (moderate-severe): 22% of arm tremor
#   - Score 4 (severe): 35% of arm tremor
#
# Applied to both accelerometer and gyroscope ankle tremor targets.

ANKLE_RATIO_BY_SCORE = {
    0: 0.00,
    1: 0.05,
    2: 0.12,
    3: 0.22,
    4: 0.35,
}

# ============================================================
# A_SUBJECT: Baseline Tremor Severity per Subject
# ============================================================
# These values represent the baseline RMS tremor amplitude for accelerometer
# in m/s² (raw sensor units before normalization).
# Higher values = more severe Parkinson's tremor
#
# Gyroscope RMS is calculated dynamically using: RMS_gyro = k_g * RMS_acc
# where k_g depends on the severity (see choose_kg_from_rms_acc function)
#
# Severity guidelines (Accelerometer RMS in m/s^2):
#   0.05-0.10 (mild), 0.25-0.5 (mild-moderate), 1.10-2.0 (moderate-severe), 4.65-5.10 (severe)
#
# Note: Gyroscope RMS is calculated from accelerometer RMS using k_g factor
#       RMS_gyro = k_g * RMS_acc (see choose_kg_from_rms_acc function)

A_SUBJECT = {
    # Subject 1-3: Mild tremor (Score 1: 0.05-0.10 m/s²)
    1: 0.07,
    2: 0.09,
    3: 0.06,
    
    # Subject 4-6: Mild–Moderate (Score 2: 0.25-0.5 m/s²)
    4: 0.35,
    5: 0.45,
    6: 0.30,
    
    # Subject 7-8: Moderate–Severe (Score 3: 1.10-2.0 m/s²)
    7: 1.3,
    8: 1.7,
    
    # Subject 9-10: Severe (Score 4: 4.65-5.10 m/s²)
    9: 4.75,
    10: 4.95,
}


# ============================================================
# FREQ_TREMOR: Tremor Frequency per Subject (Hz)
# ============================================================
# Parkinson's resting tremor typically ranges from 3.5-7 Hz
# Each subject has a characteristic tremor frequency
# Values are constrained to the range [3.5, 7.0] Hz
#
# Note: These frequencies represent the dominant tremor frequency
#       and will be used to label each window along with RMS values

FREQ_TREMOR = {
    # Subject 1-3: Mild tremor (slightly higher frequency typical)
    1: 5.5,
    2: 5.8,
    3: 5.2,
    
    # Subject 4-6: Mild–Moderate (mid-range frequency)
    4: 5.0,
    5: 4.8,
    6: 5.3,
    
    # Subject 7-8: Moderate–Severe (classic 4-6 Hz range)
    7: 4.5,
    8: 4.2,
    
    # Subject 9-10: Severe (lower frequency)
    9: 3.8,
    10: 4.0,
}

# ============================================================
# Tremor severity augmentation (interval sampling)
# ============================================================

import numpy as np
import random

# Score -> RMS_acc interval (m/s^2)
# Values derived from PD data analysis (BioStamp dataset)
SCORE_RMS_RANGE_PD_set = {
    1: (0.16, 0.28),
    2: (0.28, 0.48),
    3: (0.48, 0.72),
    4: (0.72, 1.00),
}

SCORE_RMS_RANGE = {
    1: (0.05, 0.10),
    2: (0.25, 0.50),
    3: (1.10, 2.00),
    4: (4.65, 5.10),
}

# Default augmentation mode (can be overridden per-window)
DEFAULT_AUGMENT_MODE = "mod_severe"   # Options: "clean", "mild_mod", "mod_severe"

# Optional: per-window variability (keeps realism)
SUBJECT_VARIATION_STD = 0.10   # 10% multiplicative variation
FREQ_RANGE_HZ = (3.0, 7.0)     # Tremor frequency range (Hz)

# ============================================================
# Tremor Sampling Method Selection
# ============================================================
# Choose which method to use for tremor parameter generation:
#   "subject": Subject-based (old method)
#              - Uses A_SUBJECT and FREQ_TREMOR dictionaries
#              - Each subject has fixed baseline tremor severity and frequency
#              - Modulated by body_part, activity, and jitter
#   
#   "interval": Interval-based (new method)
#              - Samples tremor parameters independently per window
#              - Uses SCORE_RMS_RANGE intervals and FREQ_RANGE_HZ
#              - More variability, suited for augmentation studies
DEFAULT_SAMPLING_METHOD = "interval"  # Options: "subject", "interval"


def sample_tremor_params(augment_mode: str = None, rng: np.random.Generator = None) -> tuple[int, float, float]:
    """
    Sample tremor parameters for one window.
    
    This function should be called for each window to get unique tremor parameters,
    ensuring realistic variability across the dataset.
    
    Args:
        augment_mode: Which severity level to sample from:
            - "clean": No tremor (score=0, rms=0, freq=0)
            - "mild_mod": Sample from score 1-2 (mild to mild-moderate)
            - "mod_severe": Sample from score 3-4 (moderate-severe to severe)
            - None: Use DEFAULT_AUGMENT_MODE
        rng: Random number generator for reproducibility (optional)
        
    Returns:
        Tuple of (score, rms_acc, freq_hz):
            - score: Tremor severity score (0-4)
            - rms_acc: Accelerometer RMS in m/s² 
            - freq_hz: Tremor frequency in Hz (3.5-7.0)
            
    Example:
        >>> rng = np.random.default_rng(42)
        >>> for i in range(5):
        ...     score, rms, freq = sample_tremor_params("mild_mod", rng)
        ...     print(f"Window {i}: Score={score}, RMS={rms:.3f}, Freq={freq:.1f} Hz")
    """
    if augment_mode is None:
        augment_mode = DEFAULT_AUGMENT_MODE
    
    if rng is None:
        rng = np.random.default_rng()
    
    if augment_mode == "clean":
        return 0, 0.0, 0.0
    
    elif augment_mode == "mild_mod":
        score = rng.choice([1, 2])
        rms_min, rms_max = SCORE_RMS_RANGE[score]
        rms = float(rng.uniform(rms_min, rms_max))
        rms *= float(rng.normal(1.0, SUBJECT_VARIATION_STD))
        # Clamp to valid range for this score to prevent falling into gaps
        rms = float(np.clip(rms, rms_min, rms_max))
        freq = float(rng.uniform(*FREQ_RANGE_HZ))
        return score, rms, freq
    
    elif augment_mode == "mod_severe":
        score = rng.choice([3, 4])
        rms_min, rms_max = SCORE_RMS_RANGE[score]
        rms = float(rng.uniform(rms_min, rms_max))
        rms *= float(rng.normal(1.0, SUBJECT_VARIATION_STD))
        # Clamp to valid range for this score to prevent falling into gaps
        rms = float(np.clip(rms, rms_min, rms_max))
        freq = float(rng.uniform(*FREQ_RANGE_HZ))
        return score, rms, freq
    
    else:
        raise ValueError(f"Unknown augment_mode: {augment_mode}. "
                        f"Must be 'clean', 'mild_mod', or 'mod_severe'")


def apply_sampled_tremor_to_window(
    X_acc: np.ndarray,
    X_gyro: np.ndarray,
    X_mag: np.ndarray,
    augment_mode: str,
    fs: float,
    mu: float = 1.0,
    sigma: float = 0.5,
    dt: float = 0.001,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Apply tremor to a sensor window by sampling parameters per window.
    
    This is the main function to use for the new per-window augmentation approach.
    It samples tremor parameters (score, RMS, frequency) for each window independently.
    
    Args:
        X_acc, X_gyro, X_mag: Sensor data arrays (N, 3)
        augment_mode: "clean", "mild_mod", or "mod_severe"
        fs: Sampling frequency (Hz)
        mu, sigma, dt: Van der Pol oscillator parameters
        seed: Random seed for reproducibility
        rng: Optional random generator (for parameter sampling)
        
    Returns:
        Tuple of (X_acc_tremor, X_gyro_tremor, X_mag_tremor, metadata)
        metadata contains: score, acc_rms, gyro_rms, freq_hz, severity
        
    Example:
        >>> X_acc = np.random.randn(256, 3)
        >>> X_gyro = np.random.randn(256, 3) 
        >>> X_mag = np.random.randn(256, 3)
        >>> X_acc_t, X_gyro_t, X_mag_t, meta = apply_sampled_tremor_to_window(
        ...     X_acc, X_gyro, X_mag, augment_mode="mild_mod", fs=50.0, seed=42
        ... )
        >>> print(f"Score: {meta['score']}, Acc RMS: {meta['acc_rms']:.3f}, "
        ...       f"Freq: {meta['freq_hz']:.1f} Hz")
    """
    # Import here to avoid circular dependency
    from tremor_simulation import Tremor
    
    # Sample tremor parameters for this window
    score, acc_rms, freq_hz = sample_tremor_params(augment_mode, rng)
    
    # Calculate gyroscope RMS from accelerometer RMS
    k_g = choose_kg_from_rms_acc(acc_rms)
    gyro_rms = k_g * acc_rms
    
    # Apply tremor with sampled parameters
    X_acc_t, X_gyro_t, X_mag_t, tremor_meta = Tremor.simulate_and_add_tremor_imu(
        X_acc=X_acc,
        X_gyro=X_gyro,
        X_mag=X_mag,
        fs=fs,
        freq_hz=freq_hz,
        acc_rms=acc_rms,
        gyro_rms=gyro_rms,
        mag_rms=0.0,  # No magnetometer tremor
        mu=mu,
        sigma=sigma,
        dt=dt,
        intermittent=False,
        seed=seed,
    )
    
    # Get severity classification
    if score == 0:
        severity = "none"
    elif score == 1:
        severity = "mild"
    elif score == 2:
        severity = "mild-moderate"
    elif score == 3:
        severity = "moderate-severe"
    else:
        severity = "severe"
    
    # Combine sampled params and tremor metadata
    metadata = {
        'score': int(score),
        'acc_rms': float(acc_rms),
        'gyro_rms': float(gyro_rms),
        'freq_hz': float(freq_hz),
        'severity': severity,
        'augment_mode': augment_mode,
        'k_g': float(k_g),
        'tremor_signal_acc': tremor_meta['s_acc'],
        'tremor_signal_gyro': tremor_meta['s_gyro'],
        'tremor_direction': tremor_meta['v'],
        'mu': mu,
        'sigma': sigma,
    }
    
    return X_acc_t, X_gyro_t, X_mag_t, metadata


# ============================================================
# C: Body Part Sensitivity Factors
# ============================================================
# Tremor typically most visible in upper limbs (hands/arms)
# Less prominent in trunk (chest) and lower limbs (ankle)
#
# Scale: 1.0 = baseline, >1.0 = more tremor, <1.0 = less tremor

C_BODY_PART = {
    "arm": 1,      # Default baseline (most affected)
    "chest": 0.6,    # Trunk less affected NOT IN USE
    "ankle": 0.8,    # Lower limbs moderately affected NOT IN USE
}

# ============================================================
# BETA: Activity-Dependent Modulation
# ============================================================
# Parkinson's resting tremor characteristics:
# - Prominent at rest
# - Often reduced during voluntary movement
# - May worsen during cognitive tasks (e.g., walking + talking)
#
# Activity labels from MHEALTH dataset:
#   0: Nothing (if exists)
#   1: Standing still
#   2: Sitting and relaxing
#   3: Lying down
#   4: Walking
#   5: Climbing stairs
#   6: Waist bends forward
#   7: Frontal elevation of arms
#   8: Knees bending (crouching)
#   9: Cycling
#   10: Jogging
#   11: Running
#   12: Jump front & back

BETA_ACTIVITY = {
    # Rest baseline (A is calibrated on rest tremor)
    1: 1.00,  # Standing still
    2: 1.00,  # Sitting and relaxing
    3: 1.00,  # Lying down (optional slight reduction)

    # Low–moderate intensity movement
    4: 0.60,  # Walking
    6: 0.60,  # Waist bends
    7: 0.10,  # Arm elevation (action condition)
    8: 0.60,  # Knees bending
    9: 0.10,  # Cycling

    # Higher motor activation
    5: 0.30,  # Climbing stairs
    10: 0.30, # Jogging
    11: 0.20, # Running
    12: 0.20, # Jumping

    0: 1.0,
}

# ============================================================
# Score-Dependent Activity Beta
# ============================================================
# Activity-dependent tremor scaling per severity score.
# Captures how tremor amplitude is modulated by the current activity
# differently depending on the severity level.
# Keys are MHEALTH activity labels (1-12, with 0 for unlabeled).

BETA_ACTIVITY_BY_SCORE = {
    1: {
        1: 0.80,  # Standing still -> calibration
        2: 1.10,  # Sitting/rest proxy
        3: 1.05,  # Lying/rest proxy
        4: 0.80,  # Walking -> similar to standing/calibration
        5: 0.90,  # Climbing stairs -> between calibration and active
        6: 1.00,  # Waist bends -> exercise
        7: 0.80,  # Arm elevation -> cardigan-like
        8: 1.00,  # Knees bending -> exercise
        9: 0.70,  # Cycling -> supported hands/feet, damped tremor
        10: 1.00, # Jogging -> exercise/key baseline
        11: 1.00, # Running -> exercise/key baseline
        12: 1.00, # Jump front & back -> exercise/key baseline
        0: 1.00,
    },

    2: {
        1: 0.89,
        2: 1.10,
        3: 1.05,
        4: 0.89,
        5: 0.94,
        6: 1.00,
        7: 0.82,  # cardigan score 2-ish
        8: 1.00,
        9: 0.70,
        10: 1.00,
        11: 1.00,
        12: 1.00,
        0: 1.00,
    },

    3: {
        1: 0.78,
        2: 1.15,
        3: 1.08,
        4: 0.78,
        5: 0.89,
        6: 1.00,
        7: 0.90,  # cardigan score 3
        8: 1.00,
        9: 0.70,
        10: 1.00,
        11: 1.00,
        12: 1.00,
        0: 1.00,
    },

    4: {
        1: 0.72,
        2: 1.20,
        3: 1.10,
        4: 0.72,
        5: 0.86,
        6: 1.00,
        7: 0.72,  # cardigan score 4
        8: 1.00,
        9: 0.65,
        10: 1.00,
        11: 1.00,
        12: 1.00,
        0: 1.00,
    },
}

def get_activity_beta_for_score(score: int, activity_label: int | None) -> float:
    """Return activity-dependent tremor scaling for a severity score and MHEALTH activity label."""
    if activity_label is None:
        return 1.0
    try:
        score_i = int(score)
        activity_i = int(activity_label)
    except Exception:
        return 1.0

    by_activity = BETA_ACTIVITY_BY_SCORE.get(score_i, {})
    return float(by_activity.get(activity_i, BETA_ACTIVITY.get(activity_i, 1.0)))


# ============================================================
# JITTER: Random Variability Parameters
# ============================================================
# Tremor naturally varies from window to window
# Add multiplicative Gaussian noise: (1 + jitter_std * randn)
#
# Recommended: 0.1-0.2 (10-20% variability)

JITTER_STD = 0.15  # 15% standard deviation

# ============================================================
# Gyroscope RMS Calculation from Accelerometer RMS
# ============================================================

def choose_kg_from_rms_acc(rms_acc: float) -> float:
    """
    Severity-dependent mapping from accelerometer tremor RMS (m/s^2) to k_g (deg/s per m/s^2).

    Intervals match Table kg_raw:
      Score 1: 0.05–0.10  -> k_g = 11
      Score 2: 0.25–0.5   -> k_g = 22
      Score 3: 1.10–2.0   -> k_g = 13
      Score 4: 4.65–5.10  -> k_g = 13

    Notes:
    - Uses left-closed, right-open bins to avoid ambiguity at boundaries.
    - Clamps extreme values to the nearest bin.
    - Returns k_g factor to calculate: RMS_gyro = k_g * RMS_acc
    """
    if rms_acc is None:
        raise ValueError("rms_acc cannot be None")
    if rms_acc < 0:
        raise ValueError(f"rms_acc must be non-negative, got {rms_acc}")

    # Clamp to modeled range (optional but keeps behavior defined)
    if rms_acc < 0.05:
        return 11
    if rms_acc < 0.10: # score 1: 0.05-0.10
        return 11
    if rms_acc < 0.25: # gap between score 1 and 2
        return 11
    if rms_acc < 0.5: # score 2: 0.25-0.5
        return 22
    if rms_acc < 1.10: # gap between score 2 and 3
        return 22
    if rms_acc < 2.0: # score 3: 1.10-2.0
        return 14
    if rms_acc < 4.65: # gap between score 3 and 4
        return 14
    if rms_acc <= 5.10: # score 4: 4.65-5.10
        return 13

    # Above modeled max -> keep most severe bin
    return 13


def get_tremor_score(rms_acc: float) -> int:
    """
    Get tremor severity score (1-4) based on accelerometer RMS.
    
    Score 1: 0.05–0.10 m/s² (mild)
    Score 2: 0.25–0.5 m/s² (mild-moderate)
    Score 3: 1.10–2.0 m/s² (moderate-severe)
    Score 4: 4.65–5.10 m/s² (severe)
    
    For values in gaps between defined ranges, assigns to nearest score
    based on k_g mapping used in choose_kg_from_rms_acc().
    
    Args:
        rms_acc: Accelerometer RMS in m/s²
        
    Returns:
        Score from 1-4 (or 0 for no tremor)
    """
    if rms_acc < 0.05:
        return 0  # No tremor or too weak
    elif rms_acc < 0.10:
        return 1  # Mild: [0.05, 0.10)
    elif rms_acc < 0.25:
        return 1  # Gap [0.10, 0.25) → use score 1 (k_g=11)
    elif rms_acc < 0.5:
        return 2  # Mild-moderate: [0.25, 0.5)
    elif rms_acc < 1.10:
        return 2  # Gap [0.5, 1.10) → use score 2 (k_g=22)
    elif rms_acc < 2.0:
        return 3  # Moderate-severe: [1.10, 2.0)
    elif rms_acc < 4.65:
        return 3  # Gap [2.0, 4.65) → use score 3 (k_g=14)
    elif rms_acc <= 5.10:
        return 4  # Severe: [4.65, 5.10]
    else:
        return 4  # Above range: [5.10, ∞) → use score 4 (k_g=13)

# ============================================================
# Helper Functions
# ============================================================

def get_tremor_rms(
    subject_id: int,
    sensor_type: str,  # "acc", "gyro", or "mag"
    body_part: str,    # "arm", "chest", or "ankle"
    activity: int,     # activity label
    jitter_std: float = JITTER_STD,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Calculate tremor RMS for a specific window using Parkinson's parameters.
    
    For accelerometer: Uses A_SUBJECT baseline directly
    For gyroscope: Calculates from accelerometer RMS using k_g factor
                   RMS_gyro = k_g * RMS_acc
    For magnetometer: Returns 0.0 (no tremor - not used in Parkinson studies)
    
    Args:
        subject_id: Subject identifier (1-10)
        sensor_type: "acc", "gyro", or "mag"
        body_part: "arm", "chest", or "ankle"
        activity: Activity label (0-12)
        jitter_std: Standard deviation for random jitter (default 0.15)
        rng: Random number generator (optional, for reproducibility)
        
    Returns:
        Target RMS value for tremor injection
        
    Example:
        >>> rms_acc = get_tremor_rms(5, "acc", "arm", 2)  # Subject 5, acc, arm, sitting
        >>> rms_gyro = get_tremor_rms(5, "gyro", "arm", 2)  # Gyro calculated from acc
        >>> print(f"Acc RMS: {rms_acc:.3f}, Gyro RMS: {rms_gyro:.3f}")
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Get base tremor for this subject (accelerometer RMS)
    if subject_id not in A_SUBJECT:
        raise ValueError(f"Subject {subject_id} not found in A_SUBJECT. Available: {list(A_SUBJECT.keys())}")
    
    A_acc = A_SUBJECT[subject_id]  # Now just a float value for accelerometer
    
    # Get body part scaling
    if body_part not in C_BODY_PART:
        raise ValueError(f"Body part '{body_part}' not found. Available: {list(C_BODY_PART.keys())}")
    
    C = C_BODY_PART[body_part]
    
    # Get activity modulation
    if activity not in BETA_ACTIVITY:
        print(f"Warning: Activity {activity} not found in BETA_ACTIVITY. Using default beta=1.0")
        beta = 1.0
    else:
        beta = BETA_ACTIVITY[activity]
    
    # Add jitter (multiplicative Gaussian noise)
    jitter = 1.0 + jitter_std * rng.standard_normal()
    jitter = max(0.5, min(1.5, jitter))  # Clip to reasonable range [0.5, 1.5]
    
    # Calculate accelerometer RMS first
    rms_acc = A_acc * C * beta * jitter
    
    # Calculate sensor-specific RMS
    if sensor_type == "acc":
        rms = rms_acc
    elif sensor_type == "gyro":
        # Calculate gyroscope RMS from accelerometer RMS using k_g factor
        k_g = choose_kg_from_rms_acc(rms_acc)
        rms = k_g * rms_acc
    elif sensor_type == "mag":
        # Magnetometer: no tremor (not used in Parkinson studies)
        rms = 0.0
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}. Must be 'acc', 'gyro', or 'mag'")
    
    return float(rms)


def get_tremor_frequency(subject_id: int) -> float:
    """
    Get the characteristic tremor frequency for a subject.
    
    Args:
        subject_id: Subject identifier (1-10)
        
    Returns:
        Tremor frequency in Hz (range: 3.5-7.0 Hz)
        
    Raises:
        ValueError: If subject_id not found
    """
    if subject_id not in FREQ_TREMOR:
        raise ValueError(f"Subject {subject_id} not found in FREQ_TREMOR. Available: {list(FREQ_TREMOR.keys())}")
    
    freq = FREQ_TREMOR[subject_id]
    
    # Validate frequency is in valid range
    if not (3.5 <= freq <= 7.0):
        raise ValueError(f"Frequency {freq} Hz for subject {subject_id} is outside valid range [3.5, 7.0] Hz")
    
    return float(freq)


def get_tremor_params(
    subject_id: int,
    body_part: str,    # "arm", "chest", or "ankle"
    activity: int,     # activity label
    jitter_std: float = JITTER_STD,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Get complete tremor parameters for a specific window.
    Returns all values needed to label the window: acc_rms, gyro_rms, and frequency.
    
    Args:
        subject_id: Subject identifier (1-10)
        body_part: "arm", "chest", or "ankle"
        activity: Activity label (0-12)
        jitter_std: Standard deviation for random jitter (default 0.15)
        rng: Random number generator (optional, for reproducibility)
        
    Returns:
        Dictionary with keys:
            - 'acc_rms': Accelerometer RMS (m/s²)
            - 'gyro_rms': Gyroscope RMS (deg/s)
            - 'freq': Tremor frequency (Hz)
            - 'subject_id': Subject ID
            - 'body_part': Body part
            - 'activity': Activity label
            - 'severity': Tremor severity classification
            
    Example:
        >>> params = get_tremor_params(5, "arm", 2)
        >>> print(f"Acc: {params['acc_rms']:.3f}, Gyro: {params['gyro_rms']:.3f}, Freq: {params['freq']:.1f} Hz")
    """
    # Get RMS values for accelerometer and gyroscope
    acc_rms = get_tremor_rms(subject_id, "acc", body_part, activity, jitter_std, rng)
    gyro_rms = get_tremor_rms(subject_id, "gyro", body_part, activity, jitter_std, rng)
    
    # Get tremor frequency
    freq = get_tremor_frequency(subject_id)
    
    # Get severity classification
    severity = get_subject_severity(subject_id)
    
    return {
        'acc_rms': float(acc_rms),
        'gyro_rms': float(gyro_rms),
        'freq': float(freq),
        'subject_id': int(subject_id),
        'body_part': str(body_part),
        'activity': int(activity),
        'severity': str(severity),
    }


def get_subject_severity(subject_id: int) -> str:
    """Get tremor severity classification for a subject."""
    if subject_id not in A_SUBJECT:
        return "unknown"
    
    acc_val = A_SUBJECT[subject_id]  # Now a float directly
    
    if acc_val < 0.10:
        return "mild"
    elif acc_val < 0.5:
        return "mild-moderate"
    elif acc_val < 2.0:
        return "moderate-severe"
    else:
        return "severe"


def apply_tremor_to_window(
    X_acc: np.ndarray,
    X_gyro: np.ndarray,
    X_mag: np.ndarray,
    subject_id: int,
    body_part: str,
    activity: int,
    fs: float,
    jitter_std: float = JITTER_STD,
    mu: float = 1.0,
    sigma: float = 0.5,
    dt: float = 0.001,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Apply Parkinson tremor to a sensor window using subject-specific parameters.
    
    This is a convenience function that:
    1. Gets tremor parameters (RMS and frequency) from config
    2. Applies tremor using Tremor.simulate_and_add_tremor_imu()
    3. Returns corrupted signals + metadata with labels
    
    Args:
        X_acc, X_gyro, X_mag: Sensor data arrays (N, 3)
        subject_id: Subject identifier (1-10)
        body_part: "arm", "chest", or "ankle"
        activity: Activity label (0-12)
        fs: Sampling frequency (Hz)
        jitter_std: RMS jitter (default 0.15)
        mu, sigma, dt: Van der Pol oscillator parameters
        seed: Random seed for reproducibility
        rng: Optional random generator (for jitter calculation)
        
    Returns:
        Tuple of (X_acc_tremor, X_gyro_tremor, X_mag_tremor, metadata)
        metadata contains tremor labels: acc_rms, gyro_rms, freq_hz
        
    Example:
        >>> X_acc = np.random.randn(256, 3)
        >>> X_gyro = np.random.randn(256, 3)
        >>> X_mag = np.random.randn(256, 3)
        >>> X_acc_t, X_gyro_t, X_mag_t, meta = apply_tremor_to_window(
        ...     X_acc, X_gyro, X_mag, subject_id=5, body_part="arm", 
        ...     activity=2, fs=50.0, seed=42
        ... )
        >>> print(f"Labels - Acc RMS: {meta['acc_rms']:.3f}, "
        ...       f"Gyro RMS: {meta['gyro_rms']:.3f}, "
        ...       f"Freq: {meta['freq_hz']:.1f} Hz")
    """
    # Import here to avoid circular dependency
    from tremor_simulation import Tremor
    
    # Get tremor parameters for this window
    params = get_tremor_params(subject_id, body_part, activity, jitter_std, rng)
    
    # Apply tremor with specified frequency and RMS values
    X_acc_t, X_gyro_t, X_mag_t, tremor_meta = Tremor.simulate_and_add_tremor_imu(
        X_acc=X_acc,
        X_gyro=X_gyro,
        X_mag=X_mag,
        fs=fs,
        freq_hz=params['freq'],  # Use subject-specific frequency!
        acc_rms=params['acc_rms'],
        gyro_rms=params['gyro_rms'],
        mag_rms=0.0,  # No magnetometer tremor
        mu=mu,
        sigma=sigma,
        dt=dt,
        intermittent=False,
        seed=seed,
    )
    
    # Combine config params and tremor metadata for complete labels
    metadata = {
        **params,  # Include all config parameters
        'tremor_signal_acc': tremor_meta['s_acc'],  # Actual tremor signal (for inspection)
        'tremor_signal_gyro': tremor_meta['s_gyro'],
        'tremor_direction': tremor_meta['v'],
        'mu': mu,
        'sigma': sigma,
    }
    
    return X_acc_t, X_gyro_t, X_mag_t, metadata


def print_tremor_summary(subject_id: int):
    """Print a summary of tremor parameters for a subject."""
    severity = get_subject_severity(subject_id)
    acc_rms = A_SUBJECT.get(subject_id, None)
    
    if acc_rms is None:
        print(f"Subject {subject_id} not found.")
        return
    
    # Calculate k_g factor for this subject's baseline
    k_g = choose_kg_from_rms_acc(acc_rms)
    gyro_rms = k_g * acc_rms
    
    # Get tremor frequency
    freq = get_tremor_frequency(subject_id)
    
    print(f"\nSubject {subject_id} - Tremor Severity: {severity.upper()}")
    print(f"  Base RMS (at rest, arm): Acc={acc_rms:.2f} m/s²")
    print(f"  Derived RMS: Gyro={gyro_rms:.2f} deg/s (k_g={k_g:.1f}), Mag=0.00 (no tremor)")
    print(f"  Tremor Frequency: {freq:.1f} Hz")
    print(f"\nBody Part Scaling:")
    for bp, scale in C_BODY_PART.items():
        print(f"  {bp:6s}: {scale:.2f}x")
    print(f"\nExample RMS values (Arm location):")
    for act in [2, 4, 10]:  # Sitting, Walking, Jogging
        rms_acc = get_tremor_rms(subject_id, "acc", "arm", act, jitter_std=0.0)
        rms_gyro = get_tremor_rms(subject_id, "gyro", "arm", act, jitter_std=0.0)
        act_name = {2: "Sitting", 4: "Walking", 10: "Jogging"}.get(act, f"Activity {act}")
        print(f"  {act_name:12s}: Acc={rms_acc:.3f}, Gyro={rms_gyro:.3f}, Freq={freq:.1f} Hz")


# ============================================================
# Validation
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PARKINSON TREMOR CONFIGURATION - PER-WINDOW SAMPLING")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Per-Window Tremor Parameter Sampling")
    print("=" * 70)
    
    # Test sampling with reproducible seed
    rng = np.random.default_rng(42)
    
    print("\nMild-Moderate (mild_mod) - 5 windows:")
    for i in range(5):
        score, rms_acc, freq = sample_tremor_params("mild_mod", rng)
        k_g = choose_kg_from_rms_acc(rms_acc)
        gyro_rms = k_g * rms_acc
        severity = get_tremor_score(rms_acc)
        print(f"  Window {i+1}: Score={score}, Acc={rms_acc:.3f} m/s², "
              f"Gyro={gyro_rms:.1f} deg/s, Freq={freq:.1f} Hz")
    
    print("\nModerate-Severe (mod_severe) - 5 windows:")
    for i in range(5):
        score, rms_acc, freq = sample_tremor_params("mod_severe", rng)
        k_g = choose_kg_from_rms_acc(rms_acc)
        gyro_rms = k_g * rms_acc
        print(f"  Window {i+1}: Score={score}, Acc={rms_acc:.3f} m/s², "
              f"Gyro={gyro_rms:.1f} deg/s, Freq={freq:.1f} Hz")
    
    print("\n" + "=" * 70)
    print("Reproducibility Test")
    print("=" * 70)
    
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    
    params1 = sample_tremor_params("mild_mod", rng1)
    params2 = sample_tremor_params("mild_mod", rng2)
    
    print(f"Run 1: Score={params1[0]}, RMS={params1[1]:.6f}, Freq={params1[2]:.3f}")
    print(f"Run 2: Score={params2[0]}, RMS={params2[1]:.6f}, Freq={params2[2]:.3f}")
    print(f"Match: {params1 == params2}")
    
    print("\n" + "=" * 70)
    print("Distribution Statistics (1000 windows)")
    print("=" * 70)
    
    rng = np.random.default_rng(999)
    
    for mode in ["mild_mod", "mod_severe"]:
        samples = [sample_tremor_params(mode, rng) for _ in range(1000)]
        scores = [s[0] for s in samples]
        rms_vals = [s[1] for s in samples]
        freqs = [s[2] for s in samples]
        
        print(f"\n{mode}:")
        print(f"  RMS Acc:  mean={np.mean(rms_vals):.3f}, std={np.std(rms_vals):.3f}, "
              f"min={np.min(rms_vals):.3f}, max={np.max(rms_vals):.3f}")
        print(f"  Freq (Hz): mean={np.mean(freqs):.2f}, std={np.std(freqs):.2f}, "
              f"min={np.min(freqs):.2f}, max={np.max(freqs):.2f}")
        from collections import Counter
        score_counts = Counter(scores)
        print(f"  Score distribution: {dict(sorted(score_counts.items()))}")
