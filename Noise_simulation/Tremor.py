import numpy as np

# Import tremor_parkinson_config - handle both package and script mode
try:
    from . import tremor_parkinson_config as pk_config
except ImportError:
    import tremor_parkinson_config as pk_config

# Default parameters for Tremor noise
# Van der Pol oscillator parameters
TREMOR_MU = 1.0                  # van der Pol nonlinearity parameter
TREMOR_SIGMA = 0.5               # stochastic noise intensity
TREMOR_DT = 0.001                # integration time step (seconds)

# Legacy tremor strength parameters (DEPRECATED - use relative RMS with alpha instead)
# These are kept for backwards compatibility with simple tremor injection
# NOTE: In modern usage, tremor is applied to RAW signal BEFORE resampling and z-score normalization
TREMOR_ACC_RMS = 0.3             # accelerometer tremor strength (legacy, in raw sensor units)
TREMOR_GYRO_RMS = 0.5            # gyroscope tremor strength (legacy, in raw sensor units)
TREMOR_MAG_RMS = 0.1            # magnetometer tremor strength (legacy, in raw sensor units)




# Intermittent tremor parameters
TREMOR_INTERMITTENT = False      # enable intermittent on/off envelope. Should not be on when mixing winwos on embedding level.
TREMOR_ON_PROB = 0.5             # IKKE I BRUK. Probability of tremor being active
TREMOR_MIN_ON_SEC = 2.0          # IKKE I BRUK. Minimum tremor activity duration
TREMOR_MAX_ON_SEC = 8.0          # IKKE I BRUK. Maximum tremor activity duration

def simulate_stochastic_vdp_states(
    T_sec: float,
    dt: float = 0.001,
    mu: float = 1.0,
    sigma: float = 0.5,
    omega: float = None,  # angular frequency (rad/s). If None, uses 2*pi*5.0 (5 Hz default)
    x1_0: float = 0.0,
    x2_0: float = 1.0,
    seed: int | None = None,
):
    """
    Simulate stochastic van der Pol oscillator (Euler–Maruyama).
    
    The van der Pol equations with frequency control:
        dx1/dt = omega * x2
        dx2/dt = omega * (mu * (1 - x1^2) * x2 - x1) + sigma * dW/dt
    
    Args:
        T_sec: Duration in seconds
        dt: Integration time step
        mu: Van der Pol nonlinearity parameter (controls limit cycle amplitude)
        sigma: Stochastic noise intensity
        omega: Angular frequency (rad/s). If None, defaults to 2*pi*5.0 (5 Hz).
               To use a specific frequency f (Hz), set omega = 2*pi*f
        x1_0, x2_0: Initial conditions
        seed: Random seed for reproducibility
    
    Returns: 
        t (n_steps,): Time vector
        x1 (n_steps,): Position state
        x2 (n_steps,): Velocity state
    """
    if omega is None:
        omega = 2.0 * np.pi * 5.0  # Default to 5 Hz
    
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T_sec / dt))
    t = np.arange(n_steps, dtype=np.float64) * dt

    x1 = np.zeros(n_steps, dtype=np.float64)
    x2 = np.zeros(n_steps, dtype=np.float64)
    x1[0], x2[0] = x1_0, x2_0

    for k in range(n_steps - 1):
        dx1 = omega * x2[k]
        dx2 = omega * (mu * (1.0 - x1[k] ** 2) * x2[k] - x1[k])
        x1[k + 1] = x1[k] + dt * dx1
        x2[k + 1] = x2[k] + dt * dx2 + sigma * np.sqrt(dt) * rng.standard_normal()

    return t, x1, x2


def resample_to_fs(t, x, fs: float, T_sec: float):
    """
    Resample signal x(t) defined on grid t (dt grid) to sensor grid at fs using interpolation.
    Returns: x_sens with length N = int(round(T_sec*fs))
    """
    N = int(round(T_sec * fs))
    t_sens = np.arange(N, dtype=np.float64) / fs
    return np.interp(t_sens, t, x)


def unit_random_direction(rng: np.random.Generator):
    """
    Sample a random 3D unit vector.
    """
    v = rng.standard_normal(3)
    return v / (np.linalg.norm(v) + 1e-12) # avoid div by zero just in case


def intermittent_envelope(
    N: int,
    fs: float,
    rng: np.random.Generator,
    on_prob: float = 0.5,
    min_on_sec: float = 2.0,
    max_on_sec: float = 8.0,
):
    """
    Piecewise-constant on/off envelope of length N.
    """
    env = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        on = rng.random() < on_prob
        dur = rng.uniform(min_on_sec, max_on_sec)
        L = max(1, int(dur * fs))
        if on:
            env[i:i + L] = 1.0
        i += L
    return env[:N]


def scale_to_rms(s: np.ndarray, target_rms: float):
    """
    Zero-mean and scale s to have desired RMS.
    """
    s0 = s - np.mean(s)
    rms = np.sqrt(np.mean(s0 ** 2)) + 1e-12
    return s0 / rms * target_rms


def add_projected_tremor(X: np.ndarray, s: np.ndarray, v: np.ndarray):
    """
    X: (N,3), s: (N,), v: (3,)
    returns: X + s[:,None]*v[None,:]
    """
    return X + s[:, None] * v[None, :]


def apply_tremor_rotation_to_magnetometer(mag_signal: np.ndarray, gyro_tremor: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply tremor-induced rotation to magnetometer signal.
    
    Physical model: Magnetometer tremor is modeled as a consequence of tremor-induced 
    orientation changes, not as independent additive tremor noise. The gyroscope tremor
    component drives small rotational perturbations that accumulate over time, changing
    the sensor's orientation relative to Earth's magnetic field.
    
    The cumulative rotation is updated at each timestep:
        R(t) = dR(t) @ R(t-1)
    where dR(t) is the incremental rotation from gyro_tremor[t] * dt.
    
    Args:
        mag_signal: Original magnetometer signal (L, 3) in original units
        gyro_tremor: Gyroscope tremor component (L, 3) in deg/s
        fs: Sampling frequency in Hz
        
    Returns:
        Rotated magnetometer signal (L, 3) with tremor-induced orientation changes
        
    Note:
        - Uses Rodrigues rotation formula for numerical stability
        - Maintains cumulative orientation over the window
        - Preserves the baseline magnetometer signal structure
        - Only applies secondary tremor-induced rotational perturbation
    """
    L = mag_signal.shape[0]
    dt = 1.0 / fs
    
    # Initialize output
    mag_rotated = np.zeros_like(mag_signal)
    
    # Convert gyro tremor from deg/s to rad/s
    gyro_tremor_rad = np.deg2rad(gyro_tremor)
    
    # Initialize cumulative rotation as identity
    R_cumulative = np.eye(3)
    
    # For each timestep, update cumulative rotation and apply to magnetometer
    for t in range(L):
        # Angular increment at this timestep (rad)
        # delta_theta = gyro_tremor * dt
        delta_theta = gyro_tremor_rad[t, :] * dt
        
        # Compute incremental rotation matrix using Rodrigues formula
        angle_norm = np.linalg.norm(delta_theta)
        
        if angle_norm < 1e-8:
            # Negligible rotation, use identity
            dR = np.eye(3)
        else:
            # Rodrigues rotation formula:
            # R = I + sin(θ)/θ [k]_× + (1-cos(θ))/θ² [k]_×²
            # where k = θ/||θ|| is the unit rotation axis
            k = delta_theta / angle_norm
            
            # Skew-symmetric cross-product matrix [k]_×
            K = np.array([
                [0,    -k[2],  k[1]],
                [k[2],  0,    -k[0]],
                [-k[1], k[0],  0]
            ])
            
            # Rodrigues formula
            dR = np.eye(3) + np.sin(angle_norm) * K + (1 - np.cos(angle_norm)) * (K @ K)
        
        # Update cumulative rotation: R = dR @ R
        # This accumulates orientation changes over time
        R_cumulative = dR @ R_cumulative
        
        # Apply cumulative rotation to original magnetometer vector
        mag_rotated[t, :] = R_cumulative @ mag_signal[t, :]
    
    return mag_rotated


def simulate_and_add_tremor_imu(
    X_acc: np.ndarray,   # (N,3)
    X_gyro: np.ndarray,  # (N,3)
    X_mag: np.ndarray,   # (N,3)
    fs: float,
    mu: float = 1.0,
    sigma: float = 0.5,
    dt: float = 0.001,
    freq_hz: float = 5.0,  # NEW: Tremor frequency in Hz (default 5 Hz)
    # Tremor strengths (RMS) in the SAME units as each sensor stream
    acc_rms: float = 0.2,
    gyro_rms: float = 0.2,
    mag_rms: float = 0.05,
    # Tremor mapping choices
    use_acc_state: str = "x1",   # "x1" or "x2"
    use_gyro_state: str = "x2",  # "x1" or "x2"
    use_mag_state: str = "x1",   # "x1" or "x2"
    # Intermittency
    intermittent: bool = False,
    on_prob: float = 0.5,
    min_on_sec: float = 2.0,
    max_on_sec: float = 8.0,
    seed: int | None = None,
):
    """
    End-to-end:
    1) simulate stochastic VdP -> x1(t), x2(t) with specified frequency
    2) resample to fs
    3) build s_acc, s_gyro, s_mag from x1/x2
    4) optional intermittent envelope
    5) project along 3D unit direction v (same for all modalities)
    6) return corrupted signals + (s's, v, env) for logging/debug
    
    Args:
        freq_hz: Tremor frequency in Hz (e.g., 4.5 for 4.5 Hz Parkinson tremor)
                 Range typically 3.5-7.0 Hz for Parkinson's disease
    """
    assert X_acc.shape == X_gyro.shape == X_mag.shape, "acc/gyro/mag must have same shape (N,3)"
    assert X_acc.shape[1] == 3, "expected (N,3) per sensor stream"

    rng = np.random.default_rng(seed)
    N = X_acc.shape[0]
    T_sec = N / fs

    # Convert frequency to angular frequency
    omega = 2.0 * np.pi * freq_hz

    # 1) simulate states on fine dt grid with specified frequency
    t, x1, x2 = simulate_stochastic_vdp_states(
        T_sec=T_sec, dt=dt, mu=mu, sigma=sigma, omega=omega, seed=seed
    )

    # 2) resample to sensor fs
    x1_s = resample_to_fs(t, x1, fs=fs, T_sec=T_sec)
    x2_s = resample_to_fs(t, x2, fs=fs, T_sec=T_sec)

    # 3) choose which state drives which modality
    def pick_state(which: str):
        return x1_s if which == "x1" else x2_s

    s_acc = pick_state(use_acc_state) # s_acc = x1
    s_gyro = pick_state(use_gyro_state) # s_gyro = x2
    s_mag = pick_state(use_mag_state) # s_mag = x1

    # 4) scale each to desired RMS (units match the stream), target RMS decides the "strength" of the tremor
    s_acc = scale_to_rms(s_acc, acc_rms) 
    s_gyro = scale_to_rms(s_gyro, gyro_rms)
    s_mag = scale_to_rms(s_mag, mag_rms)

    # 5) optional intermittent envelope (same on/off for all modalities on this body part)
    env = np.ones(N, dtype=np.float64)
    if intermittent:
        env = intermittent_envelope(N, fs, rng, on_prob=on_prob, min_on_sec=min_on_sec, max_on_sec=max_on_sec)
        s_acc *= env
        s_gyro *= env
        s_mag *= env

    # 6) direction: one random unit vector for this body part (consistent across modalities)
    v = unit_random_direction(rng)

    # 7) add projected tremor
    X_acc_t = add_projected_tremor(X_acc, s_acc, v)
    X_gyro_t = add_projected_tremor(X_gyro, s_gyro, v)
    X_mag_t = add_projected_tremor(X_mag, s_mag, v)

    meta = {
        "v": v,
        "env": env,
        "s_acc": s_acc,
        "s_gyro": s_gyro,
        "s_mag": s_mag,
        "mu": mu,
        "sigma": sigma,
        "dt": dt,
        "freq_hz": freq_hz,  # NEW: Store frequency for labeling
        "omega": omega,
    }
    
    return X_acc_t, X_gyro_t, X_mag_t, meta


def write_tremor_params_file(
    output_path, 
    mu, 
    sigma, 
    dt,
    acc_alpha,
    gyro_alpha,
    mag_alpha,
    acc_rms_min,
    acc_rms_max,
    gyro_rms_min,
    gyro_rms_max,
    mag_rms_min,
    mag_rms_max,
    intermittent, 
    on_prob, 
    min_on_sec, 
    max_on_sec,
    scenario_seed
):
    """
    Write a tremor parameters file documenting the tremor simulation settings
    with RELATIVE RMS configuration (alpha-based scaling).
    """
    from pathlib import Path
    
    tremor_lines = [
        "=" * 60,
        "TREMOR NOISE PARAMETERS",
        "=" * 60,
        "",
        "Scenario Noise: TREMOR (Relative RMS Mode)",
        f"  - Tremor mu: {mu}, sigma: {sigma}",
        f"  - Tremor alpha (relative to window RMS):",
        f"      Acc: {acc_alpha} (clipped to [{acc_rms_min}, {acc_rms_max}])",
        f"      Gyro: {gyro_alpha} (clipped to [{gyro_rms_min}, {gyro_rms_max}])",
        f"      Mag: {mag_alpha} (clipped to [{mag_rms_min}, {mag_rms_max}])",
        f"  - Note: Tremor RMS = alpha * window_signal_RMS (per window, per sensor)",
        f"  - Intermittent: {intermittent}",
    ]
    if intermittent:
        tremor_lines.append(f"  - On prob: {on_prob}, duration: {min_on_sec}-{max_on_sec}s")
    tremor_lines.extend([
        f"  - All IMU sensors corrupted in all windows",
        f"  - Scenario SEED: {scenario_seed}",
        "",
        "=" * 60,
        "Pipeline:",
        "  1. Load raw sensor windows",
        "  2. Compute window RMS per sensor",
        "  3. Calculate target tremor RMS = alpha * window_RMS",
        "  4. Clip to [min, max] range",
        "  5. Inject tremor to RAW signal",
        "  6. Resample (if needed)",
        "  7. Apply z-score normalization",
        "",
        "=" * 60,
        "Technical Details:",
        f"  - Van der Pol mu: {mu} (controls nonlinearity / limit-cycle dynamics)",
        f"  - Stochastic sigma: {sigma} (controls irregularity)",
        f"  - Integration dt: {dt} s",
        f"  - Frequency: set per window by relative RMS mode (natural ~4-6 Hz)",
        f"  - Accelerometer uses state x1(t)",
        f"  - Gyroscope uses state x2(t) (90° phase shift)",
        f"  - Magnetometer uses state x1(t)",
        f"  - Same tremor direction per body part (ankle/arm/chest)",
        f"  - Consistent across Acc/Gyro/Mag for same body part",
        "",
        "Why relative RMS?",
        "  - Makes tremor severity consistent across activities",
        "  - Standing (low signal): tremor = alpha * low_RMS = moderate impact",
        "  - Running (high signal): tremor = alpha * high_RMS = moderate impact",
        "  - Without relativity: tremor could dominate quiet activities",
        "",
        "To generate different tremor scenarios:",
        "  Change NOISE_SCENARIO_SEED to get a different realization",
        "=" * 60,
    ])
    
    Path(output_path).write_text("\n".join(tremor_lines), encoding="utf-8")
    return output_path


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate RMS of signal with small epsilon to avoid division by zero."""
    return float(np.sqrt(np.mean(x**2)) + eps)


def precompute_tremor_cache_with_relative_rms(
    window_specs,
    sensor_column_mapping,
    data_loader_func,
    fs: float,
    tremor_mu: float,
    tremor_sigma: float,
    tremor_dt: float,
    tremor_acc_alpha: float,
    tremor_gyro_alpha: float,
    tremor_mag_alpha: float,
    tremor_acc_rms_min: float,
    tremor_acc_rms_max: float,
    tremor_gyro_rms_min: float,
    tremor_gyro_rms_max: float,
    tremor_mag_rms_min: float,
    tremor_mag_rms_max: float,
    tremor_intermittent: bool,
    tremor_on_prob: float,
    tremor_min_on_sec: float,
    tremor_max_on_sec: float,
    scenario_seed: int,
):
    """
    NOTE: This is the LEGACY relative RMS version. Consider using
    precompute_tremor_cache_with_parkinson_model() for more realistic
    subject-specific, body-part-aware, and activity-modulated tremor.
    """
    """
    Pre-compute tremor noise for all windows using RELATIVE RMS from actual signals.
    
    This function is designed to be called from data generators to prepare tremor
    noise before applying it to windows. It computes tremor strength relative to
    each window's signal amplitude, making tremor effects consistent across
    different activity intensities.
    
    Args:
        window_specs: List of tuples (subj_idx, label, win_start, win_end)
        sensor_column_mapping: Dict mapping sensor names to column indices
                              e.g., {"Acc_ankle": [5,6,7], "Gyro_ankle": [8,9,10], ...}
        data_loader_func: Function that takes subject_idx and returns full data array
        fs: Sampling frequency
        tremor_mu: Van der Pol mu parameter
        tremor_sigma: Stochastic noise sigma
        tremor_dt: Integration time step
        tremor_acc_alpha: Alpha for accelerometer (target_rms = alpha * window_rms)
        tremor_gyro_alpha: Alpha for gyroscope
        tremor_mag_alpha: Alpha for magnetometer
        tremor_acc_rms_min: Minimum RMS for accelerometer (clipping)
        tremor_acc_rms_max: Maximum RMS for accelerometer (clipping)
        tremor_gyro_rms_min: Minimum RMS for gyroscope (clipping)
        tremor_gyro_rms_max: Maximum RMS for gyroscope (clipping)
        tremor_mag_rms_min: Minimum RMS for magnetometer (clipping)
        tremor_mag_rms_max: Maximum RMS for magnetometer (clipping)
        tremor_intermittent: Whether to use intermittent tremor
        tremor_on_prob: Probability of tremor being on
        tremor_min_on_sec: Minimum on duration
        tremor_max_on_sec: Maximum on duration
        scenario_seed: Random seed for deterministic tremor generation
        
    Returns:
        tremor_cache: Dict with keys (window_idx, body_part) and values containing
                     'acc_noise', 'gyro_noise', 'mag_noise' arrays and 'meta' dict
    """
    print(f"\n{'='*80}")
    print(f"Pre-computing tremor cache with RELATIVE RMS...")
    print(f"{'='*80}")
    print(f"Tremor alpha: Acc={tremor_acc_alpha}, Gyro={tremor_gyro_alpha}, Mag={tremor_mag_alpha}")
    print(f"RMS clipping: Acc=[{tremor_acc_rms_min},{tremor_acc_rms_max}], "
          f"Gyro=[{tremor_gyro_rms_min},{tremor_gyro_rms_max}], "
          f"Mag=[{tremor_mag_rms_min},{tremor_mag_rms_max}]")
    
    tremor_cache = {}
    subj_cache = {}
    
    # Group windows by body part
    body_parts_to_process = {}
    for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
        # Each window needs tremor for ankle, arm, chest
        for body_part in ['ankle', 'arm', 'chest']:
            cache_key = (w_idx, body_part)
            if cache_key not in body_parts_to_process:
                body_parts_to_process[cache_key] = (subj_idx, win_start, win_end)
    
    print(f"Processing {len(body_parts_to_process)} unique (window, body_part) combinations...")
    
    # Process each (window_idx, body_part)
    for (w_idx, body_part), (subj_idx, win_start, win_end) in body_parts_to_process.items():
        # Load subject data if not cached
        if subj_idx not in subj_cache:
            subj_cache[subj_idx] = data_loader_func(subj_idx)
        
        data = subj_cache[subj_idx]
        
        # Get sensor names for this body part
        acc_sensor = f"Acc_{body_part}"
        gyro_sensor = f"Gyro_{body_part}"
        mag_sensor = f"Mag_{body_part}"
        
        # Load raw windows (before any preprocessing)
        acc_win = data[win_start:win_end, sensor_column_mapping[acc_sensor]]
        gyro_win = data[win_start:win_end, sensor_column_mapping[gyro_sensor]] if gyro_sensor in sensor_column_mapping else None
        mag_win = data[win_start:win_end, sensor_column_mapping[mag_sensor]] if mag_sensor in sensor_column_mapping else None
        
        # Compute RMS from actual windows
        acc_win_rms = rms(acc_win)
        gyro_win_rms = rms(gyro_win) if gyro_win is not None else 0.0
        mag_win_rms = rms(mag_win) if mag_win is not None else 0.0
        
        # Scale by alpha and clip to avoid extremes
        acc_target = np.clip(tremor_acc_alpha * acc_win_rms, tremor_acc_rms_min, tremor_acc_rms_max)
        gyro_target = np.clip(tremor_gyro_alpha * gyro_win_rms, tremor_gyro_rms_min, tremor_gyro_rms_max)
        mag_target = np.clip(tremor_mag_alpha * mag_win_rms, tremor_mag_rms_min, tremor_mag_rms_max)
        
        # Generate tremor with these RMS targets
        L = win_end - win_start
        
        # Create deterministic seed for this (window_idx, body_part)
        body_part_offset = {'ankle': 0, 'arm': 1, 'chest': 2}.get(body_part, 0)
        seed = (scenario_seed + w_idx * 7919 + body_part_offset * 10000) & 0xffffffff
        
        # Create dummy signals (zeros) to apply tremor
        X_acc_dummy = np.zeros((L, 3), dtype=np.float64)
        X_gyro_dummy = np.zeros((L, 3), dtype=np.float64)
        X_mag_dummy = np.zeros((L, 3), dtype=np.float64)
        
        # Generate tremor with window-relative RMS
        X_acc_t, X_gyro_t, X_mag_t, meta = simulate_and_add_tremor_imu(
            X_acc_dummy,
            X_gyro_dummy,
            X_mag_dummy,
            fs=fs,
            mu=tremor_mu,
            sigma=tremor_sigma,
            dt=tremor_dt,
            acc_rms=acc_target,
            gyro_rms=gyro_target,
            mag_rms=mag_target,
            use_acc_state="x1",
            use_gyro_state="x2",
            use_mag_state="x1",
            intermittent=tremor_intermittent,
            on_prob=tremor_on_prob,
            min_on_sec=tremor_min_on_sec,
            max_on_sec=tremor_max_on_sec,
            seed=seed
        )
        
        # Extract noise (difference from dummy)
        acc_noise = X_acc_t - X_acc_dummy
        gyro_noise = X_gyro_t - X_gyro_dummy
        mag_noise = X_mag_t - X_mag_dummy
        
        tremor_cache[cache_key] = {
            'acc_noise': acc_noise.astype(np.float32),
            'gyro_noise': gyro_noise.astype(np.float32),
            'mag_noise': mag_noise.astype(np.float32),
            'meta': meta
        }
    
    print(f"✓ Tremor cache populated with {len(tremor_cache)} entries")
    print(f"{'='*80}\n")
    
    return tremor_cache


def precompute_tremor_cache_with_parkinson_model(
    window_specs,
    sensor_column_mapping,
    data_loader_func,
    fs: float,
    tremor_mu: float = 1.0,
    tremor_sigma: float = 0.5,
    tremor_dt: float = 0.001,
    tremor_intermittent: bool = False,
    tremor_on_prob: float = 0.5,
    tremor_min_on_sec: float = 2.0,
    tremor_max_on_sec: float = 8.0,
    scenario_seed: int = 42,
    use_jitter: bool = True,
    jitter_std: float | None = None,
    sampling_method: str | None = None,
    augment_mode: str = "mild_mod",
):
    """
    Pre-compute tremor noise for all windows using PARKINSON PATIENT MODEL.
    
    This function supports two sampling methods:
    
    1. "subject" (default): Subject-specific parameters
       - Subject-specific baseline tremor severity (A_subject) - accelerometer RMS
       - Subject-specific tremor frequency (FREQ_TREMOR)
       - Severity-dependent k_g factor for gyroscope RMS calculation
       - Body part sensitivity (C: arm > ankle > chest)
       - Activity-dependent modulation (beta: high at rest, reduced during movement)
       - Optional window-to-window jitter for natural variability
       
       The tremor RMS for each window is calculated as:
           RMS_acc = A_subject[subject_id] * C[body_part] * beta[activity] * (1 + jitter)
           RMS_gyro = k_g(RMS_acc) * RMS_acc
           Freq = FREQ_TREMOR[subject_id]
    
    2. "interval": Per-window sampling from severity ranges
       - Samples tremor parameters independently for each window
       - RMS sampled from severity intervals (SCORE_RMS_RANGE)
       - Activity-dependent scaling applied after score/RMS sampling (BETA_ACTIVITY)
       - Frequency sampled randomly from range (FREQ_RANGE_HZ)
       - More variability, suited for augmentation studies
       
       The tremor RMS for each window is sampled as:
           score = random choice based on augment_mode
           RMS_acc_base ~ Uniform(SCORE_RMS_RANGE[score]) * (1 + variation)
           RMS_acc = RMS_acc_base * beta[activity]
           RMS_gyro = k_g(RMS_acc) * RMS_acc
           Freq ~ Uniform(FREQ_RANGE_HZ)
    
    Args:
        window_specs: List of tuples (subj_idx, label, win_start, win_end)
                     where subj_idx is 1-indexed subject ID
        sensor_column_mapping: Dict mapping sensor names to column indices
                              e.g., {"Acc_ankle": [5,6,7], "Gyro_ankle": [8,9,10], ...}
        data_loader_func: Function that takes subject_idx and returns full data array
        fs: Sampling frequency
        tremor_mu: Van der Pol mu parameter (controls oscillation)
        tremor_sigma: Stochastic noise sigma (controls irregularity)
        tremor_dt: Integration time step
        tremor_intermittent: Whether to use intermittent tremor envelope
        tremor_on_prob: Probability of tremor being on (if intermittent)
        tremor_min_on_sec: Minimum on duration (if intermittent)
        tremor_max_on_sec: Maximum on duration (if intermittent)
        scenario_seed: Random seed for deterministic tremor generation
        use_jitter: Whether to add window-to-window variability
        jitter_std: Jitter standard deviation (default from pk_config.JITTER_STD)
        sampling_method: "subject" (subject-based) or "interval" (per-window sampling)
                        None uses pk_config.DEFAULT_SAMPLING_METHOD
        augment_mode: For interval method: "clean", "mild_mod", or "mod_severe"
                      Ignored for subject method
        
    Returns:
        tremor_cache: Dict with keys (window_idx, body_part) containing
                     'acc_noise', 'gyro_noise', 'mag_noise' arrays and 'meta' dict
                     
    Example (subject method):
        >>> tremor_cache = precompute_tremor_cache_with_parkinson_model(
        ...     window_specs=[(1, 2, 0, 500), (1, 4, 500, 1000)],  # subj 1, activities 2 & 4
        ...     sensor_column_mapping=sensor_cols,
        ...     data_loader_func=load_subject_data,
        ...     fs=50.0,
        ...     sampling_method="subject",
        ...     scenario_seed=42
        ... )
        
    Example (interval method):
        >>> tremor_cache = precompute_tremor_cache_with_parkinson_model(
        ...     window_specs=[(1, 2, 0, 500), (1, 4, 500, 1000)],
        ...     sensor_column_mapping=sensor_cols,
        ...     data_loader_func=load_subject_data,
        ...     fs=50.0,
        ...     sampling_method="interval",
        ...     augment_mode="mild_mod",
        ...     scenario_seed=42
        ... )
    """
    # Set defaults
    if jitter_std is None:
        jitter_std = pk_config.JITTER_STD
    
    if sampling_method is None:
        sampling_method = pk_config.DEFAULT_SAMPLING_METHOD
    
    # Validate sampling method
    if sampling_method not in ["subject", "interval"]:
        raise ValueError(f"sampling_method must be 'subject' or 'interval', got '{sampling_method}'")
    
    print(f"\n{'='*80}")
    print(f"Pre-computing tremor cache with PARKINSON MODEL")
    print(f"{'='*80}")
    print(f"Sampling method: {sampling_method.upper()}")
    
    if sampling_method == "subject":
        print(f"Model: RMS_acc = A_subject[subj] * C[body_part] * beta[activity] * (1 + jitter)")
        print(f"       RMS_gyro = k_g(RMS_acc) * RMS_acc, RMS_mag = 0.0")
        print(f"       Frequency = FREQ_TREMOR[subject_id]")
    else:  # interval
        print(f"Model: Sample parameters per window from severity ranges")
        print(f"       score ~ choice based on augment_mode='{augment_mode}'")
        print(f"       RMS_acc_base ~ Uniform(SCORE_RMS_RANGE[score]) * (1 + variation)")
        print(f"       RMS_acc = RMS_acc_base * beta[activity]")
        print(f"       RMS_gyro = k_g(RMS_acc) * RMS_acc")
        print(f"       Frequency ~ Uniform({pk_config.FREQ_RANGE_HZ[0]}-{pk_config.FREQ_RANGE_HZ[1]} Hz)")
    
    print(f"Parameters:")
    if sampling_method == "subject":
        print(f"  - Subjects with tremor severity defined: {list(pk_config.A_SUBJECT.keys())}")
        print(f"  - Tremor frequencies: {pk_config.FREQ_TREMOR}")
        print(f"  - Body part scaling: {pk_config.C_BODY_PART}")
    else:  # interval
        print(f"  - Augment mode: {augment_mode}")
        print(f"  - RMS ranges: {pk_config.SCORE_RMS_RANGE}")
        print(f"  - Activity beta map: {pk_config.BETA_ACTIVITY}")
        print(f"  - Frequency range: {pk_config.FREQ_RANGE_HZ} Hz")
    print(f"  - Jitter: {'enabled' if use_jitter else 'disabled'} (std={jitter_std:.3f})")
    print(f"  - Scenario seed: {scenario_seed}")
    
    tremor_cache = {}
    subj_cache = {}
    
    # Create RNG for jitter and interval sampling
    jitter_rng = np.random.default_rng(scenario_seed + 99999)
    interval_rng = np.random.default_rng(scenario_seed + 88888) if sampling_method == "interval" else None
    
    # Group windows by body part
    body_parts_to_process = {}
    for w_idx, (subj_idx, label, win_start, win_end) in enumerate(window_specs):
        # Each window needs tremor for arm, ankle, chest
        # Process arm first so ankle can use its parameters (in interval mode)
        for body_part in ['arm', 'ankle', 'chest']:
            cache_key = (w_idx, body_part)
            if cache_key not in body_parts_to_process:
                body_parts_to_process[cache_key] = (subj_idx, label, win_start, win_end)
    
    print(f"Processing {len(body_parts_to_process)} unique (window, body_part) combinations...")
    
    # Process each (window_idx, body_part)
    for (w_idx, body_part), (subj_idx, label, win_start, win_end) in body_parts_to_process.items():
        
        # Load subject data if not cached
        if subj_idx not in subj_cache:
            subj_cache[subj_idx] = data_loader_func(subj_idx)
        
        data = subj_cache[subj_idx]
        
        # ========================================
        # Calculate tremor parameters based on sampling method
        # ========================================
        if sampling_method == "subject":
            # Subject-based method: use subject-specific parameters
            # Check if subject has Parkinson parameters defined
            if subj_idx not in pk_config.A_SUBJECT:
                print(f"Warning: Subject {subj_idx} not in Parkinson config. Skipping tremor for this subject.")
                # Store zero tremor
                L = win_end - win_start
                tremor_cache[(w_idx, body_part)] = {
                    'acc_noise': np.zeros((L, 3), dtype=np.float32),
                    'gyro_noise': np.zeros((L, 3), dtype=np.float32),
                    'mag_noise': np.zeros((L, 3), dtype=np.float32),
                    'meta': {'subject_id': subj_idx, 'no_tremor': True}
                }
                continue
            
            # Calculate tremor RMS using subject-specific model
            acc_target = pk_config.get_tremor_rms(
                subj_idx, "acc", body_part, label,
                jitter_std=jitter_std if use_jitter else 0.0,
                rng=jitter_rng
            )
            
            gyro_target = pk_config.get_tremor_rms(
                subj_idx, "gyro", body_part, label,
                jitter_std=jitter_std if use_jitter else 0.0,
                rng=jitter_rng
            )
            
            mag_target = 0.0  # No magnetometer tremor
            
            # Get subject-specific tremor frequency
            freq_hz = pk_config.get_tremor_frequency(subj_idx)
            
        else:  # sampling_method == "interval"
            # Interval-based method: sample parameters per window
            # For ankle, use arm's parameters with scaling; otherwise sample new params
            activity_beta = float(pk_config.BETA_ACTIVITY.get(label, 1.0))

            if body_part == 'ankle':
                # Ankle is derived from arm tremor for the same window
                # (arm is always processed first, so it will be in cache)
                arm_key = (w_idx, 'arm')
                arm_meta = tremor_cache[arm_key]['meta']
                score = int(arm_meta.get('sampled_score', arm_meta.get('score', 0)))
                freq_hz = arm_meta['freq_hz']
                arm_acc_target = arm_meta['acc_target_rms']
                arm_gyro_target = arm_meta['gyro_target_rms']
                activity_beta = float(arm_meta.get('activity_beta', activity_beta))
                
                # Apply ankle scaling
                ankle_ratio = pk_config.ANKLE_RATIO_BY_SCORE[score]
                acc_target = arm_acc_target * ankle_ratio
                gyro_target = arm_gyro_target * ankle_ratio
            else:
                # For arm and chest, sample new parameters
                score, acc_target, freq_hz = pk_config.sample_tremor_params(
                    augment_mode=augment_mode,
                    rng=interval_rng
                )

                # Apply activity-dependent modulation after score/RMS sampling
                # to preserve the requested "draw score first, then scale by activity" flow.
                acc_target = max(0.0, float(acc_target) * activity_beta)
                
                # Calculate gyroscope RMS from accelerometer RMS
                k_g = pk_config.choose_kg_from_rms_acc(acc_target)
                gyro_target = k_g * acc_target
            
            mag_target = 0.0  # No magnetometer tremor
        
        # ========================================
        # Generate tremor with calculated parameters
        # ========================================
        L = win_end - win_start
        
        # Create deterministic seed for this (window_idx, body_part)
        body_part_offset = {'ankle': 0, 'arm': 1, 'chest': 2}.get(body_part, 0)
        seed = (scenario_seed + w_idx * 7919 + body_part_offset * 10000) & 0xffffffff
        
        # Create dummy signals (zeros) to apply tremor
        X_acc_dummy = np.zeros((L, 3), dtype=np.float64)
        X_gyro_dummy = np.zeros((L, 3), dtype=np.float64)
        X_mag_dummy = np.zeros((L, 3), dtype=np.float64)
        
        # Generate tremor using calculated RMS values and subject-specific frequency
        X_acc_t, X_gyro_t, X_mag_t, meta = simulate_and_add_tremor_imu(
            X_acc_dummy,
            X_gyro_dummy,
            X_mag_dummy,
            fs=fs,
            freq_hz=freq_hz,
            mu=tremor_mu,
            sigma=tremor_sigma,
            dt=tremor_dt,
            acc_rms=acc_target,
            gyro_rms=gyro_target,
            mag_rms=mag_target,
            use_acc_state="x1",
            use_gyro_state="x2",
            use_mag_state="x1",
            intermittent=tremor_intermittent,
            on_prob=tremor_on_prob,
            min_on_sec=tremor_min_on_sec,
            max_on_sec=tremor_max_on_sec,
            seed=seed
        )
        
        # Extract noise (difference from dummy)
        acc_noise = X_acc_t - X_acc_dummy
        gyro_noise = X_gyro_t - X_gyro_dummy
        mag_noise = X_mag_t - X_mag_dummy
        
        # Add metadata based on sampling method
        meta['subject_id'] = subj_idx
        meta['body_part'] = body_part
        meta['activity'] = label
        meta['acc_target_rms'] = acc_target
        meta['gyro_target_rms'] = gyro_target
        meta['mag_target_rms'] = mag_target
        meta['freq_hz'] = freq_hz
        meta['sampling_method'] = sampling_method
        
        if sampling_method == "subject":
            meta['severity'] = pk_config.get_subject_severity(subj_idx)
        else:  # interval
            sampled_score = int(score)
            scaled_score = pk_config.get_tremor_score(acc_target)

            # Keep interval-sampled rest score as canonical segment label.
            score = sampled_score

            if score == 0:
                meta['severity'] = 'none'
            elif score == 1:
                meta['severity'] = 'mild'
            elif score == 2:
                meta['severity'] = 'mild-moderate'
            elif score == 3:
                meta['severity'] = 'moderate-severe'
            else:
                meta['severity'] = 'severe'
            meta['augment_mode'] = augment_mode
            meta['score'] = score
            meta['sampled_score'] = sampled_score
            meta['scaled_score'] = scaled_score
            meta['activity_beta'] = activity_beta
        
        tremor_cache[(w_idx, body_part)] = {
            'acc_noise': acc_noise.astype(np.float32),
            'gyro_noise': gyro_noise.astype(np.float32),
            'mag_noise': mag_noise.astype(np.float32),
            'meta': meta
        }
    
    print(f"✓ Tremor cache populated with {len(tremor_cache)} entries")
    print(f"{'='*80}\n")
    
    return tremor_cache


def write_tremor_parkinson_params_file(
    output_path,
    mu,
    sigma,
    dt,
    intermittent,
    on_prob,
    min_on_sec,
    max_on_sec,
    use_jitter,
    jitter_std,
    scenario_seed
):
    """
    Write a tremor parameters file documenting the Parkinson model settings.
    """
    from pathlib import Path
    
    tremor_lines = [
        "=" * 80,
        "TREMOR NOISE PARAMETERS - PARKINSON PATIENT MODEL",
        "=" * 80,
        "",
        "Model: RMS_acc = A_subject[subj] * C[body_part] * beta[activity] * (1 + jitter)",
        "       RMS_gyro = k_g(RMS_acc) * RMS_acc",
        "       RMS_mag = 0.0 (no tremor)",
        "",
        "=" * 80,
        "Subject-Specific Baseline (A_subject) - Accelerometer RMS:",
        "=" * 80,
    ]
    
    for subj_id, acc_rms in sorted(pk_config.A_SUBJECT.items()):
        severity = pk_config.get_subject_severity(subj_id)
        k_g = pk_config.choose_kg_from_rms_acc(acc_rms)
        gyro_rms = k_g * acc_rms
        freq = pk_config.get_tremor_frequency(subj_id)
        tremor_lines.append(
            f"  Subject {subj_id:2d} ({severity:16s}): "
            f"Acc={acc_rms:.2f} m/s² -> k_g={k_g:.1f} -> Gyro={gyro_rms:.2f} deg/s, Freq={freq:.1f} Hz"
        )
    
    tremor_lines.extend([
        "",
        "=" * 80,
        "Tremor Frequency (Hz) per Subject:",
        "=" * 80,
    ])
    
    for subj_id, freq in sorted(pk_config.FREQ_TREMOR.items()):
        tremor_lines.append(f"  Subject {subj_id:2d}: {freq:.1f} Hz")
    
    tremor_lines.extend([
        "",
        "=" * 80,
        "Body Part Scaling (C):",
        "=" * 80,
    ])
    
    for bp, scale in sorted(pk_config.C_BODY_PART.items()):
        tremor_lines.append(f"  {bp.capitalize():7s}: {scale:.2f}x")
    
    tremor_lines.extend([
        "",
        "=" * 80,
        "Activity Modulation (beta):",
        "=" * 80,
        "  Resting activities (higher tremor):",
    ])
    
    for act in [1, 2, 3]:
        if act in pk_config.BETA_ACTIVITY:
            tremor_lines.append(f"    Activity {act:2d}: beta={pk_config.BETA_ACTIVITY[act]:.2f}")
    
    tremor_lines.append("  Movement activities (reduced tremor):")
    for act in [4, 9, 10, 11]:
        if act in pk_config.BETA_ACTIVITY:
            tremor_lines.append(f"    Activity {act:2d}: beta={pk_config.BETA_ACTIVITY[act]:.2f}")
    
    tremor_lines.extend([
        "",
        "=" * 80,
        "Technical Parameters:",
        "=" * 80,
        f"  Van der Pol mu: {mu} (controls nonlinearity / limit-cycle dynamics)",
        f"  Stochastic sigma: {sigma} (controls irregularity)",
        f"  Integration dt: {dt} s",
        f"  Frequency: subject-specific (freq_hz, omega = 2πf)",
        f"  Jitter: {'enabled' if use_jitter else 'disabled'}",
    ])
    
    if use_jitter:
        tremor_lines.append(f"    Jitter std: {jitter_std:.3f} ({jitter_std*100:.1f}% variability)")
    
    tremor_lines.extend([
        f"  Intermittent: {intermittent}",
    ])
    
    if intermittent:
        tremor_lines.extend([
            f"    On probability: {on_prob}",
            f"    Duration: {min_on_sec}-{max_on_sec}s",
        ])
    
    tremor_lines.extend([
        f"  Scenario SEED: {scenario_seed}",
        "",
        "=" * 80,
        "Oscillator States:",
        "=" * 80,
        "  Accelerometer: uses x1(t) (position-like)",
        "  Gyroscope: uses x2(t) (velocity-like, 90° phase shift)",
        "  Magnetometer: uses x1(t) (position-like)",
        "  Same tremor direction per body part across all sensors",
        "",
        "=" * 80,
        "Pipeline:",
        "=" * 80,
        "  1. For each window: identify subject_id, body_part, activity",
        "  2. Calculate RMS using Parkinson model formula",
        "  3. Generate van der Pol oscillator with target RMS",
        "  4. Apply tremor to RAW sensor data",
        "  5. Resample (if needed)",
        "  6. Apply z-score normalization",
        "",
        "=" * 80,
        "Clinical Realism:",
        "=" * 80,
        "  ✓ Subject-specific severity (mild/moderate/severe)",
        "  ✓ Upper limbs more affected than lower limbs",
        "  ✓ Prominent resting tremor (sitting, standing)",
        "  ✓ Reduced tremor during active movement",
        "  ✓ Natural window-to-window variability",
        "=" * 80,
    ])
    
    Path(output_path).write_text("\n".join(tremor_lines), encoding="utf-8")
    return output_path
