import numpy as np

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
    x1_0: float = 0.0,
    x2_0: float = 1.0,
    seed: int | None = None,
):
    """
    Simulate stochastic van der Pol oscillator (Euler–Maruyama).
    Returns: t (n_steps,), x1 (n_steps,), x2 (n_steps,)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T_sec / dt))
    t = np.arange(n_steps, dtype=np.float64) * dt

    x1 = np.zeros(n_steps, dtype=np.float64)
    x2 = np.zeros(n_steps, dtype=np.float64)
    x1[0], x2[0] = x1_0, x2_0

    for k in range(n_steps - 1):
        dx1 = x2[k]
        dx2 = mu * (1.0 - x1[k] ** 2) * x2[k] - x1[k]
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


def simulate_and_add_tremor_imu(
    X_acc: np.ndarray,   # (N,3)
    X_gyro: np.ndarray,  # (N,3)
    X_mag: np.ndarray,   # (N,3)
    fs: float,
    mu: float = 1.0,
    sigma: float = 0.5,
    dt: float = 0.001,
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
    1) simulate stochastic VdP -> x1(t), x2(t)
    2) resample to fs
    3) build s_acc, s_gyro, s_mag from x1/x2
    4) optional intermittent envelope
    5) project along 3D unit direction v (same for all modalities)
    6) return corrupted signals + (s's, v, env) for logging/debug
    """
    assert X_acc.shape == X_gyro.shape == X_mag.shape, "acc/gyro/mag must have same shape (N,3)"
    assert X_acc.shape[1] == 3, "expected (N,3) per sensor stream"

    rng = np.random.default_rng(seed)
    N = X_acc.shape[0]
    T_sec = N / fs

    # 1) simulate states on fine dt grid
    t, x1, x2 = simulate_stochastic_vdp_states(T_sec=T_sec, dt=dt, mu=mu, sigma=sigma, seed=seed)

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
        f"  - Van der Pol mu: {mu} (controls oscillation frequency)",
        f"  - Stochastic sigma: {sigma} (controls irregularity)",
        f"  - Integration dt: {dt} s",
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
