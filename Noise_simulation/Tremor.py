import numpy as np

# Default parameters for Tremor noise
# Van der Pol oscillator parameters
TREMOR_MU = 1.0                  # van der Pol nonlinearity parameter
TREMOR_SIGMA = 0.5               # stochastic noise intensity
TREMOR_DT = 0.001                # integration time step (seconds)

# Tremor strength per modality (RMS in sensor units, after z-score normalization)
TREMOR_ACC_RMS = 0.3             # accelerometer tremor strength
TREMOR_GYRO_RMS = 0.5            # gyroscope tremor strength
TREMOR_MAG_RMS = 0.1            # magnetometer tremor strength (typically lower)

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
    acc_rms, 
    gyro_rms, 
    mag_rms, 
    intermittent, 
    on_prob, 
    min_on_sec, 
    max_on_sec,
    scenario_seed
):
    """
    Write a tremor parameters file documenting the tremor simulation settings.
    """
    from pathlib import Path
    
    tremor_lines = [
        "=" * 60,
        "TREMOR NOISE PARAMETERS",
        "=" * 60,
        "",
        "Scenario Noise: TREMOR",
        f"  - Tremor mu: {mu}, sigma: {sigma}",
        f"  - Tremor RMS - Acc: {acc_rms}, Gyro: {gyro_rms}, Mag: {mag_rms}",
        f"  - Intermittent: {intermittent}",
    ]
    if intermittent:
        tremor_lines.append(f"  - On prob: {on_prob}, duration: {min_on_sec}-{max_on_sec}s")
    tremor_lines.extend([
        f"  - All sensors corrupted in all windows",
        f"  - Scenario SEED: {scenario_seed}",
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
        "To generate different tremor scenarios:",
        "  Change NOISE_SCENARIO_SEED to get a different realization",
        "=" * 60,
    ])
    
    Path(output_path).write_text("\n".join(tremor_lines), encoding="utf-8")
    return output_path
