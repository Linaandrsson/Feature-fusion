"""
Additive White Gaussian Noise (AWGN) simulation for sensor data corruption.

This module provides functionality to add Gaussian noise to sensor signals,
simulating hardware measurement noise. The noise is applied to RAW sensor
readings BEFORE resampling and z-score normalization, providing a more
realistic corruption model.
"""

import numpy as np

# Default parameters
AWGN_SIGMA = 0.3  # Standard deviation of Gaussian noise (legacy, for post-zscore application)
AWGN_RMS_RATIO = 0.2  # Noise RMS as fraction of signal RMS (applied to raw signal)


def apply_awgn(
    win: np.ndarray,
    sigma: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Apply Additive White Gaussian Noise to a window.
    
    Parameters
    ----------
    win : np.ndarray
        Input window of shape (L, C) where L is window length and C is number of channels.
        Expected to be z-scored (mean~0, std~1 per channel).
    sigma : float
        Standard deviation of the Gaussian noise to add.
        Since win is z-scored with std~1, sigma directly controls the noise level.
        Example: sigma=0.3 means noise with 30% of signal std.
    rng : np.random.RandomState
        Random number generator for reproducibility.
    
    Returns
    -------
    np.ndarray
        Corrupted window with same shape (L, C) as input.
    
    Notes
    -----
    - AWGN is added independently to each sample (time step and channel)
    - The noise is drawn from N(0, sigma²)
    - For z-scored signals with unit variance, sigma represents the
      signal-to-noise ratio in standard deviation units
    
    Examples
    --------
    >>> rng = np.random.RandomState(42)
    >>> clean_signal = np.random.randn(100, 3)  # (L=100, C=3)
    >>> noisy_signal = apply_awgn(clean_signal, sigma=0.3, rng=rng)
    """
    return win + sigma * rng.randn(*win.shape)


def apply_awgn_rms_ratio(
    win: np.ndarray,
    rms_ratio: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Apply AWGN to raw sensor signal with noise level relative to signal RMS.
    
    This function simulates hardware measurement noise by adding Gaussian noise
    BEFORE any preprocessing (resampling, z-score). The noise magnitude is scaled
    relative to the signal's RMS per channel, making it independent of signal amplitude.
    
    Parameters
    ----------
    win : np.ndarray
        Raw input window of shape (L, C) where L is window length and C is number of channels.
        Should be unprocessed sensor measurements at original sampling rate.
    rms_ratio : float
        Noise RMS as a fraction of signal RMS per channel.
        Example: rms_ratio=0.2 means noise RMS is 20% of signal RMS.
        Typical values: 0.1-0.3 for realistic sensor noise.
    rng : np.random.RandomState
        Random number generator for reproducibility.
    
    Returns
    -------
    np.ndarray
        Corrupted window with same shape (L, C) as input.
    
    Notes
    -----
    - Noise is added independently per channel
    - Signal RMS is computed per channel: sqrt(mean(signal²))
    - Noise is generated as N(0,1), then scaled to target RMS per channel
    - Includes numerical guard (eps=1e-8) to prevent division by zero for flat signals
    - Applied BEFORE resampling and z-score normalization
    
    Examples
    --------
    >>> rng = np.random.RandomState(42)
    >>> raw_signal = np.random.randn(50, 3) * 100  # (L=50, C=3), raw sensor units
    >>> noisy_signal = apply_awgn_rms_ratio(raw_signal, rms_ratio=0.2, rng=rng)
    """
    eps = 1e-8
    
    # Compute signal RMS per channel: sqrt(mean(x²))
    sig_rms = np.sqrt(np.mean(win ** 2, axis=0, keepdims=True))  # (1, C)
    
    # Generate unit Gaussian noise
    noise = rng.randn(*win.shape)  # (L, C)
    
    # Compute noise RMS per channel
    noise_rms = np.sqrt(np.mean(noise ** 2, axis=0, keepdims=True))  # (1, C)
    
    # Scale noise to target RMS = rms_ratio * sig_rms
    target_noise_rms = rms_ratio * sig_rms
    scaled_noise = noise * (target_noise_rms / (noise_rms + eps))
    
    return win + scaled_noise


def apply_awgn_seed_based(
    win: np.ndarray,
    sigma: float,
    seed: int
) -> np.ndarray:
    """
    Apply AWGN using a deterministic seed.
    
    Convenience function that creates an RNG from seed and applies AWGN.
    Useful for reproducible corruption per window.
    
    Parameters
    ----------
    win : np.ndarray
        Input window of shape (L, C).
    sigma : float
        Standard deviation of the Gaussian noise.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Corrupted window with same shape as input.
    """
    rng = np.random.RandomState(seed)
    return apply_awgn(win, sigma, rng)
