"""
Weak signal simulation for sensor data corruption.

This module simulates reduced signal strength or sensor degradation
by attenuating the signal amplitude through multiplication by a factor < 1.
"""

import numpy as np

# Default parameter
WEAK_SIGNAL_FACTOR = 0.2  # Multiplication factor for signal attenuation (e.g., 0.2 or 0.5)


def apply_weak_signal(
    win: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    Apply weak signal corruption by attenuating signal amplitude.
    
    Simulates reduced signal strength due to:
    - Sensor degradation or aging
    - Poor contact/connection
    - Low battery
    - Distance from signal source
    - Environmental interference
    
    Parameters
    ----------
    win : np.ndarray
        Input window of shape (L, C) where L is window length and C is number of channels.
    factor : float
        Multiplication factor to attenuate the signal.
        Should be < 1.0 for signal weakening.
        Common values:
        - 0.5: 50% signal strength (moderate degradation)
        - 0.2: 20% signal strength (severe degradation)
        - 0.1: 10% signal strength (critical degradation)
    
    Returns
    -------
    np.ndarray
        Attenuated signal with same shape (L, C) as input.
    
    Notes
    -----
    - This is a deterministic corruption (no randomness)
    - All channels are attenuated by the same factor
    - Signal shape is preserved, only amplitude is reduced
    - For z-scored signals, this reduces the variance
    
    Examples
    --------
    >>> signal = np.random.randn(100, 3)  # (L=100, C=3)
    >>> weak_signal = apply_weak_signal(signal, factor=0.2)
    >>> assert np.allclose(weak_signal, signal * 0.2)
    """
    return win * factor
