"""
Dropout noise simulation for sensor data corruption.

This module simulates sensor dropout/failure by setting all values
in a window to a constant value (typically 0), representing complete
signal loss or sensor malfunction.
"""

import numpy as np

# Default parameter
DROPOUT_VALUE = 0.0  # Constant value to set (usually 0 for complete signal loss)


def apply_dropout(
    win: np.ndarray,
    value: float = 0.0
) -> np.ndarray:
    """
    Apply dropout corruption by setting entire window to a constant value.
    
    Simulates complete sensor failure or dropout where the sensor stops
    providing meaningful readings and outputs a constant value.
    
    Parameters
    ----------
    win : np.ndarray
        Input window of shape (L, C) where L is window length and C is number of channels.
    value : float, optional
        The constant value to set all samples to. Default is 0.0.
        Common choices:
        - 0.0: Complete signal loss
        - Last known value: Sample-and-hold behavior
    
    Returns
    -------
    np.ndarray
        Corrupted window filled with constant value, same shape (L, C) as input.
    
    Notes
    -----
    - This is a deterministic corruption (no randomness within the window)
    - All channels are set to the same constant value
    - Represents worst-case scenario for sensor reliability
    
    Examples
    --------
    >>> signal = np.random.randn(100, 3)  # (L=100, C=3)
    >>> dropout_signal = apply_dropout(signal, value=0.0)
    >>> assert np.all(dropout_signal == 0.0)
    """
    return np.full_like(win, value)
