"""
Test script for noise simulation modules.
Verifies that all noise functions work correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Noise_simulation.AWGN import apply_awgn, apply_awgn_seed_based, apply_awgn_rms_ratio
from Noise_simulation.Dropout import apply_dropout
from Noise_simulation.WeakSignal import apply_weak_signal


def test_awgn():
    """Test AWGN noise application (legacy z-score method)."""
    print("\n" + "="*60)
    print("Testing AWGN (Additive White Gaussian Noise - legacy)")
    print("="*60)
    
    # Create test signal
    rng = np.random.RandomState(42)
    signal = np.random.randn(100, 3)
    
    # Apply AWGN
    noisy_signal = apply_awgn(signal, sigma=0.3, rng=rng)
    
    # Verify shape preserved
    assert signal.shape == noisy_signal.shape, "Shape mismatch!"
    
    # Verify noise added (signal changed)
    assert not np.allclose(signal, noisy_signal), "Signal unchanged!"
    
    # Compute SNR
    noise = noisy_signal - signal
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print(f"✓ Shape preserved: {signal.shape}")
    print(f"✓ Noise added successfully")
    print(f"✓ Signal power: {signal_power:.4f}")
    print(f"✓ Noise power: {noise_power:.4f}")
    print(f"✓ SNR: {snr_db:.2f} dB")
    
    # Test seed-based version
    noisy_signal_seed = apply_awgn_seed_based(signal, sigma=0.3, seed=42)
    print(f"✓ Seed-based version works")
    
    print("✓ AWGN test PASSED")


def test_awgn_rms_ratio():
    """Test AWGN with RMS ratio (raw signal method)."""
    print("\n" + "="*60)
    print("Testing AWGN with RMS Ratio (raw signal)")
    print("="*60)
    
    # Create test signal with realistic raw sensor scale
    rng = np.random.RandomState(42)
    signal = np.random.randn(50, 3) * 100  # Scale to simulate raw sensor units
    
    # Apply AWGN with 20% RMS ratio
    rms_ratio = 0.2
    noisy_signal = apply_awgn_rms_ratio(signal, rms_ratio=rms_ratio, rng=rng)
    
    # Verify shape preserved
    assert signal.shape == noisy_signal.shape, "Shape mismatch!"
    
    # Verify noise added (signal changed)
    assert not np.allclose(signal, noisy_signal), "Signal unchanged!"
    
    # Compute actual RMS ratio per channel
    sig_rms = np.sqrt(np.mean(signal ** 2, axis=0))
    noise = noisy_signal - signal
    noise_rms = np.sqrt(np.mean(noise ** 2, axis=0))
    actual_ratio = noise_rms / sig_rms
    
    print(f"✓ Shape preserved: {signal.shape}")
    print(f"✓ Noise added successfully")
    print(f"✓ Target RMS ratio: {rms_ratio}")
    print(f"✓ Actual RMS ratios per channel: {actual_ratio}")
    print(f"✓ Mean actual ratio: {np.mean(actual_ratio):.4f}")
    
    # Verify RMS ratio is approximately correct (within 10% tolerance)
    assert np.allclose(actual_ratio, rms_ratio, rtol=0.1), \
        f"RMS ratio mismatch! Expected ~{rms_ratio}, got {actual_ratio}"
    
    print("✓ AWGN RMS ratio test PASSED")


def test_dropout():
    """Test dropout corruption."""
    print("\n" + "="*60)
    print("Testing Dropout (Sensor Failure)")
    print("="*60)
    
    # Create test signal
    signal = np.random.randn(100, 3)
    
    # Apply dropout
    dropout_signal = apply_dropout(signal, value=0.0)
    
    # Verify shape preserved
    assert signal.shape == dropout_signal.shape, "Shape mismatch!"
    
    # Verify all values are zero
    assert np.all(dropout_signal == 0.0), "Not all values are zero!"
    
    print(f"✓ Shape preserved: {signal.shape}")
    print(f"✓ All values set to 0.0")
    print(f"✓ Signal completely dropped out")
    
    # Test with different value
    dropout_signal_custom = apply_dropout(signal, value=-999.0)
    assert np.all(dropout_signal_custom == -999.0), "Custom value not applied!"
    print(f"✓ Custom dropout value works")
    
    print("✓ Dropout test PASSED")


def test_weak_signal():
    """Test weak signal attenuation."""
    print("\n" + "="*60)
    print("Testing Weak Signal (Signal Attenuation)")
    print("="*60)
    
    # Create test signal
    signal = np.random.randn(100, 3)
    
    # Apply weak signal
    factor = 0.2
    weak_signal = apply_weak_signal(signal, factor=factor)
    
    # Verify shape preserved
    assert signal.shape == weak_signal.shape, "Shape mismatch!"
    
    # Verify correct attenuation
    assert np.allclose(weak_signal, signal * factor), "Incorrect attenuation!"
    
    # Compute power reduction
    signal_power = np.mean(signal ** 2)
    weak_power = np.mean(weak_signal ** 2)
    power_ratio = weak_power / signal_power
    expected_ratio = factor ** 2
    
    print(f"✓ Shape preserved: {signal.shape}")
    print(f"✓ Attenuation factor: {factor}")
    print(f"✓ Signal power: {signal_power:.4f}")
    print(f"✓ Weak signal power: {weak_power:.4f}")
    print(f"✓ Power ratio: {power_ratio:.4f} (expected: {expected_ratio:.4f})")
    print(f"✓ Signal correctly attenuated")
    
    print("✓ Weak Signal test PASSED")


def test_combined():
    """Test combining multiple noise types."""
    print("\n" + "="*60)
    print("Testing Combined Noise Application")
    print("="*60)
    
    # Create test signal
    rng = np.random.RandomState(123)
    signal = np.random.randn(100, 3)
    
    # Apply weak signal first, then AWGN
    weak = apply_weak_signal(signal, factor=0.5)
    noisy_weak = apply_awgn(weak, sigma=0.1, rng=rng)
    
    print(f"✓ Original signal shape: {signal.shape}")
    print(f"✓ After weak signal: shape={weak.shape}")
    print(f"✓ After AWGN: shape={noisy_weak.shape}")
    print(f"✓ Combined corruption successful")
    
    print("✓ Combined test PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "NOISE SIMULATION MODULE TESTS")
    print("="*70)
    
    try:
        test_awgn()
        test_awgn_rms_ratio()
        test_dropout()
        test_weak_signal()
        test_combined()
        
        print("\n" + "="*70)
        print(" "*20 + "ALL TESTS PASSED ✓")
        print("="*70)
        print("\nAll noise simulation modules are working correctly!")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
