"""
Test: Verify that precompute_tremor_cache_with_parkinson_model() 
      uses subject-specific frequencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import Tremor
import tremor_parkinson_config as pk_config

def test_tremor_cache_has_correct_frequencies():
    """
    Test that tremor cache includes subject-specific frequencies
    """
    print("=" * 80)
    print("TEST: Tremor Cache with Subject-Specific Frequencies")
    print("=" * 80)
    
    # Create simple window specs for subjects 1, 5, and 10
    # (subj_idx, activity_label, win_start, win_end)
    window_specs = [
        (1, 2, 0, 250),    # Subject 1, sitting, 5 sec @ 50Hz
        (5, 2, 0, 250),    # Subject 5, sitting, 5 sec @ 50Hz
        (10, 2, 0, 250),   # Subject 10, sitting, 5 sec @ 50Hz
    ]
    
    # Dummy sensor column mapping (not used for zeros)
    sensor_column_mapping = {
        "Acc_ankle": [0, 1, 2],
        "Gyro_ankle": [3, 4, 5],
        "Mag_ankle": [6, 7, 8],
        "Acc_arm": [9, 10, 11],
        "Gyro_arm": [12, 13, 14],
        "Mag_arm": [15, 16, 17],
        "Acc_chest": [18, 19, 20],
        "Gyro_chest": [21, 22, 23],
        "Mag_chest": [24, 25, 26],
    }
    
    # Dummy data loader (returns zeros)
    def dummy_data_loader(subj_idx):
        return np.zeros((1000, 27))  # Dummy data
    
    # Generate tremor cache
    tremor_cache = Tremor.precompute_tremor_cache_with_parkinson_model(
        window_specs=window_specs,
        sensor_column_mapping=sensor_column_mapping,
        data_loader_func=dummy_data_loader,
        fs=50.0,
        tremor_mu=1.0,
        tremor_sigma=0.5,
        tremor_dt=0.001,
        scenario_seed=42,
        use_jitter=False,  # Disable jitter for consistent testing
    )
    
    print("\n" + "=" * 80)
    print("RESULTS: Checking frequencies in cache metadata")
    print("=" * 80)
    
    # Expected frequencies from config
    expected_freqs = {
        1: 5.5,   # Mild
        5: 4.8,   # Mild-moderate
        10: 4.0,  # Severe
    }
    
    # Check each window
    print("\nWindow | Subject | Body Part | Expected Freq | Actual Freq | Match")
    print("-" * 80)
    
    all_correct = True
    for w_idx, (subj_idx, _, _, _) in enumerate(window_specs):
        for body_part in ['ankle', 'arm', 'chest']:
            cache_entry = tremor_cache.get((w_idx, body_part))
            
            if cache_entry is None:
                print(f"  {w_idx:3d}  |   {subj_idx:2d}    | {body_part:9s} | ERROR: Not in cache")
                all_correct = False
                continue
            
            meta = cache_entry['meta']
            actual_freq = meta.get('freq_hz', None)
            expected_freq = expected_freqs[subj_idx]
            
            match = "✓" if actual_freq == expected_freq else "✗"
            if actual_freq != expected_freq:
                all_correct = False
            
            print(f"  {w_idx:3d}  |   {subj_idx:2d}    | {body_part:9s} | "
                  f"   {expected_freq:.1f} Hz    |   {actual_freq:.1f} Hz   | {match}")
    
    print("\n" + "=" * 80)
    if all_correct:
        print("✓ TEST PASSED: All frequencies match expected values!")
    else:
        print("✗ TEST FAILED: Some frequencies don't match!")
    print("=" * 80)
    
    return all_correct


def test_frequency_verification_with_fft():
    """
    Verify generated tremor has correct frequency using FFT
    """
    print("\n" + "=" * 80)
    print("VERIFICATION: FFT Analysis of Generated Tremor")
    print("=" * 80)
    
    subjects = [1, 5, 10]
    fs = 50.0
    T_sec = 10.0  # Longer window for better frequency resolution
    N = int(T_sec * fs)
    
    print("\nSubject | Expected Freq | FFT Peak Freq | Match")
    print("-" * 60)
    
    all_correct = True
    for subj_idx in subjects:
        # Get expected frequency
        expected_freq = pk_config.get_tremor_frequency(subj_idx)
        
        # Generate tremor for this subject
        X_acc = np.zeros((N, 3))
        X_gyro = np.zeros((N, 3))
        X_mag = np.zeros((N, 3))
        
        acc_rms = pk_config.get_tremor_rms(subj_idx, "acc", "arm", 2, jitter_std=0.0)
        gyro_rms = pk_config.get_tremor_rms(subj_idx, "gyro", "arm", 2, jitter_std=0.0)
        freq_hz = pk_config.get_tremor_frequency(subj_idx)
        
        X_acc_t, _, _, _ = Tremor.simulate_and_add_tremor_imu(
            X_acc, X_gyro, X_mag,
            fs=fs,
            freq_hz=freq_hz,
            acc_rms=acc_rms,
            gyro_rms=gyro_rms,
            mag_rms=0.0,
            seed=subj_idx
        )
        
        # FFT analysis
        tremor_signal = X_acc_t[:, 0]
        freqs = np.fft.rfftfreq(N, 1/fs)
        fft_mag = np.abs(np.fft.rfft(tremor_signal))
        
        # Find peak frequency
        peak_idx = np.argmax(fft_mag)
        measured_freq = freqs[peak_idx]
        
        # Check if close (within 0.5 Hz tolerance due to stochastic nature and discretization)
        match = "✓" if abs(measured_freq - expected_freq) < 0.5 else "✗"
        if abs(measured_freq - expected_freq) >= 0.5:
            all_correct = False
        
        print(f"  {subj_idx:2d}    |   {expected_freq:.1f} Hz     |   {measured_freq:.2f} Hz   | {match}")
    
    print("\n" + "=" * 80)
    if all_correct:
        print("✓ FFT VERIFICATION PASSED!")
    else:
        print("✗ FFT VERIFICATION: Some peaks deviate from expected")
    print("=" * 80)
    
    return all_correct


if __name__ == "__main__":
    print("\n")
    test1_passed = test_tremor_cache_has_correct_frequencies()
    test2_passed = test_frequency_verification_with_fft()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Cache frequencies): {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (FFT verification): {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    
    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED - Frequency control is working correctly!")
    else:
        print("\n✗ SOME TESTS FAILED - Please review the output above")
    print("=" * 80 + "\n")
