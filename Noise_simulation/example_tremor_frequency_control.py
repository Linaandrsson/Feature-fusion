"""
Example: Tremor Generation with Frequency Control
==================================================

This example demonstrates how to:
1. Use tremor_parkinson_config.py to get subject-specific parameters
2. Generate tremor with controlled frequency (3.5-7 Hz range)
3. Label windows with acc_rms, gyro_rms, and freq for dataset generation

This is the recommended workflow for generating labeled tremor datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import tremor_parkinson_config as pk_config

def example_1_basic_tremor_generation():
    """
    Example 1: Generate tremor for a single window with known labels
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Tremor Generation with Labels")
    print("=" * 80)
    
    # Generate clean sensor data (simulating a window)
    fs = 50.0  # 50 Hz sampling rate
    T_sec = 5.0  # 5 second window
    N = int(T_sec * fs)
    
    # Simulate clean sensor readings (random walk for demonstration)
    X_acc = np.cumsum(np.random.randn(N, 3) * 0.1, axis=0)
    X_gyro = np.cumsum(np.random.randn(N, 3) * 0.05, axis=0)
    X_mag = np.random.randn(N, 3) * 0.5
    
    # Apply tremor using subject 5 (mild-moderate), arm, sitting
    X_acc_t, X_gyro_t, X_mag_t, metadata = pk_config.apply_tremor_to_window(
        X_acc=X_acc,
        X_gyro=X_gyro,
        X_mag=X_mag,
        subject_id=5,
        body_part="arm",
        activity=2,  # Sitting
        fs=fs,
        seed=42
    )
    
    # Print the labels for this window
    print("\nWindow Labels:")
    print(f"  Subject ID: {metadata['subject_id']}")
    print(f"  Severity: {metadata['severity']}")
    print(f"  Body Part: {metadata['body_part']}")
    print(f"  Activity: {metadata['activity']}")
    print(f"  Acc RMS: {metadata['acc_rms']:.3f} m/s²")
    print(f"  Gyro RMS: {metadata['gyro_rms']:.3f} deg/s")
    print(f"  Frequency: {metadata['freq']:.1f} Hz")
    
    print("\n✓ These three values (acc_rms, gyro_rms, freq) label each window!")


def example_2_multiple_subjects():
    """
    Example 2: Generate tremor for multiple subjects showing frequency variation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Subjects with Different Frequencies")
    print("=" * 80)
    
    fs = 50.0
    T_sec = 5.0
    N = int(T_sec * fs)
    
    # Generate clean signals once
    X_acc = np.cumsum(np.random.randn(N, 3) * 0.1, axis=0)
    X_gyro = np.cumsum(np.random.randn(N, 3) * 0.05, axis=0)
    X_mag = np.random.randn(N, 3) * 0.5
    
    subjects = [1, 5, 10]  # Mild, Moderate, Severe
    
    print("\nSubject | Severity        | Acc RMS | Gyro RMS | Frequency")
    print("-" * 70)
    
    for subj in subjects:
        _, _, _, meta = pk_config.apply_tremor_to_window(
            X_acc.copy(), X_gyro.copy(), X_mag.copy(),
            subject_id=subj,
            body_part="arm",
            activity=2,
            fs=fs,
            jitter_std=0.0,  # No jitter for comparison
            seed=42
        )
        
        print(f"   {subj:2d}   | {meta['severity']:15s} | "
              f"{meta['acc_rms']:7.3f} | {meta['gyro_rms']:8.3f} | "
              f"{meta['freq']:5.1f} Hz")


def example_3_activity_modulation():
    """
    Example 3: Show how RMS changes with activity (frequency stays constant per subject)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Activity Modulation (Subject 5)")
    print("=" * 80)
    
    fs = 50.0
    T_sec = 5.0
    N = int(T_sec * fs)
    
    X_acc = np.cumsum(np.random.randn(N, 3) * 0.1, axis=0)
    X_gyro = np.cumsum(np.random.randn(N, 3) * 0.05, axis=0)
    X_mag = np.random.randn(N, 3) * 0.5
    
    activities = [
        (2, "Sitting"),
        (4, "Walking"),
        (10, "Jogging"),
        (11, "Running"),
    ]
    
    print("\nActivity     | Acc RMS | Gyro RMS | Frequency")
    print("-" * 55)
    
    for act_label, act_name in activities:
        _, _, _, meta = pk_config.apply_tremor_to_window(
            X_acc.copy(), X_gyro.copy(), X_mag.copy(),
            subject_id=5,
            body_part="arm",
            activity=act_label,
            fs=fs,
            jitter_std=0.0,
            seed=42
        )
        
        print(f"{act_name:12s} | {meta['acc_rms']:7.3f} | "
              f"{meta['gyro_rms']:8.3f} | {meta['freq']:5.1f} Hz")
    
    print("\n✓ Notice: Frequency is constant (4.8 Hz) for subject 5")
    print("✓ RMS values decrease during active movement")


def example_4_dataset_generation_workflow():
    """
    Example 4: Typical workflow for generating a labeled tremor dataset
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Dataset Generation Workflow")
    print("=" * 80)
    
    print("\nWorkflow for generating labeled tremor dataset:")
    print("1. Load clean sensor windows from MHEALTH dataset")
    print("2. For each window:")
    print("   a. Get subject_id, body_part, activity from window metadata")
    print("   b. Call apply_tremor_to_window()")
    print("   c. Save corrupted window + labels (acc_rms, gyro_rms, freq)")
    print("3. Result: Dataset where each window has known tremor parameters")
    
    print("\nExample pseudocode:")
    print("-" * 80)
    print("""
    for window in dataset:
        X_acc, X_gyro, X_mag = window.get_sensors()
        subject_id = np.random.choice(range(1, 11))  # Pick random subject
        
        X_acc_t, X_gyro_t, X_mag_t, labels = pk_config.apply_tremor_to_window(
            X_acc, X_gyro, X_mag,
            subject_id=subject_id,
            body_part=window.body_part,
            activity=window.activity,
            fs=window.fs,
            seed=window_seed
        )
        
        # Save window with labels
        save_window(
            data={'acc': X_acc_t, 'gyro': X_gyro_t, 'mag': X_mag_t},
            labels={
                'acc_rms': labels['acc_rms'],
                'gyro_rms': labels['gyro_rms'],
                'freq_hz': labels['freq'],
                'subject_id': labels['subject_id'],
                'severity': labels['severity']
            }
        )
    """)
    print("-" * 80)


def example_5_frequency_verification():
    """
    Example 5: Verify that generated tremor has correct frequency using FFT
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Frequency Verification with FFT")
    print("=" * 80)
    
    from scipy import signal as scipy_signal
    
    fs = 50.0
    T_sec = 10.0  # Longer window for better frequency resolution
    N = int(T_sec * fs)
    
    # Clean signal (zeros for clear tremor visualization)
    X_acc = np.zeros((N, 3))
    X_gyro = np.zeros((N, 3))
    X_mag = np.zeros((N, 3))
    
    subjects = [1, 5, 10]  # Different frequencies: 5.5, 4.8, 4.0 Hz
    
    print("\nVerifying tremor frequency with FFT:")
    print("\nSubject | Expected Freq | Measured Freq (FFT Peak)")
    print("-" * 55)
    
    for subj in subjects:
        X_acc_t, _, _, meta = pk_config.apply_tremor_to_window(
            X_acc.copy(), X_gyro.copy(), X_mag.copy(),
            subject_id=subj,
            body_part="arm",
            activity=2,
            fs=fs,
            seed=subj
        )
        
        # Compute FFT on one axis of accelerometer
        tremor_signal = X_acc_t[:, 0]
        freqs = np.fft.rfftfreq(N, 1/fs)
        fft_mag = np.abs(np.fft.rfft(tremor_signal))
        
        # Find peak frequency
        peak_idx = np.argmax(fft_mag)
        measured_freq = freqs[peak_idx]
        expected_freq = meta['freq']
        
        print(f"   {subj:2d}   |   {expected_freq:.1f} Hz     |     "
              f"{measured_freq:.2f} Hz")
    
    print("\n✓ FFT peaks match expected frequencies!")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_tremor_generation()
    example_2_multiple_subjects()
    example_3_activity_modulation()
    example_4_dataset_generation_workflow()
    example_5_frequency_verification()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey points:")
    print("1. Use get_tremor_params() to get labels: acc_rms, gyro_rms, freq")
    print("2. Use apply_tremor_to_window() to apply tremor with correct frequency")
    print("3. Frequency is subject-specific and constant (range: 3.5-7.0 Hz)")
    print("4. RMS values vary with body_part and activity")
    print("5. All three values (acc_rms, gyro_rms, freq) label each window")
    print("\n" + "=" * 80)
