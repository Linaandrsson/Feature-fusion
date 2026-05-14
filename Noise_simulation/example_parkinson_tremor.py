"""
Example: Using Parkinson Tremor Model
======================================

This script demonstrates how to use the new Parkinson-aware tremor model
to generate realistic, patient-specific tremor for HAR datasets.
"""

import numpy as np
from tremor_simulation.tremor_parkinson_config import get_tremor_rms, print_tremor_summary, A_SUBJECT

def example_1_basic_rms_calculation():
    """Example 1: Calculate RMS for specific scenarios"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic RMS Calculation")
    print("="*80)
    
    rng = np.random.default_rng(42)
    
    scenarios = [
        (1, "acc", "arm", 2, "Mild patient, sitting, arm sensor"),
        (5, "acc", "arm", 2, "Moderate patient, sitting, arm sensor"),
        (10, "acc", "arm", 2, "Severe patient, sitting, arm sensor"),
        (5, "acc", "arm", 11, "Moderate patient, running, arm sensor"),
        (5, "acc", "ankle", 2, "Moderate patient, sitting, ankle sensor"),
    ]
    
    for subj, sensor, body_part, activity, description in scenarios:
        rms = get_tremor_rms(subj, sensor, body_part, activity, rng=rng)
        print(f"{description:50s} -> RMS = {rms:.3f}")


def example_2_subject_comparison():
    """Example 2: Compare tremor across different severity levels"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Subject Severity Comparison")
    print("="*80)
    
    # Compare mild, moderate, and severe patients
    for subj in [1, 5, 10]:
        print_tremor_summary(subj)


def example_3_activity_modulation():
    """Example 3: Show how tremor changes with activity"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Activity Modulation (Subject 5, Acc on Arm)")
    print("="*80)
    
    rng = np.random.default_rng(123)
    
    activities = [
        (2, "Sitting (resting)"),
        (1, "Standing (resting)"),
        (4, "Walking"),
        (9, "Cycling"),
        (10, "Jogging"),
        (11, "Running"),
    ]
    
    print(f"{'Activity':<25s} {'Beta':>6s} {'RMS':>8s}")
    print("-" * 42)
    
    from tremor_parkinson_config import BETA_ACTIVITY
    
    for act_label, act_name in activities:
        beta = BETA_ACTIVITY.get(act_label, 1.0)
        rms = get_tremor_rms(5, "acc", "arm", act_label, jitter_std=0.0, rng=rng)
        print(f"{act_name:<25s} {beta:>6.2f} {rms:>8.3f}")


def example_4_body_part_comparison():
    """Example 4: Compare tremor across body parts"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Body Part Comparison (Subject 5, Sitting)")
    print("="*80)
    
    rng = np.random.default_rng(456)
    
    body_parts = ["arm", "ankle", "chest"]
    sensors = ["acc", "gyro", "mag"]
    
    print(f"{'Body Part':<12s} {'Sensor':<8s} {'RMS':>8s}")
    print("-" * 30)
    
    for bp in body_parts:
        for sensor in sensors:
            rms = get_tremor_rms(5, sensor, bp, activity=2, jitter_std=0.0, rng=rng)
            print(f"{bp:<12s} {sensor:<8s} {rms:>8.3f}")


def example_5_jitter_demonstration():
    """Example 5: Show effect of jitter on repeated windows"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Jitter Effect (10 consecutive windows, Subject 5)")
    print("="*80)
    
    rng = np.random.default_rng(789)
    
    print("Without jitter:")
    rms_values = [get_tremor_rms(5, "acc", "arm", 2, jitter_std=0.0, rng=rng) for _ in range(10)]
    print(f"  Mean: {np.mean(rms_values):.3f}, Std: {np.std(rms_values):.3f}")
    print(f"  Values: {[f'{v:.3f}' for v in rms_values]}")
    
    print("\nWith jitter (15% std):")
    rng2 = np.random.default_rng(789)
    rms_values_jitter = [get_tremor_rms(5, "acc", "arm", 2, jitter_std=0.15, rng=rng2) for _ in range(10)]
    print(f"  Mean: {np.mean(rms_values_jitter):.3f}, Std: {np.std(rms_values_jitter):.3f}")
    print(f"  Values: {[f'{v:.3f}' for v in rms_values_jitter]}")
    
    variability = np.std(rms_values_jitter) / np.mean(rms_values_jitter) * 100
    print(f"\nCoefficient of variation: {variability:.1f}%")


def example_6_all_subjects_overview():
    """Example 6: Overview of all configured subjects"""
    print("\n" + "="*80)
    print("EXAMPLE 6: All Subjects Overview")
    print("="*80)
    
    from tremor_parkinson_config import get_subject_severity
    
    print(f"{'Subject':<10s} {'Severity':<12s} {'Acc RMS':<10s} {'Gyro RMS':<10s} {'Mag RMS':<10s}")
    print("-" * 60)
    
    for subj_id in sorted(A_SUBJECT.keys()):
        severity = get_subject_severity(subj_id)
        params = A_SUBJECT[subj_id]
        print(f"{subj_id:<10d} {severity:<12s} {params['acc']:<10.2f} "
              f"{params['gyro']:<10.2f} {params['mag']:<10.2f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PARKINSON TREMOR MODEL - USAGE EXAMPLES")
    print("="*80)
    
    # Run all examples
    example_1_basic_rms_calculation()
    example_2_subject_comparison()
    example_3_activity_modulation()
    example_4_body_part_comparison()
    example_5_jitter_demonstration()
    example_6_all_subjects_overview()
    
    print("\n" + "="*80)
    print("HOW TO USE IN DATA GENERATOR:")
    print("="*80)
    print("""
1. Import the function:
   from Noise_simulation.Tremor import precompute_tremor_cache_with_parkinson_model

2. Pre-compute tremor cache:
   tremor_cache = precompute_tremor_cache_with_parkinson_model(
       window_specs=window_specs,      # [(subj_id, label, start, end), ...]
       sensor_column_mapping=sensor_cols,
       data_loader_func=load_subject_data,
       fs=50.0,
       scenario_seed=42,
       use_jitter=True
   )

3. Apply to windows:
   for w_idx, body_part in enumerate(['ankle', 'arm', 'chest']):
       noise = tremor_cache[(w_idx, body_part)]
       acc_corrupted = acc_clean + noise['acc_noise']
       gyro_corrupted = gyro_clean + noise['gyro_noise']
       mag_corrupted = mag_clean + noise['mag_noise']

4. Customize subjects:
   Edit tremor_parkinson_config.py:
   - Modify A_SUBJECT to change patient severity
   - Adjust C_BODY_PART for body part scaling
   - Tune BETA_ACTIVITY for activity modulation
    """)
    
    print("="*80)
