#!/usr/bin/env python3
"""
Test script to demonstrate gyroscope RMS calculation from accelerometer RMS
"""
import sys
sys.path.insert(0, '/Volumes/NO NAME/Master Lina/Code')
from Noise_simulation import tremor_parkinson_config as pk

print('=' * 70)
print('GYROSCOPE RMS CALCULATION FROM ACCELEROMETER RMS')
print('=' * 70)
print()

# Test all subjects
for subj_id in [1, 5, 7, 10]:
    severity = pk.get_subject_severity(subj_id)
    acc_baseline = pk.A_SUBJECT[subj_id]
    k_g = pk.choose_kg_from_rms_acc(acc_baseline)
    gyro_baseline = k_g * acc_baseline
    
    print(f'Subject {subj_id} ({severity}):')
    print(f'  Acc baseline:  {acc_baseline:.2f} m/s²')
    print(f'  k_g factor:    {k_g:.1f}')
    print(f'  Gyro baseline: {gyro_baseline:.2f} deg/s')
    print()

print('=' * 70)
print('Example: Subject 5 during different activities (arm sensor)')
print('=' * 70)
print()

for act, name in [(2, 'Sitting'), (4, 'Walking'), (11, 'Running')]:
    rms_acc = pk.get_tremor_rms(5, 'acc', 'arm', act, jitter_std=0.0)
    rms_gyro = pk.get_tremor_rms(5, 'gyro', 'arm', act, jitter_std=0.0)
    k_g = pk.choose_kg_from_rms_acc(rms_acc)
    ratio = rms_gyro / rms_acc if rms_acc > 0 else 0
    print(f'{name:12s}: Acc={rms_acc:.3f} m/s², Gyro={rms_gyro:.3f} deg/s, k_g={k_g:.1f}')

print()
print('=' * 70)
print('k_g mapping validation')
print('=' * 70)
print()
test_values = [0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 3.5]
for val in test_values:
    k_g = pk.choose_kg_from_rms_acc(val)
    print(f'RMS_acc = {val:.2f} m/s² -> k_g = {k_g:.1f} -> RMS_gyro = {k_g * val:.2f} deg/s')
