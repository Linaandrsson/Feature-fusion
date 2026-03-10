"""
Compare clean vs severe signal data to see if tremor is actually applied
"""
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_BASE = SCRIPT_DIR.parent.parent / 'data' / 'Tremor_datagenerator_files'

# Load clean and severe variants for Acc_chest
clean_path = DATA_BASE / 's2_w2_fs50_tremor_clean' / 'Acc_chest.npz'
severe_path = DATA_BASE / 's2_w2_fs50_tremor_mod_severe' / 'Acc_chest.npz'

clean_data = np.load(clean_path)
severe_data = np.load(severe_path)

print('='*70)
print('COMPARING CLEAN vs SEVERE - Accelerometer Chest')
print('='*70)

# Check activity 0, window 0
activity = 0
window_idx = 0

# Find first window for activity 0
clean_mask = clean_data['y'] == activity
severe_mask = severe_data['y'] == activity

clean_indices = np.where(clean_mask)[0]
severe_indices = np.where(severe_mask)[0]

clean_idx = clean_indices[window_idx]
severe_idx = severe_indices[window_idx]

# Get signals
clean_signal = clean_data['X'][clean_idx]  # Shape: (C, L)
severe_signal = severe_data['X'][severe_idx]  # Shape: (C, L)

print(f'\nActivity: {activity} (Standing still)')
print(f'Window index: {window_idx}')
print(f'\nClean global index: {clean_idx}')
print(f'Severe global index: {severe_idx}')
print(f'\nSignal shape: {clean_signal.shape} (channels, time_steps)')

# Check if signals are identical
are_identical = np.allclose(clean_signal, severe_signal, rtol=1e-5, atol=1e-5)
max_diff = np.max(np.abs(clean_signal - severe_signal))
mean_diff = np.mean(np.abs(clean_signal - severe_signal))

print(f'\n{"="*70}')
print(f'SIGNAL COMPARISON:')
print(f'{"="*70}')
print(f'Signals identical? {are_identical}')
print(f'Max difference: {max_diff:.6f}')
print(f'Mean absolute difference: {mean_diff:.6f}')

# Calculate RMS for each axis
print(f'\n{"="*70}')
print(f'RMS VALUES:')
print(f'{"="*70}')
print(f'{"Axis":<10} {"Clean RMS":<15} {"Severe RMS":<15} {"Difference":<15}')
print(f'{"-"*70}')
for axis in range(3):
    clean_rms = np.sqrt(np.mean(clean_signal[axis]**2))
    severe_rms = np.sqrt(np.mean(severe_signal[axis]**2))
    diff = severe_rms - clean_rms
    print(f'Axis {axis:<4} {clean_rms:<15.6f} {severe_rms:<15.6f} {diff:<15.6f}')

# Calculate magnitude
clean_mag = np.sqrt(np.mean(clean_signal**2))
severe_mag = np.sqrt(np.mean(severe_signal**2))
print(f'{"Magnitude":<10} {clean_mag:<15.6f} {severe_mag:<15.6f} {severe_mag - clean_mag:<15.6f}')

# Show first few samples
print(f'\n{"="*70}')
print(f'FIRST 10 TIME STEPS (Axis 0):')
print(f'{"="*70}')
print(f'{"Time":<6} {"Clean":<15} {"Severe":<15} {"Diff":<15}')
print(f'{"-"*70}')
for t in range(10):
    print(f'{t:<6} {clean_signal[0, t]:<15.6f} {severe_signal[0, t]:<15.6f} {severe_signal[0, t] - clean_signal[0, t]:<15.6f}')
