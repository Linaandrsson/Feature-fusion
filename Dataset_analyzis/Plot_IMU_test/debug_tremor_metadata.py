"""
Debug tremor metadata
"""
import numpy as np
from pathlib import Path

# Load severe variant for Acc_chest
SCRIPT_DIR = Path(__file__).parent.resolve()
data_path = SCRIPT_DIR.parent.parent / 'data' / 'Tremor_datagenerator_files' / 's2_w2_fs50_tremor_mod_severe' / 'Acc_chest.npz'
data = np.load(data_path)

# Check activity 0 (Standing still)
activity_0_mask = data['y'] == 0
activity_0_indices = np.where(activity_0_mask)[0]

print('Activity 0 (Standing still) - SEVERE variant')
print('='*70)
print(f'Total windows for activity 0: {len(activity_0_indices)}')
print()

# Check first 10 windows
print('First 10 windows tremor metadata:')
print(f"{'Win':<5} {'Global':<8} {'Subject':<8} {'Tremor Freq':<12} {'Tremor Score':<12}")
print('-'*70)
for i, idx in enumerate(activity_0_indices[:10]):
    print(f"{i:<5} {idx:<8} {data['subject_id'][idx]:<8} {data['tremor_freq'][idx]:<12.2f} {data['tremor_score'][idx]:<12.4f}")

print()
print('Statistics for all activity 0 windows:')
print(f"Tremor freq - min: {data['tremor_freq'][activity_0_indices].min():.2f}, max: {data['tremor_freq'][activity_0_indices].max():.2f}, mean: {data['tremor_freq'][activity_0_indices].mean():.2f}")
print(f"Tremor score - min: {data['tremor_score'][activity_0_indices].min():.4f}, max: {data['tremor_score'][activity_0_indices].max():.4f}, mean: {data['tremor_score'][activity_0_indices].mean():.4f}")

# Also check if all values are 0
all_zero_freq = np.all(data['tremor_freq'][activity_0_indices] == 0)
all_zero_score = np.all(data['tremor_score'][activity_0_indices] == 0)
print(f"\nAll tremor_freq are 0? {all_zero_freq}")
print(f"All tremor_score are 0? {all_zero_score}")

# Check a different activity
print('\n' + '='*70)
print('Activity 3 (Walking) - SEVERE variant')
print('='*70)
activity_3_mask = data['y'] == 3
activity_3_indices = np.where(activity_3_mask)[0]
print(f'Total windows for activity 3: {len(activity_3_indices)}')
print()
print('First 5 windows tremor metadata:')
print(f"{'Win':<5} {'Global':<8} {'Subject':<8} {'Tremor Freq':<12} {'Tremor Score':<12}")
print('-'*70)
for i, idx in enumerate(activity_3_indices[:5]):
    print(f"{i:<5} {idx:<8} {data['subject_id'][idx]:<8} {data['tremor_freq'][idx]:<12.2f} {data['tremor_score'][idx]:<12.4f}")
