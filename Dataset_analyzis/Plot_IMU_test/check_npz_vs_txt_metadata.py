"""
Check TXT vs NPZ metadata discrepancy
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Volumes/NO NAME/Master Lina/Code/data/Tremor_datagenerator_files/s2_w2_fs50_tremor_mod_severe")

# Load NPZ
npz_data = np.load(DATA_DIR / "Acc_arm.npz")

# Load TXT  
txt_data = np.loadtxt(DATA_DIR / "Acc_arm.txt", delimiter=',')

print("=" * 80)
print("NPZ vs TXT METADATA COMPARISON - SEVERE VARIANT")
print("=" * 80)

print(f"\nFile sizes:")
print(f"  NPZ windows: {len(npz_data['X'])}")
print(f"  TXT rows: {txt_data.shape[0]}")
print(f"  TXT columns: {txt_data.shape[1]} (expected: 300 sensor + 7 metadata = 307)")

# NPZ metadata
print(f"\n{'='*80}")
print("NPZ METADATA (from arrays):")
print(f"{'='*80}")
print(f"First 5 windows:")
print(f"{'Win':<5} {'Activity':<10} {'SubjID':<8} {'WinIdx':<8} {'Freq(Hz)':<10} {'AccRMS':<10} {'GyroRMS':<10} {'Score':<6}")
print("-" * 80)
for i in range(5):
    print(f"{i:<5} {npz_data['y'][i]:<10} {npz_data['subject_id'][i]:<8} {npz_data['base_window_idx'][i]:<8} "
          f"{npz_data['tremor_freq'][i]:<10.2f} {npz_data['tremor_acc_rms'][i]:<10.4f} "
          f"{npz_data['tremor_gyro_rms'][i]:<10.2f} {npz_data['tremor_score'][i]:<6}")

print(f"\nNPZ Statistics:")
print(f"  Tremor freq: min={npz_data['tremor_freq'].min():.2f}, max={npz_data['tremor_freq'].max():.2f}, mean={npz_data['tremor_freq'].mean():.2f} Hz")
print(f"  Acc RMS: min={npz_data['tremor_acc_rms'].min():.4f}, max={npz_data['tremor_acc_rms'].max():.4f}, mean={npz_data['tremor_acc_rms'].mean():.4f}")
print(f"  Gyro RMS: min={npz_data['tremor_gyro_rms'].min():.2f}, max={npz_data['tremor_gyro_rms'].max():.2f}, mean={npz_data['tremor_gyro_rms'].mean():.2f}")
print(f"  Score: unique values = {np.unique(npz_data['tremor_score'])}")

# TXT metadata (last 7 columns)
print(f"\n{'='*80}")
print("TXT METADATA (from last 7 columns):")
print(f"{'='*80}")
print(f"Column mapping for TXT:")
print(f"  Columns 1-300: Sensor data")
print(f"  Column 301: Activity (1-indexed)")
print(f"  Column 302: Subject ID")
print(f"  Column 303: Window index")
print(f"  Column 304: Tremor frequency")
print(f"  Column 305: Tremor acc RMS")
print(f"  Column 306: Tremor gyro RMS")
print(f"  Column 307: Tremor score")

print(f"\nFirst 5 windows:")
print(f"{'Win':<5} {'Activity':<10} {'SubjID':<8} {'WinIdx':<8} {'Freq(Hz)':<10} {'AccRMS':<10} {'GyroRMS':<10} {'Score':<6}")
print("-" * 80)
for i in range(5):
    row = txt_data[i, -7:]  # Last 7 columns
    print(f"{i:<5} {int(row[0]):<10} {int(row[1]):<8} {int(row[2]):<8} "
          f"{row[3]:<10.2f} {row[4]:<10.4f} {row[5]:<10.2f} {int(row[6]):<6}")

print(f"\nTXT Statistics (from last columns):")
print(f"  Tremor freq: min={txt_data[:, -4].min():.2f}, max={txt_data[:, -4].max():.2f}, mean={txt_data[:, -4].mean():.2f} Hz")
print(f"  Acc RMS: min={txt_data[:, -3].min():.4f}, max={txt_data[:, -3].max():.4f}, mean={txt_data[:, -3].mean():.4f}")
print(f"  Gyro RMS: min={txt_data[:, -2].min():.2f}, max={txt_data[:, -2].max():.2f}, mean={txt_data[:, -2].mean():.2f}")
print(f"  Score: unique values = {np.unique(txt_data[:, -1].astype(int))}")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
if npz_data['tremor_freq'].max() == 0:
    print("❌ NPZ files have INCORRECT metadata (all zeros)")
else:
    print("✓ NPZ files have correct metadata")

if txt_data[:, -4].max() > 0:
    print("✓ TXT files have CORRECT metadata (non-zero tremor values)")
else:
    print("❌ TXT files have incorrect metadata")

print(f"\n💡 SOLUTION: Load tremor metadata from TXT files instead of NPZ!")
print(f"{'='*80}\n")
