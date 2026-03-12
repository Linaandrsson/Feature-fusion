"""
Plot Tremor-Branch vs Clean Signal Comparison
==============================================

Visualize tremor-branch sensor signals (non-normalized, for tremor estimation)
compared to clean baseline signals.

Only plots:
- Acc_arm_tremorbranch
- Gyro_arm_tremorbranch

These signals preserve amplitude information (no z-score normalization)
for tremor severity estimation, unlike the HAR branch.

Usage:
    python plot_tremorbranch_vs_clean.py
    
    Then follow interactive prompts to select:
    - Tremor dataset variant (e.g., mild_mod, mod_severe)
    - Activity (by index or name)
    - Window index
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# Configuration
# ============================================================

# Activity labels (0-indexed)
ACTIVITY_NAMES = {
    0: "Standing still",
    1: "Sitting and relaxing",
    2: "Lying down",
    3: "Walking",
    4: "Climbing stairs",
    5: "Waist bends forward",
    6: "Frontal elevation of arms",
    7: "Knees bending",
    8: "Cycling",
    9: "Jogging",
    10: "Running",
    11: "Jump front & back"
}

# Tremor-branch sensors only
TREMOR_BRANCH_SENSORS = {
    "Acc_arm_tremorbranch": {"axes": 3, "unit": "m/s²", "label": "Acc Arm (Tremor Branch)"},
    "Gyro_arm_tremorbranch": {"axes": 3, "unit": "deg/s", "label": "Gyro Arm (Tremor Branch)"}
}

AXIS_LABELS = ['X', 'Y', 'Z']

# Available tremor variants (excluding clean since we always compare against clean)
TREMOR_VARIANTS = ["mild_mod", "mod_severe"]

# Base path
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_BASE = SCRIPT_DIR.parent.parent / "data" / "Tremor_datagenerator_files"

# ============================================================
# Helper Functions
# ============================================================

def load_sensor_data(variant_name: str, sensor_name: str, fs: int = 30):
    """
    Load sensor data from .npz file.
    
    Args:
        variant_name: Dataset variant (e.g., "s2_w2_fs30_tremor_clean")
        sensor_name: Sensor name (e.g., "Acc_arm_tremorbranch")
        fs: Sampling frequency (default 30 Hz)
        
    Returns:
        Dictionary with data arrays (X, y, subject_id, tremor labels)
    """
    data_dir = DATA_BASE / variant_name
    sensor_file = data_dir / f"{sensor_name}.npz"
    
    if not sensor_file.exists():
        raise FileNotFoundError(f"Sensor file not found: {sensor_file}")
    
    data = np.load(sensor_file)
    return {
        'X': data['X'],              # (N, C, L)
        'y': data['y'],               # (N,) activity labels (0-11)
        'subject_id': data['subject_id'],  # (N,) subject IDs
        'tremor_freq': data['tremor_freq'],
        'tremor_acc_rms': data['tremor_acc_rms'],
        'tremor_gyro_rms': data['tremor_gyro_rms'],
        'tremor_score': data['tremor_score']
    }


def find_matching_windows(data_clean, data_tremor, activity_idx, window_idx=0, severity=None):
    """
    Find matching windows from both datasets for the specified activity.
    
    Args:
        data_clean: Clean dataset dict
        data_tremor: Tremor dataset dict
        activity_idx: Activity index (0-11)
        window_idx: Which window to select (default: 0 for first match)
        severity: Optional tremor score to filter by (e.g., 4 for severe tremor)
        
    Returns:
        Tuple of (signal_clean, signal_tremor, metadata_clean, metadata_tremor) or None if not found
    """
    # Find all windows matching the activity
    mask_clean = data_clean['y'] == activity_idx
    mask_tremor = data_tremor['y'] == activity_idx
    
    # Add severity filtering if specified (only for tremor dataset)
    if severity is not None:
        severity_mask = data_tremor['tremor_score'] == severity
        mask_tremor = mask_tremor & severity_mask
    
    if not mask_clean.any():
        print(f"❌ No samples found for activity {activity_idx} in clean dataset")
        return None
    if not mask_tremor.any():
        print(f"❌ No samples found for activity {activity_idx} in tremor dataset")
        return None
    
    # Get matching indices
    indices_clean = np.where(mask_clean)[0]
    indices_tremor = np.where(mask_tremor)[0]
    
    # Select window
    if window_idx >= len(indices_clean):
        print(f"❌ Window index {window_idx} out of range for clean dataset (max: {len(indices_clean)-1})")
        return None
    if window_idx >= len(indices_tremor):
        print(f"❌ Window index {window_idx} out of range for tremor dataset (max: {len(indices_tremor)-1})")
        return None
    
    idx_clean = indices_clean[window_idx]
    idx_tremor = indices_tremor[window_idx]
    
    # Extract signals (shape: C, L)
    signal_clean = data_clean['X'][idx_clean]
    signal_tremor = data_tremor['X'][idx_tremor]
    
    # Extract metadata
    metadata_clean = {
        'subject_id': data_clean['subject_id'][idx_clean],
        'tremor_freq': data_clean['tremor_freq'][idx_clean],
        'tremor_acc_rms': data_clean['tremor_acc_rms'][idx_clean],
        'tremor_gyro_rms': data_clean['tremor_gyro_rms'][idx_clean],
        'tremor_score': data_clean['tremor_score'][idx_clean]
    }
    
    metadata_tremor = {
        'subject_id': data_tremor['subject_id'][idx_tremor],
        'tremor_freq': data_tremor['tremor_freq'][idx_tremor],
        'tremor_acc_rms': data_tremor['tremor_acc_rms'][idx_tremor],
        'tremor_gyro_rms': data_tremor['tremor_gyro_rms'][idx_tremor],
        'tremor_score': data_tremor['tremor_score'][idx_tremor]
    }
    
    return signal_clean, signal_tremor, metadata_clean, metadata_tremor


def plot_comparison(sensor_name, signal_clean, signal_tremor, metadata_clean, metadata_tremor, 
                   activity_idx, tremor_variant_name, fs=30):
    """
    Plot side-by-side comparison of clean vs tremor signals for one sensor.
    
    Args:
        sensor_name: Name of the sensor
        signal_clean: Signal from clean dataset (C, L)
        signal_tremor: Signal from tremor dataset (C, L)
        metadata_clean: Metadata dict for clean signal
        metadata_tremor: Metadata dict for tremor signal
        activity_idx: Activity index
        tremor_variant_name: Name of tremor variant (e.g., "mild_mod")
        fs: Sampling frequency (Hz)
    """
    sensor_info = TREMOR_BRANCH_SENSORS[sensor_name]
    n_axes = sensor_info['axes']
    time = np.arange(signal_clean.shape[1]) / fs
    
    # Create figure with subplots for each axis
    fig, axes = plt.subplots(n_axes, 2, figsize=(14, 3*n_axes), sharex=True)
    if n_axes == 1:
        axes = axes.reshape(1, -1)
    
    # Activity name
    activity_name = ACTIVITY_NAMES[activity_idx]
    
    # Plot each axis
    for ax_idx in range(n_axes):
        # Left column: Clean
        axes[ax_idx, 0].plot(time, signal_clean[ax_idx], 'b-', linewidth=0.8)
        axes[ax_idx, 0].set_ylabel(f'{AXIS_LABELS[ax_idx]}\n({sensor_info["unit"]})', fontsize=10)
        axes[ax_idx, 0].grid(True, alpha=0.3)
        
        # Right column: Tremor
        axes[ax_idx, 1].plot(time, signal_tremor[ax_idx], 'r-', linewidth=0.8)
        axes[ax_idx, 1].set_ylabel(f'{AXIS_LABELS[ax_idx]}\n({sensor_info["unit"]})', fontsize=10)
        axes[ax_idx, 1].grid(True, alpha=0.3)
    
    # X-axis labels on bottom row
    axes[-1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=11)
    
    # Column titles with metadata
    title_clean = 'CLEAN (No Tremor)\n'
    title_clean += f'Subj: {metadata_clean["subject_id"]}, Score: {int(metadata_clean["tremor_score"])}\n'
    title_clean += f'RMS: {metadata_clean["tremor_acc_rms"]:.3f} m/s², Freq: {metadata_clean["tremor_freq"]:.1f} Hz'
    axes[0, 0].set_title(title_clean, fontsize=11, fontweight='bold', color='blue')
    
    title_tremor = f'{tremor_variant_name.upper()} (Tremor-Branch)\n'
    title_tremor += f'Subj: {metadata_tremor["subject_id"]}, Score: {int(metadata_tremor["tremor_score"])}\n'
    title_tremor += f'RMS: {metadata_tremor["tremor_acc_rms"]:.3f} m/s², Freq: {metadata_tremor["tremor_freq"]:.1f} Hz'
    axes[0, 1].set_title(title_tremor, fontsize=11, fontweight='bold', color='red')
    
    # Main title
    fig.suptitle(
        f'{sensor_info["label"]} (Non-Normalized) — Activity: {activity_name} (ID: {activity_idx})',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def plot_tremorbranch_comparison(tremor_variant, activity_idx, 
                                 window_idx=0, fs=30, save_dir=None, severity=None):
    """
    Plot comparison for tremor-branch sensors (Acc_arm, Gyro_arm).
    
    Args:
        tremor_variant: Tremor dataset variant (e.g., "mild_mod", "mod_severe")
        activity_idx: Activity index (0-11)
        window_idx: Window index to plot
        fs: Sampling frequency (30 or 50 Hz, determines which dataset to load)
        save_dir: Directory to save plots (optional)
        severity: Optional tremor score to filter by (e.g., 4 for severe tremor)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    activity_name = ACTIVITY_NAMES[activity_idx]
    
    # Construct full variant names
    clean_variant = f"s2_w2_fs{fs}_tremor_clean"
    tremor_variant_full = f"s2_w2_fs{fs}_tremor_{tremor_variant}"
    
    print("\n" + "="*80)
    print(f"PLOTTING TREMOR-BRANCH: CLEAN vs {tremor_variant.upper()}")
    print(f"Dataset: {clean_variant}")
    print(f"Activity: {activity_name} (ID: {activity_idx})")
    print(f"Window: {window_idx}")
    if severity is not None:
        print(f"Severity filter: Score = {severity}")
    print(f"Note: These signals are NON-NORMALIZED (preserve amplitude for tremor estimation)")
    print("="*80 + "\n")
    
    figures = []
    
    for sensor_name in TREMOR_BRANCH_SENSORS.keys():
        print(f"Loading {sensor_name}...", end=" ")
        
        try:
            # Load data for both variants
            data_clean = load_sensor_data(clean_variant, sensor_name, fs)
            data_tremor = load_sensor_data(tremor_variant_full, sensor_name, fs)
            
            # Find matching windows
            result = find_matching_windows(data_clean, data_tremor, activity_idx, window_idx, severity)
            if result is None:
                continue
            
            signal_clean, signal_tremor, metadata_clean, metadata_tremor = result
            
            # Plot
            fig = plot_comparison(
                sensor_name, signal_clean, signal_tremor, metadata_clean, metadata_tremor,
                activity_idx, tremor_variant, fs
            )
            figures.append((sensor_name, fig))
            
            # Save if requested
            if save_dir:
                filename = f"{sensor_name}_clean_vs_{tremor_variant}_act{activity_idx}_win{window_idx}.png"
                filepath = save_dir / filename
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"✓ Saved to {filename}")
            else:
                print("✓")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print(f"✓ Generated {len(figures)} tremor-branch comparison plots")
    print("="*80 + "\n")
    
    return figures


# ============================================================
# Interactive CLI
# ============================================================

def select_tremor_variant():
    """Interactive tremor variant selection."""
    print("\nSelect tremor variant to compare against clean:")
    for i, var in enumerate(TREMOR_VARIANTS, 1):
        print(f"  {i}. {var}")
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(TREMOR_VARIANTS)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(TREMOR_VARIANTS):
                return TREMOR_VARIANTS[idx]
            else:
                print("❌ Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input.")
            sys.exit(1)


def select_activity():
    """Interactive activity selection."""
    print("\nSelect activity:")
    print("  Option 1: Enter activity index (0-11)")
    print("  Option 2: Enter activity name (e.g., 'Walking')")
    
    # Display available activities
    print("\nAvailable activities:")
    for idx, name in ACTIVITY_NAMES.items():
        print(f"  {idx}: {name}")
    
    while True:
        try:
            choice = input("\nEnter activity index or name: ").strip()
            
            # Try as index
            if choice.isdigit():
                idx = int(choice)
                if idx in ACTIVITY_NAMES:
                    return idx
                else:
                    print(f"❌ Invalid index. Must be 0-11.")
            else:
                # Try as name (case-insensitive partial match)
                choice_lower = choice.lower()
                matches = [(idx, name) for idx, name in ACTIVITY_NAMES.items() 
                          if choice_lower in name.lower()]
                
                if len(matches) == 1:
                    return matches[0][0]
                elif len(matches) > 1:
                    print(f"❌ Ambiguous name. Matches: {[m[1] for m in matches]}")
                else:
                    print(f"❌ No matching activity found.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input.")
            sys.exit(1)


def select_window_index(tremor_variant, activity_idx, fs=30, severity=None):
    """Interactive window index selection."""
    # Construct full variant names
    clean_variant = f"s2_w2_fs{fs}_tremor_clean"
    tremor_variant_full = f"s2_w2_fs{fs}_tremor_{tremor_variant}"
    
    # Load one sensor to get window counts
    try:
        data_clean = load_sensor_data(clean_variant, "Acc_arm_tremorbranch", fs)
        data_tremor = load_sensor_data(tremor_variant_full, "Acc_arm_tremorbranch", fs)
        
        mask_clean = data_clean['y'] == activity_idx
        mask_tremor = data_tremor['y'] == activity_idx
        
        # Add severity filtering if specified
        if severity is not None:
            severity_mask = data_tremor['tremor_score'] == severity
            mask_tremor = mask_tremor & severity_mask
        
        count_clean = mask_clean.sum()
        count_tremor = mask_tremor.sum()
        max_windows = min(count_clean, count_tremor)
        
        print(f"\nAvailable windows for this activity:")
        print(f"  Clean dataset: {count_clean} windows")
        print(f"  Tremor dataset: {count_tremor} windows")
        print(f"  Max index: {max_windows - 1}")
        
    except Exception as e:
        print(f"⚠️  Could not determine window count: {e}")
        max_windows = 100  # fallback
    
    while True:
        try:
            choice = input(f"\nEnter window index (0-{max_windows-1}, default 0): ").strip()
            if choice == "":
                return 0
            idx = int(choice)
            if 0 <= idx < max_windows:
                return idx
            else:
                print(f"❌ Invalid index. Must be 0-{max_windows-1}.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input.")
            sys.exit(1)


def select_severity():
    """Interactive severity selection."""
    print("\n" + "="*60)
    print("SELECT SEVERITY (OPTIONAL)")
    print("="*60)
    print("Tremor severity scores:")
    print("  0: No tremor")
    print("  1: Mild")
    print("  2: Mild-Moderate")
    print("  3: Moderate-Severe")
    print("  4: Severe")
    print()
    print("Options:")
    print("  - Press Enter to skip (use all severities)")
    print("  - Enter score (0-4)")
    print()
    
    while True:
        choice = input("Filter by severity [Enter to skip]: ").strip()
        if not choice:
            return None
        try:
            severity = int(choice)
            if 0 <= severity <= 4:
                return severity
            print(f"❌ Severity must be 0-4")
        except ValueError:
            print(f"❌ Invalid number '{choice}'")


def main():
    """Main interactive function."""
    print("="*80)
    print("TREMOR-BRANCH SIGNAL COMPARISON (Non-Normalized)")
    print("="*80)
    print("Comparing tremor-branch datasets (for tremor severity estimation)")
    print("against clean baseline.")
    print()
    print("Tremor-branch signals:")
    print("  - Preserve amplitude information (no z-score normalization)")
    print("  - Include tremor/augmentation and resampling")
    print("  - Skip final augmentation noise")
    print("="*80)
    
    # Select sampling frequency (determines which dataset to load)
    print("\n" + "="*60)
    print("SELECT SAMPLING FREQUENCY")
    print("="*60)
    print("Available configurations:")
    print("  30 - s2_w2_fs30_tremor_* (30 Hz)")
    print("  50 - s2_w2_fs50_tremor_* (50 Hz)")
    print()
    
    while True:
        fs_choice = input("Enter sampling frequency (30/50, default 30): ").strip()
        if not fs_choice:
            fs = 30
            break
        try:
            fs = int(fs_choice)
            if fs in [30, 50]:
                break
            print("❌ Invalid choice. Must be 30 or 50")
        except ValueError:
            print("❌ Invalid input")
    
    # Select tremor variant (always compare against clean)
    tremor_variant = select_tremor_variant()
    
    # Select activity
    activity_idx = select_activity()
    activity_name = ACTIVITY_NAMES[activity_idx]
    
    # Select severity filter
    severity = select_severity()
    
    # Select window index
    window_idx = select_window_index(tremor_variant, activity_idx, fs, severity)
    
    # Ask if user wants to save plots
    save_choice = input("\nSave plots to file? (y/n, default n): ").strip().lower()
    save_dir = None
    if save_choice == 'y':
        severity_str = f"_score{severity}" if severity is not None else ""
        save_dir = SCRIPT_DIR / f"plots_tremorbranch_clean_vs_{tremor_variant}_fs{fs}_act{activity_idx}{severity_str}"
        print(f"Will save to: {save_dir}")
    
    # Generate plots
    figures = plot_tremorbranch_comparison(
        tremor_variant, activity_idx, window_idx, fs, save_dir, severity
    )
    
    if not save_dir and figures:
        print("Displaying plots... (close windows to exit)")
        plt.show()


if __name__ == "__main__":
    main()
