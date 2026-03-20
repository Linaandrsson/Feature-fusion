"""
Plot Clean vs Tremor Signal Comparison
=======================================

Visualize sensor signals before (clean) and after tremor application.
Displays all sensors side-by-side for easy comparison.

Usage:
    python plot_clean_vs_tremor.py
    
    Then follow interactive prompts to select:
    - Dataset variant 1 (e.g., clean)
    - Dataset variant 2 (e.g., mild_mod, mod_severe)
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

# Sensor information
SENSORS = {
    "Acc_chest": {"axes": 3, "unit": "m/s²", "label": "Acc Chest"},
    "ECG": {"axes": 2, "unit": "mV", "label": "ECG Chest"},
    "Acc_ankle": {"axes": 3, "unit": "m/s²", "label": "Acc Ankle"},
    "Gyro_ankle": {"axes": 3, "unit": "deg/s", "label": "Gyro Ankle"},
    "Mag_ankle": {"axes": 3, "unit": "μT", "label": "Mag Ankle"},
    "Acc_arm": {"axes": 3, "unit": "m/s²", "label": "Acc Arm"},
    "Gyro_arm": {"axes": 3, "unit": "deg/s", "label": "Gyro Arm"},
    "Mag_arm": {"axes": 3, "unit": "μT", "label": "Mag Arm"}
}

AXIS_LABELS = {
    3: ['X', 'Y', 'Z'],
    2: ['Lead I', 'Lead II']
}

# Available dataset variants
VARIANTS = ["clean", "mild_mod", "mod_severe"]

# Base path
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_BASE = SCRIPT_DIR.parent.parent / "data" / "Tremor_datagenerator_files"

# ============================================================
# Helper Functions
# ============================================================

def load_sensor_data(variant_name: str, sensor_name: str, fs: int = 50):
    """
    Load sensor data from .npz file.
    
    Args:
        variant_name: Dataset variant (e.g., "s2_w2_fs50_tremor_clean")
        sensor_name: Sensor name (e.g., "Acc_ankle")
        fs: Sampling frequency (default 50 Hz)
        
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


def find_matching_windows(data1, data2, activity_idx, window_idx=0, severity=None):
    """
    Find matching windows from both datasets for the specified activity.
    
    Args:
        data1: First dataset dict
        data2: Second dataset dict
        activity_idx: Activity index (0-11)
        window_idx: Which window to select (default: 0 for first match)
        severity: Optional tremor score filter for dataset 2 (1-4)
        
    Returns:
        Tuple of (signal1, signal2, metadata1, metadata2) or None if not found
    """
    # Find all windows matching the activity
    mask1 = data1['y'] == activity_idx
    mask2 = data2['y'] == activity_idx

    # Optional severity filter for dataset 2
    if severity is not None:
        severity_mask = data2['tremor_score'] == severity
        mask2 = mask2 & severity_mask
    
    if not mask1.any():
        print(f"❌ No samples found for activity {activity_idx} in dataset 1")
        return None
    if not mask2.any():
        if severity is None:
            print(f"❌ No samples found for activity {activity_idx} in dataset 2")
        else:
            print(f"❌ No samples found for activity {activity_idx} with tremor score {severity} in dataset 2")
        return None
    
    # Get matching indices
    indices1 = np.where(mask1)[0]
    indices2 = np.where(mask2)[0]
    
    # Select window
    if window_idx >= len(indices1):
        print(f"❌ Window index {window_idx} out of range for dataset 1 (max: {len(indices1)-1})")
        return None
    if window_idx >= len(indices2):
        print(f"❌ Window index {window_idx} out of range for dataset 2 (max: {len(indices2)-1})")
        return None
    
    idx1 = indices1[window_idx]
    idx2 = indices2[window_idx]
    
    # Extract signals (shape: C, L)
    signal1 = data1['X'][idx1]
    signal2 = data2['X'][idx2]
    
    # Extract metadata
    metadata1 = {
        'subject_id': data1['subject_id'][idx1],
        'tremor_freq': data1['tremor_freq'][idx1],
        'tremor_acc_rms': data1['tremor_acc_rms'][idx1],
        'tremor_gyro_rms': data1['tremor_gyro_rms'][idx1],
        'tremor_score': data1['tremor_score'][idx1]
    }
    
    metadata2 = {
        'subject_id': data2['subject_id'][idx2],
        'tremor_freq': data2['tremor_freq'][idx2],
        'tremor_acc_rms': data2['tremor_acc_rms'][idx2],
        'tremor_gyro_rms': data2['tremor_gyro_rms'][idx2],
        'tremor_score': data2['tremor_score'][idx2]
    }
    
    return signal1, signal2, metadata1, metadata2


def plot_comparison(sensor_name, signal1, signal2, metadata1, metadata2, 
                   activity_idx, variant1_name, variant2_name, fs=50):
    """
    Plot side-by-side comparison of clean vs tremor signals for one sensor.
    
    Args:
        sensor_name: Name of the sensor
        signal1: Signal from dataset 1 (C, L)
        signal2: Signal from dataset 2 (C, L)
        metadata1: Metadata dict for signal 1
        metadata2: Metadata dict for signal 2
        activity_idx: Activity index
        variant1_name: Name of variant 1 (e.g., "clean")
        variant2_name: Name of variant 2 (e.g., "mild_mod")
        fs: Sampling frequency (Hz)
    """
    sensor_info = SENSORS[sensor_name]
    n_axes = sensor_info['axes']
    time = np.arange(signal1.shape[1]) / fs
    
    # Create figure with subplots for each axis
    fig, axes = plt.subplots(n_axes, 2, figsize=(14, 3*n_axes), sharex=True)
    if n_axes == 1:
        axes = axes.reshape(1, -1)
    
    # Activity name
    activity_name = ACTIVITY_NAMES[activity_idx]
    
    # Plot each axis
    axis_labels = AXIS_LABELS[n_axes]
    for ax_idx in range(n_axes):
        # Left column: Dataset 1
        axes[ax_idx, 0].plot(time, signal1[ax_idx], 'b-', linewidth=0.8)
        axes[ax_idx, 0].set_ylabel(f'{axis_labels[ax_idx]}\n({sensor_info["unit"]})', fontsize=10)
        axes[ax_idx, 0].grid(True, alpha=0.3)
        
        # Right column: Dataset 2
        axes[ax_idx, 1].plot(time, signal2[ax_idx], 'r-', linewidth=0.8)
        axes[ax_idx, 1].set_ylabel(f'{axis_labels[ax_idx]}\n({sensor_info["unit"]})', fontsize=10)
        axes[ax_idx, 1].grid(True, alpha=0.3)
    
    # X-axis labels on bottom row
    axes[-1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=11)
    
    # Column titles with metadata
    title1 = f'{variant1_name.upper()}\n'
    title1 += f'Subj: {metadata1["subject_id"]}, Score: {int(metadata1["tremor_score"])}\n'
    title1 += f'RMS: {metadata1["tremor_acc_rms"]:.3f} m/s², Freq: {metadata1["tremor_freq"]:.1f} Hz'
    axes[0, 0].set_title(title1, fontsize=11, fontweight='bold', color='blue')
    
    title2 = f'{variant2_name.upper()}\n'
    title2 += f'Subj: {metadata2["subject_id"]}, Score: {int(metadata2["tremor_score"])}\n'
    title2 += f'RMS: {metadata2["tremor_acc_rms"]:.3f} m/s², Freq: {metadata2["tremor_freq"]:.1f} Hz'
    axes[0, 1].set_title(title2, fontsize=11, fontweight='bold', color='red')
    
    # Main title
    fig.suptitle(
        f'{sensor_info["label"]} — Activity: {activity_name} (ID: {activity_idx})',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def plot_all_sensors_comparison(variant1_name, variant2_name, activity_idx,
                                window_idx=0, fs=50, save_dir=None, severity=None):
    """
    Plot comparison for all sensors.
    
    Args:
        variant1_name: First dataset variant (e.g., "s2_w2_fs50_tremor_clean")
        variant2_name: Second dataset variant (e.g., "s2_w2_fs50_tremor_mild_mod")
        activity_idx: Activity index (0-11)
        window_idx: Window index to plot
        fs: Sampling frequency
        save_dir: Directory to save plots (optional)
        severity: Optional tremor score filter for dataset 2 (1-4)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    activity_name = ACTIVITY_NAMES[activity_idx]
    variant1_short = variant1_name.split('_tremor_')[-1]
    variant2_short = variant2_name.split('_tremor_')[-1]
    
    print("\n" + "="*80)
    print(f"PLOTTING: {variant1_short.upper()} vs {variant2_short.upper()}")
    print(f"Activity: {activity_name} (ID: {activity_idx})")
    print(f"Window: {window_idx}")
    if severity is not None:
        print(f"Severity filter (dataset 2): Score = {severity}")
    print("="*80 + "\n")
    
    figures = []
    
    for sensor_name in SENSORS.keys():
        print(f"Loading {sensor_name}...", end=" ")
        
        try:
            # Load data for both variants
            data1 = load_sensor_data(variant1_name, sensor_name, fs)
            data2 = load_sensor_data(variant2_name, sensor_name, fs)
            
            # Find matching windows
            result = find_matching_windows(data1, data2, activity_idx, window_idx, severity)
            if result is None:
                continue
            
            signal1, signal2, metadata1, metadata2 = result
            
            # Plot
            fig = plot_comparison(
                sensor_name, signal1, signal2, metadata1, metadata2,
                activity_idx, variant1_short, variant2_short, fs
            )
            figures.append((sensor_name, fig))
            
            # Save if requested
            if save_dir:
                filename = f"{sensor_name}_{variant1_short}_vs_{variant2_short}_act{activity_idx}_win{window_idx}.png"
                filepath = save_dir / filename
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"✓ Saved to {filename}")
            else:
                print("✓")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print(f"✓ Generated {len(figures)} comparison plots")
    print("="*80 + "\n")
    
    return figures


# ============================================================
# Interactive CLI
# ============================================================

def select_variant(prompt_text):
    """Interactive variant selection."""
    print(f"\n{prompt_text}")
    for i, var in enumerate(VARIANTS, 1):
        print(f"  {i}. {var}")
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(VARIANTS):
                return VARIANTS[idx]
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


def select_window_index(variant1_name, variant2_name, activity_idx, severity=None):
    """Interactive window index selection."""
    # Load one sensor to get window counts
    try:
        data1 = load_sensor_data(variant1_name, "Acc_arm")
        data2 = load_sensor_data(variant2_name, "Acc_arm")
        
        mask1 = data1['y'] == activity_idx
        mask2 = data2['y'] == activity_idx

        if severity is not None:
            severity_mask = data2['tremor_score'] == severity
            mask2 = mask2 & severity_mask
        
        count1 = mask1.sum()
        count2 = mask2.sum()
        max_windows = min(count1, count2)
        
        print(f"\nAvailable windows for this activity:")
        print(f"  Dataset 1: {count1} windows")
        print(f"  Dataset 2: {count2} windows")
        print(f"  Max index: {max_windows - 1}")

        if max_windows <= 0:
            if severity is None:
                print("❌ No overlapping windows available for selected activity.")
            else:
                print(f"❌ No windows available for selected activity with tremor score {severity} in dataset 2.")
            sys.exit(1)
        
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
    """Optional tremor severity selection for dataset 2."""
    print("\nOptional tremor grade filter for dataset 2:")
    print("  - Press Enter to skip (use all grades)")
    print("  - Enter tremor grade (1-4)")

    while True:
        choice = input("Enter tremor grade [Enter to skip]: ").strip()
        if not choice:
            return None
        try:
            severity = int(choice)
            if 1 <= severity <= 4:
                return severity
            print("❌ Tremor grade must be 1, 2, 3, or 4.")
        except ValueError:
            print(f"❌ Invalid number '{choice}'")


def main():
    """Main interactive function."""
    print("="*80)
    print("CLEAN vs TREMOR SIGNAL COMPARISON")
    print("="*80)
    
    # Select variants
    variant1 = select_variant("Select first dataset variant:")
    variant2 = select_variant("Select second dataset variant:")
    
    # Select activity
    activity_idx = select_activity()
    activity_name = ACTIVITY_NAMES[activity_idx]
    
    # Optional tremor score filter (applied to dataset 2 only)
    severity = select_severity()

    # Construct full variant names (assuming fs=50)
    fs = 50
    variant1_full = f"s2_w2_fs{fs}_tremor_{variant1}"
    variant2_full = f"s2_w2_fs{fs}_tremor_{variant2}"
    
    # Select window index
    window_idx = select_window_index(variant1_full, variant2_full, activity_idx, severity)
    
    # Ask if user wants to save plots
    save_choice = input("\nSave plots to file? (y/n, default n): ").strip().lower()
    save_dir = None
    if save_choice == 'y':
        severity_str = f"_score{severity}" if severity is not None else ""
        save_dir = SCRIPT_DIR / f"plots_{variant1}_vs_{variant2}_act{activity_idx}{severity_str}"
        print(f"Will save to: {save_dir}")
    
    # Generate plots
    figures = plot_all_sensors_comparison(
        variant1_full, variant2_full, activity_idx, window_idx, fs, save_dir, severity
    )
    
    if not save_dir and figures:
        print("Displaying plots... (close windows to exit)")
        plt.show()


if __name__ == "__main__":
    main()
