"""
Calculate RMS for Tremor-Branch Dataset Windows
================================================

Calculate Root Mean Square (RMS) values for tremor-branch sensors.
These are non-normalized signals for tremor severity estimation.

Tremor-branch sensors:
- Acc_arm_tremorbranch
- Gyro_arm_tremorbranch

These signals preserve amplitude information (no z-score normalization)
for tremor severity estimation, unlike the HAR branch.

RMS is calculated as:
    RMS = sqrt(mean(signal^2))

For 3-axis sensors, both per-axis and total magnitude RMS are calculated:
    RMS_total = sqrt(RMS_x^2 + RMS_y^2 + RMS_z^2)

Usage:
    python calculate_rms_tremorbranch.py
"""

import numpy as np
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

# Tremor-branch sensors only (non-normalized for tremor estimation)
TREMOR_BRANCH_SENSORS = {
    "Acc_arm_tremorbranch": {"axes": 3, "unit": "m/s²", "label": "Accelerometer Arm (Tremor Branch)"},
    "Gyro_arm_tremorbranch": {"axes": 3, "unit": "deg/s", "label": "Gyroscope Arm (Tremor Branch)"}
}

AXIS_LABELS = ['X', 'Y', 'Z']

# Available dataset variants
VARIANTS = {
    'clean': 's2_w2_fs30_tremor_clean',
    'mild': 's2_w2_fs30_tremor_mild_mod',
    'severe': 's2_w2_fs30_tremor_mod_severe'
}

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_BASE = SCRIPT_DIR.parent.parent / "data" / "Tremor_datagenerator_files"
OUTPUT_DIR = SCRIPT_DIR / "rms_results_tremorbranch"

# ============================================================
# Data Loading Functions
# ============================================================

def load_sensor_data(variant_name: str, sensor_name: str):
    """
    Load tremor-branch sensor data from .npz file.
    
    Args:
        variant_name: Dataset variant (e.g., "s2_w2_fs30_tremor_clean")
        sensor_name: Sensor name (e.g., "Acc_arm_tremorbranch")
        
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


def get_activity_window(variant_key: str, activity_idx: int, window_idx: int = 0, score: float = None):
    """
    Get a specific window for a given activity from tremor-branch sensors.
    
    Args:
        variant_key: Variant key ('clean', 'mild', 'severe')
        activity_idx: Activity index (0-11)
        window_idx: Which window to select (default: 0 for first match)
        score: Optional tremor score to filter by (e.g., 4.0)
        
    Returns:
        Dictionary with sensor signals and metadata
    """
    variant_name = VARIANTS[variant_key]
    
    # Step 1: Use Acc_arm_tremorbranch to find the global index
    try:
        ref_data = load_sensor_data(variant_name, 'Acc_arm_tremorbranch')
        
        # Filter by activity
        mask = ref_data['y'] == activity_idx
        
        # Add score filtering if specified
        if score is not None:
            score_mask = ref_data['tremor_score'] == score
            mask = mask & score_mask
        
        if not mask.any():
            if score is not None:
                raise ValueError(f"No samples found for activity {activity_idx} with score {score}")
            else:
                raise ValueError(f"No samples found for activity {activity_idx}")
        
        indices = np.where(mask)[0]
        if window_idx >= len(indices):
            raise ValueError(f"Window index {window_idx} out of range (found {len(indices)} windows, max index: {len(indices)-1})")
        
        # Get the global index
        global_idx = indices[window_idx]
        
    except Exception as e:
        print(f"❌ Error finding window index: {e}")
        return None
    
    # Step 2: Load all tremor-branch sensors using the global index
    result = {}
    
    for sensor_name in TREMOR_BRANCH_SENSORS.keys():
        try:
            data = load_sensor_data(variant_name, sensor_name)
            
            # Use the global index directly
            if global_idx >= len(data['X']):
                raise ValueError(f"Global index {global_idx} out of range for {sensor_name}")
            
            # Store the signal (C, L) - channels x time
            result[sensor_name] = {
                'signal': data['X'][global_idx],
                'subject_id': data['subject_id'][global_idx],
                'tremor_freq': data['tremor_freq'][global_idx],
                'tremor_acc_rms': data['tremor_acc_rms'][global_idx],
                'tremor_gyro_rms': data['tremor_gyro_rms'][global_idx],
                'tremor_score': data['tremor_score'][global_idx]
            }
            
        except Exception as e:
            print(f"❌ Error loading {sensor_name}: {e}")
            return None
    
    # Add metadata (use Acc_arm_tremorbranch for tremor labels)
    result['metadata'] = {
        'variant': variant_key,
        'variant_name': variant_name,
        'activity_idx': activity_idx,
        'activity_name': ACTIVITY_NAMES[activity_idx],
        'window_idx': window_idx,
        'global_idx': global_idx,
        'subject_id': result['Acc_arm_tremorbranch']['subject_id'],
        'tremor_freq': result['Acc_arm_tremorbranch']['tremor_freq'],
        'tremor_acc_rms': result['Acc_arm_tremorbranch']['tremor_acc_rms'],
        'tremor_gyro_rms': result['Acc_arm_tremorbranch']['tremor_gyro_rms'],
        'tremor_score': result['Acc_arm_tremorbranch']['tremor_score']
    }
    
    return result


# ============================================================
# RMS Calculation Functions
# ============================================================

def calculate_rms(signal):
    """
    Calculate RMS (Root Mean Square) of a signal.
    
    Args:
        signal: 1D array of signal values
        
    Returns:
        RMS value (float)
    """
    return np.sqrt(np.mean(signal ** 2))


def calculate_sensor_rms(sensor_signal, num_axes):
    """
    Calculate RMS for a multi-axis sensor.
    
    Args:
        sensor_signal: Array of shape (C, L) where C is number of channels/axes
        num_axes: Number of axes (always 3 for tremor-branch sensors)
        
    Returns:
        Dictionary with per-axis RMS and magnitude RMS
    """
    rms_values = {}
    
    # Calculate RMS for each axis
    for i in range(num_axes):
        rms_values[f'axis_{i}'] = calculate_rms(sensor_signal[i])
    
    # Calculate total magnitude RMS
    # RMS_total = sqrt(RMS_x^2 + RMS_y^2 + RMS_z^2)
    rms_values['magnitude'] = np.sqrt(
        rms_values['axis_0']**2 + 
        rms_values['axis_1']**2 + 
        rms_values['axis_2']**2
    )
    
    return rms_values


def calculate_all_rms(window_data):
    """
    Calculate RMS for all tremor-branch sensors in a window.
    
    Args:
        window_data: Dictionary from get_activity_window()
        
    Returns:
        Dictionary with RMS values for all sensors
    """
    if window_data is None:
        return None
    
    results = {'metadata': window_data['metadata'], 'sensors': {}}
    
    for sensor_name, sensor_info in TREMOR_BRANCH_SENSORS.items():
        if sensor_name in window_data:
            signal = window_data[sensor_name]['signal']
            num_axes = sensor_info['axes']
            rms = calculate_sensor_rms(signal, num_axes)
            results['sensors'][sensor_name] = {
                'rms': rms,
                'unit': sensor_info['unit'],
                'label': sensor_info['label'],
                'num_axes': num_axes
            }
    
    return results


# ============================================================
# Display and Export Functions
# ============================================================

def print_rms_results(results):
    """
    Print RMS results in a formatted display.
    
    Args:
        results: Dictionary with RMS values from calculate_all_rms()
    """
    if results is None:
        return
    
    meta = results['metadata']
    
    print("\n" + "="*80)
    print("RMS ANALYSIS RESULTS - TREMOR-BRANCH DATASET (Non-Normalized)")
    print("="*80)
    print(f"Variant: {meta['variant'].upper()} ({meta['variant_name']})")
    print(f"Activity: {meta['activity_name']} (index {meta['activity_idx']})")
    print(f"Window: {meta['window_idx']} (global index: {meta['global_idx']})")
    print(f"Subject: {meta['subject_id']}")
    print(f"Tremor frequency: {meta['tremor_freq']:.2f} Hz")
    print(f"Tremor score: {meta['tremor_score']:.4f}")
    print()
    print("NOTE: These are NON-NORMALIZED signals (preserve amplitude for tremor estimation)")
    print("="*80 + "\n")
    
    # Print each sensor
    for sensor_name in TREMOR_BRANCH_SENSORS.keys():
        if sensor_name in results['sensors']:
            sensor = results['sensors'][sensor_name]
            rms = sensor['rms']
            unit = sensor['unit']
            label = sensor['label']
            
            print(f"{label.upper()} ({unit})")
            print("-" * 80)
            print(f"  X-axis RMS:      {rms['axis_0']:>10.4f} {unit}")
            print(f"  Y-axis RMS:      {rms['axis_1']:>10.4f} {unit}")
            print(f"  Z-axis RMS:      {rms['axis_2']:>10.4f} {unit}")
            print(f"  Magnitude RMS:   {rms['magnitude']:>10.4f} {unit}")
            print()
    
    print("="*80 + "\n")


def export_rms_to_txt(results, output_dir=None, custom_filename=None):
    """
    Export RMS results to a readable TXT file.
    
    Args:
        results: Dictionary with RMS values
        output_dir: Output directory (default: rms_results_tremorbranch)
        custom_filename: Custom filename (default: auto-generated)
        
    Returns:
        Path to the saved file
    """
    if results is None:
        return None
    
    # Setup output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    meta = results['metadata']
    if custom_filename is None:
        filename = f"rms_tremorbranch_{meta['variant']}_act{meta['activity_idx']}_win{meta['window_idx']}.txt"
    else:
        filename = custom_filename
    
    output_path = output_dir / filename
    
    # Build the text output
    lines = []
    lines.append("=" * 80)
    lines.append("RMS ANALYSIS RESULTS - TREMOR-BRANCH DATASET (Non-Normalized)")
    lines.append("=" * 80)
    lines.append(f"Variant: {meta['variant'].upper()} ({meta['variant_name']})")
    lines.append(f"Activity: {meta['activity_name']} (index {meta['activity_idx']})")
    lines.append(f"Window: {meta['window_idx']} (global index: {meta['global_idx']})")
    lines.append(f"Subject: {meta['subject_id']}")
    lines.append(f"Tremor frequency: {meta['tremor_freq']:.2f} Hz")
    lines.append(f"Tremor score: {meta['tremor_score']:.4f}")
    lines.append(f"Tremor acc RMS: {meta['tremor_acc_rms']:.4f} m/s²")
    lines.append(f"Tremor gyro RMS: {meta['tremor_gyro_rms']:.4f} deg/s")
    lines.append("")
    lines.append("NOTE: These are NON-NORMALIZED signals")
    lines.append("      Amplitude is preserved for tremor severity estimation")
    lines.append("=" * 80)
    lines.append("")
    
    # Add each sensor
    for sensor_name in TREMOR_BRANCH_SENSORS.keys():
        if sensor_name in results['sensors']:
            sensor = results['sensors'][sensor_name]
            rms = sensor['rms']
            unit = sensor['unit']
            label = sensor['label']
            
            lines.append(f"{label.upper()} ({unit})")
            lines.append("-" * 80)
            lines.append(f"  X-axis RMS:      {rms['axis_0']:>10.4f} {unit}")
            lines.append(f"  Y-axis RMS:      {rms['axis_1']:>10.4f} {unit}")
            lines.append(f"  Z-axis RMS:      {rms['axis_2']:>10.4f} {unit}")
            lines.append(f"  Magnitude RMS:   {rms['magnitude']:>10.4f} {unit}")
            lines.append("")
    
    lines.append("=" * 80)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ RMS results exported to: {output_path}")
    return output_path


# ============================================================
# Interactive CLI
# ============================================================

def select_variant():
    """Interactive variant selection."""
    print("\n" + "="*60)
    print("SELECT TREMOR VARIANT")
    print("="*60)
    print("Available variants:")
    for key, name in VARIANTS.items():
        print(f"  {key:8s} - {name}")
    print()
    
    while True:
        choice = input("Enter variant (clean/mild/severe): ").strip().lower()
        if choice in VARIANTS:
            return choice
        print(f"❌ Invalid choice '{choice}'. Please choose: clean, mild, or severe")


def select_activity():
    """Interactive activity selection."""
    print("\n" + "="*60)
    print("SELECT ACTIVITY")
    print("="*60)
    print("Activities (0-11):")
    for idx, name in ACTIVITY_NAMES.items():
        print(f"  [{idx:2d}] {name}")
    print()
    
    while True:
        choice = input("Enter activity index (0-11) or name: ").strip()
        
        # Try as index
        try:
            idx = int(choice)
            if 0 <= idx <= 11:
                return idx
            print(f"❌ Index {idx} out of range (0-11)")
            continue
        except ValueError:
            pass
        
        # Try as name
        choice_lower = choice.lower()
        for idx, name in ACTIVITY_NAMES.items():
            if name.lower() == choice_lower:
                return idx
        
        print(f"❌ Invalid activity '{choice}'")


def select_score(variant_key):
    """Interactive tremor score selection."""
    print("\n" + "="*60)
    print("SELECT SCORE (OPTIONAL)")
    print("="*60)
    
    # Show typical scores for each variant
    score_info = {
        'clean': 'Score 0 (no tremor)',
        'mild': 'Scores 1-2 (mild tremor)',
        'severe': 'Scores 3-4 (moderate-severe tremor)'
    }
    
    print(f"Typical scores for {variant_key.upper()}:")
    print(f"  {score_info[variant_key]}")
    print("\nOptions:")
    print("  - Press Enter to skip (use all scores)")
    print("  - Enter specific score (e.g., 4)")
    print()
    
    while True:
        choice = input("Filter by score [Enter to skip]: ").strip()
        if not choice:
            return None
        try:
            score = float(choice)
            if score >= 0:
                return score
            print(f"❌ Score must be >= 0")
        except ValueError:
            print(f"❌ Invalid number '{choice}'")


def select_window_index():
    """Interactive window index selection."""
    print("\n" + "="*60)
    print("SELECT WINDOW")
    print("="*60)
    
    while True:
        choice = input("Enter window index (default: 0): ").strip()
        if not choice:
            return 0
        try:
            idx = int(choice)
            if idx >= 0:
                return idx
            print(f"❌ Window index must be >= 0")
        except ValueError:
            print(f"❌ Invalid number '{choice}'")


def main():
    """Main interactive function."""
    print("="*60)
    print("RMS ANALYSIS - TREMOR-BRANCH DATASET")
    print("="*60)
    print("Analyzing non-normalized signals for tremor estimation")
    print()
    
    # Select parameters
    variant = select_variant()
    activity_idx = select_activity()
    score = select_score(variant)
    window_idx = select_window_index()
    
    # Load window data
    print(f"\nLoading window data...")
    if score is not None:
        print(f"  Filtering: variant={variant}, activity={activity_idx}, score={score}, window={window_idx}")
    else:
        print(f"  Filtering: variant={variant}, activity={activity_idx}, window={window_idx}")
    
    window_data = get_activity_window(variant, activity_idx, window_idx, score=score)
    
    if window_data is None:
        print("❌ Failed to load window data")
        return
    
    print(f"✓ Successfully loaded window {window_idx} for activity '{ACTIVITY_NAMES[activity_idx]}'")
    
    # Calculate RMS
    print("\nCalculating RMS for tremor-branch sensors...")
    results = calculate_all_rms(window_data)
    
    # Display results
    print_rms_results(results)
    
    # Export to TXT
    output_path = export_rms_to_txt(results)
    
    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
