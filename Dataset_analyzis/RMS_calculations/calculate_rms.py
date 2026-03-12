"""
Calculate RMS for IMU Sensors
==============================

Calculate Root Mean Square (RMS) values for Accelerometer, Gyroscope, 
and Magnetometer signals within a specified time window.

RMS is calculated as:
    RMS = sqrt(mean(signal^2))

For 3-axis sensors, both per-axis and total magnitude RMS are calculated:
    RMS_total = sqrt(RMS_x^2 + RMS_y^2 + RMS_z^2)

Usage:
    python calculate_rms.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import from Plot_IMU_test (parent directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "Plot_IMU_test"))
from plot_imu_data import load_imu_data, extract_window

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "IMU_tests" / "Tremor_Session1_RWrist_Calibrated_PC.csv"

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


def calculate_sensor_rms(window_df, sensor_prefix, axes=['x', 'y', 'z']):
    """
    Calculate RMS for a multi-axis sensor.
    
    Args:
        window_df: DataFrame with windowed data
        sensor_prefix: Sensor prefix ('acc', 'gyro', 'mag')
        axes: List of axis names (default: ['x', 'y', 'z'])
        
    Returns:
        Dictionary with per-axis RMS and total magnitude RMS
    """
    rms_values = {}
    
    # Calculate RMS for each axis
    for axis in axes:
        col_name = f"{sensor_prefix}_{axis}"
        if col_name in window_df.columns:
            rms_values[axis] = calculate_rms(window_df[col_name].values)
        else:
            rms_values[axis] = np.nan
    
    # Calculate total magnitude RMS
    # RMS_total = sqrt(RMS_x^2 + RMS_y^2 + RMS_z^2)
    valid_rms = [rms_values[ax] for ax in axes if not np.isnan(rms_values[ax])]
    if valid_rms:
        rms_values['magnitude'] = np.sqrt(np.sum([v**2 for v in valid_rms]))
    else:
        rms_values['magnitude'] = np.nan
    
    return rms_values


def calculate_all_rms(df, start_time, duration):
    """
    Calculate RMS for all sensors in a time window.
    
    Args:
        df: Full DataFrame with IMU data
        start_time: Start time in seconds
        duration: Window duration in seconds
        
    Returns:
        Dictionary with RMS values for all sensors
    """
    # Extract window
    try:
        window = extract_window(df, start_time, duration)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None
    
    # Calculate RMS for each sensor
    results = {
        'window': {
            'start_time': start_time,
            'duration': duration,
            'end_time': start_time + duration,
            'num_samples': len(window)
        },
        'accelerometer': calculate_sensor_rms(window, 'acc'),
        'gyroscope': calculate_sensor_rms(window, 'gyro'),
        'magnetometer': calculate_sensor_rms(window, 'mag')
    }
    
    return results


# ============================================================
# Display Functions
# ============================================================

def print_rms_results(results):
    """
    Print RMS results in a formatted table.
    
    Args:
        results: Dictionary with RMS values from calculate_all_rms()
    """
    if results is None:
        return
    
    window_info = results['window']
    
    print("\n" + "="*80)
    print("RMS ANALYSIS RESULTS")
    print("="*80)
    print(f"Time window: [{window_info['start_time']:.2f} - {window_info['end_time']:.2f}] s")
    print(f"Duration: {window_info['duration']:.2f} s")
    print(f"Samples: {window_info['num_samples']}")
    print("="*80 + "\n")
    
    # Accelerometer
    acc = results['accelerometer']
    print("ACCELEROMETER (m/s²)")
    print("-" * 80)
    print(f"  X-axis RMS:      {acc['x']:>10.4f} m/s²")
    print(f"  Y-axis RMS:      {acc['y']:>10.4f} m/s²")
    print(f"  Z-axis RMS:      {acc['z']:>10.4f} m/s²")
    print(f"  Magnitude RMS:   {acc['magnitude']:>10.4f} m/s²")
    print()
    
    # Gyroscope
    gyro = results['gyroscope']
    print("GYROSCOPE (deg/s)")
    print("-" * 80)
    print(f"  X-axis RMS:      {gyro['x']:>10.4f} deg/s")
    print(f"  Y-axis RMS:      {gyro['y']:>10.4f} deg/s")
    print(f"  Z-axis RMS:      {gyro['z']:>10.4f} deg/s")
    print(f"  Magnitude RMS:   {gyro['magnitude']:>10.4f} deg/s")
    print()
    
    # Magnetometer
    mag = results['magnetometer']
    print("MAGNETOMETER (local flux)")
    print("-" * 80)
    print(f"  X-axis RMS:      {mag['x']:>10.4f} local flux")
    print(f"  Y-axis RMS:      {mag['y']:>10.4f} local flux")
    print(f"  Z-axis RMS:      {mag['z']:>10.4f} local flux")
    print(f"  Magnitude RMS:   {mag['magnitude']:>10.4f} local flux")
    print()
    print("="*80 + "\n")


def export_rms_to_csv(results, output_path):
    """
    Export RMS results to CSV file.
    
    Args:
        results: Dictionary with RMS values
        output_path: Path to save CSV file
    """
    if results is None:
        return
    
    # Prepare data for CSV
    rows = []
    
    window_info = results['window']
    
    # Add accelerometer data
    acc = results['accelerometer']
    rows.append({
        'sensor': 'Accelerometer',
        'axis': 'X',
        'rms': acc['x'],
        'unit': 'm/s²',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Accelerometer',
        'axis': 'Y',
        'rms': acc['y'],
        'unit': 'm/s²',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Accelerometer',
        'axis': 'Z',
        'rms': acc['z'],
        'unit': 'm/s²',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Accelerometer',
        'axis': 'Magnitude',
        'rms': acc['magnitude'],
        'unit': 'm/s²',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    
    # Add gyroscope data
    gyro = results['gyroscope']
    rows.append({
        'sensor': 'Gyroscope',
        'axis': 'X',
        'rms': gyro['x'],
        'unit': 'deg/s',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Gyroscope',
        'axis': 'Y',
        'rms': gyro['y'],
        'unit': 'deg/s',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Gyroscope',
        'axis': 'Z',
        'rms': gyro['z'],
        'unit': 'deg/s',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Gyroscope',
        'axis': 'Magnitude',
        'rms': gyro['magnitude'],
        'unit': 'deg/s',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    
    # Add magnetometer data
    mag = results['magnetometer']
    rows.append({
        'sensor': 'Magnetometer',
        'axis': 'X',
        'rms': mag['x'],
        'unit': 'local flux',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Magnetometer',
        'axis': 'Y',
        'rms': mag['y'],
        'unit': 'local flux',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Magnetometer',
        'axis': 'Z',
        'rms': mag['z'],
        'unit': 'local flux',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    rows.append({
        'sensor': 'Magnetometer',
        'axis': 'Magnitude',
        'rms': mag['magnitude'],
        'unit': 'local flux',
        'start_time': window_info['start_time'],
        'duration': window_info['duration'],
        'num_samples': window_info['num_samples']
    })
    
    # Create DataFrame and save
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"✓ RMS results exported to: {output_path}")


def export_rms_to_txt(results, output_path):
    """
    Export RMS results to a readable TXT file.
    
    Args:
        results: Dictionary with RMS values
        output_path: Path to save TXT file
    """
    if results is None:
        return
    
    window_info = results['window']
    acc = results['accelerometer']
    gyro = results['gyroscope']
    mag = results['magnetometer']
    
    # Build the text output
    lines = []
    lines.append("=" * 80)
    lines.append("RMS ANALYSIS RESULTS")
    lines.append("=" * 80)
    lines.append(f"Time window: [{window_info['start_time']:.2f} - {window_info['end_time']:.2f}] s")
    lines.append(f"Duration: {window_info['duration']:.2f} s")
    lines.append(f"Samples: {window_info['num_samples']}")
    lines.append("=" * 80)
    lines.append("")
    
    # Accelerometer
    lines.append("ACCELEROMETER (m/s²)")
    lines.append("-" * 80)
    lines.append(f"  X-axis RMS:      {acc['x']:>10.4f} m/s²")
    lines.append(f"  Y-axis RMS:      {acc['y']:>10.4f} m/s²")
    lines.append(f"  Z-axis RMS:      {acc['z']:>10.4f} m/s²")
    lines.append(f"  Magnitude RMS:   {acc['magnitude']:>10.4f} m/s²")
    lines.append("")
    
    # Gyroscope
    lines.append("GYROSCOPE (deg/s)")
    lines.append("-" * 80)
    lines.append(f"  X-axis RMS:      {gyro['x']:>10.4f} deg/s")
    lines.append(f"  Y-axis RMS:      {gyro['y']:>10.4f} deg/s")
    lines.append(f"  Z-axis RMS:      {gyro['z']:>10.4f} deg/s")
    lines.append(f"  Magnitude RMS:   {gyro['magnitude']:>10.4f} deg/s")
    lines.append("")
    
    # Magnetometer
    lines.append("MAGNETOMETER (local flux)")
    lines.append("-" * 80)
    lines.append(f"  X-axis RMS:      {mag['x']:>10.4f} local flux")
    lines.append(f"  Y-axis RMS:      {mag['y']:>10.4f} local flux")
    lines.append(f"  Z-axis RMS:      {mag['z']:>10.4f} local flux")
    lines.append(f"  Magnitude RMS:   {mag['magnitude']:>10.4f} local flux")
    lines.append("")
    lines.append("=" * 80)
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ RMS results exported to: {output_path}")


def calculate_multiple_windows(df, window_starts, duration, save_txt=True):
    """
    Calculate RMS for multiple time windows.
    
    Args:
        df: DataFrame with IMU data
        window_starts: List of start times (seconds)
        duration: Window duration (seconds)
        save_txt: Save results to TXT file (bool, default True)
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for i, start in enumerate(window_starts, 1):
        print(f"\n[{i}/{len(window_starts)}] Analyzing window starting at {start:.2f} s...")
        
        results = calculate_all_rms(df, start, duration)
        if results:
            all_results.append(results)
            print_rms_results(results)
            
            if save_txt:
                txt_path = SCRIPT_DIR / "rms_results_IMU_test" / f"rms_s{start:.1f}_d{duration:.1f}.txt"
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                export_rms_to_txt(results, txt_path)
    
    return all_results


# ============================================================
# Interactive CLI
# ============================================================

def select_start_time(total_duration, window_duration):
    """Interactive start time selection."""
    max_start = total_duration - window_duration
    
    print(f"\nSelect start time:")
    print(f"  Total duration: {total_duration:.2f} s")
    print(f"  Window duration: {window_duration:.2f} s")
    print(f"  Valid range: 0.0 - {max_start:.2f} s")
    
    while True:
        try:
            choice = input(f"\nEnter start time in seconds (0-{max_start:.1f}, default 0): ").strip()
            if choice == "":
                return 0.0
            
            start_time = float(choice)
            if 0 <= start_time <= max_start:
                return start_time
            else:
                print(f"❌ Invalid time. Must be between 0 and {max_start:.2f} s.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input.")
            sys.exit(1)


def select_window_duration(total_duration):
    """Interactive window duration selection."""
    print(f"\nSelect window duration:")
    print(f"  Total duration: {total_duration:.2f} s")
    print(f"  Suggested: 4.0 s")
    
    while True:
        try:
            choice = input(f"\nEnter window duration in seconds (default 4.0): ").strip()
            if choice == "":
                return 4.0
            
            duration = float(choice)
            if 0 < duration <= total_duration:
                return duration
            else:
                print(f"❌ Invalid duration. Must be between 0 and {total_duration:.2f} s.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input.")
            sys.exit(1)


def select_multiple_windows():
    """Ask if user wants to analyze multiple windows."""
    choice = input("\nAnalyze multiple windows? (y/n, default n): ").strip().lower()
    return choice == 'y'


def main():
    """Main interactive function."""
    print("="*80)
    print("IMU RMS CALCULATOR")
    print("="*80)
    
    # Load data
    csv_path = DEFAULT_CSV_PATH
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    print(f"\nLoading data from:\n  {csv_path}")
    df = load_imu_data(csv_path)
    
    total_duration = df['time'].max()
    
    # Select window duration
    window_duration = select_window_duration(total_duration)
    
    # Multiple windows?
    if select_multiple_windows():
        # Get multiple start times
        print("\nEnter start times (comma-separated, e.g., '0, 5, 10'):")
        start_times_str = input("Start times: ").strip()
        try:
            window_starts = [float(t.strip()) for t in start_times_str.split(',')]
        except:
            print("❌ Invalid input. Using single window at t=0.")
            window_starts = [0.0]
        
        # Ask about saving
        save_choice = input("\nSave results to TXT? (y/n, default y): ").strip().lower()
        save_txt = save_choice != 'n'
        
        # Calculate
        results = calculate_multiple_windows(df, window_starts, window_duration, save_txt)
        
        print(f"\n✓ Analyzed {len(results)} windows")
    else:
        # Single window
        start_time = select_start_time(total_duration, window_duration)
        
        # Calculate RMS
        results = calculate_all_rms(df, start_time, window_duration)
        
        # Display results
        print_rms_results(results)
        
        # Ask about saving
        save_choice = input("Save results to TXT? (y/n, default n): ").strip().lower()
        if save_choice == 'y':
            txt_path = SCRIPT_DIR / "rms_results_IMU_test" / f"rms_s{start_time:.1f}_d{window_duration:.1f}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            export_rms_to_txt(results, txt_path)


if __name__ == "__main__":
    main()
