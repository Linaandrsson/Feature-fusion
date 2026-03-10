"""
Plot IMU Test Data
==================

Visualize IMU sensor data (Accelerometer, Gyroscope, Magnetometer) from CSV files.
Allows selection of time window (4 seconds default) and displays all three sensors.

Usage:
    python plot_imu_data.py
    
Interactive mode will prompt for:
    - Start time (in seconds)
    - Window duration (default: 4 seconds)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# Configuration
# ============================================================

# Default paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "IMU_tests" / "Tremor_Session1_RWrist_Calibrated_PC.csv"

# Default window settings
DEFAULT_WINDOW_DURATION = 4.0  # seconds
DEFAULT_START_TIME = 0.0       # seconds

# ============================================================
# Data Loading
# ============================================================

def load_imu_data(csv_path):
    """
    Load IMU data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with columns: time, acc_x, acc_y, acc_z, gyro_x, gyro_y, 
                                gyro_z, mag_x, mag_y, mag_z
    """
    # Read CSV - skip row 0 (sep definition) and row 2 (units), keep row 1 (header)
    df = pd.read_csv(csv_path, skiprows=[0, 2], sep='\t')
    
    # Extract relevant columns
    data = {
        'timestamp_ms': df['RWrist_TimestampSync_Unix_CAL'].values,
        'acc_x': df['RWrist_Accel_LN_X_CAL'].values,
        'acc_y': df['RWrist_Accel_LN_Y_CAL'].values,
        'acc_z': df['RWrist_Accel_LN_Z_CAL'].values,
        'gyro_x': df['RWrist_Gyro_X_CAL'].values,
        'gyro_y': df['RWrist_Gyro_Y_CAL'].values,
        'gyro_z': df['RWrist_Gyro_Z_CAL'].values,
        'mag_x': df['RWrist_Mag_X_CAL'].values,
        'mag_y': df['RWrist_Mag_Y_CAL'].values,
        'mag_z': df['RWrist_Mag_Z_CAL'].values,
    }
    
    # Convert timestamp to seconds (relative to first sample)
    timestamp_ms = data['timestamp_ms']
    time_s = (timestamp_ms - timestamp_ms[0]) / 1000.0
    data['time'] = time_s
    
    # Create DataFrame
    df_out = pd.DataFrame(data)
    
    # Calculate sampling rate
    dt = np.diff(time_s)
    mean_dt = np.mean(dt)
    fs = 1.0 / mean_dt if mean_dt > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"IMU Data Loaded:")
    print(f"{'='*80}")
    print(f"  Total samples: {len(df_out)}")
    print(f"  Duration: {time_s[-1]:.2f} seconds")
    print(f"  Mean sampling interval: {mean_dt*1000:.2f} ms")
    print(f"  Estimated sampling rate: {fs:.2f} Hz")
    print(f"  Time range: {time_s[0]:.2f} - {time_s[-1]:.2f} s")
    print(f"{'='*80}\n")
    
    return df_out


def extract_window(df, start_time, duration):
    """
    Extract a time window from the data.
    
    Args:
        df: DataFrame with 'time' column
        start_time: Start time in seconds
        duration: Window duration in seconds
        
    Returns:
        DataFrame with data in the specified time window
    """
    end_time = start_time + duration
    mask = (df['time'] >= start_time) & (df['time'] < end_time)
    
    window_df = df[mask].copy()
    
    if len(window_df) == 0:
        raise ValueError(f"No data found in time range [{start_time:.2f}, {end_time:.2f}] s")
    
    # Make time relative to window start
    window_df['time'] = window_df['time'] - start_time
    
    return window_df


# ============================================================
# Plotting
# ============================================================

def plot_imu_window(df, window_start, window_duration, save_path=None):
    """
    Plot IMU data (Acc, Gyro, Mag) for a specified time window.
    
    Args:
        df: Full DataFrame with IMU data
        window_start: Start time in seconds
        window_duration: Duration in seconds
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    # Extract window
    try:
        window = extract_window(df, window_start, window_duration)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None
    
    time = window['time'].values
    
    # Create figure with 3 subplots (one for each sensor type)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # -------------------- Accelerometer --------------------
    ax = axes[0]
    ax.plot(time, window['acc_x'], 'r-', label='X', linewidth=1.2, alpha=0.8)
    ax.plot(time, window['acc_y'], 'g-', label='Y', linewidth=1.2, alpha=0.8)
    ax.plot(time, window['acc_z'], 'b-', label='Z', linewidth=1.2, alpha=0.8)
    ax.set_ylabel('Acceleration\n(m/s²)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Accelerometer', fontsize=13, fontweight='bold', loc='left')
    
    # -------------------- Gyroscope --------------------
    ax = axes[1]
    ax.plot(time, window['gyro_x'], 'r-', label='X', linewidth=1.2, alpha=0.8)
    ax.plot(time, window['gyro_y'], 'g-', label='Y', linewidth=1.2, alpha=0.8)
    ax.plot(time, window['gyro_z'], 'b-', label='Z', linewidth=1.2, alpha=0.8)
    ax.set_ylabel('Angular Velocity\n(deg/s)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Gyroscope', fontsize=13, fontweight='bold', loc='left')
    
    # -------------------- Magnetometer --------------------
    ax = axes[2]
    ax.plot(time, window['mag_x'], 'r-', label='X', linewidth=1.2, alpha=0.8)
    ax.plot(time, window['mag_y'], 'g-', label='Y', linewidth=1.2, alpha=0.8)
    ax.plot(time, window['mag_z'], 'b-', label='Z', linewidth=1.2, alpha=0.8)
    ax.set_ylabel('Magnetic Field\n(local flux)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Magnetometer', fontsize=13, fontweight='bold', loc='left')
    
    # Main title
    window_end = window_start + window_duration
    fig.suptitle(
        f'IMU Data — Time Window: [{window_start:.2f} - {window_end:.2f}] s  '
        f'({len(window)} samples)',
        fontsize=15, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to: {save_path}")
    
    return fig


def plot_multiple_windows(df, window_starts, window_duration, save_dir=None):
    """
    Plot multiple time windows.
    
    Args:
        df: DataFrame with IMU data
        window_starts: List of start times (seconds)
        window_duration: Duration of each window (seconds)
        save_dir: Directory to save plots (optional)
        
    Returns:
        List of Figure objects
    """
    figures = []
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, start in enumerate(window_starts, 1):
        print(f"\n[{i}/{len(window_starts)}] Plotting window starting at {start:.2f} s...")
        
        # Save path
        save_path = None
        if save_dir:
            save_path = save_dir / f"imu_window_s{start:.1f}_d{window_duration:.1f}.png"
        
        # Plot
        fig = plot_imu_window(df, start, window_duration, save_path)
        if fig:
            figures.append(fig)
    
    print(f"\n✓ Generated {len(figures)} plots")
    return figures


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
    """Ask if user wants to plot multiple windows."""
    choice = input("\nPlot multiple windows? (y/n, default n): ").strip().lower()
    return choice == 'y'


def main():
    """Main interactive function."""
    print("="*80)
    print("IMU DATA VISUALIZATION")
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
        save_choice = input("\nSave plots to file? (y/n, default y): ").strip().lower()
        save_dir = None
        if save_choice != 'n':
            save_dir = SCRIPT_DIR / "plots"
            print(f"Will save to: {save_dir}")
        
        # Plot
        figures = plot_multiple_windows(df, window_starts, window_duration, save_dir)
        
        if not save_dir and figures:
            print("\nDisplaying plots... (close windows to exit)")
            plt.show()
    else:
        # Single window
        start_time = select_start_time(total_duration, window_duration)
        
        # Ask about saving
        save_choice = input("\nSave plot to file? (y/n, default n): ").strip().lower()
        save_path = None
        if save_choice == 'y':
            save_dir = SCRIPT_DIR / "plots"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"imu_plot_s{start_time:.1f}_d{window_duration:.1f}.png"
            print(f"Will save to: {save_path}")
        
        # Plot
        fig = plot_imu_window(df, start_time, window_duration, save_path)
        
        if fig and not save_path:
            print("\nDisplaying plot... (close window to exit)")
            plt.show()


if __name__ == "__main__":
    main()
