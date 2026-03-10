"""
Example: Quick IMU data plotting
=================================

Simple example showing how to plot IMU data directly.
Modify the parameters below and run.
"""

from plot_imu_data import load_imu_data, plot_imu_window, plot_multiple_windows
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION - Modify these parameters
# ============================================================

# CSV file path
SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "IMU_tests" / "Tremor_Session1_RWrist_Calibrated_PC.csv"

# Time window settings
START_TIME = 25.0      # Start time in seconds
WINDOW_DURATION = 10.0  # Window duration in seconds

# Save plot? (True/False)
SAVE_PLOT = True

# For multiple windows, set this to a list of start times
# Example: MULTIPLE_WINDOWS = [0, 5, 10, 15, 20]
# Set to None for single window
MULTIPLE_WINDOWS = None  # [0, 10, 20, 30]

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("IMU DATA PLOTTING")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from:\n  {CSV_PATH}")
    df = load_imu_data(CSV_PATH)
    
    if MULTIPLE_WINDOWS:
        # Plot multiple windows
        print(f"\nPlotting {len(MULTIPLE_WINDOWS)} windows...")
        print(f"Start times: {MULTIPLE_WINDOWS}")
        print(f"Window duration: {WINDOW_DURATION} s")
        
        save_dir = SCRIPT_DIR / "plots" if SAVE_PLOT else None
        
        figures = plot_multiple_windows(
            df, 
            MULTIPLE_WINDOWS, 
            WINDOW_DURATION, 
            save_dir
        )
        
        if not SAVE_PLOT and figures:
            print("\nDisplaying plots... (close windows to exit)")
            plt.show()
        elif SAVE_PLOT:
            print(f"\n✓ All plots saved to: {save_dir}")
    else:
        # Plot single window
        print(f"\nPlotting window:")
        print(f"  Start time: {START_TIME} s")
        print(f"  Duration: {WINDOW_DURATION} s")
        
        save_path = None
        if SAVE_PLOT:
            save_dir = SCRIPT_DIR / "plots"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"imu_plot_s{START_TIME:.1f}_d{WINDOW_DURATION:.1f}.png"
        
        fig = plot_imu_window(df, START_TIME, WINDOW_DURATION, save_path)
        
        if fig and not SAVE_PLOT:
            print("\nDisplaying plot... (close window to exit)")
            plt.show()
        elif fig and SAVE_PLOT:
            print(f"\n✓ Plot saved to: {save_path}")
