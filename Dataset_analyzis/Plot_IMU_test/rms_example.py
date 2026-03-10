"""
Example: Quick RMS calculation for IMU data
============================================

Simple example showing how to calculate RMS values directly.
Modify the parameters below and run.
"""

from calculate_rms import calculate_all_rms, print_rms_results, export_rms_to_txt, calculate_multiple_windows
from plot_imu_data import load_imu_data
from pathlib import Path

# ============================================================
# CONFIGURATION - Modify these parameters
# ============================================================

# CSV file path
SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "IMU_tests" / "Tremor_Session1_RWrist_Calibrated_PC.csv"

# Time window settings
START_TIME = 28.0      # Start time in seconds
WINDOW_DURATION = 2.0  # Window duration in seconds

# Save results to TXT? (True/False)
SAVE_TXT = True

# For multiple windows, set this to a list of start times
# Example: MULTIPLE_WINDOWS = [0, 10, 20, 30, 40]
# Set to None for single window
MULTIPLE_WINDOWS = None  # [0, 30, 60, 90]

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("IMU RMS CALCULATION")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from:\n  {CSV_PATH}")
    df = load_imu_data(CSV_PATH)
    
    if MULTIPLE_WINDOWS:
        # Calculate RMS for multiple windows
        print(f"\nCalculating RMS for {len(MULTIPLE_WINDOWS)} windows...")
        print(f"Start times: {MULTIPLE_WINDOWS}")
        print(f"Window duration: {WINDOW_DURATION} s")
        
        results = calculate_multiple_windows(df, MULTIPLE_WINDOWS, WINDOW_DURATION, SAVE_TXT)
        
        print(f"\n✓ Completed {len(results)} RMS calculations")
    else:
        # Calculate RMS for single window
        print(f"\nCalculating RMS for window:")
        print(f"  Start time: {START_TIME} s")
        print(f"  Duration: {WINDOW_DURATION} s")
        
        results = calculate_all_rms(df, START_TIME, WINDOW_DURATION)
        
        # Display results
        print_rms_results(results)
        
        # Save to TXT if requested
        if SAVE_TXT and results:
            txt_path = SCRIPT_DIR / "rms_results_IMU_test" / f"rms_s{START_TIME:.1f}_d{WINDOW_DURATION:.1f}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            export_rms_to_txt(results, txt_path)
