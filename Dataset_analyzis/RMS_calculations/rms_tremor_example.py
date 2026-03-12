"""
Quick RMS Calculation Example for Tremor Dataset
=================================================

This script demonstrates how to calculate RMS values for tremor data
programmatically without interactive prompts.

Modify the parameters below and run:
    python rms_tremor_example.py
"""

from calculate_rms_tremor import (
    get_activity_window, 
    calculate_all_rms, 
    print_rms_results,
    export_rms_to_txt,
    ACTIVITY_NAMES
)


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

# ============================================================
# Configuration - MODIFY THESE PARAMETERS
# ============================================================

# Tremor variant: 'clean', 'mild', or 'severe'
VARIANT = 'severe'

# Activity: either index (0-11) or use ACTIVITY_NAMES
ACTIVITY_IDX = 0  # Standing still

# Window index (0 = first window of this activity)
WINDOW_IDX = 5

# Optional: Filter by score (e.g., 4.0 for severe tremor)
# Set to None to use all scores
SCORE = 4.0

# Whether to save TXT file
SAVE_TXT = True

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("RMS ANALYSIS - TREMOR DATASET (Quick Example)")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Variant: {VARIANT}")
    print(f"  Activity: {ACTIVITY_NAMES[ACTIVITY_IDX]} (index {ACTIVITY_IDX})")
    print(f"  Score filter: {SCORE if SCORE is not None else 'None (all scores)'}")
    print(f"  Window: {WINDOW_IDX}")
    print()
    
    # Load window data
    print("Loading window data...")
    window_data = get_activity_window(VARIANT, ACTIVITY_IDX, WINDOW_IDX, score=SCORE)
    
    if window_data is None:
        print("❌ Failed to load window data")
        exit(1)
    
    print(f"✓ Successfully loaded data\n")
    
    # Calculate RMS
    results = calculate_all_rms(window_data)
    
    # Display results
    print_rms_results(results)
    
    # Export to TXT file
    if SAVE_TXT:
        export_rms_to_txt(results)
    
    print("✓ Complete!")
