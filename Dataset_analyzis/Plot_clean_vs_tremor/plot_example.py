"""
Example: Quick plot of clean vs tremor comparison
==================================================

This script shows how to use the plotting functions directly
without interactive prompts.

Modify the parameters below and run to generate plots.
"""

from plot_clean_vs_tremor import plot_all_sensors_comparison, ACTIVITY_NAMES
from pathlib import Path

# ============================================================
# CONFIGURATION - Modify these parameters
# ============================================================

# Choose variants to compare (options: "clean", "mild_mod", "mod_severe")
VARIANT1 = "clean"
VARIANT2 = "mod_severe"

# Optional tremor grade filter for VARIANT2 (set to 1, 2, 3, or 4)
# Set to None to use all available tremor grades
TREMOR_GRADE = 4

# Choose activity (0-11)
# 0: Standing still
# 1: Sitting and relaxing
# 2: Lying down
# 3: Walking
# 4: Climbing stairs
# 5: Waist bends forward
# 6: Frontal elevation of arms
# 7: Knees bending
# 8: Cycling
# 9: Jogging
# 10: Running
# 11: Jump front & back
ACTIVITY_IDX = 3  # Walking

# Window index (which sample of the activity to plot)
WINDOW_IDX = 0

# Sampling frequency
FS = 50

# Save plots? (True/False)
SAVE_PLOTS = True

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Construct full variant names
    variant1_full = f"s2_w2_fs{FS}_tremor_{VARIANT1}"
    variant2_full = f"s2_w2_fs{FS}_tremor_{VARIANT2}"
    
    # Set save directory
    script_dir = Path(__file__).parent
    grade_suffix = f"_score{TREMOR_GRADE}" if TREMOR_GRADE is not None else ""
    if SAVE_PLOTS:
        save_dir = script_dir / f"plots_{VARIANT1}_vs_{VARIANT2}_act{ACTIVITY_IDX}{grade_suffix}"
    else:
        save_dir = None
    
    # Generate plots
    print(f"\n{'='*80}")
    print(f"Comparing: {VARIANT1} vs {VARIANT2}")
    print(f"Activity: {ACTIVITY_NAMES[ACTIVITY_IDX]} (ID: {ACTIVITY_IDX})")
    print(f"Window: {WINDOW_IDX}")
    if TREMOR_GRADE is not None:
        print(f"Tremor grade filter (variant 2): {TREMOR_GRADE}")
    print(f"{'='*80}\n")
    
    figures = plot_all_sensors_comparison(
        variant1_full, 
        variant2_full, 
        ACTIVITY_IDX, 
        WINDOW_IDX, 
        FS, 
        save_dir,
        TREMOR_GRADE,
    )
    
    if not SAVE_PLOTS and figures:
        import matplotlib.pyplot as plt
        print("\nDisplaying plots... (close windows to exit)")
        plt.show()
    elif SAVE_PLOTS:
        print(f"\n✓ All plots saved to: {save_dir}")
