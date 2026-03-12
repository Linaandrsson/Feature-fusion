"""
Example: Quick plot of tremor-branch vs clean comparison
=========================================================

This script shows how to plot tremor-branch sensors (non-normalized)
against clean baseline without interactive prompts.

Tremor-branch sensors:
- Acc_arm_tremorbranch
- Gyro_arm_tremorbranch

These signals preserve amplitude information (no z-score normalization)
for tremor severity estimation.

Modify the parameters below and run to generate plots.
"""

from plot_tremorbranch_vs_clean import plot_tremorbranch_comparison, ACTIVITY_NAMES
from pathlib import Path

# ============================================================
# CONFIGURATION - Modify these parameters
# ============================================================

# Choose tremor variant to compare against clean (options: "mild_mod", "mod_severe")
TREMOR_VARIANT = "mod_severe"

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
WINDOW_IDX = 6

# Sampling frequency - DETERMINES WHICH DATASET TO LOAD:
# FS = 30  →  loads from: s2_w2_fs30_tremor_*
# FS = 50  →  loads from: s2_w2_fs50_tremor_*
FS = 50  # Options: 30 or 50

# Optional: Filter by tremor severity score (0-4)
# Set to None to use all severities
# 0: No tremor
# 1: Mild
# 2: Mild-Moderate
# 3: Moderate-Severe
# 4: Severe
SEVERITY = 4  # Set to None to disable filter

# Save plots? (True/False)
SAVE_PLOTS = True

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Set save directory
    script_dir = Path(__file__).parent
    if SAVE_PLOTS:
        severity_str = f"_score{SEVERITY}" if SEVERITY is not None else ""
        save_dir = script_dir / f"plots_tremorbranch_clean_vs_{TREMOR_VARIANT}_fs{FS}_act{ACTIVITY_IDX}{severity_str}"
    else:
        save_dir = None
    
    # Determine which dataset will be loaded
    dataset_path = f"s2_w2_fs{FS}_tremor_{TREMOR_VARIANT}"
    
    # Generate plots
    print(f"\n{'='*80}")
    print(f"TREMOR-BRANCH COMPARISON (Non-Normalized)")
    print(f"{'='*80}")
    print(f"Comparing: CLEAN vs {TREMOR_VARIANT.upper()}")
    print(f"Dataset: {dataset_path}")
    print(f"Activity: {ACTIVITY_NAMES[ACTIVITY_IDX]} (ID: {ACTIVITY_IDX})")
    print(f"Window: {WINDOW_IDX}")
    if SEVERITY is not None:
        print(f"Severity filter: Score = {SEVERITY}")
    else:
        print(f"Severity filter: None (all severities)")
    print(f"Note: These signals preserve amplitude for tremor severity estimation")
    print(f"{'='*80}\n")
    
    figures = plot_tremorbranch_comparison(
        TREMOR_VARIANT, 
        ACTIVITY_IDX, 
        WINDOW_IDX, 
        FS, 
        save_dir,
        SEVERITY
    )
    
    if not SAVE_PLOTS and figures:
        import matplotlib.pyplot as plt
        print("\nDisplaying plots... (close windows to exit)")
        plt.show()
    elif SAVE_PLOTS:
        print(f"\n✓ All tremor-branch plots saved to: {save_dir}")
