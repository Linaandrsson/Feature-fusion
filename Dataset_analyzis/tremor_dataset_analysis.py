"""
Tremor Dataset Analysis Script
===============================

Analyzes tremor datasets (clean and parkinson variants) and generates:
  - Statistics on samples, subjects, activities, and tremor scores
  - Distribution of tremor severity scores
  - RMS and frequency statistics per subject
  - Visualizations (heatmaps, bar charts, distributions)

Usage:
    python tremor_dataset_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Activity labels (0-indexed in dataset, 1-indexed in original)
activity_names = {
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

# Tremor severity score labels
score_names = {
    0: "No tremor",
    1: "Mild",
    2: "Mild-Moderate",
    3: "Moderate-Severe",
    4: "Severe"
}

# RMS thresholds for scores (m/s²) - Matches get_tremor_score() logic
# These include gaps between sampling intervals
score_thresholds = {
    0: (0.0, 0.05),      # No tremor
    1: (0.05, 0.25),     # Mild (includes gap 0.10-0.25)
    2: (0.25, 1.10),     # Mild-Moderate (includes gap 0.5-1.10)
    3: (1.10, 4.65),     # Moderate-Severe (includes gap 2.0-4.65)
    4: (4.65, 10.0)      # Severe (4.65 and above)
}

# Sensor list
sensors = ["Acc_arm"]

def load_tremor_txt(file_path: Path) -> dict:
    """
    Load tremor dataset from TXT file.
    
    Returns dict with:
        - X: (N, 300) sensor data (flattened)
        - activity: (N,) activity labels (0-11)
        - subject: (N,) subject IDs (1-10)
        - window_idx: (N,) window indices
        - tremor_freq: (N,) tremor frequency (Hz)
        - tremor_acc_rms: (N,) tremor acc RMS (m/s²)
        - tremor_gyro_rms: (N,) tremor gyro RMS (deg/s)
        - tremor_score: (N,) tremor score (0-4)
    """
    data = np.loadtxt(file_path, delimiter=",")
    
    # Determine channels based on sensor type
    if "ECG" in file_path.name:
        n_channels = 2
        window_length = 100
    else:
        n_channels = 3
        window_length = 100
    
    col_offset = n_channels * window_length
    
    return {
        "X": data[:, :col_offset],
        "activity": data[:, col_offset].astype(int),
        "subject": data[:, col_offset + 1].astype(int),
        "window_idx": data[:, col_offset + 2].astype(int),
        "tremor_freq": data[:, col_offset + 3],
        "tremor_acc_rms": data[:, col_offset + 4],
        "tremor_gyro_rms": data[:, col_offset + 5],
        "tremor_score": data[:, col_offset + 6].astype(int)
    }


def analyze_dataset_variant(variant_name: str, data_dir: Path) -> dict:
    """
    Analyze one dataset variant (clean or parkinson).
    
    Returns statistics dictionary.
    """
    variant_path = data_dir / variant_name
    
    if not variant_path.exists():
        print(f"Warning: {variant_path} not found!")
        return None
    
    print(f"\nAnalyzing {variant_name}...")
    
    # Load first sensor to get labels (all sensors have same labels)
    sensor_file = variant_path / "Acc_arm.txt"
    if not sensor_file.exists():
        print(f"Warning: {sensor_file} not found!")
        return None
    
    data = load_tremor_txt(sensor_file)
    
    N = len(data["activity"])
    
    # Basic statistics
    stats = {
        "variant": variant_name,
        "total_samples": N,
        "total_seconds": N * 2,  # 2-second windows
        "total_minutes": N * 2 / 60,
        "num_subjects": len(np.unique(data["subject"])),
        "subjects": np.unique(data["subject"]).tolist(),
    }
    
    # Activity distribution
    activity_counts = {}
    for act in range(12):
        activity_counts[act] = np.sum(data["activity"] == act)
    stats["activity_counts"] = activity_counts
    
    # Score distribution
    score_counts = {}
    for score in range(5):
        score_counts[score] = np.sum(data["tremor_score"] == score)
    stats["score_counts"] = score_counts
    
    # Per-subject statistics
    subject_stats = []
    for subj in np.unique(data["subject"]):
        mask = data["subject"] == subj
        subj_data = {key: val[mask] for key, val in data.items()}
        
        # Get dominant score (most common non-zero score)
        scores = subj_data["tremor_score"]
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            dominant_score = int(np.bincount(non_zero_scores).argmax())
        else:
            dominant_score = 0
        
        subject_stats.append({
            "subject": int(subj),
            "n_samples": int(mask.sum()),
            "dominant_score": dominant_score,
            "mean_acc_rms": float(subj_data["tremor_acc_rms"].mean()),
            "std_acc_rms": float(subj_data["tremor_acc_rms"].std()),
            "mean_gyro_rms": float(subj_data["tremor_gyro_rms"].mean()),
            "mean_freq": float(subj_data["tremor_freq"].mean()),
            "score_0_count": int(np.sum(scores == 0)),
            "score_1_count": int(np.sum(scores == 1)),
            "score_2_count": int(np.sum(scores == 2)),
            "score_3_count": int(np.sum(scores == 3)),
            "score_4_count": int(np.sum(scores == 4)),
        })
    
    stats["subject_stats"] = subject_stats
    
    # RMS statistics
    stats["rms_statistics"] = {
        "acc_rms_mean": float(data["tremor_acc_rms"].mean()),
        "acc_rms_std": float(data["tremor_acc_rms"].std()),
        "acc_rms_min": float(data["tremor_acc_rms"].min()),
        "acc_rms_max": float(data["tremor_acc_rms"].max()),
        "gyro_rms_mean": float(data["tremor_gyro_rms"].mean()),
        "gyro_rms_std": float(data["tremor_gyro_rms"].std()),
        "gyro_rms_min": float(data["tremor_gyro_rms"].min()),
        "gyro_rms_max": float(data["tremor_gyro_rms"].max()),
        "freq_mean": float(data["tremor_freq"].mean()),
        "freq_std": float(data["tremor_freq"].std()),
    }
    
    # Store raw data for plotting
    stats["raw_data"] = data
    
    return stats


def save_summary_report(stats_clean, stats_parkinson, output_file: Path):
    """
    Save text summary to file.
    """
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TREMOR DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Clean dataset
        if stats_clean:
            f.write(f"Dataset: {stats_clean['variant']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {stats_clean['total_samples']}\n")
            f.write(f"Total Time: {stats_clean['total_minutes']:.2f} minutes\n")
            f.write(f"Number of Subjects: {stats_clean['num_subjects']}\n")
            f.write(f"Subjects: {stats_clean['subjects']}\n\n")
            
            f.write("Score Distribution:\n")
            for score in range(5):
                count = stats_clean['score_counts'][score]
                pct = 100 * count / stats_clean['total_samples']
                f.write(f"  Score {score} ({score_names[score]:16s}): {count:6d} samples ({pct:5.1f}%)\n")
            f.write("\n")
            
            f.write("RMS Statistics:\n")
            f.write(f"  Acc RMS:  {stats_clean['rms_statistics']['acc_rms_mean']:.4f} ± {stats_clean['rms_statistics']['acc_rms_std']:.4f} m/s²\n")
            f.write(f"            Range: [{stats_clean['rms_statistics']['acc_rms_min']:.4f}, {stats_clean['rms_statistics']['acc_rms_max']:.4f}]\n")
            f.write(f"  Gyro RMS: {stats_clean['rms_statistics']['gyro_rms_mean']:.4f} ± {stats_clean['rms_statistics']['gyro_rms_std']:.4f} deg/s\n")
            f.write(f"  Freq:     {stats_clean['rms_statistics']['freq_mean']:.2f} ± {stats_clean['rms_statistics']['freq_std']:.2f} Hz\n")
            f.write("\n\n")
        
        # Parkinson dataset
        if stats_parkinson:
            f.write(f"Dataset: {stats_parkinson['variant']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {stats_parkinson['total_samples']}\n")
            f.write(f"Total Time: {stats_parkinson['total_minutes']:.2f} minutes\n")
            f.write(f"Number of Subjects: {stats_parkinson['num_subjects']}\n")
            f.write(f"Subjects: {stats_parkinson['subjects']}\n\n")
            
            f.write("Score Distribution:\n")
            for score in range(5):
                count = stats_parkinson['score_counts'][score]
                pct = 100 * count / stats_parkinson['total_samples']
                f.write(f"  Score {score} ({score_names[score]:16s}): {count:6d} samples ({pct:5.1f}%)\n")
            f.write("\n")
            
            f.write("RMS Statistics:\n")
            f.write(f"  Acc RMS:  {stats_parkinson['rms_statistics']['acc_rms_mean']:.4f} ± {stats_parkinson['rms_statistics']['acc_rms_std']:.4f} m/s²\n")
            f.write(f"            Range: [{stats_parkinson['rms_statistics']['acc_rms_min']:.4f}, {stats_parkinson['rms_statistics']['acc_rms_max']:.4f}]\n")
            f.write(f"  Gyro RMS: {stats_parkinson['rms_statistics']['gyro_rms_mean']:.4f} ± {stats_parkinson['rms_statistics']['gyro_rms_std']:.4f} deg/s\n")
            f.write(f"  Freq:     {stats_parkinson['rms_statistics']['freq_mean']:.2f} ± {stats_parkinson['rms_statistics']['freq_std']:.2f} Hz\n")
            f.write("\n")
            
            f.write("Per-Subject Statistics:\n")
            f.write("-" * 80 + "\n")
            df = pd.DataFrame(stats_parkinson["subject_stats"])
            df_display = df[["subject", "n_samples", "dominant_score", "mean_acc_rms", "mean_freq"]]
            df_display.columns = ["Subject", "Samples", "Score", "Acc RMS", "Freq (Hz)"]
            f.write(df_display.to_string(index=False))
            f.write("\n\n")
            
            f.write("Subject-Score Distribution:\n")
            f.write("-" * 80 + "\n")
            df_scores = df[["subject", "score_0_count", "score_1_count", "score_2_count", "score_3_count", "score_4_count"]]
            df_scores.columns = ["Subj", "Score 0", "Score 1", "Score 2", "Score 3", "Score 4"]
            f.write(df_scores.to_string(index=False))
            f.write("\n")
    
    print(f"\nSummary report saved to: {output_file}")


def plot_score_distribution(stats_clean, stats_mild, stats_severe, output_dir: Path):
    """
    Plot score distribution comparison for all three variants.
    """
    all_stats = [stats_clean, stats_mild, stats_severe]
    valid_stats = [s for s in all_stats if s is not None]
    
    if not valid_stats:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, stats in enumerate([stats_clean, stats_mild, stats_severe]):
        if stats is None:
            continue
            
        ax = axes[idx]
        scores = list(range(5))
        counts = [stats['score_counts'][s] for s in scores]
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        
        bars = ax.bar(scores, counts, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Tremor Score', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        
        # Shorter title
        variant_short = stats["variant"].replace('s2_w2_tremor_', '')
        ax.set_title(f'{variant_short}: Score Distribution', fontsize=13, fontweight='bold')
        ax.set_xticks(scores)
        ax.set_xticklabels([f'{s}\n{score_names[s]}' for s in scores], fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = output_dir / "tremor_score_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()


def plot_subject_heatmap(stats, output_dir: Path):
    """
    Plot heatmap of score distribution per subject.
    """
    if not stats:
        return
    
    df = pd.DataFrame(stats["subject_stats"])
    
    # Create heatmap data: subjects x scores
    heatmap_data = df[["score_0_count", "score_1_count", "score_2_count", 
                        "score_3_count", "score_4_count"]].values
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(5))
    ax.set_yticks(range(len(df)))
    ax.set_xticklabels([f'Score {i}\n{score_names[i]}' for i in range(5)], fontsize=10)
    ax.set_yticklabels([f'Subject {s}' for s in df["subject"]], fontsize=10)
    ax.set_xlabel('Tremor Score', fontsize=12)
    ax.set_ylabel('Subject', fontsize=12)
    ax.set_title(f'{stats["variant"]}: Samples per Subject and Score', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Samples', rotation=270, labelpad=20)
    
    # Text annotations
    threshold = (heatmap_data.max() + heatmap_data.min()) / 2
    for i in range(len(df)):
        for j in range(5):
            text_color = "white" if heatmap_data[i, j] > threshold else "black"
            ax.text(j, i, int(heatmap_data[i, j]),
                   ha="center", va="center", color=text_color, 
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_file = output_dir / f"tremor_subject_score_heatmap_{stats['variant']}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()


def plot_rms_distributions(stats, output_dir: Path):
    """
    Plot RMS and frequency distributions.
    """
    if not stats:
        return
    
    data = stats["raw_data"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Acc RMS histogram
    ax = axes[0, 0]
    ax.hist(data["tremor_acc_rms"], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Acc RMS (m/s²)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Accelerometer RMS Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add vertical lines for score thresholds
    for score, (low, high) in score_thresholds.items():
        if score > 0:
            ax.axvline(low, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Gyro RMS histogram
    ax = axes[0, 1]
    ax.hist(data["tremor_gyro_rms"], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Gyro RMS (deg/s)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Gyroscope RMS Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Frequency histogram
    ax = axes[1, 0]
    ax.hist(data["tremor_freq"], bins=30, color='green', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Tremor Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Tremor Frequency Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Acc RMS vs Gyro RMS scatter
    ax = axes[1, 1]
    scores = data["tremor_score"]
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    for score in range(5):
        mask = scores == score
        ax.scatter(data["tremor_acc_rms"][mask], data["tremor_gyro_rms"][mask],
                  c=colors[score], label=f'Score {score}', alpha=0.5, s=10)
    ax.set_xlabel('Acc RMS (m/s²)', fontsize=11)
    ax.set_ylabel('Gyro RMS (deg/s)', fontsize=11)
    ax.set_title('Acc RMS vs Gyro RMS (by Score)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f"tremor_rms_distributions_{stats['variant']}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()


def plot_subject_severity(stats, output_dir: Path):
    """
    Plot subject-wise severity overview.
    """
    if not stats:
        return
    
    df = pd.DataFrame(stats["subject_stats"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot: Mean Acc RMS per subject
    ax = axes[0]
    colors_gradient = plt.cm.Reds(np.linspace(0.3, 1, len(df)))
    bars = ax.bar(df["subject"], df["mean_acc_rms"], color=colors_gradient, 
                  edgecolor='black', alpha=0.8)
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Mean Acc RMS (m/s²)', fontsize=12)
    ax.set_title('Mean Tremor Severity per Subject', fontsize=13, fontweight='bold')
    ax.set_xticks(df["subject"])
    ax.grid(axis='y', alpha=0.3)
    
    # Add score labels on bars
    for i, (subj, rms, score) in enumerate(zip(df["subject"], df["mean_acc_rms"], df["dominant_score"])):
        ax.text(subj, rms + 0.05, f'S{score}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Bar plot: Mean frequency per subject
    ax = axes[1]
    colors_gradient = plt.cm.Blues(np.linspace(0.3, 1, len(df)))
    bars = ax.bar(df["subject"], df["mean_freq"], color=colors_gradient, 
                  edgecolor='black', alpha=0.8)
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Mean Tremor Frequency (Hz)', fontsize=12)
    ax.set_title('Mean Tremor Frequency per Subject', fontsize=13, fontweight='bold')
    ax.set_xticks(df["subject"])
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([3, 7])
    
    # Add frequency values on bars
    for subj, freq in zip(df["subject"], df["mean_freq"]):
        ax.text(subj, freq + 0.05, f'{freq:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = output_dir / f"tremor_subject_severity_{stats['variant']}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()


def plot_subject_score_comparison(all_stats, output_dir: Path):
    """
    Plot per-subject score distribution across all variants.
    Each subject gets a subplot showing score distribution for each variant.
    """
    # Filter out None stats
    valid_stats = [s for s in all_stats if s is not None]
    if not valid_stats:
        return
    
    # Get all subjects (from first valid stats)
    df_first = pd.DataFrame(valid_stats[0]["subject_stats"])
    subjects = sorted(df_first["subject"].unique())
    
    # Create subplots: 2 rows x 5 columns for 10 subjects
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Colors for each variant
    variant_colors = {
        's2_w2_tremor_clean': 'lightblue',
        's2_w2_tremor_parkinson_mild': 'orange',
        's2_w2_tremor_parkinson_severe': 'red'
    }
    
    # For each subject
    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        
        # Prepare data for grouped bar chart
        x = np.arange(5)  # 5 scores: 0-4
        width = 0.25  # bar width
        
        # Plot bars for each variant
        for i, stats in enumerate(valid_stats):
            df = pd.DataFrame(stats["subject_stats"])
            subj_data = df[df["subject"] == subject]
            
            if len(subj_data) == 0:
                continue
            
            scores = [
                int(subj_data["score_0_count"].values[0]),
                int(subj_data["score_1_count"].values[0]),
                int(subj_data["score_2_count"].values[0]),
                int(subj_data["score_3_count"].values[0]),
                int(subj_data["score_4_count"].values[0])
            ]
            
            variant_name = stats["variant"]
            color = variant_colors.get(variant_name, f'C{i}')
            
            # Offset bars for each variant
            offset = (i - len(valid_stats)/2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=variant_name.replace('s2_w2_tremor_', ''),
                   color=color, edgecolor='black', alpha=0.7)
        
        ax.set_title(f'Subject {subject}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Score', fontsize=9)
        ax.set_ylabel('Samples', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle('Score Distribution per Subject across Variants', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plot_file = output_dir / "tremor_subject_comparison_all_variants.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()


def main():
    """
    Main analysis function.
    """
    print("="*80)
    print("TREMOR DATASET ANALYSIS")
    print("="*80)
    
    # Paths
    data_dir = Path("data/Tremor_datagenerator_files")
    output_dir = Path("Dataset_analyzis")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze all variants
    stats_clean = analyze_dataset_variant("s2_w2_tremor_clean", data_dir)
    stats_parkinson_mild = analyze_dataset_variant("s2_w2_tremor_parkinson_mild", data_dir)
    stats_parkinson_severe = analyze_dataset_variant("s2_w2_tremor_parkinson_severe", data_dir)
    
    # Collect all stats
    all_stats = [stats_clean, stats_parkinson_mild, stats_parkinson_severe]
    valid_stats = [s for s in all_stats if s is not None]
    
    if not valid_stats:
        print("\nError: No datasets found!")
        return
    
    # Save summary report (keeping old function for compatibility)
    output_file = output_dir / "tremor_dataset_summary.txt"
    save_summary_report(stats_clean, stats_parkinson_mild, output_file)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Score distribution comparison (all three variants)
    plot_score_distribution(stats_clean, stats_parkinson_mild, stats_parkinson_severe, output_dir)
    
    # Per-subject comparison across all variants
    print("\nGenerating per-subject comparison plot...")
    plot_subject_score_comparison(all_stats, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
