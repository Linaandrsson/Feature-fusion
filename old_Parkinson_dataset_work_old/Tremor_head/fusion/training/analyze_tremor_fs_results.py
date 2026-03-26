"""
analyze_tremor_fs_results.py

Analyze and rank tremor FS combination study results by experiment ID.

Usage:
    1. Set EXPERIMENT_ID variable below (recommended for VS Code)
    2. Run the script with the play button in VS Code
    
    OR use command line:
    python analyze_tremor_fs_results.py [experiment_id]
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - Set your experiment ID here!
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_ID = "tremor_fs_study_k2_fs10_20_30_40_50_13-30-35"  # Set to None to see available experiments
                      # Example: "tremor_fs_study_k2_fs10_30_50_14-25-30"
# ═══════════════════════════════════════════════════════════════


def load_fs_results(log_file: Path) -> List[Dict]:
    """Load all results from the FS study log file."""
    results = []
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return results
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}")
                    continue
    
    return results


def get_available_experiments(results: List[Dict]) -> Dict[str, int]:
    """Get list of unique experiment IDs and their result counts."""
    experiments = defaultdict(int)
    
    for result in results:
        exp_id = result.get('experiment_id', 'unknown')
        experiments[exp_id] += 1
    
    return dict(experiments)


def filter_by_experiment(results: List[Dict], experiment_id: str) -> List[Dict]:
    """Filter results by experiment ID."""
    filtered = []
    
    for result in results:
        exp_id = result.get('experiment_id', '')
        if exp_id == experiment_id:
            filtered.append(result)
    
    return filtered


def rank_results(results: List[Dict], metric: str = 'test_f1') -> List[Dict]:
    """Sort results by specified metric (descending)."""
    return sorted(results, key=lambda x: x.get(metric, 0), reverse=True)


def format_fs_map(fs_map: Dict[str, int]) -> str:
    """Format FS map for display."""
    return ", ".join(f"{sensor}:fs{fs}" for sensor, fs in fs_map.items())


def generate_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    """Generate human-readable text report."""
    
    if not ranked_results:
        print("No results to report")
        return
    
    # Get metadata from first result
    first = ranked_results[0]
    seed = first.get('seed', 'unknown')
    run_timestamp = first.get('run_timestamp', 'unknown')
    sensors = first.get('sensors', [])
    num_sensors = len(sensors)
    tremor_types = first.get('tremor_types_merged', [])
    
    # Calculate statistics
    test_f1s = [r['test_f1'] for r in ranked_results]
    test_accs = [r['test_accuracy'] for r in ranked_results]
    val_f1s = [r['val_f1'] for r in ranked_results]
    val_accs = [r['val_accuracy'] for r in ranked_results]
    
    mean_test_f1 = sum(test_f1s) / len(test_f1s)
    max_test_f1 = max(test_f1s)
    min_test_f1 = min(test_f1s)
    
    mean_test_acc = sum(test_accs) / len(test_accs)
    max_test_acc = max(test_accs)
    min_test_acc = min(test_accs)
    
    with open(output_file, 'w') as f:
        # Header
        f.write("="*180 + "\n")
        f.write(f"TREMOR FS COMBINATION STUDY RESULTS - RANKED BY TEST F1-SCORE\n")
        f.write("="*180 + "\n\n")
        
        # Metadata
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Sensors: {', '.join(sensors)} ({num_sensors} sensors)\n")
        f.write(f"Tremor types merged: {', '.join(tremor_types)}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total combinations tested: {len(ranked_results)}\n")
        f.write("\n")
        
        # Statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*180 + "\n")
        f.write("Test F1-score:\n")
        f.write(f"  Mean:   {mean_test_f1:.4f}\n")
        f.write(f"  Max:    {max_test_f1:.4f}\n")
        f.write(f"  Min:    {min_test_f1:.4f}\n")
        f.write(f"  Range:  {max_test_f1 - min_test_f1:.4f}\n")
        f.write("\n")
        f.write("Test Accuracy:\n")
        f.write(f"  Mean:   {mean_test_acc:.4f}\n")
        f.write(f"  Max:    {max_test_acc:.4f}\n")
        f.write(f"  Min:    {min_test_acc:.4f}\n")
        f.write(f"  Range:  {max_test_acc - min_test_acc:.4f}\n")
        f.write("\n")
        
        # Rankings
        f.write("RANKED FS COMBINATIONS (by test F1-score)\n")
        f.write("="*180 + "\n\n")
        
        # Column headers with proper spacing
        f.write(f"{'Rank':<6}    ")
        f.write(f"{'Val F1':<10}    ")
        f.write(f"{'Val Acc':<10}    ")
        f.write(f"{'Test F1':<10}    ")
        f.write(f"{'Test Acc':<10}    ")
        f.write(f"{'FS Map'}\n")
        f.write("-"*180 + "\n")
        
        # Results
        for idx, result in enumerate(ranked_results, 1):
            val_f1 = result.get('val_f1', 0)
            val_acc = result.get('val_accuracy', 0)
            test_f1 = result.get('test_f1', 0)
            test_acc = result.get('test_accuracy', 0)
            fs_map = result.get('fs_map', {})
            fs_str = format_fs_map(fs_map)
            
            f.write(f"{idx:<6}    ")
            f.write(f"{val_f1:.4f}        ")
            f.write(f"{val_acc:.4f}        ")
            f.write(f"{test_f1:.4f}        ")
            f.write(f"{test_acc:.4f}        ")
            f.write(f"{fs_str}\n")
        
        # Footer
        f.write("\n" + "="*180 + "\n")
        f.write(f"Report generated: {run_timestamp}\n")
        f.write("="*180 + "\n")
    
    print(f"✓ Text report saved to: {output_file}")


def generate_json_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    """Generate JSON report with full details."""
    
    if not ranked_results:
        print("No results to report")
        return
    
    # Calculate statistics
    test_f1s = [r['test_f1'] for r in ranked_results]
    test_accs = [r['test_accuracy'] for r in ranked_results]
    val_f1s = [r['val_f1'] for r in ranked_results]
    val_accs = [r['val_accuracy'] for r in ranked_results]
    
    report = {
        "experiment_id": experiment_id,
        "metadata": {
            "sensors": ranked_results[0].get('sensors'),
            "tremor_types_merged": ranked_results[0].get('tremor_types_merged'),
            "seed": ranked_results[0].get('seed'),
            "run_timestamp": ranked_results[0].get('run_timestamp'),
            "total_combinations": len(ranked_results),
        },
        "statistics": {
            "test_f1": {
                "mean": sum(test_f1s) / len(test_f1s),
                "max": max(test_f1s),
                "min": min(test_f1s),
                "range": max(test_f1s) - min(test_f1s),
            },
            "test_accuracy": {
                "mean": sum(test_accs) / len(test_accs),
                "max": max(test_accs),
                "min": min(test_accs),
                "range": max(test_accs) - min(test_accs),
            },
            "val_f1": {
                "mean": sum(val_f1s) / len(val_f1s),
                "max": max(val_f1s),
                "min": min(val_f1s),
                "range": max(val_f1s) - min(val_f1s),
            },
            "val_accuracy": {
                "mean": sum(val_accs) / len(val_accs),
                "max": max(val_accs),
                "min": min(val_accs),
                "range": max(val_accs) - min(val_accs),
            },
        },
        "best_combination": ranked_results[0],
        "worst_combination": ranked_results[-1],
        "ranked_results": ranked_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ JSON report saved to: {output_file}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    log_file = script_dir.parent / "logs" / "tremor_fs_study.jsonl"
    output_dir = script_dir.parent / "logs" / "tremor_fs_study_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TREMOR FS COMBINATION STUDY ANALYZER")
    print("="*80)
    
    # Load all results
    print(f"\nLoading results from: {log_file}")
    all_results = load_fs_results(log_file)
    
    if not all_results:
        print("No results found in log file.")
        return
    
    print(f"Loaded {len(all_results)} results")
    
    # Get available experiments
    experiments = get_available_experiments(all_results)
    
    # Get experiment ID (priority: config variable > command line > interactive prompt)
    experiment_id = None
    
    # 1. Check config variable first
    if EXPERIMENT_ID is not None:
        experiment_id = EXPERIMENT_ID
        print(f"\nUsing experiment ID from configuration: {experiment_id}")
    
    # 2. Check command line argument
    elif len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        print(f"\nUsing experiment ID from command line: {experiment_id}")
    
    # 3. Show available experiments
    else:
        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        print("\nSet EXPERIMENT_ID at the top of this script to analyze results.")
        print("\nAvailable experiments:")
        print("-"*80)
        for exp_id, count in sorted(experiments.items()):
            print(f"  {exp_id}: {count} combinations")
        print("-"*80)
        print("\nExample: EXPERIMENT_ID = \"tremor_fs_study_k2_fs10_30_50_14-25-30\"")
        return
    
    # Safety check
    if not experiment_id or not experiment_id.strip():
        print("Error: Empty experiment ID provided")
        return
    
    # Filter and rank results
    print(f"\nFiltering results for experiment: {experiment_id}")
    filtered_results = filter_by_experiment(all_results, experiment_id)
    
    if not filtered_results:
        print(f"Error: No results found for experiment ID: {experiment_id}")
        print("\nAvailable experiment IDs:")
        for exp_id in sorted(experiments.keys()):
            print(f"  - {exp_id}")
        return
    
    print(f"Found {len(filtered_results)} combinations")
    
    # Rank by test F1-score (primary metric for tremor classification)
    ranked_results = rank_results(filtered_results, metric='test_f1')
    
    # Generate reports
    print(f"\nGenerating reports...")
    
    # Text report
    txt_output = output_dir / f"results_{experiment_id}.txt"
    generate_report(experiment_id, ranked_results, txt_output)
    
    # JSON report
    json_output = output_dir / f"results_{experiment_id}.json"
    generate_json_report(experiment_id, ranked_results, json_output)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    best = ranked_results[0]
    print(f"\nBest combination (Test F1: {best['test_f1']:.4f}, Test Acc: {best['test_accuracy']:.4f}):")
    print(f"  FS map: {format_fs_map(best['fs_map'])}")
    
    worst = ranked_results[-1]
    print(f"\nWorst combination (Test F1: {worst['test_f1']:.4f}, Test Acc: {worst['test_accuracy']:.4f}):")
    print(f"  FS map: {format_fs_map(worst['fs_map'])}")
    
    print(f"\nReports saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
