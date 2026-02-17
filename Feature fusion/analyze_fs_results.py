"""
analyze_fs_results.py

Analyze and rank FS combination study results by experiment ID.

Usage:
    1. Set EXPERIMENT_ID variable below (recommended for VS Code)
    2. Run the script with the play button in VS Code
    
    OR use command line:
    python analyze_fs_results.py [experiment_id]
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - Set your experiment ID here!
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_ID = "fs_study_k3_fs10_30_08-12-11"  # Set to None to see available experiments
                      # Example: "fs_study_k3_fs10_20_30_50_16-30-45"
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


def rank_results(results: List[Dict]) -> List[Dict]:
    """Sort results by test accuracy (descending)."""
    return sorted(results, key=lambda x: x.get('test_accuracy', 0), reverse=True)


def format_fs_map(fs_map: Dict[str, int]) -> str:
    """Format FS map for display."""
    return ", ".join(f"{sensor}:{fs}" for sensor, fs in fs_map.items())


def generate_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    """Generate human-readable text report."""
    
    if not ranked_results:
        print("No results to report")
        return
    
    # Get metadata from first result
    first = ranked_results[0]
    train_mode = first.get('train_mode', 'unknown')
    test_mode = first.get('test_mode', 'unknown')
    fusion_mode = first.get('fusion_mode', 'unknown')
    dataset_config = first.get('dataset_config', 'unknown')
    seed = first.get('seed', 'unknown')
    run_timestamp = first.get('run_timestamp', 'unknown')
    sensors = first.get('sensors', [])
    num_sensors = len(sensors)
    
    # Determine corruption status
    if train_mode == "clean" and test_mode == "clean":
        corruption_status = "NO (clean only)"
    elif train_mode == "corrupted" and test_mode == "corrupted":
        corruption_status = "YES (both train and test)"
    elif train_mode == "clean" and test_mode == "corrupted":
        corruption_status = "TEST ONLY (clean train, corrupted test)"
    elif train_mode == "corrupted" and test_mode == "clean":
        corruption_status = "TRAIN ONLY (corrupted train, clean test)"
    else:
        corruption_status = f"MIXED (train={train_mode}, test={test_mode})"
    
    # Calculate statistics
    test_accs = [r['test_accuracy'] for r in ranked_results]
    mean_acc = sum(test_accs) / len(test_accs)
    max_acc = max(test_accs)
    min_acc = min(test_accs)
    
    with open(output_file, 'w') as f:
        # Header
        f.write("="*170 + "\n")
        f.write(f"FS COMBINATION STUDY RESULTS - RANKED BY TEST ACCURACY\n")
        f.write("="*170 + "\n\n")
        
        # Metadata (matching ablation format)
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Number of sensors: {num_sensors}\n")
        f.write(f"Fusion mode: {fusion_mode}\n")
        f.write(f"Corruption: {corruption_status}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total combinations tested: {len(ranked_results)}\n")
        f.write("\n")
        
        # Statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*170 + "\n")
        f.write(f"  Mean test accuracy:  {mean_acc:.4f}\n")
        f.write(f"  Max test accuracy:   {max_acc:.4f}\n")
        f.write(f"  Min test accuracy:   {min_acc:.4f}\n")
        f.write(f"  Range:               {max_acc - min_acc:.4f}\n")
        f.write("\n")
        
        # Rankings
        f.write("RANKED FS COMBINATIONS (by test accuracy)\n")
        f.write("="*170 + "\n\n")
        
        # Column headers with proper spacing
        f.write(f"{'Rank':<6}    ")
        f.write(f"{'Val Acc':<10}    ")
        f.write(f"{'Test Acc':<10}    ")
        f.write(f"{'FS Map'}\n")
        f.write("-"*170 + "\n")
        
        # Results
        for idx, result in enumerate(ranked_results, 1):
            val_acc = result.get('val_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            fs_map = result.get('fs_map', {})
            fs_str = format_fs_map(fs_map)
            
            f.write(f"{idx:<6}    ")
            f.write(f"{val_acc:.4f}    ")
            f.write(f"{test_acc:.4f}    ")
            f.write(f"{fs_str}\n")
        
        # Footer
        f.write("\n" + "="*170 + "\n")
        f.write(f"Report generated: {ranked_results[0].get('run_timestamp', 'unknown')}\n")
        f.write("="*170 + "\n")
    
    print(f"✓ Text report saved to: {output_file}")


def generate_json_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    """Generate JSON report with full details."""
    
    if not ranked_results:
        print("No results to report")
        return
    
    # Calculate statistics
    test_accs = [r['test_accuracy'] for r in ranked_results]
    
    report = {
        "experiment_id": experiment_id,
        "metadata": {
            "train_mode": ranked_results[0].get('train_mode'),
            "test_mode": ranked_results[0].get('test_mode'),
            "fusion_mode": ranked_results[0].get('fusion_mode'),
            "dataset_config": ranked_results[0].get('dataset_config'),
            "seed": ranked_results[0].get('seed'),
            "run_timestamp": ranked_results[0].get('run_timestamp'),
            "total_combinations": len(ranked_results),
        },
        "statistics": {
            "mean_test_accuracy": sum(test_accs) / len(test_accs),
            "max_test_accuracy": max(test_accs),
            "min_test_accuracy": min(test_accs),
            "range": max(test_accs) - min(test_accs),
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
    log_file = script_dir.parent / "fusion_logs" / "fs_combo_study.jsonl"
    base_output_dir = script_dir.parent / "fusion_logs" / "fs_study_reports"
    base_output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("FS COMBINATION STUDY ANALYZER")
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
        print("\nExample: EXPERIMENT_ID = \"fs_study_k3_fs10_20_30_50_16-30-45\"")
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
    
    # Rank by test accuracy
    ranked_results = rank_results(filtered_results)
    
    # Determine train/test mode and fusion mode from first result
    first_result = ranked_results[0]
    train_mode = first_result.get('train_mode', 'clean')
    test_mode = first_result.get('test_mode', 'clean')
    fusion_mode = first_result.get('fusion_mode', 'concat')
    
    # Map fusion_mode to directory name
    fusion_dir = "vanilla" if fusion_mode == "concat" else "gated"
    
    # Create nested directory structure: train-test_mode/fusion_type/
    scenario_dir = f"{train_mode}_train-{test_mode}_test"
    output_dir = base_output_dir / scenario_dir / fusion_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    print(f"\nGenerating reports for scenario: {scenario_dir}/{fusion_dir}...")
    
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
    print(f"\nBest combination (Test Acc: {best['test_accuracy']:.4f}):")
    print(f"  FS map: {format_fs_map(best['fs_map'])}")
    
    worst = ranked_results[-1]
    print(f"\nWorst combination (Test Acc: {worst['test_accuracy']:.4f}):")
    print(f"  FS map: {format_fs_map(worst['fs_map'])}")
    
    print(f"\nReports saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
