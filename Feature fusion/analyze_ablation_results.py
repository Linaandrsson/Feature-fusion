"""
analyze_ablation_results.py

Analyze and rank ablation study results by experiment ID.

Usage:
    1. Set EXPERIMENT_ID variable below (recommended for VS Code)
    2. Run the script with the play button in VS Code
    
    OR use command line:
    python analyze_ablation_results.py [experiment_id]
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - Set your experiment ID here!
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_ID = "ablation_k7_2026-02-16_15-35-41" # Set to None to see available experiments
                      # Example: "ablation_k7_2026-02-16_14-30-52"
# ═══════════════════════════════════════════════════════════════


def load_ablation_results(log_file: Path) -> List[Dict]:
    """Load all results from the ablation log file."""
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
        exp_id = result.get('experiment_id', result.get('experiment', 'unknown'))
        experiments[exp_id] += 1
    
    return dict(experiments)


def filter_by_experiment(results: List[Dict], experiment_id: str) -> List[Dict]:
    """Filter results by experiment ID."""
    filtered = []
    
    for result in results:
        exp_id = result.get('experiment_id', result.get('experiment', ''))
        if exp_id == experiment_id:
            filtered.append(result)
    
    return filtered


def rank_results(results: List[Dict]) -> List[Dict]:
    """Sort results by test accuracy (descending)."""
    return sorted(results, key=lambda x: x.get('test_accuracy', 0), reverse=True)


def format_sensor_list(sensors: List[str], max_length: int = 60) -> str:
    """Format sensor list for display."""
    sensor_str = ", ".join(sensors)
    if len(sensor_str) > max_length:
        return sensor_str[:max_length-3] + "..."
    return sensor_str


def generate_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    """Generate a text report with ranked results."""
    
    if not ranked_results:
        print(f"No results found for experiment: {experiment_id}")
        return
    
    # Extract metadata from first result
    first = ranked_results[0]
    num_sensors = first.get('num_sensors', 'unknown')
    fusion_mode = first.get('fusion_mode', 'unknown')
    use_corruption = first.get('use_corruption', False)
    seed = first.get('seed', 'unknown')
    
    with open(output_file, 'w') as f:
        # Header
        f.write("="*130 + "\n")
        f.write("SENSOR ABLATION RESULTS - RANKED BY TEST ACCURACY\n")
        f.write("="*130 + "\n\n")
        
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Number of sensors per combination: {num_sensors}\n")
        f.write(f"Fusion mode: {fusion_mode}\n")
        f.write(f"Corruption: {'YES' if use_corruption else 'NO (clean only)'}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total combinations tested: {len(ranked_results)}\n\n")
        
        # Best result
        best = ranked_results[0]
        f.write("-"*130 + "\n")
        f.write("BEST COMBINATION:\n")
        f.write("-"*130 + "\n")
        f.write(f"Sensors:       {', '.join(best['sensors'])}\n")
        if best.get('ablated_sensors'):
            f.write(f"Ablated:       {', '.join(best['ablated_sensors'])}\n")
        f.write(f"Test Accuracy: {best['test_accuracy']:.4f}\n")
        f.write(f"Val Accuracy:  {best['val_accuracy']:.4f}\n")
        f.write(f"Val Loss:      {best['val_loss']:.4f}\n\n")
        
        # Worst result
        worst = ranked_results[-1]
        f.write("-"*130 + "\n")
        f.write("WORST COMBINATION:\n")
        f.write("-"*130 + "\n")
        f.write(f"Sensors:       {', '.join(worst['sensors'])}\n")
        if worst.get('ablated_sensors'):
            f.write(f"Ablated:       {', '.join(worst['ablated_sensors'])}\n")
        f.write(f"Test Accuracy: {worst['test_accuracy']:.4f}\n")
        f.write(f"Val Accuracy:  {worst['val_accuracy']:.4f}\n")
        f.write(f"Val Loss:      {worst['val_loss']:.4f}\n\n")
        
        # Statistics
        test_accs = [r['test_accuracy'] for r in ranked_results]
        f.write("-"*130 + "\n")
        f.write("STATISTICS:\n")
        f.write("-"*130 + "\n")
        f.write(f"Mean Test Accuracy:   {sum(test_accs)/len(test_accs):.4f}\n")
        f.write(f"Max Test Accuracy:    {max(test_accs):.4f}\n")
        f.write(f"Min Test Accuracy:    {min(test_accs):.4f}\n")
        f.write(f"Range:                {max(test_accs) - min(test_accs):.4f}\n\n")
        
        # Full ranking
        f.write("="*170 + "\n")
        f.write("COMPLETE RANKING:\n")
        f.write("="*170 + "\n\n")
        f.write(f"{'Rank':<6}   {'Test Acc':<12}   {'Val Acc':<12}   {'Val Loss':<12}    {'Sensors':<70}    {'Ablated':<30}\n")
        f.write("-"*170 + "\n")
        
        for rank, result in enumerate(ranked_results, 1):
            sensors_str = ', '.join(result['sensors'])
            ablated_str = ', '.join(result.get('ablated_sensors', [])) if result.get('ablated_sensors') else '-'
            f.write(f"{rank:<6}   {result['test_accuracy']:<12.4f}   "
                   f"{result['val_accuracy']:<12.4f}   {result['val_loss']:<12.4f}    "
                   f"{sensors_str:<70}    {ablated_str:<30}\n")
        
        f.write("\n" + "="*170 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*170 + "\n")
    
    print(f"\n✓ Report saved to: {output_file}")


def generate_json_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    """Generate a JSON report with ranked results."""
    
    if not ranked_results:
        print(f"No results found for experiment: {experiment_id}")
        return
    
    # Calculate statistics
    test_accs = [r['test_accuracy'] for r in ranked_results]
    
    report = {
        "experiment_id": experiment_id,
        "total_combinations": len(ranked_results),
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
    log_file = script_dir.parent / "fusion_logs" / "fusion_v4_ablation.jsonl"
    base_output_dir = script_dir.parent / "fusion_logs" / "ablation_reports"
    base_output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("ABLATION RESULTS ANALYZER")
    print("="*80)
    
    # Load all results
    print(f"\nLoading results from: {log_file}")
    all_results = load_ablation_results(log_file)
    
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
    
    # 3. Show available experiments (EXPERIMENT_ID is None and no command line arg)
    else:
        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        print("\nSet EXPERIMENT_ID at the top of this script to analyze results.")
        print("\nAvailable experiments:")
        print("-"*80)
        for exp_id, count in sorted(experiments.items()):
            print(f"  {exp_id}: {count} results")
        print("-"*80)
        print("\nExample: EXPERIMENT_ID = \"ablation_k7_20260216_143052\"")
        return
    
    # Safety check for empty string
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
    
    print(f"Found {len(filtered_results)} results")
    
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
    print(f"  Sensors: {', '.join(best['sensors'])}")
    if best.get('ablated_sensors'):
        print(f"  Ablated: {', '.join(best['ablated_sensors'])}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
