"""
analyze_tremor_fs_results.py

Analyze and rank tremor FS combination study results by experiment ID.

Works with:
  Feature fusion activity/Tremor/fusion_concat_fs_study_tremor.py

Usage:
  1) Set EXPERIMENT_ID below (recommended in VS Code), or
  2) Run with command line argument:
       python analyze_tremor_fs_results.py <experiment_id>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_ID = None
# Example:
# EXPERIMENT_ID = "tremor_fs_study_k2_fs10_20_30_40_50_13-30-35"


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_fs_results(log_file: Path) -> List[Dict]:
    results = []

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return results

    with open(log_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Warning: Skipping malformed line: {error}")

    return results


def get_available_experiments(results: List[Dict]) -> Dict[str, int]:
    counts = defaultdict(int)
    for result in results:
        exp_id = result.get("experiment_id", "unknown")
        counts[exp_id] += 1
    return dict(counts)


def filter_by_experiment(results: List[Dict], experiment_id: str) -> List[Dict]:
    return [result for result in results if result.get("experiment_id") == experiment_id]


def rank_results(results: List[Dict], metric: str = "test_f1") -> List[Dict]:
    return sorted(results, key=lambda item: item.get(metric, 0), reverse=True)


def format_fs_map(fs_map: Dict[str, int]) -> str:
    return ", ".join(f"{sensor}:fs{fs}" for sensor, fs in fs_map.items())


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_text_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    if not ranked_results:
        print("No results to report")
        return

    first = ranked_results[0]
    sensors = first.get("sensors", [])
    seed = first.get("seed", "unknown")
    run_timestamp = first.get("run_timestamp", "unknown")
    tremor_types = first.get("tremor_types_merged", [])
    fusion_mode = first.get("fusion_mode", "unknown")
    embeddings_folder = first.get("embeddings_folder", "unknown")
    subject_split = first.get("subject_split", {})

    test_f1s = [item.get("test_f1", 0.0) for item in ranked_results]
    test_accs = [item.get("test_accuracy", 0.0) for item in ranked_results]
    val_f1s = [item.get("val_f1", 0.0) for item in ranked_results]
    val_accs = [item.get("val_accuracy", 0.0) for item in ranked_results]

    def fs_str_for(sensor_name: str, fs_map: Dict[str, int]) -> str:
        fs_value = fs_map.get(sensor_name)
        return f"fs{fs_value}" if fs_value is not None else "-"

    sensor_col_widths = {}
    for sensor in sensors:
        max_entry_len = max(
            len(fs_str_for(sensor, item.get("fs_map", {})))
            for item in ranked_results
        )
        sensor_col_widths[sensor] = max(len(sensor), max_entry_len, 4) + 2

    header_left = f"{'Rank':<6}    {'Val F1':<10}    {'Val Acc':<10}    {'Test F1':<10}    {'Test Acc':<10}"
    header_sensors = "    ".join(f"{sensor:^{sensor_col_widths[sensor]}}" for sensor in sensors)
    table_header = f"{header_left}    {header_sensors}" if sensors else header_left
    table_width = max(180, len(table_header))

    with open(output_file, "w") as f:
        f.write("=" * table_width + "\n")
        f.write("TREMOR FS COMBINATION STUDY RESULTS - RANKED BY TEST F1\n")
        f.write("=" * table_width + "\n\n")

        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Sensors: {', '.join(sensors)} ({len(sensors)} sensors)\n")
        f.write(f"Fusion mode: {fusion_mode}\n")
        f.write(f"Embeddings folder: {embeddings_folder}\n")
        f.write(f"Tremor types merged: {', '.join(tremor_types)}\n")
        f.write(f"Subject split: train={subject_split.get('train_subjects')} | val={subject_split.get('val_subjects')} | test={subject_split.get('test_subjects')}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total combinations tested: {len(ranked_results)}\n")
        f.write("\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * table_width + "\n")
        f.write(f"Test F1:   mean={safe_mean(test_f1s):.4f}  max={max(test_f1s):.4f}  min={min(test_f1s):.4f}  range={max(test_f1s)-min(test_f1s):.4f}\n")
        f.write(f"Test Acc:  mean={safe_mean(test_accs):.4f}  max={max(test_accs):.4f}  min={min(test_accs):.4f}  range={max(test_accs)-min(test_accs):.4f}\n")
        f.write(f"Val F1:    mean={safe_mean(val_f1s):.4f}  max={max(val_f1s):.4f}  min={min(val_f1s):.4f}  range={max(val_f1s)-min(val_f1s):.4f}\n")
        f.write(f"Val Acc:   mean={safe_mean(val_accs):.4f}  max={max(val_accs):.4f}  min={min(val_accs):.4f}  range={max(val_accs)-min(val_accs):.4f}\n")
        f.write("\n")

        f.write("RANKED FS COMBINATIONS (by test F1)\n")
        f.write("=" * table_width + "\n\n")
        f.write(table_header + "\n")
        f.write("-" * table_width + "\n")

        for rank, result in enumerate(ranked_results, 1):
            val_f1 = result.get("val_f1", 0.0)
            val_acc = result.get("val_accuracy", 0.0)
            test_f1 = result.get("test_f1", 0.0)
            test_acc = result.get("test_accuracy", 0.0)
            fs_map = result.get("fs_map", {})

            row_left = f"{rank:<6}    {val_f1:.4f}        {val_acc:.4f}        {test_f1:.4f}        {test_acc:.4f}"
            if sensors:
                row_sensors = "    ".join(
                    f"{fs_str_for(sensor, fs_map):^{sensor_col_widths[sensor]}}"
                    for sensor in sensors
                )
                f.write(f"{row_left}    {row_sensors}\n")
            else:
                f.write(f"{row_left}\n")

        f.write("\n" + "=" * table_width + "\n")
        f.write(f"Report generated from run timestamp: {run_timestamp}\n")
        f.write("=" * table_width + "\n")

    print(f"✓ Text report saved to: {output_file}")


def generate_json_report(experiment_id: str, ranked_results: List[Dict], output_file: Path):
    if not ranked_results:
        print("No results to report")
        return

    test_f1s = [item.get("test_f1", 0.0) for item in ranked_results]
    test_accs = [item.get("test_accuracy", 0.0) for item in ranked_results]
    val_f1s = [item.get("val_f1", 0.0) for item in ranked_results]
    val_accs = [item.get("val_accuracy", 0.0) for item in ranked_results]

    report = {
        "experiment_id": experiment_id,
        "metadata": {
            "sensors": ranked_results[0].get("sensors"),
            "fusion_mode": ranked_results[0].get("fusion_mode"),
            "embeddings_folder": ranked_results[0].get("embeddings_folder"),
            "tremor_types_merged": ranked_results[0].get("tremor_types_merged"),
            "subject_split": ranked_results[0].get("subject_split"),
            "seed": ranked_results[0].get("seed"),
            "run_timestamp": ranked_results[0].get("run_timestamp"),
            "total_combinations": len(ranked_results),
        },
        "statistics": {
            "test_f1": {
                "mean": safe_mean(test_f1s),
                "max": max(test_f1s),
                "min": min(test_f1s),
                "range": max(test_f1s) - min(test_f1s),
            },
            "test_accuracy": {
                "mean": safe_mean(test_accs),
                "max": max(test_accs),
                "min": min(test_accs),
                "range": max(test_accs) - min(test_accs),
            },
            "val_f1": {
                "mean": safe_mean(val_f1s),
                "max": max(val_f1s),
                "min": min(val_f1s),
                "range": max(val_f1s) - min(val_f1s),
            },
            "val_accuracy": {
                "mean": safe_mean(val_accs),
                "max": max(val_accs),
                "min": min(val_accs),
                "range": max(val_accs) - min(val_accs),
            },
        },
        "best_combination": ranked_results[0],
        "worst_combination": ranked_results[-1],
        "ranked_results": ranked_results,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ JSON report saved to: {output_file}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).parent
    log_file = script_dir / "tremor_logs" / "tremor_fusion_fs_combo_study.jsonl"
    output_dir = script_dir / "tremor_logs" / "fs_study_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TREMOR FS COMBINATION STUDY ANALYZER")
    print("=" * 80)

    print(f"\nLoading results from: {log_file}")
    all_results = load_fs_results(log_file)

    if not all_results:
        print("No results found in log file.")
        return

    print(f"Loaded {len(all_results)} results")
    experiments = get_available_experiments(all_results)

    experiment_id = None

    if EXPERIMENT_ID is not None:
        experiment_id = EXPERIMENT_ID
        print(f"\nUsing experiment ID from configuration: {experiment_id}")
    elif len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        print(f"\nUsing experiment ID from command line: {experiment_id}")
    else:
        print("\n" + "=" * 80)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 80)
        print("\nSet EXPERIMENT_ID in this file or pass it as argument.")
        print("\nAvailable experiments:")
        print("-" * 80)
        for exp_id, count in sorted(experiments.items()):
            print(f"  {exp_id}: {count} combinations")
        print("-" * 80)
        return

    if not experiment_id or not experiment_id.strip():
        print("Error: Empty experiment ID provided")
        return

    print(f"\nFiltering results for experiment: {experiment_id}")
    filtered = filter_by_experiment(all_results, experiment_id)

    if not filtered:
        print(f"Error: No results found for experiment ID: {experiment_id}")
        print("\nAvailable experiment IDs:")
        for exp_id in sorted(experiments):
            print(f"  - {exp_id}")
        return

    print(f"Found {len(filtered)} combinations")
    ranked = rank_results(filtered, metric="test_f1")

    txt_output = output_dir / f"results_{experiment_id}.txt"
    json_output = output_dir / f"results_{experiment_id}.json"

    print("\nGenerating reports...")
    generate_text_report(experiment_id, ranked, txt_output)
    generate_json_report(experiment_id, ranked, json_output)

    best = ranked[0]
    worst = ranked[-1]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBest combination (Test F1: {best.get('test_f1', 0):.4f}, Test Acc: {best.get('test_accuracy', 0):.4f})")
    print(f"  FS map: {format_fs_map(best.get('fs_map', {}))}")
    print(f"\nWorst combination (Test F1: {worst.get('test_f1', 0):.4f}, Test Acc: {worst.get('test_accuracy', 0):.4f})")
    print(f"  FS map: {format_fs_map(worst.get('fs_map', {}))}")
    print(f"\nReports saved to: {output_dir}")


if __name__ == "__main__":
    main()
