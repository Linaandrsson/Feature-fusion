"""
analyze_ablation_results.py

Analyze and rank tremor ablation study results by experiment ID.

Works with:
  Feature fusion activity/Tremor/fusion_concat_ablation_study_tremor.py

Usage:
  1) Set EXPERIMENT_ID below (recommended in VS Code), or
  2) Run with command line argument:
       python analyze_ablation_results.py <experiment_id>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_ID = None
# Example:
# EXPERIMENT_ID = "tremor_ablation_k6_0318_1430"


def resolve_log_file(script_dir: Path) -> Tuple[Path, List[Path]]:
    """Pick the best available ablation log path.

    Priority:
      1) Tremor-local log produced by fusion_concat_ablation_study_tremor.py
      2) Legacy shared fusion log
    """
    candidates = [
        script_dir / "tremor_logs" / "tremor_fusion_ablation.jsonl",
        script_dir.parent / "fusion_logs" / "fusion_v4_ablation.jsonl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates

    return candidates[0], candidates


def load_ablation_results(log_file: Path) -> List[Dict]:
    """Load all valid JSON lines from an ablation log file."""
    results = []

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return results

    with open(log_file, "r") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Warning: Skipping malformed line: {error}")

    return results


def get_available_experiments(results: List[Dict]) -> Dict[str, int]:
    """Get unique experiment IDs and number of entries per ID."""
    experiments = defaultdict(int)

    for result in results:
        exp_id = result.get("experiment_id", result.get("experiment", "unknown"))
        experiments[exp_id] += 1

    return dict(experiments)


def filter_by_experiment(results: List[Dict], experiment_id: str) -> List[Dict]:
    """Filter result entries by experiment ID."""
    filtered = []

    for result in results:
        exp_id = result.get("experiment_id", result.get("experiment", ""))
        if exp_id == experiment_id:
            filtered.append(result)

    return filtered


def choose_rank_metric(results: List[Dict]) -> Tuple[str, str]:
    """Choose best ranking metric based on available keys."""
    metric_candidates = [
        ("test_f1_macro", "Test F1 (macro)"),
        ("test_f1", "Test F1"),
        ("test_accuracy", "Test Accuracy"),
    ]

    for key, label in metric_candidates:
        if any(key in result for result in results):
            return key, label

    return "test_accuracy", "Test Accuracy"


def rank_results(results: List[Dict], metric_key: str) -> List[Dict]:
    """Sort results by selected metric (descending)."""
    return sorted(results, key=lambda item: item.get(metric_key, 0.0), reverse=True)


def _safe_stat(values: List[float]) -> Optional[Dict[str, float]]:
    """Return mean/max/min/range stats when values exist."""
    if not values:
        return None

    max_value = max(values)
    min_value = min(values)

    return {
        "mean": sum(values) / len(values),
        "max": max_value,
        "min": min_value,
        "range": max_value - min_value,
    }


def _collect_metric_values(results: List[Dict], metric_key: str) -> List[float]:
    values = []
    for result in results:
        value = result.get(metric_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _fmt_value(value: Optional[float]) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def generate_report(
    experiment_id: str,
    ranked_results: List[Dict],
    output_file: Path,
    ranking_key: str,
    ranking_label: str,
    source_log_file: Path,
):
    """Generate a text report with ranked results."""
    if not ranked_results:
        print(f"No results found for experiment: {experiment_id}")
        return

    first = ranked_results[0]
    num_sensors = first.get("num_sensors", "unknown")
    fusion_mode = first.get("fusion_mode", "unknown")
    seed = first.get("seed", "unknown")

    ranking_values = _collect_metric_values(ranked_results, ranking_key)
    test_acc_values = _collect_metric_values(ranked_results, "test_accuracy")
    val_acc_values = _collect_metric_values(ranked_results, "val_accuracy")
    val_loss_values = _collect_metric_values(ranked_results, "val_loss")

    rank_stats = _safe_stat(ranking_values)
    test_acc_stats = _safe_stat(test_acc_values)
    val_acc_stats = _safe_stat(val_acc_values)
    val_loss_stats = _safe_stat(val_loss_values)

    table_width = 190
    with open(output_file, "w") as file:
        file.write("=" * table_width + "\n")
        file.write(f"TREMOR ABLATION RESULTS - RANKED BY {ranking_label.upper()}\n")
        file.write("=" * table_width + "\n\n")

        file.write(f"Experiment ID: {experiment_id}\n")
        file.write(f"Source log: {source_log_file}\n")
        file.write(f"Number of sensors per combination: {num_sensors}\n")
        file.write(f"Fusion mode: {fusion_mode}\n")
        file.write(f"Seed: {seed}\n")
        file.write(f"Total combinations tested: {len(ranked_results)}\n\n")

        best = ranked_results[0]
        file.write("-" * table_width + "\n")
        file.write("BEST COMBINATION\n")
        file.write("-" * table_width + "\n")
        file.write(f"Sensors:       {', '.join(best.get('sensors', []))}\n")
        if best.get("ablated_sensors"):
            file.write(f"Ablated:       {', '.join(best['ablated_sensors'])}\n")
        file.write(f"{ranking_label}: {_fmt_value(best.get(ranking_key))}\n")
        if ranking_key != "test_accuracy":
            file.write(f"Test Accuracy: {_fmt_value(best.get('test_accuracy'))}\n")
        file.write(f"Val Accuracy:  {_fmt_value(best.get('val_accuracy'))}\n")
        file.write(f"Val Loss:      {_fmt_value(best.get('val_loss'))}\n")
        if "test_loss" in best:
            file.write(f"Test Loss:     {_fmt_value(best.get('test_loss'))}\n")
        if "test_f1_macro" in best:
            file.write(f"Test F1 Macro: {_fmt_value(best.get('test_f1_macro'))}\n")
        if "test_f1_weighted" in best:
            file.write(f"Test F1 Wghtd: {_fmt_value(best.get('test_f1_weighted'))}\n")
        file.write("\n")

        worst = ranked_results[-1]
        file.write("-" * table_width + "\n")
        file.write("WORST COMBINATION\n")
        file.write("-" * table_width + "\n")
        file.write(f"Sensors:       {', '.join(worst.get('sensors', []))}\n")
        if worst.get("ablated_sensors"):
            file.write(f"Ablated:       {', '.join(worst['ablated_sensors'])}\n")
        file.write(f"{ranking_label}: {_fmt_value(worst.get(ranking_key))}\n")
        if ranking_key != "test_accuracy":
            file.write(f"Test Accuracy: {_fmt_value(worst.get('test_accuracy'))}\n")
        file.write(f"Val Accuracy:  {_fmt_value(worst.get('val_accuracy'))}\n")
        file.write(f"Val Loss:      {_fmt_value(worst.get('val_loss'))}\n\n")

        file.write("-" * table_width + "\n")
        file.write("STATISTICS\n")
        file.write("-" * table_width + "\n")
        if rank_stats:
            file.write(
                f"{ranking_label}: mean={rank_stats['mean']:.4f}  "
                f"max={rank_stats['max']:.4f}  min={rank_stats['min']:.4f}  "
                f"range={rank_stats['range']:.4f}\n"
            )
        if test_acc_stats and ranking_key != "test_accuracy":
            file.write(
                f"Test Accuracy: mean={test_acc_stats['mean']:.4f}  "
                f"max={test_acc_stats['max']:.4f}  min={test_acc_stats['min']:.4f}  "
                f"range={test_acc_stats['range']:.4f}\n"
            )
        if val_acc_stats:
            file.write(
                f"Val Accuracy:  mean={val_acc_stats['mean']:.4f}  "
                f"max={val_acc_stats['max']:.4f}  min={val_acc_stats['min']:.4f}  "
                f"range={val_acc_stats['range']:.4f}\n"
            )
        if val_loss_stats:
            file.write(
                f"Val Loss:      mean={val_loss_stats['mean']:.4f}  "
                f"max={val_loss_stats['max']:.4f}  min={val_loss_stats['min']:.4f}  "
                f"range={val_loss_stats['range']:.4f}\n"
            )
        file.write("\n")

        file.write("=" * table_width + "\n")
        file.write("COMPLETE RANKING\n")
        file.write("=" * table_width + "\n\n")

        rank_col_title = ranking_label[:14]
        file.write(
            f"{'Rank':<6}   {rank_col_title:<14}   {'Test Acc':<10}   "
            f"{'Val Acc':<10}   {'Val Loss':<10}   {'Sensors':<80}   {'Ablated':<40}\n"
        )
        file.write("-" * table_width + "\n")

        for rank, result in enumerate(ranked_results, 1):
            sensors_str = ", ".join(result.get("sensors", []))
            ablated_str = ", ".join(result.get("ablated_sensors", [])) if result.get("ablated_sensors") else "-"
            row = (
                f"{rank:<6}   {_fmt_value(result.get(ranking_key)):<14}   "
                f"{_fmt_value(result.get('test_accuracy')):<10}   "
                f"{_fmt_value(result.get('val_accuracy')):<10}   "
                f"{_fmt_value(result.get('val_loss')):<10}   "
                f"{sensors_str:<80}   {ablated_str:<40}"
            )
            file.write(row + "\n")

        file.write("\n" + "=" * table_width + "\n")
        file.write("END OF REPORT\n")
        file.write("=" * table_width + "\n")

    print(f"✓ Text report saved to: {output_file}")


def generate_json_report(
    experiment_id: str,
    ranked_results: List[Dict],
    output_file: Path,
    ranking_key: str,
    ranking_label: str,
    source_log_file: Path,
):
    """Generate a JSON report with ranked results."""
    if not ranked_results:
        print(f"No results found for experiment: {experiment_id}")
        return

    report = {
        "experiment_id": experiment_id,
        "source_log_file": str(source_log_file),
        "ranking": {
            "metric_key": ranking_key,
            "metric_label": ranking_label,
        },
        "metadata": {
            "num_sensors": ranked_results[0].get("num_sensors"),
            "fusion_mode": ranked_results[0].get("fusion_mode"),
            "seed": ranked_results[0].get("seed"),
            "total_combinations": len(ranked_results),
        },
        "statistics": {
            ranking_key: _safe_stat(_collect_metric_values(ranked_results, ranking_key)),
            "test_accuracy": _safe_stat(_collect_metric_values(ranked_results, "test_accuracy")),
            "val_accuracy": _safe_stat(_collect_metric_values(ranked_results, "val_accuracy")),
            "val_loss": _safe_stat(_collect_metric_values(ranked_results, "val_loss")),
            "test_loss": _safe_stat(_collect_metric_values(ranked_results, "test_loss")),
            "test_f1_macro": _safe_stat(_collect_metric_values(ranked_results, "test_f1_macro")),
            "test_f1_weighted": _safe_stat(_collect_metric_values(ranked_results, "test_f1_weighted")),
        },
        "best_combination": ranked_results[0],
        "worst_combination": ranked_results[-1],
        "ranked_results": ranked_results,
    }

    with open(output_file, "w") as file:
        json.dump(report, file, indent=2)

    print(f"✓ JSON report saved to: {output_file}")


def main():
    script_dir = Path(__file__).parent
    log_file, searched_paths = resolve_log_file(script_dir)
    base_output_dir = script_dir / "tremor_logs" / "ablation_reports"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TREMOR ABLATION RESULTS ANALYZER")
    print("=" * 80)

    print(f"\nLoading results from: {log_file}")
    all_results = load_ablation_results(log_file)

    if not all_results:
        print("No results found in selected log file.")
        print("\nChecked these locations:")
        for path in searched_paths:
            print(f"  - {path}")
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
            print(f"  {exp_id}: {count} results")
        print("-" * 80)
        print("\nExample: EXPERIMENT_ID = \"tremor_ablation_k6_0318_1430\"")
        return

    if not experiment_id or not experiment_id.strip():
        print("Error: Empty experiment ID provided")
        return

    print(f"\nFiltering results for experiment: {experiment_id}")
    filtered_results = filter_by_experiment(all_results, experiment_id)

    if not filtered_results:
        print(f"Error: No results found for experiment ID: {experiment_id}")
        print("\nAvailable experiment IDs:")
        for exp_id in sorted(experiments.keys()):
            print(f"  - {exp_id}")
        return

    print(f"Found {len(filtered_results)} results")

    ranking_key, ranking_label = choose_rank_metric(filtered_results)
    ranked_results = rank_results(filtered_results, ranking_key)
    print(f"Ranking metric: {ranking_label} ({ranking_key})")

    first_result = ranked_results[0]
    train_mode = first_result.get("train_mode")
    test_mode = first_result.get("test_mode")
    fusion_mode = first_result.get("fusion_mode", "unknown")

    output_dir = base_output_dir
    if isinstance(train_mode, str) and train_mode and isinstance(test_mode, str) and test_mode:
        output_dir = output_dir / f"{train_mode}_train-{test_mode}_test"
    if isinstance(fusion_mode, str) and fusion_mode:
        output_dir = output_dir / fusion_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating reports in: {output_dir}")

    txt_output = output_dir / f"results_{experiment_id}.txt"
    generate_report(
        experiment_id=experiment_id,
        ranked_results=ranked_results,
        output_file=txt_output,
        ranking_key=ranking_key,
        ranking_label=ranking_label,
        source_log_file=log_file,
    )

    json_output = output_dir / f"results_{experiment_id}.json"
    generate_json_report(
        experiment_id=experiment_id,
        ranked_results=ranked_results,
        output_file=json_output,
        ranking_key=ranking_key,
        ranking_label=ranking_label,
        source_log_file=log_file,
    )

    best = ranked_results[0]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBest combination ({ranking_label}: {_fmt_value(best.get(ranking_key))}):")
    print(f"  Sensors: {', '.join(best.get('sensors', []))}")
    if best.get("ablated_sensors"):
        print(f"  Ablated: {', '.join(best['ablated_sensors'])}")
    if ranking_key != "test_accuracy":
        print(f"  Test Accuracy: {_fmt_value(best.get('test_accuracy'))}")

    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
