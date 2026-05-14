"""
aggregate_multirun.py

Aggregates multi-seed ablation results into averaged JSONL + TXT report
files that follow the same structure as single-run logs.

For each sensor combination, averages all numeric metrics across runs,
then outputs:
  1. averaged_<tag>_all_k.jsonl  — one line per combo (avg metrics + std)
  2. averaged_<tag>_k<N>_report.txt — one TXT report per k, matching
     the format of tremor_ablation_k[N]_report_*.txt

Usage:
  python aggregate_multirun.py               # aggregates 'mixed' (default)
  python aggregate_multirun.py --tag clean
  python aggregate_multirun.py --tag clean_aug
  python aggregate_multirun.py --tag mixed --tag clean  (run twice)

Output goes into:
  tremor_logs/ablation_reports_mixed/averaged_mixed_k<N>_report.txt
  tremor_logs/ablation_json_files_mixed/averaged_mixed_all_k.jsonl
  (same for clean / clean_aug)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

LOGS_DIR = Path(__file__).parents[1]

NUMERIC_FIELDS = [
    "val_accuracy", "val_f1_macro", "val_loss",
    "test_loss", "test_accuracy", "test_f1_macro",
    "gate_mean", "gate_std",
]

# Fields that should be taken from first occurrence (not averaged)
META_FIELDS = ["sensors", "ablated_sensors", "num_sensors", "fusion_mode"]


def load_all_jsonl(json_dir: Path) -> list[dict]:
    """Load every line from every .jsonl file in json_dir."""
    records = []
    for f in sorted(json_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def combo_key(record: dict) -> tuple:
    """Stable sort key for a sensor combination."""
    return tuple(sorted(record["sensors"]))


def aggregate(records: list[dict]) -> list[dict]:
    """
    Group records by sensor combination + num_sensors.
    Returns one averaged record per combo, sorted by test_f1_macro desc.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[combo_key(r)].append(r)

    averaged = []
    for key, runs in groups.items():
        first = runs[0]
        agg = {f: first[f] for f in META_FIELDS}

        for field in NUMERIC_FIELDS:
            vals = [r[field] for r in runs if r.get(field) is not None]
            if vals:
                agg[field] = float(np.mean(vals))
                agg[f"{field}_std"] = float(np.std(vals))
                agg[f"{field}_n"] = len(vals)
            else:
                agg[field] = None
                agg[f"{field}_std"] = None
                agg[f"{field}_n"] = 0

        agg["num_seeds"] = len(runs)
        agg["seeds"] = sorted({r.get("seed") for r in runs if r.get("seed") is not None})
        averaged.append(agg)

    averaged.sort(key=lambda x: x["test_f1_macro"] or 0, reverse=True)
    return averaged


def make_txt_report(combos: list[dict], k: int, tag: str, n_seeds: int) -> str:
    """
    Build a TXT report matching the format of tremor_ablation_k[N]_report_*.txt
    but with averaged metrics.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── header ───────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("TREMOR SENSOR ABLATION STUDY RESULTS  [AVERAGED OVER SEEDS]")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Generated  : {now}")
    lines.append(f"Tag        : {tag}")
    lines.append(f"K          : {k}")
    lines.append(f"Seeds used : {n_seeds}  ({combos[0]['seeds']})")
    lines.append(f"Total combinations: {len(combos)}")
    lines.append("")

    # ── ranking table ────────────────────────────────────────────────────
    # Compute column widths from actual data so nothing gets truncated
    sensor_strs  = [", ".join(c["sensors"])          for c in combos]
    ablated_strs = [", ".join(c["ablated_sensors"])   for c in combos]
    w_sensors = max(len("Sensors"),  max(len(s) for s in sensor_strs))
    w_ablated = max(len("Ablated"),  max(len(s) for s in ablated_strs))

    # Fixed-width numeric block: Rank(6) + sp(1) + sensors + sp(2) + ablated + sp(2) + metrics
    metrics_w = 9 + 1 + 7 + 2 + 9 + 1 + 8   # Test F1 ±std  Test Loss  Val Acc
    total_w   = 6 + 1 + w_sensors + 2 + w_ablated + 2 + metrics_w
    sep  = "=" * total_w
    dash = "-" * total_w
    hdr  = (
        f"{'Rank':<6} {('Sensors'):<{w_sensors}}  {('Ablated'):<{w_ablated}}  "
        f"{'Test F1':>9} {'±std':>7}  {'Test Loss':>9} {'Val Acc':>8}"
    )

    lines.append(sep)
    lines.append("RESULTS RANKED BY MEAN TEST F1 (MACRO)")
    lines.append(sep)
    lines.append(hdr)
    lines.append(dash)

    for rank, c, sensors_str, ablated_str in zip(range(1, len(combos)+1), combos, sensor_strs, ablated_strs):
        f1        = c["test_f1_macro"]
        f1_std    = c["test_f1_macro_std"]
        test_loss = c["test_loss"]
        val_acc   = c["val_accuracy"]

        lines.append(
            f"{rank:<6} {sensors_str:<{w_sensors}}  {ablated_str:<{w_ablated}}  "
            f"{f1:>9.4f} {f1_std:>7.4f}  {test_loss:>9.4f} {val_acc:>8.4f}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("BEST COMBINATION (by mean test F1)")
    lines.append("=" * 70)
    best = combos[0]
    lines.append(f"  Sensors      : {', '.join(best['sensors'])}")
    lines.append(f"  Ablated      : {', '.join(best['ablated_sensors'])}")
    lines.append(f"  Test F1      : {best['test_f1_macro']:.4f} ± {best['test_f1_macro_std']:.4f}")
    lines.append(f"  Val  F1      : {best['val_f1_macro']:.4f} ± {best['val_f1_macro_std']:.4f}")
    lines.append(f"  Test Loss    : {best['test_loss']:.4f} ± {best['test_loss_std']:.4f}")
    lines.append(f"  Val  Loss    : {best['val_loss']:.4f} ± {best['val_loss_std']:.4f}")
    lines.append(f"  Val  Acc     : {best['val_accuracy']:.4f} ± {best['val_accuracy_std']:.4f}")
    lines.append(f"  Num seeds    : {best['num_seeds']}")
    lines.append("")

    return "\n".join(lines)


def run(tag: str):
    json_dir    = LOGS_DIR / f"ablation_json_files_{tag}"
    reports_dir = LOGS_DIR / f"ablation_reports_{tag}"

    if not json_dir.exists():
        print(f"[ERROR] Directory not found: {json_dir}")
        sys.exit(1)

    reports_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"Aggregating tag='{tag}'")
    print(f"  Source : {json_dir}")
    print(f"{'=' * 65}")

    all_records = load_all_jsonl(json_dir)
    if not all_records:
        print("[ERROR] No records found.")
        sys.exit(1)

    n_seeds_total = len({r.get("seed") for r in all_records})
    print(f"  Loaded {len(all_records)} records across {n_seeds_total} seeds")

    # ── per-k aggregation ─────────────────────────────────────────────
    k_values = sorted({r["num_sensors"] for r in all_records})
    print(f"  k values found: {k_values}")

    all_averaged = []

    for k in k_values:
        k_records  = [r for r in all_records if r["num_sensors"] == k]
        k_averaged = aggregate(k_records)
        n_seeds_k  = len({r.get("seed") for r in k_records})
        all_averaged.extend(k_averaged)

        # Add experiment_id to each for JSONL
        ts = datetime.now().strftime("%m%d_%H%M")
        for c in k_averaged:
            c["experiment_id"] = f"averaged_{tag}_k{k}_{ts}"

        # TXT report
        txt = make_txt_report(k_averaged, k=k, tag=tag, n_seeds=n_seeds_k)
        txt_path = reports_dir / f"averaged_{tag}_k{k}_report.txt"
        txt_path.write_text(txt)
        print(f"  [k={k}]  {len(k_averaged)} combos, {n_seeds_k} seeds → {txt_path.name}")

    # ── combined JSONL (all k) ────────────────────────────────────────
    # Sort by num_sensors then test_f1_macro desc
    all_averaged.sort(key=lambda x: (x["num_sensors"], -(x["test_f1_macro"] or 0)))
    jsonl_path = json_dir / f"averaged_{tag}_all_k.jsonl"
    with open(jsonl_path, "w") as fh:
        for c in all_averaged:
            fh.write(json.dumps(c) + "\n")

    print(f"\n  JSONL → {jsonl_path}")
    print(f"  Done. {len(all_averaged)} total combinations written.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed ablation results")
    parser.add_argument(
        "--tag", type=str, default="mixed",
        help="Which tag to aggregate (default: mixed). "
             "E.g. mixed, clean, clean_aug, clean_on_100weak, clean_on_100awgn, ..."
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Aggregate both mixed and clean"
    )
    args = parser.parse_args()

    if args.both:
        run("mixed")
        run("clean")
    else:
        run(args.tag)


if __name__ == "__main__":
    main()
