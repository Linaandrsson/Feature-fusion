"""
plot_ablation_boxplot.py

Generates a boxplot from ablation study JSONL files.
  X-axis: Number of sensors (k)
  Y-axis: Test F1 (macro)

Two modes:
  USE_AVERAGED = True   -> averaged_*_all_k.jsonl (one mean value per combo)
                           Each point = mean F1 of one sensor combination
                           (run aggregate_multirun.py first)
  USE_AVERAGED = False  -> raw per-seed files (many points per k per combo)

Side-by-side comparison:
  COMPARE_TAGS = True  -> plots mixed + clean in the same figure
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ═══════════════════════════════════════════════════════════════

LOGS_DIR = Path(__file__).parents[2]

# Use averaged files (one mean F1 per sensor combo) or raw per-seed files
USE_AVERAGED = True

# Tag to plot when COMPARE_TAGS=False: "mixed", "clean", or "clean_aug"
TAG = "mixed"

# Plot two tags side by side (only used when COMPARE_TAGS=True)
COMPARE_TAGS = True
COMPARE_PAIR = ["clean", "mixed"]  # e.g. ["mixed", "clean"] or ["clean_aug", "clean"]

# What to plot on y-axis
Y_METRIC = "test_f1_macro"
Y_LABEL  = "Test F1 (macro)"

# Output file (saved next to this script)
OUTPUT_FILE = Path(__file__).parent / "ablation_boxplot_clean_vs_mixed.png"

# ═══════════════════════════════════════════════════════════════


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_csv(path: Path) -> list:
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({
                k: (float(v) if k not in ("timestamp", "sensors") else
                    (int(v) if k in ("k", "num_sensors", "seed") else v))
                for k, v in row.items()
            })
    return records


def load_for_tag(tag: str) -> list:
    # CSV-based tags (e.g. clean_aug) live in ablation_reports_{tag}/
    csv_dir = LOGS_DIR / f"ablation_reports_{tag}"
    if csv_dir.exists():
        paths = sorted(csv_dir.glob("*_results_*.csv"))
        if paths:
            print(f"  [{tag}] Loading {len(paths)} CSV result files from {csv_dir.name}/")
            records = []
            for p in paths:
                records.extend(load_csv(p))
            return records

    # JSONL-based tags (mixed / clean) live in ablation_json_files_{tag}/
    json_dir = LOGS_DIR / f"ablation_json_files_{tag}"
    if USE_AVERAGED:
        averaged_file = json_dir / f"averaged_{tag}_all_k.jsonl"
        if not averaged_file.exists():
            raise FileNotFoundError(
                f"Averaged file not found: {averaged_file}\n"
                f"Run aggregate_multirun.py --tag {tag} first."
            )
        print(f"  [{tag}] Loading averaged file: {averaged_file.name}")
        return load_jsonl(averaged_file)
    else:
        paths = sorted(
            p for p in json_dir.glob("*.jsonl")
            if not p.name.startswith("averaged_")
        )
        print(f"  [{tag}] Loading {len(paths)} raw seed files")
        records = []
        for p in paths:
            records.extend(load_jsonl(p))
        return records


def group_by_k(records: list) -> tuple:
    by_k = defaultdict(list)
    for r in records:
        val = r.get(Y_METRIC)
        if val is not None:
            by_k[r["num_sensors"]].append(val)
    k_values = sorted(by_k.keys())
    return k_values, [by_k[k] for k in k_values]


def print_summary(tag: str, k_values, data):
    print(f"\n  [{tag}]  {'k':>3}  {'n':>4}  {'min':>6}  {'median':>8}  {'max':>6}")
    print("  " + "-" * 44)
    for k, vals in zip(k_values, data):
        arr = np.array(vals)
        print(
            f"  [{tag}]  {k:>3}  {len(vals):>4}  {arr.min():.4f}  "
            f"{np.median(arr):.6f}  {arr.max():.4f}"
        )


def draw_boxplot(ax, k_values, data, color, label, offset=0.0, width=0.35):
    positions = [k + offset for k in k_values]
    ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor=color, alpha=0.7),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    # Proxy for legend
    ax.plot([], [], color=color, linewidth=8, alpha=0.7, label=label)


def main():
    tags = COMPARE_PAIR if COMPARE_TAGS else [TAG]
    mode_label = "averaged (mean per combo)" if USE_AVERAGED else "raw per-seed"
    print(f"\nMode: {mode_label}  |  Tags: {tags}")

    datasets = {}
    for tag in tags:
        records = load_for_tag(tag)
        k_values, data = group_by_k(records)
        print_summary(tag, k_values, data)
        datasets[tag] = (k_values, data)

    fig, ax = plt.subplots(figsize=(12 if COMPARE_TAGS else 10, 6))

    colors  = {"mixed": "#4C72B0", "clean": "#DD8452", "clean_aug": "#55A868"}
    if COMPARE_TAGS:
        offsets = {COMPARE_PAIR[0]: -0.2, COMPARE_PAIR[1]: 0.2}
    else:
        offsets = {TAG: 0.0}
    width   = 0.35 if COMPARE_TAGS else 0.6

    all_k = sorted({k for tag in tags for k in datasets[tag][0]})

    for tag in tags:
        k_values, data = datasets[tag]
        draw_boxplot(
            ax, k_values, data,
            color=colors[tag],
            label=tag.capitalize(),
            offset=offsets[tag],
            width=width,
        )

    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel("Number of sensors (k)", fontsize=13)
    ax.set_ylabel(Y_LABEL, fontsize=13)

    tag_str = " vs ".join(t.capitalize() for t in COMPARE_PAIR) if COMPARE_TAGS else TAG.capitalize()
    src_str = "mean per combo, 10 seeds" if USE_AVERAGED else "raw per-seed"
    ax.set_title(
        f"Sensor Ablation Study — {tag_str}  [{src_str}]",
        fontsize=14, pad=12
    )

    if COMPARE_TAGS:
        ax.legend(fontsize=11)

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"\n✓ Plot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
