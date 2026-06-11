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

# Tag to plot when COMPARE_TAGS=False and TAGS_BATCH is empty
TAG = "clean_train_awgn_a000"

# Generate one separate plot per tag (ignored when COMPARE_TAGS=True)
# Leave empty [] to fall back to single TAG above
TAGS_BATCH = []

# Plot two tags side by side (overrides TAG / TAGS_BATCH)
COMPARE_TAGS = False
COMPARE_PAIR = ["clean", "mixed"]  # e.g. ["mixed", "clean"] or ["clean_aug", "clean"]

# Run a batch of single-tag plots AND/OR compare-pairs in one go.
# Each entry is either a string (single plot) or a list of two strings (side-by-side).
# When non-empty, overrides TAG / TAGS_BATCH / COMPARE_TAGS / COMPARE_PAIR entirely.
PLOT_QUEUE = [
    "clean_train_awgn_a000",
    "clean_train_tremor_mixed_all3",
    "mixed_train_tremor_mixed_all3",
    ["clean_train_awgn_a000", "clean_train_tremor_mixed_all3"],
    ["clean_train_tremor_mixed_all3", "mixed_train_tremor_mixed_all3"],
    ["clean_train_awgn_a000", "mixed_train_tremor_mixed_all3"],
]

# What to plot on y-axis
Y_METRIC = "test_f1_macro"
Y_LABEL  = "Test F1 (macro)"

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
    _int_keys   = {"k", "num_sensors", "seed"}
    _str_keys   = {"timestamp", "sensors"}

    def _cast(k, v):
        if k in _str_keys:
            return v
        if k in _int_keys:
            try:
                return int(v)
            except (ValueError, TypeError):
                return v
        try:
            return float(v)
        except (ValueError, TypeError):
            return v  # leave as string (e.g. 'variant' column)

    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            records.append({k: _cast(k, v) for k, v in row.items()})
    return records


def load_for_tag(tag: str) -> list:
    json_dir = LOGS_DIR / f"ablation_json_files_{tag}"

    # When USE_AVERAGED=True, prefer the pre-computed averaged JSONL (one mean F1 per combo)
    if USE_AVERAGED:
        averaged_file = json_dir / f"averaged_{tag}_all_k.jsonl"
        if averaged_file.exists():
            print(f"  [{tag}] Loading averaged file: {averaged_file.name}")
            return load_jsonl(averaged_file)

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

    # Fall back to raw per-seed JSONL files
    if USE_AVERAGED:
        raise FileNotFoundError(
            f"Averaged file not found: {json_dir / f'averaged_{tag}_all_k.jsonl'}\n"
            f"Run aggregate_multirun.py --tag {tag} first."
        )
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


_default_colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#64B5CD"]
_color_map = {"mixed": "#4C72B0", "clean": "#DD8452", "clean_aug": "#55A868"}


def _plot_single(tag: str):
    """Generate one boxplot for a single tag."""
    mode_label = "averaged (mean per combo)" if USE_AVERAGED else "raw per-seed"
    print(f"\nMode: {mode_label}  |  Tag: {tag}")

    records = load_for_tag(tag)
    k_values, data = group_by_k(records)
    print_summary(tag, k_values, data)

    color = _color_map.get(tag, _default_colors[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_boxplot(ax, k_values, data, color=color, label=tag, offset=0.0, width=0.6)

    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("Number of sensors (k)", fontsize=13)
    ax.set_ylabel(Y_LABEL, fontsize=13)
    src_str = "mean per combo, 10 seeds" if USE_AVERAGED else "raw per-seed"
    ax.set_title(f"Sensor Ablation Study — {tag}  [{src_str}]", fontsize=14, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_file = Path(__file__).parent / f"ablation_json_files_{tag}" / f"ablation_boxplot_{tag}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"✓ Plot saved to: {out_file}")


def _plot_compare(pair: list):
    """Generate one side-by-side boxplot for two tags."""
    mode_label = "averaged (mean per combo)" if USE_AVERAGED else "raw per-seed"
    print(f"\nMode: {mode_label}  |  Tags: {pair}")

    datasets = {}
    for i, tag in enumerate(pair):
        records = load_for_tag(tag)
        k_values, data = group_by_k(records)
        print_summary(tag, k_values, data)
        datasets[tag] = (k_values, data)

    colors  = {t: _color_map.get(t, _default_colors[i % len(_default_colors)]) for i, t in enumerate(pair)}
    offsets = {pair[0]: -0.2, pair[1]: 0.2}

    fig, ax = plt.subplots(figsize=(12, 6))
    all_k = sorted({k for tag in pair for k in datasets[tag][0]})

    for tag in pair:
        k_values, data = datasets[tag]
        draw_boxplot(ax, k_values, data, color=colors[tag], label=tag,
                     offset=offsets[tag], width=0.35)

    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel("Number of sensors (k)", fontsize=13)
    ax.set_ylabel(Y_LABEL, fontsize=13)
    src_str = "mean per combo, 10 seeds" if USE_AVERAGED else "raw per-seed"
    tag_str = " vs ".join(pair)
    ax.set_title(f"Sensor Ablation Study — {tag_str}  [{src_str}]", fontsize=14, pad=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    _out_tag = pair[0] + "_vs_" + pair[1]
    out_file = Path(__file__).parent / f"ablation_json_files_{_out_tag}" / f"ablation_boxplot_{_out_tag}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"✓ Plot saved to: {out_file}")


def main():
    if PLOT_QUEUE:
        for entry in PLOT_QUEUE:
            if isinstance(entry, list):
                _plot_compare(entry)
            else:
                _plot_single(entry)
    elif COMPARE_TAGS:
        _plot_compare(COMPARE_PAIR)
    elif TAGS_BATCH:
        for tag in TAGS_BATCH:
            _plot_single(tag)
    else:
        _plot_single(TAG)


if __name__ == "__main__":
    main()
