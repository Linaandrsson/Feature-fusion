"""
plot_best_accuracy_vs_k.py

Plots the best mean test_f1_macro per number of sensors (k) for each tag.
  X-axis: Number of sensors (k)
  Y-axis: Best mean F1 macro (combo with highest mean F1 across 10 seeds)

Tags plotted: clean_aug, clean, mixed
Data source: ablation_reports_{tag}/*_results_*.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

LOGS_DIR = Path(__file__).parents[2]

TAGS = ["mixed_train_tremor_mixed_all3"]

COLORS = {
    "clean_aug":              "#55A868",
    "clean":                  "#DD8452",
    "mixed":                  "#4C72B0",
    "clean_train_awgn_a000":  "#4C72B0",
    "clean_train_tremor_mixed_all3":  "#DD8452",
    "mixed_train_tremor_mixed_all3":  "#55A868",
}

LABELS = {
    "clean_aug":              "Clean Aug",
    "clean":                  "Clean",
    "mixed":                  "Mixed",
    "clean_train_awgn_a000":  "Clean train / AWGN a000",
    "clean_train_tremor_mixed_all3":  "Clean train / Tremor mixed",
    "mixed_train_tremor_mixed_all3":  "Mixed train / Tremor mixed",
}

SHOW_ANNOTATIONS = False  # Set to True to show F1 values on data points

_tag_str = "_".join(TAGS) if len(TAGS) <= 2 else f"{len(TAGS)}_tags"
OUTPUT_FILE = Path(__file__).parent / f"best_accuracy_vs_k_{_tag_str}{'_labels' if SHOW_ANNOTATIONS else ''}.png"

# ═══════════════════════════════════════════════════════════════


import json


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
                "num_sensors": int(float(row["num_sensors"])),
                "sensors": row["sensors"],
                "test_f1_macro": float(row["test_f1_macro"]),
            })
    return records


def load_tag(tag: str) -> list:
    # Prefer averaged JSONL (one mean F1 per combo)
    json_dir = LOGS_DIR / f"ablation_json_files_{tag}"
    averaged_file = json_dir / f"averaged_{tag}_all_k.jsonl"
    if averaged_file.exists():
        print(f"  [{tag}] Loading averaged JSONL: {averaged_file.name}")
        raw = load_jsonl(averaged_file)
        return [
            {
                "num_sensors": r["num_sensors"],
                "sensors": "|".join(r["sensors"]) if isinstance(r["sensors"], list) else r["sensors"],
                "test_f1_macro": r["test_f1_macro"],
            }
            for r in raw
        ]

    # Fall back to per-seed CSV files
    csv_dir = LOGS_DIR / f"ablation_reports_{tag}"
    paths = sorted(csv_dir.glob("*_results_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No data found for tag '{tag}' — checked:\n  {averaged_file}\n  {csv_dir}")
    print(f"  [{tag}] Loading {len(paths)} CSV files")
    records = []
    for p in paths:
        records.extend(load_csv(p))
    return records


def best_per_k(records: list) -> tuple:
    # Group by (num_sensors, sensors combo) -> list of F1 macro values
    by_combo = defaultdict(list)
    for r in records:
        key = (r["num_sensors"], r.get("sensors", ""))
        by_combo[key].append(r["test_f1_macro"])

    # For each k: find the combo with the highest mean accuracy
    by_k = defaultdict(list)
    for (k, sensors), accs in by_combo.items():
        by_k[k].append(np.mean(accs))

    k_values = sorted(by_k.keys())
    best = [max(by_k[k]) for k in k_values]
    return k_values, best


def best_combo_per_k(records: list) -> list:
    """Return list of (k, sensor_names, mean_f1) for the best combo at each k."""
    by_combo = defaultdict(list)
    for r in records:
        key = (r["num_sensors"], r.get("sensors", ""))
        by_combo[key].append(r["test_f1_macro"])

    by_k: dict = {}
    for (k, sensors), accs in by_combo.items():
        mean_f1 = np.mean(accs)
        if k not in by_k or mean_f1 > by_k[k][1]:
            by_k[k] = (sensors, mean_f1)

    return [(k, by_k[k][0], by_k[k][1]) for k in sorted(by_k.keys())]


# Short display names for sensors
_SENSOR_SHORT = {
    "Acc_ankle": "Acc ankle",
    "Acc_arm":   "Acc arm",
    "Acc_chest": "Acc chest",
    "ECG":       "ECG",
    "Gyro_ankle":"Gyro ankle",
    "Gyro_arm":  "Gyro arm",
    "Mag_ankle": "Mag ankle",
    "Mag_arm":   "Mag arm",
}


def _fmt_sensors(sensors_str: str) -> str:
    """Convert pipe-separated sensor string to comma-separated display names."""
    parts = [s.strip() for s in sensors_str.split("|")]
    return ", ".join(_SENSOR_SHORT.get(s, s) for s in parts)


def main():
    # ── Load data ──────────────────────────────────────────────
    all_data    = {}
    all_combos  = {}
    for tag in TAGS:
        records = load_tag(tag)
        all_data[tag]   = best_per_k(records)
        all_combos[tag] = best_combo_per_k(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in TAGS:
        k_values, best = all_data[tag]
        ax.plot(k_values, best, marker="o", linewidth=2, markersize=7,
                color=COLORS[tag], label=LABELS[tag])
        if SHOW_ANNOTATIONS:
            for k, acc in zip(k_values, best):
                ax.annotate(f"{acc:.3f}", xy=(k, acc), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=COLORS[tag])

    ax.set_xticks(sorted({k for k_values, _ in all_data.values() for k in k_values}))
    ax.set_xlabel("Number of sensors (k)", fontsize=13)
    ax.set_ylabel("Best mean F1 macro", fontsize=13)
    ax.set_title("Best Mean F1 Macro vs Number of Sensors", fontsize=14, pad=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"✓ Combined plot saved to: {OUTPUT_FILE}")
    plt.close()

    # ── Per-tag plots ──────────────────────────────────────────
    for tag in TAGS:
        k_values, best = all_data[tag]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(k_values, best, marker="o", linewidth=2, markersize=7,
                color=COLORS[tag], label=LABELS[tag])
        if SHOW_ANNOTATIONS:
            for k, acc in zip(k_values, best):
                ax.annotate(f"{acc:.3f}", xy=(k, acc), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=9,
                            color=COLORS[tag])
        ax.set_xticks(k_values)
        ax.set_xlabel("Number of sensors (k)", fontsize=13)
        ax.set_ylabel("Best mean F1 macro", fontsize=13)
        ax.set_title(f"Best Mean F1 Macro vs Number of Sensors — {LABELS[tag]}",
                     fontsize=14, pad=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        # No legend for single-tag per-tag plot
        plt.tight_layout()
        suffix = "" if SHOW_ANNOTATIONS else "_no_labels"
        out = Path(__file__).parent / f"best_accuracy_vs_k_{tag}{suffix}.png"
        plt.savefig(out, dpi=150)
        print(f"✓ {tag} plot saved to: {out}")
        plt.close()

    # ── LaTeX tables ───────────────────────────────────────────
    for tag in TAGS:
        rows = all_combos[tag]
        label = f"tab:best_combo_{tag}"
        tag_tex = tag.replace('_', '\\_')
        caption = (
            f"Best sensor combination per $k$ for the \\texttt{{{tag_tex}}} condition. "
            "For each sensor count $k$, the combination with the highest seed-averaged "
            "test F1 macro is shown."
        )
        lines = [
            "\\begin{table}[H]",
            "    \\centering",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "    \\begin{tabular}{clc}",
            "        \\toprule",
            "        $k$ & \\textbf{Sensors} & \\textbf{Mean F1} \\\\",
            "        \\midrule",
        ]
        for k, sensors_str, f1 in rows:
            sensor_display = _fmt_sensors(sensors_str)
            lines.append(f"        {k} & {sensor_display} & {f1:.4f} \\\\")
        lines += [
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
        tex = "\n".join(lines)
        out_tex = Path(__file__).parent / f"best_combo_table_{tag}.tex"
        out_tex.write_text(tex)
        print(f"✓ LaTeX table saved to: {out_tex}")
        print(tex)


if __name__ == "__main__":
    main()
