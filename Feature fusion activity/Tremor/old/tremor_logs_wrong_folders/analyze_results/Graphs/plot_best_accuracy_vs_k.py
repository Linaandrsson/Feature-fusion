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

TAGS = ["clean_aug", "clean", "mixed"]

COLORS = {
    "clean_aug": "#55A868",
    "clean":     "#DD8452",
    "mixed":     "#4C72B0",
}

LABELS = {
    "clean_aug": "Clean Aug",
    "clean":     "Clean",
    "mixed":     "Mixed",
}

SHOW_ANNOTATIONS = False  # Set to True to show F1 values on data points

OUTPUT_FILE = Path(__file__).parent / ("best_accuracy_vs_k.png" if SHOW_ANNOTATIONS else "best_accuracy_vs_k_no_labels.png")

# ═══════════════════════════════════════════════════════════════


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
    csv_dir = LOGS_DIR / f"ablation_reports_{tag}"
    paths = sorted(csv_dir.glob("*_results_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
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


def main():
    # ── Combined plot ──────────────────────────────────────────
    all_data = {}
    for tag in TAGS:
        records = load_tag(tag)
        all_data[tag] = best_per_k(records)

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

    # ── clean_aug vs mixed only ────────────────────────────────
    subset_tags = ["clean_aug", "mixed"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in subset_tags:
        k_values, best = all_data[tag]
        ax.plot(k_values, best, marker="o", linewidth=2, markersize=7,
                color=COLORS[tag], label=LABELS[tag])
        if SHOW_ANNOTATIONS:
            for k, acc in zip(k_values, best):
                ax.annotate(f"{acc:.3f}", xy=(k, acc), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=COLORS[tag])
    ax.set_xticks(sorted({k for t in subset_tags for k in all_data[t][0]}))
    ax.set_xlabel("Number of sensors (k)", fontsize=13)
    ax.set_ylabel("Best mean F1 macro", fontsize=13)
    ax.set_title("Best Mean F1 Macro vs Number of Sensors — Clean Aug vs Mixed",
                 fontsize=14, pad=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    suffix = "" if SHOW_ANNOTATIONS else "_no_labels"
    out = Path(__file__).parent / f"best_accuracy_vs_k_clean_aug_vs_mixed{suffix}.png"
    plt.savefig(out, dpi=150)
    print(f"✓ Clean Aug vs Mixed plot saved to: {out}")
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
        plt.tight_layout()
        suffix = "" if SHOW_ANNOTATIONS else "_no_labels"
        out = Path(__file__).parent / f"best_accuracy_vs_k_{tag}{suffix}.png"
        plt.savefig(out, dpi=150)
        print(f"✓ {tag} plot saved to: {out}")
        plt.close()


if __name__ == "__main__":
    main()
