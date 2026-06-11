"""
plot_k4_clean_vs_mixed.py
--------------------------
Plots all k=4 sensor combinations from both the clean-trained and mixed-trained
AWGN robustness JSONL files on the same axes.

  • Clean-trained combos → blue family (light, thin lines) + bold best combo
  • Mixed-trained combos → orange/red family (light, thin lines) + bold best combo

The best combo per model type (highest mean F1 across all alpha levels) is
highlighted with a thicker line and labelled in the legend.

Usage:
    python plot_k4_clean_vs_mixed.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--k", type=int, default=4)
K_TARGET       = _parser.parse_args().k
EXCLUDE_ALPHAS = []          # alpha codes to skip, e.g. ["020"]
SAVE_PNG       = True
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parents[1]

JSONL = {
    "clean": LOGS_DIR / "ablation_json_files_clean_train_awgn_robustness"
                      / "robustness_clean_train_awgn_robustness_all_k.jsonl",
    "mixed": LOGS_DIR / "ablation_json_files_mixed_train_awgn_robustness"
                      / "robustness_mixed_train_awgn_robustness_all_k.jsonl",
}

PALETTE = {
    "clean": {"best": "#1f77b4",   "rest": "#aec7e8"},   # blue family
    "mixed": {"best": "#d62728",   "rest": "#f5b7b1"},   # red/orange family
}


def load_k4(path: Path, k: int) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["num_sensors"] == k:
                records.append(rec)
    return records


def write_table(best: dict, alphas_str: list, out_path: Path) -> None:
    """Write a plain-text table: AWGN alpha | clean F1 | mixed F1."""
    clean_rec = best.get("clean")
    mixed_rec = best.get("mixed")

    lines = []
    lines.append(f"AWGN robustness — best k={K_TARGET} combo per model type")
    lines.append(f"{'AWGN alpha':<12}  {'Clean F1':<10}  {'Mixed F1':<10}")
    lines.append("-" * 36)
    for a in alphas_str:
        alpha_val = f"{int(a)/100:.2f}"
        c = f"{clean_rec['test_f1_per_alpha'].get(a, float('nan')):.4f}" if clean_rec else "N/A"
        m = f"{mixed_rec['test_f1_per_alpha'].get(a, float('nan')):.4f}" if mixed_rec else "N/A"
        lines.append(f"{alpha_val:<12}  {c:<10}  {m:<10}")
    lines.append("-" * 36)
    if clean_rec:
        lines.append(f"{'Mean':<12}  {clean_rec['test_f1_macro']:.4f}      "
                     f"{mixed_rec['test_f1_macro']:.4f}" if mixed_rec else "")
        lines.append("")
        lines.append(f"Clean best sensors : {', '.join(clean_rec['sensors'])}")
    if mixed_rec:
        lines.append(f"Mixed best sensors : {', '.join(mixed_rec['sensors'])}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Table saved → {out_path}")


def run():
    fig, ax = plt.subplots(figsize=(11, 6))

    all_alphas_str = None
    best_recs: dict = {}
    legend_handles = []

    for model_type in ("clean", "mixed"):
        path = JSONL[model_type]
        if not path.exists():
            print(f"[WARN] Not found: {path}")
            continue

        records = load_k4(path, K_TARGET)
        if not records:
            print(f"[WARN] No k={K_TARGET} records in {path.name}")
            continue

        # Alpha axis (same for all records)
        sample = records[0]
        alphas_str   = [a for a in sorted(sample["alpha_levels_used"])
                        if a not in EXCLUDE_ALPHAS]
        alphas_float = [int(a) / 100.0 for a in alphas_str]
        if all_alphas_str is None:
            all_alphas_str   = alphas_str
            all_alphas_float = alphas_float

        best_rec = max(records, key=lambda r: r["test_f1_macro"])
        best_recs[model_type] = best_rec

        colors = PALETTE[model_type]

        # ── Draw best combo ───────────────────────────────────────────────
        y_best = np.array([best_rec["test_f1_per_alpha"].get(a, np.nan)
                           for a in alphas_str])
        sensors_label = ", ".join(best_rec["sensors"])
        line, = ax.plot(alphas_float, y_best,
                        color=colors["best"],
                        linewidth=2.5,
                        marker="o",
                        markersize=5,
                        zorder=3,
                        label=f"{model_type}-trained")
        legend_handles.append(line)

        print(f"\n[{model_type}] k={K_TARGET}  —  {len(records)} combinations")
        print(f"  Best: {sensors_label}")
        print(f"  Mean F1: {best_rec['test_f1_macro']:.4f}")
        for a, v in zip(alphas_str, y_best):
            print(f"    α={int(a)/100:.2f}: {v:.4f}")

    ax.set_xlabel("AWGN alpha (noise level)", fontsize=12)
    ax.set_ylabel("Test F1 Macro", fontsize=12)
    ax.set_title(
        f"AWGN robustness — k={K_TARGET} sensor combinations\n"
        f"Clean-trained vs. mixed-trained feature extractors",
        fontsize=12,
    )
    ax.set_ylim(0, 1.02)
    if all_alphas_float is not None:
        ax.set_xticks(all_alphas_float)
        ax.set_xticklabels([f"{v:.2f}" for v in all_alphas_float], rotation=45)
    ax.legend(handles=legend_handles, fontsize=8,
              loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    if SAVE_PNG:
        out_path = Path(__file__).parent / f"robustness_k{K_TARGET}_clean_vs_mixed.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved → {out_path}")

    if all_alphas_str is not None:
        tbl_path = Path(__file__).parent / f"robustness_k{K_TARGET}_clean_vs_mixed.txt"
        write_table(best_recs, all_alphas_str, tbl_path)
    else:
        plt.show()


if __name__ == "__main__":
    run()
