"""
plot_all_k_robustness.py

Plots the best sensor combination for each k (k=1..8) on the same axes,
showing test F1 macro vs AWGN alpha. Best combo per k is taken from the
cross-alpha robustness report (highest mean F1 across all alpha levels).

Edit CONFIG below and run with the VS Code run button.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_TYPE      = "mixed"   # "clean" or "mixed"
SAVE_PNG        = True      # True = save PNG, False = show interactively
EXCLUDE_ALPHAS  = ["020"]   # alpha codes to skip, e.g. ["018", "020"]
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parents[1]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run():
    robustness_path = (
        LOGS_DIR
        / f"ablation_json_files_{MODEL_TYPE}_train_awgn_robustness"
        / f"robustness_{MODEL_TYPE}_train_awgn_robustness_all_k.jsonl"
    )

    if not robustness_path.exists():
        print(f"[ERROR] Robustness JSONL not found: {robustness_path}")
        print("Run aggregate_awgn_robustness.py first.")
        sys.exit(1)

    records = load_jsonl(robustness_path)

    # ── Find best combo per k ─────────────────────────────────────────────
    k_values = sorted({r["num_sensors"] for r in records})
    best_per_k: dict[int, dict] = {}
    for k in k_values:
        k_recs = [r for r in records if r["num_sensors"] == k]
        best_per_k[k] = max(k_recs, key=lambda r: r.get("test_f1_macro") or 0)

    # Get sorted alpha list from first record (all have same set)
    sample = next(iter(best_per_k.values()))
    alphas_str   = [a for a in sorted(sample["alpha_levels_used"]) if a not in EXCLUDE_ALPHAS]
    alphas_float = [int(a) / 100.0 for a in alphas_str]

    print(f"\nModel type : {MODEL_TYPE}")
    print(f"Alpha levels: {['a'+a for a in alphas_str]}")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(k_values))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for color, k in zip(colors, k_values):
        best   = best_per_k[k]
        sensor_label = ", ".join(best["sensors"])
        f1_vals = [best["test_f1_per_alpha"].get(a) for a in alphas_str]

        # Replace None with NaN for plotting
        y = np.array([v if v is not None else np.nan for v in f1_vals])

        ax.plot(
            alphas_float, y,
            marker="o",
            linewidth=2,
            color=color,
            label=f"k={k}",
        )

        print(f"k={k}: {sensor_label}")
        print(f"  Mean F1: {best['test_f1_macro']:.4f}")
        for a, v in zip(alphas_str, f1_vals):
            print(f"    a{a}: {v:.4f}" if v is not None else f"    a{a}: N/A")
        print()

    ax.set_xlabel("AWGN alpha (noise level)", fontsize=12)
    ax.set_ylabel("Test F1 Macro", fontsize=12)
    ax.set_title(
        f"AWGN robustness — best sensor combination per k\n",
        fontsize=12,
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(alphas_float)
    ax.set_xticklabels([f"{v:.2f}" for v in alphas_float], rotation=45)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if SAVE_PNG:
        out_path = Path(__file__).parent / f"robustness_{MODEL_TYPE}_all_k_overlay.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    run()
