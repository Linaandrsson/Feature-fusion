"""
Pareto curve: best Test F1 (macro) vs. total sampling frequency (fs_total)
for each FS study scenario (Clean→Clean, Clean→Tremor, Tremor→Tremor).

fs_total = fAcc_ankle + fGyro_arm + fMag_ankle + fMag_arm
"""

import json
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

# Set to True to plot mean F1 across seeds (aggregated); False = best single-seed F1
AGGREGATE = True

SCENARIOS = {
    "Tremor→Tremor": os.path.join(
        BASE, "tremor_logs_tremor_tremor",
        "tremor_fusion_fs_combo_study_tremor_tremor.jsonl"),
}

_suffix = "agg_tremor_tremor" if AGGREGATE else "tremor_tremor"
OUT_PNG = os.path.join(BASE, f"pareto_best_f1_vs_fs_total_{_suffix}.png")
OUT_PDF = os.path.join(BASE, f"pareto_best_f1_vs_fs_total_{_suffix}.pdf")

# ---------------------------------------------------------------------------
# Parse each scenario
# ---------------------------------------------------------------------------
def load_pareto(jsonl_path):
    """Return sorted (fs_total, best_f1) list from a JSONL log (single-seed best)."""
    best = collections.defaultdict(float)
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fs_map = rec["fs_map"]
            fs_total = sum(fs_map.values())
            f1 = rec.get("test_f1_macro", rec.get("test_f1", 0.0))
            if f1 > best[fs_total]:
                best[fs_total] = f1
    return sorted(best.items())   # list of (fs_total, best_f1)


def load_pareto_agg(jsonl_path):
    """Return sorted (fs_total, mean_f1, std_f1) list, averaged across seeds per combo."""
    import statistics
    combo_f1s = collections.defaultdict(list)   # hash -> [f1, ...]
    combo_fs_total = {}                          # hash -> fs_total
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            h = rec["fs_combo_hash"]
            fs_total = sum(rec["fs_map"].values())
            f1 = rec.get("test_f1_macro", rec.get("test_f1", 0.0))
            combo_f1s[h].append(f1)
            combo_fs_total[h] = fs_total
    # best mean F1 per fs_total
    best_mean = collections.defaultdict(float)
    best_std  = collections.defaultdict(float)
    for h, f1s in combo_f1s.items():
        t = combo_fs_total[h]
        m = statistics.mean(f1s)
        s = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        if m > best_mean[t]:
            best_mean[t] = m
            best_std[t]  = s
    return sorted((t, best_mean[t], best_std[t]) for t in best_mean)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {
    "Tremor→Tremor": "#4CAF50",   # green
}
MARKERS = {
    "Tremor→Tremor": "o",
}

fig, ax = plt.subplots(figsize=(8, 5))

for scenario, path in SCENARIOS.items():
    if not os.path.exists(path):
        print(f"[SKIP] {scenario}: file not found → {path}")
        continue
    if AGGREGATE:
        pts = load_pareto_agg(path)
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ax.plot(xs, ys,
                color=COLORS[scenario],
                marker=MARKERS[scenario],
                markersize=7,
                linewidth=1.8,
                label=scenario)
    else:
        pts = load_pareto(path)
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ax.plot(xs, ys,
                color=COLORS[scenario],
                marker=MARKERS[scenario],
                markersize=7,
                linewidth=1.8,
                label=scenario)
    # annotate the best point
    best_idx = ys.index(max(ys))
    ax.annotate(f"{ys[best_idx]:.2f}%",
                xy=(xs[best_idx], ys[best_idx]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8, color=COLORS[scenario])

ax.set_xlabel("Total Sampling Frequency  $f_{\\mathrm{total}}$ [Hz]", fontsize=12)
_ylabel = "Mean Test F1 (macro) \u00b1 std [%]" if AGGREGATE else "Best Test F1 (macro) [%]"
ax.set_ylabel(_ylabel, fontsize=12)
_title_mode = "Mean F1 across seeds" if AGGREGATE else "Best single-seed F1"
ax.set_title(f"Pareto Curve: {_title_mode} vs. Total Sampling Frequency\n(Tremor\u2192Tremor)", fontsize=13)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
print(f"Saved → {OUT_PDF}")

# ---------------------------------------------------------------------------
# Print table to console (useful for LaTeX)
# ---------------------------------------------------------------------------
print("\nfs_total  | Tremor→Tremor (mean \u00b1 std)" if AGGREGATE else "\nfs_total  | Tremor→Tremor")
print("-" * 40)
if AGGREGATE:
    pts = load_pareto_agg(SCENARIOS["Tremor→Tremor"])
    for t, mean_f1, std_f1 in pts:
        print(f"{t:9d}    {mean_f1*100:8.2f}% \u00b1 {std_f1*100:.2f}%")
else:
    pts = load_pareto(SCENARIOS["Tremor→Tremor"])
    for t, f1 in pts:
        print(f"{t:9d}    {f1*100:8.2f}%")
