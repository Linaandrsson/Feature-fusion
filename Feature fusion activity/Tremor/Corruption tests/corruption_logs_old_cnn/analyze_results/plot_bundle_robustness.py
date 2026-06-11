"""
plot_bundle_robustness.py

Compacted figure combining sensor-combo tables and robustness lines.

For each k (number of sensors) a colored line shows F1 vs AWGN alpha.
At the END of each line one ring (colored to match the line) encloses
colored sensor dots showing which sensors are in the best combination
at that final alpha level.

Legend: sensor colors (dots) + k colors (lines).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np

LOGS_DIR = Path(__file__).parents[1]

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_TYPE = "clean"   # "clean" or "mixed"
SAVE_PNG   = True
# ─────────────────────────────────────────────────────────────────────────────

ALL_SENSORS = ["Acc_ankle", "Acc_arm", "Gyro_ankle", "Gyro_arm",
               "Mag_ankle", "Mag_arm", "Acc_chest", "ECG"]

SENSOR_COLORS = {
    "Acc_ankle":  "#e41a1c",   # red
    "Acc_arm":    "#377eb8",   # blue
    "Gyro_ankle": "#4daf4a",   # green
    "Gyro_arm":   "#984ea3",   # purple
    "Mag_ankle":  "#ff7f00",   # orange
    "Mag_arm":    "#a65628",   # brown
    "Acc_chest":  "#f781bf",   # pink
    "ECG":        "#999999",   # grey
}

# One distinct color per k value (tab10 palette, skipping greys)
K_COLORS = [
    "#1f77b4",  # k=1  steel blue
    "#ff7f0e",  # k=2  orange
    "#2ca02c",  # k=3  green
    "#d62728",  # k=4  red
    "#9467bd",  # k=5  purple
    "#8c564b",  # k=6  brown
    "#e377c2",  # k=7  pink
    "#17becf",  # k=8  teal
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def alpha_str_to_float(alpha: str) -> float:
    return int(alpha) / 100.0


def discover_alpha_variants(model_type: str) -> list[str]:
    pattern = f"ablation_json_files_{model_type}_train_awgn_a*"
    prefix  = f"{model_type}_train_awgn_a"
    alphas  = []
    for folder in sorted(LOGS_DIR.glob(pattern)):
        name       = folder.name.replace("ablation_json_files_", "")
        alpha_part = name[len(prefix):]
        if "_" not in alpha_part:
            alphas.append(alpha_part)
    return alphas


def averaged_jsonl_path(model_type: str, alpha: str) -> Path:
    tag = f"{model_type}_train_awgn_a{alpha}"
    return LOGS_DIR / f"ablation_json_files_{tag}" / f"averaged_{tag}_all_k.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dot_positions(n: int, rx: float, ry: float) -> list[tuple[float, float]]:
    """Return (dx, dy) offsets for n dots arranged in a ring of radius (rx, ry).
    For n=1 the dot is at center."""
    if n == 1:
        return [(0.0, 0.0)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return [(rx * np.cos(a), ry * np.sin(a)) for a in angles]


def compute_data_radii(ax, fig, visual_radius_in: float = 0.13) -> tuple[float, float]:
    """Convert a fixed visual radius (inches) to data-unit radii (rx, ry) so the
    enclosing shape appears circular regardless of x/y axis scales."""
    bbox   = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_in = fig_w * bbox.width
    ax_h_in = fig_h * bbox.height
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    rx = visual_radius_in / ax_w_in * x_range
    ry = visual_radius_in / ax_h_in * y_range
    return rx, ry


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    alphas = discover_alpha_variants(MODEL_TYPE)
    if not alphas:
        print(f"[ERROR] No alpha variants found for model_type='{MODEL_TYPE}'")
        sys.exit(1)

    # Load all records
    alpha_records: dict[str, list[dict]] = {}
    available_alphas: list[str] = []
    for alpha in alphas:
        p = averaged_jsonl_path(MODEL_TYPE, alpha)
        if p.exists():
            alpha_records[alpha] = load_jsonl(p)
            available_alphas.append(alpha)

    if not available_alphas:
        print("[ERROR] No averaged JSONL files found. Run aggregate_multirun.py first.")
        sys.exit(1)

    k_values = sorted({r["num_sensors"]
                       for recs in alpha_records.values()
                       for r in recs})
    x_ticks = [alpha_str_to_float(a) for a in available_alphas]

    # ── For each k: find the ONE sensor combo with best average F1 across all alphas ──
    # Collect all unique combos per k, compute their mean F1 across alpha levels.
    best_combo_per_k: dict[int, tuple[frozenset, list[str]]] = {}
    best_series_per_k: dict[int, list[tuple[float, float]]] = {}

    for k in k_values:
        # Gather all (frozenset→combo, per-alpha F1) entries
        combo_f1s: dict[frozenset, dict[str, float]] = {}
        combo_sensors: dict[frozenset, list[str]] = {}

        for alpha in available_alphas:
            k_recs = [r for r in alpha_records[alpha] if r.get("num_sensors") == k]
            for r in k_recs:
                key = frozenset(s.lower() for s in r["sensors"])
                if key not in combo_f1s:
                    combo_f1s[key] = {}
                    combo_sensors[key] = r["sensors"]
                combo_f1s[key][alpha] = r.get("test_f1_macro") or 0.0

        if not combo_f1s:
            continue

        # Pick combo with highest mean F1 (only average over alphas where it appears)
        best_key = max(combo_f1s, key=lambda k_: np.mean(list(combo_f1s[k_].values())))
        best_combo_per_k[k] = (best_key, combo_sensors[best_key])

        # Build (alpha_float, f1) series for the chosen combo
        series = []
        for alpha in available_alphas:
            if alpha in combo_f1s[best_key]:
                series.append((alpha_str_to_float(alpha), combo_f1s[best_key][alpha]))
        best_series_per_k[k] = series

        print(f"k={k}  best overall combo: {combo_sensors[best_key]}")
        print(f"       mean F1 = {np.mean(list(combo_f1s[best_key].values())):.4f}")

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_xlabel("AWGN α  (noise level)", fontsize=12)
    ax.set_ylabel("Test F1 Macro", fontsize=12)
    ax.set_title(
        f"AWGN Robustness — Best Overall Sensor Combination per k\n"
        f"(model_type = '{MODEL_TYPE}',  ring = sensors in combination)",
        fontsize=12
    )
    ax.set_xlim(-0.08, 1.22)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:.2f}" for v in x_ticks], rotation=45)
    ax.grid(True, alpha=0.25, zorder=0)

    # Draw colored lines (one per k)
    k_color_map = {k: K_COLORS[i % len(K_COLORS)] for i, k in enumerate(k_values)}
    line_handles = []
    for k in k_values:
        series = best_series_per_k.get(k, [])
        if not series:
            continue
        xs = [s[0] for s in series]
        ys = [s[1] for s in series]
        color = k_color_map[k]
        line, = ax.plot(xs, ys,
                        color=color,
                        linewidth=2.0,
                        marker=".",
                        markersize=5,
                        zorder=2,
                        label=f"k={k}")
        line_handles.append(line)

    fig.tight_layout()
    fig.canvas.draw()   # finalise layout before computing radii

    rx, ry = compute_data_radii(ax, fig, visual_radius_in=0.14)

    DOT_RING_SCALE = 0.55
    CIRCLE_SCALE   = 1.55
    DOT_SIZE       = 60

    # Draw ONE ring per k — distributed across different alpha positions
    # k_values[0] gets ring at series[0], k_values[1] at series[1], etc.
    for k_idx, k in enumerate(k_values):
        series = best_series_per_k.get(k, [])
        sensors = best_combo_per_k.get(k, (None, []))[1]
        if not series or not sensors:
            continue

        ring_idx = k_idx % len(series)
        ring_x, ring_y = series[ring_idx]
        color = k_color_map[k]
        offsets = dot_positions(len(sensors), rx * DOT_RING_SCALE, ry * DOT_RING_SCALE)

        # Enclosing circle — colored to match line
        ellipse = Ellipse(
            xy=(ring_x, ring_y),
            width=2 * rx * CIRCLE_SCALE,
            height=2 * ry * CIRCLE_SCALE,
            edgecolor=color,
            facecolor="white",
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
            clip_on=False,
        )
        ax.add_patch(ellipse)

        # Colored sensor dots inside
        for (dx, dy), sensor in zip(offsets, sensors):
            dot_color = SENSOR_COLORS.get(sensor, "#000000")
            sc = ax.scatter(ring_x + dx, ring_y + dy,
                            color=dot_color, s=DOT_SIZE,
                            zorder=4, edgecolors="white", linewidths=0.4)
            sc.set_clip_on(False)

    # ── Legends ───────────────────────────────────────────────────────────────
    sensor_handles = [
        mpatches.Patch(facecolor=SENSOR_COLORS[s], edgecolor="grey",
                       linewidth=0.5, label=s)
        for s in ALL_SENSORS
    ]
    leg1 = ax.legend(handles=sensor_handles, title="Sensors",
                     loc="upper right", fontsize=9, title_fontsize=10,
                     framealpha=0.92, ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=line_handles, title="k (# sensors)",
              loc="center right", fontsize=9, title_fontsize=10,
              framealpha=0.92)

    # ── Save / show ───────────────────────────────────────────────────────────
    if SAVE_PNG:
        out_path = Path(__file__).parent / f"bundle_robustness_{MODEL_TYPE}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"\nSaved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    run()

