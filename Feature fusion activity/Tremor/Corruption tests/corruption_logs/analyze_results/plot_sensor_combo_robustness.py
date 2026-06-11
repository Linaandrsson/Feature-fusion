"""
plot_sensor_combo_robustness.py

Plots test F1 macro vs AWGN alpha for a specific sensor combination.
Edit the CONFIG section below and run with the VS Code run button.

Sensor names are case-insensitive.
k is inferred from the number of sensors provided.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = Path(__file__).parents[1]

# ── CONFIG — edit these before running ───────────────────────────────────────
MODEL_TYPE = "clean"          # "clean" or "mixed"
SENSORS    = ["Acc_ankle", "Acc_arm", "Gyro_ankle", "Mag_ankle", "Mag_arm", "Acc_chest", "ECG", "Gyro_arm"]  # list of sensor names (case-insensitive)
SHOW_STD   = False            # shade ± std band around line
SAVE_PNG   = True             # True = save to PNG, False = show interactively
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def alpha_str_to_float(alpha: str) -> float:
    """Convert alpha code string to float: '010' → 0.10, '032' → 0.32"""
    return int(alpha) / 100.0


def discover_alpha_variants(model_type: str) -> list[str]:
    """Auto-discover sorted alpha variant strings for a model type."""
    pattern = f"ablation_json_files_{model_type}_train_awgn_a*"
    prefix  = f"{model_type}_train_awgn_a"
    alphas  = []
    for folder in sorted(LOGS_DIR.glob(pattern)):
        name       = folder.name.replace("ablation_json_files_", "")
        alpha_part = name[len(prefix):]
        if "_" not in alpha_part:   # exclude suffixes like _AE
            alphas.append(alpha_part)
    return alphas


def tag_for(model_type: str, alpha: str) -> str:
    return f"{model_type}_train_awgn_a{alpha}"


def averaged_jsonl_path(model_type: str, alpha: str) -> Path:
    tag = tag_for(model_type, alpha)
    return LOGS_DIR / f"ablation_json_files_{tag}" / f"averaged_{tag}_all_k.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalise_sensors(sensors: list[str]) -> frozenset[str]:
    """Lower-case frozenset for comparison."""
    return frozenset(s.lower() for s in sensors)


def find_combo(records: list[dict], target_sensors: frozenset[str]) -> dict | None:
    """Return the record whose sensor set matches target_sensors (case-insensitive)."""
    for r in records:
        if normalise_sensors(r["sensors"]) == target_sensors:
            return r
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_type: str, sensors: list[str], show_std: bool, save: bool):
    k = len(sensors)
    target = normalise_sensors(sensors)

    print(f"\n{'='*65}")
    print(f"Sensor combo robustness plot")
    print(f"  Model type : {model_type}")
    print(f"  Sensors    : {sensors}  (k={k})")
    print(f"{'='*65}")

    # ── Discover alpha variants ───────────────────────────────────────────
    all_alphas = discover_alpha_variants(model_type)
    if not all_alphas:
        print(f"[ERROR] No folders found for model_type='{model_type}'")
        sys.exit(1)

    # ── Collect F1 per alpha ──────────────────────────────────────────────
    x_vals   = []   # alpha float
    y_vals   = []   # test_f1_macro
    y_stds   = []   # test_f1_macro_std (std across seeds, already averaged)

    for alpha in all_alphas:
        p = averaged_jsonl_path(model_type, alpha)
        if not p.exists():
            print(f"  [skip] a{alpha} — averaged JSONL not found: {p}")
            continue

        records = load_jsonl(p)
        # Filter to correct k first for speed
        k_records = [r for r in records if r.get("num_sensors") == k]
        match = find_combo(k_records, target)

        if match is None:
            print(f"  [!]    a{alpha} — sensor combo not found in k={k} records")
            continue

        f1     = match["test_f1_macro"]
        f1_std = match.get("test_f1_macro_std", 0.0) or 0.0
        x_vals.append(alpha_str_to_float(alpha))
        y_vals.append(f1)
        y_stds.append(f1_std)
        print(f"  [✓]    a{alpha} ({alpha_str_to_float(alpha):.2f}) → F1 = {f1:.4f} ± {f1_std:.4f}")

    if not x_vals:
        print("[ERROR] No data points found for the specified sensor combination.")
        sys.exit(1)

    x = np.array(x_vals)
    y = np.array(y_vals)
    e = np.array(y_stds)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    sensor_label = ", ".join(sensors)

    if show_std:
        ax.fill_between(x, y - e, y + e, alpha=0.2)
        ax.plot(x, y, marker="o", linewidth=2)
    else:
        ax.plot(x, y, marker="o", linewidth=2)

    ax.set_xlabel("AWGN alpha (noise level)", fontsize=12)
    ax.set_ylabel("Test F1 Macro", fontsize=12)
    ax.set_title(
        #f"AWGN robustness — {model_type} model\n"
        f"AWGN robustness\n"
        f"Sensors: {sensor_label}  (k={k})",
        fontsize=12
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save:
        sensors_slug = "_".join(s.lower() for s in sensors)
        out_path = Path(__file__).parent / f"robustness_{model_type}_k{k}_{sensors_slug}.png"
        fig.savefig(out_path, dpi=150)
        print(f"\n  Saved → {out_path}")
    else:
        plt.show()

    print(f"{'='*65}\n")


if __name__ == "__main__":
    run(
        model_type=MODEL_TYPE,
        sensors=SENSORS,
        show_std=SHOW_STD,
        save=SAVE_PNG,
    )
