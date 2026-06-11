"""
aggregate_awgn_robustness.py

Cross-alpha AWGN robustness report.

For a given model type ("clean" or "mixed"), auto-discovers all
ablation_json_files_{model_type}_train_awgn_a???/ directories, loads
the averaged per-combo JSONL files, then for each sensor combination + k:
averages test_f1_macro across all alpha levels.
Produces per-k TXT reports ranked by mean cross-alpha F1.

Usage:
  python aggregate_awgn_robustness.py                     # model_type=clean
  python aggregate_awgn_robustness.py --model_type clean
  python aggregate_awgn_robustness.py --model_type mixed
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
]

META_FIELDS = ["sensors", "ablated_sensors", "num_sensors", "fusion_mode"]


def discover_alpha_variants(model_type: str) -> list[str]:
    """Auto-discover available alpha variants for a model type.

    Looks for folders matching ablation_json_files_{model_type}_train_awgn_a*/
    (excluding folders with extra suffixes like _AE).
    Returns sorted list of alpha strings, e.g. ['010', '018', '020', ...].
    """
    pattern = f"ablation_json_files_{model_type}_train_awgn_a*"
    prefix  = f"{model_type}_train_awgn_a"
    alphas  = []
    for folder in sorted(LOGS_DIR.glob(pattern)):
        name = folder.name.replace("ablation_json_files_", "")
        # Strip the prefix to get the alpha suffix
        alpha_part = name[len(prefix):]
        # Exclude anything with extra underscores (e.g. _AE)
        if "_" not in alpha_part:
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


def combo_key(record: dict) -> tuple:
    return tuple(sorted(record["sensors"]))


def run(model_type: str):
    print(f"\n{'='*70}")
    print(f"Cross-alpha AWGN robustness aggregation — model_type='{model_type}'")
    print(f"{'='*70}")

    # ── Auto-discover available alpha variants ────────────────────────────
    all_alphas = discover_alpha_variants(model_type)
    if not all_alphas:
        print(f"[ERROR] No folders found matching ablation_json_files_{model_type}_train_awgn_a*/")
        sys.exit(1)

    available_alphas = []
    for alpha in all_alphas:
        p = averaged_jsonl_path(model_type, alpha)
        if p.exists():
            available_alphas.append(alpha)
            print(f"  [✓] a{alpha}  →  {p.name}")
        else:
            print(f"  [✗] a{alpha}  →  NOT FOUND (run aggregate_multirun.py --tag {tag_for(model_type, alpha)})")

    if not available_alphas:
        print("[ERROR] No averaged JSONL files found. Run aggregate_multirun.py first.")
        sys.exit(1)

    print(f"\n  Using {len(available_alphas)} alpha levels: {['a'+a for a in available_alphas]}")

    # ── Load all records ─────────────────────────────────────────────────
    # Structure: {combo_key: {alpha: record}}
    combo_db_records: dict[tuple, dict[str, dict]] = defaultdict(dict)

    for alpha in available_alphas:
        records = load_jsonl(averaged_jsonl_path(model_type, alpha))
        for r in records:
            combo_db_records[combo_key(r)][alpha] = r

    print(f"  Total unique sensor combinations: {len(combo_db_records)}")

    # ── Aggregate across alpha levels ─────────────────────────────────────
    aggregated = []
    for key, alpha_map in combo_db_records.items():
        alphas_present = sorted(alpha_map.keys())
        first = alpha_map[alphas_present[0]]

        agg = {f: first[f] for f in META_FIELDS}

        for field in NUMERIC_FIELDS:
            vals_per_alpha = []
            for alpha in alphas_present:
                v = alpha_map[alpha].get(field)
                if v is not None:
                    vals_per_alpha.append(v)
            if vals_per_alpha:
                agg[field] = float(np.mean(vals_per_alpha))
                agg[f"{field}_std_across_alpha"] = float(np.std(vals_per_alpha))
            else:
                agg[field] = None
                agg[f"{field}_std_across_alpha"] = None

        agg["alpha_levels_used"] = alphas_present
        agg["num_alpha_levels"] = len(alphas_present)

        # Per-alpha breakdown for test_f1_macro
        agg["test_f1_per_alpha"] = {
            alpha: alpha_map[alpha].get("test_f1_macro") for alpha in alphas_present
        }
        aggregated.append(agg)

    # ── Output dirs ───────────────────────────────────────────────────────
    out_tag      = f"{model_type}_train_awgn_robustness"
    json_out_dir = LOGS_DIR / f"ablation_json_files_{out_tag}"
    rep_out_dir  = LOGS_DIR / f"ablation_reports_{out_tag}"
    json_out_dir.mkdir(exist_ok=True)
    rep_out_dir.mkdir(exist_ok=True)

    # ── Per-k reports ─────────────────────────────────────────────────────
    k_values = sorted({r["num_sensors"] for r in aggregated})
    ts = datetime.now().strftime("%m%d_%H%M")

    all_for_jsonl = []

    for k in k_values:
        k_recs = [r for r in aggregated if r["num_sensors"] == k]
        k_recs.sort(key=lambda x: x["test_f1_macro"] or 0, reverse=True)

        txt = make_txt_report(k_recs, k=k, model_type=model_type,
                              available_alphas=available_alphas)
        txt_path = rep_out_dir / f"robustness_{out_tag}_k{k}_report.txt"
        txt_path.write_text(txt)
        print(f"  [k={k}]  {len(k_recs)} combos → {txt_path.name}")

        for c in k_recs:
            c["experiment_id"] = f"robustness_{out_tag}_k{k}_{ts}"
        all_for_jsonl.extend(k_recs)

    # ── Combined JSONL ────────────────────────────────────────────────────
    all_for_jsonl.sort(key=lambda x: (x["num_sensors"], -(x["test_f1_macro"] or 0)))
    jsonl_path = json_out_dir / f"robustness_{out_tag}_all_k.jsonl"
    with open(jsonl_path, "w") as fh:
        for c in all_for_jsonl:
            fh.write(json.dumps(c) + "\n")

    print(f"\n  JSONL → {jsonl_path}")
    print(f"  Done. {len(all_for_jsonl)} combinations written.")
    print(f"{'='*70}\n")


def make_txt_report(combos: list[dict], k: int, model_type: str,
                    available_alphas: list[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    alpha_str = ", ".join(f"a{a}" for a in available_alphas)

    lines = []
    lines.append("=" * 80)
    lines.append("AWGN ROBUSTNESS REPORT  [CROSS-ALPHA AVERAGE]")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated    : {now}")
    lines.append(f"Model type   : {model_type}")
    lines.append(f"K            : {k}")
    lines.append(f"Alpha levels : {alpha_str}  ({len(available_alphas)} levels)")
    lines.append(f"Metric       : mean test F1 macro across all alpha levels")
    lines.append(f"Total combinations: {len(combos)}")
    lines.append("")

    # Column widths
    sensor_strs  = [", ".join(c["sensors"])        for c in combos]
    ablated_strs = [", ".join(c["ablated_sensors"]) for c in combos]
    w_s = max(len("Sensors"), max(len(s) for s in sensor_strs))
    w_a = max(len("Ablated"), max(len(s) for s in ablated_strs))

    # Per-alpha column headers (e.g. "a010", "a020", ...)
    alpha_col_w = max(6, max(len(f"a{a}") for a in available_alphas))
    alpha_header = "  ".join(f"a{a:>{alpha_col_w-1}}" for a in available_alphas)
    total_w = 6 + 1 + w_s + 2 + w_a + 2 + 11 + 2 + len(alpha_header)
    sep  = "=" * total_w
    dash = "-" * total_w

    hdr = (
        f"{'Rank':<6} {'Sensors':<{w_s}}  {'Ablated':<{w_a}}  "
        f"{'Mean F1':>9} {'±':>1}  {alpha_header}"
    )

    lines.append(sep)
    lines.append("RANKED BY MEAN TEST F1 ACROSS ALL AWGN ALPHA LEVELS")
    lines.append(sep)
    lines.append(hdr)
    lines.append(dash)

    for rank, c, s_str, a_str in zip(range(1, len(combos)+1), combos, sensor_strs, ablated_strs):
        mean_f1  = c["test_f1_macro"]
        std_f1   = c["test_f1_macro_std_across_alpha"]
        per_alpha = c["test_f1_per_alpha"]

        alpha_vals = "  ".join(
            f"{per_alpha.get(alpha):>{alpha_col_w+1}.4f}" if per_alpha.get(alpha) is not None
            else f"{'N/A':>{alpha_col_w+1}}"
            for alpha in available_alphas
        )

        lines.append(
            f"{rank:<6} {s_str:<{w_s}}  {a_str:<{w_a}}  "
            f"{mean_f1:>9.4f} {std_f1:>4.4f}  {alpha_vals}"
        )

    # Best combo summary
    best = combos[0]
    lines.append("")
    lines.append("=" * 80)
    lines.append("BEST COMBINATION (by mean cross-alpha test F1)")
    lines.append("=" * 80)
    lines.append(f"  Sensors      : {', '.join(best['sensors'])}")
    lines.append(f"  Ablated      : {', '.join(best['ablated_sensors'])}")
    lines.append(f"  Mean F1      : {best['test_f1_macro']:.4f} ± {best['test_f1_macro_std_across_alpha']:.4f}")
    lines.append(f"  Per alpha    :")
    for alpha in available_alphas:
        v = best["test_f1_per_alpha"].get(alpha)
        v_str = f"{v:.4f}" if v is not None else "N/A"
        lines.append(f"    a{alpha}  : {v_str}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cross-dB AWGN robustness results"
    )
    parser.add_argument(
        "--model_type", type=str, default="clean",
        help="Model type prefix: 'clean' or 'mixed' (default: clean)"
    )
    args = parser.parse_args()
    run(args.model_type)


if __name__ == "__main__":
    main()
