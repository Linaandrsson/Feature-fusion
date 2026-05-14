"""
aggregate_awgn_robustness.py

Cross-dB AWGN robustness report.

For a given model type (e.g. "clean" or "mixed"), loads the averaged
per-combo JSONL files from all AWGN dB-level directories, then for each
sensor combination + k: averages test_f1_macro across all dB levels.
Produces per-k TXT reports ranked by mean cross-dB F1.

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

# ── Known AWGN dB levels (edit if you add more) ──────────────────────────────
AWGN_DB_LEVELS = [0, 3, 6, 10, 14, 15, 20]

NUMERIC_FIELDS = [
    "val_accuracy", "val_f1_macro", "val_loss",
    "test_loss", "test_accuracy", "test_f1_macro",
]

META_FIELDS = ["sensors", "ablated_sensors", "num_sensors", "fusion_mode"]


def tag_for(model_type: str, db: int) -> str:
    return f"{model_type}_on_100awgn{db}db"


def averaged_jsonl_path(model_type: str, db: int) -> Path:
    tag = tag_for(model_type, db)
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
    print(f"Cross-dB AWGN robustness aggregation — model_type='{model_type}'")
    print(f"{'='*70}")

    # ── Find available dB levels ──────────────────────────────────────────
    available_dbs = []
    for db in AWGN_DB_LEVELS:
        p = averaged_jsonl_path(model_type, db)
        if p.exists():
            available_dbs.append(db)
            print(f"  [✓] {db:>3} dB  →  {p.name}")
        else:
            print(f"  [✗] {db:>3} dB  →  NOT FOUND: {p}")

    if not available_dbs:
        print("[ERROR] No AWGN data found. Run aggregate_multirun.py first.")
        sys.exit(1)

    print(f"\n  Using {len(available_dbs)} dB levels: {available_dbs}")

    # ── Load all records ─────────────────────────────────────────────────
    # Structure: {combo_key: {db: record}}
    combo_db_records: dict[tuple, dict[int, dict]] = defaultdict(dict)

    for db in available_dbs:
        records = load_jsonl(averaged_jsonl_path(model_type, db))
        for r in records:
            combo_db_records[combo_key(r)][db] = r

    print(f"  Total unique sensor combinations: {len(combo_db_records)}")

    # ── Aggregate across dB levels ────────────────────────────────────────
    aggregated = []
    for key, db_map in combo_db_records.items():
        dbs_present = sorted(db_map.keys())
        first = db_map[dbs_present[0]]

        agg = {f: first[f] for f in META_FIELDS}

        for field in NUMERIC_FIELDS:
            vals_per_db = []
            for db in dbs_present:
                v = db_map[db].get(field)
                if v is not None:
                    vals_per_db.append(v)
            if vals_per_db:
                agg[field] = float(np.mean(vals_per_db))
                agg[f"{field}_std_across_db"] = float(np.std(vals_per_db))
            else:
                agg[field] = None
                agg[f"{field}_std_across_db"] = None

        agg["db_levels_used"] = dbs_present
        agg["num_db_levels"] = len(dbs_present)

        # Per-dB breakdown for test_f1_macro
        agg["test_f1_per_db"] = {
            str(db): db_map[db].get("test_f1_macro") for db in dbs_present
        }
        aggregated.append(agg)

    # ── Output dirs ───────────────────────────────────────────────────────
    out_tag      = f"{model_type}_awgn_robustness"
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
                              available_dbs=available_dbs)
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
                    available_dbs: list[int]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    db_str = ", ".join(f"{d}dB" for d in available_dbs)

    lines = []
    lines.append("=" * 80)
    lines.append("AWGN ROBUSTNESS REPORT  [CROSS-dB AVERAGE]")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated  : {now}")
    lines.append(f"Model type : {model_type}")
    lines.append(f"K          : {k}")
    lines.append(f"dB levels  : {db_str}  ({len(available_dbs)} levels)")
    lines.append(f"Metric     : mean test F1 macro across all dB levels")
    lines.append(f"Total combinations: {len(combos)}")
    lines.append("")

    # Column widths
    sensor_strs  = [", ".join(c["sensors"])        for c in combos]
    ablated_strs = [", ".join(c["ablated_sensors"]) for c in combos]
    w_s = max(len("Sensors"), max(len(s) for s in sensor_strs))
    w_a = max(len("Ablated"), max(len(s) for s in ablated_strs))

    # Per-dB column headers
    db_col_w = 7
    db_header = "  ".join(f"{d:>{db_col_w}}dB" for d in available_dbs)
    total_w = 6 + 1 + w_s + 2 + w_a + 2 + 11 + 2 + len(db_header)
    sep  = "=" * total_w
    dash = "-" * total_w

    hdr = (
        f"{'Rank':<6} {'Sensors':<{w_s}}  {'Ablated':<{w_a}}  "
        f"{'Mean F1':>9} {'±':>1}  {db_header}"
    )

    lines.append(sep)
    lines.append("RANKED BY MEAN TEST F1 ACROSS ALL AWGN dB LEVELS")
    lines.append(sep)
    lines.append(hdr)
    lines.append(dash)

    for rank, c, s_str, a_str in zip(range(1, len(combos)+1), combos, sensor_strs, ablated_strs):
        mean_f1 = c["test_f1_macro"]
        std_f1  = c["test_f1_macro_std_across_db"]
        per_db  = c["test_f1_per_db"]

        db_vals = "  ".join(
            f"{per_db.get(str(db)):>{db_col_w+2}.4f}" if per_db.get(str(db)) is not None
            else f"{'N/A':>{db_col_w+2}}"
            for db in available_dbs
        )

        lines.append(
            f"{rank:<6} {s_str:<{w_s}}  {a_str:<{w_a}}  "
            f"{mean_f1:>9.4f} {std_f1:>4.4f}  {db_vals}"
        )

    # Best combo summary
    best = combos[0]
    lines.append("")
    lines.append("=" * 80)
    lines.append("BEST COMBINATION (by mean cross-dB test F1)")
    lines.append("=" * 80)
    lines.append(f"  Sensors    : {', '.join(best['sensors'])}")
    lines.append(f"  Ablated    : {', '.join(best['ablated_sensors'])}")
    lines.append(f"  Mean F1    : {best['test_f1_macro']:.4f} ± {best['test_f1_macro_std_across_db']:.4f}")
    lines.append(f"  Per dB     :")
    for db in available_dbs:
        v = best["test_f1_per_db"].get(str(db))
        v_str = f"{v:.4f}" if v is not None else "N/A"
        lines.append(f"    {db:>3} dB  : {v_str}")
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
