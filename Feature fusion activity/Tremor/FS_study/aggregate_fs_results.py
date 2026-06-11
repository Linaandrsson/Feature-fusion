"""
aggregate_fs_results.py

Aggregates FS combination study results across multiple seeds.
Groups by fs_combo_hash, computes mean ± std for test_f1, val_f1, test_accuracy.
Outputs a ranked summary TXT report.

Usage:
  python aggregate_fs_results.py --jsonl tremor_logs_gpu1/tremor_fusion_fs_combo_study_clean_tremor.jsonl
  python aggregate_fs_results.py --jsonl tremor_logs_gpu0/tremor_fusion_fs_combo_study_clean_clean.jsonl
  python aggregate_fs_results.py --jsonl tremor_logs_tremor_tremor/tremor_fusion_fs_combo_study_tremor_tremor.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics


def load_jsonl(path: Path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def aggregate(entries):
    groups = defaultdict(list)
    for e in entries:
        groups[e["fs_combo_hash"]].append(e)

    rows = []
    for combo_hash, runs in groups.items():
        test_f1s = [r["test_f1"] for r in runs]
        val_f1s  = [r["val_f1"]  for r in runs]
        test_accs = [r["test_accuracy"] for r in runs]
        seeds_used = sorted({r.get("seed", r.get("seed", "?")) for r in runs})

        # fs_map is the same across seeds — take from first entry
        fs_map = runs[0]["fs_map"]
        fs_total = sum(fs_map.values())

        rows.append({
            "fs_combo_hash": combo_hash,
            "fs_map": fs_map,
            "fs_total": fs_total,
            "n_seeds": len(runs),
            "seeds": seeds_used,
            "mean_test_f1":  statistics.mean(test_f1s),
            "std_test_f1":   statistics.stdev(test_f1s) if len(test_f1s) > 1 else 0.0,
            "min_test_f1":   min(test_f1s),
            "max_test_f1":   max(test_f1s),
            "mean_val_f1":   statistics.mean(val_f1s),
            "std_val_f1":    statistics.stdev(val_f1s) if len(val_f1s) > 1 else 0.0,
            "mean_test_acc": statistics.mean(test_accs),
            "std_test_acc":  statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0,
        })

    rows.sort(key=lambda r: r["mean_test_f1"], reverse=True)
    return rows


def write_report(rows, out_path: Path, jsonl_path: Path):
    seeds_all = sorted({s for r in rows for s in r["seeds"]})
    n_combos = len(rows)

    fs_col = max((len(str(r["fs_map"])) for r in rows), default=50) + 2
    fs_col = max(fs_col, 50)

    hdr = (f"{'Rank':<6} {'FS map':<{fs_col}} {'Hash':<10} "
           f"{'Mean F1':>9} {'±Std':>7} {'Min':>7} {'Max':>7} "
           f"{'MeanAcc':>9} {'Seeds':>6}")
    sep = "-" * len(hdr)

    with open(out_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FS COMBINATION STUDY — AGGREGATED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source:       {jsonl_path}\n")
        f.write(f"Combinations: {n_combos}\n")
        f.write(f"Seeds:        {seeds_all}  (n={len(seeds_all)})\n\n")
        f.write("=" * len(hdr) + "\n")
        f.write("RANKED BY MEAN TEST F1 (MACRO)\n")
        f.write("=" * len(hdr) + "\n")
        f.write(hdr + "\n")
        f.write(sep + "\n")
        for rank, r in enumerate(rows, 1):
            f.write(
                f"{rank:<6} {str(r['fs_map']):<{fs_col}} {r['fs_combo_hash']:<10} "
                f"{r['mean_test_f1']*100:>8.2f}% "
                f"{r['std_test_f1']*100:>6.2f}% "
                f"{r['min_test_f1']*100:>6.2f}% "
                f"{r['max_test_f1']*100:>6.2f}% "
                f"{r['mean_test_acc']*100:>8.2f}% "
                f"{r['n_seeds']:>5}\n"
            )
        f.write("\n")

        best = rows[0]
        worst = rows[-1]
        f.write("=" * 80 + "\n")
        f.write("TOP 10\n")
        f.write("=" * 80 + "\n")
        for rank, r in enumerate(rows[:10], 1):
            f.write(f"  {rank:>2}. {r['fs_map']}  "
                    f"Mean F1={r['mean_test_f1']*100:.2f}% ±{r['std_test_f1']*100:.2f}%  "
                    f"[{r['min_test_f1']*100:.2f}%–{r['max_test_f1']*100:.2f}%]\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("BOTTOM 5\n")
        f.write("=" * 80 + "\n")
        for rank, r in enumerate(rows[-5:], n_combos - 4):
            f.write(f"  {rank:>3}. {r['fs_map']}  "
                    f"Mean F1={r['mean_test_f1']*100:.2f}% ±{r['std_test_f1']*100:.2f}%\n")

    print(f"Rapport lagret: {out_path}")
    print(f"\nTop 5:")
    for rank, r in enumerate(rows[:5], 1):
        print(f"  {rank}. {r['fs_map']}  "
              f"Mean F1={r['mean_test_f1']*100:.2f}% ±{r['std_test_f1']*100:.2f}%  "
              f"[{r['min_test_f1']*100:.2f}%–{r['max_test_f1']*100:.2f}%]")
    print(f"\n  ({n_combos} kombinasjoner, {len(seeds_all)} seeds)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate FS study results across seeds")
    parser.add_argument("--jsonl", required=True, help="Path to .jsonl log file")
    parser.add_argument("--out", default=None, help="Output TXT path (default: same dir as jsonl)")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Feil: finner ikke {jsonl_path}")
        return

    if args.out:
        out_path = Path(args.out)
    else:
        stem = jsonl_path.stem.replace("tremor_fusion_fs_combo_study_", "")
        out_path = jsonl_path.parent / f"fs_aggregated_{stem}.txt"

    entries = load_jsonl(jsonl_path)
    print(f"Leste {len(entries)} entries fra {jsonl_path.name}")

    rows = aggregate(entries)
    print(f"Aggregerte {len(rows)} unike kombinasjoner")

    write_report(rows, out_path, jsonl_path)


if __name__ == "__main__":
    main()
