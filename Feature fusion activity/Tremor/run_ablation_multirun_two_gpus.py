"""
run_ablation_multirun.py

Runs the full ablation study (all 255 sensor combinations) N times with
different random seeds, using run_ablation_parallel.py for each run.

Each run launches k=1..8 in parallel across 2 GPUs (~10 min per run).
Runs are sequential — one full parallel run finishes before the next starts.

Results accumulate in the same tagged subdirectories:
  tremor_logs/ablation_json_files_{tag}/   ← one .jsonl per k per run
  tremor_logs/ablation_reports_{tag}/      ← .txt + .csv per run

After N runs, analyze_multirun.py aggregates mean ± std per sensor combo.

Usage:
  python run_ablation_multirun.py                           # 10 seeds, GPU 1
  python run_ablation_multirun.py --n_runs 5
  python run_ablation_multirun.py --seeds 42,43,44          # explicit seeds
  python run_ablation_multirun.py --gpus 0,1 --n_runs 10
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

SCRIPT = Path(__file__).parent / "run_ablation_parallel.py"
PYTHON = Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed ablation launcher")
    parser.add_argument(
        "--n_runs", type=int, default=10,
        help="Number of runs with different seeds (default: 10)"
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Explicit comma-separated seeds (overrides --n_runs). E.g. '42,43,44'"
    )
    parser.add_argument(
        "--start_seed", type=int, default=42,
        help="First seed when using --n_runs (seeds = start_seed .. start_seed+n_runs-1, default: 42)"
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1",
        help="Comma-separated GPU IDs passed to run_ablation_parallel.py (default: '0,1')"
    )
    parser.add_argument(
        "--k_values", type=str, default="1,2,3,4,5,6,7,8",
        help="k values to run per seed (default: all 8)"
    )
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.n_runs))

    total_runs = len(seeds)
    start_wall = time.time()
    timestamp = datetime.now().strftime("%m%d_%H%M")

    print("=" * 65)
    print(f"Tremor Ablation Multi-Run Launcher  [{timestamp}]")
    print("=" * 65)
    print(f"  Runs     : {total_runs}")
    print(f"  Seeds    : {seeds}")
    print(f"  GPUs     : {args.gpus}")
    print(f"  k values : {args.k_values}")
    print(f"  Script   : {SCRIPT}")
    print("=" * 65)
    print()

    failed_seeds = []

    for run_idx, seed in enumerate(seeds, 1):
        run_start = time.time()
        run_ts = datetime.now().strftime("%m%d_%H%M")
        print(f"{'─' * 65}")
        print(f"  Run {run_idx}/{total_runs}  seed={seed}  [{run_ts}]")
        print(f"{'─' * 65}")

        cmd = [
            str(PYTHON), str(SCRIPT),
            "--gpus", args.gpus,
            "--k_values", args.k_values,
            "--seed", str(seed),
        ]

        result = subprocess.run(cmd)
        elapsed = time.time() - run_start

        if result.returncode != 0:
            print(f"\n  ✗ Run {run_idx} (seed={seed}) FAILED (rc={result.returncode})")
            failed_seeds.append(seed)
        else:
            print(f"\n  ✓ Run {run_idx} (seed={seed}) done  ({elapsed/60:.1f} min)")

        wall_so_far = time.time() - start_wall
        remaining = total_runs - run_idx
        if run_idx > 0 and remaining > 0:
            avg_per_run = wall_so_far / run_idx
            eta_min = avg_per_run * remaining / 60
            print(f"  ETA: ~{eta_min:.0f} min remaining")
        print()

    total_elapsed = time.time() - start_wall
    print("=" * 65)
    print(f"All {total_runs} runs finished.  Total wall time: {total_elapsed/60:.1f} min")
    if failed_seeds:
        print(f"FAILED seeds: {failed_seeds}")
        sys.exit(1)
    else:
        print(f"Results in: {SCRIPT.parent}/tremor_logs/")
        print("Run analyze_multirun.py to aggregate mean ± std per combo.")


if __name__ == "__main__":
    main()
