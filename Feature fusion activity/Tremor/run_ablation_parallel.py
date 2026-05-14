"""
run_ablation_parallel.py

Launches all 8 ablation_k sizes (k=1..8) as parallel subprocesses,
distributing across 2 GPUs (round-robin).

GPU assignment (round-robin):
  GPU 0: k=1, k=3, k=5, k=7   →  8+56+56+8  = 128 combinations
  GPU 1: k=2, k=4, k=6, k=8   →  28+70+28+1 = 127 combinations

Each process writes its output to:
  tremor_logs/parallel_k{k}.log

Usage:
  python run_ablation_parallel.py [--gpus 0,1]
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

SCRIPT = Path(__file__).parent / "fusion_concat_ablation_study_tremor.py"
PYTHON = Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description="Run all ablation_k sizes in parallel across GPUs")
    parser.add_argument(
        "--gpus", type=str, default="0,1",
        help="Comma-separated GPU IDs to use (default: '0,1')"
    )
    parser.add_argument(
        "--k_values", type=str, default="1,2,3,4,5,6,7,8",
        help="Comma-separated k values to run (default: '1,2,3,4,5,6,7,8')"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed passed to the ablation script (overrides its default)"
    )
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",")]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    seed = args.seed

    start_time = time.time()
    timestamp = datetime.now().strftime("%m%d_%H%M")

    print("=" * 60)
    print(f"Tremor Ablation Parallel Launcher  [{timestamp}]")
    print("=" * 60)
    print(f"  k values : {k_values}")
    print(f"  GPUs     : {gpu_ids}")
    seed_str = str(seed) if seed is not None else "(default)"
    print(f"  Seed     : {seed_str}")
    print(f"  Script   : {SCRIPT}")
    print("=" * 60)

    # Combinatorics summary
    from math import comb
    n_sensors = 8
    for k in k_values:
        gpu = gpu_ids[(k - 1) % len(gpu_ids)]
        n_combos = comb(n_sensors, k)
        print(f"  k={k:2d}  →  GPU {gpu}  ({n_combos:3d} combinations)")
    print()

    # Log dir for stderr output
    log_dir = SCRIPT.parent / "tremor_logs" / "process_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{seed}" if seed is not None else ""

    # Spawn all subprocesses
    procs = []
    stderr_files = []
    for k in k_values:
        gpu_id = gpu_ids[(k - 1) % len(gpu_ids)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        cmd = [str(PYTHON), str(SCRIPT), "--ablation_k", str(k)]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        log_file = log_dir / f"k{k}{seed_tag}_{timestamp}.log"
        fh = open(log_file, "w")
        stderr_files.append((k, log_file, fh))
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=fh)
        procs.append((k, gpu_id, p))
        print(f"  [k={k}] PID {p.pid:6d}  GPU {gpu_id}  log→ {log_file.name}")

    print(f"\nAll {len(procs)} processes launched. Waiting for completion...\n")

    # Wait and collect results
    failed = []
    for (k, gpu_id, p), (_, log_file, fh) in zip(procs, stderr_files):
        p.wait()
        fh.close()
        elapsed = time.time() - start_time
        if p.returncode == 0:
            status = "✓"
        else:
            # Print last few lines of log to help diagnose
            try:
                lines = log_file.read_text().strip().splitlines()
                tail = "\n".join(lines[-5:]) if lines else "(empty log)"
            except Exception:
                tail = "(could not read log)"
            status = f"✗ FAILED (rc={p.returncode})  →  {log_file.name}"
            print(f"  [k={k}] GPU {gpu_id}: {status}  ({elapsed/60:.1f} min)")
            print(f"    Last lines:\n" + "\n".join(f"      {l}" for l in (lines[-5:] if lines else [])))
            failed.append(k)
            continue
        print(f"  [k={k}] GPU {gpu_id}: {status}  ({elapsed/60:.1f} min)")

    total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Total wall time: {total/60:.1f} minutes")
    if failed:
        print(f"FAILED k values: {failed}")
        print("Check the corresponding log files for details.")
        sys.exit(1)
    else:
        print("All k values completed successfully!")
        print(f"Reports + CSVs saved in: {SCRIPT.parent}/tremor_logs/")


if __name__ == "__main__":
    main()
