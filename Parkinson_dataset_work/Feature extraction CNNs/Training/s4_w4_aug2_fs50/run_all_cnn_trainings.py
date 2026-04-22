#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

TRAINING_SCRIPTS = [
    "Mag_UR_CNN.py",
    "Mag_UL_CNN.py",
    "Mag_LR_CNN.py",
    "Mag_LL_CNN.py",
    "Mag_head_CNN.py",
    "Gyro_UR_CNN.py",
    "Gyro_UL_CNN.py",
    "Gyro_LR_CNN.py",
    "Gyro_LL_CNN.py",
    "Gyro_head_CNN.py",
    "Acc_UR_CNN.py",
    "Acc_UL_CNN.py",
    "Acc_LR_CNN.py",
    "Acc_LL_CNN.py",
    "Acc_head_CNN.py",
]


def run_one(
    script_path: Path,
    log_dir: Path,
    python_exec: str,
    dry_run: bool,
    live: bool,
) -> tuple[int, float, Path]:
    start = time.time()
    log_path = log_dir / f"{script_path.stem}.log"

    if dry_run:
        print(f"[DRY-RUN] {python_exec} {script_path}")
        return 0, 0.0, log_path

    env = os.environ.copy()
    # Prevent GUI popups/blocking from matplotlib windows.
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    if live:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [python_exec, str(script_path)],
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
            proc.wait()
    else:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                [python_exec, str(script_path)],
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

    elapsed = time.time() - start
    return proc.returncode, elapsed, log_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all sensor CNN training scripts sequentially.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one script fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Stream each script output live in terminal (still saves logs).",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live output (log files only).",
    )
    args = parser.parse_args()

    log_dir = SCRIPT_DIR / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    live_mode = True
    if args.no_live:
        live_mode = False
    elif args.live:
        live_mode = True

    print("=" * 72)
    print("Batch CNN training runner")
    print(f"Folder: {SCRIPT_DIR}")
    print(f"Python: {args.python}")
    print(f"Logs:   {log_dir}")
    print(f"Live:   {'ON' if live_mode else 'OFF'}")
    print("=" * 72)

    missing = [name for name in TRAINING_SCRIPTS if not (SCRIPT_DIR / name).exists()]
    if missing:
        print("ERROR: Missing scripts:")
        for m in missing:
            print(f"  - {m}")
        return 2

    results: list[tuple[str, int, float, Path]] = []
    for idx, name in enumerate(TRAINING_SCRIPTS, start=1):
        script_path = SCRIPT_DIR / name
        print(f"\n[{idx:02d}/{len(TRAINING_SCRIPTS)}] Running {name} ...")
        code, elapsed, log_path = run_one(
            script_path,
            log_dir,
            args.python,
            args.dry_run,
            live_mode,
        )
        status = "OK" if code == 0 else f"FAIL ({code})"
        print(f"  -> {status} in {elapsed:.1f}s")
        if not args.dry_run:
            print(f"  -> log: {log_path}")
        results.append((name, code, elapsed, log_path))

        if code != 0 and args.stop_on_error:
            print("Stopping due to --stop-on-error.")
            break

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    failed = 0
    for name, code, elapsed, log_path in results:
        mark = "OK" if code == 0 else "FAIL"
        print(f"{mark:4}  {name:16}  {elapsed:8.1f}s  {log_path.name}")
        if code != 0:
            failed += 1

    if failed:
        print(f"\nCompleted with failures: {failed}/{len(results)}")
        return 1

    print(f"\nAll done: {len(results)} scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
