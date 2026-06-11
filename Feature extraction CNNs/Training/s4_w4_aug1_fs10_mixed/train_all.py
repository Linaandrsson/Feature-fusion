"""
train_all.py — Runs all CNN feature extractor training scripts sequentially.

Usage:
    python train_all.py [--gpu 0]

Features:
  - Suppresses all matplotlib GUI popups (MPLBACKEND=Agg)
  - Streams live output to terminal AND a timestamped log file
  - Reports per-script timing and a final summary
  - Exits with non-zero code if any script fails
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = Path(__file__).parents[4] / ".venv" / "bin" / "python"  # /home/linacr/projects/Code/.venv/bin/python

SCRIPTS = [
    "Acc_ankle_CNN.py",
    "Acc_arm_CNN.py",
    "Acc_chest_CNN.py",
    "ECG_chest.py",
    "Gyro_ankle_CNN.py",
    "Gyro_arm_CNN.py",
    "Mag_ankle_CNN.py",
    "Mag_arm_CNN.py",
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train all CNN feature extractors.")
parser.add_argument("--gpu", type=str, default="0",
                    help="CUDA_VISIBLE_DEVICES value (default: 0)")
parser.add_argument("--log-dir", type=str, default=str(SCRIPT_DIR / "train_logs"),
                    help="Directory for per-script log files")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
log_dir = Path(args.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_log = log_dir / f"train_all_{run_timestamp}.log"

# Resolve python executable: use current interpreter if venv not found
python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
print(f"Python: {python_exe}")
print(f"GPU:    CUDA_VISIBLE_DEVICES={args.gpu}")
print(f"Logs:   {log_dir}")
print(f"Run ID: {run_timestamp}")
print("=" * 60)

# Always delete existing splits so the first script regenerates them
# based on current dataset size. Subject-based splitting is handled
# inside each CNN script — this just ensures stale indices never cause
# IndexError when the dataset changes.
splits_dir = SCRIPT_DIR / "splits"
if splits_dir.exists():
    import shutil
    shutil.rmtree(splits_dir)
    print(f"Deleted stale splits: {splits_dir}")

# Environment: inherit current env, override display/GPU settings
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = args.gpu
env["MPLBACKEND"] = "Agg"          # Suppress all matplotlib GUI popups
env["MPLCONFIGDIR"] = str(log_dir)  # Avoid matplotlib home-dir warnings

# ---------------------------------------------------------------------------
# Run scripts
# ---------------------------------------------------------------------------
results = []

for script_name in SCRIPTS:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"\n[SKIP] {script_name} — file not found")
        results.append((script_name, "SKIP", 0.0))
        continue

    script_log = log_dir / f"{script_path.stem}_{run_timestamp}.log"
    header = f"\n{'='*60}\n[{datetime.now().strftime('%H:%M:%S')}] START: {script_name}\n{'='*60}"
    print(header)

    t_start = time.time()
    status = "OK"

    try:
        with open(script_log, "w") as log_fh:
            # Write header to log file
            log_fh.write(header + "\n")
            log_fh.flush()

            proc = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge stderr into stdout
                cwd=str(SCRIPT_DIR),
                env=env,
                bufsize=1,                  # line-buffered
                text=True,
            )

            # Stream output live to terminal AND log file
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_fh.write(line)
                log_fh.flush()

            proc.wait()
            elapsed = time.time() - t_start

            if proc.returncode != 0:
                status = f"FAIL (exit {proc.returncode})"
            else:
                status = "OK"

    except Exception as exc:
        elapsed = time.time() - t_start
        status = f"ERROR ({exc})"
        print(f"\n[ERROR] {exc}")

    summary_line = f"[{status:^20s}]  {script_name:<25s}  {elapsed:6.1f}s   log: {script_log.name}"
    print(f"\n{summary_line}")
    results.append((script_name, status, elapsed))

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
total_time = sum(r[2] for r in results)
summary_lines = [
    "",
    "=" * 60,
    f"TRAINING SUMMARY — {run_timestamp}",
    "=" * 60,
]
for name, status, elapsed in results:
    summary_lines.append(f"  {status:^20s}  {name:<25s}  {elapsed:6.1f}s")
summary_lines += [
    "-" * 60,
    f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)",
    "=" * 60,
]

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(summary_log, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {summary_log}")

# Exit with error if any script failed
failed = [r for r in results if r[1] not in ("OK", "SKIP")]
sys.exit(1 if failed else 0)
