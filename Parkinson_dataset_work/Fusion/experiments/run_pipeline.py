"""
run_pipeline.py

Re-runs both training experiments and both evaluations sequentially:

  Step 1: Train on clean CT data             → models/clean_s4_w4/
  Step 2: Train on mixed (CT + tremor sim)   → models/mixed_s4_w4/
  Step 3: Evaluate clean model on real PD    → models/clean_s4_w4/eval_results/eval_clean_pd_s4/
  Step 4: Evaluate mixed model on real PD    → models/mixed_s4_w4/eval_results/eval_mixed_pd_s4/

Usage:
    python3 run_pipeline.py

The script creates temporary patched copies of run_experiment.py and
evaluate_model.py, runs them, then deletes the temp files.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PYTHON = "/home/linacr/projects/Code/.venv/bin/python"

# Model paths (on this Linux server)
CLEAN_MODEL = (
    SCRIPT_DIR / "models" / "clean_s4_w4"
    / "clean_s4_w4__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"
)
MIXED_MODEL = (
    SCRIPT_DIR / "models" / "mixed_s4_w4"
    / "mixed_s4_w4__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"
)


def run_patched(script_path: Path, patches: dict) -> None:
    """Apply string patches to script_path, write a temp copy, run it, then delete temp."""
    content = script_path.read_text()
    for old, new in patches.items():
        if old not in content:
            raise ValueError(
                f"Patch target not found in {script_path.name}.\n"
                f"Expected to find:\n{old!r}"
            )
        content = content.replace(old, new, 1)

    tmp = script_path.parent / f"_tmp_{script_path.name}"
    tmp.write_text(content)
    try:
        subprocess.run([PYTHON, str(tmp)], check=True, cwd=str(script_path.parent))
    finally:
        tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Train clean model (run_experiment.py is already configured for clean)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("STEP 1/4  Training: clean_s4_w4  (CT clean data only)")
print("═" * 70)

subprocess.run(
    [PYTHON, str(SCRIPT_DIR / "run_experiment.py")],
    check=True,
    cwd=str(SCRIPT_DIR),
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Train mixed model (CT clean + simulated tremor mild+severe)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("STEP 2/4  Training: mixed_s4_w4  (CT + simulated tremor)")
print("═" * 70)

run_patched(
    SCRIPT_DIR / "run_experiment.py",
    patches={
        'EXPERIMENT_TAG: str = "clean_s4_w4"':
            'EXPERIMENT_TAG: str = "mixed_s4_w4"',
        'TREMOR_VARIANTS: List[str] = [\n    "s4_w4_fs50_tremor_clean",\n    # "s4_w4_fs50_tremor_mild_mod",\n    # "s4_w4_fs50_tremor_mod_severe",\n    # "s2_w2_fs50_tremor_mild_mod",\n    # "s2_w2_fs50_tremor_mod_severe",\n]':
            'TREMOR_VARIANTS: List[str] = [\n    "s4_w4_fs50_tremor_clean",\n    "s4_w4_fs50_tremor_mild_mod",\n    "s4_w4_fs50_tremor_mod_severe",\n]',
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Evaluate clean model on real PD data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("STEP 3/4  Evaluating: clean model → real PD data")
print("═" * 70)

run_patched(
    SCRIPT_DIR / "evaluate_model.py",
    patches={
        # Fix macOS path → Linux path (clean model)
        'MODEL_PATH: str = (\n'
        '    # "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion"\n'
        '    # "/experiments/models/mixed/mixed__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"\n'
        '    # "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion"\n'
        '    # "/experiments/models/clean/clean__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"\n'
        '    #"/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/clean_s4_w4/clean_s4_w4__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"\n'
        '    "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/mixed_s4_w4/mixed_s4_w4__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"\n'
        '    #"/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/mixed_v2/mixed_v2__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"\n'
        ')':
            f'MODEL_PATH: str = str(Path("{CLEAN_MODEL}"))',

        # Fix eval tag
        'EVAL_TAG: str = "eval_mixed_pd_s4"':
            'EVAL_TAG: str = "eval_clean_pd_s4"',

        # Fix matplotlib backend (headless server — no display)
        'matplotlib.use("TkAgg")   # interactive popup backend':
            'matplotlib.use("Agg")   # non-interactive backend (headless server)',

        # Disable blocking plt.show()
        '    plt.show()':
            '    # plt.show()  # disabled on headless server',
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Evaluate mixed model on real PD data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("STEP 4/4  Evaluating: mixed model → real PD data")
print("═" * 70)

run_patched(
    SCRIPT_DIR / "evaluate_model.py",
    patches={
        # Fix macOS path → Linux path (mixed model)
        'MODEL_PATH: str = (\n'
        '    # "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion"\n'
        '    # "/experiments/models/mixed/mixed__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"\n'
        '    # "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion"\n'
        '    # "/experiments/models/clean/clean__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"\n'
        '    #"/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/clean_s4_w4/clean_s4_w4__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"\n'
        '    "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/mixed_s4_w4/mixed_s4_w4__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"\n'
        '    #"/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/mixed_v2/mixed_v2__ALL-ALR-AUL-AUR-AHd-GLL-GLR-GUL-GUR-GHd-MLL-MLR-MUL-MUR-MHd_fs50_best.pth"\n'
        ')':
            f'MODEL_PATH: str = str(Path("{MIXED_MODEL}"))',

        # Eval tag is already "eval_mixed_pd_s4" — no change needed, but fix backend
        # Fix matplotlib backend
        'matplotlib.use("TkAgg")   # interactive popup backend':
            'matplotlib.use("Agg")   # non-interactive backend (headless server)',

        # Disable blocking plt.show()
        '    plt.show()':
            '    # plt.show()  # disabled on headless server',
    },
)

print("\n" + "═" * 70)
print("Pipeline complete. Results saved under:")
print(f"  {CLEAN_MODEL.parent / 'eval_results' / 'eval_clean_pd_s4'}")
print(f"  {MIXED_MODEL.parent / 'eval_results' / 'eval_mixed_pd_s4'}")
print("═" * 70)
