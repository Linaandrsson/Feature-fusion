"""
print_best_per_alpha_tables.py

For each k, prints a table showing the best sensor combination per AWGN alpha level.

Edit CONFIG below and run with the VS Code run button.
"""

import json
import sys
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_TYPE = "clean"   # "clean" or "mixed"
SAVE_TXT   = True      # True = also write to a .txt file in this folder
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parents[1]


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


def alpha_to_float(alpha: str) -> float:
    return int(alpha) / 100.0


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run():
    alphas = discover_alpha_variants(MODEL_TYPE)
    if not alphas:
        print(f"[ERROR] No folders found for model_type='{MODEL_TYPE}'")
        sys.exit(1)

    # Load all data: {alpha: [records]}
    alpha_records: dict[str, list[dict]] = {}
    available_alphas = []
    for alpha in alphas:
        p = averaged_jsonl_path(MODEL_TYPE, alpha)
        if p.exists():
            alpha_records[alpha] = load_jsonl(p)
            available_alphas.append(alpha)

    if not available_alphas:
        print("[ERROR] No averaged JSONL files found. Run aggregate_multirun.py first.")
        sys.exit(1)

    k_values = sorted({r["num_sensors"] for records in alpha_records.values() for r in records})

    output_lines = []

    def out(line=""):
        print(line)
        output_lines.append(line)

    out(f"Best sensor combination per AWGN alpha — model_type='{MODEL_TYPE}'")
    out(f"Alpha levels: {', '.join('a'+a for a in available_alphas)}")
    out()

    for k in k_values:
        # For each alpha: find the record with highest test_f1_macro for this k
        best_per_alpha: list[tuple[str, dict]] = []  # (alpha, best_record)
        for alpha in available_alphas:
            k_recs = [r for r in alpha_records[alpha] if r.get("num_sensors") == k]
            if not k_recs:
                continue
            best = max(k_recs, key=lambda r: r.get("test_f1_macro") or 0)
            best_per_alpha.append((alpha, best))

        if not best_per_alpha:
            continue

        # Column widths
        sensor_strs = [", ".join(b["sensors"]) for _, b in best_per_alpha]
        w_alpha   = max(len("Alpha"), max(len(f"a{a}") for a, _ in best_per_alpha))
        w_sensors = max(len("Best sensor combination"), max(len(s) for s in sensor_strs))
        w_f1      = 8
        total_w   = w_alpha + 2 + w_sensors + 2 + w_f1
        sep  = "=" * total_w
        dash = "-" * total_w

        out(sep)
        out(f"k = {k}  —  Best sensor combination per AWGN alpha")
        out(sep)
        out(f"{'Alpha':<{w_alpha}}  {'Best sensor combination':<{w_sensors}}  {'F1 Macro':>{w_f1}}")
        out(dash)

        for alpha, best in best_per_alpha:
            sensors_str = ", ".join(best["sensors"])
            f1          = best["test_f1_macro"]
            out(f"{'a'+alpha:<{w_alpha}}  {sensors_str:<{w_sensors}}  {f1:>{w_f1}.4f}")

        out()

    if SAVE_TXT:
        out_path = Path(__file__).parent / f"best_per_alpha_tables_{MODEL_TYPE}.txt"
        out_path.write_text("\n".join(output_lines))
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    run()
