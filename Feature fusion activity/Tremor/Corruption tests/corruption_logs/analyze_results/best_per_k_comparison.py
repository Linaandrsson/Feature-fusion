"""
best_per_k_comparison.py
------------------------
Reads the cross-alpha robustness JSONL files for clean and mixed model types,
finds the best sensor combination per k (by mean test F1 macro across all AWGN
alpha levels), and prints a side-by-side comparison table.

Usage:
    python best_per_k_comparison.py
"""

import json
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # corruption_logs/

JSONL_PATHS = {
    "clean": BASE_DIR / "ablation_json_files_clean_train_awgn_robustness"
                      / "robustness_clean_train_awgn_robustness_all_k.jsonl",
    "mixed": BASE_DIR / "ablation_json_files_mixed_train_awgn_robustness"
                      / "robustness_mixed_train_awgn_robustness_all_k.jsonl",
}

K_VALUES = list(range(1, 9))
# ─────────────────────────────────────────────────────────────────────────────


def load_best_per_k(jsonl_path: Path) -> dict:
    """Return {k: best_record} sorted by test_f1_macro (mean across alphas)."""
    best = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = rec["num_sensors"]
            f1 = rec["test_f1_macro"]
            if k not in best or f1 > best[k]["test_f1_macro"]:
                best[k] = rec
    return best


def fmt_sensors(sensors: list) -> str:
    return ", ".join(sensors)


def fmt_per_alpha(rec: dict) -> str:
    per_alpha = rec.get("test_f1_per_alpha", {})
    parts = [f"α={k}: {v:.3f}" for k, v in sorted(per_alpha.items())]
    return "  ".join(parts)


def print_table(clean: dict, mixed: dict) -> None:
    SENSOR_W = 52
    F1_W     = 8
    ACC_W    = 8

    # ── header ──────────────────────────────────────────────────────────────
    sep = "─"
    total_w = 2 + 1 + (SENSOR_W + F1_W + ACC_W + 6) * 2 + 3
    print()
    print("=" * total_w)
    print("  BEST SENSOR COMBINATION PER K  —  clean vs. mixed  (mean F1 across all AWGN α)")
    print("=" * total_w)

    col_hdr = f"{'Sensors':<{SENSOR_W}}  {'MeanF1':>{F1_W}}  {'MeanAcc':>{ACC_W}}"
    print(f"  {'k':<2}  │  {'CLEAN':^{SENSOR_W + F1_W + ACC_W + 4}}  │  {'MIXED':^{SENSOR_W + F1_W + ACC_W + 4}}")
    print(f"  {'':2}  │  {col_hdr}  │  {col_hdr}")
    print(f"  {sep*2}  ┼" + f"  {sep*(SENSOR_W + F1_W + ACC_W + 4)}  ┼" +
          f"  {sep*(SENSOR_W + F1_W + ACC_W + 4)}")

    for k in K_VALUES:
        c = clean.get(k)
        m = mixed.get(k)

        c_sensors = fmt_sensors(c["sensors"])            if c else "—"
        c_f1      = f"{c['test_f1_macro']:.4f}"         if c else "—"
        c_acc     = f"{c['test_accuracy']:.4f}"          if c else "—"

        m_sensors = fmt_sensors(m["sensors"])            if m else "—"
        m_f1      = f"{m['test_f1_macro']:.4f}"         if m else "—"
        m_acc     = f"{m['test_accuracy']:.4f}"          if m else "—"

        c_cell = f"{c_sensors:<{SENSOR_W}}  {c_f1:>{F1_W}}  {c_acc:>{ACC_W}}"
        m_cell = f"{m_sensors:<{SENSOR_W}}  {m_f1:>{F1_W}}  {m_acc:>{ACC_W}}"
        print(f"  {k:<2}  │  {c_cell}  │  {m_cell}")

    print("=" * total_w)
    print()

    # ── per-alpha breakdown ──────────────────────────────────────────────────
    print("PER-ALPHA F1 BREAKDOWN (best combo per k)")
    print("─" * total_w)
    for k in K_VALUES:
        c = clean.get(k)
        m = mixed.get(k)
        print(f"\n  k={k}")
        if c:
            print(f"    [clean]  {fmt_sensors(c['sensors'])}")
            print(f"             {fmt_per_alpha(c)}")
        if m:
            print(f"    [mixed]  {fmt_sensors(m['sensors'])}")
            print(f"             {fmt_per_alpha(m)}")
    print()


def write_txt_compact(clean: dict, mixed: dict, out_path: Path) -> None:
    lines = []
    lines.append("BEST SENSOR COMBINATION PER K  —  clean vs. mixed")
    lines.append("=" * 46)
    lines.append(f"{'k':<3}  {'Model':<6}  {'MeanF1':<8}  {'MeanAcc':<8}")
    lines.append("-" * 46)

    for k in K_VALUES:
        for label, rec in [("clean", clean.get(k)), ("mixed", mixed.get(k))]:
            if rec is None:
                continue
            mean_f1  = f"{rec['test_f1_macro']:.4f}"
            mean_acc = f"{rec['test_accuracy']:.4f}"
            lines.append(f"{k:<3}  {label:<6}  {mean_f1:<8}  {mean_acc:<8}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Compact table saved → {out_path}")


def write_txt_table(clean: dict, mixed: dict, out_path: Path) -> None:
    alpha_keys = ["000", "010", "018", "032", "050", "071", "100"]
    alpha_labels = ["α=0.00", "α=0.10", "α=0.18", "α=0.32", "α=0.50", "α=0.71", "α=1.00"]

    lines = []
    lines.append("BEST SENSOR COMBINATION PER K  —  clean vs. mixed")
    lines.append("Mean macro-F1 across all AWGN alpha levels")
    lines.append("=" * 120)
    lines.append(f"{'k':<2}  {'Model':<6}  {'Sensors':<54}  {'Mean F1':<8}  {'Mean Acc':<9}  " +
                 "  ".join(f"{a:<7}" for a in alpha_labels))
    lines.append("-" * 120)

    for k in K_VALUES:
        for label, rec in [("clean", clean.get(k)), ("mixed", mixed.get(k))]:
            if rec is None:
                continue
            sensors  = ", ".join(rec["sensors"])
            mean_f1  = f"{rec['test_f1_macro']:.4f}"
            mean_acc = f"{rec['test_accuracy']:.4f}"
            per_alpha = rec.get("test_f1_per_alpha", {})
            alpha_vals = "  ".join(f"{per_alpha.get(a, 0.0):.4f} " for a in alpha_keys)
            lines.append(f"{k:<2}  {label:<6}  {sensors:<54}  {mean_f1:<8}  {mean_acc:<9}  {alpha_vals}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Text table saved  → {out_path}")


def write_latex_table(clean: dict, mixed: dict, out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best sensor combination per $k$ for clean- and mixed-trained models, "
                 r"ranked by mean macro-F1 across all AWGN corruption levels "
                 r"($\alpha \in \{0.00, 0.10, 0.18, 0.32, 0.50, 0.71, 1.00\}$).}")
    lines.append(r"\label{tab:best_per_k_clean_mixed}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{c l r r l r r}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{\textbf{Clean-trained}}"
                 r" & \multicolumn{3}{c}{\textbf{Mixed-trained}} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"$k$ & Sensors & F1 & Acc & Sensors & F1 & Acc \\")
    lines.append(r"\midrule")

    for k in K_VALUES:
        c = clean.get(k)
        m = mixed.get(k)

        def sensor_str(rec):
            if rec is None:
                return "—", "—", "—"
            s = ", ".join(rec["sensors"]).replace("_", r"\_")
            f1  = f"{rec['test_f1_macro']:.3f}"
            acc = f"{rec['test_accuracy']:.3f}"
            return s, f1, acc

        cs, cf1, cacc = sensor_str(c)
        ms, mf1, macc = sensor_str(m)
        lines.append(rf"{k} & {cs} & {cf1} & {cacc} & {ms} & {mf1} & {macc} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LaTeX table saved → {out_path}")


def main():
    results = {}
    for model_type, path in JSONL_PATHS.items():
        if not path.exists():
            print(f"[ERROR] File not found: {path}")
            return
        results[model_type] = load_best_per_k(path)
        print(f"Loaded {len(results[model_type])} k-values from {model_type} JSONL")

    print_table(results["clean"], results["mixed"])

    latex_out = Path(__file__).parent / "best_per_k_comparison.tex"
    write_latex_table(results["clean"], results["mixed"], latex_out)

    txt_out = Path(__file__).parent / "best_per_k_comparison.txt"
    write_txt_table(results["clean"], results["mixed"], txt_out)

    txt_compact_out = Path(__file__).parent / "best_per_k_compact.txt"
    write_txt_compact(results["clean"], results["mixed"], txt_compact_out)


if __name__ == "__main__":
    main()
