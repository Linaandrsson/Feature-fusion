"""
plot_avg_confusion_matrix.py

Loads all per-seed JSONL records for a given tag + k, finds the best
sensor combination (highest mean test F1 across seeds), then produces
two confusion matrix plots side by side:

  Left  — averaged raw counts across seeds (mean number of predictions)
  Right — row-normalised (fraction correctly classified per class = recall)

Usage:
    python plot_avg_confusion_matrix.py
    python plot_avg_confusion_matrix.py --tag clean_train_awgn_a000 --k 4
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--tag", type=str, default="clean_train_awgn_a000")
_p.add_argument("--k",   type=int, default=4)
ARGS = _p.parse_args()

TAG      = ARGS.tag
K_TARGET = ARGS.k
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR   = Path(__file__).parents[1]
JSON_DIR   = LOGS_DIR / f"ablation_json_files_{TAG}"
OUT_DIR    = Path(__file__).parent

# Activity labels (1-indexed display)
N_CLASSES  = 12
CLASS_LABELS = [str(i) for i in range(1, N_CLASSES + 1)]


def load_k_records(json_dir: Path, k: int) -> list[dict]:
    records = []
    for f in sorted(json_dir.glob(f"tremor_ablation_k{k}_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def best_combo(records: list[dict]) -> tuple[tuple, list[dict]]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[tuple(sorted(r["sensors"]))].append(r)
    best_key = max(
        groups,
        key=lambda k: sum(r["test_f1_macro"] for r in groups[k]) / len(groups[k]),
    )
    return best_key, groups[best_key]


def build_cm(y_true: list, y_pred: list, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_cm(ax, cm: np.ndarray, title: str, fmt: str, cmap: str, vmax=None):
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=vmax if vmax else cm.max())
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, fontsize=8)
    ax.set_yticklabels(CLASS_LABELS, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm[i, j]
            text = format(val, fmt)
            ax.text(j, i, text,
                    ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def run():
    if not JSON_DIR.exists():
        print(f"[ERROR] Directory not found: {JSON_DIR}")
        return

    records = load_k_records(JSON_DIR, K_TARGET)
    if not records:
        print(f"[ERROR] No k={K_TARGET} records found in {JSON_DIR.name}")
        return

    sensors_key, seed_records = best_combo(records)
    mean_f1 = sum(r["test_f1_macro"] for r in seed_records) / len(seed_records)
    n_seeds  = len(seed_records)
    sensors_str = ", ".join(sensors_key)

    print(f"Tag     : {TAG}")
    print(f"k       : {K_TARGET}")
    print(f"Best    : {sensors_str}")
    print(f"Mean F1 : {mean_f1:.4f}  ({n_seeds} seeds)")

    # ── Build averaged count CM ───────────────────────────────────────────
    cms = np.stack([
        build_cm(r["y_true"], r["y_pred"], N_CLASSES)
        for r in seed_records
    ])
    cm_avg   = cms.mean(axis=0)          # mean counts
    cm_norm  = cm_avg / cm_avg.sum(axis=1, keepdims=True)  # row-normalised

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_cm(axes[0], cm_avg,
            title=f"Averaged counts  (mean over {n_seeds} seeds)",
            fmt=".1f", cmap="Blues")

    plot_cm(axes[1], cm_norm,
            title=f"Row-normalised  (recall per class)",
            fmt=".2f", cmap="Blues", vmax=1.0)

    fig.suptitle(
        f"Confusion matrix — {TAG}  k={K_TARGET}\n"
        f"Best combo: {sensors_str}  |  Mean F1={mean_f1:.4f}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    slug = "_".join(s.lower() for s in sensors_key)
    out_path = OUT_DIR / f"confusion_matrix_avg_{TAG}_k{K_TARGET}_{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    run()
