"""
evaluate_model.py

Load a saved fusion model and evaluate it on a configurable dataset.
Sensors and sampling frequencies are read automatically from the checkpoint
— no need to specify them manually.

Usage:
  1. Set MODEL_PATH to the .pth file you want to evaluate
  2. Set TREMOR_VARIANTS to the data folders to load (can differ from training)
  3. Set EVAL_TAG to label these results (used in output filenames)
  4. Run the script

Results are saved next to the model file, under:
  <model_dir>/eval_results/<eval_tag>/
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════
# USER CONFIGURATION  ← Edit this section
# ═══════════════════════════════════════════════════════════════

# Path to the saved .pth model checkpoint.
MODEL_PATH: str = (
    # "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion"
    # "/experiments/models/mixed/mixed__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"
    # "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion"
    # "/experiments/models/clean/clean__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"
    "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Fusion/experiments/models/mild_severe/mild_severe__ALL-ALR-MLL-MLR-MHd_fs50_best.pth"
)

# Data folders to evaluate on.
# These can be different from the folders used during training — useful for
# cross-dataset evaluation (e.g. train on clean, evaluate on PD patients).
#
# Examples:
#   Clean only:            ["s2_w2_fs50_tremor_clean"]
#   Simulated tremor mix:  ["s2_w2_fs50_tremor_clean",
#                           "s2_w2_fs50_tremor_mild_mod",
#                           "s2_w2_fs50_tremor_mod_severe"]
#   Real PD patients:      ["s2_w2_fs50_tremor_parkinson"]
TREMOR_VARIANTS: List[str] = [
    "s2_w2_fs50_tremor_parkinson",
]

# Short label for this evaluation run.  Appears in output filenames.
# Examples: "eval_clean", "eval_pd", "eval_mixed"
EVAL_TAG: str = "eval_tremor_pd"

# Per-variant subject filter.
# Keys are folder names from TREMOR_VARIANTS.
# A missing key or None value means ALL subjects for that variant.
# A list value means only those subject IDs are included for that variant.
#
# Examples:
#   No filtering at all:
#       VARIANT_SUBJECTS = {}
#
#   Filter one variant (keep all PD, use only test-subjects from clean):
#       VARIANT_SUBJECTS = {
#           "s2_w2_fs50_tremor_clean": [2, 7, 11],
#       }
#
#   Explicit None = same as omitting the key (all subjects included):
#       VARIANT_SUBJECTS = {
#           "s2_w2_fs50_tremor_parkinson": None,
#           "s2_w2_fs50_tremor_clean": [2, 7, 11],
#       }
VARIANT_SUBJECTS: Dict[str, Optional[List[int]]] = {
    "s2_w2_fs50_tremor_clean": [1, 14, 19],   # holdout subjects — never seen during training
}


# ═══════════════════════════════════════════════════════════════
# FIXED CONFIGURATION  (must stay identical to run_experiment.py)
# ═══════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR   = Path(__file__).parent.resolve()   # .../experiments/
FUSION_DIR   = SCRIPT_DIR.parent                  # .../Fusion/
PROJECT_ROOT = FUSION_DIR.parent                  # .../Parkinson_dataset_work/

base_data_dir          = PROJECT_ROOT / "Data" / "Tremor_datagenerator_files"
embeddings_folder_name = "Activity_ExtractedFeatures"

batch_size = 256   # larger is fine for inference


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS  (identical to run_experiment.py)
# ═══════════════════════════════════════════════════════════════

class PaperHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int,
                 hidden_dims=(256, 256), dropout_p: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class SensorGate(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
        else:
            self.net = nn.Linear(embed_dim, 1)

    def forward(self, z):
        return self.net(z)


class GatedConcatFusionModel(nn.Module):
    def __init__(self, embed_dim, num_sensors, num_classes,
                 head_hidden_dims=(128, 128), head_dropout=0.3,
                 gate_type="sigmoid", gate_hidden=0, gate_dropout=0.0,
                 use_layernorm=True, alpha_floor=0.0):
        super().__init__()
        self.num_sensors = num_sensors
        self.embed_dim   = embed_dim
        self.gate_type   = gate_type.lower()
        self.alpha_floor = alpha_floor

        assert self.gate_type in ("sigmoid", "softmax")

        self.norms = (
            nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_sensors)])
            if use_layernorm else None
        )
        self.gate_in_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else None
        self.gates = nn.ModuleList([SensorGate(embed_dim, hidden=gate_hidden)
                                    for _ in range(num_sensors)])
        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes,
                              hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        Z_proc = []
        for i, z in enumerate(Z_list):
            if self.norms is not None:
                z = self.norms[i](z)
            if self.gate_in_dropout is not None:
                z = self.gate_in_dropout(z)
            Z_proc.append(z)

        scores = torch.cat([gate(z) for gate, z in zip(self.gates, Z_proc)], dim=1)

        if self.gate_type == "softmax":
            alphas = torch.softmax(scores, dim=1)
        else:
            alphas = torch.sigmoid(scores)

        if self.alpha_floor > 0:
            alphas = self.alpha_floor + (1.0 - self.alpha_floor) * alphas

        z_cat  = torch.cat([alphas[:, i:i+1] * Z_proc[i] for i in range(self.num_sensors)], dim=1)
        logits = self.head(z_cat)
        return logits, alphas


class ConcatFusionModel(nn.Module):
    def __init__(self, embed_dim, num_sensors, num_classes,
                 head_hidden_dims=(256, 256, 128), head_dropout=0.3):
        super().__init__()
        in_dim = embed_dim * num_sensors
        self.head = PaperHead(in_dim=in_dim, num_classes=num_classes,
                              hidden_dims=head_hidden_dims, dropout_p=head_dropout)

    def forward(self, Z_list):
        z      = torch.cat(Z_list, dim=1)
        logits = self.head(z)
        return logits, None


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class MultiSensorDataset(Dataset):
    def __init__(self, Z_list: List[torch.Tensor], y: torch.Tensor):
        self.Z_list = Z_list
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple([Z[idx] for Z in self.Z_list] + [self.y[idx]])


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_embeddings_for_sensor(sensor_name: str, fs: int) -> Dict[str, np.ndarray]:
    """
    Load and concatenate ALL embeddings from TREMOR_VARIANTS for one sensor.
    No train/val/test split — evaluation uses the full dataset.

    Returns
    -------
    dict with keys: Z_all, y_all, subjects
    """
    if not TREMOR_VARIANTS:
        raise ValueError("TREMOR_VARIANTS is empty — add at least one folder name.")

    all_embeddings, all_activities, all_subjects = [], [], []

    for variant_name in TREMOR_VARIANTS:
        npz_path = (base_data_dir / variant_name / embeddings_folder_name
                    / f"{sensor_name}_embeddings.npz")

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {npz_path}\n"
                f"Make sure CNNs are extracted for {sensor_name} (variant: {variant_name})."
            )

        data = np.load(npz_path)

        splits_emb  = [data["train_embeddings"], data["val_embeddings"], data["test_embeddings"]]
        splits_acts = [data["train_activities"],  data["val_activities"],  data["test_activities"]]
        splits_subj = [data["train_subjects"],    data["val_subjects"],    data["test_subjects"]]

        # Include holdout split if present (subjects kept out of training entirely)
        if "holdout_embeddings" in data:
            splits_emb.append(data["holdout_embeddings"])
            splits_acts.append(data["holdout_activities"])
            splits_subj.append(data["holdout_subjects"])

        emb  = np.concatenate(splits_emb,  axis=0).astype(np.float32)
        acts = np.concatenate(splits_acts, axis=0).astype(np.int64)
        subj = np.concatenate(splits_subj, axis=0).astype(np.int64)

        # Apply per-variant subject filter
        keep = VARIANT_SUBJECTS.get(variant_name, None)
        if keep is not None:
            mask = np.isin(subj, keep)
            if mask.sum() == 0:
                available = sorted(np.unique(subj).astype(int).tolist())
                raise ValueError(
                    f"VARIANT_SUBJECTS['{variant_name}']={keep} matched no samples "
                    f"for {sensor_name}. Available subjects: {available}"
                )
            emb  = emb[mask]
            acts = acts[mask]
            subj = subj[mask]

        all_embeddings.append(emb)
        all_activities.append(acts)
        all_subjects.append(subj)

    Z_all    = np.concatenate(all_embeddings, axis=0)
    y_all    = np.concatenate(all_activities, axis=0)
    subj_all = np.concatenate(all_subjects, axis=0)

    subjects = sorted(np.unique(subj_all).astype(int).tolist())

    print(f"  {sensor_name} (fs={fs}): {len(Z_all)} samples across {len(subjects)} subjects ✓")
    print(f"    Subjects: {subjects}")

    return {"Z_all": Z_all, "y_all": y_all, "subjects": subjects}


# ═══════════════════════════════════════════════════════════════
# MODEL BUILDER  (reconstructs architecture from checkpoint keys)
# ═══════════════════════════════════════════════════════════════

def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Reconstruct the model architecture from checkpoint metadata."""
    embed_dim        = ckpt["embed_dim"]
    num_sensors      = ckpt["num_sensors"]
    num_classes      = ckpt["num_classes"]
    head_hidden_dims = tuple(ckpt["head_hidden_dims"])
    head_dropout     = ckpt["head_dropout"]
    use_gating       = ckpt.get("use_gating", False)

    if use_gating:
        # Gating parameters were not saved in the checkpoint (they are fixed
        # constants in run_experiment.py).  Use the same defaults.
        model = GatedConcatFusionModel(
            embed_dim=embed_dim,
            num_sensors=num_sensors,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
            gate_type="sigmoid",
            gate_hidden=64,
            gate_dropout=0.1,
            use_layernorm=True,
            alpha_floor=0.05,
        )
    else:
        model = ConcatFusionModel(
            embed_dim=embed_dim,
            num_sensors=num_sensors,
            num_classes=num_classes,
            head_hidden_dims=head_hidden_dims,
            head_dropout=head_dropout,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Load checkpoint ───────────────────────────────────────
    print("=" * 70)
    print(f"Loading checkpoint: {model_path.name}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    config_id    = ckpt["config_id"]
    sensors      = ckpt["sensors"]           # list, already sorted
    sensors_fs   = ckpt["sensors_fs"]        # {sensor: fs}
    embed_dim    = ckpt["embed_dim"]
    num_classes  = ckpt["num_classes"]
    num_sensors  = ckpt["num_sensors"]
    use_gating   = ckpt.get("use_gating", False)
    train_acc    = ckpt.get("test_accuracy", "unknown")
    train_f1     = ckpt.get("test_f1_macro", "unknown")
    train_seed   = ckpt.get("seed", "unknown")
    train_ts     = ckpt.get("timestamp", "unknown")

    print(f"\nModel info (from checkpoint):")
    print(f"  config_id   : {config_id}")
    print(f"  Sensors     : {sensors}")
    print(f"  sensors_fs  : {sensors_fs}")
    print(f"  embed_dim   : {embed_dim}  |  num_classes: {num_classes}")
    print(f"  Fusion mode : {'GATED' if use_gating else 'CONCAT BASELINE'}")
    print(f"  Trained acc : {train_acc}  |  F1(macro): {train_f1}")
    print(f"  Trained seed: {train_seed}  |  at {train_ts}")

    print(f"\nEvaluation dataset ({len(TREMOR_VARIANTS)} variant(s)):")
    for v in TREMOR_VARIANTS:
        print(f"  - {v}")
    print(f"Eval tag: {EVAL_TAG}")
    if VARIANT_SUBJECTS:
        print("Subject filters (per variant):")
        for v in TREMOR_VARIANTS:
            filt = VARIANT_SUBJECTS.get(v, None)
            label = str(filt) if filt is not None else "all"
            print(f"  {v}: {label}")
    else:
        print("Subject filter: None (all subjects in all variants)")
    print("=" * 70)

    # ── Load embeddings ───────────────────────────────────────
    print("\n[1/4] Loading embeddings...")
    embeddings: Dict[str, Dict] = {}
    for sensor in sensors:
        fs = sensors_fs[sensor]
        embeddings[sensor] = load_embeddings_for_sensor(sensor, fs)

    # Verify label alignment across sensors
    ref = sensors[0]
    ref_y = embeddings[ref]["y_all"]
    for s in sensors[1:]:
        if not np.array_equal(embeddings[s]["y_all"], ref_y):
            raise ValueError(f"Label mismatch between {ref} and {s}.")
    print("  ✓ Labels aligned across all sensors.")

    # ── Build data loader ─────────────────────────────────────
    print("\n[2/4] Building data loader...")

    Z_list = [torch.from_numpy(embeddings[s]["Z_all"]) for s in sensors]
    y_all  = torch.from_numpy(embeddings[ref]["y_all"])
    loader = DataLoader(MultiSensorDataset(Z_list, y_all),
                        batch_size=batch_size, shuffle=False)

    n_total = len(embeddings[ref]["y_all"])
    print(f"  Total samples: {n_total}")

    # ── Rebuild model ─────────────────────────────────────────
    print("\n[3/4] Rebuilding model from checkpoint...")
    model = build_model_from_checkpoint(ckpt)
    model.eval()
    print(model)

    # ── Inference ─────────────────────────────────────────────
    print("\n[4/4] Running inference...")

    y_true_np = embeddings[ref]["y_all"]
    y_pred    = []
    with torch.no_grad():
        for batch in loader:
            *Z, _ = batch
            Z = [z.to(device) for z in Z]
            logits, _ = model(Z)
            y_pred.extend(logits.argmax(1).cpu().numpy())
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true_np, y_pred)
    f1m = f1_score(y_true_np, y_pred, average="macro")
    f1w = f1_score(y_true_np, y_pred, average="weighted")

    print(f"\n{'─'*50}")
    print(f"{'Accuracy':>10} {'F1 macro':>10} {'F1 weighted':>13}")
    print(f"{'─'*50}")
    print(f"{acc:>10.4f} {f1m:>10.4f} {f1w:>13.4f}")
    print(f"{'─'*50}")

    print("\nPer-class report:")
    print(classification_report(y_true_np, y_pred))

    # ── Save outputs ──────────────────────────────────────────
    out_dir = model_path.parent / "eval_results" / EVAL_TAG
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{config_id}_{EVAL_TAG}"

    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=np.arange(num_classes)).plot(
        ax=axes[0], cmap="Blues", values_format="d"
    )
    axes[0].set_title("Absolute Counts", fontsize=14, fontweight="bold")
    for text in axes[0].texts:
        text.set_fontsize(8)

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                           display_labels=np.arange(num_classes)).plot(
        ax=axes[1], cmap="Blues", values_format=".1%"
    )
    axes[1].set_title("Recall per True Label (% of samples)", fontsize=14, fontweight="bold")
    for text in axes[1].texts:
        text.set_fontsize(8)

    fig.suptitle(
        f"Confusion Matrix — {config_id}\n"
        f"Eval: {EVAL_TAG}  |  Acc: {acc:.4f}  |  F1(macro): {f1m:.4f}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    cm_path = out_dir / f"{stem}_cm.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Confusion matrix saved → {cm_path}")

    # JSON result log
    result = {
        "timestamp":             timestamp,
        "eval_tag":              EVAL_TAG,
        "data_variants":         TREMOR_VARIANTS,
        "model_file":            model_path.name,
        "config_id":             config_id,
        "sensors":               sensors,
        "sensors_fs":            sensors_fs,
        "n_samples":             int(n_total),
        "trained_test_acc":      train_acc,
        "trained_test_f1":       train_f1,
        "trained_seed":          train_seed,
        "variant_subjects_filter": VARIANT_SUBJECTS if VARIANT_SUBJECTS else None,
        "eval_accuracy":         round(float(acc), 6),
        "eval_f1_macro":         round(float(f1m), 6),
        "eval_f1_weighted":      round(float(f1w), 6),
    }
    log_path = out_dir / f"{stem}_results.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Results saved       → {log_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print(f"  Model    : {model_path.name}")
    print(f"  Eval tag : {EVAL_TAG}")
    print(f"  Accuracy : {acc:.4f}  |  F1(macro): {f1m:.4f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    evaluate()
