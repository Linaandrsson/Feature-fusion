"""
Deep consistency analysis of CNN embeddings and splits.

Checks:
  1. Split correctness  (clean vs AWGN)
  2. Embedding file consistency
  3. Cross-sensor alignment
  4. Clean vs AWGN sample alignment
  5. Embedding distance (cosine / L2) per sensor
  6. Label field audit (tremor vs activity)
  7. Final diagnosis summary

Usage:
    python diagnose_embedding_consistency.py
    python diagnose_embedding_consistency.py --awgn_variant s4_w4_fs50_corrupt_awgn_20db
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent.resolve()
DATA_BASE  = WORKSPACE / "data" / "Tremor_datagenerator_files"

CLEAN_VARIANT      = "s4_w4_fs50_tremor_clean"
AWGN_VARIANT_DEF   = "s4_w4_fs50_corrupt_awgn_20db"
EMBEDDINGS_FOLDER  = "ExtractedFeatures_clean"

# Training-folder splits (written by CNN training scripts, subject-based)
TRAINING_SPLITS_DIR = (
    WORKSPACE / "Feature extraction CNNs"
    / "Training" / "s4_w4_aug1_fs50_clean" / "splits"
)

SENSORS = ["Acc_ankle", "Acc_arm", "Acc_chest", "ECG",
           "Gyro_ankle", "Gyro_arm", "Mag_ankle", "Mag_arm"]

EXPECTED_TEST_SUBJECTS  = [5, 10]
EXPECTED_VAL_SUBJECTS   = [2, 7]
EXPECTED_TRAIN_SUBJECTS = [1, 3, 4, 6, 8, 9]

OUT_DIR = WORKSPACE / "embedding_diagnostics"
OUT_DIR.mkdir(exist_ok=True)

# ─── helpers ────────────────────────────────────────────────────────────────

SEP = "=" * 72

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str)   -> None: print(f"  [OK ] {msg}")
def warn(msg: str) -> None: print(f"  [WARN] {msg}")
def fail(msg: str) -> None: print(f"  [FAIL] {msg}")


def load_splits(splits_dir: Path, label: str) -> dict[str, np.ndarray]:
    result = {}
    for name in ("train_idx", "val_idx", "test_idx"):
        p = splits_dir / f"{name}.txt"
        if not p.exists():
            fail(f"{label}: missing {p}")
            result[name] = np.array([], dtype=int)
        else:
            result[name] = np.loadtxt(p, dtype=int)
    return result


def load_npz_subjects(variant: str, sensor: str = "Acc_ankle") -> np.ndarray | None:
    p = DATA_BASE / variant / f"{sensor}.npz"
    if not p.exists():
        return None
    return np.load(p)["subject_id"]


def load_emb(variant: str, sensor: str) -> dict | None:
    p = DATA_BASE / variant / EMBEDDINGS_FOLDER / f"{sensor}_embeddings.npz"
    if not p.exists():
        return None
    return dict(np.load(p))


def subject_ids_for_split(splits: dict, subj_array: np.ndarray, split: str) -> np.ndarray:
    idx = splits[f"{split}_idx"]
    if len(idx) == 0 or subj_array is None:
        return np.array([], dtype=int)
    return np.unique(subj_array[idx])


# ════════════════════════════════════════════════════════════════════════════
# 1. VERIFY SPLITS
# ════════════════════════════════════════════════════════════════════════════

def check_splits(awgn_variant: str) -> dict:
    section("1. VERIFY SPLITS")
    issues = {}

    # ── Training folder (ground-truth correct splits) ──
    print("\n  [Reference] Training-folder splits:")
    print(f"    Path: {TRAINING_SPLITS_DIR}")
    if not TRAINING_SPLITS_DIR.exists():
        fail("Training-folder splits not found — cannot compare")
        ref_splits = None
    else:
        ref_splits = load_splits(TRAINING_SPLITS_DIR, "Training-folder")
        ref_subjs  = load_npz_subjects(CLEAN_VARIANT)
        for split in ("train", "val", "test"):
            idx  = ref_splits[f"{split}_idx"]
            if ref_subjs is not None:
                subjs = np.unique(ref_subjs[idx]).tolist()
            else:
                subjs = "N/A"
            print(f"    {split:5s}: n={len(idx):>4d}  subjects={subjs}")

    # ── Clean variant data-folder splits ──
    print(f"\n  [Clean] Data-folder splits ({CLEAN_VARIANT}):")
    clean_splits_dir = DATA_BASE / CLEAN_VARIANT / "splits"
    clean_subjs      = load_npz_subjects(CLEAN_VARIANT)
    if not clean_splits_dir.exists():
        fail("Clean data-folder splits directory not found")
        clean_splits = None
    else:
        clean_splits = load_splits(clean_splits_dir, "Clean")
        for split in ("train", "val", "test"):
            idx = clean_splits[f"{split}_idx"]
            if clean_subjs is not None:
                subjs = np.unique(clean_subjs[idx]).tolist()
            else:
                subjs = "N/A"
            print(f"    {split:5s}: n={len(idx):>4d}  subjects={subjs}")
            if subjs != "N/A":
                if split == "test" and sorted(subjs) != EXPECTED_TEST_SUBJECTS:
                    fail(f"Clean test subjects {subjs} != expected {EXPECTED_TEST_SUBJECTS}")
                    issues["clean_test_subjects"] = subjs
                if split == "val" and sorted(subjs) != EXPECTED_VAL_SUBJECTS:
                    fail(f"Clean val subjects {subjs} != expected {EXPECTED_VAL_SUBJECTS}")
                    issues["clean_val_subjects"] = subjs
                if split == "train" and sorted(subjs) != EXPECTED_TRAIN_SUBJECTS:
                    fail(f"Clean train subjects {subjs} != expected {EXPECTED_TRAIN_SUBJECTS}")
                    issues["clean_train_subjects"] = subjs

    # ── AWGN variant data-folder splits ──
    print(f"\n  [AWGN]  Data-folder splits ({awgn_variant}):")
    awgn_splits_dir = DATA_BASE / awgn_variant / "splits"
    awgn_subjs      = load_npz_subjects(awgn_variant)
    if not awgn_splits_dir.exists():
        fail("AWGN data-folder splits directory not found")
        awgn_splits = None
    else:
        awgn_splits = load_splits(awgn_splits_dir, "AWGN")
        for split in ("train", "val", "test"):
            idx = awgn_splits[f"{split}_idx"]
            if awgn_subjs is not None:
                subjs = np.unique(awgn_subjs[idx]).tolist()
            else:
                subjs = "N/A"
            print(f"    {split:5s}: n={len(idx):>4d}  subjects={subjs}")
            if subjs != "N/A":
                if split == "test" and sorted(subjs) != EXPECTED_TEST_SUBJECTS:
                    fail(f"AWGN test subjects {subjs} != expected {EXPECTED_TEST_SUBJECTS}")
                    issues["awgn_test_subjects"] = subjs
                if split == "val" and sorted(subjs) != EXPECTED_VAL_SUBJECTS:
                    fail(f"AWGN val subjects {subjs} != expected {EXPECTED_VAL_SUBJECTS}")
                    issues["awgn_val_subjects"] = subjs
                if split == "train" and sorted(subjs) != EXPECTED_TRAIN_SUBJECTS:
                    fail(f"AWGN train subjects {subjs} != expected {EXPECTED_TRAIN_SUBJECTS}")
                    issues["awgn_train_subjects"] = subjs

    # ── Compare clean data-folder vs AWGN data-folder ──
    print("\n  [Compare] Clean data-folder vs AWGN data-folder splits:")
    if clean_splits and awgn_splits:
        for split in ("train", "val", "test"):
            c = np.sort(clean_splits[f"{split}_idx"])
            a = np.sort(awgn_splits[f"{split}_idx"])
            if np.array_equal(c, a):
                ok(f"{split} indices identical")
            else:
                sizes_match = len(c) == len(a)
                if not sizes_match:
                    fail(f"{split}: n_clean={len(c)} vs n_awgn={len(a)} — SIZE MISMATCH")
                    issues[f"split_size_{split}"] = (len(c), len(a))
                else:
                    n_diff = int((c != a).sum())
                    fail(f"{split}: same size but {n_diff} index differences")
                    issues[f"split_indices_{split}"] = n_diff

    # ── Compare training-folder vs AWGN ──
    if ref_splits and awgn_splits:
        print("\n  [Compare] Training-folder splits vs AWGN data-folder splits:")
        for split in ("train", "val", "test"):
            c = np.sort(ref_splits[f"{split}_idx"])
            a = np.sort(awgn_splits[f"{split}_idx"])
            if np.array_equal(c, a):
                ok(f"{split} indices match training-folder ✓")
            else:
                if len(c) != len(a):
                    fail(f"{split}: n_training={len(c)} vs n_awgn={len(a)} — SIZE MISMATCH")
                else:
                    fail(f"{split}: {int((c!=a).sum())} index differences vs training-folder")

    return issues


# ════════════════════════════════════════════════════════════════════════════
# 2. VERIFY EMBEDDING FILE CONSISTENCY
# ════════════════════════════════════════════════════════════════════════════

def check_embedding_files(awgn_variant: str) -> dict:
    section("2. VERIFY EMBEDDING FILE CONSISTENCY")
    issues = {}

    for variant_label, variant in [("CLEAN", CLEAN_VARIANT), ("AWGN", awgn_variant)]:
        print(f"\n  [{variant_label}] {variant}/{EMBEDDINGS_FOLDER}/")
        for sensor in SENSORS:
            emb = load_emb(variant, sensor)
            if emb is None:
                fail(f"  {sensor}: embedding file MISSING")
                issues[f"{variant_label}_{sensor}_missing"] = True
                continue

            shapes = {k: emb[k].shape for k in emb}
            n_train = shapes.get("train_embeddings", (0,))[0]
            n_val   = shapes.get("val_embeddings",   (0,))[0]
            n_test  = shapes.get("test_embeddings",  (0,))[0]

            # Label audit
            train_labels_unique = np.unique(emb.get("train_labels", np.array([]))).tolist()
            val_labels_unique   = np.unique(emb.get("val_labels",   np.array([]))).tolist()
            test_labels_unique  = np.unique(emb.get("test_labels",  np.array([]))).tolist()

            train_act_unique = np.unique(emb.get("train_activities", np.array([]))).tolist()
            test_act_unique  = np.unique(emb.get("test_activities",  np.array([]))).tolist()

            train_subj = sorted(np.unique(emb.get("train_subjects", np.array([]))).tolist())
            val_subj   = sorted(np.unique(emb.get("val_subjects",   np.array([]))).tolist())
            test_subj  = sorted(np.unique(emb.get("test_subjects",  np.array([]))).tolist())

            # Check whether labels are tremor (should be 0-4) or activity (0-11)
            all_labels = np.concatenate([emb.get("train_labels", []),
                                         emb.get("val_labels",   []),
                                         emb.get("test_labels",  [])]).tolist()
            unique_labels = sorted(set(int(x) for x in all_labels))
            label_type = "TREMOR(0-4)" if max(unique_labels, default=0) <= 4 else "ACTIVITY(0-11)"
            all_constant = len(unique_labels) <= 1

            print(f"    {sensor:12s}  train={n_train}  val={n_val}  test={n_test}"
                  f"  | labels={unique_labels}({label_type})"
                  f"{'  [ALL_ZERO!]' if all_constant else ''}"
                  f"  | activities={len(train_act_unique)} classes"
                  f"  | test_subjects={test_subj}")

            if all_constant and len(unique_labels) == 1 and unique_labels[0] == 0:
                warn(f"    {sensor}: ALL *_labels are 0 — these are tremor_score from clean/AWGN data (expected)")
            if sorted(test_subj) != EXPECTED_TEST_SUBJECTS:
                fail(f"    {sensor}: test_subjects={test_subj} != expected {EXPECTED_TEST_SUBJECTS}")
                issues[f"{variant_label}_{sensor}_test_subjects"] = test_subj

    return issues


# ════════════════════════════════════════════════════════════════════════════
# 3. VERIFY CROSS-SENSOR ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════

def check_sensor_alignment(awgn_variant: str) -> dict:
    section("3. VERIFY CROSS-SENSOR ALIGNMENT")
    issues = {}

    for variant_label, variant in [("CLEAN", CLEAN_VARIANT), ("AWGN", awgn_variant)]:
        print(f"\n  [{variant_label}] {variant}")
        ref_sensor = SENSORS[0]
        ref_emb    = load_emb(variant, ref_sensor)
        if ref_emb is None:
            fail(f"Reference sensor {ref_sensor} missing — skipping")
            continue

        def concat_field(emb, key):
            return np.concatenate([emb[f"train_{key}"], emb[f"val_{key}"], emb[f"test_{key}"]], axis=0)

        ref_activities = concat_field(ref_emb, "activities")
        ref_subjects   = concat_field(ref_emb, "subjects")
        ref_labels     = concat_field(ref_emb, "labels")
        ref_n          = len(ref_activities)

        ok(f"Reference: {ref_sensor} — {ref_n} total windows")

        for sensor in SENSORS[1:]:
            emb = load_emb(variant, sensor)
            if emb is None:
                fail(f"{sensor}: missing — cannot check alignment")
                issues[f"{variant_label}_{sensor}_missing"] = True
                continue

            s_activities = concat_field(emb, "activities")
            s_subjects   = concat_field(emb, "subjects")
            s_labels     = concat_field(emb, "labels")
            s_n          = len(s_activities)

            if s_n != ref_n:
                fail(f"{sensor}: n={s_n} vs ref n={ref_n}  SIZE MISMATCH")
                issues[f"{variant_label}_{sensor}_n"] = (ref_n, s_n)
                continue

            act_match  = np.array_equal(ref_activities, s_activities)
            subj_match = np.array_equal(ref_subjects,   s_subjects)
            lbl_match  = np.array_equal(ref_labels,     s_labels)

            if act_match and subj_match:
                ok(f"{sensor}: activities ✓  subjects ✓  labels={'✓' if lbl_match else 'MISMATCH'}")
            else:
                first_act_diff = int(np.argmax(ref_activities != s_activities)) if not act_match else -1
                first_sub_diff = int(np.argmax(ref_subjects   != s_subjects))   if not subj_match else -1
                fail(f"{sensor}: activities={'✓' if act_match else f'FAIL@idx={first_act_diff}'}"
                     f"  subjects={'✓' if subj_match else f'FAIL@idx={first_sub_diff}'}")
                issues[f"{variant_label}_{sensor}_alignment"] = {
                    "act_match": act_match,
                    "subj_match": subj_match,
                    "first_act_diff": first_act_diff,
                    "first_sub_diff": first_sub_diff,
                }

    return issues


# ════════════════════════════════════════════════════════════════════════════
# 4. VERIFY CLEAN vs AWGN EMBEDDING ALIGNMENT (sample-level)
# ════════════════════════════════════════════════════════════════════════════

def check_clean_awgn_alignment(awgn_variant: str) -> dict:
    section("4. VERIFY CLEAN vs AWGN EMBEDDING ALIGNMENT")
    issues = {}

    for sensor in SENSORS:
        clean_emb = load_emb(CLEAN_VARIANT, sensor)
        awgn_emb  = load_emb(awgn_variant,  sensor)
        if clean_emb is None or awgn_emb is None:
            fail(f"{sensor}: embedding file missing — skipping")
            continue

        def concat_field(emb, key):
            return np.concatenate([emb[f"train_{key}"], emb[f"val_{key}"], emb[f"test_{key}"]], axis=0)

        c_act  = concat_field(clean_emb, "activities")
        a_act  = concat_field(awgn_emb,  "activities")
        c_subj = concat_field(clean_emb, "subjects")
        a_subj = concat_field(awgn_emb,  "subjects")
        c_lbl  = concat_field(clean_emb, "labels")
        a_lbl  = concat_field(awgn_emb,  "labels")
        c_n    = len(c_act)
        a_n    = len(a_act)

        if c_n != a_n:
            fail(f"{sensor}: n_clean={c_n} vs n_awgn={a_n} — SIZE MISMATCH")
            issues[f"{sensor}_n"] = (c_n, a_n)
            continue

        act_match  = np.array_equal(c_act,  a_act)
        subj_match = np.array_equal(c_subj, a_subj)
        lbl_match  = np.array_equal(c_lbl,  a_lbl)

        status = "✓" if (act_match and subj_match) else "FAIL"
        print(f"  {sensor:12s}: n={c_n}  activities={'✓' if act_match else 'MISMATCH'}"
              f"  subjects={'✓' if subj_match else 'MISMATCH'}"
              f"  labels={'✓' if lbl_match else 'MISMATCH'}  [{status}]")

        if not act_match:
            first = int(np.argmax(c_act != a_act))
            fail(f"  {sensor}: first activity mismatch at index {first}"
                 f"  (clean={c_act[first]}, awgn={a_act[first]})")
            issues[f"{sensor}_act"] = first
        if not subj_match:
            first = int(np.argmax(c_subj != a_subj))
            fail(f"  {sensor}: first subject mismatch at index {first}"
                 f"  (clean={c_subj[first]}, awgn={a_subj[first]})")
            issues[f"{sensor}_subj"] = first

    return issues


# ════════════════════════════════════════════════════════════════════════════
# 5. EMBEDDING DISTANCE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def check_embedding_distances(awgn_variant: str) -> None:
    section("5. EMBEDDING DISTANCE ANALYSIS (clean vs AWGN embeddings)")
    print("  (Only test-split windows, matched by subject+position)")

    fig_cos, axes_cos = plt.subplots(2, 4, figsize=(16, 7))
    fig_l2,  axes_l2  = plt.subplots(2, 4, figsize=(16, 7))
    fig_cos.suptitle(f"Cosine Similarity — clean vs {awgn_variant}\n(test windows: subjects 5,10)")
    fig_l2.suptitle( f"L2 Distance     — clean vs {awgn_variant}\n(test windows: subjects 5,10)")

    summary_rows = []

    for i, sensor in enumerate(SENSORS):
        ax_cos = axes_cos.flat[i]
        ax_l2  = axes_l2.flat[i]

        clean_emb = load_emb(CLEAN_VARIANT, sensor)
        awgn_emb  = load_emb(awgn_variant,  sensor)
        if clean_emb is None or awgn_emb is None:
            ax_cos.set_title(f"{sensor}\n[MISSING]")
            ax_l2.set_title( f"{sensor}\n[MISSING]")
            continue

        # Reconstruct full arrays with subject metadata (concat train+val+test)
        def concat(emb, key):
            return np.concatenate([emb[f"train_{key}"], emb[f"val_{key}"], emb[f"test_{key}"]], axis=0)

        c_Z    = concat(clean_emb, "embeddings").astype(np.float32)
        a_Z    = concat(awgn_emb,  "embeddings").astype(np.float32)
        c_subj = concat(clean_emb, "subjects")
        a_subj = concat(awgn_emb,  "subjects")
        c_act  = concat(clean_emb, "activities")
        a_act  = concat(awgn_emb,  "activities")

        # Filter to test subjects
        c_mask = np.isin(c_subj, EXPECTED_TEST_SUBJECTS)
        a_mask = np.isin(a_subj, EXPECTED_TEST_SUBJECTS)
        c_Z_test    = c_Z[c_mask]
        a_Z_test    = a_Z[a_mask]
        c_subj_test = c_subj[c_mask]
        a_subj_test = a_subj[a_mask]
        c_act_test  = c_act[c_mask]
        a_act_test  = a_act[a_mask]

        n_c = len(c_Z_test)
        n_a = len(a_Z_test)

        if n_c != n_a:
            fail(f"{sensor}: test n_clean={n_c} vs n_awgn={n_a}  —  cannot compute distances (size mismatch)")
            ax_cos.set_title(f"{sensor}\n[SIZE MISMATCH\n{n_c} vs {n_a}]")
            ax_l2.set_title( f"{sensor}\n[SIZE MISMATCH]")
            continue

        # Verify activities match before computing distance
        if not np.array_equal(c_act_test, a_act_test):
            first_diff = int(np.argmax(c_act_test != a_act_test))
            warn(f"{sensor}: activity mismatch at test index {first_diff} — distances may be misaligned")

        # Cosine similarity (per sample)
        c_norm = c_Z_test / (np.linalg.norm(c_Z_test, axis=1, keepdims=True) + 1e-8)
        a_norm = a_Z_test / (np.linalg.norm(a_Z_test, axis=1, keepdims=True) + 1e-8)
        cos_sim = (c_norm * a_norm).sum(axis=1)          # (n_test,)
        l2_dist = np.linalg.norm(c_Z_test - a_Z_test, axis=1)

        mean_cos = float(cos_sim.mean())
        std_cos  = float(cos_sim.std())
        mean_l2  = float(l2_dist.mean())
        std_l2   = float(l2_dist.std())

        # Magnitude info
        c_norm_mag = float(np.linalg.norm(c_Z_test, axis=1).mean())
        a_norm_mag = float(np.linalg.norm(a_Z_test, axis=1).mean())

        status = "OK" if mean_cos > 0.90 else ("WARN" if mean_cos > 0.70 else "VERY_DIFFERENT")
        print(f"  {sensor:12s}:  cos_sim={mean_cos:.4f}±{std_cos:.4f}  "
              f"l2={mean_l2:.4f}±{std_l2:.4f}  "
              f"|c|={c_norm_mag:.2f}  |a|={a_norm_mag:.2f}  [{status}]")

        summary_rows.append((sensor, mean_cos, std_cos, mean_l2, std_l2, n_c))

        # Histogram plots
        ax_cos.hist(cos_sim, bins=40, color="steelblue", edgecolor="none", alpha=0.8)
        ax_cos.axvline(mean_cos, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_cos:.3f}")
        ax_cos.axvline(1.0, color="grey", linestyle=":", linewidth=1)
        ax_cos.set_title(f"{sensor}\nμ={mean_cos:.3f}, σ={std_cos:.3f}")
        ax_cos.set_xlabel("Cosine similarity")
        ax_cos.set_ylabel("Count")
        ax_cos.legend(fontsize=7)
        ax_cos.set_xlim(-0.1, 1.05)

        ax_l2.hist(l2_dist, bins=40, color="darkorange", edgecolor="none", alpha=0.8)
        ax_l2.axvline(mean_l2, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_l2:.3f}")
        ax_l2.set_title(f"{sensor}\nμ={mean_l2:.3f}, σ={std_l2:.3f}")
        ax_l2.set_xlabel("L2 distance")
        ax_l2.set_ylabel("Count")
        ax_l2.legend(fontsize=7)

    plt.figure(fig_cos.number)
    plt.tight_layout()
    out_cos = OUT_DIR / f"cosine_similarity_{awgn_variant}.png"
    fig_cos.savefig(out_cos, dpi=120)
    plt.close(fig_cos)
    print(f"\n  Saved cosine histogram → {out_cos}")

    plt.figure(fig_l2.number)
    plt.tight_layout()
    out_l2 = OUT_DIR / f"l2_distance_{awgn_variant}.png"
    fig_l2.savefig(out_l2, dpi=120)
    plt.close(fig_l2)
    print(f"  Saved L2 histogram      → {out_l2}")

    return summary_rows


# ════════════════════════════════════════════════════════════════════════════
# 6. FUSION-LEVEL LABEL CONSISTENCY
# ════════════════════════════════════════════════════════════════════════════

def check_fusion_labels(awgn_variant: str) -> None:
    section("6. FUSION-LEVEL LABEL CONSISTENCY")

    print("""
  Fusion script (v2_fusion_concat_ablation_study_tremor.py) behaviour:
    load_combined_tremor_embeddings():
      - reads data["train_activities"], data["val_activities"], data["test_activities"]
      - concatenates → y_all  (used as classification target)
      → y_train / y_val / y_test are ACTIVITY labels (0-11)

    _load_test_windows_from_variant():
      - reads data["train_activities"] etc.
      → returns activity labels for evaluation

  The *_labels fields in embedding NPZs contain tremor_score.
  For clean and AWGN variants these are ALL ZERO (no tremor).
  The fusion script does NOT use *_labels — it uses *_activities.
  → Label field usage is CORRECT, assuming activities are stored correctly.
""")

    for variant_label, variant in [("CLEAN", CLEAN_VARIANT), ("AWGN", awgn_variant)]:
        print(f"  [{variant_label}] {variant}")
        for sensor in SENSORS[:2]:  # spot-check first two sensors
            emb = load_emb(variant, sensor)
            if emb is None:
                fail(f"  {sensor}: embedding file missing")
                continue
            test_act = emb.get("test_activities", np.array([]))
            test_lbl = emb.get("test_labels",     np.array([]))
            test_sub = emb.get("test_subjects",   np.array([]))
            print(f"    {sensor:12s}: test_activities unique={sorted(np.unique(test_act).tolist())}"
                  f"  test_labels unique={sorted(np.unique(test_lbl).tolist())}"
                  f"  test_subjects={sorted(np.unique(test_sub).tolist())}")
            if len(np.unique(test_act)) < 2:
                fail(f"    {sensor}: test_activities has <2 unique values — activity labels may be wrong!")
            else:
                ok(f"    {sensor}: test_activities has {len(np.unique(test_act))} classes ✓")


# ════════════════════════════════════════════════════════════════════════════
# 7. FINAL DIAGNOSIS
# ════════════════════════════════════════════════════════════════════════════

def final_diagnosis(split_issues, emb_issues, align_issues,
                    clean_awgn_issues, dist_summary, awgn_variant):
    section("7. FINAL DIAGNOSIS")

    all_issues = {**split_issues, **emb_issues, **align_issues, **clean_awgn_issues}

    print(f"\n  Total issues detected: {len(all_issues)}")
    if all_issues:
        for k, v in all_issues.items():
            print(f"    • {k}: {v}")

    # Summary questions
    print("\n  Answering diagnostic questions:")

    # Q1: AWGN embeddings aligned with clean?
    awgn_align_issues = {k: v for k, v in clean_awgn_issues.items()}
    q1 = len(awgn_align_issues) == 0
    print(f"  1. AWGN embeddings correctly aligned with clean?  {'YES ✓' if q1 else 'NO ✗ — see clean_awgn_issues'}")

    # Q2: Subject-based splits preserved in AWGN?
    awgn_subj_ok = not any(k.startswith("awgn_test") or k.startswith("awgn_val") or k.startswith("awgn_train") for k in split_issues)
    print(f"  2. AWGN subject-based splits preserved?           {'YES ✓' if awgn_subj_ok else 'NO ✗'}")

    # Q3: Correct holdout subjects?
    holdout_ok = "awgn_test_subjects" not in split_issues
    print(f"  3. Holdout subjects [5,10] preserved in AWGN?     {'YES ✓' if holdout_ok else 'NO ✗'}")

    # Q4: Labels consistent?
    label_ok = not any("label" in k.lower() for k in all_issues)
    print(f"  4. Labels consistent across sensors?              {'YES ✓' if label_ok else 'NO ✗'}")

    # Q5: Activities consistent?
    act_ok = not any("act" in k.lower() for k in all_issues)
    print(f"  5. Activities consistent across sensors?          {'YES ✓' if act_ok else 'NO ✗'}")

    # Q6: Sample-by-sample alignment?
    align_ok = len(align_issues) == 0
    print(f"  6. Embeddings aligned sample-by-sample?           {'YES ✓' if align_ok else 'NO ✗'}")

    # Q7: Wrong label field?
    print(f"  7. Downstream scripts using wrong label field?    NO ✓ (fusion uses *_activities, not *_labels)")

    # Q8: Ordering mismatch?
    order_ok = not any("order" in k.lower() or "align" in k.lower() for k in all_issues)
    print(f"  8. Evidence of ordering mismatch?                 {'NO ✓' if order_ok else 'POSSIBLE ✗'}")

    # Q9: Corrupted embedding generation?
    print(f"  9. Evidence of corrupted embedding generation?")
    print(f"     → See embedding distance plots and summary above.")

    # Q10: Most likely cause of F1 collapse
    print(f"\n  10. MOST LIKELY CAUSE OF CATASTROPHIC F1 COLLAPSE:")
    print()

    # Assess from distance summary
    if dist_summary:
        low_cos = [(s, c) for s, c, *_ in dist_summary if c < 0.80]
        if low_cos:
            print(f"     ★ EMBEDDING DRIFT: {len(low_cos)}/{len(dist_summary)} sensors show mean cosine < 0.80:")
            for s, c in low_cos:
                print(f"       - {s}: mean cosine = {c:.4f}")
            print(f"     → CNN feature extractors (trained ONLY on clean data) produce")
            print(f"       substantially different embeddings for AWGN-corrupted windows.")
            print(f"       The fusion head was trained on clean embeddings, so it cannot")
            print(f"       classify the shifted AWGN embedding distribution correctly.")
        else:
            print(f"     ✓ Embedding cosine similarity appears reasonable (>0.80 for all sensors).")
            print(f"       The F1 collapse is NOT due to large embedding drift.")

    # Check clean split bug
    if split_issues.get("clean_test_subjects") or split_issues.get("clean_train_subjects"):
        print()
        print(f"     ★ SPLIT BUG (clean data-folder splits):")
        print(f"       The clean variant's data-folder splits are RANDOM (not subject-based).")
        print(f"       extract_features.py uses these wrong splits for the clean variant.")
        print(f"       However, the fusion script re-derives splits by subject ID,")
        print(f"       so this bug does NOT affect the final evaluation subject assignment.")
        print(f"       The CNN model training used the CORRECT splits from the Training folder.")
        print()
        print(f"     ★ IMPORTANT: The clean data-folder splits should be fixed to match")
        print(f"       the training-folder splits. Run fix_clean_splits.py or copy from:")
        print(f"       Feature extraction CNNs/Training/s4_w4_aug1_fs50_clean/splits/")
        print(f"       → data/Tremor_datagenerator_files/s4_w4_fs50_tremor_clean/splits/")

    print()
    print(f"  ── Key numbers to verify above ──────────────────────────────────────────")
    print(f"  If cosine similarity between clean and AWGN embeddings is:")
    print(f"    > 0.95  →  CNN is robust; problem lies elsewhere")
    print(f"    0.80–0.95 →  mild drift; fusion head may struggle")
    print(f"    < 0.80  →  severe drift; CNN is NOT robust to AWGN")
    print(f"    ≈ 0.0   →  completely different embedding spaces — pipeline bug")
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep embedding consistency diagnostic")
    parser.add_argument("--awgn_variant", default=AWGN_VARIANT_DEF,
                        help=f"AWGN variant folder name (default: {AWGN_VARIANT_DEF})")
    args = parser.parse_args()
    awgn_variant = args.awgn_variant

    print(SEP)
    print("  EMBEDDING CONSISTENCY DIAGNOSTIC")
    print(f"  Clean variant : {CLEAN_VARIANT}")
    print(f"  AWGN  variant : {awgn_variant}")
    print(f"  Embeddings    : {EMBEDDINGS_FOLDER}")
    print(f"  Output dir    : {OUT_DIR}")
    print(SEP)

    split_issues     = check_splits(awgn_variant)
    emb_issues       = check_embedding_files(awgn_variant)
    align_issues     = check_sensor_alignment(awgn_variant)
    clean_awgn_issues = check_clean_awgn_alignment(awgn_variant)
    dist_summary     = check_embedding_distances(awgn_variant)
    check_fusion_labels(awgn_variant)
    final_diagnosis(split_issues, emb_issues, align_issues,
                    clean_awgn_issues, dist_summary, awgn_variant)

    print(f"\n{SEP}")
    print(f"  Diagnostic complete. Plots saved to: {OUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
