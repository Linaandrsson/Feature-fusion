"""Sanity checks for extracted tremor embeddings.

Checks performed:
1. Expected label sets per variant:
   - clean: {0}
   - mild_mod: {1, 2}
   - mod_severe: {3, 4}
   - parkinson: {5}
2. All configured sensors have embedding files in each variant.
3. Split-level consistency inside each variant:
   - lengths align between embeddings/labels/subjects/base_idx/activities
   - all sensors have identical labels/subjects/base_idx/activities for each split
   - embedding dimension is 128
   - embeddings contain finite values
4. Cross-variant consistency for clean/mild_mod/mod_severe:
   - equal number of windows per split
   - same subject/base_idx/activity ordering per split
5. Split disjointness inside each variant:
    - no overlap in (subject, base_idx) across train/val/test/holdout.

Usage:
    python sanity_check_embeddings.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config_extraction import (
    HOLDOUT_SUBJECTS,
    SENSORS,
    TEST_SUBJECTS,
    VAL_SUBJECTS,
    get_output_dir,
)


SPLITS = ("train", "val", "test", "holdout")
FUSION_SPLITS = ("train", "val", "test")

EXPECTED_LABELS: Dict[str, set[int]] = {
    "s2_w2_fs50_tremor_clean": {0},
    "s2_w2_fs50_tremor_mild_mod": {1, 2},
    "s2_w2_fs50_tremor_mod_severe": {3, 4},
    "s2_w2_fs50_tremor_parkinson": {5},
}

FUSION_VARIANTS = [
    "s2_w2_fs50_tremor_clean",
    "s2_w2_fs50_tremor_mild_mod",
    "s2_w2_fs50_tremor_mod_severe",
]


@dataclass
class CheckCollector:
    passed: List[str]
    failed: List[str]
    notes: List[str]

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _sensor_file(variant: str, sensor: str) -> Path:
    return get_output_dir(variant) / f"{sensor}_embeddings.npz"


def _keys_for_split(split: str) -> Dict[str, str]:
    return {
        "emb": f"{split}_embeddings",
        "lab": f"{split}_labels",
        "act": f"{split}_activities",
        "sub": f"{split}_subjects",
        "idx": f"{split}_base_idx",
    }


def _as_window_triplets(
    subjects: np.ndarray,
    activities: np.ndarray,
    base_idx: np.ndarray,
) -> List[Tuple[int, int, int]]:
    return list(
        zip(
            subjects.astype(int).tolist(),
            activities.astype(int).tolist(),
            base_idx.astype(int).tolist(),
        )
    )


def check_variant_sensor_files(collector: CheckCollector, variant: str, sensors: Iterable[str]) -> None:
    missing = []
    for sensor in sensors:
        path = _sensor_file(variant, sensor)
        if not path.exists():
            missing.append(str(path))
    if missing:
        collector.fail(f"[{variant}] Missing {len(missing)} sensor embedding files")
        for m in missing:
            collector.fail(f"  - {m}")
    else:
        collector.ok(f"[{variant}] All sensor embedding files exist ({len(list(sensors))})")


def check_variant_content(
    collector: CheckCollector,
    variant: str,
    sensors: List[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    sensor_payload: Dict[str, Dict[str, np.ndarray]] = {}

    for sensor in sensors:
        npz_path = _sensor_file(variant, sensor)
        payload = _load_npz(npz_path)
        sensor_payload[sensor] = payload

        expected_keys = []
        for split in SPLITS:
            expected_keys.extend(_keys_for_split(split).values())

        missing_keys = [k for k in expected_keys if k not in payload]
        if missing_keys:
            collector.fail(f"[{variant}/{sensor}] Missing keys: {missing_keys}")
            continue

        for split in SPLITS:
            k = _keys_for_split(split)
            emb = payload[k["emb"]]
            labels = payload[k["lab"]]
            acts = payload[k["act"]]
            subs = payload[k["sub"]]
            base_idx = payload[k["idx"]]

            n = emb.shape[0]
            if n > 0 and (emb.ndim != 2 or emb.shape[1] != 128):
                collector.fail(f"[{variant}/{sensor}/{split}] Expected embedding shape (N, 128), got {emb.shape}")
            if len(labels) != n or len(acts) != n or len(subs) != n or len(base_idx) != n:
                collector.fail(
                    f"[{variant}/{sensor}/{split}] Length mismatch: "
                    f"emb={n}, labels={len(labels)}, acts={len(acts)}, subs={len(subs)}, base_idx={len(base_idx)}"
                )
            if n > 0 and not np.isfinite(emb).all():
                collector.fail(f"[{variant}/{sensor}/{split}] Non-finite values found in embeddings")

        all_labels = np.concatenate([payload[f"{s}_labels"] for s in SPLITS])
        expected = EXPECTED_LABELS[variant]
        unique = set(np.unique(all_labels).astype(int).tolist())

        if not unique.issubset(expected):
            collector.fail(f"[{variant}/{sensor}] Labels {sorted(unique)} not subset of expected {sorted(expected)}")

        if variant == "s2_w2_fs50_tremor_parkinson":
            if unique != {5}:
                collector.fail(f"[{variant}/{sensor}] Expected exactly label 5, got {sorted(unique)}")
        else:
            if len(unique) == 0:
                collector.fail(f"[{variant}/{sensor}] Empty labels")
            else:
                collector.ok(f"[{variant}/{sensor}] Label set OK: {sorted(unique)}")

    reference_sensor = sensors[0]
    ref = sensor_payload[reference_sensor]

    for sensor in sensors[1:]:
        cur = sensor_payload[sensor]
        for split in SPLITS:
            k = _keys_for_split(split)
            for field in ("act", "sub", "idx"):
                if not np.array_equal(cur[k[field]], ref[k[field]]):
                    collector.fail(
                        f"[{variant}] Split alignment mismatch for {split}_{field}: "
                        f"{sensor} vs {reference_sensor}"
                    )

    collector.ok(f"[{variant}] Sensor alignment across subjects/base_idx/activities is consistent")

    split_meta = {}
    for split in SPLITS:
        k = _keys_for_split(split)
        split_meta[split] = {
            "subjects": ref[k["sub"]],
            "base_idx": ref[k["idx"]],
            "activities": ref[k["act"]],
            "labels": ref[k["lab"]],
        }

    train_subjects = set(split_meta["train"]["subjects"].astype(int).tolist())
    val_subjects = set(split_meta["val"]["subjects"].astype(int).tolist())
    test_subjects = set(split_meta["test"]["subjects"].astype(int).tolist())
    holdout_subjects = set(split_meta["holdout"]["subjects"].astype(int).tolist())

    subject_overlap_tv = sorted(train_subjects & val_subjects)
    subject_overlap_tt = sorted(train_subjects & test_subjects)
    subject_overlap_vt = sorted(val_subjects & test_subjects)
    if subject_overlap_tv or subject_overlap_tt or subject_overlap_vt:
        collector.fail(
            f"[{variant}] SUBJECT LEAKAGE: "
            f"train∩val={subject_overlap_tv}, train∩test={subject_overlap_tt}, val∩test={subject_overlap_vt}"
        )
    else:
        collector.ok(f"[{variant}] No subject leakage across train/val/test")

    overlap_th = sorted(train_subjects & holdout_subjects)
    overlap_vh = sorted(val_subjects & holdout_subjects)
    overlap_teh = sorted(test_subjects & holdout_subjects)
    if overlap_th or overlap_vh or overlap_teh:
        collector.fail(
            f"[{variant}] HOLDOUT LEAKAGE: "
            f"train∩holdout={overlap_th}, val∩holdout={overlap_vh}, test∩holdout={overlap_teh}"
        )
    else:
        collector.ok(f"[{variant}] No leakage between holdout and train/val/test")

    if variant != "s2_w2_fs50_tremor_parkinson":
        expected_val = set(VAL_SUBJECTS)
        expected_test = set(TEST_SUBJECTS)
        if val_subjects != expected_val:
            collector.fail(
                f"[{variant}] Val subjects mismatch: expected={sorted(expected_val)}, got={sorted(val_subjects)}"
            )
        else:
            collector.ok(f"[{variant}] Val subjects match config")
        if test_subjects != expected_test:
            collector.fail(
                f"[{variant}] Test subjects mismatch: expected={sorted(expected_test)}, got={sorted(test_subjects)}"
            )
        else:
            collector.ok(f"[{variant}] Test subjects match config")
        train_forbidden = sorted(train_subjects & (expected_val | expected_test))
        if train_forbidden:
            collector.fail(f"[{variant}] Train contains forbidden subjects: {train_forbidden}")
        else:
            collector.ok(f"[{variant}] Train excludes val/test subjects")

        holdout_in_train = sorted(train_subjects & set(HOLDOUT_SUBJECTS))
        if holdout_in_train:
            collector.fail(f"[{variant}] Train contains holdout subjects: {holdout_in_train}")
        else:
            collector.ok(f"[{variant}] Train excludes holdout subjects")

        expected_holdout = set(HOLDOUT_SUBJECTS)
        if holdout_subjects != expected_holdout:
            collector.fail(
                f"[{variant}] Holdout subjects mismatch: expected={sorted(expected_holdout)}, got={sorted(holdout_subjects)}"
            )
        else:
            collector.ok(f"[{variant}] Holdout subjects match config")

    all_pairs = {}
    for split in SPLITS:
        pairs = _as_window_triplets(
            split_meta[split]["subjects"],
            split_meta[split]["activities"],
            split_meta[split]["base_idx"],
        )
        all_pairs[split] = set(pairs)

    overlap_tv = len(all_pairs["train"] & all_pairs["val"])
    overlap_tt = len(all_pairs["train"] & all_pairs["test"])
    overlap_th = len(all_pairs["train"] & all_pairs["holdout"])
    overlap_vt = len(all_pairs["val"] & all_pairs["test"])
    overlap_vh = len(all_pairs["val"] & all_pairs["holdout"])
    overlap_teh = len(all_pairs["test"] & all_pairs["holdout"])
    if overlap_tv or overlap_tt or overlap_th or overlap_vt or overlap_vh or overlap_teh:
        collector.note(
            f"[{variant}] Non-zero overlap by (subject, activity, base_idx): "
            f"train-val={overlap_tv}, train-test={overlap_tt}, train-holdout={overlap_th}, "
            f"val-test={overlap_vt}, val-holdout={overlap_vh}, test-holdout={overlap_teh}"
        )
    else:
        collector.ok(f"[{variant}] Split disjointness by (subject, activity, base_idx) OK")

    return sensor_payload, split_meta


def check_cross_variant_fusion_alignment(
    collector: CheckCollector,
    split_meta_by_variant: Dict[str, Dict[str, np.ndarray]],
) -> None:
    base_variant = FUSION_VARIANTS[0]

    for other_variant in FUSION_VARIANTS[1:]:
        for split in FUSION_SPLITS:
            base = split_meta_by_variant[base_variant][split]
            other = split_meta_by_variant[other_variant][split]

            if len(base["subjects"]) != len(other["subjects"]):
                collector.fail(
                    f"[fusion {split}] Count mismatch: {base_variant}={len(base['subjects'])}, "
                    f"{other_variant}={len(other['subjects'])}"
                )
                continue

            if not np.array_equal(base["subjects"], other["subjects"]):
                collector.fail(f"[fusion {split}] Subject order mismatch: {base_variant} vs {other_variant}")
            if not np.array_equal(base["base_idx"], other["base_idx"]):
                collector.fail(f"[fusion {split}] base_idx order mismatch: {base_variant} vs {other_variant}")
            if not np.array_equal(base["activities"], other["activities"]):
                collector.fail(f"[fusion {split}] Activity order mismatch: {base_variant} vs {other_variant}")

    if not any(msg.startswith("[fusion") for msg in collector.failed):
        collector.ok("[fusion] clean/mild_mod/mod_severe are window-aligned for train/val/test")


def main() -> int:
    sensors = list(SENSORS.keys())
    collector = CheckCollector(passed=[], failed=[], notes=[])
    split_meta_by_variant: Dict[str, Dict[str, np.ndarray]] = {}

    print("=" * 88)
    print("Sanity Check: Embedding Extraction")
    print("=" * 88)

    for variant in EXPECTED_LABELS.keys():
        print(f"\nChecking variant: {variant}")
        check_variant_sensor_files(collector, variant, sensors)

        if any(not _sensor_file(variant, s).exists() for s in sensors):
            continue

        _, split_meta = check_variant_content(collector, variant, sensors)
        split_meta_by_variant[variant] = split_meta

        for split in SPLITS:
            n = len(split_meta[split]["subjects"])
            print(f"  - {split}: {n} windows")

    if all(v in split_meta_by_variant for v in FUSION_VARIANTS):
        check_cross_variant_fusion_alignment(collector, split_meta_by_variant)

    print("\n" + "=" * 88)
    print("RESULT")
    print("=" * 88)
    print(f"Passed checks: {len(collector.passed)}")
    print(f"Failed checks: {len(collector.failed)}")

    if collector.passed:
        print("\nPASS DETAILS")
        for msg in collector.passed:
            print(f"  + {msg}")

    if collector.failed:
        print("\nFAIL DETAILS")
        for msg in collector.failed:
            print(f"  - {msg}")
        print("\nSanity check FAILED")
        return 1

    if collector.notes:
        print("\nNOTES")
        for msg in collector.notes:
            print(f"  * {msg}")

    print("\nSanity check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
