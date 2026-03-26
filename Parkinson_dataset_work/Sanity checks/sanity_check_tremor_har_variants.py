#!/usr/bin/env python3
"""
Sanity checks for generated tremor/HAR dataset variants.

Checks performed per variant folder:
- Folder and metadata file presence
- NPZ/TXT file discovery and pairing
- tremor_score validity per variant
- NPZ shape/length consistency across key arrays
- NPZ <-> TXT consistency (rows, label, tremor_score)
- Label distribution overview
- base_idx/base_window_idx sanity (min/max/negative/NaN/unique)
- subject_id sanity overview
- window_source_map source summary
- Parkinson-specific checks:
  * only source == pd in window_source_map
  * tremor_freq/tremor_acc_rms/tremor_gyro_rms are all zero
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_VARIANT_DIRS: Dict[str, Path] = {
    "parkinson": Path(
        "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Data/Tremor_datagenerator_files/s2_w2_fs50_tremor_parkinson"
    ),
    "mod_severe": Path(
        "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Data/Tremor_datagenerator_files/s2_w2_fs50_tremor_mod_severe"
    ),
    "mild_mod": Path(
        "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Data/Tremor_datagenerator_files/s2_w2_fs50_tremor_mild_mod"
    ),
    "clean": Path(
        "/Volumes/NO NAME/Master Lina/Code/Parkinson_dataset_work/Data/Tremor_datagenerator_files/s2_w2_fs50_tremor_clean"
    ),
}

EXPECTED_TREMOR_SCORES: Dict[str, set] = {
    "clean": {0},
    "mild_mod": {1, 2},
    "mod_severe": {3, 4},
    "parkinson": {5},
}

METADATA_FILES = ["info.txt", "window_source_map.jsonl", "sensor_coverage.txt"]
EPS = 1e-8


@dataclass
class Totals:
    passed: int = 0
    failed: int = 0
    warnings: int = 0

    def ok(self, message: str) -> None:
        self.passed += 1
        print(f"[PASS] {message}")

    def fail(self, message: str) -> None:
        self.failed += 1
        print(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")


@dataclass
class NpzSnapshot:
    sensor_name: str
    n_samples: int
    y: Optional[np.ndarray]
    tremor_score: Optional[np.ndarray]
    base_idx: Optional[np.ndarray]
    subject_id: Optional[np.ndarray]


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def to_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def get_npz_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    kset = set(keys)
    for c in candidates:
        if c in kset:
            return c
    return None


def safe_load_txt(txt_path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        arr = np.loadtxt(txt_path, delimiter=",")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32), None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def parse_window_source_map(path: Path) -> Tuple[Dict[int, str], Counter, List[str]]:
    idx_to_source: Dict[int, str] = {}
    source_counts: Counter = Counter()
    parse_errors: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                idx = int(rec["base_window_idx"])
                src = str(rec["source"])
            except Exception as exc:  # pragma: no cover
                parse_errors.append(f"line {line_no}: {exc}")
                continue
            idx_to_source[idx] = src
            source_counts[src] += 1

    return idx_to_source, source_counts, parse_errors


def check_file_presence(variant_name: str, variant_dir: Path, totals: Totals) -> Tuple[List[Path], List[Path]]:
    print_header(f"Variant: {variant_name}")

    if not variant_dir.exists() or not variant_dir.is_dir():
        totals.fail(f"Variant folder missing: {variant_dir}")
        return [], []

    totals.ok(f"Variant folder exists: {variant_dir}")

    for meta_name in METADATA_FILES:
        p = variant_dir / meta_name
        if p.exists():
            totals.ok(f"Metadata file exists: {meta_name}")
        else:
            totals.fail(f"Missing metadata file: {meta_name}")

    npz_files = sorted([p for p in variant_dir.glob("*.npz") if not p.name.startswith("._")])
    txt_files = sorted(
        [
            p
            for p in variant_dir.glob("*.txt")
            if not p.name.startswith("._") and p.name not in set(METADATA_FILES)
        ]
    )

    if npz_files:
        totals.ok(f"Found {len(npz_files)} NPZ files")
    else:
        totals.fail("No NPZ files found")

    if txt_files:
        totals.ok(f"Found {len(txt_files)} sensor TXT files")
    else:
        totals.fail("No sensor TXT files found")

    print("NPZ files:", ", ".join(p.name for p in npz_files) if npz_files else "none")
    print("TXT files:", ", ".join(p.name for p in txt_files) if txt_files else "none")

    return npz_files, txt_files


def check_npz_shape_and_values(
    variant_name: str,
    npz_path: Path,
    expected_scores: set,
    totals: Totals,
) -> Optional[NpzSnapshot]:
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            keys = list(d.files)

            n_ref: Optional[int] = None
            if "X" in d:
                x = np.asarray(d["X"])
                if x.ndim >= 1:
                    n_ref = int(x.shape[0])
                else:
                    totals.fail(f"{npz_path.name}: X has invalid ndim={x.ndim}")

            check_keys = [
                "y",
                "tremor_score",
                "subject_id",
                "base_window_idx",
                "base_idx",
                "tremor_freq",
                "tremor_acc_rms",
                "tremor_gyro_rms",
            ]

            first_vector_key = None
            for k in check_keys:
                if k in d:
                    arr = np.asarray(d[k])
                    if arr.ndim == 0:
                        continue
                    first_vector_key = k
                    if n_ref is None:
                        n_ref = int(arr.shape[0])
                    break

            if n_ref is None:
                totals.fail(f"{npz_path.name}: could not infer sample dimension")
                return None

            for k in check_keys:
                if k not in d:
                    continue
                arr = np.asarray(d[k])
                if arr.ndim == 0:
                    totals.warn(f"{npz_path.name}: key {k} is scalar, skipped shape check")
                    continue
                if int(arr.shape[0]) != n_ref:
                    totals.fail(
                        f"{npz_path.name}: length mismatch for {k} ({arr.shape[0]} != {n_ref})"
                    )

            y_key = get_npz_key(keys, ["y"])
            score_key = get_npz_key(keys, ["tremor_score"])
            base_key = get_npz_key(keys, ["base_window_idx", "base_idx"])
            subj_key = get_npz_key(keys, ["subject_id"])

            y = to_1d(np.asarray(d[y_key])) if y_key else None
            tremor_score = to_1d(np.asarray(d[score_key])) if score_key else None
            base_idx = to_1d(np.asarray(d[base_key])) if base_key else None
            subject_id = to_1d(np.asarray(d[subj_key])) if subj_key else None

            if tremor_score is None:
                totals.fail(f"{npz_path.name}: missing tremor_score")
            else:
                unique_scores = np.unique(tremor_score)
                invalid_mask = ~np.isin(tremor_score, list(expected_scores))
                invalid_count = int(np.count_nonzero(invalid_mask))
                if invalid_count == 0:
                    totals.ok(
                        f"{npz_path.name}: tremor_score valid, unique={unique_scores.tolist()}"
                    )
                else:
                    invalid_unique = np.unique(tremor_score[invalid_mask]).tolist()
                    totals.fail(
                        f"{npz_path.name}: invalid tremor_score values {invalid_unique}, rows={invalid_count}"
                    )

            if y is not None:
                unique_labels = np.unique(y)
                label_min = int(np.min(unique_labels))
                label_max = int(np.max(unique_labels))
                label_counts = Counter(int(v) for v in y.tolist())
                totals.ok(
                    f"{npz_path.name}: labels unique={unique_labels.tolist()} range=[{label_min}, {label_max}]"
                )
                print(
                    f"[INFO] {npz_path.name}: label counts "
                    + ", ".join(f"{k}:{v}" for k, v in sorted(label_counts.items()))
                )
            else:
                totals.warn(f"{npz_path.name}: missing y")

            if base_idx is not None:
                if np.issubdtype(base_idx.dtype, np.floating):
                    nan_count = int(np.count_nonzero(np.isnan(base_idx)))
                else:
                    nan_count = 0
                neg_count = int(np.count_nonzero(base_idx < 0))
                unique_base = int(np.unique(base_idx).size)
                base_min = float(np.nanmin(base_idx)) if base_idx.size else float("nan")
                base_max = float(np.nanmax(base_idx)) if base_idx.size else float("nan")

                if nan_count == 0 and neg_count == 0:
                    totals.ok(
                        f"{npz_path.name}: base_idx sanity ok min={base_min} max={base_max} unique={unique_base}"
                    )
                else:
                    totals.fail(
                        f"{npz_path.name}: base_idx issues nan={nan_count}, negative={neg_count}"
                    )
            else:
                totals.warn(f"{npz_path.name}: missing base_window_idx/base_idx")

            if subject_id is not None:
                if np.issubdtype(subject_id.dtype, np.floating):
                    subj_nan = int(np.count_nonzero(np.isnan(subject_id)))
                else:
                    subj_nan = 0
                subj_neg = int(np.count_nonzero(subject_id < 0))
                subj_unique = int(np.unique(subject_id).size)
                subj_min = float(np.nanmin(subject_id)) if subject_id.size else float("nan")
                subj_max = float(np.nanmax(subject_id)) if subject_id.size else float("nan")

                if subj_nan == 0 and subj_neg == 0:
                    totals.ok(
                        f"{npz_path.name}: subject_id sanity ok min={subj_min} max={subj_max} unique={subj_unique}"
                    )
                else:
                    totals.fail(
                        f"{npz_path.name}: subject_id issues nan={subj_nan}, negative={subj_neg}"
                    )
            else:
                totals.warn(f"{npz_path.name}: missing subject_id")

            # Parkinson metadata constraints
            if variant_name == "parkinson":
                for k in ["tremor_freq", "tremor_acc_rms", "tremor_gyro_rms"]:
                    if k not in d:
                        totals.fail(f"{npz_path.name}: missing {k} in parkinson variant")
                        continue
                    arr = to_1d(np.asarray(d[k]).astype(np.float64))
                    nz = int(np.count_nonzero(np.abs(arr) > EPS))
                    if nz == 0:
                        totals.ok(f"{npz_path.name}: {k} all zeros as expected")
                    else:
                        totals.fail(f"{npz_path.name}: {k} has {nz} non-zero entries")

            sensor_name = npz_path.stem
            return NpzSnapshot(
                sensor_name=sensor_name,
                n_samples=n_ref,
                y=y,
                tremor_score=tremor_score,
                base_idx=base_idx,
                subject_id=subject_id,
            )
    except Exception as exc:  # pragma: no cover
        totals.fail(f"{npz_path.name}: failed to read NPZ ({exc})")
        return None


def check_npz_txt_consistency(npz_path: Path, snap: NpzSnapshot, totals: Totals) -> None:
    txt_path = npz_path.with_suffix(".txt")
    if not txt_path.exists():
        totals.fail(f"{npz_path.name}: missing matching TXT file {txt_path.name}")
        return

    txt_arr, txt_err = safe_load_txt(txt_path)
    if txt_err is not None or txt_arr is None:
        totals.fail(f"{txt_path.name}: failed to parse TXT ({txt_err})")
        return

    if txt_arr.shape[0] == snap.n_samples:
        totals.ok(f"{npz_path.name}: TXT rows match NPZ samples ({snap.n_samples})")
    else:
        totals.fail(
            f"{npz_path.name}: TXT row mismatch ({txt_arr.shape[0]} != {snap.n_samples})"
        )

    if txt_arr.shape[1] < 7:
        totals.fail(f"{txt_path.name}: expected at least 7 trailing label columns")
        return

    txt_label_1idx = txt_arr[:, -7]
    txt_tremor_score = txt_arr[:, -1]

    if snap.y is not None and snap.y.size == txt_label_1idx.size:
        y_from_txt = np.rint(txt_label_1idx).astype(np.int64) - 1
        mismatch = int(np.count_nonzero(y_from_txt != snap.y.astype(np.int64)))
        if mismatch == 0:
            totals.ok(f"{npz_path.name}: TXT activity labels match NPZ y")
        else:
            totals.fail(f"{npz_path.name}: TXT activity label mismatch rows={mismatch}")

    if snap.tremor_score is not None and snap.tremor_score.size == txt_tremor_score.size:
        txt_score_i = np.rint(txt_tremor_score).astype(np.int64)
        npz_score_i = np.rint(snap.tremor_score).astype(np.int64)
        mismatch = int(np.count_nonzero(txt_score_i != npz_score_i))
        if mismatch == 0:
            totals.ok(f"{npz_path.name}: TXT tremor_score matches NPZ")
        else:
            totals.fail(f"{npz_path.name}: TXT tremor_score mismatch rows={mismatch}")


def check_window_source_map(
    variant_name: str,
    variant_dir: Path,
    snapshots: List[NpzSnapshot],
    totals: Totals,
) -> None:
    source_map_path = variant_dir / "window_source_map.jsonl"
    if not source_map_path.exists():
        totals.fail("window_source_map.jsonl missing")
        return

    try:
        idx_to_source, source_counts, parse_errors = parse_window_source_map(source_map_path)
    except Exception as exc:  # pragma: no cover
        totals.fail(f"Failed to parse window_source_map.jsonl: {exc}")
        return

    if parse_errors:
        totals.fail(
            f"window_source_map.jsonl parse errors: {len(parse_errors)} (first: {parse_errors[0]})"
        )
    else:
        totals.ok("window_source_map.jsonl parsed successfully")

    if source_counts:
        print(
            "[INFO] source counts: "
            + ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
        )
    else:
        totals.fail("window_source_map.jsonl contains no valid rows")

    if variant_name == "parkinson":
        bad_sources = sorted([s for s in source_counts.keys() if s != "pd"])
        if not bad_sources and source_counts.get("pd", 0) > 0:
            totals.ok("parkinson source check: only source=pd")
        else:
            totals.fail(f"parkinson source check failed: found sources {sorted(source_counts.keys())}")
    else:
        totals.ok(
            f"non-parkinson source overview: {', '.join(f'{k}:{v}' for k, v in sorted(source_counts.items()))}"
        )

    # Compare unique base_idx in one representative NPZ against source map size.
    rep = next((s for s in snapshots if s.base_idx is not None), None)
    if rep is None:
        totals.warn("No NPZ with base_idx available for source-map consistency check")
        return

    base_unique = np.unique(rep.base_idx.astype(np.int64))
    map_unique = np.array(sorted(idx_to_source.keys()), dtype=np.int64)

    if base_unique.size == 0 or map_unique.size == 0:
        totals.warn("base_idx/source-map consistency skipped due to empty indices")
        return

    missing_in_map = np.setdiff1d(base_unique, map_unique)
    missing_in_npz = np.setdiff1d(map_unique, base_unique)

    if missing_in_map.size == 0:
        totals.ok("All base_idx values in representative NPZ exist in source map")
    else:
        totals.fail(
            f"base_idx values missing in source map: count={missing_in_map.size}, sample={missing_in_map[:10].tolist()}"
        )

    if missing_in_npz.size == 0:
        totals.ok("Source map indices are fully represented in representative NPZ")
    else:
        totals.warn(
            f"source-map indices not present in representative NPZ: count={missing_in_npz.size}, sample={missing_in_npz[:10].tolist()}"
        )


def summarize_variant_labels(snapshots: List[NpzSnapshot], totals: Totals) -> None:
    # Prefer one representative file to avoid counting same samples multiple times across sensors.
    rep = next((s for s in snapshots if s.y is not None), None)
    if rep is None:
        totals.warn("No y array found in NPZ files; label summary skipped")
        return

    counts = Counter(int(v) for v in rep.y.tolist())
    labels = sorted(counts.keys())
    if labels:
        totals.ok(f"Representative label range: [{labels[0]}, {labels[-1]}], unique={labels}")
        print("[INFO] Representative label counts:")
        for lab in labels:
            print(f"       label {lab}: {counts[lab]}")


def check_variant(variant_name: str, variant_dir: Path, totals: Totals) -> None:
    expected_scores = EXPECTED_TREMOR_SCORES[variant_name]
    npz_files, _ = check_file_presence(variant_name, variant_dir, totals)
    if not npz_files:
        return

    snapshots: List[NpzSnapshot] = []

    for npz_path in npz_files:
        snap = check_npz_shape_and_values(
            variant_name=variant_name,
            npz_path=npz_path,
            expected_scores=expected_scores,
            totals=totals,
        )
        if snap is None:
            continue
        snapshots.append(snap)
        check_npz_txt_consistency(npz_path=npz_path, snap=snap, totals=totals)

    summarize_variant_labels(snapshots, totals)
    check_window_source_map(
        variant_name=variant_name,
        variant_dir=variant_dir,
        snapshots=snapshots,
        totals=totals,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check tremor/HAR generated dataset variant folders",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero exit code if warnings are present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    totals = Totals()

    print_header("TREMOR/HAR DATASET SANITY CHECK")

    for variant_name, variant_dir in DEFAULT_VARIANT_DIRS.items():
        check_variant(variant_name=variant_name, variant_dir=variant_dir, totals=totals)

    print_header("SUMMARY")
    print(f"PASS: {totals.passed}")
    print(f"WARN: {totals.warnings}")
    print(f"FAIL: {totals.failed}")

    if totals.failed > 0:
        raise SystemExit(1)
    if args.fail_on_warning and totals.warnings > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
