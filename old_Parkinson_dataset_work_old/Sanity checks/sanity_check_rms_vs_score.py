#!/usr/bin/env python3
"""
Sanity check for consistency between tremor_acc_rms and tremor_score.

Ground-truth mapping is computed with get_tremor_score(rms_acc) from
Noise_simulation/tremor_parkinson_config.py.

Sentinel score 5 (real PD windows) is excluded from RMS-vs-score comparison
and reported separately.

Comparison modes:
- direct: compare stored tremor_acc_rms directly against stored tremor_score.
- restored: back-calculate a resting arm-equivalent RMS before recomputing score
  by reversing activity modulation (beta), body-part scaling (C), and ankle
  severity ratio when available.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

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

DEFAULT_CONFIG_PATH = Path(
    "/Volumes/NO NAME/Master Lina/Code/Noise_simulation/tremor_parkinson_config.py"
)

SENTINEL_SCORE = 5

EXPECTED_STRICT_SCORE_SET = {
    "parkinson": {5},
    "clean": {0},
    "mild_mod": {1, 2},
    "mod_severe": {3, 4},
}

# For Parkinson-specific activity labels introduced in the merged generator,
# reuse the same beta-reference mapping used during tremor-cache creation.
PARKINSON_ACTIVITY_BETA_REFERENCE = {
    13: 1,
    14: 7,
    15: 7,
    16: 7,
}


@dataclass
class FileResult:
    file_name: str
    total_samples: int = 0
    checked_samples: int = 0
    sentinel_samples: int = 0
    skipped_tremor_free: int = 0
    skipped_missing_sensor: int = 0
    skipped_unresolved: int = 0
    correct: int = 0
    mismatches: int = 0
    unique_scores: List[int] = field(default_factory=list)
    score_counts: Counter = field(default_factory=Counter)
    mismatch_examples: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    failed: bool = False
    rms_values_by_score: Dict[int, List[float]] = field(default_factory=dict)
    restored_rms_values_by_score: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class VariantResult:
    variant_name: str
    file_results: List[FileResult] = field(default_factory=list)
    total_samples: int = 0
    checked_samples: int = 0
    sentinel_samples: int = 0
    skipped_tremor_free: int = 0
    skipped_missing_sensor: int = 0
    skipped_unresolved: int = 0
    correct: int = 0
    mismatches: int = 0
    score_counts: Counter = field(default_factory=Counter)
    rms_values_by_score: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    restored_rms_values_by_score: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    failed: bool = False
    messages: List[str] = field(default_factory=list)


def print_separator() -> None:
    print("=" * 80)


def format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    ordered = sorted(counter.items(), key=lambda x: x[0])
    return "{" + ", ".join(f"{k}: {v}" for k, v in ordered) + "}"


def safe_npz_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.glob("*.npz") if p.is_file() and not p.name.startswith("._")
    )


def load_config_module(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("tremor_parkinson_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_tremor_score"):
        raise AttributeError("Config module does not define get_tremor_score")

    return module


def compute_rms_stats(values: List[float]) -> str:
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={arr.size}, min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, median={np.median(arr):.6f}"
    )


def infer_body_part(sensor_name: str) -> Optional[str]:
    if "_arm" in sensor_name:
        return "arm"
    if "_ankle" in sensor_name:
        return "ankle"
    if "_chest" in sensor_name or sensor_name == "ECG":
        return "chest"
    return None


def infer_y_offset(y_values: np.ndarray, beta_activity: Dict[int, float]) -> int:
    if y_values.size == 0:
        return 0

    y_int = np.asarray(y_values, dtype=np.int64).reshape(-1)
    if np.any(y_int == 0):
        return 1

    beta_keys = set(int(k) for k in beta_activity.keys())
    mapped0 = sum(1 for v in y_int if int(v) in beta_keys)
    mapped1 = sum(1 for v in y_int if int(v) + 1 in beta_keys)

    if mapped1 > mapped0:
        return 1
    return 0


def resolve_activity_label(y_value: int, y_offset: int) -> int:
    return int(y_value) + int(y_offset)


def resolve_activity_beta(activity_label: int, beta_activity: Dict[int, float]) -> float:
    if activity_label in beta_activity:
        return float(beta_activity[activity_label])

    ref_label = PARKINSON_ACTIVITY_BETA_REFERENCE.get(activity_label)
    if ref_label is not None and ref_label in beta_activity:
        return float(beta_activity[ref_label])

    return 1.0


def recover_rest_rms(
    rms_acc: float,
    stored_score: int,
    body_part: Optional[str],
    y_value: Optional[int],
    y_offset: int,
    beta_activity: Dict[int, float],
    c_body_part: Dict[str, float],
    ankle_ratio_by_score: Dict[int, float],
) -> tuple[Optional[float], Optional[str]]:
    if stored_score == 0:
        return float(rms_acc), None

    if y_value is None:
        beta = 1.0
    else:
        activity_label = resolve_activity_label(y_value, y_offset)
        beta = resolve_activity_beta(activity_label, beta_activity)

    if beta <= 0.0:
        return None, f"invalid beta={beta}"

    scale = beta

    if body_part == "ankle":
        ratio = ankle_ratio_by_score.get(stored_score)
        if ratio is None:
            ratio = c_body_part.get("ankle", 1.0)
        ratio = float(ratio)
        if ratio <= 0.0:
            return None, f"invalid ankle scale={ratio} for score={stored_score}"
        scale *= ratio
    elif body_part in c_body_part:
        c_val = float(c_body_part.get(body_part, 1.0))
        if c_val <= 0.0:
            return None, f"invalid body-part scale={c_val} for body_part={body_part}"
        scale *= c_val

    if scale <= 0.0:
        return None, f"invalid total scale={scale}"

    return float(rms_acc / scale), None


def apply_boundary_tolerance(
    rms_value: float,
    stored_score: int,
    score_rms_range: Dict[int, tuple[float, float]],
    tol: float = 1e-5,
) -> float:
    interval = score_rms_range.get(int(stored_score))
    if interval is None:
        return float(rms_value)

    lower = float(interval[0])
    if rms_value < lower and (lower - rms_value) <= tol:
        return lower

    return float(rms_value)


def evaluate_npz_file(
    npz_path: Path,
    get_tremor_score: Callable[[float], int],
    max_examples: int,
    score_basis: str,
    beta_activity: Dict[int, float],
    c_body_part: Dict[str, float],
    ankle_ratio_by_score: Dict[int, float],
    tremor_free_sensors: Set[str],
    skip_tremor_free_sensors: bool,
    score_rms_range: Dict[int, tuple[float, float]],
) -> FileResult:
    result = FileResult(file_name=npz_path.name)
    sensor_name = npz_path.stem
    body_part = infer_body_part(sensor_name)
    is_tremor_free = sensor_name in tremor_free_sensors

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            has_score = "tremor_score" in data
            has_rms = "tremor_acc_rms" in data
            has_y = "y" in data
            has_x = "X" in data

            if not has_score:
                result.failed = True
                result.messages.append("Missing key: tremor_score")
            if not has_rms:
                result.failed = True
                result.messages.append("Missing key: tremor_acc_rms")

            scores = None
            rms_vals = None
            y_vals = None
            x_vals = None

            if has_score:
                scores = np.asarray(data["tremor_score"]).reshape(-1).astype(np.int64, copy=False)
                result.total_samples = int(scores.size)
                result.unique_scores = np.unique(scores).astype(int).tolist()
                result.score_counts = Counter(scores.tolist())
                result.sentinel_samples = int(np.count_nonzero(scores == SENTINEL_SCORE))
            if has_rms:
                rms_vals = np.asarray(data["tremor_acc_rms"]).reshape(-1).astype(np.float64, copy=False)
                if scores is None:
                    result.total_samples = int(rms_vals.size)
            if has_y:
                y_vals = np.asarray(data["y"]).reshape(-1).astype(np.int64, copy=False)
            if has_x:
                x_vals = np.asarray(data["X"])

            if scores is None or rms_vals is None:
                return result

            if scores.size != rms_vals.size:
                result.failed = True
                result.messages.append(
                    f"Length mismatch: tremor_score={scores.size}, tremor_acc_rms={rms_vals.size}"
                )

            paired_len = int(min(scores.size, rms_vals.size))
            scores_p = scores[:paired_len]
            rms_p = rms_vals[:paired_len]
            y_p = y_vals[:paired_len] if y_vals is not None else None
            x_p = x_vals[:paired_len] if x_vals is not None and x_vals.ndim >= 3 else None

            zero_window_mask = None
            if x_p is not None and x_p.shape[0] == paired_len:
                zero_window_mask = np.all(np.isclose(x_p, 0.0, atol=1e-8), axis=(1, 2))

            for score in np.unique(scores_p):
                vals = rms_p[scores_p == score]
                finite_vals = vals[np.isfinite(vals)]
                result.rms_values_by_score[int(score)] = finite_vals.astype(np.float64).tolist()

            non_sentinel_mask = scores_p != SENTINEL_SCORE
            idx_candidates = np.where(non_sentinel_mask)[0]

            if idx_candidates.size == 0:
                return result

            if is_tremor_free and skip_tremor_free_sensors:
                result.skipped_tremor_free = int(idx_candidates.size)
                result.messages.append(
                    "Skipped RMS-vs-score checks for tremor-free sensor by policy"
                )
                return result

            y_offset = infer_y_offset(y_p, beta_activity) if y_p is not None else 0
            if y_p is None and score_basis == "restored":
                result.messages.append(
                    "Missing key: y; using beta=1.0 fallback in restored mode"
                )

            unresolved_reasons: Counter = Counter()
            checked = 0
            correct = 0
            mismatches = 0

            for global_idx in idx_candidates:
                stored_score = int(scores_p[global_idx])
                rms_raw = float(rms_p[global_idx])

                if (
                    zero_window_mask is not None
                    and bool(zero_window_mask[global_idx])
                    and stored_score > 0
                ):
                    result.skipped_missing_sensor += 1
                    continue

                if not np.isfinite(rms_raw):
                    unresolved_reasons["non-finite RMS"] += 1
                    continue

                if score_basis == "restored":
                    y_value = int(y_p[global_idx]) if y_p is not None else None
                    rms_for_score, reason = recover_rest_rms(
                        rms_acc=rms_raw,
                        stored_score=stored_score,
                        body_part=body_part,
                        y_value=y_value,
                        y_offset=y_offset,
                        beta_activity=beta_activity,
                        c_body_part=c_body_part,
                        ankle_ratio_by_score=ankle_ratio_by_score,
                    )
                    if reason is not None or rms_for_score is None:
                        unresolved_reasons[reason or "unresolved"] += 1
                        continue
                    rms_for_score = apply_boundary_tolerance(
                        float(rms_for_score),
                        stored_score,
                        score_rms_range,
                    )
                else:
                    rms_for_score = rms_raw

                predicted_score = int(get_tremor_score(float(rms_for_score)))

                checked += 1
                result.restored_rms_values_by_score.setdefault(stored_score, []).append(
                    float(rms_for_score)
                )

                if predicted_score == stored_score:
                    correct += 1
                else:
                    mismatches += 1
                    if len(result.mismatch_examples) < max_examples:
                        result.mismatch_examples.append(
                            "idx="
                            f"{int(global_idx)}, "
                            f"rms_raw={rms_raw:.6f}, "
                            f"rms_check={float(rms_for_score):.6f}, "
                            f"stored={stored_score}, "
                            f"recomputed={predicted_score}"
                        )

            result.checked_samples = int(checked)
            result.correct = int(correct)
            result.mismatches = int(mismatches)
            result.skipped_unresolved = int(idx_candidates.size - checked)
            result.skipped_unresolved -= int(result.skipped_missing_sensor)
            if result.skipped_unresolved < 0:
                result.skipped_unresolved = 0

            if result.mismatches > 0:
                result.failed = True

            if unresolved_reasons:
                result.failed = True
                reason_str = ", ".join(
                    f"{k}: {v}" for k, v in sorted(unresolved_reasons.items())
                )
                result.messages.append(
                    f"Skipped unresolved samples in check: {result.skipped_unresolved} ({reason_str})"
                )

            if result.skipped_missing_sensor > 0:
                result.messages.append(
                    f"Skipped missing-sensor placeholder samples: {result.skipped_missing_sensor}"
                )

    except Exception as exc:
        result.failed = True
        result.messages.append(f"Failed to read NPZ: {exc}")

    return result


def print_file_report(file_result: FileResult, score_basis: str) -> None:
    if file_result.checked_samples > 0:
        match_rate = 100.0 * file_result.correct / file_result.checked_samples
    else:
        match_rate = 100.0

    status = "PASS" if not file_result.failed else "FAIL"
    print(
        f"[{status}] {file_result.file_name}: "
        f"total={file_result.total_samples}, checked={file_result.checked_samples}, "
        f"sentinel={file_result.sentinel_samples}, "
        f"skipped_tremor_free={file_result.skipped_tremor_free}, "
        f"skipped_missing_sensor={file_result.skipped_missing_sensor}, "
        f"skipped_unresolved={file_result.skipped_unresolved}, "
        f"correct={file_result.correct}, mismatches={file_result.mismatches}, "
        f"match_rate={match_rate:.2f}%"
    )

    if file_result.unique_scores:
        print(f"       Unique scores: {file_result.unique_scores}")
    else:
        print("       Unique scores: none")

    print(f"       Score distribution: {format_counter(file_result.score_counts)}")

    if file_result.rms_values_by_score:
        print("       RMS stats per score (stored tremor_acc_rms):")
        for score in sorted(file_result.rms_values_by_score):
            stats = compute_rms_stats(file_result.rms_values_by_score[score])
            print(f"         score {score}: {stats}")

    if score_basis == "restored" and file_result.restored_rms_values_by_score:
        print("       RMS stats per score (restored RMS used for score check):")
        for score in sorted(file_result.restored_rms_values_by_score):
            stats = compute_rms_stats(file_result.restored_rms_values_by_score[score])
            print(f"         score {score}: {stats}")

    for msg in file_result.messages:
        print(f"       Note: {msg}")

    if file_result.mismatch_examples:
        print("       Example mismatches:")
        for ex in file_result.mismatch_examples:
            print(f"         {ex}")


def evaluate_variant(
    variant_name: str,
    variant_dir: Path,
    get_tremor_score: Callable[[float], int],
    max_examples: int,
    score_basis: str,
    beta_activity: Dict[int, float],
    c_body_part: Dict[str, float],
    ankle_ratio_by_score: Dict[int, float],
    tremor_free_sensors: Set[str],
    skip_tremor_free_sensors: bool,
    score_rms_range: Dict[int, tuple[float, float]],
) -> VariantResult:
    result = VariantResult(variant_name=variant_name)

    print_separator()
    print(f"Variant: {variant_name}")
    print(f"Path: {variant_dir}")

    if not variant_dir.exists() or not variant_dir.is_dir():
        result.failed = True
        result.messages.append("Variant folder missing")
        print(f"[FAIL] Folder missing: {variant_dir}")
        return result

    npz_files = safe_npz_files(variant_dir)
    if not npz_files:
        result.failed = True
        result.messages.append("No NPZ files found")
        print("[FAIL] No NPZ files found")
        return result

    print(f"Found {len(npz_files)} NPZ files")

    for npz_path in npz_files:
        file_result = evaluate_npz_file(
            npz_path=npz_path,
            get_tremor_score=get_tremor_score,
            max_examples=max_examples,
            score_basis=score_basis,
            beta_activity=beta_activity,
            c_body_part=c_body_part,
            ankle_ratio_by_score=ankle_ratio_by_score,
            tremor_free_sensors=tremor_free_sensors,
            skip_tremor_free_sensors=skip_tremor_free_sensors,
            score_rms_range=score_rms_range,
        )
        result.file_results.append(file_result)
        print_file_report(file_result, score_basis=score_basis)

        result.total_samples += file_result.total_samples
        result.checked_samples += file_result.checked_samples
        result.sentinel_samples += file_result.sentinel_samples
        result.skipped_tremor_free += file_result.skipped_tremor_free
        result.skipped_missing_sensor += file_result.skipped_missing_sensor
        result.skipped_unresolved += file_result.skipped_unresolved
        result.correct += file_result.correct
        result.mismatches += file_result.mismatches
        result.score_counts.update(file_result.score_counts)

        for score, vals in file_result.rms_values_by_score.items():
            result.rms_values_by_score[score].extend(vals)
        for score, vals in file_result.restored_rms_values_by_score.items():
            result.restored_rms_values_by_score[score].extend(vals)

        if file_result.failed:
            result.failed = True

    expected_set = EXPECTED_STRICT_SCORE_SET.get(variant_name)
    if expected_set is not None and result.score_counts:
        seen_scores = set(result.score_counts.keys())
        if variant_name == "parkinson":
            unexpected = sorted(seen_scores - {SENTINEL_SCORE})
            if unexpected:
                result.failed = True
                result.messages.append(
                    f"Parkinson variant contains non-sentinel scores: {unexpected}"
                )
        elif variant_name == "clean":
            unexpected = sorted(seen_scores - {0})
            if unexpected:
                result.failed = True
                result.messages.append(
                    f"Clean variant contains non-zero scores: {unexpected}"
                )
        else:
            unexpected_non_sentinel = sorted(seen_scores - expected_set - {SENTINEL_SCORE})
            if unexpected_non_sentinel:
                result.messages.append(
                    f"Non-expected non-sentinel scores observed: {unexpected_non_sentinel}"
                )

    print("Variant summary:")
    if result.checked_samples > 0:
        accuracy = 100.0 * result.correct / result.checked_samples
    else:
        accuracy = 100.0

    status = "PASS" if not result.failed else "FAIL"
    print(
        f"[{status}] {variant_name}: match_rate={accuracy:.2f}%, "
        f"total={result.total_samples}, checked={result.checked_samples}, "
        f"sentinel={result.sentinel_samples}, skipped_tremor_free={result.skipped_tremor_free}, "
        f"skipped_missing_sensor={result.skipped_missing_sensor}, "
        f"skipped_unresolved={result.skipped_unresolved}, mismatches={result.mismatches}"
    )

    print(f"Score distribution: {format_counter(result.score_counts)}")
    if result.rms_values_by_score:
        print("RMS stats per score (variant total, stored tremor_acc_rms):")
        for score in sorted(result.rms_values_by_score):
            stats = compute_rms_stats(result.rms_values_by_score[score])
            print(f"  score {score}: {stats}")

    if score_basis == "restored" and result.restored_rms_values_by_score:
        print("RMS stats per score (variant total, restored RMS for score check):")
        for score in sorted(result.restored_rms_values_by_score):
            stats = compute_rms_stats(result.restored_rms_values_by_score[score])
            print(f"  score {score}: {stats}")

    if variant_name == "parkinson":
        if result.checked_samples == 0 and not result.failed:
            print("Parkinson check: all samples are sentinel score 5 as expected.")
        elif result.checked_samples > 0:
            print("Parkinson check: non-sentinel samples found (unexpected).")

    for msg in result.messages:
        print(f"Note: {msg}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check tremor_acc_rms vs tremor_score using "
            "get_tremor_score(rms_acc) from tremor_parkinson_config.py"
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to tremor_parkinson_config.py",
    )
    parser.add_argument(
        "--parkinson-dir",
        type=Path,
        default=DEFAULT_VARIANT_DIRS["parkinson"],
        help="Path to parkinson variant folder",
    )
    parser.add_argument(
        "--mod-severe-dir",
        type=Path,
        default=DEFAULT_VARIANT_DIRS["mod_severe"],
        help="Path to mod_severe variant folder",
    )
    parser.add_argument(
        "--mild-mod-dir",
        type=Path,
        default=DEFAULT_VARIANT_DIRS["mild_mod"],
        help="Path to mild_mod variant folder",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_VARIANT_DIRS["clean"],
        help="Path to clean variant folder",
    )
    parser.add_argument(
        "--score-basis",
        choices=["direct", "restored"],
        default="restored",
        help=(
            "Score-check RMS basis: 'direct' uses stored tremor_acc_rms directly; "
            "'restored' reverses beta/C/ankle scaling before recomputing score."
        ),
    )
    parser.add_argument(
        "--include-tremor-free-check",
        action="store_true",
        help=(
            "Also run RMS-vs-score checks on tremor-free sensors listed in config "
            "(default behavior is to skip those checks)."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Number of mismatch examples to print per file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_separator()
    print("RMS vs SCORE SANITY CHECK")
    print_separator()

    try:
        config_module = load_config_module(args.config_path)
    except Exception as exc:
        print(f"[FAIL] Could not load config module: {exc}")
        return 2

    get_tremor_score = config_module.get_tremor_score
    score_rms_range = getattr(config_module, "SCORE_RMS_RANGE", None)
    if score_rms_range is None:
        score_rms_range = {}
    else:
        score_rms_range = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in score_rms_range.items()
        }
    beta_activity = {
        int(k): float(v)
        for k, v in getattr(config_module, "BETA_ACTIVITY", {}).items()
    }
    c_body_part = {
        str(k): float(v)
        for k, v in getattr(config_module, "C_BODY_PART", {}).items()
    }
    ankle_ratio_by_score = {
        int(k): float(v)
        for k, v in getattr(config_module, "ANKLE_RATIO_BY_SCORE", {}).items()
    }
    tremor_free_sensors = set(getattr(config_module, "TREMOR_FREE_SENSORS", []))

    print(f"Config: {args.config_path}")
    print(f"Score basis for check: {args.score_basis}")
    print(f"Skip tremor-free sensors: {not args.include_tremor_free_check}")
    if score_rms_range is not None:
        print(f"SCORE_RMS_RANGE (for reference): {score_rms_range}")

    variant_dirs = {
        "parkinson": args.parkinson_dir,
        "mod_severe": args.mod_severe_dir,
        "mild_mod": args.mild_mod_dir,
        "clean": args.clean_dir,
    }

    results: List[VariantResult] = []
    for variant_name, variant_dir in variant_dirs.items():
        res = evaluate_variant(
            variant_name=variant_name,
            variant_dir=variant_dir,
            get_tremor_score=get_tremor_score,
            max_examples=args.max_examples,
            score_basis=args.score_basis,
            beta_activity=beta_activity,
            c_body_part=c_body_part,
            ankle_ratio_by_score=ankle_ratio_by_score,
            tremor_free_sensors=tremor_free_sensors,
            skip_tremor_free_sensors=not args.include_tremor_free_check,
            score_rms_range=score_rms_range,
        )
        results.append(res)

    print_separator()
    print("FINAL SUMMARY")
    print_separator()

    any_fail = False
    total_mismatches = 0
    total_checked = 0
    total_sentinel = 0
    total_skipped_tremor_free = 0
    total_skipped_missing_sensor = 0
    total_skipped_unresolved = 0

    for res in results:
        total_mismatches += res.mismatches
        total_checked += res.checked_samples
        total_sentinel += res.sentinel_samples
        total_skipped_tremor_free += res.skipped_tremor_free
        total_skipped_missing_sensor += res.skipped_missing_sensor
        total_skipped_unresolved += res.skipped_unresolved
        if res.failed:
            any_fail = True

        if res.checked_samples > 0:
            acc = 100.0 * res.correct / res.checked_samples
        else:
            acc = 100.0

        status = "PASS" if not res.failed else "FAIL"
        print(
            f"[{status}] {res.variant_name}: match_rate={acc:.2f}%, "
            f"checked={res.checked_samples}, sentinel={res.sentinel_samples}, "
            f"skipped_tremor_free={res.skipped_tremor_free}, "
            f"skipped_missing_sensor={res.skipped_missing_sensor}, "
            f"skipped_unresolved={res.skipped_unresolved}, mismatches={res.mismatches}"
        )

    overall_acc = 100.0 if total_checked == 0 else (100.0 * (total_checked - total_mismatches) / total_checked)
    print(
        f"Overall: match_rate={overall_acc:.2f}%, checked={total_checked}, "
        f"sentinel={total_sentinel}, skipped_tremor_free={total_skipped_tremor_free}, "
        f"skipped_missing_sensor={total_skipped_missing_sensor}, "
        f"skipped_unresolved={total_skipped_unresolved}, mismatches={total_mismatches}"
    )

    if any_fail:
        print("\nResult: FAIL")
        return 1

    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
