"""
Merge Parkinson activity trials (Data_parkinson) into an existing simulated variant.

This script extends activity labels in an existing .npz/.txt variant by appending
new windows from Parkinson activity files (.xls). It keeps the original file format
and writes a new variant directory.

Default behavior:
- Reads source variant from data_simulated/Datagenerator_files/.../fs*/
- Uses Upper Right sensor sheet to extend arm sensors:
  - Acc_arm  <- Cal1..Cal3
  - Gyro_arm <- Cal4..Cal6
  - Mag_arm  <- Cal7..Cal9
- Adds four new activities with labels above existing classes:
  - calibration
  - key
  - cardigan
  - toast

Example:
    /opt/miniconda3/bin/python merge_parkinson_activities.py \
      --source-variant data_simulated/Datagenerator_files/s1_w1_aug2/fs10_AWGN_s0p3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample


DEFAULT_PARKINSON_ROOT = Path("Data_parkinson")
DEFAULT_OUTPUT_SUFFIX = "_plus_parkinson_activities"
PD_SOURCE = "pd"
CT_SOURCE = "ct"
# Sentinel used in tremor_score for real PD windows with unknown clinical severity.
PD_TREMOR_SENTINEL_SCORE = 5

# Old variant sensor names mapped to Data_parkinson sheet + Cal columns.
DEFAULT_SENSOR_SPECS = {
    "Acc_arm": ("Upper Right", (1, 2, 3)),
    "Gyro_arm": ("Upper Right", (4, 5, 6)),
    "Mag_arm": ("Upper Right", (7, 8, 9)),
}

# Activity order is fixed to produce stable labels.
NEW_ACTIVITY_ORDER = ["calibration", "key", "cardigan", "toast"]


@dataclass
class TrialWindowPlan:
    subject_folder: str
    source: str
    subject_id_new: int
    file_path: Path
    activity_name: str
    activity_label: int
    repetition: int
    source_fs: float
    window_len_source: int
    stride_source: int
    n_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Data_parkinson activity windows into an existing variant.",
    )
    parser.add_argument(
        "--source-variant",
        type=Path,
        required=True,
        help="Path to source variant directory (contains sensor .npz/.txt files).",
    )
    parser.add_argument(
        "--parkinson-root",
        type=Path,
        default=DEFAULT_PARKINSON_ROOT,
        help="Root directory for Parkinson .xls trials (default: Data_parkinson).",
    )
    parser.add_argument(
        "--output-variant",
        type=Path,
        default=None,
        help="Output variant directory. Defaults to <source>_plus_parkinson_activities.",
    )
    parser.add_argument(
        "--target-sensors",
        nargs="+",
        default=list(DEFAULT_SENSOR_SPECS.keys()),
        help="Sensor branches to extend (default: Acc_arm Gyro_arm Mag_arm).",
    )
    parser.add_argument(
        "--aug-size",
        type=int,
        default=None,
        help="Augmented copies per new base window. Defaults to inferred value from source.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.01,
        help="Gaussian augmentation noise level (default: 0.01).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for augmentation noise (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report merge plan without writing files.",
    )
    return parser.parse_args()


def _list_sensor_npz(variant_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in sorted(variant_dir.glob("*.npz")):
        if p.name.startswith("._"):
            continue
        mapping[p.stem] = p
    if not mapping:
        raise RuntimeError(f"No .npz sensor files found in {variant_dir}")
    return mapping


def _infer_activity_name(file_stem: str) -> Optional[str]:
    s = file_stem.lower()
    if "calibration" in s:
        return "calibration"
    if "key" in s:
        return "key"
    if "cardigan" in s:
        return "cardigan"
    if "toast" in s:
        return "toast"
    return None


def _infer_repetition(file_stem: str) -> int:
    m = re.search(r"(\d+)$", file_stem.strip())
    if m:
        return int(m.group(1))
    return 0


def _infer_subject_source(subject_folder: str) -> Optional[str]:
    name = subject_folder.lower()
    if name.startswith("pd"):
        return PD_SOURCE
    if name.startswith("ct"):
        return CT_SOURCE
    return None


def _zscore_window(win: np.ndarray) -> np.ndarray:
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def _augment_window(win: np.ndarray, rng: np.random.RandomState, noise_level: float) -> np.ndarray:
    scale = noise_level * np.max(np.abs(win), axis=0, keepdims=True)
    noise = scale * rng.randn(*win.shape)
    return win + noise


def _normalize_sheet_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _load_sheet_matrix(file_path: Path, sheet_name: str) -> np.ndarray:
    xls = pd.ExcelFile(file_path)
    normalized_to_actual = {
        _normalize_sheet_name(s): s for s in xls.sheet_names
    }
    wanted = _normalize_sheet_name(sheet_name)
    if wanted not in normalized_to_actual:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {file_path}. "
            f"Available sheets: {xls.sheet_names}"
        )

    actual_sheet_name = normalized_to_actual[wanted]
    df = pd.read_excel(xls, sheet_name=actual_sheet_name)
    if df.empty:
        raise ValueError(f"Sheet '{sheet_name}' is empty in {file_path}")

    # Expect at least Time + Cal1..Cal9 (10 columns).
    if df.shape[1] < 10:
        raise ValueError(
            f"Sheet '{sheet_name}' in {file_path} has {df.shape[1]} columns, expected >= 10"
        )

    numeric = df.iloc[:, :10].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="any")
    if numeric.empty:
        raise ValueError(f"Sheet '{sheet_name}' in {file_path} has no valid numeric rows")

    return numeric.to_numpy(dtype=np.float32)


def _estimate_fs_from_time(time_col: np.ndarray, fallback_fs: float = 50.0) -> float:
    diffs = np.diff(time_col)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return fallback_fs
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return fallback_fs
    return 1.0 / median_dt


def _resample_to_len(win: np.ndarray, target_len: int) -> np.ndarray:
    if win.shape[0] == target_len:
        return win
    out = np.zeros((target_len, win.shape[1]), dtype=np.float32)
    for c in range(win.shape[1]):
        out[:, c] = resample(win[:, c], target_len).astype(np.float32)
    return out


def _infer_aug_size(npz_data: Dict[str, np.ndarray]) -> int:
    y = npz_data["y"]
    if "base_window_idx" not in npz_data:
        return 1
    base = npz_data["base_window_idx"]
    if base.size == 0:
        return 1
    n_unique = np.unique(base).size
    if n_unique == 0:
        return 1
    est = int(round(float(y.size) / float(n_unique)))
    return max(est, 1)


def _flatten_channel_blocks(x: np.ndarray) -> np.ndarray:
    # x is (N, C, L), return (N, C*L) in channel-block format.
    n, c, _ = x.shape
    blocks = [x[:, i, :] for i in range(c)]
    return np.concatenate(blocks, axis=1).reshape(n, -1)


def _build_txt_extra_columns(
    n_rows: int,
    n_extra_cols: int,
    y_zero_indexed: np.ndarray,
    subject_ids: np.ndarray,
    base_idx: np.ndarray,
    is_pd: Optional[np.ndarray] = None,
) -> np.ndarray:
    extra = np.zeros((n_rows, n_extra_cols), dtype=np.float32)
    if n_extra_cols >= 1:
        extra[:, 0] = (y_zero_indexed + 1).astype(np.float32)
    if n_extra_cols >= 2:
        extra[:, 1] = subject_ids.astype(np.float32)
    if n_extra_cols >= 3:
        extra[:, 2] = base_idx.astype(np.float32)
    # If tremor label columns exist (7 extras total), write PD sentinel in tremor_score.
    if n_extra_cols >= 7 and is_pd is not None:
        extra[:, 6] = np.where(is_pd, PD_TREMOR_SENTINEL_SCORE, 0).astype(np.float32)
    return extra


def _load_source_txt(txt_path: Path) -> np.ndarray:
    arr = np.loadtxt(txt_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32)


def build_trial_plan(
    parkinson_root: Path,
    activity_to_label: Dict[str, int],
    subject_map: Dict[str, int],
    window_sec: float,
    stride_sec: float,
) -> List[TrialWindowPlan]:
    plans: List[TrialWindowPlan] = []

    trial_files = sorted(
        [p for p in parkinson_root.rglob("*.xls") if not p.name.startswith("._")],
        key=lambda p: (p.parent.name.lower(), p.name.lower()),
    )

    for file_path in trial_files:
        activity = _infer_activity_name(file_path.stem)
        if activity is None:
            continue

        subject_folder = file_path.parent.name
        source = _infer_subject_source(subject_folder)
        if source is None:
            continue
        subject_id_new = subject_map[subject_folder]
        repetition = _infer_repetition(file_path.stem)

        # Use one known sheet for fs estimation and row count.
        try:
            base_sheet = _load_sheet_matrix(file_path, "Upper Right")
        except Exception as exc:
            print(f"[WARN] Skipping file (cannot read base sheet): {file_path} | {exc}")
            continue

        source_fs = _estimate_fs_from_time(base_sheet[:, 0], fallback_fs=50.0)
        window_len_source = max(1, int(round(window_sec * source_fs)))
        stride_source = max(1, int(round(stride_sec * source_fs)))

        n_rows = base_sheet.shape[0]
        if n_rows < window_len_source:
            continue

        n_windows = 1 + (n_rows - window_len_source) // stride_source
        if n_windows <= 0:
            continue

        plans.append(
            TrialWindowPlan(
                subject_folder=subject_folder,
                source=source,
                subject_id_new=subject_id_new,
                file_path=file_path,
                activity_name=activity,
                activity_label=activity_to_label[activity],
                repetition=repetition,
                source_fs=source_fs,
                window_len_source=window_len_source,
                stride_source=stride_source,
                n_windows=n_windows,
            )
        )

    return plans


def main() -> None:
    args = parse_args()

    source_variant = args.source_variant.resolve()
    if not source_variant.exists() or not source_variant.is_dir():
        raise FileNotFoundError(f"Source variant directory not found: {source_variant}")

    parkinson_root = args.parkinson_root
    if not parkinson_root.is_absolute():
        parkinson_root = (Path.cwd() / parkinson_root).resolve()
    if not parkinson_root.exists() or not parkinson_root.is_dir():
        raise FileNotFoundError(f"Parkinson root not found: {parkinson_root}")

    source_npz = _list_sensor_npz(source_variant)

    target_sensors = args.target_sensors
    unknown = sorted(set(target_sensors) - set(source_npz.keys()))
    if unknown:
        raise ValueError(f"Requested target sensors not found in source variant: {unknown}")

    for s in target_sensors:
        if s not in DEFAULT_SENSOR_SPECS:
            raise ValueError(
                f"No default Data_parkinson mapping for sensor '{s}'. "
                f"Available mapped sensors: {sorted(DEFAULT_SENSOR_SPECS)}"
            )

    # Reference sensor determines shared geometry and class range.
    ref_sensor = target_sensors[0]
    with np.load(source_npz[ref_sensor], allow_pickle=False) as ref_npz:
        ref_data = {k: ref_npz[k] for k in ref_npz.files}

    fs_target = int(np.array(ref_data["fs"]).item())
    window_len_target = int(np.array(ref_data["window_len"]).item())
    stride_target = int(np.array(ref_data["stride"]).item())
    window_sec = float(window_len_target) / float(fs_target)
    stride_sec = float(stride_target) / float(fs_target)

    old_max_label = int(np.max(ref_data["y"]))
    activity_to_label = {
        name: old_max_label + 1 + i for i, name in enumerate(NEW_ACTIVITY_ORDER)
    }

    # Subject mapping: append new subject IDs after existing maximum.
    existing_subject_max = int(np.max(ref_data.get("subject_id", np.array([0], dtype=np.int64))))
    subject_dirs = sorted(
        [p for p in parkinson_root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )
    subject_map = {
        p.name: existing_subject_max + 1 + i for i, p in enumerate(subject_dirs)
    }

    if args.aug_size is None:
        aug_size = _infer_aug_size(ref_data)
    else:
        aug_size = max(int(args.aug_size), 1)

    rng = np.random.RandomState(args.seed)

    plans = build_trial_plan(
        parkinson_root=parkinson_root,
        activity_to_label=activity_to_label,
        subject_map=subject_map,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )

    if not plans:
        raise RuntimeError("No valid Parkinson trials/windows found to merge.")

    # Count expected new base windows before augmentation.
    n_new_base_windows = sum(p.n_windows for p in plans)
    n_new_samples = n_new_base_windows * aug_size

    # Build per-sensor new samples.
    new_sensor_data: Dict[str, Dict[str, List[np.ndarray]]] = {
        s: {
            "X": [],
            "y": [],
            "subject_id": [],
            "base_window_idx": [],
            "source": [],
        }
        for s in target_sensors
    }

    # Base index offset after existing base windows.
    if "base_window_idx" in ref_data:
        next_base_idx = int(np.max(ref_data["base_window_idx"])) + 1
    else:
        next_base_idx = int(ref_data["y"].shape[0])

    activity_counts = Counter()
    subject_counts = Counter()

    skipped_trials = 0
    for plan in plans:
        needed_sheets = sorted({DEFAULT_SENSOR_SPECS[s][0] for s in target_sensors})
        try:
            sheet_cache = {sheet: _load_sheet_matrix(plan.file_path, sheet) for sheet in needed_sheets}
        except Exception as exc:
            skipped_trials += 1
            print(f"[WARN] Skipping trial during merge: {plan.file_path} | {exc}")
            continue

        # Ensure all needed sheets have enough rows and are aligned by length.
        min_rows = min(sheet_cache[s].shape[0] for s in needed_sheets)
        if min_rows < plan.window_len_source:
            continue

        for base_offset in range(0, min_rows - plan.window_len_source + 1, plan.stride_source):
            base_idx = next_base_idx
            next_base_idx += 1

            for _ in range(aug_size):
                for sensor_name in target_sensors:
                    sheet_name, cols = DEFAULT_SENSOR_SPECS[sensor_name]
                    sheet_arr = sheet_cache[sheet_name]

                    win_source = sheet_arr[
                        base_offset : base_offset + plan.window_len_source,
                        list(cols),
                    ]
                    win_resampled = _resample_to_len(win_source, window_len_target)
                    win_norm = _zscore_window(win_resampled)
                    if plan.source == PD_SOURCE:
                        # Keep PD windows non-synthetic in this merge path.
                        win_aug = win_norm
                    else:
                        win_aug = _augment_window(win_norm, rng, args.noise_level)

                    # Store as (C, L)
                    new_sensor_data[sensor_name]["X"].append(win_aug.T.astype(np.float32))
                    new_sensor_data[sensor_name]["y"].append(np.int64(plan.activity_label))
                    new_sensor_data[sensor_name]["subject_id"].append(np.int64(plan.subject_id_new))
                    new_sensor_data[sensor_name]["base_window_idx"].append(np.int64(base_idx))
                    new_sensor_data[sensor_name]["source"].append(plan.source)

            activity_counts[plan.activity_name] += 1
            subject_counts[plan.subject_folder] += 1

    # Materialize arrays.
    sensor_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for sensor_name in target_sensors:
        bucket = new_sensor_data[sensor_name]
        if not bucket["X"]:
            raise RuntimeError(f"No new windows created for sensor {sensor_name}")

        x = np.stack(bucket["X"], axis=0)
        y = np.array(bucket["y"], dtype=np.int64)
        subject_id = np.array(bucket["subject_id"], dtype=np.int64)
        base_window_idx = np.array(bucket["base_window_idx"], dtype=np.int64)

        sensor_arrays[sensor_name] = {
            "X": x,
            "y": y,
            "subject_id": subject_id,
            "base_window_idx": base_window_idx,
            "is_pd": np.array([src == PD_SOURCE for src in bucket["source"]], dtype=bool),
        }

    # Dry-run report and stop early.
    print("=" * 80)
    print("MERGE PLAN")
    print("=" * 80)
    print(f"Source variant: {source_variant}")
    print(f"Parkinson root: {parkinson_root}")
    print(f"Target sensors: {target_sensors}")
    print(f"Target fs/window/stride: {fs_target} Hz / {window_len_target} / {stride_target}")
    print(f"Window seconds: {window_sec:.4f}, stride seconds: {stride_sec:.4f}")
    print(f"Augmentation copies for new data: {aug_size}")
    print(f"Old label max: {old_max_label}")
    print(f"New label map: {activity_to_label}")
    print(f"Trials included: {len(plans)}")
    print(f"Trials skipped during merge pass: {skipped_trials}")
    print(f"New base windows: {sum(activity_counts.values())}")
    print(f"New samples per extended sensor: {sensor_arrays[ref_sensor]['y'].shape[0]}")
    print("Activity base-window counts:")
    for name in NEW_ACTIVITY_ORDER:
        print(f"  - {name}: {activity_counts[name]}")
    print(f"Subjects mapped: {len(subject_map)} (new IDs start at {existing_subject_max + 1})")

    if args.dry_run:
        print("Dry-run enabled: no files written.")
        return

    if args.output_variant is None:
        output_variant = source_variant.parent / f"{source_variant.name}{DEFAULT_OUTPUT_SUFFIX}"
    else:
        output_variant = args.output_variant
        if not output_variant.is_absolute():
            output_variant = (Path.cwd() / output_variant).resolve()

    if output_variant.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_variant}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_variant)

    # Copy whole variant to keep untouched files and logs.
    shutil.copytree(source_variant, output_variant)

    # Merge selected sensor npz/txt.
    for sensor_name, source_npz_path in source_npz.items():
        out_npz_path = output_variant / source_npz_path.name
        out_txt_path = output_variant / f"{sensor_name}.txt"

        if sensor_name not in target_sensors:
            continue

        new_bucket = sensor_arrays[sensor_name]

        with np.load(source_npz_path, allow_pickle=False) as old_npz:
            merged: Dict[str, np.ndarray] = {}
            for key in old_npz.files:
                old_val = old_npz[key]

                if key == "X":
                    merged[key] = np.concatenate([old_val, new_bucket["X"]], axis=0)
                elif key == "y":
                    merged[key] = np.concatenate([old_val, new_bucket["y"]], axis=0)
                elif key == "subject_id":
                    merged[key] = np.concatenate([old_val, new_bucket["subject_id"]], axis=0)
                elif key == "base_window_idx":
                    merged[key] = np.concatenate([old_val, new_bucket["base_window_idx"]], axis=0)
                elif key == "tremor_score":
                    pd_scores = np.where(
                        new_bucket["is_pd"],
                        PD_TREMOR_SENTINEL_SCORE,
                        0,
                    ).astype(old_val.dtype)
                    merged[key] = np.concatenate([old_val, pd_scores], axis=0)
                elif key in {
                    "tremor_freq",
                    "tremor_acc_rms",
                    "tremor_gyro_rms",
                    "group_label",
                }:
                    zeros = np.zeros(new_bucket["y"].shape[0], dtype=old_val.dtype)
                    merged[key] = np.concatenate([old_val, zeros], axis=0)
                else:
                    merged[key] = old_val

        np.savez_compressed(out_npz_path, **merged)

        if out_txt_path.exists() and not out_txt_path.name.startswith("._"):
            old_txt = _load_source_txt(out_txt_path)
            x_new_flat = _flatten_channel_blocks(new_bucket["X"])
            n_extra = old_txt.shape[1] - x_new_flat.shape[1]
            if n_extra < 1:
                raise RuntimeError(
                    f"Unexpected TXT format for {out_txt_path}. Could not infer label columns."
                )

            extra = _build_txt_extra_columns(
                n_rows=x_new_flat.shape[0],
                n_extra_cols=n_extra,
                y_zero_indexed=new_bucket["y"],
                subject_ids=new_bucket["subject_id"],
                base_idx=new_bucket["base_window_idx"],
                is_pd=new_bucket["is_pd"],
            )
            new_rows = np.hstack([x_new_flat.astype(np.float32), extra.astype(np.float32)])

            if new_rows.shape[1] != old_txt.shape[1]:
                raise RuntimeError(
                    f"TXT column mismatch for {sensor_name}: old={old_txt.shape[1]} new={new_rows.shape[1]}"
                )

            merged_txt = np.vstack([old_txt, new_rows]).astype(np.float32)
            np.savetxt(out_txt_path, merged_txt, delimiter=",", fmt="%.6f")

    # Write merge metadata.
    metadata = {
        "source_variant": str(source_variant),
        "output_variant": str(output_variant),
        "parkinson_root": str(parkinson_root),
        "extended_sensors": target_sensors,
        "target_fs": fs_target,
        "target_window_len": window_len_target,
        "target_stride": stride_target,
        "window_seconds": window_sec,
        "stride_seconds": stride_sec,
        "aug_size_new_data": aug_size,
        "noise_level_new_data": args.noise_level,
        "seed": args.seed,
        "activity_to_label": activity_to_label,
        "subject_id_map": subject_map,
        "new_base_window_counts": dict(activity_counts),
        "new_base_windows_total": int(sum(activity_counts.values())),
        "new_samples_per_extended_sensor": int(sensor_arrays[ref_sensor]["y"].shape[0]),
        "pd_tremor_score_sentinel": PD_TREMOR_SENTINEL_SCORE,
    }

    merge_meta_path = output_variant / "parkinson_merge_metadata.json"
    with open(merge_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    merge_info_path = output_variant / "parkinson_merge_info.txt"
    lines = [
        "=" * 80,
        "PARKINSON ACTIVITY MERGE SUMMARY",
        "=" * 80,
        f"Source variant: {source_variant}",
        f"Output variant: {output_variant}",
        f"Parkinson root: {parkinson_root}",
        f"Extended sensors: {', '.join(target_sensors)}",
        "",
        f"Target fs: {fs_target} Hz",
        f"Target window len: {window_len_target} samples",
        f"Target stride: {stride_target} samples",
        f"Window seconds: {window_sec:.4f}",
        f"Stride seconds: {stride_sec:.4f}",
        "",
        f"Augmentation copies (new data): {aug_size}",
        f"Augmentation noise level: {args.noise_level}",
        f"Seed: {args.seed}",
        "",
        f"Old max class label (0-indexed): {old_max_label}",
        "New activity labels (0-indexed):",
    ]
    for name in NEW_ACTIVITY_ORDER:
        lines.append(f"  - {name}: {activity_to_label[name]}")

    lines.extend([
        "",
        f"New base windows total: {sum(activity_counts.values())}",
        f"New samples per extended sensor: {sensor_arrays[ref_sensor]['y'].shape[0]}",
        f"PD tremor_score sentinel: {PD_TREMOR_SENTINEL_SCORE}",
        f"  (sentinel means real PD window with unknown clinical severity)",
        "New base windows per activity:",
    ])
    for name in NEW_ACTIVITY_ORDER:
        lines.append(f"  - {name}: {activity_counts[name]}")

    lines.extend([
        "",
        f"Mapped subjects: {len(subject_map)}",
        "Metadata file: parkinson_merge_metadata.json",
        "=" * 80,
    ])

    with open(merge_info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Output: {output_variant}")
    print(f"Metadata: {merge_meta_path}")
    print(f"Info: {merge_info_path}")


if __name__ == "__main__":
    main()
