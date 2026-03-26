#!/usr/bin/env python3
"""
Merged tremor-labeled dataset generator.

Goal:
- Keep the tremor processing pipeline aligned with DataGenerator_Tremor.py
- Adapt only dataset ingestion/merge for the new structure:
  - Data/MHEALTHDATASET
  - Data/Data_parkinson

Output variants follow the same naming style as DataGenerator_Tremor.py:
  s{stride}_w{window}_fs{FS}_tremor_{mode}
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample

# Ensure top-level project modules are importable when executed from Parkinson_dataset_work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Noise_simulation.Tremor import (
    precompute_tremor_cache_with_parkinson_model,
    write_tremor_parkinson_params_file,
    apply_tremor_rotation_to_magnetometer,
)
import Noise_simulation.tremor_parkinson_config as pk_config


# ============================================================
# CONFIG - Basic parameters
# ============================================================
ORIGINAL_FS = 50
FS = 50
WINDOW_SEC = 2.0
STRIDE_SEC = 2.0
AUG_SIZE = 2
NOISE_LEVEL = 0.01
NUM_SUBJECTS = 10

# Paths
script_dir = Path(__file__).parent.resolve()
DATA_ROOT = script_dir / "Data"
MHEALTH_PATH = DATA_ROOT / "MHEALTHDATASET"
PARKINSON_PATH = DATA_ROOT / "Data_parkinson"
OUT_BASE = DATA_ROOT / "Tremor_datagenerator_files"

# Seeds
SEED = 0
TREMOR_SEED = 42

# Tremor settings (same style as DataGenerator_Tremor.py)
GENERATE_TREMOR = True
RUN_SANITY_CHECKS = False

# Source-aware tremor policy
CONTROL_LIKE_SOURCES = {"mhealth", "ct"}
PD_SOURCE = "pd"
# Sentinel: real PD window with unknown clinical tremor severity.
PD_TREMOR_SENTINEL_SCORE = 5
ALLOWED_TREMOR_SCORES_BY_MODE = {
    "clean": {0},
    "mild_mod": {1, 2},
    "mod_severe": {3, 4},
    "parkinson": {PD_TREMOR_SENTINEL_SCORE},
}

# Parameters passed into precompute_tremor_cache_with_parkinson_model
TREMOR_MU = 1.0
TREMOR_SIGMA = 0.5
TREMOR_DT = 0.001
TREMOR_INTERMITTENT = False
TREMOR_ON_PROB = 0.5
TREMOR_MIN_ON_SEC = 2.0
TREMOR_MAX_ON_SEC = 8.0
USE_TREMOR_JITTER = True
TREMOR_JITTER_STD = 0.15

# Sensor map (same as mHealth/DataGenerator_Tremor.py)
SENSORS = {
    "Acc_chest": [0, 1, 2],
    "ECG": [3, 4],
    "Acc_ankle": [5, 6, 7],
    "Gyro_ankle": [8, 9, 10],
    "Mag_ankle": [11, 12, 13],
    "Acc_arm": [14, 15, 16],
    "Gyro_arm": [17, 18, 19],
    "Mag_arm": [20, 21, 22],
}
LABEL_COL = 23

# Parkinson data: available IMU channels from "Upper Right" sheet only
PARKINSON_AVAILABLE_SENSORS = {"Acc_arm", "Gyro_arm", "Mag_arm"}

# Activity translation for Parkinson-specific classes.
# mHealth already uses labels 1..12, so Parkinson activities are assigned new
# non-overlapping labels 13..16.
PARKINSON_ACTIVITY_TO_MHEALTH_LABEL = {
    "calibration": 13,
    "key": 14,
    "cardigan": 15,
    "toast": 16,
}

# Tremor model compatibility: map new labels to existing activity modulation
# profiles from mHealth labels (beta map) for tremor-cache generation.
PARKINSON_ACTIVITY_TREMOR_BETA_REFERENCE = {
    "calibration": 1,
    "key": 7,
    "cardigan": 7,
    "toast": 7,
}


@dataclass
class WindowSpec:
    source: str  # mhealth, ct, pd
    subject_id: int
    label: int  # 1..16
    win_start: int
    win_end: int
    patient_id: str
    activity_name: str
    sensor_profile: str  # full, arm_only


# ============================================================
# Helpers (aligned with DataGenerator_Tremor.py)
# ============================================================
def _format_num(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return str(v).rstrip("0").rstrip(".")


def resample_window(win: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs == target_fs:
        return win
    L_original, C = win.shape
    L_target = int(round(L_original * target_fs / original_fs))
    out = np.zeros((L_target, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = resample(win[:, c], L_target).astype(np.float32)
    return out


def zscore_window(win: np.ndarray) -> np.ndarray:
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def augment_window(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, NOISE_LEVEL, size=win.shape)
    return win + noise


def generate_rotation_matrix(rng: np.random.Generator, max_angle_deg: float = 15.0) -> np.ndarray:
    max_angle_rad = np.deg2rad(max_angle_deg)
    theta_x, theta_y, theta_z = rng.uniform(-max_angle_rad, max_angle_rad, size=3)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return (Rz @ Ry @ Rx).astype(np.float32)


def apply_rotation_augmentation(win_raw: np.ndarray, rng: np.random.Generator, max_angle_deg: float = 15.0) -> np.ndarray:
    R = generate_rotation_matrix(rng, max_angle_deg)
    return win_raw @ R.T


def apply_awgn_raw(win_raw: np.ndarray, rng: np.random.Generator, rms_ratio: float = 0.2) -> np.ndarray:
    L, C = win_raw.shape
    noise = np.zeros_like(win_raw)
    for c in range(C):
        signal_rms = float(np.sqrt(np.mean(win_raw[:, c] ** 2)))
        if signal_rms < 1e-8:
            signal_rms = 1.0
        noise_std = rms_ratio * signal_rms
        noise[:, c] = rng.normal(0, noise_std, size=L)
    return win_raw + noise


def load_subject_data_with_retry(file_path: Path, max_retries: int = 5, delay: float = 2.0) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1 or (data.ndim > 1 and data.shape[1] < 24):
                print(f"WARNING: {file_path.name} appears incomplete ({data.shape}).")
            return data
        except (TimeoutError, OSError):
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    N, C, L = X_c_l.shape
    blocks = [X_c_l[:, c, :] for c in range(C)]
    return np.concatenate(blocks, axis=1)


def extract_body_part(sensor_name: str) -> str:
    if "ankle" in sensor_name:
        return "ankle"
    if "arm" in sensor_name:
        return "arm"
    return "chest"


def get_sensor_type(sensor_name: str) -> str:
    if "Acc" in sensor_name:
        return "acc"
    if "Gyro" in sensor_name:
        return "gyro"
    if "Mag" in sensor_name:
        return "mag"
    if "ECG" in sensor_name:
        return "ecg"
    return "unknown"


def _normalize_sheet_name(name: str) -> str:
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
    return "".join(cleaned)


def _infer_activity_name(file_name: str) -> Optional[str]:
    n = file_name.lower()
    if "calibration" in n:
        return "calibration"
    if "cardigan" in n:
        return "cardigan"
    if "key" in n or "door" in n:
        return "key"
    if "toast" in n:
        return "toast"
    return None


def _load_upper_right_calibration(file_path: Path) -> np.ndarray:
    xls = pd.ExcelFile(file_path)
    normalized_to_actual = {_normalize_sheet_name(s): s for s in xls.sheet_names}

    candidates = [
        "Upper Right",
        "UpperRight",
        "IMU_Upper_Right",
        "IMU Upper Right",
        "IMUUpperRight",
        "upper right",
    ]

    ordered_sheets: List[str] = []

    selected = None
    for c in candidates:
        key = _normalize_sheet_name(c)
        if key in normalized_to_actual:
            selected = normalized_to_actual[key]
            break

    if selected is not None:
        ordered_sheets.append(selected)

    # Prefer sheets that look like upper-right aliases.
    for key, actual in normalized_to_actual.items():
        if "upperright" in key and actual not in ordered_sheets:
            ordered_sheets.append(actual)

    # Outlier fallback: some files only expose IMU_<id> sheets.
    for key, actual in normalized_to_actual.items():
        if key.startswith("imu") and actual not in ordered_sheets:
            ordered_sheets.append(actual)

    # Final fallback: try everything.
    for actual in xls.sheet_names:
        if actual not in ordered_sheets:
            ordered_sheets.append(actual)

    last_error = "unknown"
    for sheet_name in ordered_sheets:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.shape[1] < 10:
                last_error = f"sheet '{sheet_name}' has {df.shape[1]} columns"
                continue

            numeric = df.iloc[:, :10].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.dropna(how="any")
            if numeric.empty:
                last_error = f"sheet '{sheet_name}' has no valid numeric rows"
                continue

            # Time + Cal1..Cal9 => keep Cal1..Cal9 only
            return numeric.iloc[:, 1:10].to_numpy(dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive parsing fallback
            last_error = f"sheet '{sheet_name}' parse failed: {exc}"

    raise ValueError(f"{file_path.name}: unable to parse usable IMU sheet ({last_error})")


def _build_parkinson_trial_matrix(cal9: np.ndarray, label: int) -> np.ndarray:
    """
    Build mHealth-compatible 24-column matrix from Parkinson Upper Right IMU data.
    Missing sensors remain zeros.
    """
    T = cal9.shape[0]
    out = np.zeros((T, 24), dtype=np.float32)

    # Map Cal1..Cal9 to arm sensors in mHealth layout
    out[:, 14:17] = cal9[:, 0:3]  # Acc_arm
    out[:, 17:20] = cal9[:, 3:6]  # Gyro_arm
    out[:, 20:23] = cal9[:, 6:9]  # Mag_arm
    out[:, LABEL_COL] = float(label)
    return out


def _create_clean_cache_entry(length: int, subject_id: int, body_part: str, activity_label: int) -> Dict[str, np.ndarray]:
    return {
        "acc_noise": np.zeros((length, 3), dtype=np.float32),
        "gyro_noise": np.zeros((length, 3), dtype=np.float32),
        "mag_noise": np.zeros((length, 3), dtype=np.float32),
        "meta": {
            "subject_id": subject_id,
            "body_part": body_part,
            "activity": activity_label,
            "acc_target_rms": 0.0,
            "gyro_target_rms": 0.0,
            "freq_hz": 0.0,
            "severity": "clean",
            "score": 0,
            "sampled_score": 0,
        },
    }


def ensure_parkinson_activity_beta_support() -> None:
    """
    Extend pk_config.BETA_ACTIVITY so new Parkinson labels (13..16) reuse
    known activity modulation profiles from mHealth labels.
    """
    for activity_name, new_label in PARKINSON_ACTIVITY_TO_MHEALTH_LABEL.items():
        if new_label in pk_config.BETA_ACTIVITY:
            continue

        ref_label = PARKINSON_ACTIVITY_TREMOR_BETA_REFERENCE.get(activity_name, 1)
        ref_beta = pk_config.BETA_ACTIVITY.get(ref_label, 1.0)
        pk_config.BETA_ACTIVITY[new_label] = ref_beta


def save_tremor_branch_sensor(
    variant_dir: Path,
    sensor_name: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    base_window_idx: np.ndarray,
    tremor_freq: np.ndarray,
    tremor_acc_rms: np.ndarray,
    tremor_gyro_rms: np.ndarray,
    tremor_score: np.ndarray,
    sensor_cols: List[int],
    fs: int,
    window_len: int,
    stride: int,
) -> None:
    tremor_branch_name = f"{sensor_name}_tremorbranch"

    npz_out = variant_dir / f"{tremor_branch_name}.npz"
    np.savez_compressed(
        npz_out,
        X=X,
        y=y,
        subject_id=subject_ids,
        base_window_idx=base_window_idx,
        tremor_freq=tremor_freq,
        tremor_acc_rms=tremor_acc_rms,
        tremor_gyro_rms=tremor_gyro_rms,
        tremor_score=tremor_score,
        fs=fs,
        window_len=window_len,
        stride=stride,
        sensor_name=tremor_branch_name,
        sensor_cols=np.array(sensor_cols, dtype=np.int64),
    )

    txt_out = variant_dir / f"{tremor_branch_name}.txt"
    flat_rows = flatten_channel_blocks(X)
    all_labels = np.column_stack(
        [
            (y + 1),
            subject_ids,
            base_window_idx,
            tremor_freq,
            tremor_acc_rms,
            tremor_gyro_rms,
            tremor_score,
        ]
    )
    sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
    np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")


# ============================================================
# Merge-aware dataset builders
# ============================================================
def build_merged_subject_cache(include_ct: bool, include_pd: bool) -> Tuple[Dict[int, np.ndarray], List[WindowSpec], Dict[int, Dict[str, str]]]:
    subject_data: Dict[int, np.ndarray] = {}
    window_specs: List[WindowSpec] = []
    subject_meta: Dict[int, Dict[str, str]] = {}

    window_len = int(round(WINDOW_SEC * ORIGINAL_FS))
    stride = int(round(STRIDE_SEC * ORIGINAL_FS))
    stride = max(stride, 1)

    # ---------------------------
    # mHealth subjects (1..NUM_SUBJECTS)
    # ---------------------------
    for subj_idx in range(1, NUM_SUBJECTS + 1):
        fname = MHEALTH_PATH / f"mHealth_subject{subj_idx}.log"
        if not fname.exists():
            print(f"WARNING: missing mHealth file: {fname}")
            continue

        data = load_subject_data_with_retry(fname)
        if data.ndim != 2 or data.shape[1] < 24:
            print(f"WARNING: skipping malformed mHealth file: {fname.name}, shape={data.shape}")
            continue

        subject_data[subj_idx] = data
        subject_meta[subj_idx] = {
            "source": "mhealth",
            "patient_id": f"mHealth_subject{subj_idx}",
            "sensor_profile": "full",
        }

        labels = data[:, LABEL_COL].astype(int)
        i = 0
        while i + window_len <= len(data):
            lab_win = labels[i : i + window_len]
            if np.all(lab_win >= 1):
                lab_val = int(lab_win[0])
                window_specs.append(
                    WindowSpec(
                        source="mhealth",
                        subject_id=subj_idx,
                        label=lab_val,
                        win_start=i,
                        win_end=i + window_len,
                        patient_id=f"mHealth_subject{subj_idx}",
                        activity_name=f"mhealth_label_{lab_val}",
                        sensor_profile="full",
                    )
                )
            i += stride

    # ---------------------------
    # Parkinson subjects (CT/PD)
    # ---------------------------
    next_subject_id = max(subject_data.keys(), default=0) + 1

    if PARKINSON_PATH.exists():
        for patient_dir in sorted(PARKINSON_PATH.iterdir(), key=lambda p: p.name.lower()):
            if not patient_dir.is_dir():
                continue
            if patient_dir.name.startswith("._"):
                continue

            name_upper = patient_dir.name.upper()
            if name_upper.startswith("CT"):
                source = "ct"
                if not include_ct:
                    continue
            elif name_upper.startswith("PD"):
                source = "pd"
                if not include_pd:
                    continue
            else:
                continue

            trial_files = [
                p
                for p in sorted(patient_dir.iterdir(), key=lambda x: x.name.lower())
                if p.is_file()
                and p.suffix.lower() in {".xls", ".xlsx"}
                and not p.name.startswith("._")
            ]

            if not trial_files:
                continue

            subject_rows: List[np.ndarray] = []
            patient_windows: List[WindowSpec] = []
            cursor = 0

            for trial_file in trial_files:
                activity_name = _infer_activity_name(trial_file.name)
                if activity_name is None:
                    continue

                mapped_label = PARKINSON_ACTIVITY_TO_MHEALTH_LABEL.get(activity_name)
                if mapped_label is None:
                    continue

                try:
                    cal9 = _load_upper_right_calibration(trial_file)
                except Exception as exc:
                    print(f"WARNING: skipping {trial_file}: {exc}")
                    continue

                trial_data = _build_parkinson_trial_matrix(cal9, mapped_label)
                if trial_data.shape[0] < window_len:
                    continue

                start = cursor
                end = start + trial_data.shape[0]
                subject_rows.append(trial_data)

                for offset in range(0, trial_data.shape[0] - window_len + 1, stride):
                    patient_windows.append(
                        WindowSpec(
                            source=source,
                            subject_id=next_subject_id,
                            label=mapped_label,
                            win_start=start + offset,
                            win_end=start + offset + window_len,
                            patient_id=patient_dir.name,
                            activity_name=activity_name,
                            sensor_profile="arm_only",
                        )
                    )

                cursor = end

                # Separator block with label 0 avoids accidental cross-trial windows
                sep = np.zeros((window_len, 24), dtype=np.float32)
                subject_rows.append(sep)
                cursor += window_len

            if not patient_windows:
                continue

            subject_data[next_subject_id] = np.vstack(subject_rows)
            subject_meta[next_subject_id] = {
                "source": source,
                "patient_id": patient_dir.name,
                "sensor_profile": "arm_only",
            }
            window_specs.extend(patient_windows)
            next_subject_id += 1

    return subject_data, window_specs, subject_meta


def build_tremor_cache_for_windows(
    window_specs: List[WindowSpec],
    subject_data_cache: Dict[int, np.ndarray],
    augment_mode: str,
) -> Dict[Tuple[int, str], Dict[str, np.ndarray]]:
    control_like_indexed_specs = [
        (global_idx, ws)
        for global_idx, ws in enumerate(window_specs)
        if ws.source in CONTROL_LIKE_SOURCES
    ]
    cache_window_specs = [
        (ws.subject_id, ws.label, ws.win_start, ws.win_end)
        for _, ws in control_like_indexed_specs
    ]

    cache: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
    if not GENERATE_TREMOR:
        for w_idx, ws in enumerate(window_specs):
            L = ws.win_end - ws.win_start
            for body_part in ["ankle", "arm", "chest"]:
                cache[(w_idx, body_part)] = _create_clean_cache_entry(L, ws.subject_id, body_part, ws.label)
        return cache

    def data_loader(subj_idx: int) -> np.ndarray:
        return subject_data_cache[subj_idx]

    if cache_window_specs:
        control_like_cache = precompute_tremor_cache_with_parkinson_model(
            window_specs=cache_window_specs,
            sensor_column_mapping=SENSORS,
            data_loader_func=data_loader,
            fs=ORIGINAL_FS,
            tremor_mu=TREMOR_MU,
            tremor_sigma=TREMOR_SIGMA,
            tremor_dt=TREMOR_DT,
            tremor_intermittent=TREMOR_INTERMITTENT,
            tremor_on_prob=TREMOR_ON_PROB,
            tremor_min_on_sec=TREMOR_MIN_ON_SEC,
            tremor_max_on_sec=TREMOR_MAX_ON_SEC,
            scenario_seed=TREMOR_SEED,
            use_jitter=USE_TREMOR_JITTER,
            jitter_std=TREMOR_JITTER_STD if USE_TREMOR_JITTER else 0.0,
            sampling_method=pk_config.DEFAULT_SAMPLING_METHOD,
            augment_mode=augment_mode,
        )

        for local_idx, (global_idx, _) in enumerate(control_like_indexed_specs):
            for body_part in ["ankle", "arm", "chest"]:
                entry = control_like_cache.get((local_idx, body_part))
                if entry is not None:
                    cache[(global_idx, body_part)] = entry

    # Ensure cache has all expected keys
    for w_idx, ws in enumerate(window_specs):
        L = ws.win_end - ws.win_start
        for body_part in ["ankle", "arm", "chest"]:
            if (w_idx, body_part) not in cache:
                cache[(w_idx, body_part)] = _create_clean_cache_entry(L, ws.subject_id, body_part, ws.label)

    return cache


def write_window_source_map(variant_dir: Path, window_specs: List[WindowSpec]) -> None:
    out_jsonl = variant_dir / "window_source_map.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, ws in enumerate(window_specs):
            rec = {
                "base_window_idx": idx,
                "source": ws.source,
                "subject_id": ws.subject_id,
                "patient_id": ws.patient_id,
                "activity_name": ws.activity_name,
                "mapped_activity_label": ws.label,
                "sensor_profile": ws.sensor_profile,
            }
            f.write(json.dumps(rec) + "\n")


def write_sensor_coverage(variant_dir: Path) -> None:
    lines = [
        "SENSOR COVERAGE",
        "=" * 72,
        "",
        "mHealth samples:",
        "  - Full sensor set available",
        "",
        "Parkinson samples (CT/PD):",
        "  - Available: Acc_arm, Gyro_arm, Mag_arm (from Upper Right Cal1..Cal9)",
        "  - Missing: Acc_chest, ECG, Acc_ankle, Gyro_ankle, Mag_ankle",
        "  - Missing sensors remain zeros (no synthetic fill)",
        "",
        "Important:",
        "  - Tremor pipeline follows DataGenerator_Tremor.py for available sensors.",
        "  - Missing sensors are kept as zeros to preserve source availability constraints.",
        "=" * 72,
    ]
    (variant_dir / "sensor_coverage.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main generation
# ============================================================
def generate_variant(
    variant_name: str,
    augment_mode: str,
    window_specs: List[WindowSpec],
    subject_data_cache: Dict[int, np.ndarray],
) -> None:
    print("=" * 80)
    print(f"Generating variant: {variant_name}")
    print(f"Augment mode: {augment_mode}")
    print("=" * 80)

    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    target_window_len = int(round(WINDOW_SEC * FS))
    target_stride = int(round(STRIDE_SEC * FS))

    is_parkinson_mode = augment_mode == "parkinson"
    allowed_scores = ALLOWED_TREMOR_SCORES_BY_MODE.get(augment_mode)
    if is_parkinson_mode:
        variant_window_specs = [ws for ws in window_specs if ws.source == PD_SOURCE]
    else:
        # PD windows are excluded from synthetic variants.
        variant_window_specs = [ws for ws in window_specs if ws.source in CONTROL_LIKE_SOURCES]

    if not variant_window_specs:
        print(f"WARNING: no windows available for mode '{augment_mode}'. Skipping variant generation.")
        return

    if is_parkinson_mode:
        print("STEP 1/4: Parkinson-only mode (skip tremor cache)")
        tremor_cache = {}
    else:
        print("STEP 1/4: Building tremor cache")
        tremor_cache = build_tremor_cache_for_windows(
            window_specs=variant_window_specs,
            subject_data_cache=subject_data_cache,
            augment_mode=augment_mode,
        )

    print("STEP 2/4: Processing HAR sensors")
    aug_rng = np.random.default_rng(SEED)

    for sensor_name, cols in SENSORS.items():
        print(f"  Processing {sensor_name}")

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        subj_list: List[int] = []
        base_idx_list: List[int] = []
        tremor_freq_list: List[float] = []
        tremor_acc_rms_list: List[float] = []
        tremor_gyro_rms_list: List[float] = []
        tremor_score_list: List[int] = []

        sensor_type = get_sensor_type(sensor_name)
        signal_body_part = extract_body_part(sensor_name)
        if sensor_name in pk_config.TREMOR_FREE_SENSORS:
            severity_body_part = "arm"
        else:
            severity_body_part = signal_body_part

        for w_idx, ws in enumerate(variant_window_specs):
            data = subject_data_cache.get(ws.subject_id)
            if data is None:
                continue

            is_pd = ws.source == PD_SOURCE

            if is_pd:
                tremor_score_val = PD_TREMOR_SENTINEL_SCORE
            else:
                severity_cache_entry = tremor_cache.get((w_idx, severity_body_part))
                severity_meta = severity_cache_entry.get("meta", {}) if severity_cache_entry else {}
                tremor_acc_rms_target = severity_meta.get("acc_target_rms", 0.0)
                if GENERATE_TREMOR:
                    tremor_score_val = int(
                        severity_meta.get(
                            "sampled_score",
                            severity_meta.get("score", pk_config.get_tremor_score(tremor_acc_rms_target)),
                        )
                    )
                else:
                    tremor_score_val = 0

            sensor_available = ws.sensor_profile == "full" or sensor_name in PARKINSON_AVAILABLE_SENSORS
            if not sensor_available:
                tremor_score_missing = tremor_score_val
                if allowed_scores is not None and tremor_score_missing not in allowed_scores:
                    continue
                zero_win = np.zeros((target_window_len, len(cols)), dtype=np.float32)
                for _ in range(AUG_SIZE):
                    X_list.append(zero_win.T.copy())
                    y_list.append(ws.label - 1)
                    subj_list.append(ws.subject_id)
                    base_idx_list.append(w_idx)
                    tremor_freq_list.append(0.0)
                    tremor_acc_rms_list.append(0.0)
                    tremor_gyro_rms_list.append(0.0)
                    tremor_score_list.append(tremor_score_missing)
                continue

            win_raw = data[ws.win_start : ws.win_end, cols].copy()

            if is_pd:
                # PD windows are kept as real/raw windows (no synthetic tremor simulation).
                win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
                win = zscore_window(win_resampled)

                tremor_freq = 0.0
                tremor_acc_rms = 0.0
                tremor_gyro_rms = 0.0
                tremor_score = PD_TREMOR_SENTINEL_SCORE
            else:
                apply_awgn_aug = False
                apply_rotation_aug = False

                if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                    if tremor_score_val == 0:
                        pass
                    elif tremor_score_val in [1, 2]:
                        apply_awgn_aug = True
                    else:
                        if sensor_type == "acc":
                            apply_rotation_aug = True
                        else:
                            apply_awgn_aug = True
                else:
                    signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
                    if signal_cache_entry is not None:
                        if sensor_name in pk_config.ROTATION_BASED_MAG_SENSORS:
                            gyro_tremor = signal_cache_entry["gyro_noise"]
                            win_raw = apply_tremor_rotation_to_magnetometer(
                                mag_signal=win_raw,
                                gyro_tremor=gyro_tremor,
                                fs=ORIGINAL_FS,
                            )
                        elif sensor_type in ["acc", "gyro", "mag"]:
                            noise_key = f"{sensor_type}_noise"
                            win_raw += signal_cache_entry[noise_key]

                if apply_awgn_aug:
                    win_raw = apply_awgn_raw(win_raw, aug_rng, rms_ratio=0.15)
                if apply_rotation_aug:
                    win_raw = apply_rotation_augmentation(win_raw, aug_rng, max_angle_deg=10.0)

                win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
                win = zscore_window(win_resampled)

                if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                    tremor_freq = 0.0
                    tremor_acc_rms = 0.0
                    tremor_gyro_rms = 0.0
                    tremor_score = tremor_score_val
                else:
                    signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
                    if signal_cache_entry is not None and GENERATE_TREMOR:
                        meta = signal_cache_entry["meta"]
                        tremor_freq = float(meta.get("freq_hz", 0.0))
                        tremor_acc_rms = float(meta.get("acc_target_rms", 0.0))
                        tremor_gyro_rms = float(meta.get("gyro_target_rms", 0.0))
                        tremor_score = int(
                            meta.get(
                                "sampled_score",
                                meta.get("score", pk_config.get_tremor_score(tremor_acc_rms)),
                            )
                        )
                    else:
                        tremor_freq = 0.0
                        tremor_acc_rms = 0.0
                        tremor_gyro_rms = 0.0
                        tremor_score = 0

            if allowed_scores is not None and tremor_score not in allowed_scores:
                continue

            for _ in range(AUG_SIZE):
                if is_pd:
                    win_aug = win.copy()
                else:
                    win_aug = augment_window(win.copy(), aug_rng)
                X_list.append(win_aug.T.astype(np.float32))
                y_list.append(ws.label - 1)
                subj_list.append(ws.subject_id)
                base_idx_list.append(w_idx)
                tremor_freq_list.append(tremor_freq)
                tremor_acc_rms_list.append(tremor_acc_rms)
                tremor_gyro_rms_list.append(tremor_gyro_rms)
                tremor_score_list.append(tremor_score)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        subject_ids = np.array(subj_list, dtype=np.int64)
        base_window_idx = np.array(base_idx_list, dtype=np.int64)
        tremor_freq = np.array(tremor_freq_list, dtype=np.float32)
        tremor_acc_rms = np.array(tremor_acc_rms_list, dtype=np.float32)
        tremor_gyro_rms = np.array(tremor_gyro_rms_list, dtype=np.float32)
        tremor_score = np.array(tremor_score_list, dtype=np.int8)

        npz_out = variant_dir / f"{sensor_name}.npz"
        np.savez_compressed(
            npz_out,
            X=X,
            y=y,
            subject_id=subject_ids,
            base_window_idx=base_window_idx,
            tremor_freq=tremor_freq,
            tremor_acc_rms=tremor_acc_rms,
            tremor_gyro_rms=tremor_gyro_rms,
            tremor_score=tremor_score,
            fs=FS,
            window_len=target_window_len,
            stride=target_stride,
            sensor_name=sensor_name,
            sensor_cols=np.array(cols, dtype=np.int64),
        )

        txt_out = variant_dir / f"{sensor_name}.txt"
        flat_rows = flatten_channel_blocks(X)
        all_labels = np.column_stack(
            [
                (y + 1),
                subject_ids,
                base_window_idx,
                tremor_freq,
                tremor_acc_rms,
                tremor_gyro_rms,
                tremor_score,
            ]
        )
        sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
        np.savetxt(txt_out, sensor_txt, delimiter=",", fmt="%.6f")

        print(f"    Saved {sensor_name}: X={X.shape}")

    print("STEP 3/4: Processing tremor-branch sensors")
    tremor_branch_sensors = {
        "Acc_arm": SENSORS["Acc_arm"],
        "Gyro_arm": SENSORS["Gyro_arm"],
    }

    aug_rng_tremor = np.random.default_rng(SEED)
    for sensor_name, cols in tremor_branch_sensors.items():
        X_tremor_list: List[np.ndarray] = []
        y_tremor_list: List[int] = []
        subj_tremor_list: List[int] = []
        base_idx_tremor_list: List[int] = []
        tremor_freq_tremor_list: List[float] = []
        tremor_acc_rms_tremor_list: List[float] = []
        tremor_gyro_rms_tremor_list: List[float] = []
        tremor_score_tremor_list: List[int] = []

        sensor_type = get_sensor_type(sensor_name)
        signal_body_part = extract_body_part(sensor_name)
        severity_body_part = signal_body_part

        for w_idx, ws in enumerate(variant_window_specs):
            data = subject_data_cache.get(ws.subject_id)
            if data is None:
                continue

            is_pd = ws.source == PD_SOURCE

            sensor_available = ws.sensor_profile == "full" or sensor_name in PARKINSON_AVAILABLE_SENSORS
            if not sensor_available:
                continue

            win_raw = data[ws.win_start : ws.win_end, cols].copy()

            if is_pd:
                win_tremor = resample_window(win_raw, ORIGINAL_FS, FS)
                tremor_freq = 0.0
                tremor_acc_rms = 0.0
                tremor_gyro_rms = 0.0
                tremor_score = PD_TREMOR_SENTINEL_SCORE
            else:
                severity_cache_entry = tremor_cache.get((w_idx, severity_body_part))
                severity_meta = severity_cache_entry.get("meta", {}) if severity_cache_entry else {}
                tremor_acc_rms_target = severity_meta.get("acc_target_rms", 0.0)
                if GENERATE_TREMOR:
                    _ = int(
                        severity_meta.get(
                            "sampled_score",
                            severity_meta.get("score", pk_config.get_tremor_score(tremor_acc_rms_target)),
                        )
                    )

                signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
                if signal_cache_entry is not None and sensor_type in ["acc", "gyro"]:
                    noise_key = f"{sensor_type}_noise"
                    win_raw += signal_cache_entry[noise_key]

                win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
                win_tremor = win_resampled

                if signal_cache_entry is not None and GENERATE_TREMOR:
                    meta = signal_cache_entry["meta"]
                    tremor_freq = float(meta.get("freq_hz", 0.0))
                    tremor_acc_rms = float(meta.get("acc_target_rms", 0.0))
                    tremor_gyro_rms = float(meta.get("gyro_target_rms", 0.0))
                    tremor_score = int(
                        meta.get(
                            "sampled_score",
                            meta.get("score", pk_config.get_tremor_score(tremor_acc_rms)),
                        )
                    )
                else:
                    tremor_freq = 0.0
                    tremor_acc_rms = 0.0
                    tremor_gyro_rms = 0.0
                    tremor_score = 0

            if allowed_scores is not None and tremor_score not in allowed_scores:
                continue

            X_tremor_list.append(win_tremor.T.astype(np.float32))
            y_tremor_list.append(ws.label - 1)
            subj_tremor_list.append(ws.subject_id)
            base_idx_tremor_list.append(w_idx)
            tremor_freq_tremor_list.append(tremor_freq)
            tremor_acc_rms_tremor_list.append(tremor_acc_rms)
            tremor_gyro_rms_tremor_list.append(tremor_gyro_rms)
            tremor_score_tremor_list.append(tremor_score)

        X_tremor = np.stack(X_tremor_list, axis=0)
        y_tremor = np.array(y_tremor_list, dtype=np.int64)
        subject_ids_tremor = np.array(subj_tremor_list, dtype=np.int64)
        base_window_idx_tremor = np.array(base_idx_tremor_list, dtype=np.int64)
        tremor_freq_tremor = np.array(tremor_freq_tremor_list, dtype=np.float32)
        tremor_acc_rms_tremor = np.array(tremor_acc_rms_tremor_list, dtype=np.float32)
        tremor_gyro_rms_tremor = np.array(tremor_gyro_rms_tremor_list, dtype=np.float32)
        tremor_score_tremor = np.array(tremor_score_tremor_list, dtype=np.int8)

        save_tremor_branch_sensor(
            variant_dir=variant_dir,
            sensor_name=sensor_name,
            X=X_tremor,
            y=y_tremor,
            subject_ids=subject_ids_tremor,
            base_window_idx=base_window_idx_tremor,
            tremor_freq=tremor_freq_tremor,
            tremor_acc_rms=tremor_acc_rms_tremor,
            tremor_gyro_rms=tremor_gyro_rms_tremor,
            tremor_score=tremor_score_tremor,
            sensor_cols=cols,
            fs=FS,
            window_len=target_window_len,
            stride=target_stride,
        )

    print("STEP 4/4: Writing metadata/config files")
    write_window_source_map(variant_dir, variant_window_specs)
    write_sensor_coverage(variant_dir)

    source_counts = Counter(ws.source for ws in variant_window_specs)
    if is_parkinson_mode:
        method_cache_line = "  - Tremor cache generation: skipped (parkinson-only mode)"
        method_har_line = "  - HAR branch: raw -> resample -> zscore (no tremor simulation)"
        method_tremor_line = "  - Tremor branch: raw -> resample (no tremor simulation)"
        source_rule_line = "  - Sources included: PD only"
    else:
        method_cache_line = "  - Tremor cache generation: precompute_tremor_cache_with_parkinson_model"
        method_har_line = "  - HAR branch: raw -> tremor/aug -> resample -> zscore -> augment_window"
        method_tremor_line = "  - Tremor branch: raw -> tremor/aug -> resample (no zscore, no augment_window)"
        source_rule_line = "  - Sources included: mHealth + CT (PD excluded)"

    info_lines = [
        "=" * 80,
        "MERGED TREMOR DATASET CONFIGURATION",
        "=" * 80,
        "",
        f"Variant: {variant_name}",
        f"Generate tremor: {GENERATE_TREMOR}",
        f"Augment mode: {augment_mode}",
        "",
        "Sampling:",
        f"  Original FS: {ORIGINAL_FS}",
        f"  Target FS: {FS}",
        f"  Window: {WINDOW_SEC}s ({target_window_len} samples)",
        f"  Stride: {STRIDE_SEC}s ({target_stride} samples)",
        f"  AUG_SIZE: {AUG_SIZE}",
        f"  NOISE_LEVEL: {NOISE_LEVEL}",
        "",
        "Merge summary:",
        f"  mHealth windows: {source_counts.get('mhealth', 0)}",
        f"  CT windows: {source_counts.get('ct', 0)}",
        f"  PD windows: {source_counts.get('pd', 0)}",
        f"  Total base windows: {len(variant_window_specs)}",
        f"  Total HAR samples per sensor: {len(variant_window_specs) * AUG_SIZE}",
        f"  Tremor-branch samples per sensor: {len(variant_window_specs)}",
        "",
        "Method alignment:",
        method_cache_line,
        "  - Sensor policies: tremor_parkinson_config.py",
        method_har_line,
        method_tremor_line,
        source_rule_line,
        f"  - Output score filter: {sorted(allowed_scores) if allowed_scores is not None else 'none'}",
        "",
        "Merge-specific adaptation:",
        "  - Parkinson data mapped from Upper Right Cal1..Cal9 to arm sensors",
        "  - Missing sensors for Parkinson are left as zeros",
        "  - mHealth and CT windows use synthetic tremor augmentation policies",
        "  - PD windows are NOT tremor-simulated (raw window processing path)",
        f"  - PD windows use tremor_score={PD_TREMOR_SENTINEL_SCORE} sentinel",
        f"    where {PD_TREMOR_SENTINEL_SCORE} means real PD window with unknown clinical severity",
        "  - window_source_map.jsonl stores source per base window",
        "",
        "=" * 80,
    ]

    if is_parkinson_mode:
        info_lines.extend(
            [
                "",
                "Parkinson-only variant:",
                "  - Contains only real PD windows (source == 'pd')",
                "  - Contains no mHealth or CT windows",
                "  - No synthetic tremor simulation is applied",
                f"  - tremor_score={PD_TREMOR_SENTINEL_SCORE} is a sentinel for real PD window with unknown clinical severity",
            ]
        )

    (variant_dir / "info.txt").write_text("\n".join(info_lines), encoding="utf-8")

    if GENERATE_TREMOR and not is_parkinson_mode:
        params_file = variant_dir / "tremor_parkinson_params.txt"
        write_tremor_parkinson_params_file(
            output_path=params_file,
            mu=TREMOR_MU,
            sigma=TREMOR_SIGMA,
            dt=TREMOR_DT,
            intermittent=TREMOR_INTERMITTENT,
            on_prob=TREMOR_ON_PROB,
            min_on_sec=TREMOR_MIN_ON_SEC,
            max_on_sec=TREMOR_MAX_ON_SEC,
            use_jitter=USE_TREMOR_JITTER,
            jitter_std=TREMOR_JITTER_STD,
            scenario_seed=TREMOR_SEED,
        )

    print(f"Completed: {variant_name}")


def resolve_modes(requested_variant: str) -> List[str]:
    if requested_variant == "all":
        if GENERATE_TREMOR:
            return ["clean", "mild_mod", "mod_severe", "parkinson"]
        return ["clean", "parkinson"]

    if requested_variant == "severe":
        return ["mod_severe"]

    if not GENERATE_TREMOR and requested_variant in {"mild_mod", "mod_severe"}:
        return ["clean"]

    return [requested_variant]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate merged mHealth + Parkinson tremor datasets with DataGenerator_Tremor-aligned method")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "clean", "mild_mod", "mod_severe", "severe", "parkinson"],
        help="Variant mode to generate",
    )
    parser.add_argument("--exclude-ct", action="store_true", help="Exclude CT subjects from merge")
    parser.add_argument("--exclude-pd", action="store_true", help="Exclude PD subjects from merge")
    parser.add_argument("--dry-run", action="store_true", help="Only build and report merged windows, do not generate files")
    args = parser.parse_args()

    include_ct = not args.exclude_ct
    include_pd = not args.exclude_pd

    print("=" * 80)
    print("Merged Tremor Dataset Generator")
    print("=" * 80)
    print(f"MHEALTH_PATH: {MHEALTH_PATH}")
    print(f"PARKINSON_PATH: {PARKINSON_PATH}")
    print(f"OUT_BASE: {OUT_BASE}")
    print(f"include_ct={include_ct}, include_pd={include_pd}")
    print("=" * 80)

    ensure_parkinson_activity_beta_support()
    print("Parkinson activity labels (1-indexed):")
    for act_name, lbl in PARKINSON_ACTIVITY_TO_MHEALTH_LABEL.items():
        print(f"  {act_name}: {lbl}")
    print("=" * 80)

    random.seed(SEED)
    np.random.seed(SEED)

    subject_data_cache, window_specs, subject_meta = build_merged_subject_cache(
        include_ct=include_ct,
        include_pd=include_pd,
    )

    source_counts = Counter(ws.source for ws in window_specs)
    print(f"Subjects loaded: {len(subject_data_cache)}")
    print(f"Base windows: {len(window_specs)}")
    print(
        "Source windows: "
        f"mHealth={source_counts.get('mhealth', 0)}, "
        f"CT={source_counts.get('ct', 0)}, "
        f"PD={source_counts.get('pd', 0)}"
    )

    if args.dry_run:
        print("Dry-run enabled. No output files were generated.")
        return

    stride_str = _format_num(STRIDE_SEC)
    win_str = _format_num(WINDOW_SEC)
    fs_str = f"fs{_format_num(FS)}"
    base_name = f"s{stride_str}_w{win_str}_{fs_str}_tremor"

    modes = resolve_modes(args.variant)
    print(f"Modes to generate: {modes}")

    for mode in modes:
        variant_name = f"{base_name}_{mode}"
        generate_variant(
            variant_name=variant_name,
            augment_mode=mode,
            window_specs=window_specs,
            subject_data_cache=subject_data_cache,
        )

    print("=" * 80)
    print("All requested variants generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
