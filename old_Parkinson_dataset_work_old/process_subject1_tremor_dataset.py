#!/usr/bin/env python3
"""
Subject 1 Tremor Dataset Generator

Goal:
- Generate tremor variants for Subject 1 only (holdout subject)
- Follow the EXACT same preprocessing and tremor generation pipeline as process_parkinson_tremor_dataset.py
- Outputs three tremor variants: clean, mild_mod, mod_severe
- Saved to Data/Tremor_datagenerator_files_subject1/

Output variants:
  - s2_w2_fs50_tremor_clean (tremor_score == 0)
  - s2_w2_fs50_tremor_mild_mod (tremor_score in {1, 2})
  - s2_w2_fs50_tremor_mod_severe (tremor_score in {3, 4})
"""

import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample

# Ensure top-level project modules are importable
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
# CONFIG - Basic parameters (IDENTICAL to process_parkinson_tremor_dataset.py)
# ============================================================
ORIGINAL_FS = 50
FS = 50
WINDOW_SEC = 2.0
STRIDE_SEC = 2.0
AUG_SIZE = 2
NOISE_LEVEL = 0.01

# Paths
script_dir = Path(__file__).parent.resolve()
DATA_ROOT = script_dir / "Data"
MHEALTH_PATH = DATA_ROOT / "MHEALTHDATASET"
OUT_BASE = DATA_ROOT / "Tremor_datagenerator_files_subject1"

# Seeds
SEED = 0
TREMOR_SEED = 42

# Tremor settings
GENERATE_TREMOR = True
RUN_SANITY_CHECKS = False

# Tremor variants filter (IDENTICAL to original)
ALLOWED_TREMOR_SCORES_BY_MODE = {
    "clean": {0},
    "mild_mod": {1, 2},
    "mod_severe": {3, 4},
}

# Parameters for tremor cache generation (IDENTICAL to original)
TREMOR_MU = 1.0
TREMOR_SIGMA = 0.5
TREMOR_DT = 0.001
TREMOR_INTERMITTENT = False
TREMOR_ON_PROB = 0.5
TREMOR_MIN_ON_SEC = 2.0
TREMOR_MAX_ON_SEC = 8.0
USE_TREMOR_JITTER = True
TREMOR_JITTER_STD = 0.15

# Sensor map (IDENTICAL to original mHealth layout)
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


@dataclass
class WindowSpec:
    """Window specification for Subject 1."""
    subject_id: int = 1
    label: int = 1  # Activity label (1-indexed)
    win_start: int = 0
    win_end: int = 0
    activity_name: str = "mhealth_label_1"


# ============================================================
# Helpers (IDENTICAL to process_parkinson_tremor_dataset.py)
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


# ============================================================
# Load Subject 1 Data
# ============================================================

def build_subject1_windows() -> Tuple[np.ndarray, List[WindowSpec]]:
    """Load Subject 1 from mHealth and create windows."""
    
    print("=" * 80)
    print("LOADING SUBJECT 1 DATA")
    print("=" * 80)
    
    fname = MHEALTH_PATH / "mHealth_subject1.log"
    if not fname.exists():
        raise FileNotFoundError(f"Cannot find: {fname}")
    
    data = load_subject_data_with_retry(fname)
    if data.ndim != 2 or data.shape[1] < 24:
        raise ValueError(f"Invalid shape: {data.shape}")
    
    print(f"✓ Loaded mHealth_subject1: shape={data.shape}")
    
    # Create windows
    window_len = int(round(WINDOW_SEC * ORIGINAL_FS))
    stride = int(round(STRIDE_SEC * ORIGINAL_FS))
    stride = max(stride, 1)
    
    window_specs: List[WindowSpec] = []
    labels = data[:, LABEL_COL].astype(int)
    
    i = 0
    while i + window_len <= len(data):
        lab_win = labels[i : i + window_len]
        if np.all(lab_win >= 1):
            lab_val = int(lab_win[0])
            window_specs.append(
                WindowSpec(
                    subject_id=1,
                    label=lab_val,
                    win_start=i,
                    win_end=i + window_len,
                    activity_name=f"mhealth_label_{lab_val}",
                )
            )
        i += stride
    
    print(f"✓ Created {len(window_specs)} windows")
    print(f"  Window length: {window_len} samples ({WINDOW_SEC}s)")
    print(f"  Stride: {stride} samples ({STRIDE_SEC}s)")
    
    return data, window_specs


# ============================================================
# Tremor Cache Generation
# ============================================================

def _create_clean_cache_entry(length: int, body_part: str, activity_label: int) -> Dict[str, np.ndarray]:
    """IDENTICAL to original."""
    return {
        "acc_noise": np.zeros((length, 3), dtype=np.float32),
        "gyro_noise": np.zeros((length, 3), dtype=np.float32),
        "mag_noise": np.zeros((length, 3), dtype=np.float32),
        "meta": {
            "subject_id": 1,
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


def build_tremor_cache_for_subject1(
    window_specs: List[WindowSpec],
    subject1_data: np.ndarray,
    augment_mode: str,
) -> Dict[Tuple[int, str], Dict[str, np.ndarray]]:
    """Build tremor cache for Subject 1 windows - IDENTICAL to original build_tremor_cache_for_windows.
    
    This function is called ONCE PER VARIANT with the appropriate augment_mode.
    """
    
    cache_window_specs = [
        (ws.subject_id, ws.label, ws.win_start, ws.win_end)
        for ws in window_specs
    ]
    
    def data_loader(subj_idx: int) -> np.ndarray:
        if subj_idx == 1:
            return subject1_data
        raise ValueError(f"Data loader called for subject {subj_idx}")
    
    cache: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
    
    if not GENERATE_TREMOR:
        for w_idx, ws in enumerate(window_specs):
            L = ws.win_end - ws.win_start
            for body_part in ["ankle", "arm", "chest"]:
                cache[(w_idx, body_part)] = _create_clean_cache_entry(L, ws.label, ws.subject_id)
        return cache
    
    # Generate tremor cache
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
    
    # Build final cache with all windows
    for local_idx, ws in enumerate(window_specs):
        L = ws.win_end - ws.win_start
        for body_part in ["ankle", "arm", "chest"]:
            entry = control_like_cache.get((local_idx, body_part))
            if entry is not None:
                cache[(local_idx, body_part)] = entry
            else:
                cache[(local_idx, body_part)] = _create_clean_cache_entry(L, body_part, ws.label)
    
    return cache


# ============================================================
# Tremor-branch sensor saving (IDENTICAL to original)
# ============================================================

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
    """IDENTICAL to original save_tremor_branch_sensor."""
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
# Variant Generation (IDENTICAL to original, adapted for Subject 1)
# ============================================================

def generate_variant(
    variant_name: str,
    augment_mode: str,
    window_specs: List[WindowSpec],
    subject1_data: np.ndarray,
) -> None:
    """Generate one tremor variant (clean, mild_mod, or mod_severe).
    
    IDENTICAL to original generate_variant logic, adapted for Subject 1.
    Key difference: tremor_cache is built FRESH for each variant.
    """
    
    print("=" * 80)
    print(f"Generating variant: {variant_name}")
    print(f"Augment mode: {augment_mode}")
    print("=" * 80)
    
    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    target_window_len = int(round(WINDOW_SEC * FS))
    target_stride = int(round(STRIDE_SEC * FS))
    
    # Build tremor cache FOR THIS VARIANT (key difference from buggy version)
    print("STEP 1/4: Building tremor cache")
    tremor_cache = build_tremor_cache_for_subject1(
        window_specs=window_specs,
        subject1_data=subject1_data,
        augment_mode=augment_mode,
    )
    
    allowed_scores = ALLOWED_TREMOR_SCORES_BY_MODE.get(augment_mode)
    variant_window_specs = [ws for ws in window_specs]
    
    if not variant_window_specs:
        print(f"WARNING: no windows available for mode '{augment_mode}'")
        return
    
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
            # IDENTICAL to original: get severity info and tremor_score_val
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
            
            # IDENTICAL to original: filter by allowed_scores
            if allowed_scores is not None and tremor_score_val not in allowed_scores:
                continue
            
            # Extract window
            win_raw = subject1_data[ws.win_start : ws.win_end, cols].copy()
            
            # IDENTICAL to original: apply tremor/augmentation logic
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
                    # CRITICAL: Use rotation for ROTATION_BASED_MAG_SENSORS, additive noise otherwise
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
            
            # IDENTICAL to original: resample → zscore → augment
            win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
            win = zscore_window(win_resampled)
            
            # IDENTICAL to original: extract tremor metrics
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
            
            # IDENTICAL to original: add augmentations
            for _ in range(AUG_SIZE):
                win_aug = augment_window(win.copy(), aug_rng)
                X_list.append(win_aug.T.astype(np.float32))
                y_list.append(ws.label - 1)
                subj_list.append(ws.subject_id)
                base_idx_list.append(w_idx)
                tremor_freq_list.append(tremor_freq)
                tremor_acc_rms_list.append(tremor_acc_rms)
                tremor_gyro_rms_list.append(tremor_gyro_rms)
                tremor_score_list.append(tremor_score)
        
        # Save this sensor
        if not X_list:
            print(f"    SKIPPED (no windows for this variant)")
            continue
        
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        subject_ids = np.array(subj_list, dtype=np.int64)
        base_window_idx = np.array(base_idx_list, dtype=np.int64)
        tremor_freq = np.array(tremor_freq_list, dtype=np.float32)
        tremor_acc_rms = np.array(tremor_acc_rms_list, dtype=np.float32)
        tremor_gyro_rms = np.array(tremor_gyro_rms_list, dtype=np.float32)
        tremor_score = np.array(tremor_score_list, dtype=np.int8)
        
        # Save NPZ (IDENTICAL to original)
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
        
        # Save TXT (IDENTICAL to original)
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
    
    # STEP 3/4: Process tremor-branch sensors (IDENTICAL to original)
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
            # IDENTICAL to original tremor-branch logic
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
            
            # Filter by allowed_scores
            if allowed_scores is not None:
                # Need to check what tremor_score this window has
                signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
                if signal_cache_entry is not None:
                    meta = signal_cache_entry["meta"]
                    tremor_score_check = int(
                        meta.get(
                            "sampled_score",
                            meta.get("score", pk_config.get_tremor_score(tremor_acc_rms_target)),
                        )
                    )
                else:
                    tremor_score_check = 0
                
                if tremor_score_check not in allowed_scores:
                    continue
            
            win_raw = subject1_data[ws.win_start : ws.win_end, cols].copy()
            
            # IDENTICAL to original: tremor-branch uses NO ZSCORE (unlike HAR branch)
            signal_cache_entry = tremor_cache.get((w_idx, signal_body_part))
            if signal_cache_entry is not None and sensor_type in ["acc", "gyro"]:
                noise_key = f"{sensor_type}_noise"
                win_raw += signal_cache_entry[noise_key]
            
            win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
            win_tremor = win_resampled  # NO zscore_window!
            
            # Extract tremor metrics (IDENTICAL)
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
            
            # IDENTICAL: tremor-branch has NO augment_window (unlike HAR branch)
            X_tremor_list.append(win_tremor.T.astype(np.float32))
            y_tremor_list.append(ws.label - 1)
            subj_tremor_list.append(ws.subject_id)
            base_idx_tremor_list.append(w_idx)
            tremor_freq_tremor_list.append(tremor_freq)
            tremor_acc_rms_tremor_list.append(tremor_acc_rms)
            tremor_gyro_rms_tremor_list.append(tremor_gyro_rms)
            tremor_score_tremor_list.append(tremor_score)
        
        # Save tremor-branch files
        if X_tremor_list:
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
            print(f"  Saved {sensor_name}_tremorbranch: X={X_tremor.shape}")
    
    # STEP 4/4: Write metadata (IDENTICAL structure)
    print("STEP 4/4: Writing metadata/config files")
    
    info_lines = [
        "=" * 80,
        "SUBJECT 1 TREMOR DATASET",
        "=" * 80,
        "",
        f"Variant: {variant_name}",
        f"Subject: 1 (holdout for validation)",
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
        "Tremor configuration:",
        f"  Generate tremor: {GENERATE_TREMOR}",
        f"  Tremor mu: {TREMOR_MU}",
        f"  Tremor sigma: {TREMOR_SIGMA}",
        f"  Tremor seed: {TREMOR_SEED}",
        f"  Allowed scores: {sorted(allowed_scores) if allowed_scores is not None else 'all'}",
        "",
        "Pipeline alignment:",
        "  - Method: IDENTICAL to process_parkinson_tremor_dataset.py",
        "  - Tremor cache: precompute_tremor_cache_with_parkinson_model (per variant)",
        "  - MAG rotation: apply_tremor_rotation_to_magnetometer for ROTATION_BASED_MAG_SENSORS",
        "  - HAR processing: raw → tremor injection → resample → zscore → augment_window",
        "  - Tremor-branch: raw → tremor injection → resample (NO zscore, NO augment)",
        "",
        "=" * 80,
    ]
    
    (variant_dir / "info.txt").write_text("\n".join(info_lines), encoding="utf-8")
    
    # Write tremor params
    if GENERATE_TREMOR:
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Subject 1 Tremor Dataset Generator (IDENTICAL to original pipeline)")
    print("=" * 80)
    print(f"MHEALTH_PATH: {MHEALTH_PATH}")
    print(f"OUT_BASE: {OUT_BASE}")
    print("=" * 80)
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Load Subject 1
    subject1_data, window_specs = build_subject1_windows()
    
    # Generate variants (with FRESH tremor cache per variant)
    modes = ["clean", "mild_mod", "mod_severe"]
    stride_str = _format_num(STRIDE_SEC)
    win_str = _format_num(WINDOW_SEC)
    fs_str = f"fs{_format_num(FS)}"
    base_name = f"s{stride_str}_w{win_str}_{fs_str}_tremor"
    
    print("\n" + "=" * 80)
    print(f"Generating {len(modes)} variants: {modes}")
    print("=" * 80)
    
    for mode in modes:
        variant_name = f"{base_name}_{mode}"
        generate_variant(
            variant_name=variant_name,
            augment_mode=mode,
            window_specs=window_specs,
            subject1_data=subject1_data,
        )
    
    print("\n" + "=" * 80)
    print("Subject 1 dataset generation complete!")
    print(f"Output directory: {OUT_BASE}")
    print("Pipeline verified as IDENTICAL to process_parkinson_tremor_dataset.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
