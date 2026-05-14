#!/usr/bin/env python3
"""
Tremor-labeled dataset generator for CT/PD Parkinson data only.

This script generates datasets with Parkinson tremor from CT (control) and PD
(Parkinson disease) subjects in Data_parkinson/, using the same preprocessing,
tremor simulation, and output format as DataGenerator_Tremor.py.

Pipeline:
  - CT subjects: Generate clean, mild_mod, mod_severe variants with synthetic tremor
  - PD subjects: Generate parkinson variant with real data (no tremor simulation)

Output structure:
  Data/Tremor_datagenerator_files/
    s{stride}_w{window}_fs{FS}_tremor_{mode}/
      *.npz, *.txt for each sensor
      window_source_map.jsonl (metadata)

Label conventions (matching DataGenerator_Tremor.py):
  - Activity labels: 1-indexed in NPZ/TXT files (0-indexed internally)
  - 4 activities: calibration (1), key (2), cardigan (3), toast (4)
  - Tremor scores: 0 (clean), 1-2 (mild), 3-4 (severe)
"""

import numpy as np
from pathlib import Path
import random
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import pandas as pd
from scipy.signal import resample

# Use local copies of tremor modules (Parkinson-only)
from Tremor import (
    precompute_tremor_cache_with_parkinson_model,
    write_tremor_parkinson_params_file,
    apply_tremor_rotation_to_magnetometer,
)
import tremor_parkinson_config as pk_config

# ============================================================
# CONFIG - Basic parameters
# ============================================================
ORIGINAL_FS = 50        # original sampling rate in Parkinson data (Hz)
FS = 50                 # target sampling rate (Hz)
WINDOW_SEC = 4.0        # window length in seconds
STRIDE_SEC = 4.0        # stride in seconds
AUG_SIZE = 2            # number of augmented copies per window
NOISE_LEVEL = 0.01      # augmentation noise level (NOT tremor)

# Dataset location
script_dir = Path(__file__).parent.resolve()
PARKINSON_PATH = script_dir / "Data" / "Data_parkinson"
OUT_BASE = script_dir / "Data" / "Tremor_datagenerator_files_new"

# Seeds
SEED = 0                        # Controls augmentation noise
TREMOR_SEED = 42                # Controls tremor generation

# ============================================================
# CONFIG - Tremor Generation
# ============================================================
# Set GENERATE_TREMOR = False for clean dataset (tremor labels = 0)
# Set GENERATE_TREMOR = True for Parkinson tremor with per-window labels
GENERATE_TREMOR = True

# Tremor parameters (from Tremor.py defaults, same as DataGenerator_Tremor.py)
TREMOR_MU = 1.0
TREMOR_SIGMA = 0.5
TREMOR_DT = 0.001
TREMOR_INTERMITTENT = False
TREMOR_ON_PROB = 0.5
TREMOR_MIN_ON_SEC = 2.0
TREMOR_MAX_ON_SEC = 8.0

# Jitter: adds window-to-window variability
USE_TREMOR_JITTER = True
TREMOR_JITTER_STD = 0.15  # 15% variability

# Generate an additional mixed tremor dataset where mod_severe is a tail
# component controlled by pk_config.SEVERE_TAIL_RATIO.
GENERATE_MIXED_TAIL_VARIANT = True

# ============================================================
# CONFIG - Diagnostics
# ============================================================
RUN_SANITY_CHECKS = False

# ============================================================
# CONFIG - Performance
# ============================================================
# PARALLEL_VARIANTS: run all CT variants concurrently with ThreadPoolExecutor.
# Each variant builds its own tremor cache in its own thread (read-only shared data).
# Gives ~2-4x wall-time speedup on multi-core servers.
PARALLEL_VARIANTS = True

# CACHE_XLS_AS_NPZ: on first load, write an NPZ sidecar next to each .xls file.
# Subsequent runs skip xlrd/pandas entirely and load from the fast NPZ.
# Delete <subject>/_xls_cache/ to force a full reload.
CACHE_XLS_AS_NPZ = True

# ============================================================
# SENSOR COLUMN MAP (matching DataGenerator_Tremor.py)
# ============================================================
# For Parkinson data: 5 sensor locations, each with arm-like sensor format
# Cal1..Cal3 → Acc, Cal4..Cal6 → Gyro, Cal7..Cal9 → Mag
# Sensor locations: Upper Right (UR), Lower Right (LR), Upper Left (UL), Lower Left (LL), Head

SENSOR_LOCATIONS = ["Upper Right", "Lower Right", "Upper Left", "Lower Left", "Head"]
LOCATION_CODES = {
    "Upper Right": "UR",
    "Lower Right": "LR",
    "Upper Left": "UL",
    "Lower Left": "LL",
    "Head": "head",
}

# Column indices in the 24-column matrix (all sensors use same column indices after loading)
SENSOR_COLS_24 = {
    "Acc": [14, 15, 16],   # Cal1-3
    "Gyro": [17, 18, 19],  # Cal4-6
    "Mag": [20, 21, 22],   # Cal7-9
}

# Generate sensor names for all locations
PARKINSON_SENSORS = {}
for location_name, location_code in LOCATION_CODES.items():
    for sensor_type, cols in SENSOR_COLS_24.items():
        sensor_name = f"{sensor_type}_{location_code}"
        PARKINSON_SENSORS[sensor_name] = cols

# Activity labels (1-indexed, matching mHealth convention)
PARKINSON_ACTIVITIES = {
    "calibration": 1,
    "key": 2,
    "cardigan": 3,
    "toast": 4,
}

# Sentinel tremor score for real PD windows (unknown clinical severity)
PD_TREMOR_SENTINEL_SCORE = 5


# ============================================================
# Helpers (same as DataGenerator_Tremor.py)
# ============================================================

def resample_window(win: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample window from original_fs to target_fs."""
    if original_fs == target_fs:
        return win

    L_original, C = win.shape
    L_target = int(round(L_original * target_fs / original_fs))

    resampled = np.zeros((L_target, C))
    for c in range(C):
        resampled[:, c] = resample(win[:, c], L_target)

    return resampled


def zscore_window(win: np.ndarray) -> np.ndarray:
    """Normalize per channel within window (z-score)."""
    mean = win.mean(axis=0, keepdims=True)
    std = win.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (win - mean) / std


def augment_window(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small augmentation noise to window."""
    noise = rng.normal(0, NOISE_LEVEL, size=win.shape)
    return win + noise


def generate_rotation_matrix(rng: np.random.Generator, max_angle_deg: float = 15.0) -> np.ndarray:
    """Generate a random 3D rotation matrix."""
    max_angle_rad = np.deg2rad(max_angle_deg)
    angles = rng.uniform(-max_angle_rad, max_angle_rad, size=3)
    theta_x, theta_y, theta_z = angles

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    return (Rz @ Ry @ Rx).astype(np.float32)


def apply_rotation_augmentation(win_raw: np.ndarray, rng: np.random.Generator, max_angle_deg: float = 15.0) -> np.ndarray:
    """Apply rotation-based augmentation to 3-axis sensor data."""
    R = generate_rotation_matrix(rng, max_angle_deg)
    return win_raw @ R.T


def apply_awgn_raw(win_raw: np.ndarray, rng: np.random.Generator, rms_ratio: float = 0.2) -> np.ndarray:
    """Apply AWGN to raw sensor signal."""
    L, C = win_raw.shape
    noise = np.zeros_like(win_raw)

    for c in range(C):
        signal_rms = np.sqrt(np.mean(win_raw[:, c]**2))
        if signal_rms < 1e-8:
            signal_rms = 1.0
        noise_std = rms_ratio * signal_rms
        noise[:, c] = rng.normal(0, noise_std, size=L)

    return win_raw + noise


def flatten_channel_blocks(X_c_l: np.ndarray) -> np.ndarray:
    """X_c_l: (N, C, L) -> (N, C*L) in channel-block format."""
    N, C, L = X_c_l.shape
    # C-order reshape: dims are (N, C, L) so flattening last two gives
    # [ch0_t0..ch0_tL, ch1_t0..ch1_tL, ...] — correct channel-block order.
    return X_c_l.reshape(N, C * L)


def extract_body_part(sensor_name: str) -> str:
    """Extract body part from sensor name for cache lookups.
    
    Returns "arm" for all sensors since cache is built with "arm" key.
    Individual location codes are extracted separately for scaling.
    """
    return "arm"  # Cache key - all Parkinson sensors use "arm"


def extract_location_code(sensor_name: str) -> str:
    """Extract location code from sensor name (e.g., 'Acc_UR' -> 'UR', 'Gyro_head' -> 'head')."""
    parts = sensor_name.split('_')
    if len(parts) > 1:
        return parts[-1]  # Return location code (UR, LR, UL, LL, head)
    return "ll"  # Default to lower left if parsing fails


def get_sensor_type(sensor_name: str) -> str:
    """Get sensor type: 'acc', 'gyro', or 'mag'."""
    if "Acc" in sensor_name:
        return "acc"
    elif "Gyro" in sensor_name:
        return "gyro"
    elif "Mag" in sensor_name:
        return "mag"
    return "unknown"


def _get_window_tremor_scales(sensor_name: str, tremor_score: int, activity_name: str) -> Tuple[float, float, float]:
    """Return (location_scale, activity_scale, total_scale) for one window."""
    location_code = extract_location_code(sensor_name)
    location_scale = float(pk_config.get_location_scale_for_score(int(tremor_score), location_code))
    activity_scale = float(pk_config.get_activity_beta_for_score(int(tremor_score), activity_name))
    return location_scale, activity_scale, location_scale * activity_scale


# ============================================================
# Data Loading and Parsing
# ============================================================

def _normalize_sheet_name(name: str) -> str:
    """Normalize sheet name to alphanumeric lowercase."""
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
    return "".join(cleaned)


def _infer_activity_name(file_stem: str) -> Optional[str]:
    """Infer activity name from file stem."""
    n = file_stem.lower()
    if "calibration" in n:
        return "calibration"
    if "key" in n or "door" in n:
        return "key"
    if "cardigan" in n:
        return "cardigan"
    if "toast" in n:
        return "toast"
    return None


def _infer_subject_source(subject_folder: str) -> Optional[str]:
    """Infer if subject is CT (control) or PD (Parkinson disease)."""
    name = subject_folder.upper()
    if name.startswith("CT"):
        return "ct"
    elif name.startswith("PD"):
        return "pd"
    return None


def _load_sensor_sheet(file_path: Path, sheet_name: str) -> Optional[np.ndarray]:
    """Load calibration data from specified sensor sheet in Excel file.
    
    Args:
        file_path: Path to .xls file
        sheet_name: Name of sensor sheet (e.g., "Upper Right", "Head")
        
    Returns:
        (T, 9) array with Cal1..Cal9 data, or None if sheet not found
    """
    try:
        xls = pd.ExcelFile(file_path)
        normalized_to_actual = {_normalize_sheet_name(s): s for s in xls.sheet_names}
        
        # Try to find the sheet
        sheet_key = _normalize_sheet_name(sheet_name)
        if sheet_key not in normalized_to_actual:
            return None
        
        actual_sheet = normalized_to_actual[sheet_key]
        df = pd.read_excel(xls, sheet_name=actual_sheet)
        
        if df.shape[1] < 10:
            return None
        
        # Extract Time + Cal1..Cal9
        numeric = df.iloc[:, 1:10].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(how="any")
        
        if numeric.empty:
            return None
        
        return numeric.to_numpy(dtype=np.float32)
        
    except Exception:
        return None


def _sheet_cache_key(sheet_name: str) -> str:
    """Numpy-safe savez key for a sensor location name."""
    return sheet_name.replace(" ", "_")


def _load_all_sheets(file_path: Path) -> Dict[str, Optional[np.ndarray]]:
    """Load all sensor sheets from one Excel file in a single open.

    When CACHE_XLS_AS_NPZ is True, a small NPZ sidecar is written next to the
    .xls file on the first read.  Every subsequent run loads from the NPZ,
    skipping xlrd/pandas entirely and giving a large speedup on NFS.
    Cache is invalidated automatically when the .xls is newer than its sidecar.
    """
    if CACHE_XLS_AS_NPZ:
        cache_dir  = file_path.parent / "_xls_cache"
        cache_path = cache_dir / f"{file_path.stem}.npz"
        try:
            if (
                cache_path.exists()
                and cache_path.stat().st_mtime >= file_path.stat().st_mtime
            ):
                npz = np.load(cache_path, allow_pickle=False)
                return {
                    s: npz[_sheet_cache_key(s)] if _sheet_cache_key(s) in npz.files else None
                    for s in SENSOR_LOCATIONS
                }
        except Exception:
            pass  # fall through to slow path

    # ---- Slow path: parse XLS with pandas ----
    result: Dict[str, Optional[np.ndarray]] = {}
    try:
        xls = pd.ExcelFile(file_path)
        normalized_to_actual = {_normalize_sheet_name(s): s for s in xls.sheet_names}
        for sheet_name in SENSOR_LOCATIONS:
            sheet_key = _normalize_sheet_name(sheet_name)
            if sheet_key not in normalized_to_actual:
                result[sheet_name] = None
                continue
            actual_sheet = normalized_to_actual[sheet_key]
            df = pd.read_excel(xls, sheet_name=actual_sheet)
            if df.shape[1] < 10:
                result[sheet_name] = None
                continue
            numeric = df.iloc[:, 1:10].apply(pd.to_numeric, errors="coerce").dropna(how="any")
            result[sheet_name] = None if numeric.empty else numeric.to_numpy(dtype=np.float32)
    except Exception:
        result = {s: None for s in SENSOR_LOCATIONS}

    # ---- Write NPZ sidecar for future runs (non-fatal on failure) ----
    if CACHE_XLS_AS_NPZ:
        try:
            cache_dir.mkdir(exist_ok=True)
            save_kwargs = {
                _sheet_cache_key(s): arr
                for s, arr in result.items()
                if arr is not None
            }
            if save_kwargs:
                np.savez(cache_path, **save_kwargs)
        except Exception:
            pass

    return result


# ============================================================
# Dataset Building
# ============================================================

def build_parkinson_dataset() -> Tuple[Dict[str, List], Dict, Dict]:
    """
    Load all CT and PD subjects from Data_parkinson, extracting all 5 sensor sheets.
    
    Returns:
        - window_specs: Dict with 'ct' and 'pd' keys, each containing list of window specs
        - subject_data: Dict mapping (source, subject_folder, sheet_name) -> data array
        - subject_meta: Dict mapping (source, subject_folder) -> metadata dict
    """
    window_specs = {"ct": [], "pd": []}
    subject_data = {}
    subject_meta = {}

    window_len = int(round(WINDOW_SEC * ORIGINAL_FS))
    stride = int(round(STRIDE_SEC * ORIGINAL_FS))
    stride = max(stride, 1)

    if not PARKINSON_PATH.exists():
        raise FileNotFoundError(f"Parkinson data path not found: {PARKINSON_PATH}")

    # Iterate through all patient folders
    for patient_dir in sorted(PARKINSON_PATH.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("._"):
            continue

        source = _infer_subject_source(patient_dir.name)
        if source is None:
            print(f"  Skip {patient_dir.name} (not CT or PD)")
            continue

        # Find Excel trial files
        trial_files = [
            p for p in sorted(patient_dir.iterdir())
            if p.is_file() and p.suffix.lower() in {".xls", ".xlsx"} and not p.name.startswith("._")
        ]

        if not trial_files:
            continue

        print(f"  {source.upper():2} {patient_dir.name:20} loading trials...")

        subject_rows = {sheet: [] for sheet in SENSOR_LOCATIONS}  # One list per location
        subject_windows = []
        cursor = {sheet: 0 for sheet in SENSOR_LOCATIONS}
        n_trials = 0

        # Preload all trial files in parallel — network FS benefits from
        # concurrent reads; each file is opened once for all 5 sheets.
        preloaded: Dict[Path, Dict[str, Optional[np.ndarray]]] = {}
        n_io_workers = min(8, max(1, len(trial_files)))
        if n_io_workers > 1:
            with ThreadPoolExecutor(max_workers=n_io_workers) as io_pool:
                future_to_path = {io_pool.submit(_load_all_sheets, f): f for f in trial_files}
                for fut in as_completed(future_to_path):
                    preloaded[future_to_path[fut]] = fut.result()
        else:
            for f in trial_files:
                preloaded[f] = _load_all_sheets(f)

        for trial_file in trial_files:
            activity_name = _infer_activity_name(trial_file.stem)
            if activity_name is None:
                continue

            activity_label = PARKINSON_ACTIVITIES.get(activity_name)
            if activity_label is None:
                continue

            # Use preloaded data (single file open for all 5 sheets)
            sheets_data = {
                sheet: arr
                for sheet, arr in preloaded[trial_file].items()
                if arr is not None
            }

            if not sheets_data:
                continue

            # Process each sensor sheet
            for sheet_name, cal9 in sheets_data.items():
                if cal9.shape[0] < window_len:
                    continue

                # Build 24-column matrix (matching mHealth format for compatibility)
                T = cal9.shape[0]
                trial_matrix = np.zeros((T, 24), dtype=np.float32)
                trial_matrix[:, 14:17] = cal9[:, 0:3]   # Acc (Cal1..Cal3)
                trial_matrix[:, 17:20] = cal9[:, 3:6]   # Gyro (Cal4..Cal6)
                trial_matrix[:, 20:23] = cal9[:, 6:9]   # Mag (Cal7..Cal9)
                trial_matrix[:, 23] = activity_label    # Label

                start_idx = cursor[sheet_name]
                end_idx = start_idx + trial_matrix.shape[0]
                subject_rows[sheet_name].append(trial_matrix)

                # Create windows from this trial
                for offset in range(0, trial_matrix.shape[0] - window_len + 1, stride):
                    subject_windows.append({
                        "source": source,
                        "subject_folder": patient_dir.name,
                        "sheet_name": sheet_name,
                        "activity_label": activity_label,
                        "activity_name": activity_name,
                        "win_start": start_idx + offset,
                        "win_end": start_idx + offset + window_len,
                    })

                cursor[sheet_name] = end_idx

                # Add separator
                sep = np.zeros((window_len, 24), dtype=np.float32)
                subject_rows[sheet_name].append(sep)
                cursor[sheet_name] += window_len

            n_trials += 1

        if not subject_windows:
            print(f"      No valid windows found")
            continue

        # Store data for each sheet
        for sheet_name in SENSOR_LOCATIONS:
            if subject_rows[sheet_name]:
                subject_data[(source, patient_dir.name, sheet_name)] = np.vstack(subject_rows[sheet_name])

        subject_meta[(source, patient_dir.name)] = {
            "source": source,
            "patient_folder": patient_dir.name,
            "n_trials": n_trials,
            "sheets": list(sheets_data.keys()),
        }

        window_specs[source].extend(subject_windows)
        print(f"      ✓ {len(subject_windows):4d} windows, {n_trials:2d} trials, {len(sheets_data)} sheets")

    return window_specs, subject_data, subject_meta


# ============================================================
# Tremor Cache and Processing
# ============================================================

def generate_variant(
    variant_name: str,
    augment_mode: str,
    window_specs: List[Dict],
    subject_data: Dict,
    is_pd_mode: bool = False,
) -> None:
    """
    Generate dataset variant (clean, mild_mod, mod_severe, mixed_tail, or parkinson).
    
    Args:
        variant_name: Output directory name
        augment_mode: "clean", "mild_mod", "mod_severe", "mixed_tail", or "parkinson"
        window_specs: List of window specification dicts
        subject_data: Dict mapping (source, patient_folder) -> data array
        is_pd_mode: True if generating parkinson variant (PD only, no tremor simulation)
    """
    print("=" * 80)
    print(f"Generating variant: {variant_name}")
    print(f"Mode: {augment_mode}, PD-only: {is_pd_mode}")
    print("=" * 80)

    variant_dir = OUT_BASE / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    target_window_len = int(round(WINDOW_SEC * FS))
    target_stride = int(round(STRIDE_SEC * FS))

    # Filter windows: CT for synthetic variants, PD for parkinson variant
    if is_pd_mode:
        filtered_specs = [ws for ws in window_specs if ws["source"] == "pd"]
    else:
        filtered_specs = [ws for ws in window_specs if ws["source"] == "ct"]

    if not filtered_specs:
        print(f"WARNING: No windows available for mode '{augment_mode}'. Skipping.")
        return

    print(f"Total windows: {len(filtered_specs)}")

    # -------- Create stable subject ID mapping --------
    # Map each unique subject folder to a deterministic numeric ID
    # CT subjects: 1, 2, 3, ...
    # PD subjects: 101, 102, 103, ...
    unique_subjects = sorted(set(ws["subject_folder"] for ws in filtered_specs))
    if is_pd_mode:
        subject_map = {subj: 100 + i for i, subj in enumerate(unique_subjects, 1)}
    else:
        subject_map = {subj: i for i, subj in enumerate(unique_subjects, 1)}

    print(f"Subject mapping: {subject_map}")
    
    # -------- Group windows by sheet --------
    # Create separate processing for each sensor location
    windows_by_sheet = {}
    for ws in filtered_specs:
        sheet = ws.get("sheet_name", "Upper Right")
        if sheet not in windows_by_sheet:
            windows_by_sheet[sheet] = []
        windows_by_sheet[sheet].append(ws)
    
    print(f"Sensor locations found: {list(windows_by_sheet.keys())}")

    # -------- Step 1: Build tremor cache (shared across all sheets) --------
    print("STEP 1: Building tremor cache...")
    
    if is_pd_mode:
        print("  PD-only mode (skip tremor cache)")
        tremor_cache = {}
    elif augment_mode == "clean" or not GENERATE_TREMOR:
        # Clean mode: every window gets tremor_score=0 and zero noise.
        # Skip the expensive Van der Pol tremor simulation entirely.
        # An empty cache causes all cache lookups to return None, which the
        # per-window code already handles by defaulting to tremor_score=0.
        print("  Clean/no-tremor mode — skipping tremor cache build")
        tremor_cache = {}
    else:
        # Build cache for all windows across all sheets (cache is per window, not per sheet)
        cache_specs = []
        for w_idx, ws in enumerate(filtered_specs):
            key = (ws["source"], ws["subject_folder"], ws["sheet_name"])
            subject_data_arr = subject_data.get(key)
            if subject_data_arr is not None:
                cache_specs.append((
                    w_idx,
                    ws["activity_label"],
                    ws["win_start"],
                    ws["win_end"],
                ))

        # Define data loader for tremor cache
        def data_loader_func(w_idx: int) -> np.ndarray:
            """Load data for window index."""
            key = (filtered_specs[w_idx]["source"], filtered_specs[w_idx]["subject_folder"], filtered_specs[w_idx]["sheet_name"])
            return subject_data[key]

        tremor_cache = precompute_tremor_cache_with_parkinson_model(
            window_specs=cache_specs,
            sensor_column_mapping={
                "Acc_arm": [14, 15, 16],
                "Gyro_arm": [17, 18, 19],
                "Mag_arm": [20, 21, 22],
            },
            data_loader_func=data_loader_func,
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

    # Map tremor scores to allowed scores for this variant
    allowed_tremor_scores = {
        "clean": {0},
        "mild_mod": {1, 2},
        "mod_severe": {3, 4},
        "mixed_tail": {1, 2, 3, 4},
        "parkinson": {PD_TREMOR_SENTINEL_SCORE},
    }
    allowed_scores = allowed_tremor_scores.get(augment_mode, None)

    # -------- Step 2: Process sensors for each location --------
    print("STEP 2: Processing sensors by location")

    aug_rng = np.random.default_rng(SEED)
    target_window_len = int(round(WINDOW_SEC * FS))
    
    # Build index mapping from window specs
    spec_to_global_idx = {id(ws): idx for idx, ws in enumerate(filtered_specs)}

    for sheet_name in SENSOR_LOCATIONS:
        sheet_specs = windows_by_sheet.get(sheet_name, [])
        if not sheet_specs:
            print(f"  {sheet_name}: No windows")
            continue
        
        location_code = LOCATION_CODES[sheet_name]
        print(f"\n  Location: {sheet_name} ({location_code})")
        
        # Get sensors for this location
        location_sensors = {
            name: cols for name, cols in PARKINSON_SENSORS.items()
            if name.endswith(f"_{location_code}")
        }
        
        for sensor_name, cols in location_sensors.items():
            print(f"    {sensor_name}")

            X_list = []
            y_list = []
            subj_list = []
            base_idx_list = []
            tremor_freq_list = []
            tremor_acc_rms_list = []
            tremor_gyro_rms_list = []
            tremor_score_list = []

            sensor_type = get_sensor_type(sensor_name)
            signal_body_part = extract_body_part(sensor_name)

            # Determine which body part to use for severity reference
            if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                severity_body_part = "arm"
            else:
                severity_body_part = signal_body_part

            # Process each window for this sheet
            for sheet_w_idx, ws in enumerate(sheet_specs):
                key = (ws["source"], ws["subject_folder"], ws["sheet_name"])
                data = subject_data.get(key)
                if data is None:
                    continue

                # Get global window index for tremor cache
                global_w_idx = spec_to_global_idx[id(ws)]

                is_pd = ws["source"] == "pd"

                # Extract raw window
                win_raw = data[ws["win_start"]: ws["win_end"], cols].copy()

                if is_pd:
                    # PD windows: no tremor simulation, just resample and normalize
                    win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
                    win = zscore_window(win_resampled)

                    tremor_freq = 0.0
                    tremor_acc_rms = 0.0
                    tremor_gyro_rms = 0.0
                    tremor_score = PD_TREMOR_SENTINEL_SCORE
                else:
                    # CT windows: apply tremor/augmentation policy
                    apply_awgn_aug = False
                    apply_rotation_aug = False

                    # Determine augmentation strategy based on tremor severity
                    if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                        # Tremor-free sensors: use alternative augmentation
                        severity_cache_entry = tremor_cache.get((global_w_idx, severity_body_part))
                        severity_meta = severity_cache_entry.get("meta", {}) if severity_cache_entry else {}
                        tremor_acc_rms_target = severity_meta.get("acc_target_rms", 0.0)
                        if GENERATE_TREMOR:
                            tremor_score_val = int(
                                severity_meta.get(
                                    "sampled_score",
                                    severity_meta.get("score", pk_config.get_tremor_score(tremor_acc_rms_target))
                                )
                            )
                        else:
                            tremor_score_val = 0

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
                        # Other sensors: apply tremor
                        signal_cache_entry = tremor_cache.get((global_w_idx, signal_body_part))
                        if signal_cache_entry is not None:
                            meta = signal_cache_entry.get("meta", {})
                            tremor_acc_rms_base = float(meta.get("acc_target_rms", 0.0))
                            tremor_score_val = int(
                                meta.get(
                                    "sampled_score",
                                    meta.get("score", pk_config.get_tremor_score(tremor_acc_rms_base))
                                )
                            )
                            _, _, total_scale = _get_window_tremor_scales(
                                sensor_name=sensor_name,
                                tremor_score=tremor_score_val,
                                activity_name=ws.get("activity_name", ""),
                            )
                            if sensor_name in pk_config.ROTATION_BASED_MAG_SENSORS:
                                gyro_tremor = signal_cache_entry["gyro_noise"] * total_scale
                                win_raw = apply_tremor_rotation_to_magnetometer(
                                    mag_signal=win_raw,
                                    gyro_tremor=gyro_tremor,
                                    fs=ORIGINAL_FS,
                                )
                            elif sensor_type in ["acc", "gyro", "mag"]:
                                noise_key = f"{sensor_type}_noise"
                                win_raw += signal_cache_entry[noise_key] * total_scale

                    # Apply augmentation
                    if apply_awgn_aug:
                        win_raw = apply_awgn_raw(win_raw, aug_rng, rms_ratio=0.15)
                    if apply_rotation_aug:
                        win_raw = apply_rotation_augmentation(win_raw, aug_rng, max_angle_deg=10.0)

                    # Resample and normalize
                    win_resampled = resample_window(win_raw, ORIGINAL_FS, FS)
                    win = zscore_window(win_resampled)

                    # Get tremor labels
                    if sensor_name in pk_config.TREMOR_FREE_SENSORS:
                        # Head sensors: no actual tremor, but inherit severity score from arm for filtering
                        tremor_freq = 0.0
                        tremor_acc_rms = 0.0
                        tremor_gyro_rms = 0.0
                        # Use severity score from arm cache for label consistency
                        severity_cache_entry = tremor_cache.get((global_w_idx, severity_body_part))
                        if severity_cache_entry is not None and GENERATE_TREMOR:
                            severity_meta = severity_cache_entry.get("meta", {})
                            tremor_score = int(
                                severity_meta.get(
                                    "sampled_score",
                                    severity_meta.get("score", 0)
                                )
                            )
                        else:
                            tremor_score = 0
                    else:
                        signal_cache_entry = tremor_cache.get((global_w_idx, signal_body_part))
                        if signal_cache_entry is not None and GENERATE_TREMOR:
                            meta = signal_cache_entry["meta"]
                            tremor_freq = meta.get("freq_hz", 0.0)
                            tremor_acc_rms_base = meta.get("acc_target_rms", 0.0)
                            tremor_score = int(
                                meta.get(
                                    "sampled_score",
                                    meta.get("score", pk_config.get_tremor_score(tremor_acc_rms_base))
                                )
                            )

                            # Apply location and activity scaling to align labels with
                            # the actual injected tremor amplitude for this window.
                            _, _, total_scale = _get_window_tremor_scales(
                                sensor_name=sensor_name,
                                tremor_score=tremor_score,
                                activity_name=ws.get("activity_name", ""),
                            )
                            tremor_acc_rms = tremor_acc_rms_base * total_scale
                            
                            # Recalculate gyro RMS from scaled accelerometer RMS
                            # Using the same relationship: gyro_rms = k_g(acc_rms) * acc_rms
                            if tremor_acc_rms > 0.0:
                                k_g_val = pk_config.choose_kg_from_rms_acc(tremor_acc_rms)
                                tremor_gyro_rms = k_g_val * tremor_acc_rms
                            else:
                                tremor_gyro_rms = 0.0
                        else:
                            tremor_freq = 0.0
                            tremor_acc_rms = 0.0
                            tremor_gyro_rms = 0.0
                            tremor_score = 0

                # Filter by allowed scores
                if allowed_scores is not None and tremor_score not in allowed_scores:
                    continue

                # Create augmented copies
                for _ in range(AUG_SIZE):
                    # For PD data: do NOT augment (preserve raw patient data)
                    # For CT data: apply artificial noise augmentation
                    if augment_mode == "parkinson":
                        win_aug = win.copy()  # No noise for real PD data
                    else:
                        win_aug = augment_window(win.copy(), aug_rng)  # Apply noise for synthetic variants
                    X_list.append(win_aug.T.astype(np.float32))
                    y_list.append(ws["activity_label"] - 1)  # 0-indexed
                    subj_list.append(subject_map[ws["subject_folder"]])  # Use stable subject ID mapping
                    base_idx_list.append(sheet_w_idx)
                    tremor_freq_list.append(tremor_freq)
                    tremor_acc_rms_list.append(tremor_acc_rms)
                    tremor_gyro_rms_list.append(tremor_gyro_rms)
                    tremor_score_list.append(tremor_score)

            # Stack and save
            if not X_list:
                print(f"      WARNING: No valid windows for {sensor_name}")
                continue

            X = np.stack(X_list, axis=0)
            y = np.array(y_list, dtype=np.int64)
            subject_ids = np.array(subj_list, dtype=np.int64)
            base_window_idx = np.array(base_idx_list, dtype=np.int64)
            tremor_freq = np.array(tremor_freq_list, dtype=np.float32)
            tremor_acc_rms = np.array(tremor_acc_rms_list, dtype=np.float32)
            tremor_gyro_rms = np.array(tremor_gyro_rms_list, dtype=np.float32)
            tremor_score = np.array(tremor_score_list, dtype=np.int8)

            # Save NPZ
            npz_out = variant_dir / f"{sensor_name}.npz"
            np.savez(
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

            # Save TXT
            txt_out = variant_dir / f"{sensor_name}.txt"
            flat_rows = flatten_channel_blocks(X)
            all_labels = np.column_stack([
                (y + 1),              # Activity label (1-indexed)
                subject_ids,
                base_window_idx,
                tremor_freq,
                tremor_acc_rms,
                tremor_gyro_rms,
                tremor_score
            ])
            sensor_txt = np.hstack([flat_rows, all_labels]).astype(np.float32)
            pd.DataFrame(sensor_txt).to_csv(txt_out, index=False, header=False, float_format="%.6f")

            print(f"      ✓ {sensor_name}: X={X.shape}")

    # Write metadata
    print("STEP 3: Writing metadata")
    write_window_source_map(variant_dir, filtered_specs)

    print(f"✓ Completed: {variant_name}\n")


def write_window_source_map(variant_dir: Path, window_specs: List[Dict]) -> None:
    """Write window source metadata to JSONL file."""
    out_jsonl = variant_dir / "window_source_map.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, ws in enumerate(window_specs):
            rec = {
                "base_window_idx": idx,
                "source": ws["source"],
                "patient_folder": ws["subject_folder"],
                "sheet_name": ws.get("sheet_name", "Unknown"),
                "activity_label": ws["activity_label"],
                "activity_name": ws["activity_name"],
            }
            f.write(json.dumps(rec) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Tremor Dataset Generator for CT/PD Parkinson Data Only")
    print("=" * 80)
    print(f"Parkinson data path: {PARKINSON_PATH}")
    print(f"Output base: {OUT_BASE}")
    print(f"FS: {ORIGINAL_FS} -> {FS} Hz, Window: {WINDOW_SEC}s, Stride: {STRIDE_SEC}s")
    print(f"Generate tremor: {GENERATE_TREMOR}")
    print("=" * 80)

    # Load data
    print("\nLoading Parkinson dataset...")
    window_specs, subject_data, subject_meta = build_parkinson_dataset()

    ct_count = len(window_specs["ct"])
    pd_count = len(window_specs["pd"])
    print(f"\n✓ Loaded: CT={ct_count} windows, PD={pd_count} windows")

    if ct_count == 0 and pd_count == 0:
        print("ERROR: No windows loaded!")
        return

    # Generate variants
    stride_str = str(int(STRIDE_SEC)) if STRIDE_SEC == int(STRIDE_SEC) else str(STRIDE_SEC)
    win_str = str(int(WINDOW_SEC)) if WINDOW_SEC == int(WINDOW_SEC) else str(WINDOW_SEC)
    fs_str = f"fs{int(FS)}"
    base_name = f"s{stride_str}_w{win_str}_{fs_str}_tremor"

    variants_to_generate = []

    if ct_count > 0:
        if GENERATE_TREMOR:
            variants_to_generate.extend([
                (f"{base_name}_clean", "clean", window_specs["ct"], False),
                (f"{base_name}_mild_mod", "mild_mod", window_specs["ct"], False),
                (f"{base_name}_mod_severe", "mod_severe", window_specs["ct"], False),
            ])
            if GENERATE_MIXED_TAIL_VARIANT:
                variants_to_generate.append(
                    (f"{base_name}_mixed_tail", "mixed_tail", window_specs["ct"], False)
                )
        else:
            variants_to_generate.append(
                (f"{base_name}_clean", "clean", window_specs["ct"], False)
            )

    if pd_count > 0:
        variants_to_generate.append(
            (f"{base_name}_parkinson", "parkinson", window_specs["pd"], True)
        )

    print(f"\nGenerating {len(variants_to_generate)} variants...")
    print("=" * 80)

    # Split CT and PD variants — CT variants are independent and can run in parallel.
    ct_variants = [(n, m, s, p) for n, m, s, p in variants_to_generate if not p]
    pd_variants  = [(n, m, s, p) for n, m, s, p in variants_to_generate if p]

    if PARALLEL_VARIANTS and len(ct_variants) > 1:
        print(f"\nRunning {len(ct_variants)} CT variants in parallel (ThreadPoolExecutor)...")
        print("  (per-variant progress lines may interleave — check run_logs/ for clean output)")
        with ThreadPoolExecutor(max_workers=len(ct_variants)) as pool:
            futs = {
                pool.submit(generate_variant, n, m, s, subject_data, p): n
                for n, m, s, p in ct_variants
            }
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    raise exc
    else:
        for n, m, s, p in ct_variants:
            generate_variant(n, m, s, subject_data, p)

    # PD variant runs after CT variants finish (usually just one)
    for n, m, s, p in pd_variants:
        generate_variant(n, m, s, subject_data, p)

    print("=" * 80)
    print("✓ All variants generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
