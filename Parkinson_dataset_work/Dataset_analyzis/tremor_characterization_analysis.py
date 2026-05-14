#!/usr/bin/env python3
"""
Tremor Characterization & Simulation Validation Tool
======================================================

A diagnostic tool for validating and improving the tremor simulation model.
Compares three populations side-by-side:
  - CT:  real control subjects (noise floor reference)
  - PD:  real Parkinson patients (ground truth tremor)
  - SIM: simulated tremor applied to CT signals (what our model produces)

Analyses performed:
  1. RMS distributions (acc, gyro, per-axis)
  2. Frequency estimation via Welch PSD (dominant freq, spectral centroid,
     band power 3-7 Hz) -- configurable window lengths
  3. Tremor-presence detection per window (threshold on bandpass RMS)
  4. Tremor prevalence per subject / activity / dataset
  5. Activity-specific analysis (rest vs movement; resting vs action tremor)
  6. Left-right asymmetry per subject and activity
  7. Data-driven config suggestions (SCORE_RMS_RANGE, FREQ_RANGE_HZ, k_g)
  8. Quantified simulation match score per metric
  9. Side-by-side plots: histograms, boxplots, PSD overlays, asymmetry

Output directory: Dataset_analyzis/tremor_characterization/
  report.txt          - full text report
  summary.txt         - one-page diagnostic summary
  data.json           - machine-readable statistics
  suggestions.json    - data-driven config suggestions
  plots/              - all figures

Input:  Data_parkinson/{CT,PD}*/*.xls  (Upper Right/Left, Lower Right/Left, Head)
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Keep imports inside Parkinson_dataset_work only.
# ---------------------------------------------------------------------------
PARKINSON_ROOT = Path(__file__).parent.parent.resolve()
if str(PARKINSON_ROOT) not in sys.path:
    sys.path.insert(0, str(PARKINSON_ROOT))

import Tremor as _Tremor                          # noqa: E402
import tremor_parkinson_config as pk_config       # noqa: E402


# ===========================================================================
# SECTION 1 -- CONFIGURATION
# ===========================================================================

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
DATA_DIR     = SCRIPT_DIR.parent / "Data" / "Data_parkinson"
OUTPUT_DIR   = SCRIPT_DIR / "tremor_characterization"
PLOTS_DIR    = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Data -------------------------------------------------------------------
FS_ORIGINAL   = 50          # Hz -- nominal sampling rate of raw XLS data

# --- Analysis windows -------------------------------------------------------
WINDOW_LENGTHS_SEC = [4.0, 6.0, 8.0]   # multiple window lengths analyzed
PRIMARY_WINDOW_SEC = 4.0                # primary window for per-window stats

# --- Bandpass / tremor band -------------------------------------------------
TREMOR_LO_HZ   = 2.0
TREMOR_HI_HZ   = 12.0
TREMOR_BAND_LO = 3.0     # tighter detection band
TREMOR_BAND_HI = 7.0
BUTTERWORTH_ORDER = 4

# --- Tremor-presence threshold ----------------------------------------------
TREMOR_PRESENCE_THRESHOLD = 0.30   # m/s^2 on 3-7 Hz acc RMS

# --- Simulation -------------------------------------------------------------
SIM_N_WINDOWS  = 400
SIM_AUGMODES   = ["mild_mod"]   # multiple modes; balanced sampling
SIM_SEED       = 42

# --- SIM analysis filter ---------------------------------------------------
# Controls which generated simulation modes are included in ALL analysis,
# comparisons, plots, and reports.  Options per element: "mild_mod", "mod_severe"
# Examples:
#   ["mild_mod"]                -> analyze only mild simulated tremor
#   ["mod_severe"]              -> analyze only severe simulated tremor
#   ["mild_mod", "mod_severe"]  -> analyze both together (default)
SIM_ANALYSIS_MODES = ["mild_mod", "mod_severe"]

# --- Sensor layout ----------------------------------------------------------
SENSOR_SHEETS   = ["Upper Right", "Lower Right", "Upper Left", "Lower Left", "Head"]
LOCATION_CODE   = {"Upper Right":"UR","Lower Right":"LR","Upper Left":"UL",
                   "Lower Left":"LL","Head":"Hd"}
ACC_COLS  = slice(0, 3)
GYRO_COLS = slice(3, 6)

UPPER_LOCS  = ["Upper Right", "Upper Left"]
LOWER_LOCS  = ["Lower Right", "Lower Left"]
RIGHT_LOCS  = ["Upper Right", "Lower Right"]
LEFT_LOCS   = ["Upper Left",  "Lower Left"]

ACTIVITY_MAP = {"calibration":"calibration","calib":"calibration",
                "key":"key","cardigan":"cardigan","toast":"toast"}

# --- Current simulation config (sourced from active config module) ----------
SIM_SCORE_RMS_RANGE = dict(pk_config.SCORE_RMS_RANGE)
SIM_FREQ_RANGE_HZ = tuple(pk_config.FREQ_RANGE_HZ)


def _derive_current_kg_by_severity():
    """Reference k_g per severity derived from active choose_kg config."""
    out = {}
    score_labels = {
        1: "mild",
        2: "mild-moderate",
        3: "moderate-severe",
        4: "severe",
    }
    rng = np.random.default_rng(0)
    for score, label in score_labels.items():
        lo, hi = SIM_SCORE_RMS_RANGE.get(score, (float("nan"), float("nan")))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            out[label] = float("nan")
            continue
        rms_mid = 0.5 * (lo + hi)
        samples = [pk_config.choose_kg_from_rms_acc(rms_mid, rng=rng) for _ in range(32)]
        out[label] = float(np.median(samples))
    return out


SIM_KG_BY_SEVERITY = _derive_current_kg_by_severity()
SIM_KG_CONFIG_NOTE = "Derived from tremor_parkinson_config.choose_kg_from_rms_acc"

# ===========================================================================
# SECTION 2 -- DATA LOADING
# ===========================================================================

def _normalize(s):
    return s.strip().lower()

def _infer_activity(stem):
    s = stem.strip().lower()
    for k, v in ACTIVITY_MAP.items():
        if s.startswith(k):
            return v
    return None

def _infer_group(folder):
    n = folder.upper()
    if n.startswith("PD"): return "pd"
    if n.startswith("CT"): return "ct"
    return None

def _load_sheet(file_path, sheet_name):
    """Return (T, 9) float64 array [Cal1..Cal9] or None."""
    try:
        xls   = pd.ExcelFile(file_path)
        nmap  = {_normalize(s): s for s in xls.sheet_names}
        key   = _normalize(sheet_name)
        if key not in nmap:
            return None
        df = pd.read_excel(xls, sheet_name=nmap[key])
        if df.shape[1] < 10:
            return None
        num = df.iloc[:, 1:10].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num) < int(PRIMARY_WINDOW_SEC * FS_ORIGINAL):
            return None
        return num.to_numpy(dtype=np.float64)
    except Exception:
        return None

def load_all_trials(data_dir):
    """Walk data_dir, load every valid .xls trial."""
    records = []
    dirs = sorted(d for d in data_dir.iterdir()
                  if d.is_dir() and not d.name.startswith("._"))
    for pdir in dirs:
        group = _infer_group(pdir.name)
        if group is None:
            continue
        trials = sorted(p for p in pdir.iterdir()
                        if p.is_file() and p.suffix.lower() in {".xls",".xlsx"}
                        and not p.name.startswith("._"))
        for tf in trials:
            act = _infer_activity(tf.stem)
            if act is None:
                continue
            for sheet in SENSOR_SHEETS:
                data = _load_sheet(tf, sheet)
                if data is None:
                    continue
                records.append({"subject":pdir.name, "group":group,
                                "activity":act, "location":sheet, "data":data})
    print(f"  Loaded {len(records)} records from {len(dirs)} subject folders.")
    return records


# ===========================================================================
# SECTION 3 -- SIMULATION
# ===========================================================================

def _sample_ct_windows(records, n, rng, win_samples):
    ct_recs = [r for r in records if r["group"] == "ct"
               and r["location"] in UPPER_LOCS + LOWER_LOCS]
    windows = []
    for rec in ct_recs:
        acc  = rec["data"][:, ACC_COLS]
        gyro = rec["data"][:, GYRO_COLS]
        n_w  = len(acc) // win_samples
        for w in range(n_w):
            s, e = w*win_samples, (w+1)*win_samples
            windows.append({"acc":acc[s:e].copy(), "gyro":gyro[s:e].copy(),
                             "location":rec["location"], "activity":rec["activity"],
                             "subject":rec["subject"]})
    if len(windows) == 0:
        return []
    idx = rng.choice(len(windows), size=min(n, len(windows)), replace=False)
    return [windows[i] for i in idx]


def generate_simulated_windows(records):
    """Apply tremor simulation to CT windows, one batch per SIM_AUGMODES entry."""
    rng = np.random.default_rng(SIM_SEED)
    win_samples = int(round(PRIMARY_WINDOW_SEC * FS_ORIGINAL))

    severe_ratio = float(getattr(pk_config, "SEVERE_TAIL_RATIO", 0.144))
    modes = list(SIM_AUGMODES)
    if set(modes) == {"mild_mod", "mod_severe"}:
        n_sev = int(round(max(0.0, min(1.0, severe_ratio)) * SIM_N_WINDOWS))
        mode_counts = {
            "mod_severe": max(0, n_sev),
            "mild_mod": max(0, SIM_N_WINDOWS - n_sev),
        }
    else:
        n_per_mode = max(1, SIM_N_WINDOWS // max(1, len(modes)))
        mode_counts = {m: n_per_mode for m in modes}

    sim_records = []
    for mode in modes:
        n_mode = int(mode_counts.get(mode, 0))
        if n_mode <= 0:
            print(f"  Mode '{mode}': skipped (0 windows requested).")
            continue
        ct_wins = _sample_ct_windows(records, n_mode, rng, win_samples)
        if not ct_wins:
            print(f"  WARNING: No CT windows available for mode '{mode}'.")
            continue
        count = 0
        for w in ct_wins:
            acc  = w["acc"].astype(np.float64)
            gyro = w["gyro"].astype(np.float64)
            mag  = np.zeros_like(acc)

            score, acc_rms_base, freq_hz = pk_config.sample_tremor_params(mode, rng)
            activity_scale = float(pk_config.get_activity_beta_for_score(score, w.get("activity")))
            location_scale = float(pk_config.get_location_scale_for_score(score, w.get("location")))
            total_scale = activity_scale * location_scale
            acc_rms = max(0.0, float(acc_rms_base) * total_scale)
            k_g = pk_config.choose_kg_from_rms_acc(acc_rms)
            gyro_rms_target = k_g * acc_rms

            acc_t, gyro_t, _, _ = _Tremor.simulate_and_add_tremor_imu(
                X_acc=acc, X_gyro=gyro, X_mag=mag,
                fs=float(FS_ORIGINAL),
                freq_hz=freq_hz,
                acc_rms=acc_rms,
                gyro_rms=gyro_rms_target,
                mag_rms=0.0,
                seed=int(rng.integers(0, 2**31)),
            )
            sim_records.append({"subject": w["subject"]+"_sim",
                                 "group":   "sim",
                                 "_sim_augmode": mode,
                                 "activity": w["activity"],
                                 "location": w["location"],
                                 "data_acc":  acc_t,
                                 "data_gyro": gyro_t,
                                 "_sim_acc_rms_base": acc_rms_base,
                                 "_sim_acc_rms_injected": acc_rms,
                                 "_sim_gyro_rms_injected": gyro_rms_target,
                                 "_sim_activity_scale": activity_scale,
                                 "_sim_location_scale": location_scale,
                                 "_sim_total_scale": total_scale,
                                 "_sim_freq_injected": freq_hz,
                                 "_sim_score": score})
            count += 1
        print(f"  Mode '{mode}': {count} windows generated.")
    print(f"  Total simulated: {len(sim_records)} windows "
            f"(modes: {', '.join(modes)}, severe_tail_ratio={severe_ratio:.3f}, seed={SIM_SEED})")
    return sim_records


# ===========================================================================
# SECTION 4 -- SIGNAL PROCESSING
# ===========================================================================

def _bp_filter(sig, lo, hi, fs, order=BUTTERWORTH_ORDER):
    """Zero-phase Butterworth bandpass on (T,) or (T,C)."""
    nyq = 0.5 * fs
    b, a = butter(order, [max(lo/nyq,1e-4), min(hi/nyq,1-1e-4)], btype="band")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sig.ndim == 1:
            return filtfilt(b, a, sig)
        return np.column_stack([filtfilt(b, a, sig[:,c]) for c in range(sig.shape[1])])


def _rms_3d(arr):
    return float(np.sqrt(np.mean(np.sum(arr**2, axis=1))))

def _rms_per_axis(arr):
    return np.sqrt(np.mean(arr**2, axis=0))


def _welch_features(sig, fs,
                    lo=TREMOR_LO_HZ, hi=TREMOR_HI_HZ,
                    band_lo=TREMOR_BAND_LO, band_hi=TREMOR_BAND_HI,
                    nperseg=None):
    """Compute Welch PSD features. Returns dict with dom_freq, spectral_cent,
    band_power, total_power, band_ratio."""
    if sig.ndim > 1:
        sig = np.sqrt(np.sum(sig**2, axis=1))
    n = len(sig)
    if nperseg is None:
        nperseg = min(n, max(32, n // 2))
    f, pxx = welch(sig, fs=fs, nperseg=nperseg)

    mask_full = (f >= lo) & (f <= hi)
    mask_band = (f >= band_lo) & (f <= band_hi)

    if not np.any(mask_full):
        return dict(dom_freq=float("nan"), spectral_cent=float("nan"),
                    band_power=float("nan"), total_power=float("nan"),
                    band_ratio=float("nan"))

    pxx_f = pxx[mask_full]; f_f = f[mask_full]
    dom_freq     = float(f_f[np.argmax(pxx_f)])
    total_power  = float(np.trapezoid(pxx_f, f_f))
    spectral_cent = (float(np.sum(f_f * pxx_f) / np.sum(pxx_f))
                     if np.sum(pxx_f) > 0 else float("nan"))
    band_power   = (float(np.trapezoid(pxx[mask_band], f[mask_band]))
                    if np.any(mask_band) else 0.0)
    band_ratio   = band_power / total_power if total_power > 0 else float("nan")

    return dict(dom_freq=dom_freq, spectral_cent=spectral_cent,
                band_power=band_power, total_power=total_power,
                band_ratio=band_ratio)


def _welch_psd_curve(sig, fs, nperseg=None):
    if sig.ndim > 1:
        sig = np.sqrt(np.sum(sig**2, axis=1))
    n = len(sig)
    if nperseg is None:
        nperseg = min(n, max(32, n // 2))
    return welch(sig, fs=fs, nperseg=nperseg)


# ===========================================================================
# SECTION 5 -- PER-WINDOW ANALYSIS
# ===========================================================================

def _analyze_window_array(acc_bp, gyro_bp, acc_raw, fs):
    acc_rms  = _rms_3d(acc_bp)
    gyro_rms = _rms_3d(gyro_bp)
    kg_ratio = gyro_rms / acc_rms if acc_rms > 1e-9 else float("nan")

    acc_bp37 = _bp_filter(acc_raw, TREMOR_BAND_LO, TREMOR_BAND_HI, fs)
    acc_rms_band37 = _rms_3d(acc_bp37)
    tremor_present = bool(acc_rms_band37 > TREMOR_PRESENCE_THRESHOLD)

    wf = _welch_features(acc_bp, fs)
    return dict(
        acc_rms=acc_rms, gyro_rms=gyro_rms, kg_ratio=kg_ratio,
        acc_rms_band37=acc_rms_band37,
        tremor_present=tremor_present,
        dom_freq=wf["dom_freq"],
        spectral_cent=wf["spectral_cent"],
        band_power=wf["band_power"],
        band_ratio=wf["band_ratio"],
        acc_rms_axes=_rms_per_axis(acc_bp),
        gyro_rms_axes=_rms_per_axis(gyro_bp),
    )


def analyze_windows(records, sim_windows=None):
    """Slide primary window over records; ingest sim_windows as-is."""
    win = int(round(PRIMARY_WINDOW_SEC * FS_ORIGINAL))
    results = []

    for rec in records:
        acc_full  = rec["data"][:, ACC_COLS]
        gyro_full = rec["data"][:, GYRO_COLS]
        if len(acc_full) < win:
            continue
        acc_bp  = _bp_filter(acc_full,  TREMOR_LO_HZ, TREMOR_HI_HZ, FS_ORIGINAL)
        gyro_bp = _bp_filter(gyro_full, TREMOR_LO_HZ, TREMOR_HI_HZ, FS_ORIGINAL)

        n_w = len(acc_bp) // win
        for w in range(n_w):
            s, e = w*win, (w+1)*win
            feat = _analyze_window_array(acc_bp[s:e], gyro_bp[s:e],
                                         acc_full[s:e], FS_ORIGINAL)
            results.append({**feat,
                             "subject":  rec["subject"],
                             "group":    rec["group"],
                             "activity": rec["activity"],
                             "location": rec["location"]})

    if sim_windows:
        for sw in sim_windows:
            acc  = sw["data_acc"].astype(np.float64)
            gyro = sw["data_gyro"].astype(np.float64)
            acc_bp  = _bp_filter(acc,  TREMOR_LO_HZ, TREMOR_HI_HZ, FS_ORIGINAL)
            gyro_bp = _bp_filter(gyro, TREMOR_LO_HZ, TREMOR_HI_HZ, FS_ORIGINAL)
            feat = _analyze_window_array(acc_bp, gyro_bp, acc, FS_ORIGINAL)
            results.append({**feat,
                             "subject":  sw["subject"],
                             "group":    "sim",
                             "_sim_augmode": sw.get("_sim_augmode", "unknown"),
                             "activity": sw["activity"],
                             "location": sw["location"],
                             "_sim_acc_rms_injected": sw.get("_sim_acc_rms_injected"),
                             "_sim_freq_injected":    sw.get("_sim_freq_injected")})

    return results


# ===========================================================================
# SECTION 5b -- MULTI-WINDOW FREQUENCY ANALYSIS
# ===========================================================================

def analyze_frequency_by_window_length(records):
    """For each window length, compute Welch dominant frequency for PD and CT."""
    out = {}
    for wl in WINDOW_LENGTHS_SEC:
        win = int(round(wl * FS_ORIGINAL))
        pd_freqs, ct_freqs = [], []
        for rec in records:
            acc_full = rec["data"][:, ACC_COLS]
            if len(acc_full) < win:
                continue
            acc_bp = _bp_filter(acc_full, TREMOR_LO_HZ, TREMOR_HI_HZ, FS_ORIGINAL)
            n_w = len(acc_bp) // win
            freq_list = pd_freqs if rec["group"] == "pd" else ct_freqs
            for w in range(n_w):
                seg = acc_bp[w*win:(w+1)*win]
                wf  = _welch_features(seg, FS_ORIGINAL, nperseg=win)
                freq_list.append(wf["dom_freq"])
        out[wl] = {
            "pd": _stats(np.array(pd_freqs)),
            "ct": _stats(np.array(ct_freqs)),
        }
    return out


# ===========================================================================
# SECTION 6 -- STATISTICAL AGGREGATION
# ===========================================================================

def _stats(values):
    v = values[np.isfinite(values)] if hasattr(values, "__len__") else values
    if len(v) == 0:
        return {"n":0,"mean":float("nan"),"std":float("nan"),
                "p5":float("nan"),"p25":float("nan"),"median":float("nan"),
                "p75":float("nan"),"p95":float("nan")}
    return {"n":int(len(v)),"mean":float(np.mean(v)),"std":float(np.std(v)),
            "p5":float(np.percentile(v,5)),"p25":float(np.percentile(v,25)),
            "median":float(np.median(v)),"p75":float(np.percentile(v,75)),
            "p95":float(np.percentile(v,95))}


SCALAR_METRICS = ("acc_rms","gyro_rms","kg_ratio","dom_freq","spectral_cent",
                  "band_power","band_ratio","acc_rms_band37")


def _top_bottom_n(vals: np.ndarray, n: int) -> dict:
    """Return mean (and count used) for the top-n and bottom-n values."""
    v = vals[np.isfinite(vals) & (vals > 0)]
    if len(v) == 0:
        return {"n_total": 0,
                "top_n_used": 0,  "top_n_mean": float("nan"),
                "bottom_n_used": 0, "bottom_n_mean": float("nan")}
    k = min(n, len(v))
    sorted_desc = np.sort(v)[::-1]
    sorted_asc  = np.sort(v)
    return {
        "n_total":       int(len(v)),
        "top_n_used":    int(k),
        "top_n_mean":    float(np.mean(sorted_desc[:k])),
        "bottom_n_used": int(k),
        "bottom_n_mean": float(np.mean(sorted_asc[:k])),
    }


def _group_subset(windows, group):
    return [w for w in windows if w["group"] == group]

def _extract(subset, metric):
    return np.array([w[metric] for w in subset], dtype=float)


def aggregate_results(windows):
    groups = {g: _group_subset(windows, g) for g in ("pd","ct","sim")}

    by_group = {}
    for g, sub in groups.items():
        if not sub: continue
        by_group[g] = {m: _stats(_extract(sub, m)) for m in SCALAR_METRICS}
        tp = np.array([w["tremor_present"] for w in sub])
        by_group[g]["tremor_prevalence"] = float(np.mean(tp))

    by_location = {}
    for g, sub in groups.items():
        if not sub: continue
        by_location[g] = {}
        for loc in SENSOR_SHEETS:
            s = [w for w in sub if w["location"] == loc]
            if not s: continue
            by_location[g][loc] = {m: _stats(_extract(s,m)) for m in ("acc_rms","gyro_rms","dom_freq")}
            by_location[g][loc]["tremor_prevalence"] = float(np.mean([w["tremor_present"] for w in s]))

    activities = sorted({w["activity"] for w in windows})
    by_activity = {}
    for g, sub in groups.items():
        if not sub: continue
        by_activity[g] = {}
        for act in activities:
            s = [w for w in sub if w["activity"] == act]
            if not s: continue
            by_activity[g][act] = {m: _stats(_extract(s,m)) for m in ("acc_rms","gyro_rms","dom_freq")}
            by_activity[g][act]["tremor_prevalence"] = float(np.mean([w["tremor_present"] for w in s]))

    pd_arm = [w for w in groups.get("pd",[])
              if w["location"] in UPPER_LOCS + LOWER_LOCS]
    subjects = sorted({w["subject"] for w in pd_arm})
    per_subject = []
    for subj in subjects:
        sw = [w for w in pd_arm if w["subject"] == subj]
        if not sw: continue
        per_subject.append({
            "subject":    subj,
            "n_windows":  len(sw),
            "acc_rms_med":  float(np.nanmedian([w["acc_rms"]  for w in sw])),
            "gyro_rms_med": float(np.nanmedian([w["gyro_rms"] for w in sw])),
            "kg_ratio_med": float(np.nanmedian([w["kg_ratio"] for w in sw])),
            "dom_freq_med": float(np.nanmedian([w["dom_freq"] for w in sw])),
            "tremor_prev":  float(np.mean([w["tremor_present"] for w in sw])),
        })

    pd_all = groups.get("pd", [])
    upper_acc  = _extract([w for w in pd_all if w["location"] in UPPER_LOCS], "acc_rms")
    lower_acc  = _extract([w for w in pd_all if w["location"] in LOWER_LOCS], "acc_rms")
    upper_gyro = _extract([w for w in pd_all if w["location"] in UPPER_LOCS], "gyro_rms")
    lower_gyro = _extract([w for w in pd_all if w["location"] in LOWER_LOCS], "gyro_rms")
    upper_vs_lower = {
        "mean_acc_upper":  float(np.nanmean(upper_acc)),
        "mean_acc_lower":  float(np.nanmean(lower_acc)),
        "ratio_acc":       float(np.nanmean(lower_acc)/np.nanmean(upper_acc)) if np.nanmean(upper_acc) > 0 else float("nan"),
        "mean_gyro_upper": float(np.nanmean(upper_gyro)),
        "mean_gyro_lower": float(np.nanmean(lower_gyro)),
        "ratio_gyro":      float(np.nanmean(lower_gyro)/np.nanmean(upper_gyro)) if np.nanmean(upper_gyro) > 0 else float("nan"),
    }

    lr_asym_per_subject = _compute_lr_asymmetry(windows)

    return {
        "by_group":       by_group,
        "by_location":    by_location,
        "by_activity":    by_activity,
        "per_subject":    per_subject,
        "upper_vs_lower": upper_vs_lower,
        "lr_asymmetry":   lr_asym_per_subject,
        "activities":     activities,
    }


def _compute_lr_asymmetry(windows):
    """Compute left-right acc RMS asymmetry per PD subject per activity."""
    results = []
    pd_wins = _group_subset(windows, "pd")
    subjects   = sorted({w["subject"] for w in pd_wins})
    activities = sorted({w["activity"] for w in pd_wins})

    for subj in subjects:
        for act in activities:
            sw = [w for w in pd_wins if w["subject"]==subj and w["activity"]==act]
            if not sw: continue

            ur = [w for w in sw if w["location"]=="Upper Right"]
            ul = [w for w in sw if w["location"]=="Upper Left"]
            r_acc = float(np.nanmedian(_extract(ur,"acc_rms"))) if ur else float("nan")
            l_acc = float(np.nanmedian(_extract(ul,"acc_rms"))) if ul else float("nan")
            asym_upper = abs(l_acc-r_acc)/(l_acc+r_acc+1e-9) if np.isfinite(l_acc+r_acc) else float("nan")

            lr_ = [w for w in sw if w["location"]=="Lower Right"]
            ll_ = [w for w in sw if w["location"]=="Lower Left"]
            r_low = float(np.nanmedian(_extract(lr_,"acc_rms"))) if lr_ else float("nan")
            l_low = float(np.nanmedian(_extract(ll_,"acc_rms"))) if ll_ else float("nan")
            asym_lower = abs(l_low-r_low)/(l_low+r_low+1e-9) if np.isfinite(l_low+r_low) else float("nan")

            results.append({"subject":subj,"activity":act,
                             "asym_upper_arm":asym_upper, "asym_lower_arm":asym_lower,
                             "R_upper":r_acc,"L_upper":l_acc,
                             "R_lower":r_low,"L_lower":l_low})
    return results


def tremor_only_freq_analysis(windows):
    """Frequency stats from tremor-present windows only (arm locations).
    Uses already-computed dom_freq / spectral_cent / band_power -- no re-filtering.
    Returns stats dicts for PD and SIM."""
    result = {}
    for g in ("pd", "sim"):
        sub = [w for w in _group_subset(windows, g)
               if w["tremor_present"]
               and w["location"] in UPPER_LOCS]   # arm only
        if not sub:
            # fall back to all arm locations if no tremor-present found
            sub = [w for w in _group_subset(windows, g)
                   if w["location"] in UPPER_LOCS]
        result[g] = {
            "n":              len(sub),
            "dom_freq":       _stats(_extract(sub, "dom_freq")),
            "spectral_cent":  _stats(_extract(sub, "spectral_cent")),
            "band_power":     _stats(_extract(sub, "band_power")),
            "acc_rms":        _stats(_extract(sub, "acc_rms")),
            "gyro_rms":       _stats(_extract(sub, "gyro_rms")),
            "kg_ratio":       _stats(_extract(sub, "kg_ratio")),
        }
    return result


def pd_upper_end_rms_stats(windows):
    """Upper-end RMS statistics from tremor-present PD arm windows.

    Goal: determine whether the PD dataset actually contains strong/severe tremor
    (high-amplitude windows) or is predominantly mild/moderate.

    Uses only:
      - PD group windows
      - tremor_present == True
      - arm locations (Upper Left/Right and Lower Left/Right)
    Reuses already-computed acc_rms and gyro_rms -- no re-filtering.
    """
    pd_arm = [w for w in _group_subset(windows, "pd")
              if w["tremor_present"]
              and w["location"] in UPPER_LOCS + LOWER_LOCS]

    if not pd_arm:
        return {"error": "No tremor-present PD arm windows found"}

    n = len(pd_arm)
    acc_sorted  = np.sort(_extract(pd_arm, "acc_rms"))[::-1]
    gyro_sorted = np.sort(_extract(pd_arm, "gyro_rms"))[::-1]

    def _top_mean(vals, k):
        k_actual = min(k, len(vals))
        return float(np.mean(vals[:k_actual])), k_actual

    acc_top50_mean,  acc_n50  = _top_mean(acc_sorted,  50)
    acc_top100_mean, acc_n100 = _top_mean(acc_sorted, 100)
    gyro_top50_mean,  gyro_n50  = _top_mean(gyro_sorted,  50)
    gyro_top100_mean, gyro_n100 = _top_mean(gyro_sorted, 100)

    return {
        "n_tremor_present_arm_windows": n,
        "note": ("Tremor-present PD windows at arm locations (Upper+Lower Left/Right). "
                 "Uses pre-computed bandpass RMS -- no re-filtering."),
        "acc_rms": {
            "max":           float(acc_sorted[0]),
            "mean_top_50":   acc_top50_mean,
            "mean_top_100":  acc_top100_mean,
            "n_top_50_used":  acc_n50,
            "n_top_100_used": acc_n100,
        },
        "gyro_rms": {
            "max":           float(gyro_sorted[0]),
            "mean_top_50":   gyro_top50_mean,
            "mean_top_100":  gyro_top100_mean,
            "n_top_50_used":  gyro_n50,
            "n_top_100_used": gyro_n100,
        },
    }


def kg_extremes_by_location(windows, n: int = 100) -> dict:
    """Mean of the top-n and bottom-n per-segment k_g values, broken down by group and location.

    k_g is computed per window as gyro_rms / acc_rms (bandpass 2-12 Hz).
    Only windows with acc_rms > 1e-9 and finite k_g are included.

    Returns nested dict:  out[group][location]  with keys:
        n_total, top_n_used, top_n_mean, bottom_n_used, bottom_n_mean
    An "_all" key aggregates across all locations for each group.
    """
    out = {}
    for g in ("ct", "pd", "sim"):
        sub = _group_subset(windows, g)
        if not sub:
            continue
        out[g] = {}
        # Overall across all locations
        all_kg = _extract(sub, "kg_ratio")
        out[g]["_all"] = _top_bottom_n(all_kg, n)
        # Per body location
        for loc in SENSOR_SHEETS:
            loc_sub = [w for w in sub if w["location"] == loc]
            if not loc_sub:
                continue
            out[g][loc] = _top_bottom_n(_extract(loc_sub, "kg_ratio"), n)
    return out


def _assign_score_from_reference_amplitude(ref_amp):
    """Assign severity score (1-4) from reference amplitude using SCORE_RMS_RANGE.

    Values outside configured ranges are clamped to nearest valid score.
    """
    if not np.isfinite(ref_amp):
        return 1

    ranges = []
    for k, vr in SIM_SCORE_RMS_RANGE.items():
        try:
            score = int(k)
            lo, hi = float(vr[0]), float(vr[1])
        except Exception:
            continue
        if np.isfinite(lo) and np.isfinite(hi):
            ranges.append((score, lo, hi))

    if not ranges:
        return 1

    ranges.sort(key=lambda x: x[0])
    min_score, min_lo, _ = ranges[0]
    max_score, _, max_hi = ranges[-1]

    if ref_amp <= min_lo:
        return int(min_score)
    if ref_amp >= max_hi:
        return int(max_score)

    for score, lo, hi in ranges:
        if lo <= ref_amp <= hi:
            return int(score)

    # If inside a gap, choose the nearest interval endpoint.
    best_score = min_score
    best_dist = float("inf")
    for score, lo, hi in ranges:
        dist = min(abs(ref_amp - lo), abs(ref_amp - hi))
        if dist < best_dist:
            best_dist = dist
            best_score = score
    return int(best_score)


def derive_activity_scaling_from_pd(windows, metric="acc_rms_band37"):
    """Derive activity-dependent scaling ratios from tremor-present PD arm windows.

    Method:
      1) Per subject/activity median amplitude on tremor-present PD arm windows
      2) Per subject reference activity = highest median amplitude
      3) Subject severity score from reference amplitude via SCORE_RMS_RANGE
      4) Subject activity ratios relative to reference activity
      5) Aggregate ratios by severity and suggest BETA_ACTIVITY_BY_SCORE
    """
    pd_arm_tremor = [
        w for w in _group_subset(windows, "pd")
        if w.get("tremor_present", False)
        and w.get("location") in UPPER_LOCS + LOWER_LOCS
    ]
    if not pd_arm_tremor:
        return {"error": "No tremor-present PD arm windows found for activity-scaling analysis."}

    canonical_acts = ["calibration", "key", "cardigan", "toast"]
    acts_present = sorted({w["activity"] for w in pd_arm_tremor})
    activities = [a for a in canonical_acts if a in acts_present] + [
        a for a in acts_present if a not in canonical_acts
    ]
    subjects = sorted({w["subject"] for w in pd_arm_tremor})

    per_subject_activity = {}
    per_subject_reference = []
    per_subject_ratios = []

    ratio_bucket = {
        s: {a: [] for a in activities}
        for s in (1, 2, 3, 4)
    }
    global_ratio_bucket = {a: [] for a in activities}

    for subj in subjects:
        sub_wins = [w for w in pd_arm_tremor if w["subject"] == subj]
        activity_rows = {}
        for act in activities:
            vals = np.array([w.get(metric, float("nan")) for w in sub_wins if w["activity"] == act], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            activity_rows[act] = {
                "median_amplitude": float(np.median(vals)),
                "n_windows": int(len(vals)),
            }

        if not activity_rows:
            continue

        ref_activity = max(
            activity_rows.items(),
            key=lambda kv: kv[1]["median_amplitude"],
        )[0]
        ref_amp = float(activity_rows[ref_activity]["median_amplitude"])
        score = _assign_score_from_reference_amplitude(ref_amp)

        per_subject_activity[subj] = activity_rows
        per_subject_reference.append({
            "subject": subj,
            "reference_activity": ref_activity,
            "reference_median_amplitude": ref_amp,
            "assigned_severity_score": int(score),
        })

        ratio_map = {}
        for act, row in activity_rows.items():
            if ref_amp > 0 and np.isfinite(ref_amp):
                ratio = float(row["median_amplitude"] / ref_amp)
            else:
                ratio = float("nan")
            ratio_map[act] = {
                "ratio_to_reference": ratio,
                "median_amplitude": float(row["median_amplitude"]),
                "n_windows": int(row["n_windows"]),
            }
            if np.isfinite(ratio):
                ratio_bucket[int(score)][act].append(ratio)
                global_ratio_bucket[act].append(ratio)

        per_subject_ratios.append({
            "subject": subj,
            "assigned_severity_score": int(score),
            "reference_activity": ref_activity,
            "reference_median_amplitude": ref_amp,
            "activity_ratios": ratio_map,
        })

    if not per_subject_reference:
        return {"error": "No valid PD subject/activity medians available for activity scaling."}

    aggregated_by_severity = {}
    suggested_beta = {}
    for score in (1, 2, 3, 4):
        aggregated_by_severity[score] = {}
        suggested_beta[score] = {}
        for act in activities:
            vals = np.array(ratio_bucket[score].get(act, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            med = float(np.median(vals)) if len(vals) else float("nan")
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            n_subj = int(len(vals))
            aggregated_by_severity[score][act] = {
                "median_ratio": med,
                "mean_ratio": mean,
                "n_subjects": n_subj,
            }

            if np.isfinite(med):
                suggested = med
            else:
                gvals = np.array(global_ratio_bucket.get(act, []), dtype=float)
                gvals = gvals[np.isfinite(gvals)]
                suggested = float(np.median(gvals)) if len(gvals) else 1.0
            suggested_beta[score][act] = float(suggested)

    return {
        "method": {
            "metric": metric,
            "group": "pd",
            "subset": "tremor_present arm windows (Upper/Lower Left/Right)",
            "reference_rule": "per subject: activity with highest median amplitude",
            "severity_rule": "reference median mapped to SCORE_RMS_RANGE with clamping",
            "ratio_rule": "activity_median / reference_activity_median",
            "main_suggestion_stat": "median_ratio",
        },
        "activities": activities,
        "n_subjects_total": int(len(per_subject_reference)),
        "per_subject_activity_medians": per_subject_activity,
        "per_subject_reference": per_subject_reference,
        "per_subject_activity_ratios": per_subject_ratios,
        "aggregated_ratios_by_severity": aggregated_by_severity,
        "suggested_BETA_ACTIVITY_BY_SCORE": suggested_beta,
    }


def derive_pd_coverage_by_sim(windows):
    """Quantify how well SIM covers tremor-present PD arm-window distributions.

    Uses already-computed per-window features only (no re-filtering).
    """
    pd = [
        w for w in _group_subset(windows, "pd")
        if w.get("tremor_present", False)
        and w.get("location") in UPPER_LOCS + LOWER_LOCS
    ]
    sim = [
        w for w in _group_subset(windows, "sim")
        if w.get("tremor_present", False)
        and w.get("location") in UPPER_LOCS + LOWER_LOCS
    ]

    if not pd:
        return {"error": "No tremor-present PD arm windows found for coverage analysis."}
    if not sim:
        return {"error": "No tremor-present SIM arm windows found for coverage analysis."}

    features = ("acc_rms", "dom_freq", "spectral_cent")
    feature_coverage = {}
    sim_intervals = {}

    for feat in features:
        pd_vals = _extract(pd, feat)
        sim_vals = _extract(sim, feat)
        pd_vals = pd_vals[np.isfinite(pd_vals)]
        sim_vals = sim_vals[np.isfinite(sim_vals)]

        if len(pd_vals) == 0 or len(sim_vals) == 0:
            feature_coverage[feat] = {
                "n_pd": int(len(pd_vals)),
                "n_sim": int(len(sim_vals)),
                "sim_p5": float("nan"),
                "sim_p95": float("nan"),
                "pd_pct_inside": float("nan"),
                "pd_pct_below": float("nan"),
                "pd_pct_above": float("nan"),
            }
            sim_intervals[feat] = {"p5": float("nan"), "p95": float("nan")}
            continue

        p5 = float(np.percentile(sim_vals, 5))
        p95 = float(np.percentile(sim_vals, 95))
        inside = (pd_vals >= p5) & (pd_vals <= p95)
        below = pd_vals < p5
        above = pd_vals > p95

        feature_coverage[feat] = {
            "n_pd": int(len(pd_vals)),
            "n_sim": int(len(sim_vals)),
            "sim_p5": p5,
            "sim_p95": p95,
            "pd_pct_inside": float(np.mean(inside) * 100.0),
            "pd_pct_below": float(np.mean(below) * 100.0),
            "pd_pct_above": float(np.mean(above) * 100.0),
        }
        sim_intervals[feat] = {"p5": p5, "p95": p95}

    def _joint_coverage(feat_a, feat_b):
        p5a = sim_intervals.get(feat_a, {}).get("p5", float("nan"))
        p95a = sim_intervals.get(feat_a, {}).get("p95", float("nan"))
        p5b = sim_intervals.get(feat_b, {}).get("p5", float("nan"))
        p95b = sim_intervals.get(feat_b, {}).get("p95", float("nan"))
        sub = [
            w for w in pd
            if np.isfinite(w.get(feat_a, float("nan")))
            and np.isfinite(w.get(feat_b, float("nan")))
        ]
        if not sub or not (np.isfinite(p5a) and np.isfinite(p95a) and np.isfinite(p5b) and np.isfinite(p95b)):
            return {"n_pd": int(len(sub)), "pct_inside_both": float("nan")}
        va = np.array([w[feat_a] for w in sub], dtype=float)
        vb = np.array([w[feat_b] for w in sub], dtype=float)
        inside_both = (va >= p5a) & (va <= p95a) & (vb >= p5b) & (vb <= p95b)
        return {"n_pd": int(len(sub)), "pct_inside_both": float(np.mean(inside_both) * 100.0)}

    joint_coverage = {
        "acc_rms__dom_freq": _joint_coverage("acc_rms", "dom_freq"),
        "acc_rms__spectral_cent": _joint_coverage("acc_rms", "spectral_cent"),
    }

    # Poorly covered PD windows: outside SIM p5-p95 for dom_freq OR acc_rms
    p5_acc = sim_intervals.get("acc_rms", {}).get("p5", float("nan"))
    p95_acc = sim_intervals.get("acc_rms", {}).get("p95", float("nan"))
    p5_f = sim_intervals.get("dom_freq", {}).get("p5", float("nan"))
    p95_f = sim_intervals.get("dom_freq", {}).get("p95", float("nan"))

    poor = []
    for w in pd:
        acc = w.get("acc_rms", float("nan"))
        frq = w.get("dom_freq", float("nan"))
        if not (np.isfinite(acc) and np.isfinite(frq)):
            continue
        out_acc = not (p5_acc <= acc <= p95_acc) if np.isfinite(p5_acc) and np.isfinite(p95_acc) else False
        out_frq = not (p5_f <= frq <= p95_f) if np.isfinite(p5_f) and np.isfinite(p95_f) else False
        if out_acc or out_frq:
            poor.append(w)

    def _median_metric(sub, feat):
        vals = np.array([w.get(feat, float("nan")) for w in sub], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if len(vals) else float("nan")

    def _distribution_by_key(sub, key):
        total = len(sub)
        if total == 0:
            return []
        counts = {}
        for w in sub:
            k = str(w.get(key, "unknown"))
            counts[k] = counts.get(k, 0) + 1
        rows = []
        for k, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append({
                key: k,
                "n_windows": int(c),
                "pct_windows": float(c / total * 100.0),
            })
        return rows

    poor_profile = {
        "n_windows": int(len(poor)),
        "pct_of_pd_tremor_present_arm": float(len(poor) / len(pd) * 100.0) if len(pd) else float("nan"),
        "median_acc_rms": _median_metric(poor, "acc_rms"),
        "median_dom_freq": _median_metric(poor, "dom_freq"),
        "median_spectral_cent": _median_metric(poor, "spectral_cent"),
        "median_kg_ratio": _median_metric(poor, "kg_ratio"),
        "activity_distribution": _distribution_by_key(poor, "activity"),
        "subject_distribution": _distribution_by_key(poor, "subject"),
    }

    # Optional binned summary: PD vs SIM on tremor-present arm windows.
    s2_lo = float(SIM_SCORE_RMS_RANGE.get(2, (0.28, 0.48))[0])
    s3_lo = float(SIM_SCORE_RMS_RANGE.get(3, (0.48, 0.72))[0])
    s4_lo = float(SIM_SCORE_RMS_RANGE.get(4, (0.72, 1.00))[0])

    def _pct_distribution(vals, bins):
        vals = np.array(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        total = len(vals)
        out = {}
        for label, mask_fn in bins:
            if total == 0:
                out[label] = float("nan")
            else:
                out[label] = float(np.mean(mask_fn(vals)) * 100.0)
        out["n"] = int(total)
        return out

    rms_bins = [
        ("low", lambda v: v < s2_lo),
        ("medium", lambda v: (v >= s2_lo) & (v < s3_lo)),
        ("high", lambda v: (v >= s3_lo) & (v < s4_lo)),
        ("very_high", lambda v: v >= s4_lo),
    ]
    freq_bins = [
        ("lt_3p5_hz", lambda v: v < 3.5),
        ("hz_3p5_to_5", lambda v: (v >= 3.5) & (v < 5.0)),
        ("hz_5_to_7", lambda v: (v >= 5.0) & (v <= 7.0)),
        ("gt_7_hz", lambda v: v > 7.0),
    ]

    binned_summary = {
        "rms_bins": {
            "bin_edges_reference": {
                "score2_low": s2_lo,
                "score3_low": s3_lo,
                "score4_low": s4_lo,
            },
            "pd": _pct_distribution([w.get("acc_rms", float("nan")) for w in pd], rms_bins),
            "sim": _pct_distribution([w.get("acc_rms", float("nan")) for w in sim], rms_bins),
        },
        "frequency_bins": {
            "pd": _pct_distribution([w.get("dom_freq", float("nan")) for w in pd], freq_bins),
            "sim": _pct_distribution([w.get("dom_freq", float("nan")) for w in sim], freq_bins),
        },
    }

    return {
        "method": {
            "subset": "tremor-present arm windows (Upper/Lower Left/Right)",
            "features": ["acc_rms", "dom_freq", "spectral_cent"],
            "sim_interval": "p5-p95",
            "poor_definition": "PD windows outside SIM p5-p95 for acc_rms OR dom_freq",
        },
        "n_pd_windows": int(len(pd)),
        "n_sim_windows": int(len(sim)),
        "feature_coverage": feature_coverage,
        "joint_coverage": joint_coverage,
        "poorly_covered_pd_windows": poor_profile,
        "binned_summary": binned_summary,
    }


def build_tremor_only_comparison(tremor_freq_stats):
    """Match scores computed on tremor-present arm windows only."""
    pd_s  = tremor_freq_stats.get("pd",  {})
    sim_s = tremor_freq_stats.get("sim", {})
    metrics = ["dom_freq", "acc_rms", "gyro_rms", "kg_ratio"]
    comp = {}
    for m in metrics:
        comp[m] = {
            "pd":  pd_s.get(m, {}),
            "sim": sim_s.get(m, {}),
            "match_score": _match_score(pd_s, sim_s, m),
        }
    scores = [comp[m]["match_score"] for m in metrics
              if np.isfinite(comp[m]["match_score"])]
    comp["overall_match_score"] = float(np.mean(scores)) if scores else float("nan")
    return comp


def threshold_sensitivity(windows, thresholds=(0.10, 0.15, 0.20, 0.25)):
    """Compute tremor prevalence per group at multiple detection thresholds.
    Uses acc_rms_band37 (already computed) -- no re-filtering needed."""
    results = {}
    for thr in thresholds:
        row = {}
        for g in ("ct", "pd", "sim"):
            sub = _group_subset(windows, g)
            if not sub:
                row[g] = float("nan")
                continue
            vals = _extract(sub, "acc_rms_band37")
            row[g] = float(np.mean(vals[np.isfinite(vals)] > thr))
        results[thr] = row
    return results


def aggregate_by_sim_mode(windows):
    """Return per-mode stats for simulated windows."""
    sim_wins = _group_subset(windows, "sim")
    modes = sorted({w.get("_sim_augmode", "unknown") for w in sim_wins})
    out = {}
    for mode in modes:
        sub = [w for w in sim_wins if w.get("_sim_augmode") == mode]
        out[mode] = {
            "n_windows":        len(sub),
            "acc_rms":          _stats(_extract(sub, "acc_rms")),
            "gyro_rms":         _stats(_extract(sub, "gyro_rms")),
            "dom_freq":         _stats(_extract(sub, "dom_freq")),
            "tremor_prevalence": float(np.mean([w["tremor_present"] for w in sub])),
        }
    return out


def pd_lower_arm_tremor_iqr_stats(records):
    """IQR statistics for tremor-present PD lower-arm windows.

    Subset:
      - PD group only
      - Lower arm locations only (Lower Right, Lower Left)
      - Tremor-present windows only (acc_rms_band37 > TREMOR_PRESENCE_THRESHOLD)

    Features computed per window:
      - dom_freq        : dominant frequency via Welch PSD (2-12 Hz bandpass input)
      - acc_rms_band37  : 3D acc RMS bandpass-filtered to 3-7 Hz
      - gyro_rms_band37 : 3D gyro RMS bandpass-filtered to 3-7 Hz
      - k_g             : gyro_rms_band37 / acc_rms_band37
    """
    win = int(round(PRIMARY_WINDOW_SEC * FS_ORIGINAL))
    dom_freqs, acc_vals, gyro_vals = [], [], []

    for rec in records:
        if rec["group"] != "pd":
            continue
        if rec["location"] not in LOWER_LOCS:
            continue

        acc_full  = rec["data"][:, ACC_COLS]
        gyro_full = rec["data"][:, GYRO_COLS]
        if len(acc_full) < win:
            continue

        acc_bp37  = _bp_filter(acc_full,  TREMOR_BAND_LO, TREMOR_BAND_HI, FS_ORIGINAL)
        gyro_bp37 = _bp_filter(gyro_full, TREMOR_BAND_LO, TREMOR_BAND_HI, FS_ORIGINAL)
        acc_bp    = _bp_filter(acc_full,  TREMOR_LO_HZ,   TREMOR_HI_HZ,   FS_ORIGINAL)

        n_w = len(acc_full) // win
        for w in range(n_w):
            s, e = w * win, (w + 1) * win
            acc_rms37 = _rms_3d(acc_bp37[s:e])
            if acc_rms37 <= TREMOR_PRESENCE_THRESHOLD:
                continue  # skip non-tremor windows
            gyro_rms37 = _rms_3d(gyro_bp37[s:e])
            wf = _welch_features(acc_bp[s:e], FS_ORIGINAL)
            acc_vals.append(acc_rms37)
            gyro_vals.append(gyro_rms37)
            dom_freqs.append(wf["dom_freq"])

    acc_arr  = np.array(acc_vals,  dtype=float)
    gyro_arr = np.array(gyro_vals, dtype=float)
    freq_arr = np.array(dom_freqs, dtype=float)

    valid  = (acc_arr > 1e-9) & np.isfinite(acc_arr) & np.isfinite(gyro_arr)
    kg_arr = np.where(valid, gyro_arr / acc_arr, float("nan"))

    def _iqr(arr):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return {"p25": float("nan"), "median": float("nan"), "p75": float("nan"), "n": 0}
        return {
            "p25":    float(np.percentile(v, 25)),
            "median": float(np.median(v)),
            "p75":    float(np.percentile(v, 75)),
            "n":      int(len(v)),
        }

    return {
        "n_windows":       int(len(acc_arr)),
        "dom_freq":        _iqr(freq_arr),
        "acc_rms_band37":  _iqr(acc_arr),
        "gyro_rms_band37": _iqr(gyro_arr),
        "k_g":             _iqr(kg_arr),
    }


# ===========================================================================
# SECTION 7 -- COMPARISON: REAL vs SIMULATED
# ===========================================================================

def _match_score(pd_stats, sim_stats, metric):
    """Score 0-1 for how close sim median is to PD median."""
    m_pd  = pd_stats.get(metric,{}).get("median", float("nan"))
    m_sim = sim_stats.get(metric,{}).get("median", float("nan"))
    if not (np.isfinite(m_pd) and np.isfinite(m_sim)):
        return float("nan")
    if metric in ("acc_rms","gyro_rms","band_power"):
        ratio = m_sim / (m_pd + 1e-12)
        log_err = abs(np.log(max(ratio, 1e-6)))
        return float(np.exp(-log_err))
    else:
        rel_err = abs(m_sim - m_pd) / (abs(m_pd) + 1e-9)
        return float(max(0.0, 1.0 - rel_err))


def build_comparison(agg):
    bg = agg["by_group"]
    pd_s  = bg.get("pd",{})
    ct_s  = bg.get("ct",{})
    sim_s = bg.get("sim",{})

    metrics_to_compare = ["acc_rms","gyro_rms","dom_freq","kg_ratio","band_ratio"]

    comparison = {}
    for m in metrics_to_compare:
        comparison[m] = {
            "pd":  pd_s.get(m,{}),
            "ct":  ct_s.get(m,{}),
            "sim": sim_s.get(m,{}),
            "match_score_sim_vs_pd": _match_score(pd_s, sim_s, m),
        }

    scores = [comparison[m]["match_score_sim_vs_pd"]
              for m in metrics_to_compare
              if np.isfinite(comparison[m]["match_score_sim_vs_pd"])]
    comparison["overall_match_score"] = float(np.mean(scores)) if scores else float("nan")

    comparison["tremor_prevalence"] = {
        "pd_mean":  pd_s.get("tremor_prevalence", float("nan")),
        "ct_mean":  ct_s.get("tremor_prevalence", float("nan")),
        "sim_mean": sim_s.get("tremor_prevalence", float("nan")),
    }

    return comparison


def _relative_diff_pct(pd_value, sim_value):
    if not (np.isfinite(pd_value) and np.isfinite(sim_value)):
        return float("nan")
    if abs(pd_value) < 1e-9:
        return float("nan")
    return float((sim_value - pd_value) / abs(pd_value) * 100.0)


def _prevalence_delta_pct_points(pd_prev, sim_prev):
    if not (np.isfinite(pd_prev) and np.isfinite(sim_prev)):
        return float("nan")
    return float((sim_prev - pd_prev) * 100.0)


def _prevalence_match_score(pd_prev, sim_prev):
    if not (np.isfinite(pd_prev) and np.isfinite(sim_prev)):
        return float("nan")
    return float(max(0.0, 1.0 - abs(sim_prev - pd_prev)))


def _activity_match_label(score):
    if not np.isfinite(score):
        return "Unavailable"
    if score > 0.8:
        return "Good match"
    if score >= 0.6:
        return "Moderate mismatch"
    return "Poor match"


def compare_pd_vs_sim_per_activity(windows):
    """Direct PD vs SIM comparison per activity using existing window features."""
    metrics = ("acc_rms", "dom_freq", "spectral_cent", "kg_ratio")
    activities = sorted({w["activity"] for w in windows if w["group"] in ("pd", "sim")})
    out = {}

    for act in activities:
        pd_sub = [w for w in windows if w["group"] == "pd" and w["activity"] == act]
        sim_sub = [w for w in windows if w["group"] == "sim" and w["activity"] == act]
        if not pd_sub or not sim_sub:
            continue

        act_out = {
            "n_pd_windows": len(pd_sub),
            "n_sim_windows": len(sim_sub),
        }
        scores = []

        for metric in metrics:
            pd_stats = _stats(_extract(pd_sub, metric))
            sim_stats = _stats(_extract(sim_sub, metric))
            match_score = _match_score({metric: pd_stats}, {metric: sim_stats}, metric)
            rel_diff_pct = _relative_diff_pct(pd_stats.get("median", float("nan")),
                                              sim_stats.get("median", float("nan")))
            act_out[metric] = {
                "pd": pd_stats,
                "sim": sim_stats,
                "relative_diff_pct": rel_diff_pct,
                "match_score": match_score,
            }
            if np.isfinite(match_score):
                scores.append(match_score)

        pd_prev = float(np.mean([w["tremor_present"] for w in pd_sub]))
        sim_prev = float(np.mean([w["tremor_present"] for w in sim_sub]))
        prev_score = _prevalence_match_score(pd_prev, sim_prev)
        act_out["tremor_prevalence"] = {
            "pd": pd_prev,
            "sim": sim_prev,
            "delta_pct_points": _prevalence_delta_pct_points(pd_prev, sim_prev),
            "match_score": prev_score,
        }
        if np.isfinite(prev_score):
            scores.append(prev_score)

        overall = float(np.mean(scores)) if scores else float("nan")
        act_out["overall_match_score"] = overall
        act_out["match_label"] = _activity_match_label(overall)
        out[act] = act_out

    return out


def compare_pd_vs_sim_per_location(windows):
    """Direct PD vs SIM comparison per body location using existing features."""
    metrics = ("acc_rms", "gyro_rms", "dom_freq")
    out = {}

    for loc in SENSOR_SHEETS:
        pd_sub = [w for w in windows if w["group"] == "pd" and w["location"] == loc]
        sim_sub = [w for w in windows if w["group"] == "sim" and w["location"] == loc]

        loc_out = {
            "n_pd_windows": len(pd_sub),
            "n_sim_windows": len(sim_sub),
        }
        scores = []

        for metric in metrics:
            pd_stats = _stats(_extract(pd_sub, metric)) if pd_sub else _stats(np.array([]))
            sim_stats = _stats(_extract(sim_sub, metric)) if sim_sub else _stats(np.array([]))
            match_score = _match_score({metric: pd_stats}, {metric: sim_stats}, metric)
            rel_diff_pct = _relative_diff_pct(
                pd_stats.get("median", float("nan")),
                sim_stats.get("median", float("nan")),
            )
            loc_out[metric] = {
                "pd": pd_stats,
                "sim": sim_stats,
                "relative_diff_pct": rel_diff_pct,
                "match_score": match_score,
            }
            if np.isfinite(match_score):
                scores.append(match_score)

        pd_prev = float(np.mean([w["tremor_present"] for w in pd_sub])) if pd_sub else float("nan")
        sim_prev = float(np.mean([w["tremor_present"] for w in sim_sub])) if sim_sub else float("nan")
        prev_score = _prevalence_match_score(pd_prev, sim_prev)
        loc_out["tremor_prevalence"] = {
            "pd": pd_prev,
            "sim": sim_prev,
            "delta_pct_points": _prevalence_delta_pct_points(pd_prev, sim_prev),
            "match_score": prev_score,
        }
        if np.isfinite(prev_score):
            scores.append(prev_score)

        overall = float(np.mean(scores)) if scores else float("nan")
        loc_out["overall_match_score"] = overall
        loc_out["match_label"] = _activity_match_label(overall)
        out[loc] = loc_out

    ranked = [
        {
            "location": loc,
            "overall_match_score": val.get("overall_match_score", float("nan")),
            "match_label": val.get("match_label", "Unavailable"),
        }
        for loc, val in out.items()
        if np.isfinite(val.get("overall_match_score", float("nan")))
    ]
    ranked.sort(key=lambda r: r["overall_match_score"])

    return {
        "per_location": out,
        "worst_matching_locations": ranked,
    }


def derive_location_scaling_from_pd(windows, metric="acc_rms_band37"):
    """Derive location scaling ratios by severity from tremor-present PD windows.

    Reference per subject: stronger of Lower Right / Lower Left.
    """
    pd_tremor = [
        w for w in _group_subset(windows, "pd")
        if w.get("tremor_present", False)
        and w.get("location") in SENSOR_SHEETS
    ]
    if not pd_tremor:
        return {"error": "No tremor-present PD windows found for location scaling analysis."}

    locations = list(SENSOR_SHEETS)
    subjects = sorted({w["subject"] for w in pd_tremor})

    per_subject_location = {}
    per_subject_reference = []
    per_subject_ratios = []

    ratio_bucket = {s: {loc: [] for loc in locations} for s in (1, 2, 3, 4)}
    global_ratio_bucket = {loc: [] for loc in locations}

    for subj in subjects:
        sub_wins = [w for w in pd_tremor if w["subject"] == subj]
        loc_rows = {}
        for loc in locations:
            vals = np.array([w.get(metric, float("nan")) for w in sub_wins if w["location"] == loc], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            loc_rows[loc] = {
                "median_amplitude": float(np.median(vals)),
                "n_windows": int(len(vals)),
            }

        if not loc_rows:
            continue

        lower_candidates = {loc: loc_rows[loc]["median_amplitude"] for loc in ("Lower Right", "Lower Left") if loc in loc_rows}
        if not lower_candidates:
            continue
        ref_location = max(lower_candidates.items(), key=lambda kv: kv[1])[0]
        ref_amp = float(loc_rows[ref_location]["median_amplitude"])
        score = _assign_score_from_reference_amplitude(ref_amp)

        per_subject_location[subj] = loc_rows
        per_subject_reference.append({
            "subject": subj,
            "reference_location": ref_location,
            "reference_median_amplitude": ref_amp,
            "assigned_severity_score": int(score),
        })

        ratio_map = {}
        for loc, row in loc_rows.items():
            ratio = float(row["median_amplitude"] / ref_amp) if ref_amp > 0 and np.isfinite(ref_amp) else float("nan")
            ratio_map[loc] = {
                "ratio_to_reference": ratio,
                "median_amplitude": float(row["median_amplitude"]),
                "n_windows": int(row["n_windows"]),
            }
            if np.isfinite(ratio):
                ratio_bucket[int(score)][loc].append(ratio)
                global_ratio_bucket[loc].append(ratio)

        per_subject_ratios.append({
            "subject": subj,
            "assigned_severity_score": int(score),
            "reference_location": ref_location,
            "reference_median_amplitude": ref_amp,
            "location_ratios": ratio_map,
        })

    if not per_subject_reference:
        return {"error": "No valid subject references (Lower Right/Lower Left) found for location scaling."}

    aggregated_by_severity = {}
    suggested_beta = {}
    for score in (1, 2, 3, 4):
        aggregated_by_severity[score] = {}
        suggested_beta[score] = {}
        for loc in locations:
            vals = np.array(ratio_bucket[score].get(loc, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            med = float(np.median(vals)) if len(vals) else float("nan")
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            n_subj = int(len(vals))
            aggregated_by_severity[score][loc] = {
                "median_ratio": med,
                "mean_ratio": mean,
                "n_subjects": n_subj,
            }

            if np.isfinite(med):
                suggested = med
            else:
                gvals = np.array(global_ratio_bucket.get(loc, []), dtype=float)
                gvals = gvals[np.isfinite(gvals)]
                suggested = float(np.median(gvals)) if len(gvals) else 1.0
            suggested_beta[score][loc] = float(suggested)

    return {
        "method": {
            "metric": metric,
            "group": "pd",
            "subset": "tremor_present windows (all body locations)",
            "reference_rule": "per subject: stronger of Lower Right / Lower Left",
            "severity_rule": "reference median mapped to SCORE_RMS_RANGE with clamping",
            "ratio_rule": "location_median / reference_location_median",
            "main_suggestion_stat": "median_ratio",
        },
        "locations": locations,
        "n_subjects_total": int(len(per_subject_reference)),
        "per_subject_location_medians": per_subject_location,
        "per_subject_reference": per_subject_reference,
        "per_subject_location_ratios": per_subject_ratios,
        "aggregated_ratios_by_severity": aggregated_by_severity,
        "suggested_BETA_LOCATION_BY_SCORE": suggested_beta,
    }


# ===========================================================================
# SECTION 8 -- DATA-DRIVEN CONFIG SUGGESTIONS
# ===========================================================================

def build_suggestions(agg, windows):
    pd_arm = [w for w in _group_subset(windows, "pd")
              if w["location"] in UPPER_LOCS + LOWER_LOCS and w["tremor_present"]]

    if not pd_arm:
        return {"error": "No tremor-present PD windows found -- lower TREMOR_PRESENCE_THRESHOLD"}

    acc_vals  = _extract(pd_arm, "acc_rms")
    freq_vals = _extract(pd_arm, "dom_freq")
    kg_vals   = _extract(pd_arm, "kg_ratio")

    acc_vals  = acc_vals[np.isfinite(acc_vals)]
    freq_vals = freq_vals[np.isfinite(freq_vals)]
    kg_vals   = kg_vals[np.isfinite(kg_vals)]

    q   = [0, 25, 50, 75, 100]
    pct = np.percentile(acc_vals, q) if len(acc_vals) >= 4 else [0]*5
    sug_score_rms = {
        1: (round(float(pct[0]),3), round(float(pct[1]),3)),
        2: (round(float(pct[1]),3), round(float(pct[2]),3)),
        3: (round(float(pct[2]),3), round(float(pct[3]),3)),
        4: (round(float(pct[3]),3), round(float(pct[4]),3)),
    }

    # Frequency: use arm-location tremor-present PD windows only
    pd_arm_tremor = [w for w in _group_subset(windows, "pd")
                     if w["location"] in UPPER_LOCS and w["tremor_present"]]
    freq_vals_arm = _extract(pd_arm_tremor, "dom_freq")
    freq_vals_arm = freq_vals_arm[np.isfinite(freq_vals_arm)]
    # Fall back to all tremor-present if arm has too few samples
    if len(freq_vals_arm) >= 20:
        freq_vals_for_sug = freq_vals_arm
        freq_source_note  = "arm locations (Upper Left/Right), tremor-present PD windows"
    else:
        freq_vals_for_sug = freq_vals
        freq_source_note  = "all locations, tremor-present PD windows"
    sug_freq = (round(float(np.percentile(freq_vals_for_sug, 5)), 2),
                round(float(np.percentile(freq_vals_for_sug, 95)), 2)) if len(freq_vals_for_sug) else (None, None)

    sug_kg_med = round(float(np.median(kg_vals)), 3) if len(kg_vals) else None
    sug_kg_p25 = round(float(np.percentile(kg_vals, 25)), 3) if len(kg_vals) else None
    sug_kg_p75 = round(float(np.percentile(kg_vals, 75)), 3) if len(kg_vals) else None

    return {
        "n_tremor_present_windows": len(pd_arm),
        "note_kg_units": ("k_g measured in raw XLS units. "
                          "If gyro is in rad/s: multiply by 180/pi to get deg/s. "
                          "Simulation uses deg/s, so measured k_g ~ X -> sim k_g = X*(180/pi)."),
        "suggested_SCORE_RMS_RANGE": sug_score_rms,
        "current_SCORE_RMS_RANGE":   SIM_SCORE_RMS_RANGE,
        "suggested_FREQ_RANGE_HZ": sug_freq,
        "suggested_FREQ_RANGE_HZ_source": freq_source_note,
        "current_FREQ_RANGE_HZ":   SIM_FREQ_RANGE_HZ,
        "suggested_kg_median":     sug_kg_med,
        "suggested_kg_p25_p75":    (sug_kg_p25, sug_kg_p75),
        "current_KG_values":       SIM_KG_BY_SEVERITY,
    }


# ===========================================================================
# SECTION 9 -- VISUALIZATION
# ===========================================================================

COLORS = {"pd":"#E05252","ct":"#5278E0","sim":"#3DAA5A"}
LABELS = {"pd":"PD (real)","ct":"CT (control)","sim":"SIM (simulated)"}


def _hist3(ax, windows, metric, xlabel, title, lo_pct=1.0, hi_pct=99.0, bins=35):
    all_vals = []
    for g in ("ct","pd","sim"):
        sub = _group_subset(windows, g)
        if not sub: continue
        v = _extract(sub, metric); v = v[np.isfinite(v)]
        all_vals.append(v)
    if not all_vals: return
    flat = np.concatenate(all_vals)
    lo, hi = np.percentile(flat, lo_pct), np.percentile(flat, hi_pct)
    for g in ("ct","pd","sim"):
        sub = _group_subset(windows, g)
        if not sub: continue
        v = _extract(sub, metric); v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, range=(lo,hi), density=True, alpha=0.5,
                color=COLORS[g], label=LABELS[g])
        ax.axvline(np.median(v), color=COLORS[g], linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel); ax.set_title(title); ax.legend(fontsize=7)
    ax.set_ylabel("Density")


def plot_group_distributions(windows):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("CT vs PD vs SIM - Metric Distributions (all locations)", fontsize=12)
    specs = [
        ("acc_rms",    "Acc RMS (bandpass 2-12 Hz)", axes[0,0]),
        ("gyro_rms",   "Gyro RMS (bandpass 2-12 Hz)", axes[0,1]),
        ("dom_freq",   "Dominant Frequency (Hz)",     axes[0,2]),
        ("kg_ratio",   "k_g = Gyro/Acc RMS",          axes[1,0]),
        ("band_ratio", "Band power ratio (3-7/2-12 Hz)", axes[1,1]),
        ("acc_rms_band37","Acc RMS 3-7 Hz band",      axes[1,2]),
    ]
    for metric, xlabel, ax in specs:
        _hist3(ax, windows, metric, xlabel, metric.replace("_"," ").title())
    plt.tight_layout()
    _save("01_group_distributions.png")


def plot_per_location(windows):
    groups = [g for g in ("ct","pd","sim") if _group_subset(windows, g)]
    fig, axes = plt.subplots(len(groups), 2, figsize=(13, 4*len(groups)))
    if len(groups) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Acc RMS & Dominant Frequency per Location", fontsize=11)
    for row, g in enumerate(groups):
        sub = _group_subset(windows, g)
        for col, (metric, ylabel) in enumerate([
            ("acc_rms", "Acc RMS (m/s^2)"), ("dom_freq", "Dom. Freq (Hz)")
        ]):
            ax = axes[row, col]
            data, labels = [], []
            for loc in SENSOR_SHEETS:
                v = _extract([w for w in sub if w["location"]==loc], metric)
                v = v[np.isfinite(v)]
                if len(v): data.append(v); labels.append(LOCATION_CODE[loc])
            if data:
                ax.boxplot(data, tick_labels=labels, patch_artist=True,
                           boxprops=dict(facecolor=COLORS[g], alpha=0.6),
                           medianprops=dict(color="black", linewidth=2))
            ax.set_title(f"{LABELS[g]}  -  {ylabel}"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("02_per_location.png")


def plot_per_activity(windows):
    activities = sorted({w["activity"] for w in windows})
    groups = [g for g in ("ct","pd","sim") if _group_subset(windows, g)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Per Activity - Acc RMS and Tremor Prevalence", fontsize=11)

    ax = axes[0]; x = np.arange(len(activities)); bw = 0.25
    for i, g in enumerate(groups):
        sub = _group_subset(windows, g)
        meds = []
        for act in activities:
            v = _extract([w2 for w2 in sub if w2["activity"]==act], "acc_rms")
            meds.append(float(np.nanmedian(v[np.isfinite(v)])) if len(v) else 0)
        ax.bar(x + i*bw, meds, width=bw, label=LABELS[g], color=COLORS[g], alpha=0.75)
    ax.set_xticks(x + bw); ax.set_xticklabels(activities, rotation=20)
    ax.set_ylabel("Median Acc RMS"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Acc RMS per Activity")

    ax = axes[1]
    for i, g in enumerate(groups):
        sub = _group_subset(windows, g)
        prevs = []
        for act in activities:
            s2 = [w2 for w2 in sub if w2["activity"]==act]
            prevs.append(float(np.mean([w2["tremor_present"] for w2 in s2])) if s2 else 0)
        ax.bar(x + i*bw, prevs, width=bw, label=LABELS[g], color=COLORS[g], alpha=0.75)
    ax.set_xticks(x + bw); ax.set_xticklabels(activities, rotation=20)
    ax.set_ylabel("Fraction tremor-present"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Tremor Prevalence per Activity")
    plt.tight_layout(); _save("03_per_activity.png")


def plot_psd_comparison(records, sim_windows):
    def _mean_psd(recs_or_wins, is_sim=False):
        psds = []
        for item in recs_or_wins:
            if is_sim:
                sig = np.sqrt(np.sum(item["data_acc"]**2, axis=1))
            else:
                acc = item["data"][:, ACC_COLS]
                sig = np.sqrt(np.sum(acc**2, axis=1))
            n = len(sig); nperseg = min(n, max(32, n//2))
            f, p = welch(sig, fs=FS_ORIGINAL, nperseg=nperseg)
            psds.append((f, p))
        if not psds: return None, None
        f_common = psds[0][0]
        pstack = np.vstack([np.interp(f_common, f, p) for f, p in psds])
        return f_common, np.mean(pstack, axis=0)

    pd_arm = [r for r in records if r["group"]=="pd" and r["location"] in UPPER_LOCS]
    ct_arm = [r for r in records if r["group"]=="ct" and r["location"] in UPPER_LOCS]
    fig, ax = plt.subplots(figsize=(9, 5))
    for recs, g, is_sim in [(ct_arm,"ct",False),(pd_arm,"pd",False),(sim_windows,"sim",True)]:
        f, p = _mean_psd(recs, is_sim=is_sim)
        if f is None: continue
        mask = (f >= 0.5) & (f <= 15)
        ax.semilogy(f[mask], p[mask], color=COLORS[g], linewidth=2, label=LABELS[g])
    ax.axvspan(TREMOR_BAND_LO, TREMOR_BAND_HI, alpha=0.12, color="gray", label="3-7 Hz band")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("Mean Welch PSD - CT vs PD vs SIM (arm locations)")
    ax.legend(); ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout(); _save("04_psd_comparison.png")


def plot_per_subject(agg):
    subjects = agg["per_subject"]
    if not subjects: return
    names = [s["subject"] for s in subjects]
    x     = np.arange(len(names))
    fig, axes = plt.subplots(4, 1, figsize=(max(10, len(names)*0.7), 14))
    fig.suptitle("PD Subjects - Arm Locations, All Activities", fontsize=11)
    rows = [
        ([s["acc_rms_med"]  for s in subjects], "Acc RMS (m/s^2)",    "#E05252"),
        ([s["dom_freq_med"] for s in subjects], "Dominant Freq (Hz)", "#52A0E0"),
        ([s["kg_ratio_med"] for s in subjects], "k_g = Gyro/Acc",     "#60C060"),
        ([s["tremor_prev"]  for s in subjects], "Tremor Prevalence",  "#C060C0"),
    ]
    for ax, (vals, ylabel, color) in zip(axes, rows):
        bars = ax.bar(x, vals, color=color, alpha=0.75)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    axes[1].axhline(SIM_FREQ_RANGE_HZ[0], color="orange", linestyle="--", lw=1.2,
                    label=f"Sim lo ({SIM_FREQ_RANGE_HZ[0]} Hz)")
    axes[1].axhline(SIM_FREQ_RANGE_HZ[1], color="orange", linestyle="-",  lw=1.2,
                    label=f"Sim hi ({SIM_FREQ_RANGE_HZ[1]} Hz)")
    axes[1].legend(fontsize=7)
    plt.tight_layout(); _save("05_per_subject.png")


def plot_lr_asymmetry(agg):
    asym = agg["lr_asymmetry"]
    if not asym: return
    subjects   = sorted({r["subject"]  for r in asym})
    activities = sorted({r["activity"] for r in asym})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PD Left-Right Asymmetry (Acc RMS)", fontsize=11)
    for ax, arm_key, title in [
        (axes[0], "asym_upper_arm", "Upper Arm (UR vs UL)"),
        (axes[1], "asym_lower_arm", "Lower Arm (LR vs LL)"),
    ]:
        x = np.arange(len(subjects)); bw = 0.2
        for i, act in enumerate(activities):
            vals = []
            for subj in subjects:
                rows = [r for r in asym if r["subject"]==subj and r["activity"]==act]
                vals.append(rows[0][arm_key] if rows and np.isfinite(rows[0][arm_key]) else 0)
            ax.bar(x + i*bw, vals, width=bw, alpha=0.7, label=act)
        ax.set_xticks(x + bw*len(activities)/2)
        ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Asymmetry |L-R|/(L+R)"); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save("06_lr_asymmetry.png")


def plot_kg_scatter(windows):
    fig, ax = plt.subplots(figsize=(7, 6))
    for g in ("ct","pd","sim"):
        sub = [w for w in _group_subset(windows, g)
               if w["location"] in UPPER_LOCS + LOWER_LOCS]
        if not sub: continue
        acc  = _extract(sub,"acc_rms"); gyro = _extract(sub,"gyro_rms")
        m    = np.isfinite(acc) & np.isfinite(gyro) & (acc > TREMOR_PRESENCE_THRESHOLD)
        ax.scatter(acc[m], gyro[m], s=8, alpha=0.25, color=COLORS[g], label=LABELS[g])
    x_line = np.linspace(0, 3, 200)
    colors_line = ["#1a9e4f","#d4a017","#1a7abf","#7b1ab5"]
    for (label, kg), c in zip(SIM_KG_BY_SEVERITY.items(), colors_line):
        ax.plot(x_line, kg*x_line, "--", lw=1.5, color=c, label=f"Sim k_g={kg} ({label})")
    ax.set_xlabel("Acc RMS (bandpass)"); ax.set_ylabel("Gyro RMS (bandpass)")
    ax.set_title("k_g Scatter: Gyro vs Acc RMS"); ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout(); _save("07_kg_scatter.png")


def plot_freq_window_comparison(freq_by_wl):
    wls = sorted(freq_by_wl.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Dominant Frequency vs Window Length (Welch PSD)", fontsize=11)
    for ax, g, gname in [(axes[0],"pd","PD"),(axes[1],"ct","CT")]:
        meds = [freq_by_wl[wl][g]["median"] for wl in wls]
        p5s  = [freq_by_wl[wl][g]["p5"]    for wl in wls]
        p95s = [freq_by_wl[wl][g]["p95"]   for wl in wls]
        ax.plot(wls, meds, "o-", color=COLORS[g], linewidth=2, label="median")
        ax.fill_between(wls, p5s, p95s, alpha=0.2, color=COLORS[g], label="p5-p95")
        ax.axhspan(SIM_FREQ_RANGE_HZ[0], SIM_FREQ_RANGE_HZ[1], alpha=0.12,
                   color="orange", label="Sim freq range")
        ax.set_xlabel("Window length (s)"); ax.set_ylabel("Dominant Frequency (Hz)")
        ax.set_title(f"{gname} - Welch dominant freq vs window size")
        ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout(); _save("08_freq_vs_window_length.png")


def _save(name):
    p = PLOTS_DIR / name
    plt.savefig(p, dpi=120); plt.close()
    print(f"  Saved: {name}")


# ===========================================================================
# SECTION 10 -- REPORT GENERATION
# ===========================================================================

def _fmt(v, d=3):
    if isinstance(v, (float, np.floating)) and not np.isfinite(v): return "n/a"
    if v is None: return "n/a"
    return f"{v:.{d}f}"

def _sr(s):
    return (f"n={s.get('n',0):5d}  mean={_fmt(s.get('mean'))}  "
            f"median={_fmt(s.get('median'))}  std={_fmt(s.get('std'))}  "
            f"[p5={_fmt(s.get('p5'))}, p95={_fmt(s.get('p95'))}]")

def _bar(score, width=20):
    if not np.isfinite(score): return "[n/a]"
    filled = int(round(score * width))
    return "[" + chr(9608)*filled + chr(9617)*(width-filled) + f"] {score:.2f}"


def _fmt_pct_diff(v, d=1):
    if not np.isfinite(v):
        return "n/a"
    return f"{v:+.{d}f}%"


def _fmt_pp_diff(v, d=1):
    if not np.isfinite(v):
        return "n/a"
    return f"{v:+.{d}f} pp"


def _activity_mismatch_reasons(act_comp):
    candidates = []
    metric_specs = [
        ("acc_rms", "relative_diff_pct"),
        ("dom_freq", "relative_diff_pct"),
        ("spectral_cent", "relative_diff_pct"),
        ("kg_ratio", "relative_diff_pct"),
        ("tremor_prevalence", "delta_pct_points"),
    ]
    for metric, diff_key in metric_specs:
        metric_data = act_comp.get(metric, {})
        score = metric_data.get("match_score", float("nan"))
        if not np.isfinite(score):
            continue
        candidates.append((score, metric, metric_data.get(diff_key, float("nan"))))

    candidates.sort(key=lambda x: x[0])
    reasons = []
    for _, metric, diff in candidates:
        if metric == "acc_rms" and np.isfinite(diff) and abs(diff) >= 20:
            reasons.append("SIM too strong" if diff > 0 else "SIM too weak")
        elif metric == "dom_freq" and np.isfinite(diff) and abs(diff) >= 20:
            reasons.append("tremor too fast" if diff > 0 else "tremor too slow")
        elif metric == "spectral_cent" and np.isfinite(diff) and abs(diff) >= 20:
            reasons.append("frequency centroid too high" if diff > 0 else "frequency centroid too low")
        elif metric == "kg_ratio" and np.isfinite(diff) and abs(diff) >= 20:
            reasons.append("k_g too high" if diff > 0 else "k_g too low")
        elif metric == "tremor_prevalence" and np.isfinite(diff) and abs(diff) >= 10:
            reasons.append("too much tremor" if diff > 0 else "too little tremor")
        if len(reasons) == 2:
            break
    return reasons


def _activity_summary_line(activity, act_comp):
    label = _activity_match_label(act_comp.get("overall_match_score", float("nan")))
    reasons = _activity_mismatch_reasons(act_comp)
    if label == "Good match":
        detail = "SIM closely matches PD"
    elif reasons:
        detail = ", ".join(reasons)
    elif label == "Moderate mismatch":
        detail = "moderate mismatch"
    else:
        detail = "clear mismatch"
    return f"- {activity.capitalize()}: {label} ({detail})"


def generate_report(agg, comparison, suggestions, freq_by_wl,
                    sim_mode_stats=None, thr_sensitivity=None,
                    tremor_freq_stats=None, tremor_only_comp=None,
                    pd_vs_sim_activity=None, pd_upper_end=None,
                    pd_activity_scaling=None,
                    pd_coverage_sim=None,
                    pd_vs_sim_location=None,
                    pd_location_scaling=None,
                    kg_extremes=None):
    """Return (full_report, summary)."""
    lines = []
    def h1(t): lines.append(f"\n{'='*72}\n  {t}\n{'='*72}")
    def h2(t): lines.append(f"\n  -- {t} --")
    def ln(t=""): lines.append(t)

    lines.append("TREMOR CHARACTERIZATION & SIMULATION VALIDATION REPORT")
    lines.append(f"Bandpass: {TREMOR_LO_HZ}-{TREMOR_HI_HZ} Hz  |  Detection band: "
                 f"{TREMOR_BAND_LO}-{TREMOR_BAND_HI} Hz  |  FS={FS_ORIGINAL} Hz  |  "
                 f"Window={PRIMARY_WINDOW_SEC}s  |  Threshold={TREMOR_PRESENCE_THRESHOLD}")
    lines.append(f"SIM analysis modes included: {SIM_ANALYSIS_MODES}"
                 f"  (generated from: {SIM_AUGMODES})")

    # 1. Group overview
    h1("1. GROUP OVERVIEW  (CT / PD / SIM)")
    bg = agg["by_group"]
    for g in ("ct","pd","sim"):
        if g not in bg: continue
        h2(LABELS[g])
        s = bg[g]
        ln(f"  Acc RMS       : {_sr(s['acc_rms'])}")
        ln(f"  Gyro RMS      : {_sr(s['gyro_rms'])}")
        ln(f"  k_g           : {_sr(s['kg_ratio'])}")
        ln(f"  Dom. freq (Hz): {_sr(s['dom_freq'])}")
        ln(f"  Band ratio    : {_sr(s['band_ratio'])}")
        ln(f"  Tremor prev.  : {s.get('tremor_prevalence', float('nan')):.1%}")

    if sim_mode_stats:
        h2("SIM breakdown by augmentation mode")
        ln(f"  {'Mode':<18} {'N_wins':>8} {'Acc RMS med':>12} {'Gyro RMS med':>13} {'Dom Freq med':>13} {'Tremor%':>9}")
        for mode, ms in sorted(sim_mode_stats.items()):
            ln(f"  {mode:<18} {ms['n_windows']:>8} "
               f"{_fmt(ms['acc_rms']['median']):>12} "
               f"{_fmt(ms['gyro_rms']['median']):>13} "
               f"{_fmt(ms['dom_freq']['median']):>13} "
               f"{ms['tremor_prevalence']:>8.1%}")

    # 2. Tremor-only frequency analysis
    h1("2. TREMOR-ONLY FREQUENCY ANALYSIS  (tremor-present windows, arm locations)")
    ln("  NOTE: Tremor-specific parameters (e.g., frequency range) are derived from")
    ln("  tremor-present windows to better reflect actual tremor characteristics,")
    ln("  not diluted by non-tremor motion or noise.")
    if tremor_freq_stats:
        for g, glabel in (("pd", "PD (real)"), ("sim", "SIM (simulated)")):
            s = tremor_freq_stats.get(g, {})
            if not s: continue
            h2(f"{glabel}  (n={s.get('n',0)} tremor-present arm windows)")
            ln(f"  Dom. freq (Hz): {_sr(s['dom_freq'])}")
            ln(f"  Spectral cent : {_sr(s['spectral_cent'])}")
            ln(f"  Band power    : {_sr(s['band_power'])}")
            ln(f"  Acc RMS       : {_sr(s['acc_rms'])}")
            ln(f"  Gyro RMS      : {_sr(s['gyro_rms'])}")
            ln(f"  k_g           : {_sr(s['kg_ratio'])}")
        if tremor_only_comp:
            h2("SIM vs PD match (tremor-present arm windows only)")
            to_metrics = [("dom_freq","Dom. Freq (Hz)"),("acc_rms","Acc RMS"),
                          ("gyro_rms","Gyro RMS"),("kg_ratio","k_g")]
            hdr2 = f"  {'Metric':<18} {'PD median':>12} {'SIM median':>12} {'Rel. diff':>11} {'Match':>8}"
            ln(hdr2); ln("  " + "-"*len(hdr2.strip()))
            for m, label in to_metrics:
                c      = tremor_only_comp[m]
                pd_med  = c["pd"].get("median",  float("nan"))
                sim_med = c["sim"].get("median", float("nan"))
                if np.isfinite(pd_med) and np.isfinite(sim_med) and pd_med != 0:
                    rel_str = f"{(sim_med-pd_med)/abs(pd_med)*100:+.1f}%"
                else:
                    rel_str = "n/a"
                ln(f"  {label:<18} {_fmt(pd_med):>12} {_fmt(sim_med):>12} "
                   f"{rel_str:>11}  {_bar(c['match_score'], 10)}")
            ln(f"\n  Overall match (tremor-only): {_bar(tremor_only_comp['overall_match_score'])}")
    else:
        ln("  (not available)")

    # 2B. PD upper-end RMS statistics
    h1("2B. PD UPPER-END RMS STATISTICS  (tremor-present arm windows)")
    ln("  Goal: assess whether the PD dataset contains strong/severe tremor windows,")
    ln("  or is predominantly mild/moderate.")
    ln("  Source: tremor-present PD windows at arm locations (Upper+Lower Left/Right).")
    if pd_upper_end and "error" not in pd_upper_end:
        n_ue = pd_upper_end["n_tremor_present_arm_windows"]
        ln(f"  N tremor-present arm windows: {n_ue}")
        ln()

        # Acc RMS
        a = pd_upper_end["acc_rms"]
        ln(f"  Acc RMS -- upper-end reference:")
        ln(f"    Max observed          : {_fmt(a['max'])} m/s^2")
        ln(f"    Mean of top {a['n_top_50_used']:>3} windows : {_fmt(a['mean_top_50'])} m/s^2")
        ln(f"    Mean of top {a['n_top_100_used']:>3} windows : {_fmt(a['mean_top_100'])} m/s^2")
        if a['n_top_100_used'] < 100:
            ln(f"    (fewer than 100 windows available; used {a['n_top_100_used']})")

        ln()

        # Gyro RMS
        g_ue = pd_upper_end["gyro_rms"]
        ln(f"  Gyro RMS -- upper-end reference:")
        ln(f"    Max observed          : {_fmt(g_ue['max'])} (raw units)")
        ln(f"    Mean of top {g_ue['n_top_50_used']:>3} windows : {_fmt(g_ue['mean_top_50'])} (raw units)")
        ln(f"    Mean of top {g_ue['n_top_100_used']:>3} windows : {_fmt(g_ue['mean_top_100'])} (raw units)")
        if g_ue['n_top_100_used'] < 100:
            ln(f"    (fewer than 100 windows available; used {g_ue['n_top_100_used']})")

        ln()
        # Interpret severity
        score4_hi = SIM_SCORE_RMS_RANGE.get(4, (float("nan"), float("nan")))[1]
        if np.isfinite(score4_hi) and np.isfinite(a['max']):
            if a['max'] >= score4_hi:
                ln(f"  -> PD max acc RMS ({_fmt(a['max'])}) >= score-4 ceiling ({_fmt(score4_hi)}):  SEVERE tremor present.")
            elif a['mean_top_50'] >= SIM_SCORE_RMS_RANGE.get(3, (0,0))[0]:
                ln(f"  -> PD top-50 mean ({_fmt(a['mean_top_50'])}) reaches score-3+ range: moderate-severe windows present.")
            else:
                ln(f"  -> PD upper-end appears mild/moderate (top-50 mean = {_fmt(a['mean_top_50'])}).")
    elif pd_upper_end and "error" in pd_upper_end:
        ln(f"  ERROR: {pd_upper_end['error']}")
    else:
        ln("  (not available)")

    # 3. Frequency by window length
    h1("3. FREQUENCY ESTIMATION vs WINDOW LENGTH  (Welch PSD, ALL windows)")
    ln(f"  {'Window':>8}  {'PD median':>12}  {'PD p5-p95':>18}  {'CT median':>12}")
    for wl in sorted(freq_by_wl.keys()):
        pd_s = freq_by_wl[wl]["pd"]; ct_s = freq_by_wl[wl]["ct"]
        ln(f"  {wl:>6.1f}s   {_fmt(pd_s['median']):>12}"
           f"  [{_fmt(pd_s['p5'])}, {_fmt(pd_s['p95'])}]  {_fmt(ct_s['median']):>12}")
    ln(f"\n  Current simulation FREQ_RANGE_HZ = {SIM_FREQ_RANGE_HZ}")

    # 4. Tremor prevalence
    h1(f"4. TREMOR-PRESENT WINDOWS  (threshold = {TREMOR_PRESENCE_THRESHOLD:.3f})")
    tp = comparison["tremor_prevalence"]
    ln(f"  PD:  {tp['pd_mean']:.1%} of windows classified as tremor-present")
    ln(f"  CT:  {tp['ct_mean']:.1%}")
    ln(f"  SIM: {tp['sim_mean']:.1%}")
    if np.isfinite(tp['ct_mean']) and tp['ct_mean'] > 0.15:
        ln(f"  *** WARNING: CT prevalence {tp['ct_mean']:.1%} > 15% -- "
           f"threshold may be too low (detecting noise as tremor) ***")
    if np.isfinite(tp['pd_mean']) and tp['pd_mean'] < 0.25:
        ln(f"  *** WARNING: PD prevalence {tp['pd_mean']:.1%} < 25% -- "
           f"threshold may be too high (missing tremor) ***")
    h2("Per PD subject")
    ln(f"  {'Subject':<12} {'N_wins':>7} {'Tremor%':>9} {'acc_rms':>9} {'freq':>8}")
    for s in sorted(agg["per_subject"], key=lambda x: x["subject"]):
        ln(f"  {s['subject']:<12} {s['n_windows']:>7} "
           f"{s['tremor_prev']:>8.1%} {_fmt(s['acc_rms_med']):>9} {_fmt(s['dom_freq_med']):>8}")

    if thr_sensitivity:
        h2("Threshold sensitivity (tremor prevalence at different thresholds)")
        ln(f"  {'Threshold':>12}  {'CT prev':>9}  {'PD prev':>9}  {'SIM prev':>10}  Notes")
        for thr, row in sorted(thr_sensitivity.items()):
            ct_p  = row.get("ct",  float("nan"))
            pd_p  = row.get("pd",  float("nan"))
            sim_p = row.get("sim", float("nan"))
            flags = []
            if np.isfinite(ct_p) and ct_p > 0.15: flags.append("CT>15% noise?")
            if np.isfinite(pd_p) and pd_p < 0.25: flags.append("PD<25% missed?")
            marker = " << current" if abs(thr - TREMOR_PRESENCE_THRESHOLD) < 1e-6 else ""
            note = ", ".join(flags) + marker
            ln(f"  {thr:>12.2f}  {ct_p:>8.1%}  {pd_p:>8.1%}  {sim_p:>9.1%}  {note}")

    # 5. Activity analysis
    h1("5. ACTIVITY-SPECIFIC ANALYSIS  (PD)")
    acts = agg.get("activities", [])
    if "pd" in agg["by_activity"]:
        ln(f"  {'Activity':<14} {'Acc RMS':>9} {'Dom freq':>10} {'Tremor%':>9}")
        for act in acts:
            s = agg["by_activity"]["pd"].get(act, {})
            tp_val   = s.get("tremor_prevalence", float("nan"))
            med_acc  = s.get("acc_rms",{}).get("median", float("nan"))
            med_freq = s.get("dom_freq",{}).get("median", float("nan"))
            ln(f"  {act:<14} {_fmt(med_acc):>9} {_fmt(med_freq):>10} {tp_val:>8.1%}")
    ln("\n  Expected: calibration (rest) has LOWER RMS and HIGHER tremor%")
    ln("  (Parkinson tremor is a RESTING tremor - reduced during active movement)")

    h1("5B. ACTIVITY-WISE SIMULATION VALIDATION (PD vs SIM)")
    if pd_vs_sim_activity:
        for act in sorted(pd_vs_sim_activity.keys()):
            comp = pd_vs_sim_activity[act]
            ln(f"\nActivity: {act}")
            ln(f"  Windows        : PD={comp.get('n_pd_windows', 0)}   SIM={comp.get('n_sim_windows', 0)}")

            acc = comp["acc_rms"]
            ln(f"  Acc RMS        : PD={_fmt(acc['pd'].get('median')):<7} "
               f"SIM={_fmt(acc['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(acc.get('relative_diff_pct')):>7})   {_bar(acc.get('match_score', float('nan')), 10)}")

            freq = comp["dom_freq"]
            ln(f"  Frequency (Hz) : PD={_fmt(freq['pd'].get('median')):<7} "
               f"SIM={_fmt(freq['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(freq.get('relative_diff_pct')):>7})   {_bar(freq.get('match_score', float('nan')), 10)}")

            cent = comp["spectral_cent"]
            ln(f"  Spectral cent  : PD={_fmt(cent['pd'].get('median')):<7} "
               f"SIM={_fmt(cent['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(cent.get('relative_diff_pct')):>7})   {_bar(cent.get('match_score', float('nan')), 10)}")

            kg = comp["kg_ratio"]
            ln(f"  k_g            : PD={_fmt(kg['pd'].get('median')):<7} "
               f"SIM={_fmt(kg['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(kg.get('relative_diff_pct')):>7})   {_bar(kg.get('match_score', float('nan')), 10)}")

            prev = comp["tremor_prevalence"]
            ln(f"  Tremor prev    : PD={prev.get('pd', float('nan')):>5.1%}   "
               f"SIM={prev.get('sim', float('nan')):>5.1%}   "
                f"({_fmt_pp_diff(prev.get('delta_pct_points', float('nan'))):>9})   "
               f"{_bar(prev.get('match_score', float('nan')), 10)}")

            ln(f"  Overall match  : {_bar(comp.get('overall_match_score', float('nan')))}  "
               f"{comp.get('match_label', 'Unavailable')}")

    h1("5C. ACTIVITY-DEPENDENT TREMOR SCALING FROM PD DATA")
    ln("  Method summary:")
    ln("    1) Tremor-present PD windows only, arm locations only")
    ln("    2) Per subject/activity median tremor amplitude (acc_rms_band37)")
    ln("    3) Per subject reference activity = highest median amplitude")
    ln("    4) Subject severity from reference amplitude")
    ln("    5) Activity ratios aggregated by severity")
    if pd_activity_scaling and "error" not in pd_activity_scaling:
        activities = pd_activity_scaling.get("activities", [])
        agg_sc = pd_activity_scaling.get("aggregated_ratios_by_severity", {})
        suggested = pd_activity_scaling.get("suggested_BETA_ACTIVITY_BY_SCORE", {})
        ln(f"\n  Subjects contributing: {pd_activity_scaling.get('n_subjects_total', 0)}")
        ln(f"  Amplitude metric: {pd_activity_scaling.get('method', {}).get('metric', 'n/a')}")

        for score in (1, 2, 3, 4):
            h2(f"Severity score {score} - aggregated activity ratios")
            ln(f"  {'Activity':<14} {'Median ratio':>12} {'Mean ratio':>11} {'N subj':>8}")
            for act in activities:
                row = agg_sc.get(score, {}).get(act, {})
                ln(f"  {act:<14} {_fmt(row.get('median_ratio')):>12} {_fmt(row.get('mean_ratio')):>11} {int(row.get('n_subjects', 0)):>8}")

        h2("Suggested BETA_ACTIVITY_BY_SCORE (median-based)")
        for score in (1, 2, 3, 4):
            row = suggested.get(score, {})
            if not row:
                continue
            row_txt = ", ".join([f"{act}={_fmt(row.get(act), 3)}" for act in activities])
            ln(f"  Score {score}: {row_txt}")
    elif pd_activity_scaling and "error" in pd_activity_scaling:
        ln(f"  ERROR: {pd_activity_scaling['error']}")
    else:
        ln("  (not available)")

    h1("5D. LOCATION-WISE SIMULATION VALIDATION (PD vs SIM)")
    if pd_vs_sim_location and pd_vs_sim_location.get("per_location"):
        loc_comp = pd_vs_sim_location["per_location"]
        for loc in SENSOR_SHEETS:
            comp = loc_comp.get(loc)
            if not comp:
                continue
            ln(f"\nLocation: {loc}")
            ln(f"  Windows        : PD={comp.get('n_pd_windows', 0)}   SIM={comp.get('n_sim_windows', 0)}")

            acc = comp["acc_rms"]
            ln(f"  Acc RMS        : PD={_fmt(acc['pd'].get('median')):<7} "
               f"SIM={_fmt(acc['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(acc.get('relative_diff_pct')):>7})   {_bar(acc.get('match_score', float('nan')), 10)}")

            gyro = comp["gyro_rms"]
            ln(f"  Gyro RMS       : PD={_fmt(gyro['pd'].get('median')):<7} "
               f"SIM={_fmt(gyro['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(gyro.get('relative_diff_pct')):>7})   {_bar(gyro.get('match_score', float('nan')), 10)}")

            freq = comp["dom_freq"]
            ln(f"  Frequency (Hz) : PD={_fmt(freq['pd'].get('median')):<7} "
               f"SIM={_fmt(freq['sim'].get('median')):<7} "
               f"({_fmt_pct_diff(freq.get('relative_diff_pct')):>7})   {_bar(freq.get('match_score', float('nan')), 10)}")

            prev = comp["tremor_prevalence"]
            ln(f"  Tremor prev    : PD={prev.get('pd', float('nan')):>5.1%}   "
               f"SIM={prev.get('sim', float('nan')):>5.1%}   "
               f"({_fmt_pp_diff(prev.get('delta_pct_points', float('nan'))):>9})   "
               f"{_bar(prev.get('match_score', float('nan')), 10)}")

            ln(f"  Overall match  : {_bar(comp.get('overall_match_score', float('nan')))}  "
               f"{comp.get('match_label', 'Unavailable')}")

        worst = pd_vs_sim_location.get("worst_matching_locations", [])
        if worst:
            h2("Worst-matching locations")
            for i, row in enumerate(worst, 1):
                ln(f"  {i}. {row.get('location', 'n/a'):<12}  score={_fmt(row.get('overall_match_score'), 3)}  {row.get('match_label', 'Unavailable')}")
    else:
        ln("  (not available)")

    h1("5E. BODY-LOCATION SCALING FROM PD DATA")
    ln("  Method summary:")
    ln("    1) Tremor-present PD windows only, all body locations")
    ln("    2) Per subject/location median tremor amplitude (acc_rms_band37)")
    ln("    3) Per subject reference location = stronger of Lower Right / Lower Left")
    ln("    4) Subject severity from reference amplitude")
    ln("    5) Location ratios aggregated by severity")
    if pd_location_scaling and "error" not in pd_location_scaling:
        locations = pd_location_scaling.get("locations", [])
        agg_sc = pd_location_scaling.get("aggregated_ratios_by_severity", {})
        suggested = pd_location_scaling.get("suggested_BETA_LOCATION_BY_SCORE", {})
        ln(f"\n  Subjects contributing: {pd_location_scaling.get('n_subjects_total', 0)}")
        ln(f"  Amplitude metric: {pd_location_scaling.get('method', {}).get('metric', 'n/a')}")

        for score in (1, 2, 3, 4):
            h2(f"Severity score {score} - aggregated location ratios")
            ln(f"  {'Location':<14} {'Median ratio':>12} {'Mean ratio':>11} {'N subj':>8}")
            for loc in locations:
                row = agg_sc.get(score, {}).get(loc, {})
                ln(f"  {loc:<14} {_fmt(row.get('median_ratio')):>12} {_fmt(row.get('mean_ratio')):>11} {int(row.get('n_subjects', 0)):>8}")

        h2("Suggested BETA_LOCATION_BY_SCORE (median-based)")
        for score in (1, 2, 3, 4):
            row = suggested.get(score, {})
            if not row:
                continue
            row_txt = ", ".join([f"{loc}={_fmt(row.get(loc), 3)}" for loc in locations])
            ln(f"  Score {score}: {row_txt}")
    elif pd_location_scaling and "error" in pd_location_scaling:
        ln(f"  ERROR: {pd_location_scaling['error']}")
    else:
        ln("  (not available)")

    h1("5F. PD COVERAGE BY SIM DISTRIBUTION")
    ln("  Method: tremor-present arm windows only (PD and SIM).")
    ln("  SIM coverage interval per feature = p5-p95, then evaluate where PD falls.")
    if pd_coverage_sim and "error" not in pd_coverage_sim:
        fc = pd_coverage_sim.get("feature_coverage", {})
        ln(f"\n  {'Feature':<14} {'SIM p5':>10} {'SIM p95':>10} {'PD in':>8} {'PD below':>10} {'PD above':>10}")
        for feat in ("acc_rms", "dom_freq", "spectral_cent"):
            row = fc.get(feat, {})
            ln(f"  {feat:<14} {_fmt(row.get('sim_p5')):>10} {_fmt(row.get('sim_p95')):>10} "
               f"{_fmt(row.get('pd_pct_inside'),1):>7}% {_fmt(row.get('pd_pct_below'),1):>9}% {_fmt(row.get('pd_pct_above'),1):>9}%")

        jc = pd_coverage_sim.get("joint_coverage", {})
        j1 = jc.get("acc_rms__dom_freq", {})
        j2 = jc.get("acc_rms__spectral_cent", {})
        ln(f"\n  Joint coverage (PD inside BOTH intervals):")
        ln(f"    Acc RMS + Dom. freq    : {_fmt(j1.get('pct_inside_both'),1)}%  (n={int(j1.get('n_pd', 0))})")
        ln(f"    Acc RMS + Spectral cent: {_fmt(j2.get('pct_inside_both'),1)}%  (n={int(j2.get('n_pd', 0))})")

        poor = pd_coverage_sim.get("poorly_covered_pd_windows", {})
        ln(f"\n  Poorly covered PD windows (outside SIM interval for acc_rms OR dom_freq):")
        ln(f"    N windows: {int(poor.get('n_windows', 0))}  ({_fmt(poor.get('pct_of_pd_tremor_present_arm'),1)}% of PD tremor-present arm windows)")
        ln(f"    Median acc_rms      : {_fmt(poor.get('median_acc_rms'))}")
        ln(f"    Median dom_freq     : {_fmt(poor.get('median_dom_freq'))}")
        ln(f"    Median spectral_cent: {_fmt(poor.get('median_spectral_cent'))}")
        ln(f"    Median kg_ratio     : {_fmt(poor.get('median_kg_ratio'))}")

        ln("\n    Activity distribution (poor subset):")
        for row in poor.get("activity_distribution", [])[:8]:
            ln(f"      {row.get('activity','?'):<12} n={int(row.get('n_windows', 0)):>5}  ({_fmt(row.get('pct_windows'),1)}%)")

        ln("\n    Subject distribution (poor subset):")
        for row in poor.get("subject_distribution", [])[:10]:
            ln(f"      {row.get('subject','?'):<12} n={int(row.get('n_windows', 0)):>5}  ({_fmt(row.get('pct_windows'),1)}%)")

        bins = pd_coverage_sim.get("binned_summary", {})
        rb = bins.get("rms_bins", {})
        fb = bins.get("frequency_bins", {})
        ln("\n  Binned PD vs SIM (tremor-present arm windows):")
        ln("    RMS bins (%):")
        for label in ("low", "medium", "high", "very_high"):
            ln(f"      {label:<10} PD={_fmt(rb.get('pd', {}).get(label),1)}%   SIM={_fmt(rb.get('sim', {}).get(label),1)}%")
        ln("    Frequency bins (%):")
        for label in ("lt_3p5_hz", "hz_3p5_to_5", "hz_5_to_7", "gt_7_hz"):
            ln(f"      {label:<10} PD={_fmt(fb.get('pd', {}).get(label),1)}%   SIM={_fmt(fb.get('sim', {}).get(label),1)}%")
    elif pd_coverage_sim and "error" in pd_coverage_sim:
        ln(f"  ERROR: {pd_coverage_sim['error']}")
    else:
        ln("  (not available)")

    # 6. Left-right asymmetry
    h1("6. LEFT-RIGHT ASYMMETRY  (PD subjects)")
    asym = agg["lr_asymmetry"]
    if asym:
        ln(f"  {'Subject':<12} {'Activity':<14} {'Asym Upper':>12} {'Asym Lower':>12}")
        for r in sorted(asym, key=lambda x:(x["subject"],x["activity"])):
            ln(f"  {r['subject']:<12} {r['activity']:<14} "
               f"{_fmt(r['asym_upper_arm']):>12} {_fmt(r['asym_lower_arm']):>12}")

    # 7. Upper vs lower arm
    h1("7. UPPER vs LOWER ARM  (PD)")
    uvl = agg["upper_vs_lower"]
    ln(f"  Upper arm acc RMS mean:  {_fmt(uvl['mean_acc_upper'])}")
    ln(f"  Lower arm acc RMS mean:  {_fmt(uvl['mean_acc_lower'])}")
    ln(f"  Lower / Upper ratio:     {_fmt(uvl['ratio_acc'])}")
    ln(f"  Upper gyro RMS mean:     {_fmt(uvl['mean_gyro_upper'])}")
    ln(f"  Lower gyro RMS mean:     {_fmt(uvl['mean_gyro_lower'])}")
    ln(f"  Lower / Upper gyro:      {_fmt(uvl['ratio_gyro'])}")
    ln(f"\n  Current sim ANKLE_RATIO_BY_SCORE (ankle vs arm):")
    for score, ratio in pk_config.ANKLE_RATIO_BY_SCORE.items():
        ln(f"    Score {score}: {ratio}")

    # 8. Comparison
    h1("8. REAL vs SIMULATED - SIDE-BY-SIDE COMPARISON  (ALL windows)")
    comp_metrics = [("acc_rms","Acc RMS"),("gyro_rms","Gyro RMS"),
                    ("dom_freq","Dom. Freq (Hz)"),("kg_ratio","k_g"),
                    ("band_ratio","Band ratio")]
    hdr = f"  {'Metric':<18} {'CT median':>12} {'PD median':>12} {'SIM median':>12} {'Match':>8}"
    ln(hdr); ln("  " + "-"*len(hdr.strip()))
    for m, label in comp_metrics:
        c = comparison[m]
        pd_med  = c["pd"].get("median",  float("nan"))
        ct_med  = c["ct"].get("median",  float("nan"))
        sim_med = c["sim"].get("median", float("nan"))
        ms      = c["match_score_sim_vs_pd"]
        ln(f"  {label:<18} {_fmt(ct_med):>12} {_fmt(pd_med):>12} {_fmt(sim_med):>12}  {_bar(ms, 10)}")
    ln(f"\n  Overall simulation match score: {_bar(comparison['overall_match_score'])}")

    h2("Top 3 mismatches (worst SIM-vs-PD fit)")
    sorted_m = sorted(comp_metrics,
                      key=lambda x: comparison[x[0]]["match_score_sim_vs_pd"]
                      if np.isfinite(comparison[x[0]]["match_score_sim_vs_pd"]) else -1.0)
    ln(f"  {'Rank':<6} {'Metric':<18} {'PD median':>12} {'SIM median':>12} {'Rel. diff':>11} Score")
    ln("  " + "-"*68)
    for rank, (m, label) in enumerate(sorted_m[:3], 1):
        c       = comparison[m]
        pd_med  = c["pd"].get("median",  float("nan"))
        sim_med = c["sim"].get("median", float("nan"))
        if np.isfinite(pd_med) and np.isfinite(sim_med) and pd_med != 0:
            rel_diff = (sim_med - pd_med) / abs(pd_med) * 100.0
            rel_str  = f"{rel_diff:+.1f}%"
        else:
            rel_str = "n/a"
        ms = c["match_score_sim_vs_pd"]
        ln(f"  {rank:<6} {label:<18} {_fmt(pd_med):>12} {_fmt(sim_med):>12} {rel_str:>11}  {_bar(ms, 8)}")

    # 8B. k_g extremes by location
    h1("8B. K_G EXTREMES BY LOCATION  (top-N and bottom-N per-segment k_g = gyro_rms / acc_rms)")
    ln("  Computed on bandpass-filtered (2-12 Hz) RMS values per window.")
    ln("  Only windows with positive, finite k_g are included.")
    _kg_n = None
    if kg_extremes:
        for g in ("ct", "pd", "sim"):
            if g not in kg_extremes:
                continue
            h2(f"{LABELS[g]}")
            g_data = kg_extremes[g]
            # Infer N from _all entry
            _all = g_data.get("_all", {})
            _kg_n = int(_all.get("top_n_used", 100)) if _all else 100
            ln(f"  {'Location':<16} {'N total':>9} {'Top-{n} mean':>13} {'Bottom-{n} mean':>16}".replace("{n}", str(_kg_n)))
            ln("  " + "-" * 58)
            # Overall row first
            if "_all" in g_data:
                r = g_data["_all"]
                ln(f"  {'ALL (combined)':<16} {r.get('n_total',0):>9} "
                   f"{_fmt(r.get('top_n_mean')):>13} {_fmt(r.get('bottom_n_mean')):>16}")
            # Per-location rows
            for loc in SENSOR_SHEETS:
                if loc not in g_data:
                    continue
                r = g_data[loc]
                code = LOCATION_CODE.get(loc, loc)
                ln(f"  {loc:<16} {r.get('n_total',0):>9} "
                   f"{_fmt(r.get('top_n_mean')):>13} {_fmt(r.get('bottom_n_mean')):>16}  [{code}]")
    else:
        ln("  (not available)")

    # 9. Suggestions
    h1("9. DATA-DRIVEN CONFIG SUGGESTIONS  (from tremor-present PD windows)")
    ln("  DISCLAIMER: These values are empirical estimates derived from this dataset")
    ln("  and should be treated as initial guidelines, not absolute ground truth.")
    ln("  Always validate against additional data before modifying simulation parameters.")
    if "error" in suggestions:
        ln(f"\n  ERROR: {suggestions['error']}")
    else:
        ln(f"\n  N tremor-present windows used: {suggestions['n_tremor_present_windows']}")
        ln(f"\n  NOTE: {suggestions['note_kg_units']}")
        ln(f"\n  Suggested SCORE_RMS_RANGE (quartile-based from PD tremor-present):")
        for score, rng2 in suggestions["suggested_SCORE_RMS_RANGE"].items():
            cur = SIM_SCORE_RMS_RANGE.get(score, (None,None))
            ln(f"    Score {score}:  suggested {rng2}   current {cur}")
        ln(f"\n  Suggested FREQ_RANGE_HZ: {suggestions['suggested_FREQ_RANGE_HZ']}")
        ln(f"  Source: {suggestions.get('suggested_FREQ_RANGE_HZ_source', 'n/a')}")
        ln(f"  Current  FREQ_RANGE_HZ: {suggestions['current_FREQ_RANGE_HZ']}")
        ln(f"\n  Suggested k_g (median):    {_fmt(suggestions['suggested_kg_median'])}")
        ln(f"  Suggested k_g (p25-p75):   {suggestions['suggested_kg_p25_p75']}")
        ln(f"  Current   k_g mode:        {SIM_KG_CONFIG_NOTE}")
        ln(f"  Current   k_g by severity: {suggestions['current_KG_values']}")

    full_report = "\n".join(lines)

    # Summary
    s_lines = []
    s_lines.append("=" * 60)
    s_lines.append("SIMULATION DIAGNOSTIC SUMMARY")
    s_lines.append("=" * 60)
    s_lines.append(f"\nSIM modes in this run: {SIM_ANALYSIS_MODES}")

    oms = comparison.get("overall_match_score", float("nan"))
    s_lines.append(f"\nOVERALL MATCH SCORE (SIM vs PD): {_bar(oms)}")
    s_lines.append("\nPer-metric match:")
    for m, label in comp_metrics:
        ms = comparison[m]["match_score_sim_vs_pd"]
        s_lines.append(f"  {label:<18}: {_bar(ms, 12)}")

    s_lines.append("\nBIGGEST MISMATCHES:")
    sorted_metrics = sorted(comp_metrics,
                             key=lambda x: comparison[x[0]]["match_score_sim_vs_pd"]
                             if np.isfinite(comparison[x[0]]["match_score_sim_vs_pd"]) else 1.0)
    for m, label in sorted_metrics[:3]:
        c = comparison[m]
        s_lines.append(f"  {label:<18}: PD={_fmt(c['pd'].get('median'))}"
                       f"  SIM={_fmt(c['sim'].get('median'))}"
                       f"  score={_fmt(c['match_score_sim_vs_pd'])}")

    s_lines.append("\nCONCRETE RECOMMENDATIONS:")
    if not np.isfinite(comparison["dom_freq"]["match_score_sim_vs_pd"]) or \
       comparison["dom_freq"]["match_score_sim_vs_pd"] < 0.7:
        sug_freq = suggestions.get("suggested_FREQ_RANGE_HZ", ("?","?"))
        s_lines.append(f"  1. FREQ: Change FREQ_RANGE_HZ from {SIM_FREQ_RANGE_HZ} -> {sug_freq}")
    if not np.isfinite(comparison["acc_rms"]["match_score_sim_vs_pd"]) or \
       comparison["acc_rms"]["match_score_sim_vs_pd"] < 0.7:
        sug_rms = suggestions.get("suggested_SCORE_RMS_RANGE", {})
        s_lines.append(f"  2. RMS:  Score 2 -> {sug_rms.get(2)}, Score 3 -> {sug_rms.get(3)}")
    if not np.isfinite(comparison["kg_ratio"]["match_score_sim_vs_pd"]) or \
       comparison["kg_ratio"]["match_score_sim_vs_pd"] < 0.7:
        sug_kg = suggestions.get("suggested_kg_median", "?")
        kg_v = float(sug_kg) if isinstance(sug_kg, (int, float)) else float("nan")
        kg_deg = kg_v * 180 / 3.141592653589793 if np.isfinite(kg_v) else float("nan")
        s_lines.append(f"  3. K_G:  Measured median k_g = {_fmt(sug_kg)}")
        s_lines.append(f"          -> If gyro in rad/s: sim k_g should be ~{_fmt(kg_deg)}")
    tp = comparison["tremor_prevalence"]
    if abs(tp.get("sim_mean",0) - tp.get("pd_mean",0)) > 0.1:
        s_lines.append(f"  4. PREVALENCE: PD={tp['pd_mean']:.1%} vs SIM={tp['sim_mean']:.1%}"
                       f"  -> adjust TREMOR_PRESENCE_THRESHOLD or SIM_AUGMODE")

    s_lines.append(f"\nACTIVITY RESTING TREMOR CHECK:")
    if "pd" in agg["by_activity"]:
        calib = agg["by_activity"]["pd"].get("calibration", {})
        other_acts = [agg["by_activity"]["pd"].get(a,{}) for a in ["key","cardigan","toast"]]
        c_med  = calib.get("acc_rms",{}).get("median", float("nan"))
        o_meds = [o.get("acc_rms",{}).get("median", float("nan")) for o in other_acts if o]
        if np.isfinite(c_med) and o_meds:
            diff = float(np.nanmean(o_meds)) - c_med
            direction = "LOWER at rest (resting tremor confirmed)" if diff > 0 else "HIGHER at rest (unexpected)"
            s_lines.append(f"  Acc RMS calibration={_fmt(c_med)} vs active mean={_fmt(float(np.nanmean(o_meds)))} -> {direction}")

    if pd_vs_sim_activity:
        s_lines.append("\nACTIVITY-LEVEL FINDINGS:")
        for act in sorted(pd_vs_sim_activity.keys()):
            s_lines.append(f"  {_activity_summary_line(act, pd_vs_sim_activity[act])}")

    if pd_upper_end and "error" not in pd_upper_end:
        a = pd_upper_end["acc_rms"]
        n_ue = pd_upper_end["n_tremor_present_arm_windows"]
        s_lines.append(f"\nPD UPPER-END TREMOR AMPLITUDE  (n={n_ue} tremor-present arm windows):")
        s_lines.append(f"  Acc RMS max          : {_fmt(a['max'])} m/s^2")
        s_lines.append(f"  Acc RMS mean top-50  : {_fmt(a['mean_top_50'])} m/s^2")
        s_lines.append(f"  Acc RMS mean top-100 : {_fmt(a['mean_top_100'])} m/s^2")
        g_ue = pd_upper_end["gyro_rms"]
        s_lines.append(f"  Gyro RMS max         : {_fmt(g_ue['max'])}")
        s_lines.append(f"  Gyro RMS mean top-50 : {_fmt(g_ue['mean_top_50'])}")
        # Severity interpretation
        score4_hi = SIM_SCORE_RMS_RANGE.get(4, (float("nan"), float("nan")))[1]
        if np.isfinite(score4_hi) and np.isfinite(a['max']):
            if a['max'] >= score4_hi:
                s_lines.append(f"  -> SEVERE tremor present in PD data (max >= score-4 ceiling {_fmt(score4_hi)})")
            elif a['mean_top_50'] >= SIM_SCORE_RMS_RANGE.get(3, (0,0))[0]:
                s_lines.append(f"  -> Moderate-severe windows present (top-50 mean reaches score-3+ range)")
            else:
                s_lines.append(f"  -> PD upper-end appears mild/moderate (check threshold or PD severity)")

    if pd_activity_scaling and "error" not in pd_activity_scaling:
        suggested = pd_activity_scaling.get("suggested_BETA_ACTIVITY_BY_SCORE", {})

        def _collect_metric(act_name):
            vals = []
            for score in (1, 2, 3, 4):
                v = suggested.get(score, {}).get(act_name, float("nan"))
                if np.isfinite(v):
                    vals.append(float(v))
            return float(np.median(vals)) if vals else float("nan")

        cal_med = _collect_metric("calibration")
        key_med = _collect_metric("key")
        card_med = _collect_metric("cardigan")
        toast_med = _collect_metric("toast")

        s_lines.append("\nPD-DERIVED ACTIVITY SCALING PATTERN:")
        if np.isfinite(cal_med):
            s_lines.append(f"  calibration: {_fmt(cal_med)} (consistently reduced vs reference)")
        if np.isfinite(key_med):
            s_lines.append(f"  key        : {_fmt(key_med)} (moderately reduced)")
        if np.isfinite(card_med) or np.isfinite(toast_med):
            s_lines.append(f"  cardigan/toast: {_fmt(card_med)} / {_fmt(toast_med)} (near baseline)")

    if pd_coverage_sim and "error" not in pd_coverage_sim:
        fc = pd_coverage_sim.get("feature_coverage", {})
        freq_cov = fc.get("dom_freq", {})
        acc_cov = fc.get("acc_rms", {})
        poor = pd_coverage_sim.get("poorly_covered_pd_windows", {})

        s_lines.append("\nPD COVERAGE BY SIM DISTRIBUTION:")
        s_lines.append(
            f"  PD inside SIM p5-p95: freq={_fmt(freq_cov.get('pd_pct_inside'),1)}%  acc={_fmt(acc_cov.get('pd_pct_inside'),1)}%"
        )
        if np.isfinite(freq_cov.get("pd_pct_below", float("nan"))) and np.isfinite(freq_cov.get("pd_pct_above", float("nan"))):
            if freq_cov.get("pd_pct_below", 0.0) > freq_cov.get("pd_pct_above", 0.0):
                s_lines.append("  Most unmatched PD windows are lower-frequency than SIM.")
            elif freq_cov.get("pd_pct_above", 0.0) > freq_cov.get("pd_pct_below", 0.0):
                s_lines.append("  Most unmatched PD windows are higher-frequency than SIM.")
        s_lines.append(
            f"  Poorly covered subset size: {int(poor.get('n_windows', 0))} windows ({_fmt(poor.get('pct_of_pd_tremor_present_arm'),1)}%)."
        )

    if pd_vs_sim_location and pd_vs_sim_location.get("worst_matching_locations"):
        worst = pd_vs_sim_location.get("worst_matching_locations", [])
        top = worst[0] if worst else {}
        if top:
            s_lines.append("\nLOCATION-WISE PD vs SIM:")
            s_lines.append(
                f"  Worst location match: {top.get('location', 'n/a')} (score={_fmt(top.get('overall_match_score'), 3)})."
            )

    if pd_location_scaling and "error" not in pd_location_scaling:
        suggested_loc = pd_location_scaling.get("suggested_BETA_LOCATION_BY_SCORE", {})

        def _collect_loc(loc_name):
            vals = []
            for score in (1, 2, 3, 4):
                v = suggested_loc.get(score, {}).get(loc_name, float("nan"))
                if np.isfinite(v):
                    vals.append(float(v))
            return float(np.median(vals)) if vals else float("nan")

        ll_med = _collect_loc("Lower Left")
        lr_med = _collect_loc("Lower Right")
        ul_med = _collect_loc("Upper Left")
        ur_med = _collect_loc("Upper Right")
        hd_med = _collect_loc("Head")

        s_lines.append("\nPD-DERIVED LOCATION SCALING PATTERN:")
        s_lines.append(f"  lower arm (LL/LR): {_fmt(ll_med)} / {_fmt(lr_med)} (reference-strongest)")
        s_lines.append(f"  upper arm (UL/UR): {_fmt(ul_med)} / {_fmt(ur_med)} (reduced vs lower arm)")
        s_lines.append(f"  head            : {_fmt(hd_med)} (typically much lower)")

    summary = "\n".join(s_lines)
    return full_report, summary


# ===========================================================================
# SECTION 11 -- MAIN
# ===========================================================================

def _jsonify(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict):       return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):       return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)) and not np.isfinite(obj): return None
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj


def main():
    print("=" * 60)
    print("Tremor Characterization & Simulation Validation Tool")
    print("=" * 60)

    print("\n[1/7] Loading XLS trial data...")
    records = load_all_trials(DATA_DIR)
    n_pd = len({r["subject"] for r in records if r["group"]=="pd"})
    n_ct = len({r["subject"] for r in records if r["group"]=="ct"})
    print(f"  PD subjects: {n_pd}  |  CT subjects: {n_ct}")

    print("\n[2/7] Generating simulated tremor windows...")
    sim_windows = generate_simulated_windows(records)
    # Apply analysis-mode filter: only keep the modes configured in SIM_ANALYSIS_MODES
    sim_windows = [w for w in sim_windows if w.get("_sim_augmode") in SIM_ANALYSIS_MODES]
    print(f"  Retained {len(sim_windows)} windows for analysis  (SIM_ANALYSIS_MODES={SIM_ANALYSIS_MODES})")

    print("\n[3/7] Computing per-window features...")
    windows = analyze_windows(records, sim_windows)
    n_by_g = {g: len(_group_subset(windows, g)) for g in ("ct","pd","sim")}
    print(f"  Windows: CT={n_by_g['ct']}  PD={n_by_g['pd']}  SIM={n_by_g['sim']}")

    print("\n[4/7] Frequency analysis across window lengths...")
    freq_by_wl = analyze_frequency_by_window_length(records)

    print("\n[5/7] Aggregating statistics...")
    agg = aggregate_results(windows)

    print("\n[6/7] Building comparison, config suggestions, and sensitivity analysis...")
    comparison          = build_comparison(agg)
    pd_vs_sim_activity  = compare_pd_vs_sim_per_activity(windows)
    suggestions         = build_suggestions(agg, windows)
    pd_activity_scaling = derive_activity_scaling_from_pd(windows, metric="acc_rms_band37")
    pd_vs_sim_location  = compare_pd_vs_sim_per_location(windows)
    pd_location_scaling = derive_location_scaling_from_pd(windows, metric="acc_rms_band37")
    pd_coverage_sim     = derive_pd_coverage_by_sim(windows)
    sim_mode_stats      = aggregate_by_sim_mode(windows)
    thr_sensitivity     = threshold_sensitivity(windows)
    tremor_freq_stats   = tremor_only_freq_analysis(windows)
    tremor_only_comp    = build_tremor_only_comparison(tremor_freq_stats)
    pd_upper_end        = pd_upper_end_rms_stats(windows)
    kg_extremes         = kg_extremes_by_location(windows, n=100)

    print("\n[7/7] Generating plots...")
    plot_group_distributions(windows)
    plot_per_location(windows)
    plot_per_activity(windows)
    plot_psd_comparison(records, sim_windows)
    plot_per_subject(agg)
    plot_lr_asymmetry(agg)
    plot_kg_scatter(windows)
    plot_freq_window_comparison(freq_by_wl)

    full_report, summary = generate_report(agg, comparison, suggestions, freq_by_wl,
                                           sim_mode_stats=sim_mode_stats,
                                           thr_sensitivity=thr_sensitivity,
                                           tremor_freq_stats=tremor_freq_stats,
                                           tremor_only_comp=tremor_only_comp,
                                           pd_vs_sim_activity=pd_vs_sim_activity,
                                           pd_upper_end=pd_upper_end,
                                           pd_activity_scaling=pd_activity_scaling,
                                           pd_vs_sim_location=pd_vs_sim_location,
                                           pd_location_scaling=pd_location_scaling,
                                           pd_coverage_sim=pd_coverage_sim,
                                           kg_extremes=kg_extremes)

    (OUTPUT_DIR / "report.txt").write_text(full_report, encoding="utf-8")
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"\n  Report:  {OUTPUT_DIR}/report.txt")
    print(f"  Summary: {OUTPUT_DIR}/summary.txt")

    json_out = {
        "aggregated":            _jsonify(agg),
        "comparison":            _jsonify(comparison),
        "pd_vs_sim_per_activity": _jsonify(pd_vs_sim_activity),
        "suggestions":           _jsonify(suggestions),
        "freq_by_window_length": _jsonify(freq_by_wl),
        "sim_mode_stats":        _jsonify(sim_mode_stats),
        "threshold_sensitivity":  _jsonify({str(k): v for k, v in thr_sensitivity.items()}),
        "tremor_only_freq_stats": _jsonify(tremor_freq_stats),
        "tremor_only_comparison": _jsonify(tremor_only_comp),
        "pd_upper_end_rms":       _jsonify(pd_upper_end),
        "pd_activity_scaling":    _jsonify(pd_activity_scaling),
        "pd_vs_sim_per_location": _jsonify(pd_vs_sim_location),
        "pd_location_scaling":    _jsonify(pd_location_scaling),
        "pd_coverage_by_sim":     _jsonify(pd_coverage_sim),
        "kg_extremes_by_location": _jsonify(kg_extremes),
        "sim_analysis_modes":     SIM_ANALYSIS_MODES,
    }
    (OUTPUT_DIR / "data.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "suggestions.json").write_text(
        json.dumps(_jsonify(suggestions), indent=2), encoding="utf-8")
    print(f"  Data:    {OUTPUT_DIR}/data.json")
    print(f"  Config suggestions: {OUTPUT_DIR}/suggestions.json")

    print("\n" + summary)
    print(f"\nAll outputs -> {OUTPUT_DIR}")

    # -----------------------------------------------------------------------
    # Additional: IQR statistics for PD tremor-present lower-arm windows
    # -----------------------------------------------------------------------
    iqr = pd_lower_arm_tremor_iqr_stats(records)

    def _fiqr(s, key):
        v = s.get(key, float("nan"))
        return "n/a" if not np.isfinite(v) else f"{v:.4f}"

    print("")
    print("-- PD tremor-present (lower arm only) --")
    print(f"   (n={iqr['n_windows']} windows, locations: Lower Right + Lower Left,"
          f" threshold={TREMOR_PRESENCE_THRESHOLD})")
    print("")
    print("Dominant frequency (Hz):")
    s = iqr["dom_freq"]
    print(f"  p25={_fiqr(s,'p25')}, median={_fiqr(s,'median')}, p75={_fiqr(s,'p75')}")
    print("")
    print("Acc tremor RMS (3\u20137 Hz):")
    s = iqr["acc_rms_band37"]
    print(f"  p25={_fiqr(s,'p25')}, median={_fiqr(s,'median')}, p75={_fiqr(s,'p75')}")
    print("")
    print("Gyro tremor RMS (3\u20137 Hz):")
    s = iqr["gyro_rms_band37"]
    print(f"  p25={_fiqr(s,'p25')}, median={_fiqr(s,'median')}, p75={_fiqr(s,'p75')}")
    print("")
    print("k_g (gyro/acc):")
    s = iqr["k_g"]
    print(f"  p25={_fiqr(s,'p25')}, median={_fiqr(s,'median')}, p75={_fiqr(s,'p75')}")


if __name__ == "__main__":
    main()
