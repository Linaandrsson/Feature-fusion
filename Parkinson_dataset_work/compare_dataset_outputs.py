#!/usr/bin/env python3
"""
Compare two dataset output directories (old vs new) to verify
that the new optimized script produces identical results.

Usage:
    python compare_dataset_outputs.py [--old DIR_OLD] [--new DIR_NEW]

Defaults:
    --old  Data/Tremor_datagenerator_files
    --new  Data/Tremor_datagenerator_files_new
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN = "\033[0;32m"; YELLOW = "\033[1;33m"; RED = "\033[0;31m"
BOLD = "\033[1m"; RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}── {msg} ──{RESET}")

# ── helpers ───────────────────────────────────────────────────────────────────

# Keys that must be bit-for-bit identical (labels / indices)
_EXACT_KEYS = {"y", "subject_id", "base_window_idx", "tremor_score", "sensor_cols"}
# Keys compared with np.allclose
_APPROX_KEYS = {"X", "tremor_freq", "tremor_acc_rms", "tremor_gyro_rms"}


def load_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=False))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def compare_arrays(key: str, a: np.ndarray, b: np.ndarray,
                   rtol: float = 1e-4, atol: float = 1e-4) -> bool:
    """Compare two arrays — exact for discrete keys, allclose for continuous."""
    if a.shape != b.shape:
        fail(f"  {key}: shape mismatch  old={a.shape}  new={b.shape}")
        return False
    if a.dtype != b.dtype:
        warn(f"  {key}: dtype differs  old={a.dtype}  new={b.dtype}")

    # Determine the base key name (strip sensor prefix if present)
    base_key = key.split("/")[-1]

    try:
        if base_key in _EXACT_KEYS:
            if not np.array_equal(a, b):
                n_diff = int(np.sum(a != b))
                fail(f"  {key}: exact mismatch  {n_diff}/{a.size} elements differ")
                return False
        elif base_key in _APPROX_KEYS:
            a_f = a.astype(np.float64)
            b_f = b.astype(np.float64)
            if not np.allclose(a_f, b_f, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.abs(a_f - b_f)
                idx = np.unravel_index(np.argmax(diff), diff.shape)
                fail(f"  {key}: values differ  max_abs_diff={diff.max():.6e} at index={idx}")
                return False
        else:
            warn(f"  {key}: unknown key — using exact comparison")
            if not np.array_equal(a, b):
                fail(f"  {key}: mismatch (unknown key)")
                return False
    except Exception as exc:
        fail(f"  {key}: comparison error — {exc}")
        return False
    return True


# ── main comparison ───────────────────────────────────────────────────────────

def compare_variant(variant_name: str,
                    dir_old: Path, dir_new: Path) -> tuple[int, int]:
    """Compare one variant folder. Returns (n_ok, n_fail)."""
    n_ok = n_fail = 0

    npz_old = sorted(f for f in dir_old.glob("*.npz") if not f.name.startswith("._"))
    npz_new = sorted(f for f in dir_new.glob("*.npz") if not f.name.startswith("._"))
    txt_old = sorted(f for f in dir_old.glob("*.txt") if not f.name.startswith("._"))
    txt_new = sorted(f for f in dir_new.glob("*.txt") if not f.name.startswith("._"))

    names_old = {f.name for f in npz_old}
    names_new = {f.name for f in npz_new}
    only_old = names_old - names_new
    only_new = names_new - names_old

    if only_old:
        for n in sorted(only_old):
            fail(f"NPZ only in OLD: {n}")
            n_fail += 1
    if only_new:
        for n in sorted(only_new):
            fail(f"NPZ only in NEW: {n}")
            n_fail += 1

    # ---- per-sensor NPZ comparison ----
    for npz_path in npz_old:
        if npz_path.name not in names_new:
            continue
        sensor = npz_path.stem
        old_d = load_npz(npz_path)
        new_d = load_npz(dir_new / npz_path.name)

        keys_old = set(old_d.keys())
        keys_new = set(new_d.keys())
        if keys_old != keys_new:
            warn(f"  {sensor}: key sets differ  "
                 f"only_old={keys_old-keys_new}  only_new={keys_new-keys_old}")

        all_ok = True
        for k in sorted(keys_old & keys_new):
            a, b = old_d[k], new_d[k]
            is_scalar = np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0)
            if is_scalar:
                a_v = a[()] if isinstance(a, np.ndarray) else a
                b_v = b[()] if isinstance(b, np.ndarray) else b
                try:
                    if not np.isclose(float(a_v), float(b_v), rtol=1e-4, atol=1e-4):
                        fail(f"  {sensor}/{k}: scalar differs  old={a_v}  new={b_v}")
                        all_ok = False
                except (TypeError, ValueError):
                    # Non-numeric scalar (e.g. string) — use direct equality
                    if a_v != b_v:
                        fail(f"  {sensor}/{k}: scalar differs  old={a_v}  new={b_v}")
                        all_ok = False
            else:
                if not compare_arrays(f"{sensor}/{k}", a, b):
                    all_ok = False

        if all_ok:
            X = old_d["X"]
            ok(f"{sensor}: {X.shape[0]} windows, shape={X.shape} — all arrays match")
            n_ok += 1
        else:
            n_fail += 1

    # ---- TXT row count + SHA256 ----
    hdr(f"{variant_name} · TXT shape check")
    txt_names_old = {f.name for f in txt_old}
    txt_names_new = {f.name for f in txt_new}
    for name in sorted(txt_names_old & txt_names_new):
        rows_o = sum(1 for _ in (dir_old / name).open())
        rows_n = sum(1 for _ in (dir_new / name).open())
        if rows_o != rows_n:
            fail(f"{name}: row count differs  old={rows_o}  new={rows_n}")
            n_fail += 1
        else:
            hash_o = sha256_file(dir_old / name)
            hash_n = sha256_file(dir_new / name)
            if hash_o == hash_n:
                ok(f"{name}: {rows_o} rows, SHA256 match")
            else:
                warn(f"{name}: {rows_o} rows match but SHA256 differs  "
                     f"old={hash_o[:12]}…  new={hash_n[:12]}…")

    for name in sorted(txt_names_old - txt_names_new):
        fail(f"TXT only in OLD: {name}")
        n_fail += 1
    for name in sorted(txt_names_new - txt_names_old):
        fail(f"TXT only in NEW: {name}")
        n_fail += 1

    return n_ok, n_fail


def compare_all(dir_old: Path, dir_new: Path) -> int:
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"{BOLD} Dataset comparison{RESET}")
    print(f"  OLD: {dir_old}")
    print(f"  NEW: {dir_new}")
    print(f"{BOLD}{'='*72}{RESET}")

    # ---- variant-level overview ----
    hdr("Variants present")
    variants_old = {d.name for d in dir_old.iterdir() if d.is_dir()}
    variants_new = {d.name for d in dir_new.iterdir() if d.is_dir()}

    if variants_old == variants_new:
        ok(f"Same {len(variants_old)} variants in both")
    else:
        if variants_old - variants_new:
            fail(f"Only in OLD: {sorted(variants_old - variants_new)}")
        if variants_new - variants_old:
            fail(f"Only in NEW: {sorted(variants_new - variants_old)}")

    total_ok = total_fail = 0

    for variant in sorted(variants_old & variants_new):
        hdr(f"Variant: {variant}")
        n_ok, n_fail = compare_variant(
            variant,
            dir_old / variant,
            dir_new / variant,
        )
        total_ok   += n_ok
        total_fail += n_fail
        summary = f"{n_ok} sensors OK, {n_fail} FAIL"
        (ok if n_fail == 0 else fail)(summary)

    # ---- final verdict ----
    print(f"\n{BOLD}{'='*72}{RESET}")
    if total_fail == 0:
        print(f"{GREEN}{BOLD} ✓  All {total_ok} sensors match across all variants{RESET}")
    else:
        print(f"{RED}{BOLD} ✗  {total_fail} sensors FAILED, {total_ok} OK{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")

    return 1 if total_fail > 0 else 0


def main() -> int:
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(description="Compare old vs new dataset outputs.")
    parser.add_argument(
        "--old",
        default=str(script_dir / "Data" / "Tremor_datagenerator_files"),
        help="Path to old (reference) output directory",
    )
    parser.add_argument(
        "--new",
        default=str(script_dir / "Data" / "Tremor_datagenerator_files_new"),
        help="Path to new (optimized) output directory",
    )
    args = parser.parse_args()

    dir_old = Path(args.old)
    dir_new = Path(args.new)

    for d, label in [(dir_old, "OLD"), (dir_new, "NEW")]:
        if not d.exists():
            print(f"{RED}ERROR:{RESET} {label} directory not found: {d}")
            return 2

    return compare_all(dir_old, dir_new)


if __name__ == "__main__":
    raise SystemExit(main())
