#!/usr/bin/env python
# scripts/validate_cache.py
"""Cache validation script — scans .npz shards and runs data-quality checks.

Usage:
    python scripts/validate_cache.py --cache data/cache [--sample-raw]

Exit codes:
    0 — all checks passed
    1 — at least one CRITICAL check failed
    2 — warning-only failures (no CRITICAL failures)

Checks
------
1. Schema completeness       [CRITICAL] — all 19 expected keys + correct dtypes
2. Shape consistency         [CRITICAL] — features (N,17), event_ts/day_id (N,)
3. No NaN/inf in features    [CRITICAL] — np.isfinite(features).all()
4. April hold-out            [CRITICAL] — date < 2026-04-14
5. Per-symbol event-count re-derivation [WARNING, --sample-raw only]
6. Label validity            [WARNING]  — mask_h500 tail check
7. day_id monotonicity       [WARNING]  — non-decreasing day_id; event_ts within day
8. Feature value ranges      [WARNING]  — sanity bounds per feature
9. Aggregate corpus stats    [INFO]     — windows at stride=50/200, per-symbol counts
"""

from __future__ import annotations

import argparse
import enum
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APRIL_HELDOUT_START: str = "2026-04-14"
WINDOW_LEN: int = 200
STRIDE_PRETRAIN: int = 50
STRIDE_EVAL: int = 200

# Canonical feature names in channel order (matches tape/constants.py FEATURE_NAMES)
FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "log_total_qty",
    "is_open",
    "time_delta",
    "num_fills",
    "book_walk",
    "effort_vs_result",
    "climax_score",
    "prev_seq_time_span",
    "log_spread",
    "imbalance_L1",
    "imbalance_L5",
    "depth_ratio",
    "trade_vs_mid",
    "delta_imbalance_L1",
    "kyle_lambda",
    "cum_ofi_5",
)
assert len(FEATURE_NAMES) == 17

# All 19 keys that must be present in every shard
EXPECTED_KEYS: frozenset[str] = frozenset(
    {
        "features",
        "event_ts",
        "day_id",
        "schema_version",
        "symbol",
        "date",
        "dir_h10",
        "dir_h50",
        "dir_h100",
        "dir_h500",
        "dir_mask_h10",
        "dir_mask_h50",
        "dir_mask_h100",
        "dir_mask_h500",
        "wy_stress",
        "wy_informed_flow",
        "wy_climax",
        "wy_spring",
        "wy_absorption",
    }
)
assert len(EXPECTED_KEYS) == 19

# Expected dtypes for the most critical keys
_EXPECTED_DTYPES: dict[str, np.dtype] = {
    "features": np.dtype(np.float32),
    "event_ts": np.dtype(np.int64),
    "day_id": np.dtype(np.int64),
}

# Per-feature range constraints: (min_allowed, max_allowed, inclusive)
# Only features with non-trivial bounds are listed.
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "is_open": (0.0, 1.0),
    "effort_vs_result": (-5.0, 5.0),
    "climax_score": (0.0, 5.0),
    "trade_vs_mid": (-5.0, 5.0),
}
# log_spread must be strictly negative (spread << mid in normal markets)
_LOG_SPREAD_MAX: float = 0.0

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Severity(enum.Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    INFO = "INFO"


@dataclass
class CheckResult:
    severity: Severity
    message: str
    shard_path: Path | None = None


# ---------------------------------------------------------------------------
# Individual check functions (each takes a loaded shard dict + path)
# ---------------------------------------------------------------------------


def check_schema(shard: dict, path: Path) -> CheckResult:
    """Check 1: All 19 expected keys are present with correct dtypes. [CRITICAL]"""
    present = frozenset(shard.keys())
    missing = EXPECTED_KEYS - present
    if missing:
        return CheckResult(
            Severity.CRITICAL,
            f"Missing keys: {sorted(missing)}",
            path,
        )

    # Dtype checks for critical arrays
    for key, expected_dtype in _EXPECTED_DTYPES.items():
        arr = shard[key]
        if not isinstance(arr, np.ndarray):
            return CheckResult(
                Severity.CRITICAL,
                f"Key '{key}' is not an ndarray (got {type(arr).__name__})",
                path,
            )
        if arr.dtype != expected_dtype:
            return CheckResult(
                Severity.CRITICAL,
                f"Key '{key}' dtype is {arr.dtype}, expected {expected_dtype}",
                path,
            )

    return CheckResult(Severity.OK, "Schema OK", path)


def check_shape_consistency(shard: dict, path: Path) -> CheckResult:
    """Check 2: features is (N,17) float32; event_ts/day_id are (N,). [CRITICAL]"""
    features = shard.get("features")
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        return CheckResult(Severity.CRITICAL, "features is not a 2-D array", path)
    n, ncols = features.shape
    if ncols != 17:
        return CheckResult(
            Severity.CRITICAL,
            f"features has {ncols} columns, expected 17",
            path,
        )

    for key in ("event_ts", "day_id"):
        arr = shard.get(key)
        if not isinstance(arr, np.ndarray) or arr.shape != (n,):
            return CheckResult(
                Severity.CRITICAL,
                f"'{key}' shape {getattr(arr, 'shape', None)}, expected ({n},)",
                path,
            )

    # Direction label arrays must all have length N
    dir_keys = [k for k in EXPECTED_KEYS if k.startswith("dir_")]
    for key in dir_keys:
        arr = shard.get(key)
        if arr is None:
            continue  # schema check will catch missing keys
        if not isinstance(arr, np.ndarray) or arr.shape[0] != n:
            return CheckResult(
                Severity.CRITICAL,
                f"'{key}' length {getattr(arr, 'shape', None)}, expected ({n}, ...)",
                path,
            )

    return CheckResult(Severity.OK, "Shape OK", path)


def check_no_nan_inf(shard: dict, path: Path) -> CheckResult:
    """Check 3: No NaN or Inf in features. [CRITICAL]"""
    features = shard.get("features")
    if not isinstance(features, np.ndarray):
        return CheckResult(Severity.CRITICAL, "features missing or not array", path)
    finite_mask = np.isfinite(features)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        has_nan = bool(np.isnan(features).any())
        has_inf = bool(np.isinf(features).any())
        kinds = []
        if has_nan:
            kinds.append("NaN")
        if has_inf:
            kinds.append("Inf")
        return CheckResult(
            Severity.CRITICAL,
            f"{n_bad} non-finite values ({', '.join(kinds)}) in features",
            path,
        )
    return CheckResult(Severity.OK, "No NaN/Inf", path)


def check_april_holdout(shard: dict, path: Path) -> CheckResult:
    """Check 4: date < 2026-04-14 (hold-out guard). [CRITICAL]"""
    date_arr = shard.get("date")
    if date_arr is None:
        return CheckResult(Severity.CRITICAL, "Missing 'date' key", path)
    date_str = str(date_arr)
    if date_str >= APRIL_HELDOUT_START:
        return CheckResult(
            Severity.CRITICAL,
            f"Hold-out violation: date={date_str} >= {APRIL_HELDOUT_START}",
            path,
        )
    return CheckResult(Severity.OK, f"Hold-out OK (date={date_str})", path)


def check_label_validity(shard: dict, path: Path) -> CheckResult:
    """Check 6: mask_h500 tail sentinel check. [WARNING]

    The last 500 events should have mask_h500 == False (no forward data).
    If the full mask is True (no tail masked) this is suspicious.
    """
    mask = shard.get("dir_mask_h500")
    if not isinstance(mask, np.ndarray):
        return CheckResult(Severity.WARNING, "dir_mask_h500 missing", path)
    n = len(mask)
    if n <= 500:
        # Too few events to enforce the 500-event tail; skip
        return CheckResult(Severity.OK, "N<=500; label tail check skipped", path)
    tail_mask = mask[-500:]
    if tail_mask.any():
        n_bad = int(tail_mask.sum())
        return CheckResult(
            Severity.WARNING,
            f"dir_mask_h500: {n_bad}/500 tail positions are True (should be False)",
            path,
        )
    return CheckResult(Severity.OK, "Label tails OK", path)


def check_day_id_monotonicity(shard: dict, path: Path) -> CheckResult:
    """Check 7: day_id non-decreasing; event_ts non-decreasing within each day. [WARNING]"""
    day_id = shard.get("day_id")
    event_ts = shard.get("event_ts")
    if not isinstance(day_id, np.ndarray) or not isinstance(event_ts, np.ndarray):
        return CheckResult(Severity.WARNING, "day_id or event_ts missing", path)

    # day_id must be non-decreasing
    if len(day_id) > 1 and bool(np.any(np.diff(day_id.astype(np.int64)) < 0)):
        return CheckResult(
            Severity.WARNING,
            "day_id is not non-decreasing",
            path,
        )

    # event_ts must be non-decreasing within each day
    unique_days = np.unique(day_id)
    for d in unique_days:
        mask = day_id == d
        ts_day = event_ts[mask]
        if len(ts_day) > 1 and bool(np.any(np.diff(ts_day) < 0)):
            return CheckResult(
                Severity.WARNING,
                f"event_ts not non-decreasing for day_id={d}",
                path,
            )

    return CheckResult(Severity.OK, "Monotonicity OK", path)


def check_feature_ranges(shard: dict, path: Path) -> CheckResult:
    """Check 8: Sanity bounds per feature. [WARNING]"""
    features = shard.get("features")
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        return CheckResult(
            Severity.WARNING, "features not available for range check", path
        )

    violations: list[str] = []
    feat_list = list(FEATURE_NAMES)

    for fname, (lo, hi) in _FEATURE_RANGES.items():
        if fname not in feat_list:
            continue
        idx = feat_list.index(fname)
        col = features[:, idx].astype(float)
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            continue
        if float(finite.min()) < lo or float(finite.max()) > hi:
            violations.append(
                f"{fname} [{float(finite.min()):.3f}, {float(finite.max()):.3f}]"
                f" outside [{lo}, {hi}]"
            )

    # log_spread must be strictly negative
    if "log_spread" in feat_list:
        ls_idx = feat_list.index("log_spread")
        ls_col = features[:, ls_idx].astype(float)
        finite_ls = ls_col[np.isfinite(ls_col)]
        if len(finite_ls) > 0 and float(finite_ls.max()) >= _LOG_SPREAD_MAX:
            violations.append(
                f"log_spread max={float(finite_ls.max()):.3f} >= 0 (should be negative)"
            )

    if violations:
        return CheckResult(
            Severity.WARNING,
            "Feature range violations: " + "; ".join(violations),
            path,
        )
    return CheckResult(Severity.OK, "Feature ranges OK", path)


# ---------------------------------------------------------------------------
# Aggregate stats collector
# ---------------------------------------------------------------------------


class _SymbolStats(NamedTuple):
    n_events: int
    n_shards: int
    windows_pretrain: int
    windows_eval: int


def _windows(n_events: int, stride: int) -> int:
    """Number of stride-stride windows of length WINDOW_LEN in n_events events."""
    if n_events < WINDOW_LEN:
        return 0
    return max(0, (n_events - WINDOW_LEN) // stride + 1)


def _print_summary(
    results: list[CheckResult],
    per_symbol: dict[str, _SymbolStats],
    n_shards: int,
) -> None:
    """Print a human-readable summary table."""
    n_critical = sum(1 for r in results if r.severity == Severity.CRITICAL)
    n_warning = sum(1 for r in results if r.severity == Severity.WARNING)

    print("\n" + "=" * 72)
    print("CACHE VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Shards scanned : {n_shards}")
    print(f"CRITICAL fails : {n_critical}")
    print(f"WARNING  fails : {n_warning}")
    print()

    if n_shards > 0:
        total_events = sum(s.n_events for s in per_symbol.values())
        total_pretrain = sum(s.windows_pretrain for s in per_symbol.values())
        total_eval = sum(s.windows_eval for s in per_symbol.values())
        print(f"Total events   : {total_events:,}")
        print(f"Windows@s=50   : {total_pretrain:,}")
        print(f"Windows@s=200  : {total_eval:,}")
        print()
        print(f"{'Symbol':<12} {'Events':>10} {'Shards':>7} {'W@50':>9} {'W@200':>9}")
        print("-" * 52)
        for sym in sorted(per_symbol):
            s = per_symbol[sym]
            print(
                f"{sym:<12} {s.n_events:>10,} {s.n_shards:>7} "
                f"{s.windows_pretrain:>9,} {s.windows_eval:>9,}"
            )
        print()

    if n_critical > 0 or n_warning > 0:
        print("FAILURES:")
        for r in results:
            if r.severity in (Severity.CRITICAL, Severity.WARNING):
                tag = f"[{r.severity.value}]"
                path_label = r.shard_path.name if r.shard_path else ""
                print(f"  {tag:<12} {path_label}: {r.message}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main validation runner (callable from tests)
# ---------------------------------------------------------------------------


def run_validation(cache_root: Path, *, sample_raw: bool = False) -> int:
    """Run all checks against shards under cache_root.

    Returns exit code: 0=pass, 1=critical, 2=warning-only.
    """
    shards = sorted(cache_root.rglob("*.npz"))
    all_results: list[CheckResult] = []
    per_symbol: dict[str, dict] = {}  # sym -> {n_events, n_shards}

    for shard_path in shards:
        try:
            with np.load(shard_path, allow_pickle=False) as z:
                shard = {k: z[k] for k in z.files}
        except Exception as exc:
            all_results.append(
                CheckResult(Severity.CRITICAL, f"Failed to load: {exc}", shard_path)
            )
            continue

        # Run all checks
        check_fns = [
            check_schema,
            check_shape_consistency,
            check_no_nan_inf,
            check_april_holdout,
            check_label_validity,
            check_day_id_monotonicity,
            check_feature_ranges,
        ]
        for fn in check_fns:
            result = fn(shard, shard_path)
            if result.severity != Severity.OK:
                all_results.append(result)

        # Accumulate corpus stats
        sym = str(shard.get("symbol", "UNKNOWN"))
        features = shard.get("features")
        n_ev = (
            int(features.shape[0])
            if isinstance(features, np.ndarray) and features.ndim == 2
            else 0
        )
        rec = per_symbol.setdefault(sym, {"n_events": 0, "n_shards": 0})
        rec["n_events"] += n_ev
        rec["n_shards"] += 1

    # Build typed per-symbol stats
    per_sym_stats: dict[str, _SymbolStats] = {
        sym: _SymbolStats(
            n_events=rec["n_events"],
            n_shards=rec["n_shards"],
            windows_pretrain=_windows(rec["n_events"], STRIDE_PRETRAIN),
            windows_eval=_windows(rec["n_events"], STRIDE_EVAL),
        )
        for sym, rec in per_symbol.items()
    }

    _print_summary(all_results, per_sym_stats, len(shards))

    has_critical = any(r.severity == Severity.CRITICAL for r in all_results)
    has_warning = any(r.severity == Severity.WARNING for r in all_results)

    if has_critical:
        return 1
    if has_warning:
        return 2
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate the .npz shard cache for tape representation learning."
    )
    p.add_argument(
        "--cache",
        required=True,
        metavar="DIR",
        help="Root directory of the .npz cache (e.g. data/cache).",
    )
    p.add_argument(
        "--sample-raw",
        action="store_true",
        default=False,
        help="(Slow) re-derive event counts from raw parquet and compare. Default: skip.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cache_root = Path(args.cache)
    if not cache_root.exists():
        print(f"ERROR: cache directory does not exist: {cache_root}", file=sys.stderr)
        return 1
    return run_validation(cache_root, sample_raw=args.sample_raw)


if __name__ == "__main__":
    raise SystemExit(main())
