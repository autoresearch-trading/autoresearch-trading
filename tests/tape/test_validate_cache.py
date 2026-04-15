# tests/tape/test_validate_cache.py
"""Unit tests for scripts/validate_cache.py — all checks exercised on synthetic shards.

Design: No real data required. Shards are built in tmp dirs via make_shard().
Each test exercises one check in isolation by calling the check function directly,
then verifies the aggregated exit code via run_validation().
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to sys.path so we can import scripts/ module
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.validate_cache import (  # noqa: E402
    EXPECTED_KEYS,
    CheckResult,
    Severity,
    check_april_holdout,
    check_day_id_monotonicity,
    check_feature_ranges,
    check_label_validity,
    check_no_nan_inf,
    check_schema,
    check_shape_consistency,
    run_validation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 600  # number of events in synthetic shard


def make_features(n: int = N, *, nan_row: int = -1, inf_row: int = -1) -> np.ndarray:
    """Valid (N, 17) float32 feature matrix with optional corruption."""
    rng = np.random.default_rng(42)
    feats = rng.standard_normal((n, 17)).astype(np.float32)
    # Enforce per-feature constraints so range check passes by default
    feat_names = [
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
    ]
    is_open_idx = feat_names.index("is_open")
    evr_idx = feat_names.index("effort_vs_result")
    climax_idx = feat_names.index("climax_score")
    log_spread_idx = feat_names.index("log_spread")
    tvm_idx = feat_names.index("trade_vs_mid")

    feats[:, is_open_idx] = rng.uniform(0.0, 1.0, n).astype(np.float32)
    feats[:, evr_idx] = rng.uniform(-4.9, 4.9, n).astype(np.float32)
    feats[:, climax_idx] = rng.uniform(0.0, 4.9, n).astype(np.float32)
    feats[:, log_spread_idx] = rng.uniform(-10.0, -0.001, n).astype(np.float32)
    feats[:, tvm_idx] = rng.uniform(-4.9, 4.9, n).astype(np.float32)

    if nan_row >= 0:
        feats[nan_row, 0] = np.nan
    if inf_row >= 0:
        feats[inf_row, 1] = np.inf
    return feats


def make_shard(
    tmp_path: Path,
    *,
    symbol: str = "BTC",
    date: str = "2025-11-01",
    n: int = N,
    features: np.ndarray | None = None,
    drop_key: str | None = None,
    corrupt_dtype: str | None = None,
    day_id_nonmono: bool = False,
    event_ts_nonmono: bool = False,
) -> Path:
    """Save a synthetic .npz shard and return the path."""
    rng = np.random.default_rng(0)

    if features is None:
        features = make_features(n)

    # day_id: monotonically non-decreasing
    import datetime

    epoch = datetime.date(1970, 1, 1)
    day_int = (datetime.date.fromisoformat(date) - epoch).days
    day_id = np.full(n, day_int, dtype=np.int64)
    if day_id_nonmono:
        day_id[n // 2] = day_int - 1  # retrograde step

    # event_ts: monotonically non-decreasing (1ms steps)
    base_ts = 1_700_000_000_000  # ms epoch
    event_ts = np.arange(base_ts, base_ts + n, dtype=np.int64)
    if event_ts_nonmono:
        event_ts[n // 2] = event_ts[n // 2 - 1] - 1  # retrograde step

    payload: dict[str, np.ndarray] = {
        "features": features,
        "event_ts": event_ts,
        "day_id": day_id,
        "schema_version": np.array(1, dtype=np.int32),
        "symbol": np.array(symbol),
        "date": np.array(date),
        # direction labels — h500 tail masked
        "dir_h10": np.ones(n, dtype=np.int8),
        "dir_h50": np.ones(n, dtype=np.int8),
        "dir_h100": np.ones(n, dtype=np.int8),
        "dir_h500": np.zeros(n, dtype=np.int8),  # tail 500 are sentinels
        "dir_mask_h10": np.concatenate(
            [np.ones(n - 10, dtype=bool), np.zeros(10, dtype=bool)]
        ),
        "dir_mask_h50": np.concatenate(
            [np.ones(n - 50, dtype=bool), np.zeros(50, dtype=bool)]
        ),
        "dir_mask_h100": np.concatenate(
            [np.ones(n - 100, dtype=bool), np.zeros(100, dtype=bool)]
        ),
        "dir_mask_h500": np.concatenate(
            [np.ones(n - 500, dtype=bool), np.zeros(500, dtype=bool)]
        ),
        # wyckoff
        "wy_stress": rng.integers(0, 2, n, dtype=np.int8),
        "wy_informed_flow": rng.integers(0, 2, n, dtype=np.int8),
        "wy_climax": rng.integers(0, 2, n, dtype=np.int8),
        "wy_spring": rng.integers(0, 2, n, dtype=np.int8),
        "wy_absorption": rng.integers(0, 2, n, dtype=np.int8),
    }

    if drop_key is not None:
        payload.pop(drop_key, None)

    if corrupt_dtype == "features_wrong_dtype":
        payload["features"] = features.astype(np.float64)

    shard_dir = tmp_path / symbol
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{symbol}__{date}.npz"
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]
    return path


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Check 1: Schema completeness [CRITICAL]
# ---------------------------------------------------------------------------


class TestCheckSchema:
    def test_happy_path_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path)
        shard = load_npz(path)
        result = check_schema(shard, path)
        assert result.severity == Severity.OK

    def test_missing_key_critical(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, drop_key="dir_h500")
        shard = load_npz(path)
        result = check_schema(shard, path)
        assert result.severity == Severity.CRITICAL
        assert "dir_h500" in result.message

    def test_wrong_dtype_critical(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, corrupt_dtype="features_wrong_dtype")
        shard = load_npz(path)
        result = check_schema(shard, path)
        assert result.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Check 2: Shape consistency [CRITICAL]
# ---------------------------------------------------------------------------


class TestCheckShapeConsistency:
    def test_happy_path_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path)
        shard = load_npz(path)
        result = check_shape_consistency(shard, path)
        assert result.severity == Severity.OK

    def test_wrong_feature_columns_critical(self, tmp_path: Path) -> None:
        bad_feats = np.zeros((N, 16), dtype=np.float32)  # 16 instead of 17
        path = make_shard(tmp_path, features=bad_feats)
        shard = load_npz(path)
        result = check_shape_consistency(shard, path)
        assert result.severity == Severity.CRITICAL

    def test_mismatched_label_length_critical(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path)
        shard = load_npz(path)
        # Corrupt one label array length
        shard["dir_h10"] = np.ones(N - 1, dtype=np.int8)
        result = check_shape_consistency(shard, path)
        assert result.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Check 3: No NaN/inf [CRITICAL]
# ---------------------------------------------------------------------------


class TestCheckNoNanInf:
    def test_happy_path_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path)
        shard = load_npz(path)
        result = check_no_nan_inf(shard, path)
        assert result.severity == Severity.OK

    def test_nan_in_features_critical(self, tmp_path: Path) -> None:
        feats = make_features(N, nan_row=10)
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_no_nan_inf(shard, path)
        assert result.severity == Severity.CRITICAL
        assert "NaN" in result.message or "nan" in result.message.lower()

    def test_inf_in_features_critical(self, tmp_path: Path) -> None:
        feats = make_features(N, inf_row=5)
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_no_nan_inf(shard, path)
        assert result.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Check 4: April hold-out [CRITICAL]
# ---------------------------------------------------------------------------


class TestCheckAprilHoldout:
    def test_pre_april_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, date="2026-04-13")
        shard = load_npz(path)
        result = check_april_holdout(shard, path)
        assert result.severity == Severity.OK

    def test_april14_critical(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, date="2026-04-14")
        shard = load_npz(path)
        result = check_april_holdout(shard, path)
        assert result.severity == Severity.CRITICAL

    def test_april15_critical(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, date="2026-04-15")
        shard = load_npz(path)
        result = check_april_holdout(shard, path)
        assert result.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Check 6: Label validity [WARNING]
# ---------------------------------------------------------------------------


class TestCheckLabelValidity:
    def test_correct_tail_sentinel_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, n=N)
        shard = load_npz(path)
        result = check_label_validity(shard, path)
        assert result.severity in (Severity.OK, Severity.WARNING)

    def test_all_mask_true_warning(self, tmp_path: Path) -> None:
        """If mask_h500 is all True (no sentinel tail) → warning."""
        path = make_shard(tmp_path, n=N)
        shard = load_npz(path)
        # Corrupt: set all mask_h500 True (no tail)
        shard["dir_mask_h500"] = np.ones(N, dtype=bool)
        result = check_label_validity(shard, path)
        assert result.severity == Severity.WARNING


# ---------------------------------------------------------------------------
# Check 7: Day_id monotonicity [WARNING]
# ---------------------------------------------------------------------------


class TestCheckDayIdMonotonicity:
    def test_monotone_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path)
        shard = load_npz(path)
        result = check_day_id_monotonicity(shard, path)
        assert result.severity == Severity.OK

    def test_day_id_not_monotone_warning(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, day_id_nonmono=True)
        shard = load_npz(path)
        result = check_day_id_monotonicity(shard, path)
        assert result.severity == Severity.WARNING

    def test_event_ts_not_monotone_warning(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path, event_ts_nonmono=True)
        shard = load_npz(path)
        result = check_day_id_monotonicity(shard, path)
        assert result.severity == Severity.WARNING


# ---------------------------------------------------------------------------
# Check 8: Feature value ranges [WARNING]
# ---------------------------------------------------------------------------


class TestCheckFeatureRanges:
    def test_valid_ranges_passes(self, tmp_path: Path) -> None:
        path = make_shard(tmp_path)
        shard = load_npz(path)
        result = check_feature_ranges(shard, path)
        assert result.severity == Severity.OK

    def test_evr_out_of_range_warning(self, tmp_path: Path) -> None:
        feats = make_features(N)
        evr_idx = 6  # effort_vs_result
        feats[0, evr_idx] = 6.0  # outside [-5, 5]
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_feature_ranges(shard, path)
        assert result.severity == Severity.WARNING

    def test_climax_negative_warning(self, tmp_path: Path) -> None:
        feats = make_features(N)
        climax_idx = 7  # climax_score
        feats[5, climax_idx] = -0.1  # outside [0, 5]
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_feature_ranges(shard, path)
        assert result.severity == Severity.WARNING

    def test_is_open_outside_01_warning(self, tmp_path: Path) -> None:
        feats = make_features(N)
        is_open_idx = 2
        feats[3, is_open_idx] = 1.5  # outside [0, 1]
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_feature_ranges(shard, path)
        assert result.severity == Severity.WARNING

    def test_log_spread_positive_warning(self, tmp_path: Path) -> None:
        feats = make_features(N)
        log_spread_idx = 9
        feats[0, log_spread_idx] = 0.5  # positive log_spread → spread > 100% of mid
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_feature_ranges(shard, path)
        assert result.severity == Severity.WARNING

    def test_trade_vs_mid_out_of_range_warning(self, tmp_path: Path) -> None:
        feats = make_features(N)
        tvm_idx = 13  # trade_vs_mid
        feats[2, tvm_idx] = 6.0  # outside [-5, 5]
        path = make_shard(tmp_path, features=feats)
        shard = load_npz(path)
        result = check_feature_ranges(shard, path)
        assert result.severity == Severity.WARNING


# ---------------------------------------------------------------------------
# Integration: run_validation exit codes
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_happy_path_exit0(self, tmp_path: Path) -> None:
        make_shard(tmp_path, symbol="BTC", date="2025-11-01")
        make_shard(tmp_path, symbol="ETH", date="2025-11-01")
        code = run_validation(tmp_path, sample_raw=False)
        assert code == 0

    def test_missing_key_exit1(self, tmp_path: Path) -> None:
        make_shard(tmp_path, drop_key="features")
        code = run_validation(tmp_path, sample_raw=False)
        assert code == 1

    def test_nan_in_features_exit1(self, tmp_path: Path) -> None:
        feats = make_features(N, nan_row=0)
        make_shard(tmp_path, features=feats)
        code = run_validation(tmp_path, sample_raw=False)
        assert code == 1

    def test_april14_shard_exit1(self, tmp_path: Path) -> None:
        make_shard(tmp_path, date="2026-04-14")
        code = run_validation(tmp_path, sample_raw=False)
        assert code == 1

    def test_warning_only_exit2(self, tmp_path: Path) -> None:
        """Range violation = WARNING only → exit 2, not 1."""
        feats = make_features(N)
        feats[0, 6] = 6.0  # effort_vs_result out of range
        make_shard(tmp_path, features=feats)
        code = run_validation(tmp_path, sample_raw=False)
        assert code == 2

    def test_empty_cache_exit0(self, tmp_path: Path) -> None:
        """Empty cache dir → 0 shards → no failures, exit 0."""
        code = run_validation(tmp_path, sample_raw=False)
        assert code == 0


# ---------------------------------------------------------------------------
# EXPECTED_KEYS contract
# ---------------------------------------------------------------------------


def test_expected_keys_count() -> None:
    assert (
        len(EXPECTED_KEYS) == 19
    ), f"Expected 19 keys, got {len(EXPECTED_KEYS)}: {EXPECTED_KEYS}"
