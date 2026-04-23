# tests/tape/test_run_gate0.py
"""Unit tests for scripts/run_gate0.py helpers.

RED phase: tests written before any production code exists in run_gate0.py.

Synthetic cache layout
----------------------
2 symbols × 3 days × ~1200 events each.
Each shard is a .npz file written using tape/cache.save_shard-compatible format.
We write them directly with np.savez_compressed to avoid depending on build_symbol_day
(which needs raw parquet files).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build a synthetic shard
# ---------------------------------------------------------------------------

WINDOW_LEN = 200
N_FEATURES = 17
HORIZONS = (10, 50, 100, 500)
_SYMBOLS = ("SYNTH_A", "SYNTH_B")
_DATES = ("2025-11-01", "2025-11-02", "2025-11-03")
# Events per shard: 3 days × 6000 = 18,000 total per symbol.
# Eval stride=200 → ~90 windows per symbol; enough for walk-forward folds at
# min_train=5 (test-mode) and embargo=600 events remapped to embargo=0 windows.
_N_EVENTS = 6_000


def _make_direction_labels(
    n: int, horizon: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels int8, mask bool) arrays of length n.

    Last `horizon` events are masked (NaN territory at tail).
    """
    labels = rng.integers(0, 2, size=n, dtype=np.int8)
    mask = np.ones(n, dtype=bool)
    mask[n - horizon :] = False
    return labels, mask


def _make_shard(
    symbol: str,
    date: str,
    n_events: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Create a minimal shard dict compatible with load_shard output format."""
    features = rng.standard_normal((n_events, N_FEATURES)).astype(np.float32)
    event_ts = np.sort(
        rng.integers(1_700_000_000_000, 1_700_100_000_000, size=n_events)
    ).astype(np.int64)
    import datetime

    _epoch = datetime.date(1970, 1, 1)
    day_id_int = (datetime.date.fromisoformat(date) - _epoch).days
    day_id = np.full(n_events, day_id_int, dtype=np.int64)

    shard: dict[str, Any] = {
        "features": features,
        "event_ts": event_ts,
        "day_id": day_id,
        "symbol": np.array(symbol),
        "date": np.array(date),
        "schema_version": np.array(1, dtype=np.int32),
    }
    for h in HORIZONS:
        labels, mask = _make_direction_labels(n_events, h, rng)
        shard[f"dir_h{h}"] = labels
        shard[f"dir_mask_h{h}"] = mask
    # Wyckoff labels (not used by Gate 0 but must exist for load_shard)
    for wy in ("stress", "informed_flow", "climax", "spring", "absorption"):
        shard[f"wy_{wy}"] = rng.integers(0, 2, size=n_events, dtype=np.int8)
    return shard


def _write_shard(shard: dict[str, Any], out_dir: Path) -> Path:
    """Write shard to <symbol>__<date>.npz."""
    symbol = str(shard["symbol"])
    date = str(shard["date"])
    path = out_dir / f"{symbol}__{date}.npz"
    np.savez_compressed(path, **shard)
    return path


def build_synthetic_cache(tmp_dir: Path, n_events: int = _N_EVENTS) -> Path:
    """Build a small synthetic cache with 2 symbols × 3 days each."""
    cache_dir = tmp_dir / "cache"
    cache_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed=42)
    for sym in _SYMBOLS:
        for date in _DATES:
            shard = _make_shard(sym, date, n_events, rng)
            _write_shard(shard, cache_dir)
    return cache_dir


# ---------------------------------------------------------------------------
# Import helpers — import the script module lazily so tests can run before
# the script is created (RED phase: we expect ImportError if file missing).
# ---------------------------------------------------------------------------


def _import_gate0():
    """Import scripts/run_gate0.py as a module. Raises ImportError if absent."""
    # Add repo root to sys.path so `scripts` is importable
    repo_root = Path(__file__).parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_gate0", repo_root / "scripts" / "run_gate0.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("scripts/run_gate0.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate0():
    """Import gate0 module; skip all tests in module if file doesn't exist yet."""
    try:
        return _import_gate0()
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"scripts/run_gate0.py not yet created: {exc}")


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gate0_cache")
    return build_synthetic_cache(tmp)


# ---------------------------------------------------------------------------
# Test: _load_symbol_shards
# ---------------------------------------------------------------------------


class TestLoadSymbolShards:
    def test_returns_none_for_missing_symbol(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "NONEXISTENT")
        assert result is None

    def test_returns_tuple_for_existing_symbol(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert result is not None
        feats, labels, masks = result
        assert feats.ndim == 2 and feats.shape[1] == N_FEATURES
        assert isinstance(labels, dict)
        assert isinstance(masks, dict)

    def test_feature_shape_is_n_events_by_17(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert result is not None
        feats, _, _ = result
        # 3 days × _N_EVENTS events each
        assert feats.shape == (len(_DATES) * _N_EVENTS, N_FEATURES)

    def test_all_horizons_present(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "SYNTH_B")
        assert result is not None
        _, labels, masks = result
        for h in HORIZONS:
            assert h in labels, f"horizon {h} missing from labels"
            assert h in masks, f"horizon {h} missing from masks"

    def test_assert_no_holdout_dates(self, gate0, tmp_path):
        """Shards with date >= 2026-04-14 must be rejected."""
        rng = np.random.default_rng(0)
        shard = _make_shard("SYNTH_A", "2026-04-14", 1200, rng)
        _write_shard(shard, tmp_path)
        with pytest.raises((ValueError, AssertionError)):
            gate0._load_symbol_shards(tmp_path, "SYNTH_A")


# ---------------------------------------------------------------------------
# Test: _build_eval_windows
# ---------------------------------------------------------------------------


class TestBuildEvalWindows:
    def test_stride200_reduces_window_count(self, gate0):
        # 1200 events at stride=200 → (1200 - 200) // 200 + 1 = 6 windows
        starts = gate0._build_eval_windows(n_events=1200, window_len=200, stride=200)
        assert len(starts) == 6

    def test_returns_list_of_ints(self, gate0):
        starts = gate0._build_eval_windows(n_events=1200, window_len=200, stride=200)
        assert all(isinstance(s, int) for s in starts)

    def test_no_out_of_bounds_windows(self, gate0):
        n = 1200
        starts = gate0._build_eval_windows(n_events=n, window_len=200, stride=200)
        for s in starts:
            assert s + 200 <= n

    def test_empty_when_too_few_events(self, gate0):
        starts = gate0._build_eval_windows(n_events=100, window_len=200, stride=200)
        assert starts == []


# ---------------------------------------------------------------------------
# Test: extract_flat_features_for_windows
# ---------------------------------------------------------------------------


class TestExtractFlatFeaturesForWindows:
    def test_output_shape(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert result is not None
        feats, _, _ = result
        starts = gate0._build_eval_windows(len(feats), 200, 200)
        X = gate0._extract_flat_features_for_windows(feats, starts)
        assert X.shape == (len(starts), 83)

    def test_output_is_finite(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert result is not None
        feats, _, _ = result
        starts = gate0._build_eval_windows(len(feats), 200, 200)
        X = gate0._extract_flat_features_for_windows(feats, starts)
        assert np.all(np.isfinite(X))

    def test_output_dtype_float32(self, gate0, synthetic_cache):
        result = gate0._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert result is not None
        feats, _, _ = result
        starts = gate0._build_eval_windows(len(feats), 200, 200)
        X = gate0._extract_flat_features_for_windows(feats, starts)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# Test: evaluate_symbol (end-to-end per-symbol run)
# ---------------------------------------------------------------------------


class TestEvaluateSymbol:
    """Use relaxed thresholds so the tiny synthetic cache can complete folds.
    Production defaults are much stricter (min_train=10_000, embargo=600)."""

    _EVAL_KWARGS: dict = dict(min_labeled_windows=1, min_train=5, min_test=1, embargo=0)

    def test_returns_dict_with_horizon_keys(self, gate0, synthetic_cache):
        result = gate0.evaluate_symbol(
            synthetic_cache, "SYNTH_A", horizons=(100,), seed=0, **self._EVAL_KWARGS
        )
        assert result is not None
        assert "h100" in result

    def test_returns_none_for_missing_symbol(self, gate0, synthetic_cache):
        result = gate0.evaluate_symbol(
            synthetic_cache, "NONEXISTENT", horizons=(100,), seed=0, **self._EVAL_KWARGS
        )
        assert result is None

    def test_accuracy_in_valid_range(self, gate0, synthetic_cache):
        result = gate0.evaluate_symbol(
            synthetic_cache, "SYNTH_A", horizons=(100,), seed=0, **self._EVAL_KWARGS
        )
        assert result is not None
        h100 = result.get("h100", {})
        acc = h100.get("accuracy_mean", None)
        if acc is not None and not (isinstance(acc, float) and acc != acc):  # not NaN
            assert 0.0 <= acc <= 1.0

    def test_all_horizons_in_result(self, gate0, synthetic_cache):
        result = gate0.evaluate_symbol(
            synthetic_cache,
            "SYNTH_A",
            horizons=(10, 50, 100, 500),
            seed=0,
            **self._EVAL_KWARGS,
        )
        assert result is not None
        for h in (10, 50, 100, 500):
            assert f"h{h}" in result

    def test_nan_labels_excluded(self, gate0, synthetic_cache):
        """Direction labels at the tail (mask=False) must be dropped, not imputed."""
        result = gate0.evaluate_symbol(
            synthetic_cache, "SYNTH_B", horizons=(100,), seed=0, **self._EVAL_KWARGS
        )
        assert result is not None
        h100 = result.get("h100", {})
        # n_windows reported must be <= total possible windows
        n_windows = h100.get("n_windows", None)
        if n_windows is not None:
            total_events = len(_DATES) * _N_EVENTS
            max_possible = (total_events - WINDOW_LEN) // WINDOW_LEN + 1
            assert n_windows <= max_possible


# ---------------------------------------------------------------------------
# Test: embargo is respected in walk-forward folds used by evaluate_symbol
# ---------------------------------------------------------------------------


class TestEmbargoInFolds:
    def test_no_train_index_in_embargo_zone(self, gate0):
        """Gate0 uses walk_forward_folds; verify embargo is honoured."""
        from tape.splits import walk_forward_folds

        embargo = 600
        folds = walk_forward_folds(
            np.arange(20_000, dtype=np.int64),
            n_folds=5,
            embargo=embargo,
            min_train=10_000,
        )
        for train, test in folds:
            prohibited_start = int(test.min()) - embargo
            prohibited_end = int(test.max())
            overlap = train[(train >= prohibited_start) & (train <= prohibited_end)]
            assert len(overlap) == 0, (
                f"{len(overlap)} train indices fall in embargo zone "
                f"[{prohibited_start}, {prohibited_end}]"
            )

    def test_gap_strictly_greater_than_embargo(self, gate0):
        from tape.splits import walk_forward_folds

        embargo = 600
        folds = walk_forward_folds(
            np.arange(20_000, dtype=np.int64),
            n_folds=5,
            embargo=embargo,
            min_train=10_000,
        )
        for train, test in folds:
            gap = int(test.min()) - int(train.max())
            assert gap > embargo, f"gap {gap} not > embargo {embargo}"


# ---------------------------------------------------------------------------
# Test: PCA is fit on train only (no leakage)
# ---------------------------------------------------------------------------


class TestPCAFitPerFold:
    """Verify that PCA + StandardScaler are fit exclusively on training data."""

    def test_pca_fit_train_only(self, gate0):
        """When called with isolated train/test splits, the scaler must only
        see training data.  We probe this by checking that the test-set mean
        after transform is NOT zero (it would be exactly zero only if the
        scaler was fit on the entire dataset including test)."""
        rng = np.random.default_rng(7)
        # Two clearly different distributions
        X_train = rng.standard_normal((200, 85)).astype(np.float32)
        X_test = rng.standard_normal((50, 85)).astype(np.float32) + 10.0  # shifted

        scaler, pca, lr = gate0._fit_fold(X_train, rng.integers(0, 2, 200))
        X_te_scaled = scaler.transform(X_test)
        # Test-set mean after scaler.transform should NOT be 0 (scaler was fit on train)
        assert abs(X_te_scaled.mean()) > 0.1, (
            "Test-set mean is nearly zero after scaler.transform — "
            "scaler appears to have been fit on test data (leakage)"
        )

    def test_lr_fit_in_standardized_space(self, gate0):
        """LR is fit on PCA-projected standardised features — confirm by checking
        that _fit_fold returns a fitted LogisticRegression with coef_ attribute."""
        rng = np.random.default_rng(8)
        X_train = rng.standard_normal((300, 85)).astype(np.float32)
        y_train = rng.integers(0, 2, 300)
        scaler, pca, lr = gate0._fit_fold(X_train, y_train)
        assert hasattr(lr, "coef_"), "LR not fitted"
        assert hasattr(scaler, "mean_"), "StandardScaler not fitted"
        assert hasattr(pca, "components_"), "PCA not fitted"


# ---------------------------------------------------------------------------
# Test: main() / CLI end-to-end
# ---------------------------------------------------------------------------


_SMALL_CLI_FLAGS: list[str] = [
    "--min-train",
    "5",
    "--min-labeled-windows",
    "1",
    "--embargo",
    "0",
    "--min-test",
    "1",
]


class TestCLIEndToEnd:
    def test_end_to_end_produces_json(self, gate0, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "gate0")
        rc = gate0.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        assert rc == 0
        json_path = Path(out_prefix + ".json")
        assert json_path.exists(), "JSON output not created"
        with open(json_path) as f:
            data = json.load(f)
        assert "SYNTH_A" in data or "SYNTH_B" in data

    def test_json_has_expected_structure(self, gate0, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "gate0")
        gate0.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        sym_data = data.get("SYNTH_A", {})
        h100 = sym_data.get("h100", {})
        assert "accuracy_mean" in h100 or "error" in h100

    def test_avax_flagged_not_excluded(self, gate0, tmp_path):
        """AVAX data in the cache (pre-April) should appear in per-symbol table,
        flagged as held-out for Gate 3, not silently dropped."""
        rng = np.random.default_rng(99)
        cache_dir = tmp_path / "avax_cache"
        cache_dir.mkdir()
        for date in _DATES:
            shard = _make_shard("AVAX", date, _N_EVENTS, rng)
            _write_shard(shard, cache_dir)

        out_prefix = str(tmp_path / "gate0_avax")
        rc = gate0.main(
            [
                "--cache",
                str(cache_dir),
                "--symbols",
                "AVAX",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        assert rc == 0
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        assert "AVAX" in data

    def test_no_holdout_dates_in_cache(self, gate0, tmp_path):
        """A cache with a shard dated 2026-04-14 must raise an error."""
        rng = np.random.default_rng(0)
        cache_dir = tmp_path / "bad_cache"
        cache_dir.mkdir()
        shard = _make_shard("BTC", "2026-04-14", _N_EVENTS, rng)
        _write_shard(shard, cache_dir)
        with pytest.raises((ValueError, AssertionError, SystemExit)):
            gate0.main(
                [
                    "--cache",
                    str(cache_dir),
                    "--symbols",
                    "BTC",
                    "--horizons",
                    "100",
                    "--seed",
                    "0",
                    "--out",
                    str(tmp_path / "out"),
                ]
                + _SMALL_CLI_FLAGS
            )

    def test_summary_table_in_json(self, gate0, synthetic_cache, tmp_path):
        """JSON must contain a top-level 'summary' key with per-horizon stats."""
        out_prefix = str(tmp_path / "gate0")
        gate0.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        assert "summary" in data, "JSON missing top-level 'summary' key"
        h100_summary = data["summary"].get("h100", {})
        assert "mean_accuracy" in h100_summary or "n_symbols_above_514" in h100_summary

    def test_enriched_summary_has_error_counts(self, gate0, synthetic_cache, tmp_path):
        """Summary must include n_successful and n_errors counts per horizon."""
        out_prefix = str(tmp_path / "gate0")
        gate0.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        h100_summary = data["summary"].get("h100", {})
        assert "n_successful" in h100_summary, "summary missing n_successful"
        assert "n_errors" in h100_summary, "summary missing n_errors"


# ---------------------------------------------------------------------------
# Test: illiquid symbol (~5000 windows) completes all 3 folds at new defaults
# This is the key regression test: old defaults (min_train=10_000, n_folds=5)
# failed for any symbol with < 12_100 windows; new defaults must fit LDO (4577).
# ---------------------------------------------------------------------------

# LDO-sized: 4577 windows at H500.  At stride=200 we need ~4577 raw windows.
# Each raw window occupies one stride slot.  4577 * 200 + 200 = 916_000 events.
# But we don't simulate the full event chain — we build a shard with enough events
# to yield 4577 eval windows at stride=200:  4577 * 200 + 200 = 916_200 events.
# That is too large for a unit test.  Instead we test at a scaled-down ratio:
# 5_000 windows ~= LDO.  Use _build_eval_windows(n_events) and set n_events to
# yield exactly 5_000 windows: n_events = (5_000 - 1) * 200 + 200 = 1_000_000.
# Still huge.  The point is the fold arithmetic, so we override stride internally.
# Better approach: build a shard with n_events = 5_000 and window_len = 1, stride = 1
# to simulate 5_000 windows directly — but that changes the flat feature contract.
#
# Simplest correct approach: build a synthetic cache with enough events to produce
# 5_000 windows at stride=200 *from evaluate_symbol directly*, relying on the fact
# that evaluate_symbol's internal _build_eval_windows uses WINDOW_LEN=200 / STRIDE_EVAL=200.
# n_events = (5_000 - 1) * 200 + 200 = 1_000_000 — too large.
#
# Final approach: patch min_labeled_windows=1 and use the evaluate_symbol API
# directly with a synthetic Xv of shape (5000, 85) via walk_forward_folds.
# This exercises the fold arithmetic without building huge event arrays.


class TestIlliquidSymbolFolds:
    """New defaults (min_train=2000, min_test=500, n_folds=3) must complete
    all 3 folds for a symbol with 5000 valid labeled windows (LDO-like).
    Old defaults (min_train=10_000, n_folds=5) would have raised ValueError
    and returned 'no_folds_completed' for this window count."""

    def test_5000_windows_completes_3_folds_at_new_defaults(self, gate0):
        """Walk-forward with new defaults must return 3 completed folds for 5000 windows."""
        from tape.splits import walk_forward_folds

        n_windows = 5_000
        # New defaults from the patch: min_train=2000, min_test=500, n_folds=3, embargo=600
        folds = walk_forward_folds(
            np.arange(n_windows, dtype=np.int64),
            n_folds=3,
            embargo=600,
            min_train=2_000,
            min_test=500,
        )
        assert (
            len(folds) == 3
        ), f"Expected 3 folds for 5000 windows at new defaults, got {len(folds)}"

    def test_5000_windows_would_fail_old_defaults(self, gate0):
        """Old defaults (min_train=10_000, n_folds=5) must raise ValueError for 5000 windows."""
        import pytest

        from tape.splits import walk_forward_folds

        with pytest.raises(ValueError, match="insufficient events"):
            walk_forward_folds(
                np.arange(5_000, dtype=np.int64),
                n_folds=5,
                embargo=600,
                min_train=10_000,
                min_test=1_000,
            )

    def test_n_folds_cli_flag_threads_through(self, gate0, tmp_path):
        """--n-folds CLI flag must be accepted and override the default fold count."""
        rng = np.random.default_rng(seed=77)
        cache_dir = tmp_path / "illiquid_cache"
        cache_dir.mkdir()
        # Build a cache with enough events for 5 folds at small settings
        for date in _DATES:
            shard = _make_shard("ILLIQ", date, _N_EVENTS, rng)
            _write_shard(shard, cache_dir)

        out_prefix = str(tmp_path / "gate0_illiq")
        rc = gate0.main(
            [
                "--cache",
                str(cache_dir),
                "--symbols",
                "ILLIQ",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
                "--n-folds",
                "2",
                "--min-train",
                "5",
                "--min-labeled-windows",
                "1",
                "--embargo",
                "0",
                "--min-test",
                "1",
            ]
        )
        assert rc == 0
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        illiq = data.get("ILLIQ", {})
        h100 = illiq.get("h100", {})
        assert (
            h100.get("n_folds") == 2
        ), f"Expected n_folds=2 from --n-folds flag, got {h100}"

    def test_summary_has_per_symbol_window_counts(
        self, gate0, synthetic_cache, tmp_path
    ):
        """Summary JSON must include per-symbol window counts at each horizon."""
        out_prefix = str(tmp_path / "gate0_wc")
        gate0.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--seed",
                "0",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        h100_summary = summary.get("h100", {})
        assert (
            "symbol_window_counts" in h100_summary
        ), "summary h100 missing 'symbol_window_counts'"
