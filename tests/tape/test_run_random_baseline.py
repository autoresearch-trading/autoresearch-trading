# tests/tape/test_run_random_baseline.py
"""Unit tests for scripts/run_random_baseline.py.

TDD RED phase: tests written before any production code exists.

Tests cover:
1. Projection matrix determinism (same seed → same matrix).
2. Projection matrix is fixed across folds (non-adaptive).
3. End-to-end smoke: synthetic cache → JSON output with expected schema.
4. Output schema matches Gate 0 (same keys so caller can delta-compare).
5. April hold-out guard passes through unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Re-use the synthetic cache builder from the Gate 0 test module
# (avoids copy-paste of shard helpers)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent


def _import_random_baseline():
    """Import scripts/run_random_baseline.py as a module."""
    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_random_baseline",
        _REPO_ROOT / "scripts" / "run_random_baseline.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("scripts/run_random_baseline.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Shard helpers (minimal copy from test_run_gate0 to avoid cross-test coupling)
# ---------------------------------------------------------------------------

WINDOW_LEN = 200
N_FEATURES = 17
HORIZONS = (10, 50, 100, 500)
_SYMBOLS = ("SYNTH_A", "SYNTH_B")
_DATES = ("2025-11-01", "2025-11-02", "2025-11-03")
_N_EVENTS = 6_000


def _make_direction_labels(
    n: int, horizon: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
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
    import datetime

    features = rng.standard_normal((n_events, N_FEATURES)).astype(np.float32)
    event_ts = np.sort(
        rng.integers(1_700_000_000_000, 1_700_100_000_000, size=n_events)
    ).astype(np.int64)
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
    for wy in ("stress", "informed_flow", "climax", "spring", "absorption"):
        shard[f"wy_{wy}"] = rng.integers(0, 2, size=n_events, dtype=np.int8)
    return shard


def _write_shard(shard: dict[str, Any], out_dir: Path) -> Path:
    symbol = str(shard["symbol"])
    date = str(shard["date"])
    path = out_dir / f"{symbol}__{date}.npz"
    np.savez_compressed(path, **shard)
    return path


def build_synthetic_cache(tmp_dir: Path, n_events: int = _N_EVENTS) -> Path:
    cache_dir = tmp_dir / "cache"
    cache_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed=42)
    for sym in _SYMBOLS:
        for date in _DATES:
            shard = _make_shard(sym, date, n_events, rng)
            _write_shard(shard, cache_dir)
    return cache_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rb():
    """Import run_random_baseline module; skip if not yet created."""
    try:
        return _import_random_baseline()
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"scripts/run_random_baseline.py not yet created: {exc}")


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rp_cache")
    return build_synthetic_cache(tmp)


# ---------------------------------------------------------------------------
# Small CLI flags matching Gate 0 test pattern
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


# ---------------------------------------------------------------------------
# Test 1: Projection matrix determinism
# ---------------------------------------------------------------------------


class TestProjectionMatrixDeterminism:
    def test_same_seed_same_matrix(self, rb):
        """build_projection_matrix(seed=42) must return the identical matrix on
        every call — it is a fixed, non-adaptive transform."""
        P1 = rb.build_projection_matrix(seed=42)
        P2 = rb.build_projection_matrix(seed=42)
        np.testing.assert_array_equal(P1, P2)

    def test_different_seeds_different_matrices(self, rb):
        """Different seeds must produce different projection matrices."""
        P42 = rb.build_projection_matrix(seed=42)
        P99 = rb.build_projection_matrix(seed=99)
        assert not np.array_equal(P42, P99)

    def test_matrix_shape_83_by_20(self, rb):
        """Projection matrix must map 83-dim flat features to 20 dims."""
        P = rb.build_projection_matrix(seed=42)
        assert P.shape == (83, 20), f"Expected (83, 20), got {P.shape}"

    def test_columns_have_unit_variance_in_expectation(self, rb):
        """Column normalization: each output dim should have std ~ 1 when applied
        to zero-mean unit-variance input.  Check via column norms of P."""
        P = rb.build_projection_matrix(seed=42)
        # Each column of P was drawn from N(0,1) and column-normalized so that
        # var(X @ P[:,j]) = 1 when X ~ N(0,I_{83}).
        # Equivalently ||P[:,j]||^2 should be 1.0 for each j.
        col_norms_sq = np.sum(P**2, axis=0)  # shape (20,)
        np.testing.assert_allclose(col_norms_sq, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Projection is applied identically across folds (non-adaptive)
# ---------------------------------------------------------------------------


class TestProjectionNonAdaptive:
    def test_projection_matrix_unchanged_after_fit_fold(self, rb):
        """_fit_fold_rp must NOT modify the projection matrix — it must be the
        same object (or at least identical values) before and after fitting."""
        rng = np.random.default_rng(7)
        X_train = rng.standard_normal((200, 83)).astype(np.float32)
        y_train = rng.integers(0, 2, 200)
        P = rb.build_projection_matrix(seed=42)
        P_before = P.copy()

        rb._fit_fold_rp(X_train, y_train, P)

        np.testing.assert_array_equal(
            P, P_before, err_msg="Projection matrix was mutated"
        )

    def test_fit_fold_rp_returns_scaler_and_lr(self, rb):
        """_fit_fold_rp returns (StandardScaler, LogisticRegression) — no PCA."""
        rng = np.random.default_rng(8)
        X_train = rng.standard_normal((300, 83)).astype(np.float32)
        y_train = rng.integers(0, 2, 300)
        P = rb.build_projection_matrix(seed=42)
        result = rb._fit_fold_rp(X_train, y_train, P)
        assert (
            len(result) == 2
        ), f"Expected (scaler, lr), got tuple of len {len(result)}"
        scaler, lr = result
        assert hasattr(scaler, "mean_"), "StandardScaler not fitted"
        assert hasattr(lr, "coef_"), "LogisticRegression not fitted"


# ---------------------------------------------------------------------------
# Test 3: End-to-end smoke — produces JSON with expected schema
# ---------------------------------------------------------------------------


class TestCLIEndToEnd:
    def test_produces_json_and_md(self, rb, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "rp_baseline")
        rc = rb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--seed",
                "42",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        assert rc == 0
        assert Path(out_prefix + ".json").exists(), "JSON not created"
        assert Path(out_prefix + ".md").exists(), "MD not created"

    def test_json_schema_matches_gate0_schema(self, rb, synthetic_cache, tmp_path):
        """JSON output must have same top-level structure as Gate 0 so the
        delta-comparison script can subtract them directly."""
        out_prefix = str(tmp_path / "rp_schema")
        rb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--seed",
                "42",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)

        # Must have a 'summary' key
        assert "summary" in data, "JSON missing top-level 'summary'"

        # Per-symbol data must have h{horizon} → accuracy_mean or error
        sym_data = data.get("SYNTH_A", {})
        h100 = sym_data.get("h100", {})
        assert (
            "accuracy_mean" in h100 or "error" in h100
        ), f"h100 missing both accuracy_mean and error: {h100}"

    def test_summary_has_standard_fields(self, rb, synthetic_cache, tmp_path):
        """Summary must include mean_accuracy, n_symbols_above_514, n_successful, n_errors."""
        out_prefix = str(tmp_path / "rp_summary")
        rb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--seed",
                "42",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        h100_summary = data["summary"].get("h100", {})
        for field in (
            "mean_accuracy",
            "n_symbols_above_514",
            "n_successful",
            "n_errors",
        ):
            assert field in h100_summary, f"summary['h100'] missing '{field}'"

    def test_method_tag_in_json(self, rb, synthetic_cache, tmp_path):
        """JSON must include a top-level 'method' key identifying this as random projection."""
        out_prefix = str(tmp_path / "rp_method")
        rb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--seed",
                "42",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        assert "method" in data, "JSON missing 'method' key"
        assert (
            "random" in data["method"].lower()
        ), f"'method' should identify random projection, got: {data['method']}"

    def test_april_holdout_guard_propagated(self, rb, tmp_path):
        """Shards with date >= 2026-04-14 must raise an error (same as Gate 0)."""
        rng = np.random.default_rng(0)
        cache_dir = tmp_path / "bad_cache"
        cache_dir.mkdir()
        shard = _make_shard("BTC", "2026-04-14", _N_EVENTS, rng)
        _write_shard(shard, cache_dir)
        with pytest.raises((ValueError, AssertionError, SystemExit)):
            rb.main(
                [
                    "--cache",
                    str(cache_dir),
                    "--symbols",
                    "BTC",
                    "--horizons",
                    "100",
                    "--seed",
                    "42",
                    "--out",
                    str(tmp_path / "out"),
                ]
                + _SMALL_CLI_FLAGS
            )
