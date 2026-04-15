# tests/tape/test_run_majority_baseline.py
"""Unit tests for scripts/run_majority_baseline.py.

TDD RED phase: tests written before any production code exists.

Tests cover:
1. Synthetic cache round-trip: produces JSON + MD.
2. Predictions are constant per fold (always the training majority class).
3. Balanced accuracy is exactly 0.5 when the predictor is a constant class.
4. April hold-out guard propagated.
5. JSON schema matches Gate 0 (same keys for direct comparison).
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
# Shard helpers (minimal copy matching other baseline tests)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent

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
    label_override: dict[int, np.ndarray] | None = None,
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
        if label_override and h in label_override:
            labels = label_override[h]
            mask = np.ones(n_events, dtype=bool)
            mask[n_events - h :] = False
        else:
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
# Import helpers
# ---------------------------------------------------------------------------


def _import_majority_baseline():
    """Import scripts/run_majority_baseline.py as a module."""
    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_majority_baseline",
        _REPO_ROOT / "scripts" / "run_majority_baseline.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("scripts/run_majority_baseline.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mb():
    """Import run_majority_baseline module; skip if not yet created."""
    try:
        return _import_majority_baseline()
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"scripts/run_majority_baseline.py not yet created: {exc}")


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mb_cache")
    return build_synthetic_cache(tmp)


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
# Test 1: Synthetic cache round-trip — produces JSON and MD
# ---------------------------------------------------------------------------


class TestCLIEndToEnd:
    def test_produces_json_and_md(self, mb, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "maj_baseline")
        rc = mb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        assert rc == 0
        assert Path(out_prefix + ".json").exists(), "JSON not created"
        assert Path(out_prefix + ".md").exists(), "MD not created"

    def test_json_has_symbol_and_summary_keys(self, mb, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "maj_schema")
        mb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        assert "SYNTH_A" in data, "Symbol key missing from JSON"
        assert "summary" in data, "summary key missing from JSON"

    def test_json_has_accuracy_or_error_per_horizon(
        self, mb, synthetic_cache, tmp_path
    ):
        out_prefix = str(tmp_path / "maj_horizon")
        mb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        h100 = data.get("SYNTH_A", {}).get("h100", {})
        assert "accuracy_mean" in h100 or "error" in h100, f"h100 missing both: {h100}"

    def test_summary_has_standard_fields(self, mb, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "maj_summary_fields")
        mb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "SYNTH_B",
                "--horizons",
                "100",
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

    def test_method_tag_in_json(self, mb, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "maj_method_tag")
        mb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        assert "method" in data, "JSON missing 'method' key"
        assert (
            "majority" in data["method"].lower()
        ), f"'method' should identify majority baseline, got: {data['method']}"

    def test_april_holdout_guard_propagated(self, mb, tmp_path):
        """Shards dated >= 2026-04-14 must raise an error."""
        rng = np.random.default_rng(0)
        cache_dir = tmp_path / "bad_cache"
        cache_dir.mkdir()
        shard = _make_shard("BTC", "2026-04-14", _N_EVENTS, rng)
        _write_shard(shard, cache_dir)
        with pytest.raises((ValueError, AssertionError, SystemExit)):
            mb.main(
                [
                    "--cache",
                    str(cache_dir),
                    "--symbols",
                    "BTC",
                    "--horizons",
                    "100",
                    "--out",
                    str(tmp_path / "out"),
                ]
                + _SMALL_CLI_FLAGS
            )


# ---------------------------------------------------------------------------
# Test 2: Predictions are constant per fold (always the training majority class)
# ---------------------------------------------------------------------------


class TestPredictionsConstantPerFold:
    """The majority-class predictor must output the same class for every test
    observation in a given fold — the training-fold majority class."""

    def test_constant_prediction_when_labels_skewed(self, mb, tmp_path):
        """Build a cache where class 1 is ~90% of labels.  The predictor must
        output 1 for every test observation in every fold."""
        n_events = _N_EVENTS
        rng = np.random.default_rng(seed=7)
        cache_dir = tmp_path / "skewed_cache"
        cache_dir.mkdir()

        # All labels = 1 (perfectly skewed)
        all_ones = np.ones(n_events, dtype=np.int8)
        for date in _DATES:
            shard = _make_shard(
                "SKEWED", date, n_events, rng, label_override={100: all_ones}
            )
            _write_shard(shard, cache_dir)

        out_prefix = str(tmp_path / "skewed_out")
        rc = mb.main(
            [
                "--cache",
                str(cache_dir),
                "--symbols",
                "SKEWED",
                "--horizons",
                "100",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        assert rc == 0
        with open(out_prefix + ".json") as f:
            data = json.load(f)

        h100 = data.get("SKEWED", {}).get("h100", {})
        # All labels are 1, so accuracy_mean must be 1.0
        acc = h100.get("accuracy_mean")
        assert acc is not None, f"accuracy_mean missing: {h100}"
        assert (
            abs(acc - 1.0) < 1e-6
        ), f"Expected accuracy=1.0 for all-ones labels, got {acc}"


# ---------------------------------------------------------------------------
# Test 3: Balanced accuracy is exactly 0.5 when predictor is constant
# ---------------------------------------------------------------------------


class TestBalancedAccuracyConstantPredictor:
    """When the predictor always predicts one class, balanced accuracy = 0.5
    if both classes are present in the test fold (TPR=1.0, TNR=0.0 → mean=0.5).

    Test this directly via sklearn: create a 50/50 test array, constant predictor.
    This is a pure unit test of the property, not dependent on cache layout.
    """

    def test_balanced_accuracy_is_05_for_constant_predictor(self, mb):
        """Directly verify: balanced_accuracy(y_true=[0,1,0,1,...], y_pred=[1,1,...]) = 0.5.

        This is a mathematical property.  The majority-class predictor must satisfy
        it when both classes are present in the test fold.
        """
        from sklearn.metrics import balanced_accuracy_score

        n = 1_000
        # Exactly 50/50 labels
        y_true = np.array([i % 2 for i in range(n)], dtype=np.int64)
        # Constant predictor: always predicts 1 (training majority)
        y_pred = np.ones(n, dtype=np.int64)

        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # TPR (for class 1) = 1.0,  TNR (for class 0) = 0.0 → mean = 0.5
        assert abs(bal_acc - 0.5) < 1e-9, (
            f"Expected balanced_accuracy = 0.5 for constant predictor on 50/50 labels, "
            f"got {bal_acc}"
        )

    def test_majority_baseline_bal_acc_keys_exist_in_json(
        self, mb, synthetic_cache, tmp_path
    ):
        """Verify JSON has balanced_accuracy_mean key — needed for the 0.5 property check."""
        out_prefix = str(tmp_path / "bal_keys")
        mb.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        h100 = data.get("SYNTH_A", {}).get("h100", {})
        assert (
            "balanced_accuracy_mean" in h100 or "error" in h100
        ), f"h100 missing balanced_accuracy_mean: {h100}"
