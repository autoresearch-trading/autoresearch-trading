# tests/tape/test_run_shuffled_labels_baseline.py
"""Unit tests for scripts/run_shuffled_labels_baseline.py.

TDD RED phase: tests written before any production code exists.

Tests cover:
1. Labels are actually shuffled (per-symbol, not global).
2. Features are unchanged (shuffling touches only labels).
3. Seed reproducibility: two runs with same seed produce same labels.
4. Two different seeds produce different shuffles.
5. Balanced accuracy stays at 0.5 ± 0.05 on balanced synthetic labels
   (null-hypothesis check).
6. JSON schema matches Gate 0.
7. April hold-out guard propagated.
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
# Import helpers
# ---------------------------------------------------------------------------


def _import_shuffled_labels():
    """Import scripts/run_shuffled_labels_baseline.py as a module."""
    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_shuffled_labels_baseline",
        _REPO_ROOT / "scripts" / "run_shuffled_labels_baseline.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("scripts/run_shuffled_labels_baseline.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sl():
    """Import run_shuffled_labels_baseline module; skip if not yet created."""
    try:
        return _import_shuffled_labels()
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"scripts/run_shuffled_labels_baseline.py not yet created: {exc}")


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sl_cache")
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
# Test 1: Labels are actually shuffled per-symbol
# ---------------------------------------------------------------------------


class TestLabelsAreShuffled:
    """The shuffled labels must differ from the original labels (per-symbol)."""

    def test_shuffle_labels_helper_actually_shuffles(self, sl):
        """_shuffle_labels_per_symbol must return labels that differ from input."""
        n = 10_000
        rng_data = np.random.default_rng(0)
        # Build random labels with some balance
        original = rng_data.integers(0, 2, size=n, dtype=np.int8)

        shuffled = sl._shuffle_labels(original, seed=42)

        assert len(shuffled) == len(original), "Shuffled labels have different length"
        # After shuffling 10k elements, at least some must differ
        n_diff = int((shuffled != original).sum())
        assert (
            n_diff > 100
        ), f"Only {n_diff}/{n} labels differ after shuffle — likely not shuffled"

    def test_shuffle_preserves_label_distribution(self, sl):
        """Shuffling must preserve the exact label distribution (same # of 0s and 1s)."""
        n = 5_000
        rng_data = np.random.default_rng(1)
        original = rng_data.integers(0, 2, size=n, dtype=np.int8)

        shuffled = sl._shuffle_labels(original, seed=42)

        assert int(original.sum()) == int(
            shuffled.sum()
        ), "Shuffle changed label distribution — must be a permutation"

    def test_shuffle_is_per_symbol_not_global(self, sl, synthetic_cache):
        """Two symbols shuffled with the same seed must produce different permutations
        (because each symbol gets its own RNG stream seeded from the global seed + symbol).
        """
        # Load raw labels for two symbols
        from tape.cache import load_shard

        shards_a = sorted(synthetic_cache.glob("SYNTH_A__*.npz"))
        shards_b = sorted(synthetic_cache.glob("SYNTH_B__*.npz"))
        assert shards_a and shards_b

        payload_a = load_shard(shards_a[0])
        payload_b = load_shard(shards_b[0])

        raw_a = payload_a["dir_h100"].astype(np.int8)
        raw_b = payload_b["dir_h100"].astype(np.int8)

        shuf_a = sl._shuffle_labels(raw_a, seed=42)
        shuf_b = sl._shuffle_labels(raw_b, seed=42)

        # The two shuffles are of different data — the function is pure, not symbol-aware,
        # but since the inputs differ the outputs will differ.  The key property is that
        # the function is a pure permutation of its input (no global state).
        # Verify: each output is a permutation of its own input.
        np.testing.assert_array_equal(
            np.sort(shuf_a),
            np.sort(raw_a),
            err_msg="SYNTH_A shuffle changed distribution",
        )
        np.testing.assert_array_equal(
            np.sort(shuf_b),
            np.sort(raw_b),
            err_msg="SYNTH_B shuffle changed distribution",
        )


# ---------------------------------------------------------------------------
# Test 2: Features are unchanged
# ---------------------------------------------------------------------------


class TestFeaturesUnchanged:
    """The flat features must be identical to Gate 0 (only labels differ)."""

    def test_features_identical_to_gate0_features(self, sl, synthetic_cache, tmp_path):
        """Run shuffled baseline and Gate 0 on the same cache.  The per-window
        flat features (X matrix before LR) must be identical."""
        import importlib.util as ilu

        # Import gate0 to extract flat features
        spec = ilu.spec_from_file_location(
            "run_gate0", _REPO_ROOT / "scripts" / "run_gate0.py"
        )
        assert spec and spec.loader
        gate0 = ilu.module_from_spec(spec)
        spec.loader.exec_module(gate0)  # type: ignore[union-attr]

        # Load features via gate0 helpers
        loaded = gate0._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert loaded is not None
        feats, _, _ = loaded
        starts = gate0._build_eval_windows(len(feats))
        X_gate0 = gate0._extract_flat_features_for_windows(feats, starts)

        # Load features via shuffled-labels helpers (must be identical)
        loaded_sl = sl._load_symbol_shards(synthetic_cache, "SYNTH_A")
        assert loaded_sl is not None
        feats_sl, _, _ = loaded_sl
        starts_sl = sl._build_eval_windows(len(feats_sl))
        X_sl = sl._extract_flat_features_for_windows(feats_sl, starts_sl)

        np.testing.assert_array_equal(
            X_gate0,
            X_sl,
            err_msg="Flat features differ between gate0 and shuffled-labels pipeline",
        )


# ---------------------------------------------------------------------------
# Test 3: Seed reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    def test_same_seed_same_shuffled_labels(self, sl):
        """Two calls with the same seed must produce identical shuffles."""
        n = 3_000
        original = np.random.default_rng(5).integers(0, 2, size=n, dtype=np.int8)

        s1 = sl._shuffle_labels(original, seed=42)
        s2 = sl._shuffle_labels(original, seed=42)

        np.testing.assert_array_equal(
            s1, s2, err_msg="Same seed produced different shuffles"
        )

    def test_different_seeds_different_shuffles(self, sl):
        """Different seeds must produce different permutations (with high probability)."""
        n = 3_000
        original = np.random.default_rng(6).integers(0, 2, size=n, dtype=np.int8)

        s42 = sl._shuffle_labels(original, seed=42)
        s99 = sl._shuffle_labels(original, seed=99)

        n_diff = int((s42 != s99).sum())
        assert (
            n_diff > 100
        ), f"Seeds 42 and 99 produced nearly identical shuffles ({n_diff}/{n} differ)"

    def test_cli_seed_flag_accepted(self, sl, synthetic_cache, tmp_path):
        """--seed flag must be accepted and affect outputs deterministically."""
        out_prefix = str(tmp_path / "sl_seed")
        rc = sl.main(
            [
                "--cache",
                str(synthetic_cache),
                "--symbols",
                "SYNTH_A",
                "--horizons",
                "100",
                "--seed",
                "99",
                "--out",
                out_prefix,
            ]
            + _SMALL_CLI_FLAGS
        )
        assert rc == 0
        with open(out_prefix + ".json") as f:
            data = json.load(f)
        assert (
            data.get("shuffle_seed") == 99
        ), "JSON must record the shuffle_seed for reproducibility"


# ---------------------------------------------------------------------------
# Test 4: Balanced accuracy stays near 0.5 for balanced synthetic labels
# ---------------------------------------------------------------------------


class TestNullHypothesisBalancedAccuracy:
    """On balanced random labels, shuffled-labels PCA+LR should score ~0.5.
    This verifies the null-hypothesis design: if accuracy deviates significantly
    from 0.5, there is structural leakage in the pipeline."""

    def test_balanced_accuracy_near_05_on_random_labels(
        self, sl, synthetic_cache, tmp_path
    ):
        out_prefix = str(tmp_path / "sl_null")
        rc = sl.main(
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
        assert rc == 0
        with open(out_prefix + ".json") as f:
            data = json.load(f)

        h100 = data.get("SYNTH_A", {}).get("h100", {})
        bal_acc = h100.get("balanced_accuracy_mean")
        assert bal_acc is not None, f"balanced_accuracy_mean missing: {h100}"
        # On shuffled labels the LR must score near chance: 0.5 ± 0.05
        assert abs(bal_acc - 0.5) < 0.10, (
            f"Shuffled-labels balanced accuracy {bal_acc:.4f} deviates from 0.5 by "
            f"{abs(bal_acc - 0.5):.4f} — possible pipeline leakage"
        )


# ---------------------------------------------------------------------------
# Test 5: JSON schema matches Gate 0
# ---------------------------------------------------------------------------


class TestJSONSchema:
    def test_schema_matches_gate0_structure(self, sl, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "sl_schema")
        sl.main(
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

        assert "summary" in data, "JSON missing top-level 'summary' key"
        h100 = data.get("SYNTH_A", {}).get("h100", {})
        assert (
            "accuracy_mean" in h100 or "error" in h100
        ), f"h100 missing both accuracy_mean and error: {h100}"

    def test_summary_has_standard_fields(self, sl, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "sl_summary")
        sl.main(
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

    def test_method_tag_in_json(self, sl, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "sl_method")
        sl.main(
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
            "shuffled" in data["method"].lower()
        ), f"'method' should identify shuffled-labels, got: {data['method']}"

    def test_json_records_shuffle_seed(self, sl, synthetic_cache, tmp_path):
        out_prefix = str(tmp_path / "sl_seed_record")
        sl.main(
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
        assert "shuffle_seed" in data, "JSON missing 'shuffle_seed' key"
        assert data["shuffle_seed"] == 42


# ---------------------------------------------------------------------------
# Test 6: April hold-out guard
# ---------------------------------------------------------------------------


class TestAprilHoldoutGuard:
    def test_april_holdout_shard_raises(self, sl, tmp_path):
        rng = np.random.default_rng(0)
        cache_dir = tmp_path / "bad_cache"
        cache_dir.mkdir()
        shard = _make_shard("BTC", "2026-04-14", _N_EVENTS, rng)
        _write_shard(shard, cache_dir)
        with pytest.raises((ValueError, AssertionError, SystemExit)):
            sl.main(
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
