# tests/tape/test_build_cache.py
"""Unit tests for the Task 7 integration layer functions.

Covers:
  - compute_trade_vs_mid: clip((vwap - mid) / max(spread, 1e-8*mid), -5, 5)
  - compute_real_kyle_lambda: trade-attributed signed notional per snapshot,
    rolling 50-snapshot Cov(Δmid, cum_signed_notional)/Var(cum_signed_notional)
  - save_shard / load_shard: round-trip persistence
  - dedup_dispatch: correct dedup path based on date string
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tape.cache import (
    compute_real_kyle_lambda,
    compute_trade_vs_mid,
    load_shard,
    save_shard,
)
from tape.dedup import dedup_trades_pre_april, filter_trades_april

# ---------------------------------------------------------------------------
# compute_trade_vs_mid
# ---------------------------------------------------------------------------


class TestComputeTradeVsMid:
    def test_basic_positive_direction(self) -> None:
        """vwap above mid gives positive result."""
        vwap = np.array([100.5])
        mid = np.array([100.0])
        spread = np.array([0.2])
        result = compute_trade_vs_mid(vwap, mid, spread)
        # (100.5 - 100.0) / 0.2 = 2.5
        assert result.shape == (1,)
        assert abs(result[0] - 2.5) < 1e-6

    def test_basic_negative_direction(self) -> None:
        """vwap below mid gives negative result."""
        vwap = np.array([99.5])
        mid = np.array([100.0])
        spread = np.array([0.2])
        result = compute_trade_vs_mid(vwap, mid, spread)
        # (99.5 - 100.0) / 0.2 = -2.5
        assert abs(result[0] - (-2.5)) < 1e-6

    def test_clips_at_plus_5(self) -> None:
        """Values beyond +5 are clipped."""
        vwap = np.array([200.0])
        mid = np.array([100.0])
        spread = np.array([0.1])
        result = compute_trade_vs_mid(vwap, mid, spread)
        assert result[0] == pytest.approx(5.0)

    def test_clips_at_minus_5(self) -> None:
        """Values below -5 are clipped."""
        vwap = np.array([0.0])
        mid = np.array([100.0])
        spread = np.array([0.1])
        result = compute_trade_vs_mid(vwap, mid, spread)
        assert result[0] == pytest.approx(-5.0)

    def test_zero_spread_guarded_by_mid(self) -> None:
        """When spread == 0, denominator falls back to 1e-8 * mid (gotcha #10).

        With spread=0, mid=100, denom = max(0, 1e-8 * 100) = 1e-6.
        (101 - 100) / 1e-6 = 1e6, clips to +5.
        """
        vwap = np.array([101.0])
        mid = np.array([100.0])
        spread = np.array([0.0])
        result = compute_trade_vs_mid(vwap, mid, spread)
        assert result[0] == pytest.approx(5.0)

    def test_returns_float32(self) -> None:
        """Output dtype is float32."""
        vwap = np.array([100.0], dtype=float)
        mid = np.array([100.0], dtype=float)
        spread = np.array([0.2], dtype=float)
        result = compute_trade_vs_mid(vwap, mid, spread)
        assert result.dtype == np.float32

    def test_vectorized_multiple_events(self) -> None:
        """Works correctly across multiple events."""
        n = 10
        vwap = np.full(n, 100.5)
        mid = np.full(n, 100.0)
        spread = np.full(n, 0.2)
        result = compute_trade_vs_mid(vwap, mid, spread)
        assert result.shape == (n,)
        assert np.allclose(result, 2.5)

    def test_at_mid_is_zero(self) -> None:
        """Trade at mid = 0 trade_vs_mid."""
        vwap = np.array([50.0])
        mid = np.array([50.0])
        spread = np.array([0.5])
        result = compute_trade_vs_mid(vwap, mid, spread)
        assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_real_kyle_lambda
# ---------------------------------------------------------------------------


class TestComputeRealKyleLambda:
    """Tests for the trade-attributed Kyle's lambda per snapshot."""

    def _make_trades(
        self,
        snap_ts: np.ndarray,
        n_trades_per_interval: int = 5,
        is_buy: bool = True,
    ) -> pd.DataFrame:
        """Build a minimal trades DataFrame with one trade between each snapshot pair."""
        rows = []
        for i in range(len(snap_ts) - 1):
            t_start = snap_ts[i]
            t_end = snap_ts[i + 1]
            dt = (t_end - t_start) // (n_trades_per_interval + 1)
            side = "open_long" if is_buy else "open_short"
            for j in range(n_trades_per_interval):
                rows.append(
                    {
                        "ts_ms": t_start + dt * (j + 1),
                        "qty": 1.0,
                        "price": 100.0,
                        "side": side,
                    }
                )
        return pd.DataFrame(rows)

    def test_output_shape_matches_snap_count(self) -> None:
        """Output has one value per snapshot."""
        n_snaps = 60
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = np.ones(n_snaps) * 100.0
        trades = self._make_trades(snap_ts, n_trades_per_interval=3)
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        assert result.shape == (n_snaps,)

    def test_first_49_snapshots_are_zero(self) -> None:
        """First KYLE_LAMBDA_WINDOW-1=49 snapshots have 0 (insufficient history)."""
        from tape.constants import KYLE_LAMBDA_WINDOW

        n_snaps = 60
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = np.ones(n_snaps) * 100.0
        trades = self._make_trades(snap_ts)
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        assert np.all(
            result[: KYLE_LAMBDA_WINDOW - 1] == 0.0
        ), f"Expected first {KYLE_LAMBDA_WINDOW - 1} entries to be 0"

    def test_consistent_buy_flow_positive_lambda(self) -> None:
        """Consistent buy flow with rising mid should give positive lambda."""
        n_snaps = 60
        # Rising mid: each snapshot the mid increases
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = 100.0 + np.arange(n_snaps, dtype=float) * 0.01
        # All buys between each pair of snapshots
        trades = self._make_trades(snap_ts, n_trades_per_interval=3, is_buy=True)
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        # After window fills (index 49+), lambda should be positive
        assert (
            result[59] > 0.0
        ), f"Expected positive lambda at index 59, got {result[59]}"

    def test_consistent_sell_flow_negative_lambda(self) -> None:
        """Consistent sell flow with falling mid should give positive lambda
        (Kyle lambda is always >= 0 in the price-impact sense, but here we use
        signed notional — see sign convention: sells give negative signed_notional
        with falling prices → Cov is positive → lambda positive).
        """
        n_snaps = 60
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        # Falling mid
        snap_mid = 100.0 - np.arange(n_snaps, dtype=float) * 0.01
        trades = self._make_trades(snap_ts, n_trades_per_interval=3, is_buy=False)
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        # Sells = negative signed notional, mid falling (negative Δmid)
        # Cov(neg Δmid, neg notional) > 0 → lambda > 0
        assert (
            result[59] > 0.0
        ), f"Expected positive lambda at index 59, got {result[59]}"

    def test_zero_variance_in_signed_notional_gives_zero_not_nan(self) -> None:
        """If signed_notional is constant (zero variance), lambda = 0 not NaN."""
        n_snaps = 60
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = np.ones(n_snaps) * 100.0
        # Flat mid, flat trades
        trades = self._make_trades(snap_ts, n_trades_per_interval=3, is_buy=True)
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        assert np.all(
            np.isfinite(result)
        ), "lambda must be finite even with zero variance"

    def test_no_trades_in_interval_uses_zero(self) -> None:
        """Intervals with no trades contribute 0 to signed_notional (not skipped)."""
        n_snaps = 60
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = 100.0 + np.arange(n_snaps, dtype=float) * 0.01
        # No trades at all
        trades = pd.DataFrame(
            {
                "ts_ms": pd.Series([], dtype=np.int64),
                "qty": pd.Series([], dtype=float),
                "price": pd.Series([], dtype=float),
                "side": pd.Series([], dtype=str),
            }
        )
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        # All signed_notional = 0 → zero variance → lambda = 0, never NaN
        assert result.shape == (n_snaps,)
        assert np.all(np.isfinite(result))

    def test_is_buy_sign_convention(self) -> None:
        """open_long/close_short = +1; open_short/close_long = -1."""
        n_snaps = 60
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = 100.0 + np.arange(n_snaps, dtype=float) * 0.01

        # buy-side only
        buy_trades = pd.DataFrame(
            {
                "ts_ms": (snap_ts[:-1] + 1).tolist() * 1,
                "qty": [1.0] * (n_snaps - 1),
                "price": [100.0] * (n_snaps - 1),
                "side": ["open_long"] * (n_snaps - 1),
            }
        )
        r_buy = compute_real_kyle_lambda(snap_ts, snap_mid, buy_trades)

        # sell-side only (inverted sign)
        sell_trades = pd.DataFrame(
            {
                "ts_ms": (snap_ts[:-1] + 1).tolist() * 1,
                "qty": [1.0] * (n_snaps - 1),
                "price": [100.0] * (n_snaps - 1),
                "side": ["open_short"] * (n_snaps - 1),
            }
        )
        # Flip mid direction so sells also correlate with price moves
        snap_mid_down = 100.0 - np.arange(n_snaps, dtype=float) * 0.01
        r_sell = compute_real_kyle_lambda(snap_ts, snap_mid_down, sell_trades)

        # Both should be positive (price impact is positive for both sides
        # when signed notional matches mid direction)
        assert r_buy[59] > 0.0
        assert r_sell[59] > 0.0

    def test_output_is_finite(self) -> None:
        """All output values must be finite (no NaN or inf)."""
        n_snaps = 55
        snap_ts = np.arange(n_snaps, dtype=np.int64) * 24_000
        snap_mid = 100.0 + np.random.default_rng(42).normal(0, 0.01, n_snaps).cumsum()
        trades = self._make_trades(snap_ts, n_trades_per_interval=2)
        result = compute_real_kyle_lambda(snap_ts, snap_mid, trades)
        assert np.all(np.isfinite(result)), "lambda contains non-finite values"


# ---------------------------------------------------------------------------
# save_shard / load_shard round-trip
# ---------------------------------------------------------------------------


class TestShardRoundTrip:
    def _make_shard(self, n: int = 500) -> dict:
        date_str = "2025-11-01"
        day_id_val = int((pd.Timestamp(date_str) - pd.Timestamp("1970-01-01")).days)
        return {
            "features": np.random.randn(n, 17).astype(np.float32),
            "event_ts": np.arange(n, dtype=np.int64),
            "day_id": np.full(n, day_id_val, dtype=np.int64),
            "directions": (
                {f"h{h}": np.zeros(n, dtype=np.int8) for h in (10, 50, 100, 500)}
                | {f"mask_h{h}": np.ones(n, dtype=bool) for h in (10, 50, 100, 500)}
            ),
            "wyckoff": {
                k: np.zeros(n, dtype=np.int8)
                for k in ("stress", "informed_flow", "climax", "spring", "absorption")
            },
            "symbol": "BTC",
            "date": date_str,
            "schema_version": 1,
        }

    def test_round_trip_preserves_features_shape(self, tmp_path: Path) -> None:
        n = 500
        shard = self._make_shard(n)
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert payload["features"].shape == (n, 17)

    def test_round_trip_schema_version(self, tmp_path: Path) -> None:
        shard = self._make_shard()
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert int(payload["schema_version"]) == 1

    def test_round_trip_direction_keys(self, tmp_path: Path) -> None:
        """Direction keys are prefixed with 'dir_' in the .npz."""
        shard = self._make_shard()
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert "dir_h100" in payload
        assert "dir_mask_h100" in payload

    def test_round_trip_wyckoff_keys(self, tmp_path: Path) -> None:
        """Wyckoff keys are prefixed with 'wy_' in the .npz."""
        shard = self._make_shard()
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert "wy_stress" in payload
        assert "wy_absorption" in payload

    def test_round_trip_event_ts(self, tmp_path: Path) -> None:
        n = 100
        shard = self._make_shard(n)
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert payload["event_ts"].shape == (n,)
        assert payload["event_ts"].dtype == np.int64

    def test_file_name_includes_symbol_and_date(self, tmp_path: Path) -> None:
        shard = self._make_shard()
        path = save_shard(shard, tmp_path)
        assert "BTC" in path.name
        assert "2025-11-01" in path.name

    def test_features_values_preserved(self, tmp_path: Path) -> None:
        """Saved float32 values survive round-trip within float32 tolerance."""
        rng = np.random.default_rng(0)
        n = 100
        shard = self._make_shard(n)
        shard["features"] = rng.random((n, 17), dtype=np.float32)
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        np.testing.assert_array_equal(payload["features"], shard["features"])


# ---------------------------------------------------------------------------
# dedup_dispatch tests
# ---------------------------------------------------------------------------


class TestDedupDispatch:
    """Verify correct dedup path is taken for pre-April vs April+ data."""

    def _make_pre_april_trades(self) -> pd.DataFrame:
        """Two counterparty rows per fill: same ts_ms/qty/price, differ on side."""
        return pd.DataFrame(
            {
                "ts_ms": [1000, 1000, 2000, 2000],
                "qty": [1.0, 1.0, 2.0, 2.0],
                "price": [100.0, 100.0, 101.0, 101.0],
                "side": ["open_long", "open_short", "open_long", "open_short"],
            }
        )

    def test_pre_april_dedup_collapses_counterparty_pairs(self) -> None:
        """Pre-April: two rows with same (ts_ms, qty, price) → collapsed to one."""
        df = self._make_pre_april_trades()
        deduped = dedup_trades_pre_april(df)
        assert len(deduped) == 2, f"Expected 2 events, got {len(deduped)}"

    def test_pre_april_dedup_does_not_use_side(self) -> None:
        """Dedup key must NOT include side — both counterparty rows must collapse."""
        df = pd.DataFrame(
            {
                "ts_ms": [1000, 1000],
                "qty": [1.0, 1.0],
                "price": [100.0, 100.0],
                "side": ["open_long", "open_short"],
            }
        )
        result = dedup_trades_pre_april(df)
        assert len(result) == 1

    def test_april_filter_keeps_taker_only(self) -> None:
        """April+: only fulfill_taker rows are kept."""
        df = pd.DataFrame(
            {
                "ts_ms": [1000, 1000, 2000],
                "qty": [1.0, 1.0, 2.0],
                "price": [100.0, 100.0, 101.0],
                "side": ["open_long", "open_short", "open_long"],
                "event_type": ["fulfill_taker", "fulfill_maker", "fulfill_taker"],
            }
        )
        result = filter_trades_april(df)
        assert len(result) == 2
        assert (result["event_type"] == "fulfill_taker").all()

    def test_april_filter_requires_event_type_column(self) -> None:
        """April filter raises ValueError if event_type column is missing."""
        df = pd.DataFrame(
            {
                "ts_ms": [1000],
                "qty": [1.0],
                "price": [100.0],
                "side": ["open_long"],
            }
        )
        with pytest.raises(ValueError, match="event_type"):
            filter_trades_april(df)


# ---------------------------------------------------------------------------
# Fix 1: hold-out guard must raise ValueError
# ---------------------------------------------------------------------------


class TestHoldOutGuardRaises:
    def test_build_symbol_day_raises_on_heldout_date(self, tmp_path: Path) -> None:
        """build_symbol_day must raise ValueError for dates >= APRIL_HELDOUT_START."""
        from tape.cache import build_symbol_day

        with pytest.raises(ValueError, match="hold-out"):
            build_symbol_day(
                "BTC",
                "2026-04-14",
                raw_root=tmp_path,
                out_root=tmp_path,
            )

    def test_build_symbol_day_raises_for_later_heldout_date(
        self, tmp_path: Path
    ) -> None:
        """Dates well past the hold-out boundary also raise."""
        from tape.cache import build_symbol_day

        with pytest.raises(ValueError, match="hold-out"):
            build_symbol_day(
                "ETH",
                "2026-05-01",
                raw_root=tmp_path,
                out_root=tmp_path,
            )

    def test_build_symbol_day_accepts_last_pre_heldout_date(
        self, tmp_path: Path
    ) -> None:
        """2026-04-13 (one day before hold-out) must NOT raise — returns None (no data)."""
        from tape.cache import build_symbol_day

        # No parquet files on disk → returns None, does not raise
        result = build_symbol_day(
            "BTC",
            "2026-04-13",
            raw_root=tmp_path,
            out_root=tmp_path,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Fix 2: delta_imbalance_L1 prior-day pre-warming
# ---------------------------------------------------------------------------


def _make_ob_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Minimal 10-level OB DataFrame."""
    rng = np.random.default_rng(seed)
    ts = (np.arange(n, dtype=np.int64) * 24_000) + 1_700_000_000_000
    mid = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    spread = np.abs(rng.normal(0.05, 0.01, size=n)) + 1e-4
    bid = mid - spread / 2
    ask = mid + spread / 2
    data: dict[str, object] = {"ts_ms": ts}
    for lvl in range(1, 11):
        data[f"bid{lvl}_price"] = bid - (lvl - 1) * 0.01
        data[f"ask{lvl}_price"] = ask + (lvl - 1) * 0.01
        data[f"bid{lvl}_qty"] = rng.gamma(2.0, 5.0, size=n)
        data[f"ask{lvl}_qty"] = rng.gamma(2.0, 5.0, size=n)
    return pd.DataFrame(data)


class TestDeltaImbalancePreWarm:
    def test_none_prior_gives_zero_first_delta(self) -> None:
        """When prior_imbalance_l1=None, first snapshot delta_imbalance_L1 == 0."""
        from tape.features_ob import compute_snapshot_features

        ob = _make_ob_df(20)
        snap = compute_snapshot_features(ob, prior_imbalance_l1=None)
        assert snap["delta_imbalance_L1"].iloc[0] == pytest.approx(0.0)

    def test_prior_provided_first_delta_reflects_it(self) -> None:
        """When prior_imbalance_l1 is provided, first delta uses it (not 0)."""
        from tape.features_ob import compute_snapshot_features

        ob = _make_ob_df(20, seed=42)
        snap_no_prior = compute_snapshot_features(ob, prior_imbalance_l1=None)
        first_imb = float(snap_no_prior["imbalance_L1"].iloc[0])

        # Use a known prior that differs from first_imb
        prior = first_imb + 0.15
        snap_with_prior = compute_snapshot_features(ob, prior_imbalance_l1=prior)
        expected_delta = first_imb - prior
        assert snap_with_prior["delta_imbalance_L1"].iloc[0] == pytest.approx(
            expected_delta, abs=1e-7
        )

    def test_prior_does_not_affect_subsequent_deltas(self) -> None:
        """Deltas at index >= 1 are unaffected by prior_imbalance_l1."""
        from tape.features_ob import compute_snapshot_features

        ob = _make_ob_df(20, seed=7)
        snap_no = compute_snapshot_features(ob, prior_imbalance_l1=None)
        snap_yes = compute_snapshot_features(ob, prior_imbalance_l1=99.0)
        np.testing.assert_array_equal(
            snap_no["delta_imbalance_L1"].to_numpy()[1:],
            snap_yes["delta_imbalance_L1"].to_numpy()[1:],
        )

    def test_positional_call_still_works(self) -> None:
        """compute_snapshot_features(ob) without keyword arg (old callers) still works."""
        from tape.features_ob import compute_snapshot_features

        ob = _make_ob_df(10)
        snap = compute_snapshot_features(ob)
        assert "delta_imbalance_L1" in snap.columns


# ---------------------------------------------------------------------------
# Fix 3: day_id in .npz shard
# ---------------------------------------------------------------------------


class TestDayIdInShard:
    def _make_shard_with_day_id(self, n: int = 100, date: str = "2025-11-01") -> dict:
        import pandas as pd

        day_id = int((pd.Timestamp(date) - pd.Timestamp("1970-01-01")).days)
        return {
            "features": np.random.default_rng(0).random((n, 17)).astype(np.float32),
            "event_ts": np.arange(n, dtype=np.int64),
            "day_id": np.full(n, day_id, dtype=np.int64),
            "directions": (
                {f"h{h}": np.zeros(n, dtype=np.int8) for h in (10, 50, 100, 500)}
                | {f"mask_h{h}": np.ones(n, dtype=bool) for h in (10, 50, 100, 500)}
            ),
            "wyckoff": {
                k: np.zeros(n, dtype=np.int8)
                for k in ("stress", "informed_flow", "climax", "spring", "absorption")
            },
            "symbol": "BTC",
            "date": date,
            "schema_version": 1,
        }

    def test_day_id_present_in_saved_shard(self, tmp_path: Path) -> None:
        """day_id key must be present in the .npz after save_shard."""
        shard = self._make_shard_with_day_id()
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert "day_id" in payload, "day_id missing from saved shard"

    def test_day_id_correct_shape(self, tmp_path: Path) -> None:
        """day_id shape matches n_events."""
        n = 123
        shard = self._make_shard_with_day_id(n=n)
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert payload["day_id"].shape == (n,)

    def test_day_id_correct_dtype(self, tmp_path: Path) -> None:
        """day_id dtype is int64."""
        shard = self._make_shard_with_day_id()
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert payload["day_id"].dtype == np.int64

    def test_day_id_correct_value(self, tmp_path: Path) -> None:
        """day_id value equals (date - 1970-01-01).days."""
        date_str = "2025-11-01"
        expected = int((pd.Timestamp(date_str) - pd.Timestamp("1970-01-01")).days)
        shard = self._make_shard_with_day_id(date=date_str)
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert int(payload["day_id"][0]) == expected

    def test_all_day_id_values_identical(self, tmp_path: Path) -> None:
        """All events in a shard must share the same day_id (np.full)."""
        n = 50
        shard = self._make_shard_with_day_id(n=n)
        path = save_shard(shard, tmp_path)
        payload = load_shard(path)
        assert len(set(payload["day_id"].tolist())) == 1
