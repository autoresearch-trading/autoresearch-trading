# tests/tape/test_features_ob.py
"""TDD tests for tape/features_ob.py — 8 orderbook features.

Written BEFORE the implementation (TDD red phase).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tape.constants import OB_FEATURES


def _fake_ob(n: int, seed: int = 0) -> pd.DataFrame:
    """Minimal well-formed 10-level OB DataFrame for testing."""
    rng = np.random.default_rng(seed)
    ts = (np.arange(n, dtype=np.int64) * 24_000) + 1_700_000_000_000
    mid = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    spread = np.abs(rng.normal(0.05, 0.01, size=n)) + 1e-4  # keep spread > 0
    bid = mid - spread / 2
    ask = mid + spread / 2
    data: dict[str, object] = {"ts_ms": ts}
    for lvl in range(1, 11):
        data[f"bid{lvl}_price"] = bid - (lvl - 1) * 0.01
        data[f"ask{lvl}_price"] = ask + (lvl - 1) * 0.01
        data[f"bid{lvl}_qty"] = rng.gamma(2.0, 5.0, size=n)
        data[f"ask{lvl}_qty"] = rng.gamma(2.0, 5.0, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# compute_snapshot_features
# ---------------------------------------------------------------------------


class TestComputeSnapshotFeatures:

    def test_returns_all_required_columns(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(100)
        snap = compute_snapshot_features(ob)
        required = {
            "log_spread",
            "imbalance_L1",
            "imbalance_L5",
            "depth_ratio",
            "delta_imbalance_L1",
            "kyle_lambda",
            "cum_ofi_5",
            "mid",
            "spread",
            "ts_ms",
        }
        assert required.issubset(set(snap.columns))

    def test_output_length_matches_input(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(75)
        snap = compute_snapshot_features(ob)
        assert len(snap) == 75

    def test_all_outputs_are_finite(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(200)
        snap = compute_snapshot_features(ob)
        for col in (
            "log_spread",
            "imbalance_L1",
            "imbalance_L5",
            "depth_ratio",
            "delta_imbalance_L1",
            "kyle_lambda",
            "cum_ofi_5",
        ):
            assert np.isfinite(
                snap[col].to_numpy()
            ).all(), f"{col} contains non-finite values"

    # --- log_spread ---

    def test_log_spread_is_negative(self) -> None:
        """spread/mid < 1 for normal markets, so log_spread < 0."""
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        assert (snap["log_spread"] < 0).all()

    def test_log_spread_monotone_in_spread(self) -> None:
        """Wider spread → larger (less negative) log_spread."""
        from tape.features_ob import compute_snapshot_features

        n = 10
        base = _fake_ob(n, seed=1)
        wide = base.copy()
        for lvl in range(1, 11):
            # Push asks further out, bids further in → wider spread
            wide[f"ask{lvl}_price"] = base[f"ask{lvl}_price"] * 1.01
            wide[f"bid{lvl}_price"] = base[f"bid{lvl}_price"] * 0.99
        snap_base = compute_snapshot_features(base)
        snap_wide = compute_snapshot_features(wide)
        assert (snap_wide["log_spread"] > snap_base["log_spread"]).all()

    # --- imbalance_L1 ---

    def test_imbalance_L1_in_range(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(100)
        snap = compute_snapshot_features(ob)
        assert (snap["imbalance_L1"] >= -1.0).all()
        assert (snap["imbalance_L1"] <= 1.0).all()

    def test_imbalance_L1_positive_when_bid_heavy(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(10)
        # Make bid notional >> ask notional
        ob["bid1_qty"] = 1000.0
        ob["ask1_qty"] = 1.0
        snap = compute_snapshot_features(ob)
        assert (snap["imbalance_L1"] > 0).all()

    def test_imbalance_L1_negative_when_ask_heavy(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(10)
        ob["bid1_qty"] = 1.0
        ob["ask1_qty"] = 1000.0
        snap = compute_snapshot_features(ob)
        assert (snap["imbalance_L1"] < 0).all()

    # --- depth_ratio ---

    def test_depth_ratio_finite_on_one_sided_asks_only(self) -> None:
        """All bids depleted (flash bid-side depletion) → very negative but finite."""
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(50)
        for lvl in range(1, 11):
            ob[f"bid{lvl}_qty"] = 0.0
        snap = compute_snapshot_features(ob)
        assert np.isfinite(snap["depth_ratio"].to_numpy()).all()
        assert (snap["depth_ratio"] < 0).all()

    def test_depth_ratio_finite_on_one_sided_bids_only(self) -> None:
        """All asks depleted → very positive but finite."""
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(50)
        for lvl in range(1, 11):
            ob[f"ask{lvl}_qty"] = 0.0
        snap = compute_snapshot_features(ob)
        assert np.isfinite(snap["depth_ratio"].to_numpy()).all()
        assert (snap["depth_ratio"] > 0).all()

    def test_depth_ratio_uses_full_10_levels(self) -> None:
        """Removing levels 6-10 should change depth_ratio."""
        from tape.features_ob import compute_snapshot_features

        ob_full = _fake_ob(20, seed=42)
        ob_l5 = ob_full.copy()
        for lvl in range(6, 11):
            ob_l5[f"bid{lvl}_qty"] = 0.0
            ob_l5[f"ask{lvl}_qty"] = 0.0
        snap_full = compute_snapshot_features(ob_full)
        snap_l5 = compute_snapshot_features(ob_l5)
        assert not np.allclose(snap_full["depth_ratio"], snap_l5["depth_ratio"])

    # --- delta_imbalance_L1 ---

    def test_delta_imbalance_L1_first_snapshot_is_zero(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        assert snap["delta_imbalance_L1"].iloc[0] == pytest.approx(0.0)

    def test_delta_imbalance_L1_is_diff_of_imbalance(self) -> None:
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        expected = np.concatenate([[0.0], np.diff(snap["imbalance_L1"].to_numpy())])
        np.testing.assert_allclose(
            snap["delta_imbalance_L1"].to_numpy(), expected, atol=1e-12
        )

    # --- kyle_lambda ---

    def test_kyle_lambda_placeholder_zero_before_full_window(self) -> None:
        """Validates placeholder behavior only — real Kyle's λ implemented in Task 7."""
        from tape.constants import KYLE_LAMBDA_WINDOW
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(200)
        snap = compute_snapshot_features(ob)
        # First KYLE_LAMBDA_WINDOW - 1 rows have insufficient history (< 50 snapshots)
        # Row KYLE_LAMBDA_WINDOW - 1 (index 49) is the first row with a full window
        assert (snap["kyle_lambda"].iloc[: KYLE_LAMBDA_WINDOW - 1] == 0.0).all()

    def test_kyle_lambda_placeholder_nonzero_after_full_window(self) -> None:
        """Validates placeholder behavior only — real Kyle's λ implemented in Task 7."""
        from tape.constants import KYLE_LAMBDA_WINDOW
        from tape.features_ob import compute_snapshot_features

        ob = _fake_ob(200)
        snap = compute_snapshot_features(ob)
        # Index KYLE_LAMBDA_WINDOW - 1 (=49) has exactly 50 snapshots of history
        # and must be nonzero (off-by-one fix: was previously left as 0)
        assert snap["kyle_lambda"].iloc[KYLE_LAMBDA_WINDOW - 1] != 0.0
        # The tail from that index onward should have non-zero values
        tail = snap["kyle_lambda"].iloc[KYLE_LAMBDA_WINDOW - 1 :]
        assert (tail != 0.0).any()

    def test_kyle_lambda_same_value_within_snapshot(self) -> None:
        """Gotcha #13: per-snapshot, not per-event. When two events align to
        the same OB snapshot, they must get the same kyle_lambda value."""
        from tape.constants import KYLE_LAMBDA_WINDOW
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(200)
        snap = compute_snapshot_features(ob)
        ob_ts = snap["ts_ms"].to_numpy()
        # Two events bracketing the same snapshot (between idx 60 and 61)
        t60 = int(ob_ts[60])
        event_ts = np.array([t60, t60 + 100, t60 + 200], dtype=np.int64)
        aligned = align_ob_features_to_events(snap, event_ts)
        # All three events align to snapshot 60 → same kyle_lambda
        assert aligned["kyle_lambda"].iloc[0] == pytest.approx(
            aligned["kyle_lambda"].iloc[1]
        )
        assert aligned["kyle_lambda"].iloc[1] == pytest.approx(
            aligned["kyle_lambda"].iloc[2]
        )

    # --- cum_ofi_5 ---

    def test_cum_ofi_5_piecewise_when_bid_price_rises(self) -> None:
        """Gotcha #15: when bid price rises, OFI_bid = +new_bid_qty (not qty delta).
        This is the canonical correctness check for piecewise Cont.

        Critical scenario: bid price rises while ask stays FLAT.
        - Naive delta-qty: bid_q[1] - bid_q[0] = 0 → OFI_bid = 0 (WRONG)
        - Piecewise Cont: bid price rose → OFI_bid = +bid_q[1]*bid_p[1] > 0 (CORRECT)
        """
        from tape.features_ob import compute_snapshot_features

        ob = pd.DataFrame(
            {
                "ts_ms": np.array([0, 1, 2], dtype=np.int64),
                "bid1_price": [100.0, 101.0, 101.0],  # bid rose at t=1, flat at t=2
                "bid1_qty": [
                    10.0,
                    10.0,
                    10.0,
                ],  # same qty — naive delta = 0, piecewise = +10*101
                "ask1_price": [
                    101.0,
                    101.0,
                    101.0,
                ],  # ask stays FLAT (key: not symmetric)
                "ask1_qty": [10.0, 10.0, 10.0],
            }
        )
        # Fill levels 2-10 with dummy data
        for lvl in range(2, 11):
            ob[f"bid{lvl}_price"] = ob["bid1_price"] - (lvl - 1) * 0.1
            ob[f"ask{lvl}_price"] = ob["ask1_price"] + (lvl - 1) * 0.1
            ob[f"bid{lvl}_qty"] = 1.0
            ob[f"ask{lvl}_qty"] = 1.0
        snap = compute_snapshot_features(ob)
        # At idx=1: bid rose, ask flat, same qtys
        # Piecewise: OFI_bid = +10*101 = +1010, OFI_ask = -(10-10)*101 = 0 → net +1010
        # Naive: OFI_bid = (10-10)*101 = 0 → net 0 (would fail this assertion)
        # cum_ofi_5[1] = OFI[0] + OFI[1] = 0 + 1010 > 0
        assert snap["cum_ofi_5"].iloc[1] > 0

    def test_cum_ofi_5_piecewise_wrong_sign_if_naive(self) -> None:
        """Verify that the scenario where naive gives 0 but piecewise gives non-zero."""
        from tape.features_ob import compute_snapshot_features

        # Same bid qty before and after a price rise: naive delta-qty = 0
        # Piecewise: bid price rose → OFI_bid = +new_bid_qty (large positive)
        ob = pd.DataFrame(
            {
                "ts_ms": np.array([0, 1], dtype=np.int64),
                "bid1_price": [100.0, 102.0],  # rose by 2
                "bid1_qty": [5.0, 5.0],  # unchanged → naive Δ = 0
                "ask1_price": [101.0, 103.0],  # rose symmetrically
                "ask1_qty": [5.0, 5.0],
            }
        )
        for lvl in range(2, 11):
            ob[f"bid{lvl}_price"] = ob["bid1_price"] - (lvl - 1) * 0.1
            ob[f"ask{lvl}_price"] = ob["ask1_price"] + (lvl - 1) * 0.1
            ob[f"bid{lvl}_qty"] = 1.0
            ob[f"ask{lvl}_qty"] = 1.0
        snap = compute_snapshot_features(ob)
        # Piecewise: bid rose → OFI_bid = +5*102 = +510
        #            ask rose → OFI_ask = -5*101 = -505 (sign-inverted: ask_up means OFI_ask contribution is negative)
        # Net OFI[1] > 0 (buy pressure)
        # cum_ofi_5[1] > 0
        assert snap["cum_ofi_5"].iloc[1] > 0


# ---------------------------------------------------------------------------
# align_ob_features_to_events
# ---------------------------------------------------------------------------


class TestAlignObFeaturesToEvents:

    def test_output_length_matches_event_count(self) -> None:
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(100)
        snap = compute_snapshot_features(ob)
        event_ts = snap["ts_ms"].to_numpy()[:10]
        out = align_ob_features_to_events(snap, event_ts)
        assert len(out) == 10

    def test_event_before_first_snapshot_is_nan(self) -> None:
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        first_ts = snap["ts_ms"].iloc[0]
        event_ts = np.array([first_ts - 1], dtype=np.int64)
        out = align_ob_features_to_events(snap, event_ts)
        assert np.isnan(out["log_spread"].iloc[0])

    def test_event_exactly_at_first_snapshot_gets_features(self) -> None:
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        event_ts = np.array([snap["ts_ms"].iloc[0]], dtype=np.int64)
        out = align_ob_features_to_events(snap, event_ts)
        assert not np.isnan(out["log_spread"].iloc[0])

    def test_event_between_snapshots_uses_prior(self) -> None:
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(100)
        snap = compute_snapshot_features(ob)
        ob_ts = snap["ts_ms"].to_numpy()
        # Event between snapshot 10 and 11
        between_ts = int(ob_ts[10]) + 100
        event_ts = np.array([between_ts], dtype=np.int64)
        out = align_ob_features_to_events(snap, event_ts)
        assert out["log_spread"].iloc[0] == pytest.approx(snap["log_spread"].iloc[10])

    def test_event_after_last_snapshot_uses_last(self) -> None:
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        last_ts = snap["ts_ms"].iloc[-1]
        event_ts = np.array([last_ts + 1_000_000], dtype=np.int64)
        out = align_ob_features_to_events(snap, event_ts)
        assert not np.isnan(out["log_spread"].iloc[0])
        assert out["log_spread"].iloc[0] == pytest.approx(snap["log_spread"].iloc[-1])

    def test_output_columns_include_ob_features_minus_trade_vs_mid(self) -> None:
        """align_ob_features_to_events returns the 7 pure-OB features + mid + spread.
        trade_vs_mid is deferred to the integration layer (Task 7)."""
        from tape.features_ob import (
            align_ob_features_to_events,
            compute_snapshot_features,
        )

        ob = _fake_ob(50)
        snap = compute_snapshot_features(ob)
        event_ts = snap["ts_ms"].to_numpy()[:5]
        out = align_ob_features_to_events(snap, event_ts)
        for col in (
            "log_spread",
            "imbalance_L1",
            "imbalance_L5",
            "depth_ratio",
            "delta_imbalance_L1",
            "kyle_lambda",
            "cum_ofi_5",
        ):
            assert col in out.columns, f"Missing column: {col}"
        # mid and spread are aux columns needed by the integration layer
        assert "mid" in out.columns
        assert "spread" in out.columns


# ---------------------------------------------------------------------------
# OB_FEATURES constant check
# ---------------------------------------------------------------------------


def test_ob_features_constant_has_correct_names() -> None:
    """Regression guard: OB_FEATURES must be exactly the 8 spec-defined names."""
    expected = (
        "log_spread",
        "imbalance_L1",
        "imbalance_L5",
        "depth_ratio",
        "trade_vs_mid",
        "delta_imbalance_L1",
        "kyle_lambda",
        "cum_ofi_5",
    )
    assert OB_FEATURES == expected
