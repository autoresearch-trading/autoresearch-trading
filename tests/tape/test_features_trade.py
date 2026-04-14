# tests/tape/test_features_trade.py
import numpy as np
import pandas as pd
import pytest

from tape.constants import WINDOW_LEN
from tape.features_trade import compute_trade_features


def _fake_events(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts_ms = (
        np.cumsum(rng.integers(100, 5_000, size=n)).astype(np.int64) + 1_700_000_000_000
    )
    vwap = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    total_qty = rng.gamma(2.0, 1.0, size=n)
    return pd.DataFrame(
        {
            "ts_ms": ts_ms,
            "vwap": vwap,
            "total_qty": total_qty,
            "is_open_frac": rng.uniform(size=n),
            "n_fills": rng.integers(1, 5, size=n).astype(np.int64),
            "book_walk_abs": rng.uniform(0, 0.5, size=n),
            "first_ts": ts_ms,
            "last_ts": ts_ms,
        }
    )


def test_compute_trade_features_returns_9_columns():
    ev = _fake_events(2_000)
    out = compute_trade_features(
        ev, spread=np.full(2_000, 0.1), mid=np.full(2_000, 100.0)
    )
    expected_cols = {
        "log_return",
        "log_total_qty",
        "is_open",
        "time_delta",
        "num_fills",
        "book_walk",
        "effort_vs_result",
        "climax_score",
        "prev_seq_time_span",
    }
    # Exact equality — no extra columns allowed
    assert set(out.columns) == expected_cols
    assert len(out) == len(ev)
    # Global finiteness check
    assert np.isfinite(out.values).all(), "Output contains NaN or inf"


def test_log_return_first_event_is_zero():
    ev = _fake_events(10)
    out = compute_trade_features(ev, spread=np.full(10, 0.1), mid=np.full(10, 100.0))
    assert out["log_return"].iloc[0] == 0.0


def test_effort_vs_result_is_clipped_minus5_to_5():
    # Gotcha #5
    ev = _fake_events(1_500)
    ev.loc[5, "total_qty"] = 1e20  # huge qty → large log_total_qty
    # tiny |return| via nearly identical vwap will push EVR toward +5
    out = compute_trade_features(
        ev, spread=np.full(1_500, 0.1), mid=np.full(1_500, 100.0)
    )
    assert (out["effort_vs_result"] >= -5.0).all()
    assert (out["effort_vs_result"] <= 5.0).all()


def test_climax_score_is_continuous_nonnegative():
    ev = _fake_events(2_000, seed=1)
    out = compute_trade_features(
        ev, spread=np.full(2_000, 0.1), mid=np.full(2_000, 100.0)
    )
    assert (out["climax_score"] >= 0.0).all()
    assert (out["climax_score"] <= 5.0).all()


def test_book_walk_zero_spread_guard():
    # Gotcha #10: spread=0 must NOT produce inf.
    ev = _fake_events(100)
    spread = np.zeros(100)
    mid = np.full(100, 100.0)
    out = compute_trade_features(ev, spread=spread, mid=mid)
    assert np.isfinite(out["book_walk"]).all()


def test_prev_seq_time_span_sliding_window():
    """Sliding-window semantics for prev_seq_time_span (Fix 1).

    For event i >= WINDOW_LEN:
        prev_seq_time_span[i] = log(ts_ms[i-1] - ts_ms[i-WINDOW_LEN] + 1)

    Key assertions:
    - First WINDOW_LEN events are exactly 0.0 (no prior window).
    - Event WINDOW_LEN value equals log(ts_ms[WINDOW_LEN-1] - ts_ms[0] + 1).
    - Each subsequent event uses its own shifted 200-event window (not a block).
    - Strict causality: event i only sees ts_ms[i-WINDOW_LEN .. i-1].
    """
    n = 500
    ev = _fake_events(n)
    out = compute_trade_features(ev, spread=np.full(n, 0.1), mid=np.full(n, 100.0))
    ts = ev["ts_ms"].to_numpy(dtype=np.int64)

    # First WINDOW_LEN events must be 0.0
    assert (out["prev_seq_time_span"].iloc[:WINDOW_LEN] == 0.0).all()

    # Event exactly at WINDOW_LEN: sees span of events [0, WINDOW_LEN-1]
    expected_at_200 = float(np.log(ts[WINDOW_LEN - 1] - ts[0] + 1))
    np.testing.assert_allclose(
        out["prev_seq_time_span"].iloc[WINDOW_LEN],
        expected_at_200,
        rtol=1e-10,
        atol=0,
    )

    # Spot-check several later events: each must reference its own window
    for i in [201, 250, 300, 350, 400, 499]:
        expected_i = float(np.log(ts[i - 1] - ts[i - WINDOW_LEN] + 1))
        np.testing.assert_allclose(
            out["prev_seq_time_span"].iloc[i],
            expected_i,
            rtol=1e-10,
            atol=0,
            err_msg=f"Mismatch at event i={i}",
        )

    # Events 201-399 must NOT all share the same value (old block behaviour)
    vals_201_399 = out["prev_seq_time_span"].iloc[201:400].to_numpy()
    assert (
        vals_201_399.std() > 0.0
    ), "prev_seq_time_span is a step-function — sliding window not implemented"


def test_no_day_start_saturation():
    """Fix 2: cold-start sigma floor + event-0 effort_vs_result guard.

    Two distinct cold-start problems:

    1. climax_score: with min_periods=10 and fillna(1e-10), z = qty/1e-10 → 5.0
       for events 1-9.  Fix: min_periods=1 + sigma floor (_SIGMA_FLOOR=0.1).
       The first 20 events must not saturate.

    2. effort_vs_result event 0: log_return[0] = 0 → log(0 + 1e-6) = -13.8 →
       evr = log_total_qty - (-13.8) → clips to +5.  Fix: zero out event 0.
       Events 1+ may legitimately clip to ±5 with tiny real price moves (the
       formula log(qty/med) - log(|ret|+1e-6) is designed to be near the clip
       boundary for small-return, normal-volume events — that is correct
       behaviour, not a bug).
    """
    ev = _fake_events(2_000)
    out = compute_trade_features(
        ev, spread=np.full(2_000, 0.1), mid=np.full(2_000, 100.0)
    )
    # climax_score: must not saturate for the first 20 events (cold-start fix)
    assert (
        out["climax_score"].iloc[:20] < 4.5
    ).all(), "climax_score saturates to 5.0 at day start — cold-start spike present"
    # effort_vs_result event 0: log_return[0]=0 is undefined — must be zeroed
    assert (
        out["effort_vs_result"].iloc[0] == 0.0
    ), "effort_vs_result[0] must be 0.0 (log_return[0] is undefined)"
    # Global finiteness
    assert np.isfinite(out.values).all(), "Output contains NaN or inf"
