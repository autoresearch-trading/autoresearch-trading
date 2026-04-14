# tests/tape/test_features_trade.py
import numpy as np
import pandas as pd
import pytest

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
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == len(ev)


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


def test_prev_seq_time_span_no_lookahead():
    # Gotcha #8: prev_seq_time_span is the PRIOR 200-event window's span, not
    # the current one's (which would be a hard lookahead).
    ev = _fake_events(500)
    out = compute_trade_features(ev, spread=np.full(500, 0.1), mid=np.full(500, 100.0))
    # First 200 events must have prev_seq_time_span = 0 (no prior window yet).
    assert (out["prev_seq_time_span"].iloc[:200] == 0.0).all()
    # From event 200 onward, prev_seq_time_span equals log(last_ts[0:200] - first_ts[0:200] + 1) for 200:400
    span = float(ev["ts_ms"].iloc[199] - ev["ts_ms"].iloc[0])
    expected = float(np.log(span + 1.0))
    np.testing.assert_allclose(
        out["prev_seq_time_span"].iloc[200:400], expected, rtol=0, atol=0
    )
