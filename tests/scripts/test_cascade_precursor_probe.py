# tests/scripts/test_cascade_precursor_probe.py
"""Smoke tests for cascade_precursor_probe — synthetic label, lift, and headroom math."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def test_rolling_p99_cutoff_basic_shape():
    """Rolling 5000-window 99th-percentile cutoff has correct shape and is causal."""
    from scripts.cascade_precursor_probe import _rolling_quantile_causal

    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    out = _rolling_quantile_causal(x, window=50, q=0.99, min_periods=20)
    assert out.shape == x.shape
    # First (min_periods - 1) entries are NaN (no rolling cutoff yet)
    assert np.isnan(out[:19]).all()
    # Once we have ≥min_periods entries, cutoff is finite
    assert np.isfinite(out[20:]).all()
    # Causal: cutoff at idx i must NOT depend on x[i+1:] — verify by mutation
    x_alt = x.copy()
    x_alt[100:] = -100.0
    out_alt = _rolling_quantile_causal(x_alt, window=50, q=0.99, min_periods=20)
    assert np.allclose(out[:100], out_alt[:100], equal_nan=True)


def test_synthetic_cascade_label_exclusive_above_p99():
    """Synthetic label fires when |fwd_ret| > rolling cutoff (first few extremes
    after warmup; the rolling cutoff then absorbs them)."""
    from scripts.cascade_precursor_probe import _synthetic_cascade_label

    rng = np.random.default_rng(1)
    fwd_ret = rng.normal(scale=0.001, size=2000)
    # Inject a SINGLE extreme after warmup
    fwd_ret[1500] = 1.0  # one huge move
    label = _synthetic_cascade_label(
        fwd_ret, rolling_window=500, q=0.99, min_periods=200
    )
    # First 199 windows have no rolling cutoff → label = 0 (we don't fire on NaN)
    assert (label[:199] == 0).all()
    # The single extreme is far above the rolling 99th-pct of N(0, 1e-6) → fires.
    assert int(label[1500]) == 1
    # Base rate is ~1% by construction (99th-pct cutoff)
    base_rate = float(label[200:].mean())
    assert 0.001 < base_rate < 0.05


def test_lift_and_precision_at_top_decile_basic():
    from scripts.cascade_precursor_probe import _precision_recall_at_top_decile

    # 1000 windows, 50 cascades, perfectly ranked: top 100 contain all 50
    n = 1000
    n_pos = 50
    proba = np.linspace(0, 1, n)  # higher → more likely
    labels = np.zeros(n, dtype=int)
    labels[-n_pos:] = 1  # cascades are the top n_pos
    p, r, lift = _precision_recall_at_top_decile(proba, labels)
    # Top decile = top 100 windows. All 50 cascades are in there.
    # precision = 50/100 = 0.5, recall = 50/50 = 1.0, base = 0.05, lift = 10
    assert math.isclose(p, 0.5, abs_tol=1e-9)
    assert math.isclose(r, 1.0, abs_tol=1e-9)
    assert math.isclose(lift, 10.0, abs_tol=1e-9)


def test_lift_returns_one_for_random_predictions():
    from scripts.cascade_precursor_probe import _precision_recall_at_top_decile

    rng = np.random.default_rng(0)
    n = 5000
    proba = rng.random(n)
    labels = (rng.random(n) < 0.05).astype(int)  # 5% base rate
    p, r, lift = _precision_recall_at_top_decile(proba, labels)
    # Random ranker: lift ~= 1.0, expect within 0.5 in expectation
    assert 0.5 < lift < 1.5


def test_headroom_formula_components():
    """headroom = lift × E[|fwd|=cascade] × bps - cost_round_trip."""
    from scripts.cascade_precursor_probe import _headroom_top_decile_bps

    # lift=4, mean_edge_in_cascade_bps = 50, slip = 1bp, taker = 4bp
    # Per the prompt: gross_per_trade_bps = lift × E[|fwd_ret|cascade]_bps - cost_round_trip
    # cost_round_trip = 2*4 + 2*|slip| = 10
    # gross_per_trade = 4 × 50 - 10 = 190
    out = _headroom_top_decile_bps(
        lift=4.0,
        mean_edge_in_cascade_bps=50.0,
        slip_bps=1.0,
        taker_fee_bps_per_side=4.0,
    )
    assert math.isclose(out, 190.0, abs_tol=1e-9)


def test_headroom_zero_lift_returns_negative_cost():
    from scripts.cascade_precursor_probe import _headroom_top_decile_bps

    # No lift → no expected gross → just lose the round-trip cost
    out = _headroom_top_decile_bps(
        lift=1.0,
        mean_edge_in_cascade_bps=10.0,
        slip_bps=1.0,
        taker_fee_bps_per_side=4.0,
    )
    # gross = 1 × 10 - (2*4 + 2*1) = 10 - 10 = 0
    assert math.isclose(out, 0.0, abs_tol=1e-9)


def test_per_window_lookup_to_dict_smoke():
    """Build a small per-window dataframe; the aggregation columns exist."""
    from scripts.cascade_precursor_probe import _row_for_window

    row = _row_for_window(
        symbol="BTC",
        horizon=100,
        fold="2026-02",
        anchor_ts=1700000000000,
        date="2026-02-01",
        window_start=0,
        pred_proba=0.42,
        synthetic_cascade_label=0,
        real_cascade_label=None,
        top_decile_bool=False,
        edge_bps=12.5,
        slip_bps=1.5,
    )
    assert row["symbol"] == "BTC"
    assert row["horizon"] == 100
    assert row["fold"] == "2026-02"
    assert math.isclose(float(row["pred_proba"]), 0.42)
    assert int(row["synthetic_cascade_label"]) == 0
    assert pd.isna(row["real_cascade_label"])
    assert bool(row["top_decile_bool"]) is False


# ---------------------------------------------------------------------------
# Stage-2 (real cascade label) helpers
# ---------------------------------------------------------------------------


def test_bootstrap_auc_ci_perfect_separation_is_one():
    """Bootstrap CI on perfect separation should be tight near 1.0."""
    from scripts.cascade_precursor_probe import _bootstrap_auc_ci

    rng = np.random.default_rng(0)
    n = 500
    labels = np.zeros(n, dtype=np.int64)
    labels[: n // 2] = 1
    # Perfect ranker: probas correlate perfectly with labels
    proba = labels.astype(np.float64) + rng.normal(scale=0.01, size=n)
    point, lo, hi = _bootstrap_auc_ci(proba, labels, n_boot=200, seed=0)
    assert point > 0.99
    assert lo > 0.95
    assert hi <= 1.0


def test_bootstrap_auc_ci_random_brackets_half():
    """Random predictions on balanced labels: AUC CI brackets 0.5."""
    from scripts.cascade_precursor_probe import _bootstrap_auc_ci

    rng = np.random.default_rng(1)
    n = 1000
    labels = (rng.random(n) < 0.5).astype(np.int64)
    proba = rng.random(n)
    point, lo, hi = _bootstrap_auc_ci(proba, labels, n_boot=200, seed=1)
    # Random ranker → AUC near 0.5; 95% CI should bracket 0.5
    assert lo < 0.5 < hi
    assert 0.4 < point < 0.6


def test_bootstrap_auc_ci_degenerate_returns_nan():
    """All-one or all-zero labels → AUC undefined → NaN tuple."""
    from scripts.cascade_precursor_probe import _bootstrap_auc_ci

    n = 100
    proba = np.linspace(0, 1, n)
    labels_all_zero = np.zeros(n, dtype=np.int64)
    point, lo, hi = _bootstrap_auc_ci(proba, labels_all_zero, n_boot=50, seed=0)
    assert math.isnan(point)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_real_cascade_label_for_window_anchor_horizon_interval():
    """Real-cascade label = 1 iff a liquidation ts falls in (anchor_ts, ts_at(anchor+H)]."""
    from scripts.cascade_precursor_probe import _real_cascade_label_with_event_ts

    # 5 events at ts=10,20,30,40,50; anchor at idx=2 (ts=30); horizon=2 → end_idx=4 (ts=50)
    event_ts = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    # Window starts at idx=0 → anchor is start + WINDOW_LEN - 1 (in this synthetic test
    # we bypass WINDOW_LEN by passing window_starts that imply anchor_idx via the formula
    # in the helper).  Easier: import WINDOW_LEN and adapt.
    from tape.constants import WINDOW_LEN

    # We need window_starts such that start + WINDOW_LEN - 1 = 2 (anchor at idx=2)
    # So start = 2 - WINDOW_LEN + 1.  Build a fake event_ts array large enough.
    n = WINDOW_LEN + 10
    event_ts = np.arange(n, dtype=np.int64) * 10  # ts = 0, 10, 20, ...
    window_starts = np.array([0], dtype=np.int64)  # anchor_idx = WINDOW_LEN - 1
    anchor_idx = WINDOW_LEN - 1
    anchor_ts = np.array([event_ts[anchor_idx]], dtype=np.int64)  # 10*(WINDOW_LEN-1)
    horizon = 5
    end_ts = event_ts[anchor_idx + horizon]

    # Liquidation at ts=anchor_ts+5 (strictly inside (anchor_ts, end_ts])
    liq_ts_inside = np.array([int(anchor_ts[0]) + 5], dtype=np.int64)
    out = _real_cascade_label_with_event_ts(
        anchor_ts=anchor_ts,
        window_starts=window_starts,
        event_ts=event_ts,
        horizon=horizon,
        liq_ts=liq_ts_inside,
    )
    assert out.shape == (1,)
    assert int(out[0]) == 1

    # Liquidation exactly at anchor_ts (excluded by left-open interval)
    liq_ts_at_anchor = np.array([int(anchor_ts[0])], dtype=np.int64)
    out2 = _real_cascade_label_with_event_ts(
        anchor_ts=anchor_ts,
        window_starts=window_starts,
        event_ts=event_ts,
        horizon=horizon,
        liq_ts=liq_ts_at_anchor,
    )
    assert int(out2[0]) == 0

    # Liquidation past end_ts (excluded)
    liq_ts_past = np.array([int(end_ts) + 1], dtype=np.int64)
    out3 = _real_cascade_label_with_event_ts(
        anchor_ts=anchor_ts,
        window_starts=window_starts,
        event_ts=event_ts,
        horizon=horizon,
        liq_ts=liq_ts_past,
    )
    assert int(out3[0]) == 0


def test_precision_recall_at_top_pct_basic():
    """Top-1% precision/recall on a perfectly ranked array."""
    from scripts.cascade_precursor_probe import _precision_recall_at_top_pct

    n = 1000
    # 5 cascades, perfectly ranked at the top
    proba = np.linspace(0, 1, n)
    labels = np.zeros(n, dtype=np.int64)
    labels[-5:] = 1
    p, r = _precision_recall_at_top_pct(proba, labels, top_pct=0.01)
    # top 1% = 10 windows; 5 of them are cascades
    assert math.isclose(p, 0.5, abs_tol=1e-9)
    assert math.isclose(r, 1.0, abs_tol=1e-9)


def test_precision_recall_at_top_pct_no_positives_returns_nan():
    from scripts.cascade_precursor_probe import _precision_recall_at_top_pct

    n = 100
    proba = np.linspace(0, 1, n)
    labels = np.zeros(n, dtype=np.int64)
    p, r = _precision_recall_at_top_pct(proba, labels, top_pct=0.01)
    # No positives → precision=0, recall=NaN (denom=0)
    assert math.isclose(p, 0.0, abs_tol=1e-9)
    assert math.isnan(r)


def test_signal_distinguishable_from_baseline_basic():
    """CI lower bound must exceed baseline upper bound for distinguishable=True."""
    from scripts.cascade_precursor_probe import _signal_distinguishable_from_baseline

    # Distinguishable: real CI [0.62, 0.70] vs baseline CI [0.45, 0.55]
    assert _signal_distinguishable_from_baseline(
        real_lo=0.62, real_hi=0.70, baseline_lo=0.45, baseline_hi=0.55
    )
    # Not distinguishable: overlap
    assert not _signal_distinguishable_from_baseline(
        real_lo=0.50, real_hi=0.65, baseline_lo=0.45, baseline_hi=0.55
    )
    # Tied at the boundary — strict greater-than → not distinguishable
    assert not _signal_distinguishable_from_baseline(
        real_lo=0.55, real_hi=0.65, baseline_lo=0.45, baseline_hi=0.55
    )
    # NaN inputs propagate to False
    assert not _signal_distinguishable_from_baseline(
        real_lo=float("nan"), real_hi=0.7, baseline_lo=0.45, baseline_hi=0.55
    )


def test_april_diagnostic_dates_listing():
    """Helper returns the canonical April 1-13 date list (calendar dates, not gated on data)."""
    from scripts.cascade_precursor_probe import _april_diagnostic_dates

    out = _april_diagnostic_dates()
    assert out[0] == "2026-04-01"
    assert out[-1] == "2026-04-13"
    assert len(out) == 13
    # Strictly chronological
    assert out == sorted(out)
