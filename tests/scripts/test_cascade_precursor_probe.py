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
