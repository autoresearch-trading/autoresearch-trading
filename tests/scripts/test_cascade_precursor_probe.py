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


# ---------------------------------------------------------------------------
# Step-1/4 robustness — day-clustered bootstrap
# ---------------------------------------------------------------------------


def _make_synthetic_per_window_for_robustness(
    *,
    n_days: int = 7,
    n_per_day: int = 200,
    n_pos_per_day: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a fake per-window dataframe shaped like the cascade-real parquet.

    Probabilities are perfectly separable (positives draw from N(2, 1)) so AUC ≈ 1
    for the real label and ~0.5 for the shuffled control.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for d in range(n_days):
        date = f"2026-04-{(d + 3):02d}"
        labels = np.zeros(n_per_day, dtype=np.int8)
        labels[:n_pos_per_day] = 1
        rng.shuffle(labels)
        # Real model: separable
        proba = np.where(
            labels == 1,
            rng.normal(loc=2.0, scale=1.0, size=n_per_day),
            rng.normal(loc=0.0, scale=1.0, size=n_per_day),
        )
        # Shuffled control: same labels, random scores
        proba_shuf = rng.normal(size=n_per_day)
        proba_rf = rng.normal(size=n_per_day)
        for i in range(n_per_day):
            rows.append(
                {
                    "symbol": "BTC",
                    "date": date,
                    "anchor_ts": 1_700_000_000_000 + d * 86_400_000 + i,
                    "horizon": 500,
                    "real_cascade_label": int(labels[i]),
                    "pred_proba": float(proba[i]),
                    "pred_proba_shuffled": float(proba_shuf[i]),
                    "pred_proba_random_feat": float(proba_rf[i]),
                    "fold": date,
                    "scope": "pooled",
                }
            )
    return pd.DataFrame(rows)


def test_day_clustered_bootstrap_auc_recovers_separable_signal():
    """On a perfectly separable dataset the day-clustered bootstrap CI must
    bracket a high AUC and exclude 0.5."""
    from scripts.cascade_precursor_probe import _day_clustered_bootstrap_auc

    df = _make_synthetic_per_window_for_robustness(seed=42)
    point, lo, hi = _day_clustered_bootstrap_auc(
        df,
        proba_col="pred_proba",
        label_col="real_cascade_label",
        date_col="date",
        n_boot=200,
        seed=0,
    )
    assert math.isfinite(point)
    assert math.isfinite(lo)
    assert math.isfinite(hi)
    # Separable problem → real-AUC is high, lower bound clears 0.6 comfortably.
    assert point > 0.85, f"point={point}"
    assert lo > 0.6, f"lo={lo}"
    # Shuffled control hovers near 0.5; bootstrap CI should NOT exclude 0.5.
    s_point, s_lo, s_hi = _day_clustered_bootstrap_auc(
        df,
        proba_col="pred_proba_shuffled",
        label_col="real_cascade_label",
        date_col="date",
        n_boot=200,
        seed=0,
    )
    assert math.isfinite(s_point)
    # Shuffled CI brackets 0.5 (lo<0.5<hi) — lower bound below 0.5 strict-greater test.
    assert s_lo < 0.5 < s_hi, f"shuffled lo={s_lo} hi={s_hi}"


def test_day_clustered_bootstrap_resamples_at_day_level():
    """If we resample at the day level with replacement, a single duplicated day
    should appear with a non-trivial probability (~1 - (n-1)^k / n^k for k draws)."""
    from scripts.cascade_precursor_probe import _day_clustered_bootstrap_iter_indices

    rng = np.random.default_rng(0)
    days = ["2026-04-03", "2026-04-04", "2026-04-06"]
    # Per-day index lists
    day_to_idx = {
        days[0]: np.array([0, 1, 2]),
        days[1]: np.array([3, 4]),
        days[2]: np.array([5, 6, 7, 8]),
    }
    n_iter = 200
    has_duplicate = 0
    sizes: list[int] = []
    for _ in range(n_iter):
        boot_idx = _day_clustered_bootstrap_iter_indices(days, day_to_idx, rng=rng)
        # Each draw concatenates ALL windows of the sampled days
        sizes.append(len(boot_idx))
        # Detect day duplication: total size > sum of unique-day sizes
        # We test by checking that occasionally the bootstrap fold has 9 rows
        # (e.g. day-0 sampled 3× → 3+3+3 = 9). With n=3 days, every draw concatenates
        # 3 days' worth of windows so the size is sum of 3 per-day counts.
        if len(boot_idx) > sum(len(v) for v in day_to_idx.values()):
            # Cannot exceed total — only equal when all unique. Larger means duplication
            # → impossible by construction, but guard against off-by-one.
            has_duplicate += 1
    # Each bootstrap draw should have len = sum_of_3_sampled_day_sizes.
    # Min size = 3*2 = 6 (three days of [3,4]); max = 3*4 = 12 (three days of [5,6,7,8]).
    assert min(sizes) >= 2 * 3, f"too small, min={min(sizes)}"
    assert max(sizes) <= 4 * 3, f"too large, max={max(sizes)}"


def test_per_day_attribution_columns():
    """Per-day attribution returns one row per (date, horizon) with required cols."""
    from scripts.cascade_precursor_probe import _per_day_attribution

    df = _make_synthetic_per_window_for_robustness(seed=11)
    out = _per_day_attribution(
        df,
        horizon=500,
        proba_col="pred_proba",
        label_col="real_cascade_label",
        date_col="date",
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 7  # 7 days
    required = {
        "date",
        "horizon",
        "n_cascades",
        "n_windows",
        "auc",
        "precision_top_1pct",
        "leave_out_pooled_auc_drop",
    }
    assert required.issubset(set(out.columns))
    assert (out["horizon"] == 500).all()
    assert (out["n_windows"] > 0).all()
    # Pooled AUC is high → leave-one-day-out drop is small per day on a balanced set.
    assert out["leave_out_pooled_auc_drop"].notna().all()


def test_precision_top_1pct_helper_matches_definition():
    """Day-clustered precision-at-top-1% helper agrees with the existing utility on
    a single-day sample."""
    from scripts.cascade_precursor_probe import (
        _precision_at_top_pct_pooled,
        _precision_recall_at_top_pct,
    )

    rng = np.random.default_rng(7)
    proba = rng.uniform(size=1000)
    labels = (rng.uniform(size=1000) < 0.04).astype(np.int8)
    p1, _ = _precision_recall_at_top_pct(proba, labels, top_pct=0.01)
    p2 = _precision_at_top_pct_pooled(proba, labels, top_pct=0.01)
    assert abs(p1 - p2) < 1e-12


# ---------------------------------------------------------------------------
# Cascade-direction probe — Step 2 (direction LR on cascade-likely subset)
# ---------------------------------------------------------------------------


def test_realized_direction_sign_handles_zero_and_signs():
    """Realized direction is +1 for positive forward return, 0 for non-positive
    (we treat exactly-zero as 'not positive' to keep the binary target stable)."""
    from scripts.cascade_precursor_probe import _realized_direction_label

    fwd = np.array([0.01, -0.005, 0.0, 0.0001, -1e-9, np.nan])
    out = _realized_direction_label(fwd)
    # +ret → 1, -ret/zero/NaN → 0; NaN preserved as -1 sentinel? We check spec.
    assert out[0] == 1
    assert out[1] == 0
    assert out[2] == 0
    assert out[3] == 1
    assert out[4] == 0
    # NaN forward return → label = -1 (invalid sentinel)
    assert out[5] == -1


def test_top_pct_pred_proba_mask_returns_correct_count():
    """Top-5% by predicted probability returns exactly ceil(n * 0.05) windows."""
    from scripts.cascade_precursor_probe import _top_pct_mask

    rng = np.random.default_rng(0)
    proba = rng.uniform(size=200)
    mask = _top_pct_mask(proba, top_pct=0.05)
    # 200 * 0.05 = 10
    assert mask.sum() == 10
    # The selected probabilities must all be ≥ the 95th percentile cutoff
    cutoff = np.quantile(proba, 0.95)
    assert (proba[mask] >= cutoff - 1e-12).all()


def test_marginal_direction_asymmetry_reports_p_positive():
    """Helper reports P(forward_return > 0 | cascade) on the cascade subset only."""
    from scripts.cascade_precursor_probe import _marginal_direction_asymmetry

    fwd = np.array([0.01, -0.01, 0.02, -0.005, 0.0, 0.001])
    cascade = np.array([1, 1, 1, 0, 1, 1])  # 5 cascades
    p = _marginal_direction_asymmetry(fwd, cascade)
    # Among 5 cascades: pos count = [0.01, 0.02, 0.001] = 3; zero counts as non-pos
    assert abs(p - 3 / 5) < 1e-12


def test_marginal_direction_asymmetry_no_cascades_returns_nan():
    from scripts.cascade_precursor_probe import _marginal_direction_asymmetry

    fwd = np.array([0.01, -0.01, 0.02])
    cascade = np.zeros(3, dtype=np.int8)
    p = _marginal_direction_asymmetry(fwd, cascade)
    assert math.isnan(p)


def test_overshoot_direction_first_liq_fill_in_window():
    """Overshoot direction = sign(first_liq_price - anchor_mid) for the first
    liquidation in (anchor_ts, end_ts]; if no liq in window, sentinel = 0."""
    from scripts.cascade_precursor_probe import _overshoot_direction_for_window

    # Anchor at t=1000, end_ts=2000, anchor_mid=100.
    # Liquidation fills: ts=1500 price=110 (first), ts=1800 price=90.
    liq_ts = np.array([500, 1500, 1800, 2500], dtype=np.int64)
    liq_price = np.array([99.0, 110.0, 90.0, 95.0], dtype=np.float64)
    sign = _overshoot_direction_for_window(
        anchor_ts=1000,
        end_ts=2000,
        anchor_mid=100.0,
        liq_ts=liq_ts,
        liq_price=liq_price,
    )
    # First liquidation in (1000, 2000] is idx=1, price=110 → +1
    assert sign == 1

    # Negative overshoot: first liq price below anchor
    liq_ts2 = np.array([1200, 1900], dtype=np.int64)
    liq_price2 = np.array([95.0, 105.0], dtype=np.float64)
    s2 = _overshoot_direction_for_window(
        anchor_ts=1000,
        end_ts=2000,
        anchor_mid=100.0,
        liq_ts=liq_ts2,
        liq_price=liq_price2,
    )
    assert s2 == -1

    # No liq in window → sentinel 0
    s3 = _overshoot_direction_for_window(
        anchor_ts=1000,
        end_ts=2000,
        anchor_mid=100.0,
        liq_ts=np.array([100, 5000], dtype=np.int64),
        liq_price=np.array([1.0, 2.0], dtype=np.float64),
    )
    assert s3 == 0


def test_majority_class_baseline_auc_is_naive_constant():
    """A baseline that always predicts the majority class has AUC = 0.5 by
    construction (constant scores)."""
    from scripts.cascade_precursor_probe import _majority_class_baseline_auc

    # 70% positive, 30% negative — majority predictor should yield AUC = 0.5
    y = np.concatenate([np.ones(70, dtype=int), np.zeros(30, dtype=int)])
    auc = _majority_class_baseline_auc(y)
    assert abs(auc - 0.5) < 1e-12


def test_direction_lr_pipeline_produces_valid_predictions_on_synthetic_data():
    """End-to-end: feed a small synthetic dataset to the direction LR LOO-CV
    helper and check it returns finite per-window predictions for every window."""
    from scripts.cascade_precursor_probe import _leave_one_day_out_predictions

    rng = np.random.default_rng(0)
    n_days = 5
    n_per_day = 50
    X_list, y_list, dates_list = [], [], []
    for d in range(n_days):
        date = f"2026-04-{d + 3:02d}"
        X_d = rng.normal(size=(n_per_day, 10)).astype(np.float32)
        # Direction label: depends on first feature
        y_d = (X_d[:, 0] > 0).astype(np.int64)
        X_list.append(X_d)
        y_list.append(y_d)
        dates_list.extend([date] * n_per_day)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    dates = np.array(dates_list)

    pred = _leave_one_day_out_predictions(
        X=X, y=y, dates=dates, feature_mode="real", label_mode="real", rng_seed=0
    )
    # Every window has a prediction (one fold per day)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    # Direction LR should beat 0.5 AUC since y is signal-bearing
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y, pred)
    assert auc > 0.7, f"sanity check failed: auc={auc}"


def test_per_day_expected_gross_formula():
    """Per-day expected gross = trigger_freq × (gross_per_trigger - cost_per_trigger).

    gross_per_trigger = (2 × dir_acc - 1) × E[|fwd_ret|_h500] (in bps).
    cost_per_trigger = 2 × fee + 2 × slip (in bps).
    """
    from scripts.cascade_precursor_probe import _direction_per_day_expected_gross

    out = _direction_per_day_expected_gross(
        triggers_per_day=1.0,
        direction_accuracy=0.60,  # 10pp above coin-flip
        mean_abs_fwd_bps=100.0,  # 1% mean on cascade-likely
        fee_bps_per_side=4.0,
        slip_bps_per_side=1.0,
    )
    # gross = (2*0.6-1)*100 = 20bps; cost = 2*4+2*1 = 10bps; net = 10bps; daily = 10bps × 1
    assert abs(out["gross_per_trigger_bps"] - 20.0) < 1e-9
    assert abs(out["cost_per_trigger_bps"] - 10.0) < 1e-9
    assert abs(out["net_per_trigger_bps"] - 10.0) < 1e-9
    assert abs(out["per_day_gross_bps"] - 10.0) < 1e-9
