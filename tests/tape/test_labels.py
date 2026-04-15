# tests/tape/test_labels.py
import numpy as np
import pandas as pd
import pytest

from tape.constants import DIRECTION_HORIZONS, SPRING_SIGMA_MULT
from tape.labels import (
    DirectionLabels,
    compute_direction_labels,
    compute_wyckoff_labels,
)

# ---------------------------------------------------------------------------
# Direction label tests
# ---------------------------------------------------------------------------


def test_direction_labels_all_horizons():
    n = 2_000
    rng = np.random.default_rng(0)
    vwap = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    out: DirectionLabels = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        key = f"h{h}"
        assert key in out, f"Missing key {key}"
        assert len(out[key]) == n, f"Wrong length for {key}"
        # Last h events have no label — sentinel is 0
        assert (out[key][-h:] == 0).all(), f"Tail of {key} not zero-sentinel"
    # mask_h100: True where valid, False in tail
    assert out["mask_h100"][-100:].sum() == 0, "mask_h100 tail must be all False"
    assert out["mask_h100"][:-100].all(), "mask_h100 body must be all True"


def test_direction_labels_correct_sign_monotone_increasing():
    """On a strictly increasing VWAP, all valid labels must be 1 (up)."""
    n = 600
    vwap = np.linspace(100.0, 200.0, n)  # strictly increasing
    out = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        valid = out[f"mask_h{h}"]
        assert valid.sum() == n - h, f"Expected {n-h} valid events for h{h}"
        assert (
            out[f"h{h}"][valid] == 1
        ).all(), f"All valid labels for h{h} should be 1 on monotone vwap"


def test_direction_labels_correct_sign_monotone_decreasing():
    """On a strictly decreasing VWAP, all valid labels must be 0 (down/flat)."""
    n = 600
    vwap = np.linspace(200.0, 100.0, n)  # strictly decreasing
    out = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        valid = out[f"mask_h{h}"]
        assert (
            out[f"h{h}"][valid] == 0
        ).all(), f"All valid labels for h{h} should be 0 on decreasing vwap"


def test_direction_labels_short_series_all_masked():
    """Series shorter than smallest horizon: all events masked, all labels zero."""
    vwap = np.array([100.0, 100.1, 100.2, 99.5, 100.5], dtype=float)  # 5 events
    out = compute_direction_labels(vwap)
    # h=10 > 5 events: no valid labels
    assert out["h10"].shape == (5,)
    assert (out["mask_h10"] == False).all()
    assert (out["h10"] == 0).all()


def test_direction_labels_exactly_h_events():
    """Series with exactly H events: all masked (no forward data)."""
    for h in DIRECTION_HORIZONS:
        vwap = np.linspace(100.0, 110.0, h)
        out = compute_direction_labels(vwap)
        assert (
            out[f"mask_h{h}"] == False
        ).all(), f"h{h}: exactly h events must all be masked"
        assert (out[f"h{h}"] == 0).all()


def test_direction_labels_h_plus_one_events():
    """Series with H+1 events: exactly the first event is valid."""
    for h in DIRECTION_HORIZONS:
        vwap = np.linspace(100.0, 110.0, h + 1)  # increasing -> label=1
        out = compute_direction_labels(vwap)
        assert out[f"mask_h{h}"][0] == True, f"h{h}: first event must be valid"
        assert out[f"h{h}"][0] == 1, f"h{h}: first label must be 1 on increasing vwap"
        assert (out[f"mask_h{h}"][1:] == False).all()


def test_direction_labels_flat_vwap_is_zero():
    """Flat VWAP (vwap[i+h] == vwap[i]): label = 0 (down-or-flat)."""
    n = 200
    vwap = np.full(n, 100.0)
    out = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        valid = out[f"mask_h{h}"]
        assert (out[f"h{h}"][valid] == 0).all(), f"Flat vwap should give 0 for h{h}"


def test_direction_labels_no_nan_inf():
    """No NaN or inf values in any label or mask array."""
    n = 1_000
    rng = np.random.default_rng(42)
    vwap = 100.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    out = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        arr = out[f"h{h}"].astype(float)
        assert np.all(np.isfinite(arr)), f"h{h} labels contain NaN/inf"


def test_direction_labels_day_boundary_masking():
    """Day-boundary: within the last H events of a day, labels must be masked.

    We pass in just one day's events (600 events). For each horizon H,
    the last H events must be masked. This tests the day-boundary contract
    described in gotcha #26.
    """
    n = 600
    rng = np.random.default_rng(7)
    vwap = 100.0 + np.cumsum(rng.normal(0, 0.05, size=n))
    out = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        tail_mask = out[f"mask_h{h}"][-h:]
        assert tail_mask.sum() == 0, (
            f"Day boundary: last {h} events of mask_h{h} must be False "
            f"(no cross-day lookahead)"
        )


def test_direction_labels_dtype():
    """Direction labels must be int8; masks must be bool."""
    n = 300
    vwap = np.linspace(100.0, 150.0, n)
    out = compute_direction_labels(vwap)
    for h in DIRECTION_HORIZONS:
        assert out[f"h{h}"].dtype == np.int8, f"h{h} must be int8"
        assert out[f"mask_h{h}"].dtype == bool, f"mask_h{h} must be bool"


# ---------------------------------------------------------------------------
# Wyckoff label tests
# ---------------------------------------------------------------------------


def _make_wyckoff_inputs(n: int, seed: int = 0) -> dict:
    """Helper: return keyword-ready dict of zero arrays for compute_wyckoff_labels."""
    rng = np.random.default_rng(seed)
    return dict(
        log_return=rng.normal(0, 0.01, n),
        effort_vs_result=np.zeros(n, dtype=float),
        is_open=np.full(n, 0.5),
        climax_score=np.zeros(n, dtype=float),
        z_qty=np.zeros(n, dtype=float),
        z_ret=np.zeros(n, dtype=float),
        log_spread=rng.normal(-5, 1, n),
        depth_ratio=rng.normal(0, 1, n),
        kyle_lambda=rng.normal(0, 1, n),
        cum_ofi_5=rng.normal(0, 1, n),
    )


def test_wyckoff_labels_keys_present():
    """compute_wyckoff_labels must return all five expected label keys."""
    n = 500
    wl = compute_wyckoff_labels(**_make_wyckoff_inputs(n))
    for key in ("stress", "informed_flow", "climax", "spring", "absorption"):
        assert key in wl, f"Missing Wyckoff label key: {key}"


def test_wyckoff_labels_shapes():
    """All Wyckoff label arrays must have shape (n,)."""
    n = 800
    wl = compute_wyckoff_labels(**_make_wyckoff_inputs(n))
    for key in ("stress", "informed_flow", "climax", "spring", "absorption"):
        assert wl[key].shape == (n,), f"{key} has wrong shape"


def test_wyckoff_labels_dtype_int8():
    """All Wyckoff labels must be int8."""
    n = 400
    wl = compute_wyckoff_labels(**_make_wyckoff_inputs(n))
    for key in ("stress", "informed_flow", "climax", "spring", "absorption"):
        assert wl[key].dtype == np.int8, f"{key} dtype must be int8"


def test_wyckoff_labels_binary_values():
    """All Wyckoff labels are in {0, 1}."""
    n = 1_200
    wl = compute_wyckoff_labels(**_make_wyckoff_inputs(n))
    for key in ("stress", "informed_flow", "climax", "spring", "absorption"):
        unique = np.unique(wl[key])
        assert set(unique).issubset(
            {0, 1}
        ), f"{key} contains values outside {{0,1}}: {unique}"


def test_wyckoff_labels_no_nan_inf():
    """No NaN or inf in any Wyckoff label."""
    n = 1_000
    wl = compute_wyckoff_labels(**_make_wyckoff_inputs(n))
    for key in ("stress", "informed_flow", "climax", "spring", "absorption"):
        assert np.all(np.isfinite(wl[key].astype(float))), f"{key} contains NaN/inf"


# ---------------------------------------------------------------------------
# Spring label — sigma multiplier tests (falsifiability prereq #4)
# ---------------------------------------------------------------------------


def test_spring_sigma_mult_constant_is_3():
    """SPRING_SIGMA_MULT from tape.constants must equal 3.0 exactly."""
    assert SPRING_SIGMA_MULT == 3.0, f"Expected 3.0, got {SPRING_SIGMA_MULT}"


def test_spring_does_not_fire_at_2pt5_sigma():
    """Falsifiability prereq #4: spring must NOT fire at -2.5σ (< 3.0σ threshold).

    Uses a fully deterministic alternating ±0.02 series so that the rolling σ
    is tightly controlled (~0.020).  A spike of -0.025 is only ~1.25σ under
    this regime, which is well below the 3.0σ threshold.  All other spring
    conditions (evr>1, is_open>0.5, recent_mean>0) are also satisfied so
    the only reason spring should NOT fire is the depth condition.
    """
    n = 200
    # Alternating ±0.02 gives rolling std ≈ 0.020 -> 3σ threshold ≈ 0.060
    log_ret = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(n)], dtype=float)
    # Spike at event 100: -0.025 (< 0.060 threshold, so spring must NOT fire)
    log_ret[100] = -0.025
    # Ensure cond4: mean of last 10 returns at events 100-109 must be positive.
    # Set events 101-109 to +0.02 so the window mean is positive after recovery.
    log_ret[101:110] = 0.02

    evr = np.full(n, 1.5)  # cond2 satisfied everywhere
    is_open = np.full(n, 0.8)  # cond3 satisfied everywhere

    wl = compute_wyckoff_labels(
        log_return=log_ret,
        effort_vs_result=evr,
        is_open=is_open,
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=np.zeros(n),
        depth_ratio=np.zeros(n),
        kyle_lambda=np.zeros(n),
        cum_ofi_5=np.zeros(n),
    )
    assert wl["spring"].sum() == 0, (
        f"Spring must not fire at ~1.25σ with SPRING_SIGMA_MULT=3.0. "
        f"Fired at indices: {np.where(wl['spring'])[0]}"
    )


def test_spring_fires_at_deep_sigma():
    """Spring MUST fire when all conditions are met at >> 3.0σ depth.

    Deterministic construction (n=300):
    - Event 100: large spike of -0.10 (buried well before recovery window)
    - Events 111-121: small positive returns (+0.005) — the "recovery" window
    - At event 120: rolling std ≈ 0.0092; threshold ≈ 0.028; spike=-0.10 << threshold
    - cond2 (evr>1.0), cond3 (is_open>0.5), cond4 (mean_last10 > 0) all satisfied
    """
    n = 300
    log_ret = np.zeros(n, dtype=float)
    log_ret[100] = -0.10  # large negative spike (~10σ given rolling std ≈ 0.009)
    log_ret[111:121] = 0.005  # positive recovery so cond4 fires at event 120

    evr = np.full(n, 1.5)  # cond2 satisfied everywhere
    is_open = np.full(n, 0.8)  # cond3 satisfied everywhere

    wl = compute_wyckoff_labels(
        log_return=log_ret,
        effort_vs_result=evr,
        is_open=is_open,
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=np.zeros(n),
        depth_ratio=np.zeros(n),
        kyle_lambda=np.zeros(n),
        cum_ofi_5=np.zeros(n),
    )
    assert (
        wl["spring"].sum() > 0
    ), "Spring must fire when log_return dips far below -3.0σ with all conditions met"
    # Verify the fire is in the expected window (after recovery begins at 111)
    assert (
        wl["spring"][120] == 1
    ), "Spring must fire at event 120 where recovery is complete"


# ---------------------------------------------------------------------------
# Climax label tests
# ---------------------------------------------------------------------------


def test_climax_requires_both_z_qty_and_z_ret():
    """Climax fires only when BOTH z_qty > 2 AND z_ret > 2."""
    n = 50
    base = _make_wyckoff_inputs(n)
    base["z_qty"] = np.zeros(n)
    base["z_ret"] = np.zeros(n)

    # Only z_qty high
    base_qty_only = {**base, "z_qty": np.full(n, 3.0), "z_ret": np.zeros(n)}
    wl_qty = compute_wyckoff_labels(**base_qty_only)
    assert wl_qty["climax"].sum() == 0, "Climax must not fire on z_qty alone"

    # Only z_ret high
    base_ret_only = {**base, "z_qty": np.zeros(n), "z_ret": np.full(n, 3.0)}
    wl_ret = compute_wyckoff_labels(**base_ret_only)
    assert wl_ret["climax"].sum() == 0, "Climax must not fire on z_ret alone"

    # Both high
    base_both = {**base, "z_qty": np.full(n, 3.0), "z_ret": np.full(n, 3.0)}
    wl_both = compute_wyckoff_labels(**base_both)
    assert (
        wl_both["climax"].sum() == n
    ), "Climax must fire when both z_qty>2 and z_ret>2"


def test_climax_threshold_exactly_at_boundary():
    """Climax does not fire at exactly z=2.0 (strict >)."""
    n = 50
    base = _make_wyckoff_inputs(n)
    wl = compute_wyckoff_labels(
        **{**base, "z_qty": np.full(n, 2.0), "z_ret": np.full(n, 2.0)}
    )
    assert wl["climax"].sum() == 0, "Climax must use strict > 2.0"


# ---------------------------------------------------------------------------
# Stress label tests
# ---------------------------------------------------------------------------


def test_stress_fires_when_both_percentiles_crossed():
    """Stress fires when log_spread > p90 AND |depth_ratio| > p90."""
    n = 1_200
    rng = np.random.default_rng(0)
    log_spread = rng.normal(-5, 1, size=n)
    depth_ratio = rng.normal(0, 1, size=n)
    # Inject a clear extreme value at position 1100
    log_spread[1100] = 10.0
    depth_ratio[1100] = 10.0
    wl = compute_wyckoff_labels(
        log_return=np.zeros(n),
        effort_vs_result=np.zeros(n),
        is_open=np.zeros(n),
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=log_spread,
        depth_ratio=depth_ratio,
        kyle_lambda=np.zeros(n),
        cum_ofi_5=np.zeros(n),
    )
    assert (
        wl["stress"][1100] == 1
    ), "Stress must fire at extreme log_spread + depth_ratio"


def test_stress_does_not_fire_when_only_one_condition():
    """Stress requires BOTH conditions. One alone must not fire."""
    n = 1_200
    rng = np.random.default_rng(1)
    log_spread = rng.normal(-5, 1, size=n)
    depth_ratio = rng.normal(0, 1, size=n)

    # Extreme log_spread only
    log_spread_high = log_spread.copy()
    log_spread_high[1100] = 10.0
    wl1 = compute_wyckoff_labels(
        log_return=np.zeros(n),
        effort_vs_result=np.zeros(n),
        is_open=np.zeros(n),
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=log_spread_high,
        depth_ratio=depth_ratio,
        kyle_lambda=np.zeros(n),
        cum_ofi_5=np.zeros(n),
    )
    assert wl1["stress"][1100] == 0, "Stress must not fire on extreme log_spread alone"

    # Extreme depth_ratio only
    depth_ratio_high = depth_ratio.copy()
    depth_ratio_high[1100] = 10.0
    wl2 = compute_wyckoff_labels(
        log_return=np.zeros(n),
        effort_vs_result=np.zeros(n),
        is_open=np.zeros(n),
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=log_spread,
        depth_ratio=depth_ratio_high,
        kyle_lambda=np.zeros(n),
        cum_ofi_5=np.zeros(n),
    )
    assert (
        wl2["stress"][1100] == 0
    ), "Stress must not fire on extreme |depth_ratio| alone"


# ---------------------------------------------------------------------------
# Absorption label tests
# ---------------------------------------------------------------------------


def test_absorption_fires_on_high_effort_low_return():
    """Absorption fires when effort_vs_result is high AND |log_return| is low."""
    n = 1_500
    rng = np.random.default_rng(3)
    # Mostly low effort
    evr = rng.uniform(-1, 1, n)
    log_ret = rng.normal(0, 0.01, n)
    # Last 100 events: very high effort, near-zero return (should be absorption)
    evr[-100:] = 5.0
    log_ret[-100:] = 0.0001
    wl = compute_wyckoff_labels(
        log_return=log_ret,
        effort_vs_result=evr,
        is_open=np.zeros(n),
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=np.zeros(n),
        depth_ratio=np.zeros(n),
        kyle_lambda=np.zeros(n),
        cum_ofi_5=np.zeros(n),
    )
    assert (
        wl["absorption"][-50:].sum() > 0
    ), "Absorption must fire when effort_vs_result is extreme and |log_return| is near zero"


# ---------------------------------------------------------------------------
# Informed flow label tests
# ---------------------------------------------------------------------------


def test_informed_flow_requires_sign_consistency():
    """informed_flow requires 3 consecutive same-sign cum_ofi_5 values."""
    n = 200
    rng = np.random.default_rng(5)
    # High kyle_lambda and |cum_ofi_5| values
    kyle_lambda = np.full(n, 10.0)
    cum_ofi_5_alternating = (
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)]) * 5.0
    )
    wl = compute_wyckoff_labels(
        log_return=np.zeros(n),
        effort_vs_result=np.zeros(n),
        is_open=np.zeros(n),
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=np.zeros(n),
        depth_ratio=np.zeros(n),
        kyle_lambda=kyle_lambda,
        cum_ofi_5=cum_ofi_5_alternating,
    )
    # Alternating signs never produce 3 consecutive consistent signs
    assert (
        wl["informed_flow"].sum() == 0
    ), "informed_flow must not fire with alternating cum_ofi_5 signs"


def test_informed_flow_fires_with_consistent_signs():
    """informed_flow fires when all three conditions are met with consistent signs."""
    n = 500
    rng = np.random.default_rng(6)
    # Build something where most values are low (so p75/p50 thresholds are easy to cross)
    kyle_lambda = rng.uniform(0, 0.5, n)
    cum_ofi_5 = rng.uniform(-0.5, 0.5, n)
    # Inject a window where all 3 conditions are clearly met
    kyle_lambda[300:350] = 100.0  # far above p75
    cum_ofi_5[300:350] = 50.0  # far above p50, consistently positive
    wl = compute_wyckoff_labels(
        log_return=np.zeros(n),
        effort_vs_result=np.zeros(n),
        is_open=np.zeros(n),
        climax_score=np.zeros(n),
        z_qty=np.zeros(n),
        z_ret=np.zeros(n),
        log_spread=np.zeros(n),
        depth_ratio=np.zeros(n),
        kyle_lambda=kyle_lambda,
        cum_ofi_5=cum_ofi_5,
    )
    # At least some of the injected window should fire
    assert (
        wl["informed_flow"][302:350].sum() > 0
    ), "informed_flow must fire in the high-kyle_lambda + consistent-OFI window"
