# tests/scripts/test_encoder_confidence_probe.py
"""Smoke tests for encoder_confidence_probe — confidence quintile + cost-band logic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _has_train_and_test_shards(symbol: str = "BTC") -> bool:
    cache = Path("data/cache")
    if not cache.exists():
        return False
    train_ok = any(cache.glob(f"{symbol}__2025-1*-*.npz")) or any(
        cache.glob(f"{symbol}__2026-01-*.npz")
    )
    test_ok = any(cache.glob(f"{symbol}__2026-02-*.npz"))
    return train_ok and test_ok


def _has_per_window_parquet() -> bool:
    return Path("docs/experiments/goal-a-feasibility/per_window.parquet").exists()


def _has_checkpoint() -> bool:
    return Path("runs/step3-r2/encoder-best.pt").exists()


def test_binomial_2sigma_lower_bound_basics():
    from scripts.encoder_confidence_probe import _binomial_2sigma_lower

    # Trivial: 50% over 100 → SE = 0.05 → lower = 0.40
    assert math.isclose(_binomial_2sigma_lower(0.5, 100), 0.40, abs_tol=1e-9)
    # Larger n shrinks band
    assert _binomial_2sigma_lower(0.55, 1000) > _binomial_2sigma_lower(0.55, 100)
    # Floor at 0
    assert _binomial_2sigma_lower(0.001, 5) >= 0.0
    # Degenerate n
    assert math.isnan(_binomial_2sigma_lower(0.5, 0))


def test_per_cell_quintile_assign_partitions_into_5():
    from scripts.encoder_confidence_probe import _per_cell_quintile_assign

    # Uniform random over [0.5, 1.0] → 5 equal-frequency buckets
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=1000)
    qs = _per_cell_quintile_assign(conf)
    assert qs.min() == 1
    assert qs.max() == 5
    counts = np.bincount(qs, minlength=6)[1:]
    # equal-frequency → all bucket counts within ±5% of 200
    assert all(abs(c - 200) <= 20 for c in counts)


def test_per_cell_quintile_degenerate_returns_zeros():
    from scripts.encoder_confidence_probe import _per_cell_quintile_assign

    # All identical confidences → can't form 5 non-trivial buckets
    qs = _per_cell_quintile_assign(np.full(100, 0.7))
    assert (qs == 0).all()
    # Too-small sample
    qs = _per_cell_quintile_assign(np.array([0.5, 0.6, 0.7]))
    assert (qs == 0).all()


def test_walk_forward_logreg_returns_pred_and_proba():
    from scripts.encoder_confidence_probe import _walk_forward_logreg

    rng = np.random.default_rng(42)
    Xtr = rng.normal(size=(500, 8))
    # Plant a weak signal: y depends on first feature
    ytr = (Xtr[:, 0] + 0.5 * rng.normal(size=500) > 0).astype(np.int64)
    Xte = rng.normal(size=(100, 8))
    pred, p_up = _walk_forward_logreg(Xtr, ytr, Xte)
    assert pred.shape == (100,)
    assert p_up.shape == (100,)
    assert pred.dtype == np.int64
    assert p_up.dtype == np.float64
    assert ((p_up >= 0.0) & (p_up <= 1.0)).all()
    # Confidence = max(p, 1-p) ∈ [0.5, 1.0]
    conf = np.maximum(p_up, 1.0 - p_up)
    assert (conf >= 0.5 - 1e-9).all()
    assert (conf <= 1.0 + 1e-9).all()


def test_walk_forward_logreg_degenerate_train_returns_majority():
    from scripts.encoder_confidence_probe import _walk_forward_logreg

    Xtr = np.random.normal(size=(50, 4))
    ytr = np.zeros(50, dtype=np.int64)  # all class 0
    Xte = np.random.normal(size=(10, 4))
    pred, p_up = _walk_forward_logreg(Xtr, ytr, Xte)
    assert (pred == 0).all()
    assert np.allclose(p_up, 0.5)


def test_aggregate_cells_basic_math():
    """Synthetic per-window rows → cells with correct accuracy + headroom."""
    from scripts.encoder_confidence_probe import _aggregate_cells

    # 10 windows, all correct, ~10bp edge, ~1bp slip
    rows = []
    for i in range(10):
        rows.append(
            {
                "symbol": "TST",
                "date": "2026-02-01",
                "horizon": 100,
                "fold": "2026-02",
                "anchor_ts": 1_700_000_000_000 + i,
                "window_start": i * 200,
                "pred_label": 1,
                "p_up": 0.9,
                "confidence": 0.9,
                "realized_label": 1,
                "pred_correct": 1,
                "quintile": 5,
            }
        )
    pw = pd.DataFrame(rows)
    cost_rows = [
        {
            "symbol": "TST",
            "date": "2026-02-01",
            "anchor_ts": 1_700_000_000_000 + i,
            "horizon": 100,
            "edge_bps": 10.0,
            "slip_avg_bps": 1.0,
            "fillable": True,
        }
        for i in range(10)
    ]
    cost = pd.DataFrame(cost_rows)
    cells = _aggregate_cells(pw, cost)
    assert len(cells) == 1
    row = cells.iloc[0]
    assert row["directional_accuracy"] == 1.0
    assert row["mean_abs_edge_bps"] == 10.0
    # gross = (2*1 - 1) * 10 = 10. cost = 2*4 + 2*1 = 10. headroom = 0.
    assert math.isclose(row["headroom_bps"], 0.0, abs_tol=1e-9)
    # tradeable: acc > 0.55 ✓, binomial_lo > 0.51 ✓ at n=10 acc=1
    # but headroom_bps > 0 fails → tradeable False
    assert bool(row["tradeable"]) is False


def test_build_per_window_lookup_filters_size_and_horizon():
    """If the goal-a per_window.parquet exists, lookup filters correctly."""
    if not _has_per_window_parquet():
        pytest.skip("per_window.parquet not on disk")
    from scripts.encoder_confidence_probe import _build_per_window_lookup

    df = _build_per_window_lookup(
        Path("docs/experiments/goal-a-feasibility/per_window.parquet"),
        horizons=(100, 500),
        size_usd=1000.0,
    )
    assert bool(df["horizon"].isin([100, 500]).all())
    # one row per (symbol, date, anchor_ts, horizon)
    n_dups = df.duplicated(
        subset=["symbol", "date", "anchor_ts", "horizon"], keep=False
    ).sum()
    assert n_dups == 0


@pytest.mark.skipif(
    not (
        _has_train_and_test_shards("BTC")
        and _has_checkpoint()
        and _has_per_window_parquet()
    ),
    reason="need cache shards + checkpoint + per_window.parquet",
)
def test_end_to_end_smoke_one_symbol(tmp_path):
    """Run the full pipeline on a single symbol; verify outputs exist."""
    import sys

    from scripts.encoder_confidence_probe import main as run

    out_dir = tmp_path / "out"
    argv_orig = sys.argv
    try:
        sys.argv = [
            "encoder_confidence_probe",
            "--checkpoint",
            "runs/step3-r2/encoder-best.pt",
            "--cache",
            "data/cache",
            "--per-window",
            "docs/experiments/goal-a-feasibility/per_window.parquet",
            "--out-dir",
            str(out_dir),
            "--symbols",
            "BTC",
            "--horizons",
            "100",
            "500",
            "--batch-size",
            "32",
        ]
        rc = run()
    finally:
        sys.argv = argv_orig
    assert rc == 0
    assert (out_dir / "encoder_confidence_per_window.parquet").exists()
    assert (out_dir / "encoder_confidence_table.csv").exists()
    assert (out_dir / "encoder_confidence.md").exists()
    df = pd.read_parquet(out_dir / "encoder_confidence_per_window.parquet")
    assert len(df) > 0
    assert {
        "symbol",
        "horizon",
        "fold",
        "anchor_ts",
        "confidence",
        "quintile",
    }.issubset(df.columns)
