# tests/tape/test_io_parquet.py
"""Regression tests for tape/io_parquet.py — expand_ob_levels zero-fill.

Regression for: fix(tape): zero-fill missing OB levels instead of NaN
Root cause: np.full(..., np.nan) left short book snapshots with NaN at
missing depth levels, poisoning depth_ratio / imbalance_L5 / cum_ofi_5 /
delta_imbalance_L1 downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tape.io_parquet import expand_ob_levels


def _make_level_list(n: int, base_price: float, side: str) -> list[dict]:
    """Return a list of n level dicts with synthetic price/qty."""
    levels = []
    for i in range(n):
        if side == "bid":
            price = base_price - i * 0.01
        else:
            price = base_price + i * 0.01
        levels.append({"price": price, "qty": float(10 * (i + 1))})
    return levels


def _make_raw_ob(rows: list[tuple[int, int]]) -> pd.DataFrame:
    """Build a raw OB DataFrame.

    rows: list of (n_bid_levels, n_ask_levels) per snapshot.
    Prices are synthetic but structurally valid.
    """
    records = []
    for i, (n_bid, n_ask) in enumerate(rows):
        records.append(
            {
                "ts_ms": 1_700_000_000_000 + i * 24_000,
                "bids": _make_level_list(n_bid, base_price=100.0, side="bid"),
                "asks": _make_level_list(n_ask, base_price=100.05, side="ask"),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core regression: 3-bid / 10-ask snapshot
# ---------------------------------------------------------------------------


class TestExpandObLevelsShortBids:
    """Snapshot with only 3 bid levels — levels 4-10 must be 0.0, not NaN."""

    @pytest.fixture(scope="class")
    def result(self) -> pd.DataFrame:
        raw = _make_raw_ob([(3, 10)])
        return expand_ob_levels(raw)

    def test_shape(self, result: pd.DataFrame) -> None:
        # 1 row, ts_ms + 4*10 level columns
        assert result.shape == (1, 41), f"unexpected shape {result.shape}"

    def test_bid_levels_present_filled(self, result: pd.DataFrame) -> None:
        for lvl in range(1, 4):
            assert result[f"bid{lvl}_price"].iloc[0] > 0.0
            assert result[f"bid{lvl}_qty"].iloc[0] > 0.0

    def test_bid_levels_missing_zero_price(self, result: pd.DataFrame) -> None:
        for lvl in range(4, 11):
            val = result[f"bid{lvl}_price"].iloc[0]
            assert val == 0.0, f"bid{lvl}_price expected 0.0 got {val}"

    def test_bid_levels_missing_zero_qty(self, result: pd.DataFrame) -> None:
        for lvl in range(4, 11):
            val = result[f"bid{lvl}_qty"].iloc[0]
            assert val == 0.0, f"bid{lvl}_qty expected 0.0 got {val}"

    def test_ask_levels_all_filled(self, result: pd.DataFrame) -> None:
        for lvl in range(1, 11):
            assert result[f"ask{lvl}_price"].iloc[0] > 0.0
            assert result[f"ask{lvl}_qty"].iloc[0] > 0.0

    def test_all_finite(self, result: pd.DataFrame) -> None:
        numeric = result.drop(columns=["ts_ms"])
        assert np.isfinite(numeric.to_numpy()).all(), "NaN or Inf found in output"


# ---------------------------------------------------------------------------
# Zero-bid-level snapshot (flash crash / feed outage)
# ---------------------------------------------------------------------------


class TestExpandObLevelsZeroBids:
    """Snapshot with 0 bid levels — all bid columns must be 0.0."""

    @pytest.fixture(scope="class")
    def result(self) -> pd.DataFrame:
        raw = _make_raw_ob([(0, 10)])
        return expand_ob_levels(raw)

    def test_all_bid_prices_zero(self, result: pd.DataFrame) -> None:
        for lvl in range(1, 11):
            assert result[f"bid{lvl}_price"].iloc[0] == 0.0

    def test_all_bid_qtys_zero(self, result: pd.DataFrame) -> None:
        for lvl in range(1, 11):
            assert result[f"bid{lvl}_qty"].iloc[0] == 0.0

    def test_all_finite(self, result: pd.DataFrame) -> None:
        numeric = result.drop(columns=["ts_ms"])
        assert np.isfinite(numeric.to_numpy()).all()


# ---------------------------------------------------------------------------
# Multi-row: mix of full and partial snapshots
# ---------------------------------------------------------------------------


class TestExpandObLevelsMixedRows:
    """Multiple rows with varying level counts — only finite output allowed."""

    @pytest.fixture(scope="class")
    def result(self) -> pd.DataFrame:
        raw = _make_raw_ob([(10, 10), (3, 10), (9, 5), (0, 0), (10, 10)])
        return expand_ob_levels(raw)

    def test_row_count(self, result: pd.DataFrame) -> None:
        assert len(result) == 5

    def test_all_finite(self, result: pd.DataFrame) -> None:
        numeric = result.drop(columns=["ts_ms"])
        assert np.isfinite(numeric.to_numpy()).all(), "NaN or Inf in mixed-row output"

    def test_full_row_unaffected(self, result: pd.DataFrame) -> None:
        """First row (10/10) must have all levels > 0."""
        row = result.iloc[0]
        for lvl in range(1, 11):
            assert row[f"bid{lvl}_price"] > 0.0
            assert row[f"ask{lvl}_price"] > 0.0

    def test_partial_bid_row_missing_zero(self, result: pd.DataFrame) -> None:
        """Row index 1 has 3 bids — levels 4-10 must be 0."""
        row = result.iloc[1]
        for lvl in range(4, 11):
            assert row[f"bid{lvl}_price"] == 0.0
            assert row[f"bid{lvl}_qty"] == 0.0

    def test_zero_zero_row_all_zero(self, result: pd.DataFrame) -> None:
        """Row index 3 (0 bid, 0 ask) — everything 0.0."""
        row = result.iloc[3]
        for lvl in range(1, 11):
            assert row[f"bid{lvl}_price"] == 0.0
            assert row[f"ask{lvl}_price"] == 0.0


# ---------------------------------------------------------------------------
# Full 10x10 snapshot is unaffected (no regression)
# ---------------------------------------------------------------------------


def test_full_snapshot_unaffected() -> None:
    """10-level snapshot must still produce all non-zero prices/qtys."""
    raw = _make_raw_ob([(10, 10)])
    result = expand_ob_levels(raw)
    numeric = result.drop(columns=["ts_ms"])
    assert np.isfinite(numeric.to_numpy()).all()
    for lvl in range(1, 11):
        assert result[f"bid{lvl}_price"].iloc[0] > 0.0
        assert result[f"ask{lvl}_price"].iloc[0] > 0.0
