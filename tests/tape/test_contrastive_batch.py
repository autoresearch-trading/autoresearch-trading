# tests/tape/test_contrastive_batch.py
import numpy as np
import pytest

from tape.constants import HELD_OUT_SYMBOL, LIQUID_CONTRASTIVE_SYMBOLS
from tape.contrastive_batch import build_soft_positive_matrix, hour_bucket_from_ms


def test_hour_bucket_basic():
    # 1700000000000 ms -> some UTC hour 0..23
    h = hour_bucket_from_ms(np.array([1700000000000], dtype=np.int64))
    assert 0 <= int(h[0]) <= 23


def test_soft_positive_matrix_pairs_same_date_same_hour_liquid_symbols():
    # Build a fake batch metadata: 4 windows, 3 in liquid symbols same date+hour, 1 in illiquid
    symbols = np.array(["BTC", "ETH", "SOL", "DOGE"])
    dates = np.array(["2026-02-01"] * 4)
    hours = np.array([10, 10, 10, 10], dtype=np.int64)
    sym_match = np.array([s in LIQUID_CONTRASTIVE_SYMBOLS for s in symbols])
    soft = build_soft_positive_matrix(symbols, dates, hours, sym_match)
    assert soft.shape == (4, 4)
    # BTC <-> ETH <-> SOL all paired (off-diagonal 1s)
    assert soft[0, 1] == 1
    assert soft[1, 0] == 1
    assert soft[0, 2] == 1
    # No self-pairs
    assert soft[0, 0] == 0
    assert soft[1, 1] == 0
    # DOGE is not in LIQUID_CONTRASTIVE_SYMBOLS -> no pairs touching index 3
    assert soft[3].sum() == 0
    assert soft[:, 3].sum() == 0


def test_avax_rejected_from_soft_positives():
    """Even if metadata claims AVAX, the helper must drop it (defense in depth)."""
    symbols = np.array(["BTC", "AVAX", "ETH"])
    dates = np.array(["2026-02-01"] * 3)
    hours = np.array([10, 10, 10], dtype=np.int64)
    sym_match = np.array(
        [s in LIQUID_CONTRASTIVE_SYMBOLS and s != HELD_OUT_SYMBOL for s in symbols]
    )
    soft = build_soft_positive_matrix(symbols, dates, hours, sym_match)
    # AVAX (index 1) must have zero rows and columns
    assert soft[1].sum() == 0
    assert soft[:, 1].sum() == 0
    # BTC <-> ETH still paired
    assert soft[0, 2] == 1


def test_pairs_only_within_same_date_hour():
    symbols = np.array(["BTC", "ETH", "BTC", "ETH"])
    dates = np.array(["2026-02-01", "2026-02-01", "2026-02-02", "2026-02-02"])
    hours = np.array([10, 10, 10, 10], dtype=np.int64)
    sym_match = np.array([True, True, True, True])
    soft = build_soft_positive_matrix(symbols, dates, hours, sym_match)
    # Date 1 pair: 0 <-> 1
    assert soft[0, 1] == 1
    # Date 2 pair: 2 <-> 3
    assert soft[2, 3] == 1
    # Cross-date should be zero
    assert soft[0, 2] == 0
    assert soft[1, 3] == 0
