# tests/tape/test_sampler.py
"""Tests for tape/sampler.py — EqualSymbolSampler.

TDD: tests written before implementation.
"""
from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tape.cache import save_shard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime.date(1970, 1, 1)


def _make_shard(
    tmp_path: Path,
    sym: str,
    date_str: str,
    n: int,
    seed: int = 0,
) -> Path:
    rng = np.random.RandomState(seed)
    day_id_int = (datetime.date.fromisoformat(date_str) - _EPOCH).days
    shard = {
        "features": rng.randn(n, 17).astype(np.float32),
        "event_ts": np.arange(n, dtype=np.int64),
        "day_id": np.full(n, day_id_int, dtype=np.int64),
        "directions": {f"h{h}": np.zeros(n, dtype=np.int8) for h in (10, 50, 100, 500)}
        | {f"mask_h{h}": np.ones(n, dtype=bool) for h in (10, 50, 100, 500)},
        "wyckoff": {
            k: np.zeros(n, dtype=np.int8)
            for k in ("stress", "informed_flow", "climax", "spring", "absorption")
        },
        "symbol": sym,
        "date": date_str,
        "schema_version": 1,
    }
    out_dir = tmp_path / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_shard(shard, out_dir)


# ---------------------------------------------------------------------------
# EqualSymbolSampler — basic round-robin
# ---------------------------------------------------------------------------


def test_equal_symbol_sampler_round_robins_three_equal_groups(tmp_path: Path) -> None:
    """BTC/ETH/SOL each with 1000 events → 17 windows each → 51 total per epoch."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    paths = [
        _make_shard(tmp_path, sym, "2025-11-01", n=1000, seed=i)
        for i, sym in enumerate(["BTC", "ETH", "SOL"])
    ]
    ds = TapeDataset(paths, stride=50)
    sampler = EqualSymbolSampler(ds, seed=0)

    seq = list(sampler)
    assert len(seq) == 51  # min_group=17, num_symbols=3

    # Each symbol must appear exactly 17 times
    counts = Counter(ds._refs[i].symbol for i in seq)
    for sym in ("BTC", "ETH", "SOL"):
        assert counts[sym] == 17, f"{sym}: expected 17, got {counts[sym]}"


def test_equal_symbol_sampler_caps_large_group(tmp_path: Path) -> None:
    """Symbol A (1000 events→17 windows) and B (200 events→1 window).

    min_group=1, so sampler yields 1*2=2 total (one per symbol per epoch).
    """
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    p_a = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000, seed=0)
    p_b = _make_shard(tmp_path, "ETH", "2025-11-02", n=200, seed=1)
    ds = TapeDataset([p_a, p_b], stride=50)
    sampler = EqualSymbolSampler(ds, seed=0)

    seq = list(sampler)
    counts = Counter(ds._refs[i].symbol for i in seq)
    # Each symbol appears min_group times (1)
    assert counts["BTC"] == counts["ETH"]
    assert len(seq) == counts["BTC"] + counts["ETH"]


def test_equal_symbol_sampler_with_target_per_symbol(tmp_path: Path) -> None:
    """target_per_symbol=5 caps large group to 5 while not inflating small group.

    A has 1000 events (17 windows), B has 400 events (5 windows).
    With target_per_symbol=5: A yields 5, B yields 5 → 10 total.
    """
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    p_a = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000, seed=0)
    p_b = _make_shard(tmp_path, "ETH", "2025-11-02", n=400, seed=1)
    ds = TapeDataset([p_a, p_b], stride=50)
    sampler = EqualSymbolSampler(ds, seed=0, target_per_symbol=5)

    seq = list(sampler)
    counts = Counter(ds._refs[i].symbol for i in seq)
    assert counts["BTC"] == 5
    assert counts["ETH"] == 5
    assert len(seq) == 10


def test_equal_symbol_sampler_target_does_not_exceed_available(tmp_path: Path) -> None:
    """target_per_symbol larger than available windows → capped at available."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    # 200 events → exactly 1 window at stride=50
    p_a = _make_shard(tmp_path, "BTC", "2025-11-01", n=200, seed=0)
    p_b = _make_shard(tmp_path, "ETH", "2025-11-02", n=1000, seed=1)
    ds = TapeDataset([p_a, p_b], stride=50)
    # target=50 but BTC only has 1 window
    sampler = EqualSymbolSampler(ds, seed=0, target_per_symbol=50)
    seq = list(sampler)
    counts = Counter(ds._refs[i].symbol for i in seq)
    assert counts["BTC"] == 1
    assert counts["ETH"] == 1


# ---------------------------------------------------------------------------
# EqualSymbolSampler — determinism
# ---------------------------------------------------------------------------


def test_equal_symbol_sampler_same_seed_same_sequence(tmp_path: Path) -> None:
    """Same seed → identical index sequence across two sampler instances."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    paths = [
        _make_shard(tmp_path, sym, "2025-11-01", n=1000, seed=i)
        for i, sym in enumerate(["BTC", "ETH"])
    ]
    ds = TapeDataset(paths, stride=50)
    seq_a = list(EqualSymbolSampler(ds, seed=42))
    seq_b = list(EqualSymbolSampler(ds, seed=42))
    assert seq_a == seq_b


def test_equal_symbol_sampler_different_seeds_different_sequence(
    tmp_path: Path,
) -> None:
    """Different seeds → different orderings (not guaranteed but statistically expected)."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    paths = [
        _make_shard(tmp_path, sym, "2025-11-01", n=1000, seed=i)
        for i, sym in enumerate(["BTC", "ETH"])
    ]
    ds = TapeDataset(paths, stride=50)
    seq_a = list(EqualSymbolSampler(ds, seed=0))
    seq_b = list(EqualSymbolSampler(ds, seed=999))
    # With 17 windows per symbol, the probability of identical sequences is negligible
    assert seq_a != seq_b


# ---------------------------------------------------------------------------
# EqualSymbolSampler — set_epoch
# ---------------------------------------------------------------------------


def test_equal_symbol_sampler_set_epoch_same_epoch_same_sequence(
    tmp_path: Path,
) -> None:
    """set_epoch(n) is deterministic: calling twice yields the same sequence."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    paths = [
        _make_shard(tmp_path, sym, "2025-11-01", n=1000, seed=i)
        for i, sym in enumerate(["BTC", "ETH"])
    ]
    ds = TapeDataset(paths, stride=50)
    sampler = EqualSymbolSampler(ds, seed=7)
    sampler.set_epoch(3)
    seq_a = list(sampler)
    sampler.set_epoch(3)
    seq_b = list(sampler)
    assert seq_a == seq_b


def test_equal_symbol_sampler_set_epoch_different_epochs_differ(tmp_path: Path) -> None:
    """Different epoch numbers produce different permutations."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    paths = [
        _make_shard(tmp_path, sym, "2025-11-01", n=1000, seed=i)
        for i, sym in enumerate(["BTC", "ETH"])
    ]
    ds = TapeDataset(paths, stride=50)
    sampler = EqualSymbolSampler(ds, seed=0)
    sampler.set_epoch(0)
    seq_0 = list(sampler)
    sampler.set_epoch(1)
    seq_1 = list(sampler)
    assert seq_0 != seq_1


# ---------------------------------------------------------------------------
# EqualSymbolSampler — __len__
# ---------------------------------------------------------------------------


def test_equal_symbol_sampler_len_matches_yielded(tmp_path: Path) -> None:
    """__len__ must equal the actual number of indices yielded."""
    from tape.dataset import TapeDataset
    from tape.sampler import EqualSymbolSampler

    paths = [
        _make_shard(tmp_path, sym, "2025-11-01", n=1000, seed=i)
        for i, sym in enumerate(["BTC", "ETH", "SOL"])
    ]
    ds = TapeDataset(paths, stride=50)
    sampler = EqualSymbolSampler(ds, seed=0)
    assert len(sampler) == len(list(sampler))
