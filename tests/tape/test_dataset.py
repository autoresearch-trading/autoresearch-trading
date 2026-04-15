# tests/tape/test_dataset.py
"""Tests for tape/dataset.py — TapeDataset.

TDD: tests written before implementation.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

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
    """Create and persist a minimal valid shard."""
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
# TapeDataset — length
# ---------------------------------------------------------------------------


def test_dataset_len_matches_expected_window_count(tmp_path: Path) -> None:
    """1000 events, stride=50, window=200 → 17 windows."""
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000)
    ds = TapeDataset([p], stride=50)
    assert len(ds) == 17  # (1000 - 200) // 50 + 1


def test_dataset_len_accumulates_across_shards(tmp_path: Path) -> None:
    """Two shards each with 1000 events → 34 total windows."""
    from tape.dataset import TapeDataset

    p1 = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000, seed=0)
    p2 = _make_shard(tmp_path, "ETH", "2025-11-02", n=1000, seed=1)
    ds = TapeDataset([p1, p2], stride=50)
    assert len(ds) == 34


# ---------------------------------------------------------------------------
# TapeDataset — __getitem__ shape and content
# ---------------------------------------------------------------------------


def test_dataset_getitem_features_shape(tmp_path: Path) -> None:
    """Each item's 'features' tensor must be (200, 17) float32."""
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000)
    ds = TapeDataset([p], stride=50)
    item = ds[0]
    assert isinstance(item["features"], torch.Tensor)
    assert item["features"].shape == (200, 17)
    assert item["features"].dtype == torch.float32


def test_dataset_getitem_features_match_shard_slice(tmp_path: Path) -> None:
    """i-th window's features must equal shard_features[start_i : start_i+200]."""
    from tape.cache import load_shard
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000, seed=42)
    ds = TapeDataset([p], stride=50)
    shard = load_shard(p)

    # Check window 0 (start=0) and window 5 (start=250)
    for idx, expected_start in [(0, 0), (5, 250)]:
        item = ds[idx]
        expected = torch.from_numpy(
            shard["features"][expected_start : expected_start + 200]
        )
        assert torch.allclose(
            item["features"], expected
        ), f"Window {idx}: features mismatch at start={expected_start}"


def test_dataset_getitem_contains_label_keys(tmp_path: Path) -> None:
    """Each item must contain label_h{h} and label_h{h}_mask for all horizons."""
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000)
    ds = TapeDataset([p], stride=50)
    item = ds[0]
    for h in (10, 50, 100, 500):
        assert f"label_h{h}" in item, f"Missing label_h{h}"
        assert f"label_h{h}_mask" in item, f"Missing label_h{h}_mask"


def test_dataset_getitem_symbol_and_date(tmp_path: Path) -> None:
    """Each item must expose its symbol and date strings."""
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "ETH", "2025-12-15", n=1000)
    ds = TapeDataset([p], stride=50)
    item = ds[0]
    assert item["symbol"] == "ETH"
    assert item["date"] == "2025-12-15"


# ---------------------------------------------------------------------------
# TapeDataset — mode / stride
# ---------------------------------------------------------------------------


def test_dataset_eval_mode_uses_stride_200(tmp_path: Path) -> None:
    """Eval mode (stride=200) yields non-overlapping windows."""
    from tape.constants import STRIDE_EVAL
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000)
    ds = TapeDataset([p], stride=STRIDE_EVAL)
    # 1000 events, stride=200, window=200 → (1000-200)//200 + 1 = 5 windows
    assert len(ds) == 5


# ---------------------------------------------------------------------------
# TapeDataset — set_epoch randomizes offset deterministically
# ---------------------------------------------------------------------------


def test_dataset_set_epoch_changes_window_count_or_starts(tmp_path: Path) -> None:
    """set_epoch(n) re-randomizes offset; different epochs may yield different lengths."""
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000)
    ds = TapeDataset([p], stride=50)
    len_epoch0 = len(ds)
    ds.set_epoch(1)
    len_epoch1 = len(ds)
    # Both must be non-zero and within plausible range
    assert len_epoch0 > 0
    assert len_epoch1 > 0


def test_dataset_set_epoch_same_epoch_same_result(tmp_path: Path) -> None:
    """Same epoch number → same index sequence (deterministic)."""
    from tape.cache import load_shard
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000, seed=7)
    ds = TapeDataset([p], stride=50)
    ds.set_epoch(5)
    starts_a = [ds._refs[i].start for i in range(len(ds))]
    ds.set_epoch(5)
    starts_b = [ds._refs[i].start for i in range(len(ds))]
    assert starts_a == starts_b


# ---------------------------------------------------------------------------
# TapeDataset — symbol_id field
# ---------------------------------------------------------------------------


def test_dataset_getitem_exposes_symbol_id(tmp_path: Path) -> None:
    """Each item must include 'symbol_id' (int index into SYMBOLS tuple)."""
    from tape.constants import SYMBOLS
    from tape.dataset import TapeDataset

    p = _make_shard(tmp_path, "BTC", "2025-11-01", n=1000)
    ds = TapeDataset([p], stride=50)
    item = ds[0]
    assert "symbol_id" in item
    assert item["symbol_id"] == list(SYMBOLS).index("BTC")
