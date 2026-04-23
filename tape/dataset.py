# tape/dataset.py
"""TapeDataset — PyTorch Dataset backed by cached .npz shards.

Each shard is one symbol-day produced by tape/cache.py.  The dataset turns
them into 200-event windows with a configurable stride, exposing feature
tensors, direction labels, and metadata.

Design notes:
  - Shards are loaded lazily and kept in a small LRU cache (default 8 entries)
    to avoid loading all 40 GB at once (data pipeline rule #5).
  - set_epoch() re-randomizes the per-shard random_offset for stride=50
    pretraining diversity (gotcha #16, CLAUDE.md).
  - Each shard is one day so day-boundary enforcement (gotcha #26) is
    handled automatically by build_window_starts.
  - symbol_id is looked up from SYMBOLS tuple for the equal-symbol sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from tape.cache import load_shard
from tape.constants import STRIDE_EVAL, STRIDE_PRETRAIN, SYMBOLS, WINDOW_LEN
from tape.windowing import build_window_starts

# ---------------------------------------------------------------------------
# WindowRef — lightweight record stored in the flat global index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowRef:
    """Identifies one 200-event window within a shard."""

    shard_path: Path
    start: int  # global index into shard arrays
    symbol: str
    date: str
    symbol_id: int


# ---------------------------------------------------------------------------
# TapeDataset
# ---------------------------------------------------------------------------


class TapeDataset(Dataset):
    """Indexable dataset over (symbol, day, window_start) tuples.

    Parameters
    ----------
    shard_paths : list[Path]
        Paths to .npz shards (one per symbol-day, from cache.py / save_shard).
    stride : int
        Step between consecutive windows. Use STRIDE_PRETRAIN=50 for
        pretraining, STRIDE_EVAL=200 for evaluation probes.
    mode : {"pretrain", "eval"}
        Convenience alias: "eval" overrides stride to STRIDE_EVAL.
    cache_size : int
        Number of shards to keep in the in-memory LRU cache.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        stride: int = STRIDE_PRETRAIN,
        mode: Literal["pretrain", "eval"] = "pretrain",
        cache_size: int = 8,
    ) -> None:
        if mode == "eval":
            stride = STRIDE_EVAL

        self.shard_paths: list[Path] = sorted(shard_paths)
        self.stride: int = stride
        self._epoch: int = 0
        self._cache: dict[Path, dict] = {}
        self._cache_order: list[Path] = []
        self._cache_size: int = cache_size

        # Flat list of (shard_path, start, symbol, date) for every window
        self._refs: list[WindowRef] = []
        self._build_index(random_offset=0)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self, random_offset: int) -> None:
        """Populate self._refs from all shards."""
        self._refs = []
        sym_lookup: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}

        for p in self.shard_paths:
            payload = self._load_shard_uncached(p)
            day_id: np.ndarray = payload["day_id"]
            sym = str(payload["symbol"])
            date = str(payload["date"])
            sym_id = sym_lookup.get(sym, -1)

            starts = build_window_starts(
                day_id,
                window_len=WINDOW_LEN,
                stride=self.stride,
                random_offset=random_offset,
            )
            for s in starts:
                self._refs.append(
                    WindowRef(
                        shard_path=p,
                        start=int(s),
                        symbol=sym,
                        date=date,
                        symbol_id=sym_id,
                    )
                )

    # ------------------------------------------------------------------
    # Epoch / offset control
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Re-randomize the per-window offset for stride=50 pretraining.

        Uses a deterministic seed so that the same epoch always produces the
        same window set.  Rebuilds the full index (cheap — just numpy arange).
        """
        self._epoch = epoch
        rng = np.random.default_rng(seed=epoch)
        offset = int(rng.integers(0, self.stride))
        self._build_index(random_offset=offset)

    # ------------------------------------------------------------------
    # Shard LRU cache
    # ------------------------------------------------------------------

    def _load_shard_uncached(self, path: Path) -> dict:
        """Load shard without touching the LRU cache (used during index build)."""
        return load_shard(path)

    def _get_shard(self, path: Path) -> dict:
        """Return shard dict, loading and caching on first access."""
        if path in self._cache:
            return self._cache[path]
        if len(self._cache) >= self._cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        payload = load_shard(path)
        self._cache[path] = payload
        self._cache_order.append(path)
        return payload

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, i: int) -> dict:
        ref = self._refs[i]
        payload = self._get_shard(ref.shard_path)
        s = ref.start
        e = s + WINDOW_LEN

        features = payload["features"][s:e].astype(np.float32)

        # Emit ts_first_ms so downstream probes / contrastive can compute real
        # UTC hour. Without this, `_collate` and `_run_probe_trio` in
        # scripts/run_pretrain.py silently fall back to 0 → the "hour_of_day"
        # probe measures event-index-mod-24, not wall-clock hour (council-5
        # Bug B diagnosis, 2026-04-23).
        ts_first_ms = int(payload["event_ts"][s]) if "event_ts" in payload else 0
        out: dict = {
            "features": torch.from_numpy(features),
            "symbol": ref.symbol,
            "date": ref.date,
            "symbol_id": ref.symbol_id,
            "start": s,
            "ts_first_ms": ts_first_ms,
        }

        # Direction labels: use the last event of the window (e-1) as the label
        # position (gotcha: "label at end of window", plan Task 8 note).
        label_idx = e - 1
        for h in (10, 50, 100, 500):
            dir_key = f"dir_h{h}"
            mask_key = f"dir_mask_h{h}"
            n_arr = len(payload[dir_key])
            out[f"label_h{h}"] = (
                int(payload[dir_key][label_idx]) if label_idx < n_arr else 0
            )
            out[f"label_h{h}_mask"] = (
                bool(payload[mask_key][label_idx]) if label_idx < n_arr else False
            )

        return out
