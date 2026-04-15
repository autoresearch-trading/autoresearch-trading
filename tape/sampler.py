# tape/sampler.py
"""Equal-symbol sampler for balanced pretraining minibatches (gotcha #27).

BTC contributes ~133K windows at stride=50; illiquid symbols contribute ~2K.
Without rebalancing, BTC dominates and the encoder learns BTC-specific patterns
rather than universal tape representations.

Strategy: at the start of each epoch, for each symbol sample
  min(n_windows_this_symbol, target_per_symbol)
window indices without replacement.  Concatenate + shuffle.

target_per_symbol defaults to the minimum group size (strictest equalisation).
A configurable value allows relaxed equalisation (e.g. median group size).
"""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler

from tape.dataset import TapeDataset


class EqualSymbolSampler(Sampler[int]):
    """Round-robin by symbol, random within symbol, per epoch.

    Parameters
    ----------
    dataset : TapeDataset
        The underlying dataset (must have _refs populated).
    target_per_symbol : int | None
        Maximum windows to draw per symbol per epoch.  None (default) uses the
        minimum group size so all symbols are perfectly balanced.
    seed : int
        Base random seed.  set_epoch(n) incorporates epoch into the seed for
        per-epoch diversity while remaining deterministic.
    """

    def __init__(
        self,
        dataset: TapeDataset,
        *,
        target_per_symbol: int | None = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.seed = seed
        self._epoch: int = 0
        self._target_per_symbol: int | None = target_per_symbol

        # Group global indices by symbol
        self._by_symbol: dict[str, list[int]] = {}
        for i, ref in enumerate(dataset._refs):
            self._by_symbol.setdefault(ref.symbol, []).append(i)

    def _effective_target(self) -> int:
        """Compute the actual number of windows drawn per symbol this epoch."""
        min_group = min(len(v) for v in self._by_symbol.values())
        if self._target_per_symbol is None:
            return min_group
        return min(self._target_per_symbol, min_group)

    def __len__(self) -> int:
        """Number of indices yielded per epoch: target * num_symbols."""
        return self._effective_target() * len(self._by_symbol)

    def set_epoch(self, epoch: int) -> None:
        """Update epoch so that __iter__ produces a different permutation."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Combine base seed with epoch for deterministic per-epoch diversity
        rng = random.Random(self.seed ^ (self._epoch * 2654435761))

        target = self._effective_target()

        # For each symbol: shuffle the full index list, take the first `target`
        symbol_pools: dict[str, list[int]] = {}
        for sym, indices in self._by_symbol.items():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            symbol_pools[sym] = shuffled[:target]

        # Interleave: round-robin across symbols (shuffled symbol order per round)
        syms = list(symbol_pools.keys())
        result: list[int] = []
        for i in range(target):
            order = syms.copy()
            rng.shuffle(order)
            for s in order:
                result.append(symbol_pools[s][i])

        # Final global shuffle so symbol order is not predictable within a batch
        rng.shuffle(result)
        yield from result
