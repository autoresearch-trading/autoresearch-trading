# tape/masking.py
"""MEM masking primitives.

Block masking covers most features (sequential signals).  Two features have
lag-5 autocorrelation > 0.8 (prev_seq_time_span, kyle_lambda) — for those, the
SPECIFICATION uses random-position masking via tape/constants.MEM_RANDOM_MASK_FEATURES,
but the MEM target mask EXCLUDES three carry-forward features entirely
(delta_imbalance_L1, kyle_lambda, cum_ofi_5) — gotcha #22.

Note: kyle_lambda appears in both lists.  The exclusion wins — kyle_lambda is
not a reconstruction target at all.  prev_seq_time_span is the only feature
that uses random-position masking AND is reconstructed.
"""

from __future__ import annotations

import numpy as np
import torch

from tape.constants import (
    FEATURE_NAMES,
    MEM_BLOCK_LEN,
    MEM_EXCLUDED_FEATURES,
    MEM_MASK_FRACTION,
)


def block_mask(
    *,
    window_len: int,
    block_len: int = MEM_BLOCK_LEN,
    fraction: float = MEM_MASK_FRACTION,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bool mask of length window_len with ~fraction*window_len True positions
    arranged in contiguous blocks of block_len.

    Strategy: choose ceil(fraction*window_len/block_len) random block starts
    in the valid range [0, window_len - block_len], without overlap.
    """
    n_blocks = max(1, int(np.ceil(fraction * window_len / block_len)))
    # Choose start positions that snap to block_len-aligned slots so blocks
    # never overlap.
    n_slots = window_len // block_len
    pick = rng.choice(n_slots, size=min(n_blocks, n_slots), replace=False)
    out = np.zeros(window_len, dtype=bool)
    for slot in pick:
        s = int(slot) * block_len
        out[s : s + block_len] = True
    return out


def random_mask(
    *,
    window_len: int,
    fraction: float = MEM_MASK_FRACTION,
    rng: np.random.Generator,
) -> np.ndarray:
    """I.i.d. Bernoulli(fraction) mask per position."""
    return rng.random(window_len) < fraction


def build_mem_target_mask() -> torch.Tensor:
    """Return bool[17] — True where the feature IS a reconstruction target."""
    excluded = set(MEM_EXCLUDED_FEATURES)
    return torch.tensor(
        [name not in excluded for name in FEATURE_NAMES],
        dtype=torch.bool,
    )
