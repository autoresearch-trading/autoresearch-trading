# tape/contrastive_batch.py
"""Cross-symbol contrastive pairing for SimCLR (spec §Training).

Same-date, same-UTC-hour windows from the 6 LIQUID_CONTRASTIVE_SYMBOLS
(BTC, ETH, SOL, BNB, LINK, LTC) get a soft-positive weight 0.5 in the
NT-Xent loss.  AVAX is the Gate 3 held-out symbol — must NEVER appear
in pairs (gotcha #25).
"""

from __future__ import annotations

import numpy as np

from tape.constants import HELD_OUT_SYMBOL, LIQUID_CONTRASTIVE_SYMBOLS


def hour_bucket_from_ms(ts_ms: np.ndarray) -> np.ndarray:
    """Map ms timestamps to UTC hour (0–23)."""
    seconds = ts_ms // 1_000
    return ((seconds // 3_600) % 24).astype(np.int64)


def build_soft_positive_matrix(
    symbols: np.ndarray,  # (B,) of str
    dates: np.ndarray,  # (B,) of "YYYY-MM-DD" str
    hours: np.ndarray,  # (B,) of int 0..23
    eligible_mask: np.ndarray,  # (B,) of bool — True if symbol is in liquid set AND not AVAX
) -> np.ndarray:
    """Return (B, B) {0, 1} matrix of cross-symbol same-date-same-hour pairs.

    Diagonal is always zero.  Pairs only between distinct eligible symbols.
    AVAX is rejected here as defense in depth even if the dataset already
    excludes it.
    """
    B = len(symbols)
    # Hard reject AVAX regardless of caller's eligible_mask
    safe_mask = eligible_mask & (symbols != HELD_OUT_SYMBOL)
    out = np.zeros((B, B), dtype=np.float32)
    for i in range(B):
        if not safe_mask[i]:
            continue
        for j in range(B):
            if i == j or not safe_mask[j]:
                continue
            if symbols[i] == symbols[j]:
                continue
            if dates[i] != dates[j]:
                continue
            if hours[i] != hours[j]:
                continue
            out[i, j] = 1.0
    return out


def is_eligible_for_contrastive(symbol: str) -> bool:
    return symbol in LIQUID_CONTRASTIVE_SYMBOLS and symbol != HELD_OUT_SYMBOL
