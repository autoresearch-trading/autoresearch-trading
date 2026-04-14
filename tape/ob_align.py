"""Align trade/event timestamps to the nearest prior orderbook snapshot.

See CLAUDE.md gotcha #2: `np.searchsorted(ob_ts, trade_ts, side="right") - 1`,
vectorised. Never use a Python for-loop.
"""

from __future__ import annotations

import numpy as np


def align_events_to_ob(event_ts: np.ndarray, ob_ts: np.ndarray) -> np.ndarray:
    """Return idx such that `ob_ts[idx[i]]` is the latest OB snapshot at or
    before `event_ts[i]`. Events before the first snapshot get idx = -1; the
    caller must mask them (or drop them).

    Parameters
    ----------
    event_ts : int64 array, shape (n_events,)
    ob_ts    : int64 array, shape (n_snapshots,), must be non-decreasing

    Returns
    -------
    idx : int64 array, shape (n_events,), values in [-1, n_snapshots - 1]
    """
    if ob_ts.ndim != 1 or event_ts.ndim != 1:
        raise ValueError("both inputs must be 1-D")
    if ob_ts.size > 1 and np.any(np.diff(ob_ts) < 0):
        raise ValueError("ob_ts must be non-decreasing")
    idx = np.searchsorted(ob_ts, event_ts, side="right") - 1
    return idx.astype(np.int64)
