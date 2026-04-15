# tape/windowing.py
"""Window index generation for the tape pipeline.

Produces int64 arrays of window START indices respecting:
  - Configurable stride and window length.
  - Day-boundary enforcement (gotcha #26): no window may cross a day boundary.
  - Random offset in [0, stride) per epoch for stride=50 pretraining diversity.

Each shard is one symbol-day so in the normal case day_id is constant, but
multi-day arrays are also supported for multi-shard Dataset usage.
"""

from __future__ import annotations

import numpy as np

from tape.constants import STRIDE_PRETRAIN, WINDOW_LEN


def build_window_starts(
    day_id: np.ndarray,
    *,
    window_len: int = WINDOW_LEN,
    stride: int = STRIDE_PRETRAIN,
    random_offset: int = 0,
) -> np.ndarray:
    """Return int64 array of window start indices respecting day boundaries.

    For each contiguous segment of equal ``day_id`` values, enumerate start
    indices ``s`` such that:
      - ``s + window_len <= segment_end``  (window fits within the segment)
      - ``day_id[s:s+window_len]`` is all the same day (guaranteed by construction)

    The ``random_offset`` (reduced modulo ``stride``) shifts the first window
    within EACH day segment.  This provides per-epoch diversity during
    pretraining without introducing lookahead (gotcha #26).

    Parameters
    ----------
    day_id : int64 array, shape (N,)
        Day identifier per event (days since epoch, same as cache.py output).
    window_len : int
        Number of events per window. Default WINDOW_LEN=200.
    stride : int
        Step between consecutive windows. Default STRIDE_PRETRAIN=50.
    random_offset : int
        Offset in [0, stride) applied to the first window of each day segment.
        Values >= stride are reduced modulo stride.

    Returns
    -------
    np.ndarray, dtype=int64
        Sorted array of window start indices (global, into the full day_id array).
    """
    if len(day_id) == 0:
        return np.zeros(0, dtype=np.int64)

    offset = int(random_offset) % stride
    all_starts: list[np.ndarray] = []

    # Identify contiguous day segments via the change-points in day_id.
    # np.where gives indices where day_id differs from the previous element.
    change_pts = np.where(np.diff(day_id))[0] + 1  # positions where a new day begins
    seg_starts = np.concatenate([[0], change_pts])
    seg_ends = np.concatenate([change_pts, [len(day_id)]])

    for seg_s, seg_e in zip(seg_starts, seg_ends):
        seg_len = int(seg_e - seg_s)
        if seg_len < window_len:
            continue
        first_win = int(seg_s) + offset
        last_win = int(seg_e) - window_len  # last valid start (inclusive)
        if first_win > last_win:
            continue
        seg_window_starts = np.arange(first_win, last_win + 1, stride, dtype=np.int64)
        all_starts.append(seg_window_starts)

    if not all_starts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(all_starts)


def window_view(
    features: np.ndarray,
    start: int,
    window_len: int = WINDOW_LEN,
) -> np.ndarray:
    """Return a zero-copy view of features[start : start + window_len].

    Parameters
    ----------
    features : np.ndarray, shape (N, F)
        Full feature matrix for a shard.
    start : int
        Window start index.
    window_len : int
        Window length (default WINDOW_LEN=200).

    Returns
    -------
    np.ndarray, shape (window_len, F)
        A numpy view (no copy) for memory efficiency.
    """
    return features[start : start + window_len]
