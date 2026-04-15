# tape/splits.py
"""Walk-forward cross-validation with a 600-event embargo (CLAUDE.md gotcha #12).

For a symbol's event-ordered cache, yield (train_idx, test_idx) where the test
window immediately follows train with at least `embargo` events of gap.

K-fold expanding window:
    train grows each fold; test window slides forward.

Rolling window (walk_forward_folds only):
    fixed-size train window slides forward.

Embargo rationale
-----------------
Labels at H500 use future events 500 ahead.  If training ends at event t,
testing must start at t + 500 + 100 = t + 600 so no training sample's label
overlaps a test sample's feature window (spec §Fine-Tuning; CLAUDE.md gotcha #12).
"""

from __future__ import annotations

from typing import Iterator, Literal

import numpy as np

from tape.constants import EMBARGO_EVENTS


def walk_forward_splits(
    n_events: int,
    *,
    k: int = 5,
    embargo: int = EMBARGO_EVENTS,
    min_train: int = 10_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) event-index arrays for each of k folds.

    This is an *expanding* walk-forward generator.  Each successive fold adds
    more training events; the test window slides forward without overlap.

    If n_events is too small to produce even one fold the generator yields
    nothing (silent skip — callers should check for zero folds).

    Parameters
    ----------
    n_events:
        Total number of events in the sequence.
    k:
        Number of folds.
    embargo:
        Minimum number of events between max(train) and min(test).
    min_train:
        Minimum events required in the first training set.
    """
    # Per plan reference: return early rather than raise when too small.
    if n_events < min_train + embargo + k * 1_000:
        return

    test_len = (n_events - min_train - embargo) // k

    for fi in range(k):
        test_start = min_train + embargo + fi * test_len
        test_end = test_start + test_len
        train = np.arange(0, test_start - embargo, dtype=np.int64)
        test = np.arange(test_start, test_end, dtype=np.int64)
        yield train, test


def walk_forward_folds(
    event_ts: np.ndarray,
    *,
    n_folds: int,
    embargo: int = EMBARGO_EVENTS,
    mode: Literal["expanding", "rolling"] = "expanding",
    min_train: int = 10_000,
    min_test: int = 1_000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) pairs for walk-forward evaluation.

    Parameters
    ----------
    event_ts:
        1-D int64 array of event timestamps (or a positional arange).  Only the
        length is used; the function works on positional indices so it is safe
        to pass ``np.arange(n)`` for synthetic data.
    n_folds:
        Number of folds to generate.
    embargo:
        Events to leave between max(train) and min(test).  Must satisfy
        ``min(test) - max(train) > embargo`` for every fold.
    mode:
        ``"expanding"`` — train window grows each fold (default).
        ``"rolling"``   — train window is fixed-size and slides forward.
    min_train:
        Minimum number of events required in each training set.
    min_test:
        Minimum number of events required in each test set.

    Returns
    -------
    list of (train_idx, test_idx) tuples; each array is int64.

    Raises
    ------
    ValueError
        If ``len(event_ts)`` is insufficient to build ``n_folds`` folds with
        the requested embargo, min_train, and min_test constraints.
    """
    n = len(event_ts)

    # Minimum total events needed:
    #   min_train + embargo + n_folds * min_test
    # (expanding: train of the last fold grows, but the minimum is the first)
    needed = min_train + embargo + n_folds * min_test
    if n < needed:
        raise ValueError(
            f"insufficient events: need >= {needed} for {n_folds} folds "
            f"(min_train={min_train}, embargo={embargo}, min_test={min_test}), "
            f"got {n}"
        )

    # Partition the region *after* min_train + embargo into n_folds test windows.
    available_for_test = n - min_train - embargo
    test_len = available_for_test // n_folds  # floor division — equal-size test windows

    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fi in range(n_folds):
        test_start = min_train + embargo + fi * test_len
        test_end = test_start + test_len

        if mode == "expanding":
            # Train = everything before the embargo zone of this test window.
            train_end = test_start - embargo  # exclusive upper bound
            train = np.arange(0, train_end, dtype=np.int64)
        else:  # rolling
            # Fixed-size train window ending just before the embargo zone.
            train_end = test_start - embargo
            train_start = max(0, train_end - min_train)
            train = np.arange(train_start, train_end, dtype=np.int64)

        test = np.arange(test_start, test_end, dtype=np.int64)

        # --- Internal embargo assertion (gotcha #12) ---
        # The prohibited zone for training indices is [test_start - embargo, test_end).
        prohibited_start = int(test.min()) - embargo
        overlapping = train[(train >= prohibited_start) & (train <= int(test.max()))]
        assert len(overlapping) == 0, (
            f"fold {fi}: {len(overlapping)} train indices fall inside the "
            f"embargo zone [{prohibited_start}, {int(test.max())}]"
        )

        # Sanity: gap must be strictly greater than embargo.
        gap = int(test.min()) - int(train.max())
        assert gap > embargo, f"fold {fi}: gap={gap} is not > embargo={embargo}"

        folds.append((train, test))

    return folds
