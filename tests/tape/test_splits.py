# tests/tape/test_splits.py
"""Tests for walk-forward split generators with 600-event embargo.

RED phase: all tests are written before any production code exists.
"""
from __future__ import annotations

import numpy as np
import pytest

# Import both the plan's generator API and the task's list API.
from tape.splits import walk_forward_folds, walk_forward_splits

# ---------------------------------------------------------------------------
# walk_forward_splits (generator, per plan reference)
# ---------------------------------------------------------------------------


class TestWalkForwardSplits:
    """Tests for the generator-style API."""

    def test_returns_k_folds(self):
        folds = list(walk_forward_splits(50_000, k=5, embargo=600, min_train=10_000))
        assert len(folds) == 5

    def test_no_overlap_between_train_and_test(self):
        for train, test in walk_forward_splits(
            50_000, k=5, embargo=600, min_train=10_000
        ):
            assert set(train.tolist()).isdisjoint(set(test.tolist()))

    def test_embargo_respected(self):
        """Gap between max(train) and min(test) must be >= embargo."""
        for train, test in walk_forward_splits(
            50_000, k=5, embargo=600, min_train=10_000
        ):
            gap = int(test.min()) - int(train.max())
            assert gap > 600, f"Gap {gap} not > 600"

    def test_train_sizes_increase_monotonically(self):
        """Expanding window: each fold has more training data than the previous."""
        sizes = [
            len(tr)
            for tr, _ in walk_forward_splits(50_000, k=5, embargo=600, min_train=10_000)
        ]
        for i in range(1, len(sizes)):
            assert (
                sizes[i] >= sizes[i - 1]
            ), f"fold {i} train size {sizes[i]} < fold {i-1} size {sizes[i-1]}"

    def test_no_empty_fold(self):
        for train, test in walk_forward_splits(
            50_000, k=5, embargo=600, min_train=10_000
        ):
            assert len(train) > 0, "train is empty"
            assert len(test) > 0, "test is empty"

    def test_indices_are_int64(self):
        for train, test in walk_forward_splits(
            50_000, k=5, embargo=600, min_train=10_000
        ):
            assert train.dtype == np.int64, f"train dtype {train.dtype}"
            assert test.dtype == np.int64, f"test dtype {test.dtype}"

    def test_indices_within_bounds(self):
        n = 50_000
        for train, test in walk_forward_splits(n, k=5, embargo=600, min_train=10_000):
            assert int(train.min()) >= 0
            assert int(train.max()) < n
            assert int(test.min()) >= 0
            assert int(test.max()) < n

    def test_too_small_returns_no_folds(self):
        """When n_events is too small, the generator yields nothing (plan says 'return')."""
        folds = list(walk_forward_splits(100, k=5, embargo=600, min_train=10_000))
        assert folds == []

    def test_different_k_values(self):
        for k in (2, 3, 10):
            folds = list(
                walk_forward_splits(200_000, k=k, embargo=600, min_train=10_000)
            )
            assert len(folds) == k, f"expected {k} folds, got {len(folds)}"

    def test_custom_embargo(self):
        """Verify embargo parameter is honored."""
        embargo = 1200
        for train, test in walk_forward_splits(
            100_000, k=3, embargo=embargo, min_train=10_000
        ):
            gap = int(test.min()) - int(train.max())
            assert gap > embargo, f"gap {gap} not > {embargo}"

    def test_test_windows_do_not_overlap_across_folds(self):
        """Test sets from different folds should be disjoint (no data leakage)."""
        all_test_idx: set[int] = set()
        for _, test in walk_forward_splits(50_000, k=5, embargo=600, min_train=10_000):
            fold_set = set(test.tolist())
            assert all_test_idx.isdisjoint(
                fold_set
            ), "test indices overlap across folds"
            all_test_idx |= fold_set


# ---------------------------------------------------------------------------
# walk_forward_folds (list API, per task deliverable)
# ---------------------------------------------------------------------------


class TestWalkForwardFolds:
    """Tests for the list-returning API with expanding + rolling modes."""

    def test_expanding_returns_n_folds(self):
        folds = walk_forward_folds(np.arange(50_000, dtype=np.int64), n_folds=5)
        assert len(folds) == 5

    def test_rolling_returns_n_folds(self):
        folds = walk_forward_folds(
            np.arange(50_000, dtype=np.int64), n_folds=5, mode="rolling"
        )
        assert len(folds) == 5

    def test_expanding_embargo_respected(self):
        folds = walk_forward_folds(
            np.arange(50_000, dtype=np.int64), n_folds=5, embargo=600
        )
        for i, (train, test) in enumerate(folds):
            gap = int(test.min()) - int(train.max())
            assert gap > 600, f"fold {i}: gap {gap} not > 600"

    def test_rolling_embargo_respected(self):
        folds = walk_forward_folds(
            np.arange(50_000, dtype=np.int64), n_folds=5, embargo=600, mode="rolling"
        )
        for i, (train, test) in enumerate(folds):
            gap = int(test.min()) - int(train.max())
            assert gap > 600, f"fold {i}: gap {gap} not > 600"

    def test_expanding_train_sizes_increase(self):
        folds = walk_forward_folds(np.arange(50_000, dtype=np.int64), n_folds=5)
        sizes = [len(tr) for tr, _ in folds]
        for i in range(1, len(sizes)):
            assert (
                sizes[i] >= sizes[i - 1]
            ), f"fold {i} train {sizes[i]} < fold {i-1} {sizes[i-1]}"

    def test_rolling_train_sizes_equal(self):
        folds = walk_forward_folds(
            np.arange(50_000, dtype=np.int64), n_folds=5, mode="rolling"
        )
        sizes = [len(tr) for tr, _ in folds]
        assert len(set(sizes)) == 1, f"rolling train sizes not equal: {sizes}"

    def test_no_empty_fold(self):
        folds = walk_forward_folds(np.arange(50_000, dtype=np.int64), n_folds=5)
        for i, (train, test) in enumerate(folds):
            assert len(train) > 0, f"fold {i}: train is empty"
            assert len(test) > 0, f"fold {i}: test is empty"

    def test_indices_are_int64(self):
        folds = walk_forward_folds(np.arange(50_000, dtype=np.int64), n_folds=5)
        for train, test in folds:
            assert train.dtype == np.int64
            assert test.dtype == np.int64

    def test_indices_within_bounds(self):
        n = 50_000
        event_ts = np.arange(n, dtype=np.int64)
        folds = walk_forward_folds(event_ts, n_folds=5, embargo=600)
        for train, test in folds:
            assert int(train.min()) >= 0 and int(train.max()) < n
            assert int(test.min()) >= 0 and int(test.max()) < n

    def test_raises_on_insufficient_data(self):
        """Too-small input should raise ValueError with a clear message."""
        with pytest.raises(ValueError, match="insufficient"):
            walk_forward_folds(np.arange(100, dtype=np.int64), n_folds=5, embargo=600)

    def test_train_test_no_overlap(self):
        folds = walk_forward_folds(np.arange(50_000, dtype=np.int64), n_folds=5)
        for train, test in folds:
            assert set(train.tolist()).isdisjoint(set(test.tolist()))

    def test_returns_list_not_generator(self):
        result = walk_forward_folds(np.arange(50_000, dtype=np.int64), n_folds=5)
        assert isinstance(result, list)

    def test_internal_embargo_assertion_fires(self):
        """The function must assert the embargo internally (cannot be bypassed)."""
        # We can't easily bypass it from outside, but we can verify the embargo
        # is always satisfied in all return values.
        folds = walk_forward_folds(
            np.arange(100_000, dtype=np.int64), n_folds=4, embargo=600
        )
        for train, test in folds:
            prohibited_start = int(test.min()) - 600
            prohibited_end = int(test.max())
            # train must not contain any index in [prohibited_start, prohibited_end]
            overlapping = train[(train >= prohibited_start) & (train <= prohibited_end)]
            assert (
                len(overlapping) == 0
            ), f"{len(overlapping)} train indices in embargo zone"
