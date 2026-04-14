import numpy as np
import pytest

from tape.ob_align import align_events_to_ob


def test_each_event_mapped_to_nearest_prior_snapshot():
    ob_ts = np.array([100, 200, 300, 400], dtype=np.int64)
    event_ts = np.array([50, 100, 150, 200, 250, 399, 400, 500], dtype=np.int64)
    idx = align_events_to_ob(event_ts, ob_ts)
    # Event at 50 is before any snapshot → idx = -1 (caller masks)
    # Event at 100 → snapshot at 100 (idx 0)   [side="right"-1 maps 100 → idx 0]
    # Event at 150 → snapshot at 100 (idx 0)
    # Event at 200 → snapshot at 200 (idx 1)
    # Event at 250 → snapshot at 200 (idx 1)
    # Event at 399 → snapshot at 300 (idx 2)
    # Event at 400 → snapshot at 400 (idx 3)
    # Event at 500 → snapshot at 400 (idx 3)
    expected = np.array([-1, 0, 0, 1, 1, 2, 3, 3], dtype=np.int64)
    np.testing.assert_array_equal(idx, expected)


def test_vectorised_not_loop_for_large_input():
    # If someone reimplements as a Python for-loop, this takes >1s.
    rng = np.random.default_rng(0)
    ob_ts = np.sort(rng.integers(0, 10**9, size=100_000)).astype(np.int64)
    event_ts = np.sort(rng.integers(0, 10**9, size=1_000_000)).astype(np.int64)
    import time

    t = time.time()
    idx = align_events_to_ob(event_ts, ob_ts)
    assert (time.time() - t) < 0.5, "alignment must be vectorised"
    assert idx.shape == (1_000_000,)
    assert idx.max() < len(ob_ts)
    assert idx.min() >= -1


def test_monotonically_non_decreasing_ob_ts_required():
    ob_ts = np.array([100, 50, 200], dtype=np.int64)
    with pytest.raises(ValueError, match="non-decreasing"):
        align_events_to_ob(np.array([100], dtype=np.int64), ob_ts)
