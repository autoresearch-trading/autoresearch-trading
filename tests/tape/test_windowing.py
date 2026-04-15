# tests/tape/test_windowing.py
"""Tests for tape/windowing.py — build_window_starts and window_view.

TDD: tests written before implementation. Run to confirm RED, then implement.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# build_window_starts — day-id-aware windowing
# ---------------------------------------------------------------------------


def test_stride50_1000_events_single_day_yields_17_windows() -> None:
    """Stride=50, 1000 events, window=200 → 17 windows: starts 0,50,...,800."""
    from tape.windowing import build_window_starts

    day_id = np.zeros(1000, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50, random_offset=0)
    assert starts[0] == 0
    assert starts[-1] == 800
    assert len(starts) == 17
    assert np.all(np.diff(starts) == 50)


def test_random_offset_shifts_first_window() -> None:
    """random_offset=17 → first window in day starts at 17."""
    from tape.windowing import build_window_starts

    day_id = np.zeros(1000, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50, random_offset=17)
    assert starts[0] == 17
    assert np.all(starts <= 1000 - 200)
    assert np.all(np.diff(starts) == 50)


def test_random_offset_is_modded_by_stride() -> None:
    """random_offset >= stride is reduced modulo stride."""
    from tape.windowing import build_window_starts

    day_id = np.zeros(1000, dtype=np.int64)
    # offset=50 mod stride=50 = 0 → same as no offset
    starts_0 = build_window_starts(day_id, window_len=200, stride=50, random_offset=0)
    starts_50 = build_window_starts(day_id, window_len=200, stride=50, random_offset=50)
    assert np.array_equal(starts_0, starts_50)


def test_short_shard_yields_empty_array() -> None:
    """Fewer than window_len events → no windows."""
    from tape.windowing import build_window_starts

    day_id = np.zeros(100, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50)
    assert len(starts) == 0


def test_exactly_window_len_events_yields_one_window() -> None:
    """Exactly 200 events → one window starting at 0."""
    from tape.windowing import build_window_starts

    day_id = np.zeros(200, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50, random_offset=0)
    assert len(starts) == 1
    assert starts[0] == 0


def test_day_boundary_enforced_no_cross_day_windows() -> None:
    """Windows must not cross day boundaries (gotcha #26).

    Construct: day 0 has 300 events, day 1 has 300 events.
    No window start should span both days.
    """
    from tape.windowing import build_window_starts

    day_id = np.array([0] * 300 + [1] * 300, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50, random_offset=0)
    assert len(starts) > 0
    for s in starts:
        # All events in window must share the same day_id
        assert day_id[s] == day_id[s + 199], (
            f"Window at {s} crosses day boundary: day_id[{s}]={day_id[s]} "
            f"!= day_id[{s+199}]={day_id[s+199]}"
        )


def test_multi_day_segments_both_contribute_windows() -> None:
    """Each day segment independently generates windows."""
    from tape.windowing import build_window_starts

    # day 0: 500 events → (500-200)//50 + 1 = 7 windows
    # day 1: 500 events → 7 windows
    day_id = np.array([0] * 500 + [1] * 500, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50, random_offset=0)
    # Verify each day contributes windows
    day0_starts = starts[day_id[starts] == 0]
    day1_starts = starts[day_id[starts] == 1]
    assert len(day0_starts) == 7
    assert len(day1_starts) == 7


def test_day_boundary_first_window_per_day_respects_offset() -> None:
    """With random_offset, each day segment's first window is offset from that day's start."""
    from tape.windowing import build_window_starts

    day_id = np.array([0] * 600 + [1] * 600, dtype=np.int64)
    starts = build_window_starts(day_id, window_len=200, stride=50, random_offset=20)
    day0_starts = starts[day_id[starts] == 0]
    day1_starts = starts[day_id[starts] == 1]
    # Day 0 starts at index 0+20=20
    assert day0_starts[0] == 20
    # Day 1 starts at global index 600 (day boundary), first window at 600+20=620
    assert day1_starts[0] == 620


# ---------------------------------------------------------------------------
# window_view — zero-copy slice
# ---------------------------------------------------------------------------


def test_window_view_returns_correct_slice() -> None:
    """window_view returns features[start:start+window_len] with same data."""
    from tape.windowing import window_view

    features = np.arange(2000, dtype=np.float32).reshape(1000, 2)
    view = window_view(features, start=50, window_len=200)
    assert view.shape == (200, 2)
    assert np.array_equal(view, features[50:250])


def test_window_view_is_a_view_not_copy() -> None:
    """window_view must return a numpy view (no copy) for memory efficiency."""
    from tape.windowing import window_view

    features = np.zeros((1000, 17), dtype=np.float32)
    view = window_view(features, start=10, window_len=200)
    # Modify view → original changes (proves it's a view)
    view[0, 0] = 99.0
    assert features[10, 0] == 99.0


def test_window_view_default_window_len_is_200() -> None:
    """Default window_len is WINDOW_LEN=200."""
    from tape.constants import WINDOW_LEN
    from tape.windowing import window_view

    features = np.zeros((500, 17), dtype=np.float32)
    view = window_view(features, start=0)
    assert view.shape[0] == WINDOW_LEN
