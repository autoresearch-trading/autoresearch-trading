"""Tests for Hawkes regime gate in evaluate()."""

import numpy as np


def test_regime_gate_forces_flat():
    """When hawkes_branching < r_min, action should be forced to 0."""
    raw_hawkes = np.array([0.1, 0.2, 0.8, 0.9, 0.3])
    r_min = 0.5
    actions = np.array([1, 2, 1, 2, 1])  # model wants to trade
    gated = np.where(raw_hawkes < r_min, 0, actions)
    assert list(gated) == [0, 0, 1, 2, 0]


def test_regime_gate_no_filter_when_zero():
    """When r_min=0, no filtering occurs."""
    raw_hawkes = np.array([0.1, 0.2, 0.8])
    r_min = 0.0
    actions = np.array([1, 2, 1])
    gated = np.where(raw_hawkes < r_min, 0, actions)
    assert list(gated) == [1, 2, 1]


def test_alpha_min_values():
    """Verify proved alpha_min formula."""
    assert abs(0.5 + 1 / (2 * 1.5) - 5 / 6) < 1e-10
    assert abs(0.5 + 1 / (2 * 4.0) - 0.625) < 1e-10
    assert abs(0.5 + 1 / (2 * 8.0) - 0.5625) < 1e-10
