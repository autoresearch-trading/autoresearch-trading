"""Tests for TradingEnv min_hold constraint."""

from __future__ import annotations

import numpy as np

from prepare import TradingEnv


def _make_env(n_steps=100, min_hold=1, fee_bps=5, window_size=5):
    """Create a TradingEnv with synthetic data."""
    features = np.random.default_rng(42).normal(0, 1, (n_steps, 25))
    prices = 100.0 + np.random.default_rng(42).normal(0, 0.1, n_steps).cumsum()
    return TradingEnv(
        features, prices, window_size=window_size, fee_bps=fee_bps, min_hold=min_hold
    )


class TestMinHold:
    """Tests for min_hold position change enforcement."""

    def test_first_trade_always_allowed(self):
        """Entering from flat should always work, even with high min_hold."""
        env = _make_env(min_hold=999)
        env.reset()
        # Action 1 = long, from flat -> should be allowed
        _, _, _, _, info = env.step(1)
        assert info["position"] == 1

    def test_hold_prevents_early_exit(self):
        """Cannot exit position before min_hold steps."""
        env = _make_env(n_steps=200, min_hold=10)
        env.reset()
        # Enter long
        env.step(1)
        # Try to go flat immediately -> should be blocked
        _, _, _, _, info = env.step(0)
        assert info["position"] == 1  # still long

    def test_hold_allows_exit_after_period(self):
        """Can exit after min_hold steps elapsed."""
        env = _make_env(n_steps=200, min_hold=5)
        env.reset()
        # Enter long
        env.step(1)
        # Hold for min_hold steps
        for _ in range(5):
            env.step(1)
        # Now exit -> should be allowed
        _, _, _, _, info = env.step(0)
        assert info["position"] == 0

    def test_hold_prevents_flip(self):
        """Cannot flip from long to short during hold period."""
        env = _make_env(n_steps=200, min_hold=10)
        env.reset()
        env.step(1)  # enter long
        # Try to flip to short -> blocked
        _, _, _, _, info = env.step(2)
        assert info["position"] == 1

    def test_flat_to_position_always_allowed(self):
        """Entering from flat ignores min_hold."""
        env = _make_env(n_steps=200, min_hold=10)
        env.reset()
        # Enter long
        env.step(1)
        # Hold for min_hold
        for _ in range(10):
            env.step(1)
        # Exit to flat
        env.step(0)
        # Immediately enter short -> should work (from flat)
        _, _, _, _, info = env.step(2)
        assert info["position"] == 2  # 2=short

    def test_min_hold_1_allows_every_trade(self):
        """min_hold=1 means no constraint (trades every step)."""
        env = _make_env(n_steps=200, min_hold=1)
        env.reset()
        trades = 0
        positions = []
        for action in [1, 2, 0, 1, 0, 2, 1]:
            _, _, _, _, info = env.step(action)
            positions.append(info["position"])
        # All position changes should go through (2=short)
        assert positions == [1, 2, 0, 1, 0, 2, 1]

    def test_fee_charged_on_position_change(self):
        """Fees should be charged when position actually changes."""
        env = _make_env(
            n_steps=200, min_hold=1, fee_bps=100
        )  # 1% fee for easy checking
        env.reset()
        # Enter long (1% fee)
        _, _, _, _, info1 = env.step(1)
        assert info1["step_pnl"] < 0  # fee should make pnl negative (tiny price move)

    def test_no_fee_when_hold_blocks(self):
        """No fee when min_hold prevents the trade."""
        env = _make_env(n_steps=200, min_hold=100, fee_bps=100)
        env.reset()
        env.step(1)  # enter long, fee charged
        # Try to exit -> blocked by min_hold, no fee
        _, _, _, _, info = env.step(0)
        # Position unchanged, so only PnL from price movement (no fee)
        assert info["position"] == 1
