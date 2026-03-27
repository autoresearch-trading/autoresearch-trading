"""Tests for funding rate cost in TradingEnv."""

import numpy as np

from prepare import TradingEnv


def _make_env(n=100, funding_rates=None):
    """Helper: flat-price env with zero fees, optional funding."""
    features = np.random.randn(n, 13).astype(np.float32)
    prices = np.full(n, 100.0)
    env = TradingEnv(features, prices, window_size=10, fee_bps=0, min_hold=1)
    env.spread_bps = np.zeros(n)  # zero spread (impact_buffer still 3bps)
    env.funding_rates = funding_rates
    return env


def _hold_pnl(env, action, steps=10):
    """Enter position, then return PnL from holding steps only (excludes entry)."""
    env.reset(seed=42, options={"sequential": True})
    env.step(action)  # entry step — ignore its PnL
    total = 0.0
    for _ in range(steps):
        _, _, _, _, info = env.step(action)
        total += info["step_pnl"]
    return total


def test_funding_charged_when_holding_long():
    """Long pays positive funding: holding PnL should be negative."""
    n = 100
    env = _make_env(n, funding_rates=np.full(n, 0.001))  # 10bps/step
    pnl = _hold_pnl(env, action=1, steps=10)
    assert pnl < -0.005, f"Long should pay ~10bps/step funding, got {pnl}"


def test_funding_charged_when_holding_short():
    """Short pays negative funding: holding PnL should be negative."""
    n = 100
    env = _make_env(n, funding_rates=np.full(n, -0.001))  # shorts pay
    pnl = _hold_pnl(env, action=2, steps=10)
    assert pnl < -0.005, f"Short should pay negative funding, got {pnl}"


def test_funding_not_charged_when_flat():
    """Flat position should not pay funding."""
    n = 100
    env = _make_env(n, funding_rates=np.full(n, 0.001))
    env.reset(seed=42, options={"sequential": True})
    total = 0.0
    for _ in range(10):
        _, _, _, _, info = env.step(0)
        total += info["step_pnl"]
    assert total == 0.0, f"Flat should not pay funding, got {total}"


def test_short_receives_positive_funding():
    """Short receives positive funding: holding PnL should be positive."""
    n = 100
    env = _make_env(n, funding_rates=np.full(n, 0.001))  # shorts receive
    pnl = _hold_pnl(env, action=2, steps=10)
    assert pnl > 0.005, f"Short should receive ~10bps/step funding, got {pnl}"


def test_funding_none_means_no_charge():
    """No funding_rates: holding PnL should be zero (flat price, no fees)."""
    n = 100
    env = _make_env(n, funding_rates=None)
    pnl = _hold_pnl(env, action=1, steps=10)
    assert pnl == 0.0, f"No funding should mean zero holding cost, got {pnl}"


def test_funding_amount_is_correct():
    """Verify funding cost is exactly rate per step."""
    n = 100
    rate = 0.0005  # 5bps per step
    env = _make_env(n, funding_rates=np.full(n, rate))
    pnl = _hold_pnl(env, action=1, steps=10)
    expected = -rate * 10  # long pays positive funding for 10 steps
    assert abs(pnl - expected) < 1e-10, f"Expected {expected}, got {pnl}"
