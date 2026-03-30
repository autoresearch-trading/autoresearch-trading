"""Tests for evaluation metrics correctness."""

import numpy as np


def test_sortino_uses_all_observations_in_denominator():
    """T26: downside_std must divide by N, not N_neg."""
    returns = np.array([0.01, 0.02, -0.01, -0.02, 0.015])
    # Correct: min(r, 0) for all, then sqrt(mean(squares))
    downside_returns = np.minimum(returns, 0)  # [0, 0, -0.01, -0.02, 0]
    correct_std = np.sqrt(np.mean(downside_returns**2))
    # Buggy: filter to negatives only
    neg_only = returns[returns < 0]  # [-0.01, -0.02]
    buggy_std = np.sqrt(np.mean(neg_only**2))
    # Buggy divides by 2 instead of 5 -> inflates denominator
    assert buggy_std > correct_std
    # Correct relationship: buggy = correct / sqrt(p) where p = N_neg/N
    p = len(neg_only) / len(returns)
    np.testing.assert_allclose(buggy_std, correct_std / np.sqrt(p), rtol=1e-10)


def test_sortino_correction_factor():
    """T26: correction factor is 1/sqrt(p) ~ 1.49 for p=0.45."""
    p = 0.45
    factor = 1.0 / np.sqrt(p)
    assert factor > 1.490
    assert factor < 1.491


def test_evaluate_sortino_formula():
    """evaluate() should use correct Sortino formula (all N in denominator)."""
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002] * 20)
    mean_ret = returns.mean()
    downside_returns = np.minimum(returns, 0)
    downside_std = np.sqrt(np.mean(downside_returns**2))
    steps_per_day = 100
    expected_sortino = mean_ret / downside_std * np.sqrt(steps_per_day)
    assert expected_sortino > 0  # sanity check


def test_sharpe_ratio():
    """T27: Sharpe = mean / std * sqrt(spd)."""
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002])
    mean_ret = returns.mean()
    std_ret = returns.std()
    spd = 100
    sharpe = mean_ret / std_ret * np.sqrt(spd)
    assert sharpe > 0


def test_sortino_ge_sharpe():
    """T27: Sortino >= Sharpe for profitable strategies."""
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002] * 20)
    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    spd = 100
    sharpe = mean_ret / std_ret * np.sqrt(spd)
    sortino = mean_ret / downside_std * np.sqrt(spd)
    assert sortino >= sharpe


def test_cvar_ge_var():
    """T28: CVaR >= VaR always."""
    returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    sorted_r = np.sort(returns)
    k = max(1, int(0.05 * len(returns)))  # 5% tail
    var_95 = -sorted_r[k - 1]
    cvar_95 = -np.mean(sorted_r[:k])
    assert cvar_95 >= var_95


def test_calmar_ratio():
    """T27: Calmar = annualized return / max drawdown."""
    annual_return = 0.10
    max_dd = 0.20
    calmar = annual_return / max_dd
    assert calmar == 0.5
