#!/usr/bin/env python3
"""T46: Estimate expected variance of Sortino ratio across walk-forward folds.

Uses bootstrap resampling on the full dataset to estimate sampling distribution
of Sortino, then compares to observed fold variance.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from prepare import DEFAULT_SYMBOLS, evaluate, make_env

EXCLUDED = {"CRV", "XPL"}


def compute_sortino_from_returns(returns, steps_per_day):
    """Compute Sortino ratio from step returns array."""
    if len(returns) == 0:
        return 0.0
    mean_ret = returns.mean()
    downside = np.minimum(returns, 0)
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-10:
        return 0.0
    return mean_ret / downside_std * np.sqrt(steps_per_day)


def main():
    # Collect step returns from all passing symbols on the full test set
    # (uses the default test split: Feb 17 - Mar 25, 36 days)
    print("Loading test environments and collecting step returns...")
    all_returns = {}

    for sym in [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED]:
        env = make_env(sym, "test", window_size=50, trade_batch=100, min_hold=1200)
        n_steps = env.num_steps - env.window_size
        all_returns[sym] = {
            "prices": env.prices,
            "n_steps": n_steps,
        }

    # For each symbol, compute per-step returns assuming we're always in
    # a directional position (approximation for variance estimation)
    print(f"\nCollected data for {len(all_returns)} symbols")

    # Walk-forward observed values
    observed_sortinos = [0.0148, 0.3827, 0.5586, 0.0888]
    observed_mean = np.mean(observed_sortinos)
    observed_std = np.std(observed_sortinos)

    print(f"\n=== OBSERVED WALK-FORWARD RESULTS ===")
    print(f"  Fold Sortinos: {observed_sortinos}")
    print(f"  Mean: {observed_mean:.4f}")
    print(f"  Std:  {observed_std:.4f}")
    print(f"  CV (std/mean): {observed_std/observed_mean:.2f}")

    # T46 analytical bound:
    # For portfolio Sortino S computed from N steps with mean return mu
    # and downside std sigma_d:
    #   S = mu / sigma_d * sqrt(steps_per_day)
    #
    # The variance of the sample mean is var(mu_hat) = sigma^2 / N
    # where sigma is the total std of returns.
    #
    # By delta method, Var(S_hat) ≈ (sqrt(steps_per_day) / sigma_d)^2 * (sigma^2 / N)
    #   = steps_per_day * sigma^2 / (sigma_d^2 * N)
    #
    # For a 20-day test window with ~25K steps/symbol and 9 passing symbols:
    #   N_effective ≈ 25K * 9 = 225K per fold (if independent)
    #   But symbols are correlated (rho ≈ 0.28), so effective N is lower

    # Estimate from the actual full test period
    # Use price returns as proxy for step returns (most steps are in position)
    total_steps_per_symbol = []
    return_stds = []
    return_means = []
    downside_stds = []

    for sym, data in all_returns.items():
        px = data["prices"]
        ret = np.diff(px) / px[:-1]
        ret = ret[~np.isnan(ret) & ~np.isinf(ret)]
        if len(ret) > 100:
            total_steps_per_symbol.append(len(ret))
            return_stds.append(ret.std())
            return_means.append(ret.mean())
            downside = np.minimum(ret, 0)
            downside_stds.append(np.sqrt(np.mean(downside**2)))

    avg_steps = np.mean(total_steps_per_symbol)
    avg_std = np.mean(return_stds)
    avg_downside_std = np.mean(downside_stds)
    avg_mean = np.mean(return_means)

    # Steps per fold (20 days ≈ 20/36 * total steps)
    steps_per_fold = avg_steps * 20 / 36
    steps_per_day_fold = steps_per_fold / 20

    print(f"\n=== DATA STATISTICS (per symbol, full test set) ===")
    print(f"  Avg steps/symbol: {avg_steps:.0f}")
    print(f"  Steps per 20d fold: {steps_per_fold:.0f}")
    print(f"  Steps per day: {steps_per_day_fold:.0f}")
    print(f"  Avg return std: {avg_std:.6f}")
    print(f"  Avg downside std: {avg_downside_std:.6f}")
    print(f"  Avg mean return: {avg_mean:.8f}")

    # T46 Formula: Std(Sortino_hat) for a single symbol
    # SE(mu_hat) = sigma / sqrt(N)
    # SE(Sortino) ≈ sqrt(steps_per_day) * sigma / (sigma_d * sqrt(N))
    #            = Sortino_true * sigma / (mu * sqrt(N))  ... but mu is tiny
    #
    # More precisely, for ratio estimator mu/sigma_d:
    # Var(mu/sigma_d) ≈ (1/sigma_d^2) * Var(mu) + (mu^2/sigma_d^4) * Var(sigma_d)
    #
    # Var(mu) = sigma^2 / N
    # Var(sigma_d) ≈ sigma_d^2 / (2*N)  (for downside, approximately)
    #
    # So: SE(S) ≈ sqrt(steps_per_day) * sqrt(sigma^2/(sigma_d^2 * N) + S^2/(2*N*steps_per_day))

    N_per_symbol = steps_per_fold
    n_passing = 9  # avg passing symbols
    rho = 0.28  # empirical from T44

    # Effective N for portfolio (accounting for correlation)
    N_eff_symbols = n_passing / (1 + (n_passing - 1) * rho)
    N_portfolio = N_per_symbol * N_eff_symbols

    sigma = avg_std
    sigma_d = avg_downside_std
    spd = steps_per_day_fold
    S_true = observed_mean  # use observed mean as estimate

    # Standard error of Sortino (delta method)
    var_term1 = sigma**2 / (sigma_d**2 * N_portfolio)  # from mean estimation
    var_term2 = S_true**2 / (2 * N_portfolio * spd)  # from downside_std estimation
    se_sortino = np.sqrt(spd) * np.sqrt(var_term1 + var_term2)

    print(f"\n=== T46: SORTINO VARIANCE BOUND ===")
    print(f"  N per symbol per fold: {N_per_symbol:.0f}")
    print(f"  N_eff symbols (rho={rho}): {N_eff_symbols:.1f}")
    print(f"  N_portfolio effective: {N_portfolio:.0f}")
    print(f"  SE(Sortino) from delta method: {se_sortino:.4f}")
    print(f"  Expected std across folds: {se_sortino:.4f}")
    print(f"  Observed std across folds: {observed_std:.4f}")
    print(f"  Ratio observed/expected: {observed_std/se_sortino:.1f}x")

    if observed_std / se_sortino < 2.5:
        print(f"\n  CONCLUSION: Fold variance is CONSISTENT with sampling noise.")
        print(
            f"  The edge appears stable — fold variation is expected for {N_per_symbol:.0f} steps/fold."
        )
    else:
        print(
            f"\n  CONCLUSION: Fold variance EXCEEDS sampling noise by {observed_std/se_sortino:.1f}x."
        )
        print(f"  This suggests genuine REGIME SHIFTS between folds.")

    # Also: how many steps do we need for SE < 0.1 (reasonably precise Sortino)?
    target_se = 0.1
    N_needed = spd * (sigma**2 / (sigma_d**2 * target_se**2))
    days_needed = N_needed / steps_per_day_fold
    print(f"\n  Steps needed for SE(Sortino) < {target_se}: {N_needed:.0f}")
    print(f"  Days needed: {days_needed:.0f}")

    # Bootstrap validation: resample 20-day windows from the full test period
    print(f"\n=== BOOTSTRAP VALIDATION ===")
    # Collect all step returns across symbols for the full test period
    # Then sample 20-day windows and compute portfolio Sortino
    all_price_returns = []
    for sym, data in all_returns.items():
        px = data["prices"]
        ret = np.diff(px) / px[:-1]
        ret = ret[~np.isnan(ret) & ~np.isinf(ret)]
        all_price_returns.append(ret[: int(avg_steps)])

    # Align lengths
    min_len = min(len(r) for r in all_price_returns)
    aligned = np.array([r[:min_len] for r in all_price_returns])  # (n_symbols, n_steps)

    # Sample random 20-day windows (20/36 of total steps)
    window_size = int(min_len * 20 / 36)
    n_bootstrap = 1000
    bootstrap_sortinos = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        start = rng.integers(0, min_len - window_size)
        window = aligned[:, start : start + window_size]
        # Portfolio mean return (equal weight across symbols)
        port_ret = window.mean(axis=0)  # average across symbols per step
        s = compute_sortino_from_returns(port_ret, window_size / 20)
        bootstrap_sortinos.append(s)

    bootstrap_sortinos = np.array(bootstrap_sortinos)
    print(f"  Bootstrap windows: {n_bootstrap}, window size: {window_size} steps")
    print(f"  Bootstrap mean Sortino: {bootstrap_sortinos.mean():.4f}")
    print(f"  Bootstrap std Sortino: {bootstrap_sortinos.std():.4f}")
    print(f"  Bootstrap 5th percentile: {np.percentile(bootstrap_sortinos, 5):.4f}")
    print(f"  Bootstrap 95th percentile: {np.percentile(bootstrap_sortinos, 95):.4f}")
    print(f"  Observed std: {observed_std:.4f}")
    print(f"  Bootstrap/Observed ratio: {bootstrap_sortinos.std()/observed_std:.2f}")


if __name__ == "__main__":
    main()
