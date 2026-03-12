#!/usr/bin/env python3
"""Find the trade frequency where alpha survives fees.

Approach: use the top3 signal (imbalance + microprice + ofi) but vary
how often we're ALLOWED to trade. Impose a minimum hold period.
"""

import numpy as np

from prepare import CACHE_DIR, TRAIN_END, TRAIN_START, VAL_END, load_cached

SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "CRV"]
WARMUP = 200
FEE_BPS = 5


def load_val(sym):
    return load_cached(sym, CACHE_DIR, TRAIN_END, VAL_END, 100)


def backtest_with_hold_period(features, prices, weights, min_hold, enter_thresh=0.5):
    """Only allow position changes every min_hold steps."""
    n = len(features)
    signal = features @ weights
    fee_frac = FEE_BPS / 10000

    position = 0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0
    steps_since_trade = min_hold  # allow immediate first trade
    step_pnls = []

    for t in range(WARMUP, n - 1):
        s = signal[t]
        steps_since_trade += 1

        # Only consider changing position if hold period elapsed
        if steps_since_trade >= min_hold:
            if s > enter_thresh:
                new_pos = 1
            elif s < -enter_thresh:
                new_pos = -1
            else:
                new_pos = 0

            if new_pos != position:
                # Commit to trade
                steps_since_trade = 0
            else:
                new_pos = position
        else:
            new_pos = position

        # P&L
        ret = (prices[t + 1] - prices[t]) / prices[t] if prices[t] > 0 else 0.0
        pnl = position * ret

        if new_pos != position:
            if position != 0:
                pnl -= fee_frac
            if new_pos != 0:
                pnl -= fee_frac
            trades += 1
            position = new_pos

        equity *= 1 + pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        step_pnls.append(pnl)

    pnls = np.array(step_pnls)
    n_steps = len(pnls)
    sharpe = pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(n_steps / 100)

    return {
        "sharpe": sharpe,
        "total_return": equity - 1.0,
        "max_dd": max_dd,
        "trades": trades,
        "trades_per_day": trades / max(n_steps / 2800, 1),  # ~2800 batches/day
    }


if __name__ == "__main__":
    weights = np.zeros(20)
    weights[14] = 1.0  # imbalance
    weights[15] = 1.0  # microprice_dev
    weights[16] = 1.0  # ofi

    # ── Sweep 1: min_hold period with fixed entry threshold ──
    print("=" * 80)
    print("SWEEP 1: Min hold period (enter_thresh=0.5, 5bps fees)")
    print("  min_hold in batches (~1 batch = 1 minute)")
    print("=" * 80)

    print(
        f"\n  {'hold':>6s} {'~mins':>6s} {'sharpe':>8s} {'pass':>6s} {'ret%':>8s} {'dd%':>6s} {'trades':>7s} {'tr/day':>7s}"
    )

    for hold in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
        results = []
        for sym in SYMBOLS:
            cached = load_val(sym)
            if cached is None:
                continue
            features, _, prices = cached
            if len(features) < WARMUP + 100:
                continue
            r = backtest_with_hold_period(
                features, prices, weights, hold, enter_thresh=0.5
            )
            results.append(r)

        if not results:
            continue

        sharpes = [r["sharpe"] for r in results]
        rets = [r["total_return"] for r in results]
        dds = [r["max_dd"] for r in results]
        trades = [r["trades"] for r in results]
        tpd = [r["trades_per_day"] for r in results]
        passing = sum(1 for s in sharpes if s > 0)

        print(
            f"  {hold:>6d} {hold:>6d} {np.mean(sharpes):>8.3f} "
            f"{passing}/{len(results):>3d} {np.mean(rets)*100:>8.2f} "
            f"{np.mean(dds)*100:>6.1f} {np.mean(trades):>7.0f} {np.mean(tpd):>7.1f}"
        )

    # ── Sweep 2: entry threshold with best hold period ──
    print("\n" + "=" * 80)
    print("SWEEP 2: Entry threshold (min_hold=100, 5bps fees)")
    print("=" * 80)

    print(
        f"\n  {'thresh':>6s} {'sharpe':>8s} {'pass':>6s} {'ret%':>8s} {'dd%':>6s} {'trades':>7s} {'tr/day':>7s}"
    )

    for thresh in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        results = []
        for sym in SYMBOLS:
            cached = load_val(sym)
            if cached is None:
                continue
            features, _, prices = cached
            if len(features) < WARMUP + 100:
                continue
            r = backtest_with_hold_period(
                features, prices, weights, 100, enter_thresh=thresh
            )
            results.append(r)

        sharpes = [r["sharpe"] for r in results]
        rets = [r["total_return"] for r in results]
        dds = [r["max_dd"] for r in results]
        trades = [r["trades"] for r in results]
        tpd = [r["trades_per_day"] for r in results]
        passing = sum(1 for s in sharpes if s > 0)

        print(
            f"  {thresh:>6.1f} {np.mean(sharpes):>8.3f} "
            f"{passing}/{len(results):>3d} {np.mean(rets)*100:>8.2f} "
            f"{np.mean(dds)*100:>6.1f} {np.mean(trades):>7.0f} {np.mean(tpd):>7.1f}"
        )

    # ── Sweep 3: 2D grid around promising region ──
    print("\n" + "=" * 80)
    print("SWEEP 3: 2D Grid (hold × threshold, 5bps fees)")
    print("=" * 80)

    best_sharpe = -999
    best_config = None

    holds = [50, 100, 200, 500, 1000]
    threshs = [0.5, 1.0, 1.5, 2.0, 2.5]

    print(
        f"\n  {'hold':>6s} {'thresh':>6s} {'sharpe':>8s} {'pass':>6s} {'ret%':>8s} {'dd%':>6s} {'trades':>7s} {'tr/day':>7s}"
    )

    for hold in holds:
        for thresh in threshs:
            results = []
            for sym in SYMBOLS:
                cached = load_val(sym)
                if cached is None:
                    continue
                features, _, prices = cached
                if len(features) < WARMUP + 100:
                    continue
                r = backtest_with_hold_period(
                    features, prices, weights, hold, enter_thresh=thresh
                )
                results.append(r)

            sharpes = [r["sharpe"] for r in results]
            rets = [r["total_return"] for r in results]
            dds = [r["max_dd"] for r in results]
            trades = [r["trades"] for r in results]
            tpd = [r["trades_per_day"] for r in results]
            passing = sum(1 for s in sharpes if s > 0)

            avg_sharpe = np.mean(sharpes)
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_config = (hold, thresh)

            flag = " <-- BEST" if avg_sharpe == best_sharpe else ""
            print(
                f"  {hold:>6d} {thresh:>6.1f} {avg_sharpe:>8.3f} "
                f"{passing}/{len(results):>3d} {np.mean(rets)*100:>8.2f} "
                f"{np.mean(dds)*100:>6.1f} {np.mean(trades):>7.0f} {np.mean(tpd):>7.1f}{flag}"
            )

    print(
        f"\n  BEST: hold={best_config[0]}, thresh={best_config[1]}, sharpe={best_sharpe:.3f}"
    )

    # ── Per-symbol detail at best config ──
    hold, thresh = best_config
    print(f"\n  Per-symbol at best config (hold={hold}, thresh={thresh}):")
    for sym in SYMBOLS:
        cached = load_val(sym)
        if cached is None:
            continue
        features, _, prices = cached
        r = backtest_with_hold_period(
            features, prices, weights, hold, enter_thresh=thresh
        )
        tag = "PASS" if r["sharpe"] > 0 else "FAIL"
        print(
            f"    {sym:<8s} sharpe={r['sharpe']:>7.3f} "
            f"ret={r['total_return']*100:>7.2f}% "
            f"dd={r['max_dd']*100:>5.1f}% "
            f"trades={r['trades']:>5d} tr/day={r['trades_per_day']:>5.1f} [{tag}]"
        )
