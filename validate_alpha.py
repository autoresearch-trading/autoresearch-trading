#!/usr/bin/env python3
"""Validate whether features have alpha net of fees with simple models.

No RL. Just: can features predict returns? Does the prediction survive fees?
"""

import numpy as np
from scipy.stats import spearmanr

from prepare import (
    CACHE_DIR,
    DEFAULT_SYMBOLS,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    load_cached,
    make_env,
)

FEATURE_NAMES = [
    "returns",
    "r_5",
    "r_20",
    "r_100",
    "realvol_10",
    "bipower_var_20",
    "tfi",
    "volume_spike_ratio",
    "large_trade_share",
    "kyle_lambda_50",
    "amihud_illiq_50",
    "trade_arrival_rate",
    "spread_bps",
    "log_total_depth",
    "weighted_imbalance_5lvl",
    "microprice_dev",
    "ofi",
    "ob_slope_asym",
    "funding_zscore",
    "utc_hour_linear",
]

WARMUP = 200


def load_split(symbol, split):
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (TRAIN_END, VAL_END),
        "test": (VAL_END, TEST_END),
    }
    start, end = splits[split]
    return load_cached(symbol, CACHE_DIR, start, end, 100)


def ensure_cached(symbols, split):
    """Make sure features are cached for all symbols in split."""
    for sym in symbols:
        cached = load_split(sym, split)
        if cached is None:
            print(f"  Computing {sym} {split}...")
            make_env(sym, split)


# ============================================================
# Backtest with z-score thresholds and holding logic
# ============================================================
def backtest(features, prices, weights, enter_thresh=0.5, exit_thresh=0.1, fee_bps=5):
    """Backtest with entry/exit thresholds on z-score signal.

    - Enter long when signal > enter_thresh
    - Enter short when signal < -enter_thresh
    - Exit when |signal| < exit_thresh (go flat)
    - Hold otherwise (hysteresis reduces turnover)
    """
    n = len(features)
    signal = features @ weights

    fee_frac = fee_bps / 10000
    position = 0  # 0=flat, 1=long, -1=short
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0
    step_pnls = []

    for t in range(WARMUP, n - 1):
        s = signal[t]

        # Position logic with hysteresis
        if position == 0:
            if s > enter_thresh:
                new_pos = 1
            elif s < -enter_thresh:
                new_pos = -1
            else:
                new_pos = 0
        elif position == 1:
            if s < -enter_thresh:
                new_pos = -1  # flip
            elif s < exit_thresh:
                new_pos = 0  # exit
            else:
                new_pos = 1  # hold
        else:  # position == -1
            if s > enter_thresh:
                new_pos = 1  # flip
            elif s > -exit_thresh:
                new_pos = 0  # exit
            else:
                new_pos = -1  # hold

        # Price return
        ret = (prices[t + 1] - prices[t]) / prices[t] if prices[t] > 0 else 0.0

        # P&L
        pnl = position * ret

        # Fee on position change
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
    total_ret = equity - 1.0
    sharpe = pnls.mean() / max(pnls.std(), 1e-10) * np.sqrt(len(pnls) / 100)

    return {
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": trades,
        "equity": equity,
    }


def run_strategy(
    symbols, split, feature_indices, weights, enter_thresh, exit_thresh, fee_bps=5
):
    """Run a strategy across symbols, return aggregate results."""
    results = []
    for sym in symbols:
        cached = load_split(sym, split)
        if cached is None:
            continue
        features, _, prices = cached
        if len(features) < WARMUP + 100:
            continue

        full_w = np.zeros(features.shape[1])
        for idx, w in zip(feature_indices, weights):
            full_w[idx] = w

        res = backtest(features, prices, full_w, enter_thresh, exit_thresh, fee_bps)
        results.append((sym, res))
    return results


def print_summary(results, label=""):
    if not results:
        print(f"  {label}: no data")
        return
    sharpes = [r["sharpe"] for _, r in results]
    returns = [r["total_return"] for _, r in results]
    max_dds = [r["max_dd"] for _, r in results]
    trades = [r["trades"] for _, r in results]
    passing = sum(1 for s in sharpes if s > 0)
    print(
        f"  {label:<35s} "
        f"sharpe={np.mean(sharpes):>7.3f} "
        f"pass={passing}/{len(results)} "
        f"ret={np.mean(returns)*100:>7.2f}% "
        f"dd={np.mean(max_dds)*100:>5.1f}% "
        f"trades={np.mean(trades):>7.0f}"
    )


if __name__ == "__main__":
    test_syms = ["BTC", "ETH", "SOL", "DOGE", "CRV"]

    # Ensure caches exist
    print("Ensuring caches...")
    ensure_cached(test_syms, "val")
    ensure_cached(test_syms, "test")

    # ── Test 1: Feature signal ──
    print("\n" + "=" * 70)
    print("TEST 1: Feature Predictive Signal (train split)")
    print("=" * 70)
    agg = {name: [] for name in FEATURE_NAMES}
    for sym in DEFAULT_SYMBOLS:
        cached = load_split(sym, "train")
        if cached is None:
            continue
        features, _, prices = cached
        n = len(features)
        if n < WARMUP + 100:
            continue
        future_ret = np.zeros(n)
        future_ret[:-1] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))
        for i, name in enumerate(FEATURE_NAMES):
            corr, _ = spearmanr(features[WARMUP:-1, i], future_ret[WARMUP:-1])
            if not np.isnan(corr):
                agg[name].append(corr)

    ranked = [(n, np.mean(v), np.std(v), len(v)) for n, v in agg.items() if v]
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {'Feature':<25s} {'mean_corr':>10s} {'std':>8s}")
    for name, m, s, cnt in ranked:
        flag = " ***" if abs(m) > 0.02 else ""
        print(f"  {name:<25s} {m:>10.4f} {s:>8.4f}{flag}")

    # ── Test 2: Threshold sweep (z-score thresholds, not bps) ──
    print("\n" + "=" * 70)
    print("TEST 2: Threshold Sweep — top3 features (val split, 5bps fees)")
    print("=" * 70)
    print("  enter/exit thresholds on z-score signal (imbalance + microprice + ofi)")

    idxs = [14, 15, 16]
    ws = [1.0, 1.0, 1.0]

    for enter in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        exit_t = enter * 0.3  # exit at 30% of entry threshold
        res = run_strategy(test_syms, "val", idxs, ws, enter, exit_t, fee_bps=5)
        print_summary(res, f"enter={enter:.1f} exit={exit_t:.1f}")

    # ── Test 3: Fee sensitivity at best threshold ──
    print("\n" + "=" * 70)
    print("TEST 3: Fee Sensitivity (enter=1.0, exit=0.3)")
    print("=" * 70)

    for fee in [0, 1, 2, 3, 5, 7, 10]:
        res = run_strategy(test_syms, "val", idxs, ws, 1.0, 0.3, fee_bps=fee)
        print_summary(res, f"fee={fee}bps")

    # ── Test 4: Different feature combos at best threshold ──
    print("\n" + "=" * 70)
    print("TEST 4: Feature Combos (enter=1.0, exit=0.3, 5bps fees, val)")
    print("=" * 70)

    combos = {
        "microprice_dev only": ([15], [1.0]),
        "imbalance only": ([14], [1.0]),
        "ofi only": ([16], [1.0]),
        "top2: imb+micro": ([14, 15], [1.0, 1.0]),
        "top3: imb+micro+ofi": ([14, 15, 16], [1.0, 1.0, 1.0]),
        "top4: +returns": ([0, 14, 15, 16], [1.0, 1.0, 1.0, 1.0]),
        "top5: +tfi": ([0, 6, 14, 15, 16], [1.0, 1.0, 1.0, 1.0, 1.0]),
        "micro+ofi (no imb)": ([15, 16], [1.0, 1.0]),
    }

    for name, (idxs_c, ws_c) in combos.items():
        res = run_strategy(test_syms, "val", idxs_c, ws_c, 1.0, 0.3, fee_bps=5)
        print_summary(res, name)

    # ── Test 5: Out of sample ──
    print("\n" + "=" * 70)
    print("TEST 5: Out-of-Sample (TEST split)")
    print("=" * 70)

    best_idxs = [14, 15, 16]
    best_ws = [1.0, 1.0, 1.0]

    for enter in [0.5, 1.0, 1.5, 2.0]:
        exit_t = enter * 0.3
        res = run_strategy(
            test_syms, "test", best_idxs, best_ws, enter, exit_t, fee_bps=5
        )
        print_summary(res, f"enter={enter:.1f} exit={exit_t:.1f}")

    # Per-symbol detail at best threshold
    print(f"\n  Per-symbol (enter=1.0, test split):")
    res = run_strategy(test_syms, "test", best_idxs, best_ws, 1.0, 0.3, fee_bps=5)
    for sym, r in res:
        tag = "PASS" if r["sharpe"] > 0 else "FAIL"
        print(
            f"    {sym:<8s} sharpe={r['sharpe']:>7.3f} "
            f"ret={r['total_return']*100:>7.2f}% "
            f"dd={r['max_dd']*100:>5.1f}% "
            f"trades={r['trades']:>5d} [{tag}]"
        )

    # ── Test 6: All 25 symbols at best config ──
    print(f"\n  All 25 symbols (enter=1.0, val split):")
    ensure_cached(DEFAULT_SYMBOLS, "val")
    res = run_strategy(DEFAULT_SYMBOLS, "val", best_idxs, best_ws, 1.0, 0.3, fee_bps=5)
    passing_syms = []
    for sym, r in res:
        tag = "PASS" if r["sharpe"] > 0 else "FAIL"
        if r["sharpe"] > 0:
            passing_syms.append(sym)
        print(
            f"    {sym:<10s} sharpe={r['sharpe']:>7.3f} "
            f"ret={r['total_return']*100:>7.2f}% "
            f"dd={r['max_dd']*100:>5.1f}% "
            f"trades={r['trades']:>5d} [{tag}]"
        )
    print(f"\n  PASSING: {len(passing_syms)}/{len(res)} — {passing_syms}")
