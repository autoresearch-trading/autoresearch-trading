#!/usr/bin/env python3
"""Walk-forward validation: 4-fold rolling window (80d train, 20d test, 20d step).

Tests whether the Sortino edge is stable across different market regimes
or overfit to one specific test period.
"""
import sys
import time

sys.path.insert(0, ".")

import prepare
from train import (
    BEST_PARAMS,
    EXCLUDED_SYMBOLS,
    FINAL_BUDGET,
    FINAL_SEEDS,
    full_run,
)

# 4-fold rolling windows: 80d train, 20d test, 20d step
FOLDS = [
    {
        "name": "Fold 1",
        "train_start": "2025-10-16",
        "train_end": "2026-01-03",
        "test_start": "2026-01-04",
        "test_end": "2026-01-23",
    },
    {
        "name": "Fold 2",
        "train_start": "2025-11-05",
        "train_end": "2026-01-23",
        "test_start": "2026-01-24",
        "test_end": "2026-02-12",
    },
    {
        "name": "Fold 3",
        "train_start": "2025-11-25",
        "train_end": "2026-02-12",
        "test_start": "2026-02-13",
        "test_end": "2026-03-04",
    },
    {
        "name": "Fold 4",
        "train_start": "2025-12-15",
        "train_end": "2026-03-04",
        "test_start": "2026-03-05",
        "test_end": "2026-03-25",
    },
]


def run_fold(fold, symbols, params):
    """Run one walk-forward fold by overriding prepare.py date constants."""
    # Save originals
    orig_train_start = prepare.TRAIN_START
    orig_train_end = prepare.TRAIN_END
    orig_val_end = prepare.VAL_END
    orig_test_end = prepare.TEST_END

    try:
        # Override date constants for this fold
        # train split uses (TRAIN_START, TRAIN_END)
        # test split uses (VAL_END, TEST_END)
        prepare.TRAIN_START = fold["train_start"]
        prepare.TRAIN_END = fold["train_end"]
        prepare.VAL_END = fold["test_start"]
        prepare.TEST_END = fold["test_end"]

        result = full_run(
            symbols, params, FINAL_BUDGET, FINAL_SEEDS, split="test", verbose=True
        )
        # result = (sortino, passing, trades, dd, steps, updates, wr, pf, sharpe, calmar, cvar)
        return {
            "sortino": result[0],
            "passing": result[1],
            "trades": result[2],
            "max_dd": result[3],
            "win_rate": result[6],
            "profit_factor": result[7],
            "sharpe": result[8],
            "calmar": result[9],
            "cvar": result[10],
        }
    finally:
        # Restore originals
        prepare.TRAIN_START = orig_train_start
        prepare.TRAIN_END = orig_train_end
        prepare.VAL_END = orig_val_end
        prepare.TEST_END = orig_test_end


def main():
    symbols = [s for s in prepare.DEFAULT_SYMBOLS if s not in EXCLUDED_SYMBOLS]
    params = dict(BEST_PARAMS)
    n_symbols = len(symbols)

    print("=" * 78)
    print("WALK-FORWARD VALIDATION")
    print(f"  Folds: {len(FOLDS)} (80d train / 20d test / 20d step)")
    print(f"  Symbols: {n_symbols} (excluded: {EXCLUDED_SYMBOLS})")
    print(f"  Seeds: {FINAL_SEEDS}, Budget: {FINAL_BUDGET}s")
    print(f"  Params: {params}")
    print("=" * 78)

    results = []
    for i, fold in enumerate(FOLDS):
        print(f"\n{'='*78}")
        print(
            f"{fold['name']}: Train {fold['train_start']} → {fold['train_end']}  "
            f"Test {fold['test_start']} → {fold['test_end']}"
        )
        print("=" * 78)

        t0 = time.time()
        r = run_fold(fold, symbols, params)
        elapsed = time.time() - t0

        results.append(r)
        print(
            f"\n>>> {fold['name']}: Sortino={r['sortino']:.4f}  "
            f"Passing={r['passing']}/{n_symbols}  Trades={r['trades']}  "
            f"DD={r['max_dd']:.4f}  WR={r['win_rate']:.4f}  PF={r['profit_factor']:.4f}  "
            f"({elapsed:.0f}s)"
        )

    # Summary
    sortinos = [r["sortino"] for r in results]
    passings = [r["passing"] for r in results]
    trades = [r["trades"] for r in results]
    import numpy as np

    print("\n" + "=" * 78)
    print("WALK-FORWARD SUMMARY")
    print("=" * 78)
    print(
        f"{'Fold':<8} {'Train Period':<28} {'Test Period':<28} "
        f"{'Sortino':>8} {'Pass':>6} {'Trades':>7} {'DD':>6} {'WR':>6} {'PF':>6}"
    )
    print("-" * 110)
    for fold, r in zip(FOLDS, results):
        train_period = f"{fold['train_start']} → {fold['train_end']}"
        test_period = f"{fold['test_start']} → {fold['test_end']}"
        print(
            f"{fold['name']:<8} {train_period:<28} {test_period:<28} "
            f"{r['sortino']:>8.4f} {r['passing']:>4}/{n_symbols} "
            f"{r['trades']:>7} {r['max_dd']:>6.3f} {r['win_rate']:>6.3f} "
            f"{r['profit_factor']:>6.3f}"
        )
    print("-" * 110)
    print(
        f"{'Mean':<8} {'':<28} {'':<28} "
        f"{np.mean(sortinos):>8.4f} {np.mean(passings):>4.0f}/{n_symbols} "
        f"{np.mean(trades):>7.0f}"
    )
    print(f"{'Std':<8} {'':<28} {'':<28} " f"{np.std(sortinos):>8.4f}")
    print(f"{'Min':<8} {'':<28} {'':<28} " f"{np.min(sortinos):>8.4f}")
    print(f"{'Max':<8} {'':<28} {'':<28} " f"{np.max(sortinos):>8.4f}")

    # Verdict
    print("\n" + "=" * 78)
    all_positive = all(s > 0 for s in sortinos)
    mean_s = np.mean(sortinos)
    if all_positive:
        print(
            f"VERDICT: EDGE IS CONSISTENT — all {len(FOLDS)} folds positive (mean={mean_s:.4f})"
        )
    elif mean_s > 0:
        neg = sum(1 for s in sortinos if s <= 0)
        print(
            f"VERDICT: EDGE IS FRAGILE — {neg}/{len(FOLDS)} folds negative (mean={mean_s:.4f})"
        )
    else:
        print(f"VERDICT: NO EDGE — mean Sortino={mean_s:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
