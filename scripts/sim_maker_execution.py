#!/usr/bin/env python3
"""Simulate maker execution to estimate T41 upper bound.

Same trained models, but evaluate with maker costs:
- fee = 1.5 bps (Pacifica Tier 1 maker)
- slippage = 0 (no spread crossing, no impact)

This answers: "If we built limit order infrastructure, what Sortino could we achieve?"
"""

import io
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from prepare import DEFAULT_SYMBOLS, evaluate, make_env
from train import (
    BEST_PARAMS,
    DEVICE,
    EXCLUDED_SYMBOLS,
    FEE_BPS,
    FINAL_BUDGET,
    FINAL_SEEDS,
    MAX_HOLD_STEPS,
    MIN_HOLD,
    TRADE_BATCH,
    WINDOW_SIZE,
    make_ensemble_fn,
    train_one_model,
)


def main():
    tradeable = [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED_SYMBOLS]

    print("=" * 60)
    print("T41 SIMULATION: Maker Execution Upper Bound")
    print("=" * 60)
    print(f"Maker fee: 1.5 bps (vs taker {FEE_BPS} bps)")
    print(f"Slippage: 0 bps (no spread crossing)")
    print(f"Symbols: {len(tradeable)} (T40 filtered)")
    print(f"Config: {BEST_PARAMS}\n")

    # Train models (same as taker — training uses barrier labels, not execution costs)
    print("Training ensemble...")
    old_stdout = sys.stdout
    sys.stdout = open("/dev/null", "w")
    train_envs = {}
    for sym in tradeable:
        try:
            env = make_env(
                sym,
                "train",
                window_size=WINDOW_SIZE,
                trade_batch=TRADE_BATCH,
                min_hold=MIN_HOLD,
            )
            train_envs[sym] = env
        except Exception:
            pass
    sys.stdout.close()
    sys.stdout = old_stdout

    active = list(train_envs.keys())
    weights = np.array([train_envs[s].num_steps for s in active], dtype=np.float64)
    weights /= weights.sum()
    obs_shape = train_envs[active[0]].observation_space.shape

    models = []
    for seed in range(FINAL_SEEDS):
        print(f"  Training seed {seed}...")
        model, _, _ = train_one_model(
            train_envs,
            active,
            weights,
            obs_shape,
            BEST_PARAMS,
            FINAL_BUDGET // FINAL_SEEDS,
            seed,
        )
        models.append(model)
        model.eval()

    ensemble_fn = make_ensemble_fn(models, DEVICE)

    # Evaluate with MAKER costs
    print("\nEvaluating with maker execution...")
    passing = []
    for sym in tradeable:
        env_test = make_env(
            sym,
            "test",
            window_size=WINDOW_SIZE,
            trade_batch=TRADE_BATCH,
            min_hold=MIN_HOLD,
        )

        # Override to maker costs
        env_test.fee_bps = 1.5  # maker fee
        env_test.spread_bps = None  # no spread crossing (limit order at mid)

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        sh = evaluate(
            env_test,
            ensemble_fn,
            min_trades=10,
            r_min=BEST_PARAMS.get("r_min", 0.0),
            vpin_max_z=BEST_PARAMS.get("vpin_max_z", 0.0),
            fee_mult=BEST_PARAMS.get("fee_mult", 1.0),
        )
        sys.stdout = old
        out = buf.getvalue()

        t, d, wr, pf = 0, 0.0, 0.0, 0.0
        for ln in out.strip().split("\n"):
            if ln.startswith("num_trades:"):
                t = int(ln.split()[1])
            elif ln.startswith("max_drawdown:"):
                d = float(ln.split()[1])
            elif ln.startswith("win_rate:"):
                wr = float(ln.split()[1])
            elif ln.startswith("profit_factor:"):
                pf = float(ln.split()[1])

        passed = (t >= 10 and d <= 0.20) if t > 0 else False
        tag = "PASS" if passed else "FAIL"
        extra = f" wr={wr:.2f} pf={pf:.2f}" if wr > 0 else ""
        print(f"  {sym}: sortino={sh:.4f} trades={t} dd={d:.4f}{extra} [{tag}]")
        if passed:
            passing.append(sh)

    mean_sortino = float(np.mean(passing)) if passing else 0.0
    print(f"\n{'='*60}")
    print("MAKER EXECUTION SUMMARY")
    print(f"  sortino: {mean_sortino:.6f}")
    print(f"  symbols_passing: {len(passing)}/{len(tradeable)}")

    # Compare
    print(f"\n  vs TAKER (current): Sortino=0.184, 6/23")
    print(f"  Improvement: {mean_sortino - 0.184:+.4f} Sortino")


if __name__ == "__main__":
    main()
