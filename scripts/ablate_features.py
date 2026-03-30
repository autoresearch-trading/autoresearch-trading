#!/usr/bin/env python3
"""Feature ablation for v11: train once, evaluate with each feature zeroed out.

Usage:
    uv run python scripts/ablate_features.py

Trains one ensemble (5 seeds) on all 17 features, then evaluates:
  1. Baseline (all 17 features)
  2. Drop each of the 8 new features (indices 9-16) one at a time
  3. Drop all 8 new features at once (back to v10's 9)

Zeroing a z-scored feature = setting it to its mean (neutral signal).
"""

import io
import sys
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, ".")

from prepare import DEFAULT_SYMBOLS, V9_FEATURE_NAMES, evaluate, make_env
from train import (
    BEST_PARAMS,
    DEVICE,
    FEE_BPS,
    FINAL_BUDGET,
    FINAL_SEEDS,
    MAX_HOLD_STEPS,
    MIN_HOLD,
    TRADE_BATCH,
    WINDOW_SIZE,
    full_run,
    make_ensemble_fn,
    make_labeled_dataset,
    train_one_model,
)

# Features 9-16 are the 8 new v11 additions
NEW_FEATURE_INDICES = list(range(9, 17))
NEW_FEATURE_NAMES = V9_FEATURE_NAMES[9:17]

# Ablation configs: (name, indices_to_zero)
ABLATION_CONFIGS = [
    ("baseline (all 17)", []),
    ("drop all 8 new (back to v10 9-feat)", NEW_FEATURE_INDICES),
]
# Add individual feature drops
for idx, name in zip(NEW_FEATURE_INDICES, NEW_FEATURE_NAMES):
    ABLATION_CONFIGS.append((f"drop {name} (#{idx})", [idx]))


def eval_with_zeroed_features(ensemble_fn, symbols, drop_indices, params):
    """Evaluate ensemble with specific features zeroed out at test time."""
    p_ref = params or {}
    passing = []
    trades_all = 0
    worst_dd = 0.0

    for sym in symbols:
        try:
            env_test = make_env(
                sym,
                "test",
                window_size=WINDOW_SIZE,
                trade_batch=TRADE_BATCH,
                min_hold=MIN_HOLD,
            )
            # Zero out features at test time
            if drop_indices:
                env_test.features[:, drop_indices] = 0.0

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            sh = evaluate(
                env_test,
                ensemble_fn,
                min_trades=10,
                r_min=p_ref.get("r_min", 0.0),
                vpin_max_z=p_ref.get("vpin_max_z", 0.0),
                fee_mult=p_ref.get("fee_mult", 1.0),
            )
            sys.stdout = old
            out = buf.getvalue()

            t, d = 0, 0.0
            for ln in out.strip().split("\n"):
                if ln.startswith("num_trades:"):
                    t = int(ln.split()[1])
                elif ln.startswith("max_drawdown:"):
                    d = float(ln.split()[1])

            passed = (t >= 10 and d <= 0.20) if t > 0 else False
            if passed:
                passing.append(sh)
            trades_all += t
            worst_dd = max(worst_dd, d)
        except Exception as e:
            print(f"  {sym}: ERROR ({e})", file=sys.stderr)

    mean_sortino = float(np.mean(passing)) if passing else 0.0
    return mean_sortino, len(passing), trades_all, worst_dd


def main():
    print("=" * 70)
    print("FEATURE ABLATION — v11 (17 features)")
    print("=" * 70)
    print(f"Params: {BEST_PARAMS}")
    print(f"Seeds: {FINAL_SEEDS}, Symbols: {len(DEFAULT_SYMBOLS)}")
    print()

    # Step 1: Train ensemble on all 17 features
    print("Step 1: Training ensemble on all 17 features...")
    t0 = time.time()

    # Suppress output during env loading
    old_stdout = sys.stdout
    sys.stdout = open("/dev/null", "w")
    train_envs = {}
    for sym in DEFAULT_SYMBOLS:
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
        model, steps, updates = train_one_model(
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

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s\n")

    ensemble_fn = make_ensemble_fn(models, DEVICE)

    # Step 2: Run ablation experiments
    print("Step 2: Running ablation experiments...")
    print(
        f"{'Config':<45} {'Sortino':>8} {'Pass':>6} {'Trades':>7} {'MaxDD':>7} {'Delta':>8}"
    )
    print("-" * 85)

    baseline_sortino = None
    results = []

    for name, drop_indices in ABLATION_CONFIGS:
        t1 = time.time()
        sortino, passing, trades, max_dd = eval_with_zeroed_features(
            ensemble_fn,
            DEFAULT_SYMBOLS,
            drop_indices,
            BEST_PARAMS,
        )
        elapsed = time.time() - t1

        if baseline_sortino is None:
            baseline_sortino = sortino
            delta = ""
        else:
            diff = sortino - baseline_sortino
            delta = f"{diff:+.4f}"

        print(
            f"{name:<45} {sortino:>8.4f} {passing:>4}/25 {trades:>7} {max_dd:>7.4f} {delta:>8}"
        )
        results.append((name, sortino, passing, trades, max_dd))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_s = results[0][1]
    v10_s = results[1][1]
    print(f"\nBaseline (17 feat): Sortino={baseline_s:.4f}, {results[0][2]}/25 passing")
    print(f"v10 only (9 feat):  Sortino={v10_s:.4f}, {results[1][2]}/25 passing")

    if v10_s > baseline_s:
        print("\n>>> NEW FEATURES ARE HURTING. v10 (9 feat) > v11 (17 feat).")
    else:
        print("\n>>> New features provide lift over v10 baseline.")

    # Rank individual feature drops by impact
    individual = results[2:]  # skip baseline and "drop all 8"
    ranked = sorted(individual, key=lambda x: x[1], reverse=True)

    print("\nFeature impact (drop = higher Sortino means feature HURTS):")
    for name, sortino, passing, trades, max_dd in ranked:
        diff = sortino - baseline_s
        direction = "HURTS" if diff > 0.005 else "HELPS" if diff < -0.005 else "neutral"
        print(f"  {name:<40} Sortino={sortino:.4f} ({diff:+.4f}) [{direction}]")


if __name__ == "__main__":
    main()
