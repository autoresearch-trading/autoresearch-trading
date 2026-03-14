#!/usr/bin/env python3
"""Supervised forward-return classifier for trading."""

import argparse
import hashlib
import io
import os
import sys
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.distributions import Categorical

from prepare import DEFAULT_SYMBOLS, TRAIN_BUDGET_SECONDS, evaluate, make_env

# ── Configuration ──────────────────────────────────────────────
SEARCH_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "CRV"]
SEARCH_BUDGET = 90
SEARCH_SEEDS = 2
SEARCH_TRIALS = 20
FINAL_SEEDS = 3
FINAL_BUDGET = TRAIN_BUDGET_SECONDS  # 300s
WINDOW_SIZE = 50
TRADE_BATCH = 100
MIN_HOLD = 200  # ~3h between trades — breakeven from fee analysis
FEE_BPS = 5
FORWARD_HORIZON = 200  # steps to look ahead for labeling

DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 1e-3,
    "hdim": 256,
    "nlayers": 3,
    "batch_size": 256,
    "fee_mult": 1.5,  # multiply fee by this to set label threshold
}


# ── Network ────────────────────────────────────────────────────
def _ortho_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class DirectionClassifier(nn.Module):
    def __init__(self, obs_shape, n_classes, hidden_dim, num_layers):
        super().__init__()
        n_time, n_feat = obs_shape
        flat_dim = n_time * n_feat + 2 * n_feat  # flat + temporal mean + std
        layers = [_ortho_init(nn.Linear(flat_dim, hidden_dim)), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([_ortho_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = _ortho_init(nn.Linear(hidden_dim, n_classes), gain=0.01)

    def forward(self, x):
        # x: (batch, time, feat)
        t_mean = x.mean(dim=1)
        t_std = x.std(dim=1)
        flat = x.flatten(start_dim=1)
        x = torch.cat([flat, t_mean, t_std], dim=1)
        return self.head(self.trunk(x))


# ── Data labeling ──────────────────────────────────────────────
def make_labeled_dataset(env, horizon, fee_threshold, max_samples=5000):
    """Extract (obs, label) pairs from env data using forward returns.
    Samples max_samples random indices to keep memory manageable."""
    features = env.features
    prices = env.prices
    n = len(features)
    window = env.window_size

    # Valid indices range
    valid_start = window
    valid_end = n - horizon
    if valid_end <= valid_start:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    # Sample random indices
    all_idx = np.arange(valid_start, valid_end)
    if len(all_idx) > max_samples:
        idx = np.random.choice(all_idx, max_samples, replace=False)
        idx.sort()
    else:
        idx = all_idx

    obs_list = []
    labels = []

    for i in idx:
        if prices[i] <= 0:
            continue
        fwd_return = (prices[i + horizon] - prices[i]) / prices[i]

        if fwd_return > fee_threshold:
            label = 1  # long
        elif fwd_return < -fee_threshold:
            label = 2  # short
        else:
            label = 0  # flat

        obs_list.append(features[i - window : i])
        labels.append(label)

    if not obs_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    return np.array(obs_list, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── Training ───────────────────────────────────────────────────
def train_one_model(train_envs, active_symbols, weights, obs_shape, p, budget, seed):
    """Train a classifier on forward return labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    fee_threshold = (2 * FEE_BPS / 10000) * p["fee_mult"]

    # Collect labeled data from all symbols
    all_obs = []
    all_labels = []
    for sym in active_symbols:
        env = train_envs[sym]
        obs, labels = make_labeled_dataset(env, FORWARD_HORIZON, fee_threshold)
        if len(obs) > 0:
            all_obs.append(obs)
            all_labels.append(labels)

    X = np.concatenate(all_obs)
    y = np.concatenate(all_labels)

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for cls, cnt in zip(unique, counts):
        names = {0: "flat", 1: "long", 2: "short"}
        print(f"    {names[cls]}: {cnt} ({100*cnt/total:.1f}%)")

    # Compute class weights for balanced training
    class_weights = torch.ones(3)
    for cls in range(3):
        cnt = (y == cls).sum()
        if cnt > 0:
            class_weights[cls] = total / (3 * cnt)

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

    model = DirectionClassifier(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    batch_size = p["batch_size"]
    start_time = time.time()
    total_steps = 0
    num_updates = 0
    best_loss = float("inf")

    while (time.time() - start_time) < budget:
        # Shuffle
        perm = torch.randperm(len(X_t), device=DEVICE)
        X_shuf = X_t[perm]
        y_shuf = y_t[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_t), batch_size):
            batch_x = X_shuf[start : start + batch_size]
            batch_y = y_shuf[start : start + batch_size]

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            num_updates += 1
            total_steps += len(batch_x)

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

    return model, total_steps, num_updates


# ── Evaluation ─────────────────────────────────────────────────
def eval_policy(policy_fn, symbols, split="test"):
    """Run policy_fn on all symbols. Returns (sharpe, passing, trades, dd)."""
    passing = []
    trades_all = 0
    worst_dd = 0.0

    for sym in symbols:
        try:
            env_test = make_env(
                sym,
                split,
                window_size=WINDOW_SIZE,
                trade_batch=TRADE_BATCH,
                min_hold=MIN_HOLD,
            )
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            sh = evaluate(env_test, policy_fn, min_trades=10)
            sys.stdout = old
            out = buf.getvalue()

            t, d = 0, 0.0
            for ln in out.strip().split("\n"):
                if ln.startswith("num_trades:"):
                    t = int(ln.split()[1])
                elif ln.startswith("max_drawdown:"):
                    d = float(ln.split()[1])

            passed = (t >= 10 and d <= 0.20) if t > 0 else False
            tag = "PASS" if passed else "FAIL"
            print(f"  {sym}: sharpe={sh:.4f} trades={t} dd={d:.4f} [{tag}]")
            if passed:
                passing.append(sh)
            trades_all += t
            worst_dd = max(worst_dd, d)
        except Exception as e:
            print(f"  {sym}: ERROR ({e})")

    return (
        float(np.mean(passing)) if passing else 0.0,
        len(passing),
        trades_all,
        worst_dd,
    )


def make_ensemble_fn(models, device):
    """Create ensemble policy function using argmax of summed logits."""

    def fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits_sum = None
            for m in models:
                logits = m(obs_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            return logits_sum.argmax(dim=-1).item()

    return fn


# ── Full training run ──────────────────────────────────────────
def full_run(symbols, p, budget, n_seeds, split="test", verbose=True):
    """Train n_seeds models, evaluate ensemble."""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    train_envs = {}
    env_weights = {}
    for sym in symbols:
        try:
            env = make_env(
                sym,
                "train",
                window_size=WINDOW_SIZE,
                trade_batch=TRADE_BATCH,
                min_hold=MIN_HOLD,
            )
            train_envs[sym] = env
            env_weights[sym] = env.num_steps
        except Exception:
            pass
    sys.stdout.close()
    sys.stdout = old_stdout

    active = list(train_envs.keys())
    weights = np.array([env_weights[s] for s in active], dtype=np.float64)
    weights /= weights.sum()
    obs_shape = train_envs[active[0]].observation_space.shape

    budget_per_seed = budget // n_seeds
    models = []
    total_steps_all = 0
    total_updates_all = 0
    for seed in range(n_seeds):
        if verbose:
            print(f"  Training seed {seed} ({budget_per_seed}s)...")
        model, steps, updates = train_one_model(
            train_envs, active, weights, obs_shape, p, budget_per_seed, seed
        )
        models.append(model)
        total_steps_all += steps
        total_updates_all += updates

    ensemble_fn = make_ensemble_fn(models, DEVICE)
    sh, ps, tr, dd = eval_policy(ensemble_fn, symbols, split=split)
    return sh, ps, tr, dd, total_steps_all, total_updates_all


# ── Optuna objective ───────────────────────────────────────────
def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hdim": trial.suggest_categorical("hdim", [128, 256]),
        "nlayers": trial.suggest_categorical("nlayers", [2, 3]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "fee_mult": trial.suggest_float("fee_mult", 1.0, 3.0),
    }

    print(f"\n{'='*50}")
    print(f"Trial {trial.number}")
    for k, v in p.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    try:
        t0 = time.time()
        sh, ps, tr, dd, _, _ = full_run(
            SEARCH_SYMBOLS, p, SEARCH_BUDGET, SEARCH_SEEDS, split="val", verbose=False
        )
        elapsed = time.time() - t0
        print(
            f"  => sharpe={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} "
            f"trades={tr} dd={dd:.4f} ({elapsed:.0f}s)"
        )
        return sh
    except Exception as e:
        print(f"  => FAILED: {e}")
        return -999.0


def _code_hash():
    with open(__file__, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"device: {DEVICE}")

    if args.search:
        print(
            f"=== SEARCH: {SEARCH_TRIALS} trials x {SEARCH_BUDGET}s "
            f"({SEARCH_SEEDS} seeds) on {SEARCH_SYMBOLS} ===\n"
        )
        study_name = f"sup_{_code_hash()}"
        print(f"Optuna study: {study_name}")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            storage="sqlite:///optuna_study.db",
            study_name=study_name,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=SEARCH_TRIALS)

        print(f"\n{'='*50}")
        print("TOP 5 TRIALS:")
        ranked = sorted(
            study.trials, key=lambda t: t.value if t.value else -999, reverse=True
        )
        for t in ranked[:5]:
            print(f"  #{t.number}: sharpe={t.value:.4f}  {t.params}")

        b = study.best_params
        bp = {
            "lr": b["lr"],
            "hdim": b["hdim"],
            "nlayers": b["nlayers"],
            "batch_size": b["batch_size"],
            "fee_mult": b["fee_mult"],
        }
        print(f"\nHint: update BEST_PARAMS in train.py with: {bp}")
    else:
        print("=== FAST MODE: using BEST_PARAMS ===\n")
        bp = BEST_PARAMS

    print(
        f"\n=== FINAL: {FINAL_SEEDS} seeds x "
        f"{FINAL_BUDGET // FINAL_SEEDS}s on all {len(DEFAULT_SYMBOLS)} symbols ==="
    )
    print(f"params: {bp}\n")

    sh, ps, tr, dd, total_steps, total_updates = full_run(
        DEFAULT_SYMBOLS, bp, FINAL_BUDGET, FINAL_SEEDS, split="test", verbose=True
    )

    print("---")
    print("=== PORTFOLIO SUMMARY ===")
    print(f"symbols_passing: {ps}/{len(DEFAULT_SYMBOLS)}")
    print(f"val_sharpe: {sh:.6f}")
    print(f"num_trades: {tr}")
    print(f"max_drawdown: {dd:.4f}")
    print(f"training_seconds: {FINAL_BUDGET:.1f}")
    print(f"total_steps: {total_steps}")
    print(f"num_updates: {total_updates}")
    print(f"\nbest_params: {bp}")


if __name__ == "__main__":
    main()
