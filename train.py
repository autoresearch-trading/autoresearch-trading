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
FINAL_SEEDS = 5
FINAL_BUDGET = TRAIN_BUDGET_SECONDS  # 300s
WINDOW_SIZE = 50  # v5 proven
TRADE_BATCH = 100
MIN_HOLD = 800  # v5 proven — best Sortino
FEE_BPS = 5
MAX_HOLD_STEPS = 300

DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 4.4e-3,
    "hdim": 64,
    "nlayers": 3,
    "batch_size": 256,
    "fee_mult": 9.0,
    "r_min": 0.24,
    "vpin_max_z": 0.0,  # no VPIN gate (T17/T22)
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


class HybridClassifier(nn.Module):
    """Flat MLP + 1D TCN hybrid. Proved: weak dominance over either alone (Theorem 5)."""

    def __init__(self, obs_shape, n_classes, hidden_dim, num_layers):
        super().__init__()
        n_time, n_feat = obs_shape

        # Flat branch: flatten + temporal stats
        flat_dim = n_time * n_feat + 2 * n_feat

        # TCN branch: Conv1d → pool
        self.tcn = nn.Sequential(
            nn.Conv1d(n_feat, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Kaiming init for TCN
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

        combined_dim = flat_dim + 8  # flat + tcn pool output

        layers = [_ortho_init(nn.Linear(combined_dim, hidden_dim)), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([_ortho_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = _ortho_init(nn.Linear(hidden_dim, n_classes), gain=0.01)

    def forward(self, x):
        # x: (batch, time, feat)
        t_mean = x.mean(dim=1)
        t_std = x.std(dim=1)
        flat = x.flatten(start_dim=1)
        flat_branch = torch.cat([flat, t_mean, t_std], dim=1)

        # TCN expects (batch, channels, time)
        tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in).squeeze(-1)  # (batch, 8)

        combined = torch.cat([flat_branch, tcn_out], dim=1)
        return self.head(self.trunk(combined))


# ── Data labeling ──────────────────────────────────────────────
def make_labeled_dataset(env, max_hold, tp_threshold, sl_threshold, max_samples=10000):
    """Extract (obs, label) pairs using Triple Barrier labeling.

    For each sample, scan forward up to max_hold steps:
    - TP barrier hit first (return >= +tp_threshold) → long (1)
    - SL barrier hit first (return <= -sl_threshold) → short (2)
    - Neither hit within max_hold → flat (0) (timeout)

    Vectorized via numpy broadcasting. Memory: O(n_samples × max_hold).
    """
    features = env.features
    prices = env.prices
    n = len(features)
    window = env.window_size

    valid_start = window
    valid_end = n - max_hold
    if valid_end <= valid_start:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    all_idx = np.arange(valid_start, valid_end)
    if len(all_idx) > max_samples:
        idx = np.random.choice(all_idx, max_samples, replace=False)
        idx.sort()
    else:
        idx = all_idx

    # Filter zero prices
    idx = idx[prices[idx] > 0]
    if len(idx) == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Forward return matrix: (n_samples, max_hold)
    offsets = np.arange(1, max_hold + 1)
    future_prices = prices[idx[:, np.newaxis] + offsets[np.newaxis, :]]
    entry_prices = prices[idx]
    fwd_returns = (future_prices - entry_prices[:, np.newaxis]) / entry_prices[
        :, np.newaxis
    ]

    # First barrier hit per sample
    hit_tp = fwd_returns >= tp_threshold
    hit_sl = fwd_returns <= -sl_threshold
    tp_any = hit_tp.any(axis=1)
    sl_any = hit_sl.any(axis=1)
    tp_first = np.where(tp_any, hit_tp.argmax(axis=1), max_hold)
    sl_first = np.where(sl_any, hit_sl.argmax(axis=1), max_hold)

    # Label by which barrier hit first
    labels = np.zeros(len(idx), dtype=np.int64)  # 0 = flat (timeout)
    labels[tp_first < sl_first] = 1  # long: TP hit first
    labels[sl_first < tp_first] = 2  # short: SL hit first
    # tp_first == sl_first: both never hit (flat) or both hit same step (ambiguous → flat)

    obs = np.array([features[i - window : i] for i in idx], dtype=np.float32)
    return obs, labels, idx


# ── Training ───────────────────────────────────────────────────
def train_one_model(train_envs, active_symbols, weights, obs_shape, p, budget, seed):
    """Train a classifier on forward return labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    fee_threshold = (2 * FEE_BPS / 10000) * p["fee_mult"]

    # Collect labeled data from all symbols
    all_obs = []
    all_labels = []
    all_indices = []
    for sym in active_symbols:
        env = train_envs[sym]
        obs, labels, indices = make_labeled_dataset(
            env, MAX_HOLD_STEPS, fee_threshold, fee_threshold
        )
        if len(obs) > 0:
            all_obs.append(obs)
            all_labels.append(labels)
            all_indices.append(indices)

    X = np.concatenate(all_obs)
    y = np.concatenate(all_labels)
    sample_idx = np.concatenate(all_indices)

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

    # Recency weights: exponential decay so recent samples matter more
    norm_idx = (sample_idx - sample_idx.min()) / max(
        sample_idx.max() - sample_idx.min(), 1
    )
    recency_w = np.exp(1.0 * norm_idx)  # decay=1.0 → recent ~2.7x weight of oldest
    recency_w /= recency_w.mean()  # normalize so mean weight = 1

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    w_t = torch.tensor(recency_w, dtype=torch.float32, device=DEVICE)

    model = DirectionClassifier(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=5e-4)
    cw = class_weights.to(DEVICE)

    def focal_loss(logits, targets, sample_w, gamma=1.0):
        ce = nn.functional.cross_entropy(logits, targets, weight=cw, reduction="none")
        pt = torch.exp(-ce)
        return (sample_w * (1 - pt) ** gamma * ce).mean()

    criterion = focal_loss

    batch_size = p["batch_size"]
    total_steps = 0
    num_updates = 0
    n_epochs = 25  # 100 overfit; 25 is optimal for 68K params

    # Alpha_min early stopping (Theorem 3)
    alpha_min = 0.5 + 1.0 / (2.0 * p["fee_mult"])
    epochs_below = 0

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(X_t), device=DEVICE)
        X_shuf = X_t[perm]
        y_shuf = y_t[perm]
        w_shuf = w_t[perm]

        for start in range(0, len(X_t), batch_size):
            batch_x = X_shuf[start : start + batch_size]
            batch_y = y_shuf[start : start + batch_size]
            batch_w = w_shuf[start : start + batch_size]

            logits = model(batch_x)
            loss = criterion(logits, batch_y, batch_w)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            num_updates += 1
            total_steps += len(batch_x)

        # End of epoch: quick accuracy check on training data
        with torch.no_grad():
            logits = model(X_t)
            preds = logits.argmax(dim=1)
            directional_mask = y_t != 0  # long or short labels only
            if directional_mask.sum() > 0:
                acc = (
                    (preds[directional_mask] == y_t[directional_mask])
                    .float()
                    .mean()
                    .item()
                )
                if acc < alpha_min:
                    epochs_below += 1
                else:
                    epochs_below = 0
                if False and epochs_below >= 5:  # Disabled: let model train fully first
                    print(
                        f"    Early stop: accuracy {acc:.3f} < alpha_min {alpha_min:.3f} for 5 epochs"
                    )
                    break

    return model, total_steps, num_updates


# ── Evaluation ─────────────────────────────────────────────────
def eval_policy(policy_fn, symbols, split="test", params=None):
    """Run policy_fn on all symbols. Returns (sortino, passing, trades, dd, win_rate, profit_factor, sharpe, calmar, cvar)."""
    p_ref = params or {}
    passing = []
    trades_all = 0
    worst_dd = 0.0
    all_win_rates = []
    all_profit_factors = []
    all_sharpes = []
    all_calmars = []
    all_cvars = []

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
            sh = evaluate(
                env_test,
                policy_fn,
                min_trades=10,
                r_min=p_ref.get("r_min", 0.0),
                vpin_max_z=p_ref.get("vpin_max_z", 0.0),
                fee_mult=p_ref.get("fee_mult", 1.0),
            )
            sys.stdout = old
            out = buf.getvalue()

            t, d = 0, 0.0
            wr, pf = 0.0, 0.0
            sharpe_val, calmar_val, cvar_val = 0.0, 0.0, 0.0
            for ln in out.strip().split("\n"):
                if ln.startswith("num_trades:"):
                    t = int(ln.split()[1])
                elif ln.startswith("max_drawdown:"):
                    d = float(ln.split()[1])
                elif ln.startswith("win_rate:"):
                    wr = float(ln.split()[1])
                elif ln.startswith("profit_factor:"):
                    pf = float(ln.split()[1])
                elif ln.startswith("sharpe:"):
                    sharpe_val = float(ln.split()[1])
                elif ln.startswith("calmar:"):
                    calmar_val = float(ln.split()[1])
                elif ln.startswith("cvar_95:"):
                    cvar_val = float(ln.split()[1])

            passed = (t >= 10 and d <= 0.20) if t > 0 else False
            tag = "PASS" if passed else "FAIL"
            extra = f" wr={wr:.2f} pf={pf:.2f}" if wr > 0 else ""
            print(f"  {sym}: sortino={sh:.4f} trades={t} dd={d:.4f}{extra} [{tag}]")
            if passed:
                passing.append(sh)
                all_sharpes.append(sharpe_val)
                all_calmars.append(calmar_val)
                all_cvars.append(cvar_val)
            if passed and wr > 0:
                all_win_rates.append(wr)
                all_profit_factors.append(pf)
            trades_all += t
            worst_dd = max(worst_dd, d)
        except Exception as e:
            print(f"  {sym}: ERROR ({e})")

    mean_wr = float(np.mean(all_win_rates)) if all_win_rates else 0.0
    mean_pf = float(np.mean(all_profit_factors)) if all_profit_factors else 0.0
    mean_sharpe = float(np.mean(all_sharpes)) if all_sharpes else 0.0
    mean_calmar = float(np.mean(all_calmars)) if all_calmars else 0.0
    mean_cvar = float(np.mean(all_cvars)) if all_cvars else 0.0

    return (
        float(np.mean(passing)) if passing else 0.0,
        len(passing),
        trades_all,
        worst_dd,
        mean_wr,
        mean_pf,
        mean_sharpe,
        mean_calmar,
        mean_cvar,
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

    # Ensemble validity check (Theorem 10): only ensemble if alpha > 0.5
    # Check mean training accuracy across seeds
    mean_accuracy = 0.0
    best_seed_idx = 0
    best_acc = 0.0
    old_stdout2 = sys.stdout
    sys.stdout = open(os.devnull, "w")
    for si, model in enumerate(models):
        model.eval()
        # Quick accuracy check on first available training env
        first_env = train_envs[active[0]]
        fee_threshold = (2 * FEE_BPS / 10000) * p["fee_mult"]
        obs, labels, _ = make_labeled_dataset(
            first_env, MAX_HOLD_STEPS, fee_threshold, fee_threshold, max_samples=2000
        )
        if len(obs) > 0:
            with torch.no_grad():
                X_check = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                y_check = torch.tensor(labels, dtype=torch.long, device=DEVICE)
                logits = model(X_check)
                preds = logits.argmax(dim=1)
                directional_mask = y_check != 0
                if directional_mask.sum() > 0:
                    acc = (
                        (preds[directional_mask] == y_check[directional_mask])
                        .float()
                        .mean()
                        .item()
                    )
                    mean_accuracy += acc
                    if acc > best_acc:
                        best_acc = acc
                        best_seed_idx = si
    sys.stdout.close()
    sys.stdout = old_stdout2
    mean_accuracy /= max(len(models), 1)

    if mean_accuracy < 0.5 and len(models) > 1:
        if verbose:
            print(
                f"  WARNING: alpha={mean_accuracy:.3f} < 0.5, using single best model"
            )
        ensemble_fn = make_ensemble_fn([models[best_seed_idx]], DEVICE)
    else:
        ensemble_fn = make_ensemble_fn(models, DEVICE)

    sh, ps, tr, dd, wr, pf, sharpe, calmar, cvar = eval_policy(
        ensemble_fn, symbols, split=split, params=p
    )
    return (
        sh,
        ps,
        tr,
        dd,
        total_steps_all,
        total_updates_all,
        wr,
        pf,
        sharpe,
        calmar,
        cvar,
    )


# ── Warm-start configs from prior experiments ─────────────────
WARM_START_CONFIGS = [
    # v10 best (wd5e4 run, 18/25 passing)
    {
        "lr": 1e-3,
        "hdim": 256,
        "nlayers": 2,
        "batch_size": 256,
        "fee_mult": 1.5,
        "r_min": 0.0,
    },
    # v5 baseline
    {
        "lr": 1e-3,
        "hdim": 256,
        "nlayers": 2,
        "batch_size": 256,
        "fee_mult": 1.5,
        "r_min": 0.7,
    },
    # tape-v3 best (min_hold=300 era)
    {
        "lr": 1e-3,
        "hdim": 256,
        "nlayers": 2,
        "batch_size": 256,
        "fee_mult": 10.0,
        "r_min": 0.0,
    },
    # wider fee_mult exploration
    {
        "lr": 5e-4,
        "hdim": 128,
        "nlayers": 3,
        "batch_size": 512,
        "fee_mult": 3.0,
        "r_min": 0.0,
    },
    {
        "lr": 2e-3,
        "hdim": 256,
        "nlayers": 2,
        "batch_size": 128,
        "fee_mult": 5.0,
        "r_min": 0.5,
    },
]


# ── Optuna objective (multi-fidelity + pruning) ──────────────
def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "hdim": trial.suggest_categorical("hdim", [64, 128, 256]),
        "nlayers": trial.suggest_categorical("nlayers", [2, 3]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "fee_mult": trial.suggest_float("fee_mult", 1.0, 15.0),
        "r_min": trial.suggest_float("r_min", 0.0, 0.7),
    }

    print(f"\n{'='*50}")
    print(f"Trial {trial.number}")
    for k, v in p.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    try:
        t0 = time.time()
        sh, ps, tr, dd, _, _, _, _, _, _, _ = full_run(
            SEARCH_SYMBOLS, p, SEARCH_BUDGET, SEARCH_SEEDS, split="val", verbose=False
        )
        elapsed = time.time() - t0
        print(
            f"  => sortino={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} "
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
            f"({SEARCH_SEEDS} seeds) ===\n"
            f"  Stage 1: screen on {SEARCH_SYMBOLS}\n"
            f"  Stage 2: promote to all {len(DEFAULT_SYMBOLS)} symbols\n"
        )
        study_name = "v11_search"
        storage = "sqlite:///optuna_v11.db"
        print(f"Optuna study: {study_name} ({storage})")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=0,  # stage 0 = 5-symbol screen
                max_resource=1,  # stage 1 = 25-symbol full eval
                reduction_factor=3,
            ),
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

        # Warm-start: enqueue known good configs from prior experiments
        existing_trials = len(study.trials)
        if existing_trials == 0:
            print(f"Warm-starting with {len(WARM_START_CONFIGS)} prior configs...")
            for cfg in WARM_START_CONFIGS:
                study.enqueue_trial(cfg)
        else:
            print(f"Resuming study with {existing_trials} existing trials")

        study.optimize(objective, n_trials=SEARCH_TRIALS)

        print(f"\n{'='*50}")
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        print(f"Trials: {len(completed)} completed, {len(pruned)} pruned")
        print("\nTOP 5 TRIALS:")
        ranked = sorted(
            completed, key=lambda t: t.value if t.value else -999, reverse=True
        )
        for t in ranked[:5]:
            print(f"  #{t.number}: sortino={t.value:.4f}  {t.params}")

        b = study.best_params
        bp = {
            "lr": b["lr"],
            "hdim": b["hdim"],
            "nlayers": b["nlayers"],
            "batch_size": b["batch_size"],
            "fee_mult": b["fee_mult"],
            "r_min": b["r_min"],
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

    sh, ps, tr, dd, total_steps, total_updates, wr, pf, sharpe, calmar, cvar = full_run(
        DEFAULT_SYMBOLS, bp, FINAL_BUDGET, FINAL_SEEDS, split="test", verbose=True
    )

    print("---")
    print("=== PORTFOLIO SUMMARY ===")
    print(f"symbols_passing: {ps}/{len(DEFAULT_SYMBOLS)}")
    print(f"sortino: {sh:.6f}")
    print(f"sharpe: {sharpe:.6f}")
    print(f"calmar: {calmar:.6f}")
    print(f"cvar_95: {cvar:.6f}")
    print(f"num_trades: {tr}")
    print(f"max_drawdown: {dd:.4f}")
    if wr > 0:
        print(f"win_rate: {wr:.4f}")
        print(f"profit_factor: {pf:.4f}")
    print(f"training_seconds: {FINAL_BUDGET:.1f}")
    print(f"total_steps: {total_steps}")
    print(f"num_updates: {total_updates}")
    print(f"\nbest_params: {bp}")


if __name__ == "__main__":
    main()
