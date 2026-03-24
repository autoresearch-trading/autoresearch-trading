#!/usr/bin/env python3
"""Permutation importance for v5 features (31).

Trains v5 model, then for each feature: shuffle it across test data,
re-evaluate Sortino, measure the drop. Bigger drop = more important feature.
"""

import io
import sys

import numpy as np
import torch

from prepare import DEFAULT_SYMBOLS, evaluate, make_env

# v5 config (Run 7)
WINDOW_SIZE = 50
TRADE_BATCH = 100
MIN_HOLD = 800
FEE_BPS = 5
MAX_HOLD_STEPS = 300
DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 1e-3,
    "hdim": 256,
    "nlayers": 2,
    "batch_size": 256,
    "fee_mult": 1.5,
}

# Import model and training from train
from train import (
    DirectionClassifier,
    _ortho_init,
    make_ensemble_fn,
    make_labeled_dataset,
)


def train_models(n_seeds=5):
    """Train n_seeds models on all symbols, return models + obs_shape."""
    print("Loading training data...")
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
    obs_shape = train_envs[active[0]].observation_space.shape
    print(f"Training on {len(active)} symbols, obs_shape={obs_shape}")

    fee_threshold = (2 * FEE_BPS / 10000) * BEST_PARAMS["fee_mult"]
    p = BEST_PARAMS

    models = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        all_obs, all_labels, all_indices = [], [], []
        for sym in active:
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

        # Class weights
        total = len(y)
        class_weights = torch.ones(3)
        for cls in range(3):
            cnt = (y == cls).sum()
            if cnt > 0:
                class_weights[cls] = total / (3 * cnt)

        # Recency weights
        norm_idx = (sample_idx - sample_idx.min()) / max(
            sample_idx.max() - sample_idx.min(), 1
        )
        recency_w = np.exp(1.0 * norm_idx)
        recency_w /= recency_w.mean()

        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
        w_t = torch.tensor(recency_w, dtype=torch.float32, device=DEVICE)

        model = DirectionClassifier(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=5e-4)
        cw = class_weights.to(DEVICE)

        def focal_loss(logits, targets, sample_w, gamma=1.0):
            ce = torch.nn.functional.cross_entropy(
                logits, targets, weight=cw, reduction="none"
            )
            pt = torch.exp(-ce)
            return (sample_w * (1 - pt) ** gamma * ce).mean()

        batch_size = p["batch_size"]
        for epoch in range(25):
            perm = torch.randperm(len(X_t), device=DEVICE)
            X_shuf, y_shuf, w_shuf = X_t[perm], y_t[perm], w_t[perm]
            for start in range(0, len(X_t), batch_size):
                bx = X_shuf[start : start + batch_size]
                by = y_shuf[start : start + batch_size]
                bw = w_shuf[start : start + batch_size]
                logits = model(bx)
                loss = focal_loss(logits, by, bw)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        models.append(model)

    return models, obs_shape


def eval_sortino(models, symbol, permute_feature=None):
    """Evaluate ensemble on one symbol. Optionally permute a feature index."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf

    env_test = make_env(
        symbol,
        "test",
        window_size=WINDOW_SIZE,
        trade_batch=TRADE_BATCH,
        min_hold=MIN_HOLD,
    )

    # If permuting, shuffle that feature across the entire feature matrix
    if permute_feature is not None:
        rng = np.random.RandomState(42)
        perm_idx = rng.permutation(len(env_test.features))
        env_test.features[:, permute_feature] = env_test.features[
            perm_idx, permute_feature
        ]

    ensemble_fn = make_ensemble_fn(models, DEVICE)
    sortino = evaluate(env_test, ensemble_fn, min_trades=10)

    sys.stdout = old
    return sortino


def main():
    models, obs_shape = train_models(n_seeds=5)
    n_features = obs_shape[1]

    # Feature names from CLAUDE.md
    feature_names = [
        "returns",
        "r_5",
        "r_20",
        "r_100",  # 0-3
        "realvol_10",
        "bipower_var_20",  # 4-5
        "tfi",
        "volume_spike_ratio",
        "large_trade_share",  # 6-8
        "kyle_lambda_50",
        "amihud_illiq_50",
        "trade_arrival_rate",  # 9-11
        "spread_bps",
        "log_total_depth",
        "weighted_imbalance_5lvl",  # 12-14
        "microprice_dev",
        "ofi",
        "ob_slope_asym",  # 15-17
        "funding_zscore",
        "utc_hour_linear",  # 18-19
        "r_500",
        "r_2800",
        "cum_tfi_100",
        "cum_tfi_500",  # 20-23
        "funding_rate_raw",  # 24
        "VPIN",
        "delta_TFI",  # 25-26
        "Hurst",
        "realized_skew",
        "vol_of_vol",
        "sign_autocorr",  # 27-30
    ]
    if len(feature_names) < n_features:
        feature_names.extend(
            [f"feat_{i}" for i in range(len(feature_names), n_features)]
        )

    # Baseline: evaluate all symbols without permutation
    print(f"\n=== Baseline evaluation (no permutation) ===")
    baseline_sortinos = {}
    for sym in DEFAULT_SYMBOLS:
        try:
            s = eval_sortino(models, sym)
            baseline_sortinos[sym] = s
            print(f"  {sym}: {s:.4f}")
        except Exception as e:
            print(f"  {sym}: ERROR ({e})")

    baseline_mean = np.mean([v for v in baseline_sortinos.values() if v != 0])
    print(f"  Mean Sortino: {baseline_mean:.4f}")

    # Permutation importance: shuffle each feature, measure Sortino drop
    print(f"\n=== Permutation Importance ({n_features} features) ===")
    importance = {}

    for feat_idx in range(n_features):
        feat_name = (
            feature_names[feat_idx]
            if feat_idx < len(feature_names)
            else f"feat_{feat_idx}"
        )
        sortinos = {}
        for sym in DEFAULT_SYMBOLS:
            try:
                s = eval_sortino(models, sym, permute_feature=feat_idx)
                sortinos[sym] = s
            except Exception:
                pass

        perm_mean = np.mean([v for v in sortinos.values() if v != 0]) if sortinos else 0
        drop = baseline_mean - perm_mean
        importance[feat_idx] = drop
        print(
            f"  [{feat_idx:2d}] {feat_name:<25s} perm_sortino={perm_mean:+.4f}  drop={drop:+.4f}"
        )

    # Rank by importance
    print(f"\n=== FEATURE RANKING (by Sortino drop when shuffled) ===")
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    for rank, (idx, drop) in enumerate(ranked, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        bar = "█" * max(0, int(drop * 200))
        print(f"  {rank:2d}. [{idx:2d}] {name:<25s} drop={drop:+.6f} {bar}")


if __name__ == "__main__":
    main()
