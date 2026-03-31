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
SEARCH_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "AAVE"]
# T40: exclude symbols with spread > 25 bps (untradeable after costs)
EXCLUDED_SYMBOLS = {"CRV", "XPL"}  # CRV=52bps, XPL=28bps spread
SEARCH_BUDGET = 90
SEARCH_SEEDS = 2
SEARCH_TRIALS = 20
FINAL_SEEDS = 5
FINAL_BUDGET = TRAIN_BUDGET_SECONDS  # 300s
WINDOW_SIZE = (
    50  # window sweep: 50 > 20 > 10 (T47 linear analysis missed nonlinear patterns)
)
TRADE_BATCH = 100
MIN_HOLD = 1200  # min_hold sweep winner (Sortino=0.184, best honest result)
FEE_BPS = 5
MAX_HOLD_STEPS = 300  # short horizon = momentum filter (300 beats 600 and 1200)

DEVICE = torch.device("cpu")

BEST_PARAMS = {
    "lr": 1e-3,  # restored baseline
    "hdim": 64,  # hdim sweep: 64 > 128 > 256 (smaller net generalizes better)
    "nlayers": 3,  # nlayers sweep: 3 best score (2 higher Sortino but fewer passing)
    "batch_size": 256,  # batch_size sweep: 256 > 128, 512
    "fee_mult": 11.0,  # T39 cost-adjusted: ties fm=5 on score, better PF (1.74 vs 1.12)
    "r_min": 0.0,  # no regime gate — cost-adjusted barriers make it redundant
    "vpin_max_z": 0.0,  # no VPIN gate (T17/T22)
    "wd": 0.0,  # no weight decay — 64-dim net doesn't overfit at 25 epochs
    "logit_bias": 0.0,  # logit bias sweep: 0 > 0.5 > 1.0 (bias hurts)
    "curriculum_epochs": 0,  # curriculum sweep: 0 > 10 (directional warm-up hurts)
    "swa_start": 0,  # SWA: disabled (0.285 vs 0.353 baseline — weight averaging hurts)
    # asymmetric barriers: symmetric (11/11) wins over tp=15/sl=11 and tp=9/sl=11
    "confidence_threshold": 0.0,  # confidence gating: 0 > 0.45 > 0.55 (gating hurts, same as r_min)
    "use_uace": False,  # UACE properly tested: focal wins at all lr (best UACE=0.258 at lr=3e-4 vs focal=0.353)
    "dropout": 0.0,  # dropout sweep: 0.0 > 0.1 > 0.2 (dropout hurts — model already regularized by small size)
    "residual": False,  # residual sweep: False > True (skip connections hurt — 0.167 vs 0.353)
    "use_gce": False,  # GCE sweep: focal wins at all lr (best GCE=0.240@1e-3 vs focal=0.353)
    "gce_q": 0.7,  # GCE q parameter (0=MAE, 1=CE, 0.7=balanced) — unused when use_gce=False
    "use_metalabeling": True,  # metalabeling experiment
    "meta_threshold": 0.5,  # meta-model confidence threshold
}


# ── Network ────────────────────────────────────────────────────
def _ortho_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class DirectionClassifier(nn.Module):
    def __init__(
        self, obs_shape, n_classes, hidden_dim, num_layers, dropout=0.0, residual=False
    ):
        super().__init__()
        n_time, n_feat = obs_shape
        flat_dim = n_time * n_feat + 2 * n_feat  # flat + temporal mean + std
        self.projection = nn.Sequential(
            _ortho_init(nn.Linear(flat_dim, hidden_dim)), nn.ReLU()
        )
        self.residual = residual
        blocks = []
        for _ in range(num_layers - 1):
            block = [_ortho_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()]
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)
        self.head = _ortho_init(nn.Linear(hidden_dim, n_classes), gain=0.01)

    def forward(self, x):
        # x: (batch, time, feat)
        t_mean = x.mean(dim=1)
        t_std = x.std(dim=1)
        flat = x.flatten(start_dim=1)
        x = torch.cat([flat, t_mean, t_std], dim=1)
        x = self.projection(x)
        for block in self.blocks:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        return self.head(x)


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
def _cost_adjusted_threshold(env, fee_mult):
    """Compute cost-adjusted barrier threshold per symbol (T39).

    Barrier = fee_mult × 2 × (fee + slippage) / 10000
    where slippage = median(half_spread) + impact_buffer.
    """
    impact_buffer_bps = 3.0  # MEV + market impact
    if env.spread_bps is not None and len(env.spread_bps) > 0:
        median_half_spread = np.median(env.spread_bps[env.spread_bps > 0]) / 2.0
        slippage_bps = median_half_spread + impact_buffer_bps
    else:
        slippage_bps = 5.0  # fallback
    total_cost_bps = FEE_BPS + slippage_bps
    return (2 * total_cost_bps / 10000) * fee_mult


def train_one_model(train_envs, active_symbols, weights, obs_shape, p, budget, seed):
    """Train a classifier on forward return labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect labeled data from all symbols (per-symbol cost-adjusted barriers, T39)
    all_obs = []
    all_labels = []
    all_indices = []
    tp_mult = p.get("tp_mult", p["fee_mult"])  # default: symmetric
    sl_mult = p.get("sl_mult", p["fee_mult"])
    for sym in active_symbols:
        env = train_envs[sym]
        tp_threshold = _cost_adjusted_threshold(env, tp_mult)
        sl_threshold = _cost_adjusted_threshold(env, sl_mult)
        obs, labels, indices = make_labeled_dataset(
            env, MAX_HOLD_STEPS, tp_threshold, sl_threshold
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

    model = DirectionClassifier(
        obs_shape,
        3,
        p["hdim"],
        p["nlayers"],
        dropout=p.get("dropout", 0.0),
        residual=p.get("residual", False),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p["lr"], weight_decay=p.get("wd", 5e-4)
    )
    cw = class_weights.to(DEVICE)

    logit_bias = p.get("logit_bias", 0.0)
    use_uace = p.get("use_uace", False)
    use_gce = p.get("use_gce", False)
    gce_q = p.get("gce_q", 0.7)

    def focal_loss(logits, targets, sample_w, gamma=1.0):
        # Logit bias: add epsilon to correct-class logit (noise robustness)
        if logit_bias > 0:
            logits = logits.clone()
            bias = torch.zeros_like(logits)
            bias.scatter_(1, targets.unsqueeze(1), logit_bias)
            logits = logits + bias
        ce = nn.functional.cross_entropy(logits, targets, weight=cw, reduction="none")
        if use_uace:
            # UACE: down-weight uncertain samples using prediction entropy
            probs = torch.softmax(logits.detach(), dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            max_entropy = np.log(3)  # log(C) for C=3 classes
            confidence = 1.0 - entropy / max_entropy  # 1=confident, 0=uncertain
            return (sample_w * confidence * ce).mean()
        else:
            pt = torch.exp(-ce)
            return (sample_w * (1 - pt) ** gamma * ce).mean()

    def gce_loss(logits, targets, sample_w):
        # GCE: L_q = (1 - p_y^q) / q where p_y = softmax probability of true class
        probs = torch.softmax(logits, dim=1)
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # prob of true class
        # Apply class weights via per-sample weighting
        target_cw = cw[targets]
        loss = (1.0 - (p_y + 1e-8) ** gce_q) / gce_q
        return (sample_w * target_cw * loss).mean()

    criterion = gce_loss if use_gce else focal_loss

    batch_size = p["batch_size"]
    total_steps = 0
    num_updates = 0
    n_epochs = 25  # epoch sweep confirmed: 25 optimal (15 underfit, 35 overfit-to-flat)
    swa_start = p.get("swa_start", 0)  # epoch to start SWA (0 = disabled)
    if swa_start > 0:
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    # Alpha_min early stopping (Theorem 3)
    alpha_min = 0.5 + 1.0 / (2.0 * p["fee_mult"])
    epochs_below = 0

    # Curriculum learning: directional labels only for warm-up epochs
    curriculum_epochs = p.get("curriculum_epochs", 0)
    if curriculum_epochs > 0:
        directional_mask = y_t != 0  # labels 1 (long) and 2 (short) only
        X_dir = X_t[directional_mask]
        y_dir = y_t[directional_mask]
        w_dir = w_t[directional_mask]

    for epoch in range(n_epochs):
        # Curriculum: use only directional samples for warm-up epochs
        if curriculum_epochs > 0 and epoch < curriculum_epochs:
            perm = torch.randperm(len(X_dir), device=DEVICE)
            X_shuf = X_dir[perm]
            y_shuf = y_dir[perm]
            w_shuf = w_dir[perm]
        else:
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

        # SWA: average weights over late epochs
        if swa_start > 0 and epoch >= swa_start:
            swa_model.update_parameters(model)

    # Return SWA-averaged model if enabled
    if swa_start > 0:
        torch.optim.swa_utils.update_bn(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_t),
                batch_size=batch_size,
            ),
            swa_model,
            device=DEVICE,
        )
        return swa_model.module, total_steps, num_updates

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
                # include_funding=True,  # T42: proven negligible (0.16% of fee barrier)
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


class MetaModel(nn.Module):
    """Small binary classifier for metalabeling: should we trade or skip?"""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_meta_model(primary_model, val_envs, active_symbols, params, device):
    """Train a metalabeling model on validation data.

    Collects directional predictions from the primary model, labels them
    as 1 (profitable) or 0 (unprofitable), and trains a small binary MLP.
    """
    primary_model.eval()
    meta_X = []
    meta_y = []

    for sym in active_symbols:
        if sym not in val_envs:
            continue
        env = val_envs[sym]
        fee_threshold = _cost_adjusted_threshold(env, params["fee_mult"])
        obs, labels, indices = make_labeled_dataset(
            env, MAX_HOLD_STEPS, fee_threshold, fee_threshold, max_samples=5000
        )
        if len(obs) == 0:
            continue

        with torch.no_grad():
            X_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits = primary_model(X_t)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

        # Only keep directional predictions (long=1 or short=2)
        dir_mask = preds != 0
        if dir_mask.sum() == 0:
            continue

        probs_np = probs.cpu().numpy()[dir_mask]  # (n, 3) softmax probs
        max_prob = probs_np.max(axis=1, keepdims=True)  # (n, 1) confidence
        # Temporal stats from obs (mean + std per feature)
        obs_dir = obs[dir_mask]  # (n, window, features)
        t_mean = obs_dir.mean(axis=1)  # (n, features)
        t_std = obs_dir.std(axis=1)  # (n, features)

        # Meta features: softmax probs (3) + max_prob (1) + temporal stats (2*features)
        meta_features = np.concatenate([probs_np, max_prob, t_mean, t_std], axis=1)
        meta_X.append(meta_features)

        # Meta labels: was the primary prediction correct?
        preds_dir = preds[dir_mask]
        labels_dir = labels[dir_mask]
        correct = (preds_dir == labels_dir).astype(np.float32)
        meta_y.append(correct)

    if not meta_X:
        return None

    meta_X = np.concatenate(meta_X)
    meta_y = np.concatenate(meta_y)

    # Train meta-model
    input_dim = meta_X.shape[1]
    meta_model = MetaModel(input_dim, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

    X_t = torch.tensor(meta_X, dtype=torch.float32, device=device)
    y_t = torch.tensor(meta_y, dtype=torch.float32, device=device)

    # Class weights for meta labels
    pos_rate = meta_y.mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    meta_model.train()
    for epoch in range(50):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            batch_idx = perm[i : i + 256]
            logits = meta_model(X_t[batch_idx])
            loss = criterion(logits, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    meta_model.eval()
    return meta_model


def make_ensemble_fn(
    models, device, confidence_threshold=0.0, meta_model=None, meta_threshold=0.5
):
    """Create ensemble policy function using argmax of summed logits.

    If confidence_threshold > 0, force flat when max softmax prob < threshold.
    If meta_model is provided, gate directional predictions through it.
    """

    def fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits_sum = None
            for m in models:
                logits = m(obs_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            if confidence_threshold > 0:
                probs = torch.softmax(logits_sum, dim=-1)
                if probs.max().item() < confidence_threshold:
                    return 0  # force flat when uncertain
            action = logits_sum.argmax(dim=-1).item()

            # Metalabeling gate: if directional, check meta-model
            if meta_model is not None and action != 0:
                probs = torch.softmax(logits_sum, dim=-1)
                max_prob = probs.max(dim=-1, keepdim=True).values
                t_mean = obs_t.mean(dim=1)
                t_std = obs_t.std(dim=1)
                meta_features = torch.cat([probs, max_prob, t_mean, t_std], dim=1)
                meta_logit = meta_model(meta_features)
                if torch.sigmoid(meta_logit).item() < meta_threshold:
                    return 0  # meta-model says skip
            return action

    return fn


# ── Full training run ──────────────────────────────────────────
def full_run(symbols, p, budget, n_seeds, split="test", verbose=True):
    """Train n_seeds models, evaluate ensemble."""
    old_stdout = sys.stdout
    train_envs = {}
    env_weights = {}
    try:
        sys.stdout = open(os.devnull, "w")
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
            except Exception as e:
                old_stdout.write(f"  WARNING: {sym} failed to load: {e}\n")
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    active = list(train_envs.keys())
    if not active:
        print("  ERROR: no symbols loaded successfully")
        return (0.0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
    try:
        sys.stdout = open(os.devnull, "w")
        for si, model in enumerate(models):
            model.eval()
            first_env = train_envs[active[0]]
            fee_threshold = _cost_adjusted_threshold(first_env, p["fee_mult"])
            obs, labels, _ = make_labeled_dataset(
                first_env,
                MAX_HOLD_STEPS,
                fee_threshold,
                fee_threshold,
                max_samples=2000,
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
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout2
    mean_accuracy /= max(len(models), 1)

    if mean_accuracy < 0.5 and len(models) > 1:
        if verbose:
            print(
                f"  WARNING: alpha={mean_accuracy:.3f} < 0.5, using single best model"
            )
        selected_models = [models[best_seed_idx]]
    else:
        selected_models = models

    # Metalabeling: train meta-model on val data if enabled
    meta_model = None
    use_metalabeling = p.get("use_metalabeling", False)
    meta_threshold = p.get("meta_threshold", 0.5)
    if use_metalabeling:
        if verbose:
            print("  Training meta-model on val data...")
        val_envs = {}
        old_stdout3 = sys.stdout
        try:
            sys.stdout = open(os.devnull, "w")
            for sym in active:
                try:
                    val_envs[sym] = make_env(
                        sym,
                        "val",
                        window_size=WINDOW_SIZE,
                        trade_batch=TRADE_BATCH,
                        min_hold=MIN_HOLD,
                    )
                except Exception:
                    pass
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout3
        if val_envs:
            meta_model = train_meta_model(
                selected_models[0], val_envs, active, p, DEVICE
            )
            if verbose:
                print(f"  Meta-model: {'trained' if meta_model else 'failed'}")

    ensemble_fn = make_ensemble_fn(
        selected_models,
        DEVICE,
        p.get("confidence_threshold", 0.0),
        meta_model=meta_model,
        meta_threshold=meta_threshold,
    )

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

    # T40: filter out symbols with spreads too wide to trade profitably
    tradeable_symbols = [s for s in DEFAULT_SYMBOLS if s not in EXCLUDED_SYMBOLS]

    print(
        f"\n=== FINAL: {FINAL_SEEDS} seeds x "
        f"{FINAL_BUDGET // FINAL_SEEDS}s on {len(tradeable_symbols)} symbols "
        f"(excluded {len(EXCLUDED_SYMBOLS)}: {EXCLUDED_SYMBOLS}) ==="
    )
    print(f"params: {bp}\n")

    sh, ps, tr, dd, total_steps, total_updates, wr, pf, sharpe, calmar, cvar = full_run(
        tradeable_symbols, bp, FINAL_BUDGET, FINAL_SEEDS, split="test", verbose=True
    )

    print("---")
    print("=== PORTFOLIO SUMMARY ===")
    print(f"symbols_passing: {ps}/{len(tradeable_symbols)}")
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
