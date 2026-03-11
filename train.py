#!/usr/bin/env python3
"""RL trading agent — Optuna hyperparameter search with SB3 PPO."""

import contextlib
import io
import os
import sys
import time

import gymnasium
import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from prepare import DEFAULT_SYMBOLS, TRAIN_BUDGET_SECONDS, evaluate, make_env

# ── Configuration ──────────────────────────────────────────────
SEARCH_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "CRV"]
SEARCH_BUDGET = 90  # seconds per search trial
SEARCH_TRIALS = 20
FINAL_SEEDS = 3
FINAL_BUDGET = TRAIN_BUDGET_SECONDS  # 300s
WINDOW_SIZE = 50
TRADE_BATCH = 100
BATCH_SIZE = 256

# SB3 recommends CPU for MlpPolicy (GPU transfer overhead > compute for small MLPs)
DEVICE = torch.device("cpu")


# ── Utilities ──────────────────────────────────────────────────
@contextlib.contextmanager
def quiet():
    """Suppress stdout (for noisy cache-loading messages)."""
    old = sys.stdout
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old
        devnull.close()


class TimeBudget(BaseCallback):
    """Stop SB3 training after a wall-clock budget."""

    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        return (time.time() - self.t0) < self.seconds


# ── Reward wrapper ─────────────────────────────────────────────
class SortinoReward(gymnasium.Wrapper):
    """Sortino-style reward with Welford running normalization."""

    def __init__(self, env, lam_vol=0.5, lam_draw=1.0):
        super().__init__(env)
        self.lam_vol = lam_vol
        self.lam_draw = lam_draw
        self._hist = []
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def reset(self, **kw):
        self._hist = []
        return self.env.reset(**kw)

    def step(self, action):
        obs, _, done, trunc, info = self.env.step(action)
        r = self._raw_reward(info)
        r = self._normalize(r)
        return obs, float(r), done, trunc, info

    def _raw_reward(self, info):
        pnl = info["step_pnl"]
        self._hist.append(pnl)
        if len(self._hist) > 100:
            self._hist = self._hist[-100:]
        dv = 0.0
        if len(self._hist) > 10:
            neg = [p for p in self._hist if p < 0]
            if len(neg) > 2:
                dv = float(np.std(neg))
        return pnl - self.lam_vol * dv - self.lam_draw * info["drawdown"]

    def _normalize(self, x):
        self._n += 1
        d = x - self._mean
        self._mean += d / self._n
        self._m2 += d * (x - self._mean)
        if self._n < 2:
            return x
        return x / max(np.sqrt(self._m2 / self._n), 1e-8)


# ── Environment factories ─────────────────────────────────────
def _env_factory(sym, split, lv, ld):
    """Return a callable that creates a reward-wrapped env."""

    def _init():
        with quiet():
            env = make_env(sym, split, window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH)
        return SortinoReward(env, lam_vol=lv, lam_draw=ld)

    return _init


# ── Training ───────────────────────────────────────────────────
def train_model(symbols, p, budget, seed=0, tb_log=None):
    """Train one PPO model with SB3, return the model."""
    n_envs = len(symbols)
    vec = DummyVecEnv(
        [_env_factory(s, "train", p["lam_vol"], p["lam_draw"]) for s in symbols]
    )

    n_steps = p["n_steps"]
    buf = n_steps * n_envs
    bs = min(BATCH_SIZE, buf)
    while buf % bs != 0:
        bs -= 1

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=p["lr"],
        n_steps=n_steps,
        batch_size=bs,
        n_epochs=p["n_epochs"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lam"],
        ent_coef=p["ent"],
        clip_range=p["clip"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device=DEVICE,
        tensorboard_log=tb_log,
        verbose=0,
        policy_kwargs=dict(net_arch=[p["hdim"]] * p["nlayers"]),
    )

    model.learn(total_timesteps=999_999_999, callback=TimeBudget(budget))
    vec.close()
    return model


# ── Evaluation ─────────────────────────────────────────────────
def _run_eval(policy_fn, symbols, split="test"):
    """Run policy through symbols on given split. Returns (sharpe, passing, trades, dd)."""
    passing = []
    trades_all = 0
    worst_dd = 0.0

    for sym in symbols:
        try:
            with quiet():
                et = make_env(
                    sym, split, window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH
                )
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            sh = evaluate(et, policy_fn)
            sys.stdout = old
            out = buf.getvalue()

            t, d = 0, 0.0
            for ln in out.strip().split("\n"):
                if ln.startswith("num_trades:"):
                    t = int(ln.split()[1])
                elif ln.startswith("max_drawdown:"):
                    d = float(ln.split()[1])

            tag = "PASS" if sh > 0 else "FAIL"
            print(f"  {sym}: sharpe={sh:.4f} trades={t} dd={d:.4f} [{tag}]")

            if sh > 0:
                passing.append(sh)
            trades_all += t
            worst_dd = max(worst_dd, d)
        except Exception as e:
            print(f"  {sym}: ERROR ({e})")

    mean_sh = float(np.mean(passing)) if passing else 0.0
    return mean_sh, len(passing), trades_all, worst_dd


def eval_single(model, symbols, split="test"):
    """Evaluate a single SB3 model."""

    def fn(obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)

    return _run_eval(fn, symbols, split)


def eval_ensemble(models, symbols, split="test"):
    """Evaluate ensemble by averaging logits across models."""

    def fn(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = None
        with torch.no_grad():
            for m in models:
                feat = m.policy.extract_features(obs_t, m.policy.features_extractor)
                lp, _ = m.policy.mlp_extractor(feat)
                lg = m.policy.action_net(lp)
                logits = lg if logits is None else logits + lg
        return logits.argmax(-1).item()

    return _run_eval(fn, symbols, split)


# ── Optuna objective ───────────────────────────────────────────
def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 5e-5, 3e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
        "n_epochs": 4,
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lam": 0.95,
        "ent": trial.suggest_float("ent", 0.001, 0.05, log=True),
        "clip": 0.2,
        "hdim": trial.suggest_categorical("hdim", [64, 128, 256, 512]),
        "nlayers": 3,
        "lam_vol": trial.suggest_float("lam_vol", 0.0, 2.0),
        "lam_draw": trial.suggest_float("lam_draw", 0.1, 5.0, log=True),
    }

    print(f"\n{'='*50}")
    print(f"Trial {trial.number}")
    for k in ["lr", "n_steps", "gamma", "ent", "hdim", "lam_vol", "lam_draw"]:
        v = p[k]
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    try:
        t0 = time.time()
        m = train_model(SEARCH_SYMBOLS, p, SEARCH_BUDGET, seed=trial.number)
        # Evaluate on VAL split (not test) to avoid overfitting during search
        sh, ps, tr, dd = eval_single(m, SEARCH_SYMBOLS, split="val")
        elapsed = time.time() - t0
        print(
            f"  => sharpe={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} "
            f"trades={tr} dd={dd:.4f} ({elapsed:.0f}s)"
        )
        del m
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return sh
    except Exception as e:
        print(f"  => FAILED: {e}")
        return -999.0


# ── Main ───────────────────────────────────────────────────────
def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"device: {DEVICE}")
    print(
        f"=== SEARCH: {SEARCH_TRIALS} trials x {SEARCH_BUDGET}s "
        f"on {SEARCH_SYMBOLS} ===\n"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage="sqlite:///optuna_study.db",
        study_name="ppo_v3",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=SEARCH_TRIALS)

    # Top results
    print(f"\n{'='*50}")
    print("TOP 5 TRIALS:")
    ranked = sorted(
        study.trials, key=lambda t: t.value if t.value else -999, reverse=True
    )
    for t in ranked[:5]:
        print(f"  #{t.number}: sharpe={t.value:.4f}  {t.params}")

    # Final training with best params
    b = study.best_params
    bp = {
        "lr": b["lr"],
        "n_steps": b["n_steps"],
        "n_epochs": 4,
        "gamma": b["gamma"],
        "gae_lam": 0.95,
        "ent": b["ent"],
        "clip": 0.2,
        "hdim": b["hdim"],
        "nlayers": 3,
        "lam_vol": b["lam_vol"],
        "lam_draw": b["lam_draw"],
    }
    seed_budget = FINAL_BUDGET // FINAL_SEEDS

    print(
        f"\n=== FINAL: {FINAL_SEEDS} seeds x {seed_budget}s, "
        f"all {len(DEFAULT_SYMBOLS)} symbols ==="
    )
    print(f"params: {bp}\n")

    models = []
    for s in range(FINAL_SEEDS):
        print(f"Training seed {s}...")
        models.append(
            train_model(DEFAULT_SYMBOLS, bp, seed_budget, seed=s, tb_log="./tb_logs")
        )

    # Ensemble evaluation on TEST split
    print("\n---")
    print("=== PER-SYMBOL EVALUATION ===")
    sh, ps, tr, dd = eval_ensemble(models, DEFAULT_SYMBOLS, split="test")

    print("---")
    print("=== PORTFOLIO SUMMARY ===")
    print(f"symbols_passing: {ps}/{len(DEFAULT_SYMBOLS)}")
    print(f"val_sharpe: {sh:.6f}")
    print(f"num_trades: {tr}")
    print(f"max_drawdown: {dd:.4f}")
    print(f"training_seconds: {FINAL_BUDGET:.1f}")
    print(f"total_steps: 0")
    print(f"num_updates: 0")
    print(f"\nbest_params: {bp}")


if __name__ == "__main__":
    main()
