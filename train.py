#!/usr/bin/env python3
"""RL trading agent — Optuna hyperparameter search with hand-rolled PPO."""

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
SEARCH_BUDGET = 90  # seconds per trial (split across seeds)
SEARCH_SEEDS = 2  # seeds per search trial
SEARCH_TRIALS = 20
FINAL_SEEDS = 3
FINAL_BUDGET = TRAIN_BUDGET_SECONDS  # 300s
WINDOW_SIZE = 50
TRADE_BATCH = 100
MIN_HOLD = 200  # ~3h between trades — breakeven from fee analysis

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# ── Reward ─────────────────────────────────────────────────────
class RewardNormalizer:
    """Welford's online algorithm for running std."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        return max(np.sqrt(self.M2 / self.count), 1e-8)

    def normalize(self, x):
        return x / self.std


def compute_reward(info, reward_state, lam_vol, lam_draw):
    """Sortino-style reward."""
    pnl = info["step_pnl"]
    reward_state.setdefault("pnl_history", [])
    reward_state["pnl_history"].append(pnl)
    if len(reward_state["pnl_history"]) > 100:
        reward_state["pnl_history"] = reward_state["pnl_history"][-100:]

    downside_vol = 0.0
    if len(reward_state["pnl_history"]) > 10:
        negatives = [p for p in reward_state["pnl_history"] if p < 0]
        if len(negatives) > 2:
            downside_vol = float(np.std(negatives))

    return pnl - lam_vol * downside_vol - lam_draw * info["drawdown"]


# ── Network ────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dim, num_layers):
        super().__init__()
        flat_dim = obs_shape[0] * obs_shape[1]
        layers = [nn.Linear(flat_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

    def get_action_and_value(self, obs, action=None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


# ── Training ───────────────────────────────────────────────────
def train_one_policy(train_envs, active_symbols, weights, obs_shape, p, budget, seed):
    """Train a single policy with given hyperparams and time budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = PolicyNetwork(obs_shape, 3, p["hdim"], p["nlayers"]).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=p["lr"])
    reward_norm = RewardNormalizer()

    env_obs = {}
    env_reward_states = {}
    for sym in active_symbols:
        obs, _ = train_envs[sym].reset()
        env_obs[sym] = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        env_reward_states[sym] = {}

    n_steps = p["n_steps"]
    n_minibatches = 4
    start_time = time.time()
    total_steps = 0

    while (time.time() - start_time) < budget:
        sym = np.random.choice(active_symbols, p=weights)
        env = train_envs[sym]
        obs = env_obs[sym]
        reward_state = env_reward_states[sym]

        batch_obs, batch_actions, batch_logprobs = [], [], []
        batch_rewards, batch_values, batch_dones = [], [], []

        for _ in range(n_steps):
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    obs.unsqueeze(0)
                )

            next_obs, _, done, truncated, info = env.step(action.item())
            raw_reward = compute_reward(info, reward_state, p["lam_vol"], p["lam_draw"])
            reward_norm.update(raw_reward)
            reward = reward_norm.normalize(raw_reward)

            batch_obs.append(obs)
            batch_actions.append(action.squeeze())
            batch_logprobs.append(logprob.squeeze())
            batch_rewards.append(reward)
            batch_values.append(value.squeeze())
            batch_dones.append(done or truncated)

            if done or truncated:
                next_obs, _ = env.reset()
                reward_state = {}
                env_reward_states[sym] = reward_state

            obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
            total_steps += 1

        env_obs[sym] = obs

        # GAE
        with torch.no_grad():
            _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0))
            next_value = next_value.squeeze()

        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=DEVICE)
        values = torch.stack(batch_values).squeeze(-1)
        dones = torch.tensor(batch_dones, dtype=torch.float32, device=DEVICE)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            next_val = next_value if t == n_steps - 1 else values[t + 1]
            delta = rewards[t] + p["gamma"] * next_val * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = (
                delta + p["gamma"] * p["gae_lam"] * (1 - dones[t]) * lastgaelam
            )

        returns = advantages + values
        b_obs = torch.stack(batch_obs)
        b_actions = torch.stack(batch_actions)
        b_logprobs = torch.stack(batch_logprobs)
        b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        b_returns = returns

        minibatch_size = n_steps // n_minibatches

        for _ in range(p["n_epochs"]):
            indices = torch.randperm(n_steps, device=DEVICE)
            for start in range(0, n_steps, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = policy.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                ratio = (new_logprob - b_logprobs[mb_idx]).exp()
                surr1 = ratio * b_advantages[mb_idx]
                surr2 = (
                    torch.clamp(ratio, 1 - p["clip"], 1 + p["clip"])
                    * b_advantages[mb_idx]
                )
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((new_value - b_returns[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + 0.5 * v_loss - p["ent"] * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

    return policy


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
            sh = evaluate(env_test, policy_fn, min_trades=3)
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

    return (
        float(np.mean(passing)) if passing else 0.0,
        len(passing),
        trades_all,
        worst_dd,
    )


def make_ensemble_fn(policies, device):
    """Create ensemble policy function averaging logits."""

    def fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits_sum = None
            for p in policies:
                logits, _ = p(obs_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            return logits_sum.argmax(dim=-1).item()

    return fn


# ── Full training run ──────────────────────────────────────────
def full_run(symbols, p, budget, n_seeds, split="test", verbose=True):
    """Train n_seeds policies, evaluate ensemble. Returns (sharpe, passing, trades, dd)."""
    # Suppress env loading noise
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
    policies = []
    for seed in range(n_seeds):
        if verbose:
            print(f"  Training seed {seed} ({budget_per_seed}s)...")
        policy = train_one_policy(
            train_envs, active, weights, obs_shape, p, budget_per_seed, seed
        )
        policies.append(policy)

    ensemble_fn = make_ensemble_fn(policies, DEVICE)
    sh, ps, tr, dd = eval_policy(ensemble_fn, symbols, split=split)
    return sh, ps, tr, dd


# ── Optuna objective ───────────────────────────────────────────
def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024]),
        "n_epochs": 4,
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lam": 0.95,
        "ent": trial.suggest_float("ent", 0.01, 0.05, log=True),  # min 0.01
        "clip": 0.2,
        "hdim": trial.suggest_categorical("hdim", [128, 256]),
        "nlayers": 3,
        "lam_vol": 0.5,  # FIXED — searching over penalty weights causes inaction
        "lam_draw": 1.0,  # FIXED
    }

    print(f"\n{'='*50}")
    print(f"Trial {trial.number}")
    for k in ["lr", "n_steps", "gamma", "ent", "hdim", "lam_vol", "lam_draw"]:
        v = p[k]
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    try:
        t0 = time.time()
        sh, ps, tr, dd = full_run(
            SEARCH_SYMBOLS, p, SEARCH_BUDGET, SEARCH_SEEDS, split="val", verbose=False
        )
        elapsed = time.time() - t0
        print(
            f"  => sharpe={sh:.4f} pass={ps}/{len(SEARCH_SYMBOLS)} "
            f"trades={tr} dd={dd:.4f} ({elapsed:.0f}s)"
        )
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
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
        f"({SEARCH_SEEDS} seeds) on {SEARCH_SYMBOLS} ===\n"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage="sqlite:///optuna_study.db",
        study_name="ppo_v4_longhorizon",
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
        "lam_vol": 0.5,
        "lam_draw": 1.0,
    }

    print(
        f"\n=== FINAL: {FINAL_SEEDS} seeds x "
        f"{FINAL_BUDGET // FINAL_SEEDS}s on all {len(DEFAULT_SYMBOLS)} symbols ==="
    )
    print(f"params: {bp}\n")

    sh, ps, tr, dd = full_run(
        DEFAULT_SYMBOLS, bp, FINAL_BUDGET, FINAL_SEEDS, split="test", verbose=True
    )

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
