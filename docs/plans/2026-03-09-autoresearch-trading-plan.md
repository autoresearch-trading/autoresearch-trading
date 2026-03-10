# autoresearch-trading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an autonomous RL research loop that lets Claude Code experiment with trading agents on 36GB of DEX perpetual futures data, following Karpathy's autoresearch pattern.

**Architecture:** Event-driven Gym environment reading Parquet trade/orderbook/funding data, CleanRL-style single-file RL training, autoresearch iteration loop with git commit/revert, out-of-sample Sharpe as the single metric.

**Tech Stack:** Python 3.12+, PyTorch (MPS), Gymnasium, numpy, pandas, pyarrow. Reuses signal calculators from signal-engine/src/signals/.

---

### Task 1: Project Scaffold

**Files:**
- Create: `autoresearch-trading/pyproject.toml`
- Create: `autoresearch-trading/.python-version`

**Step 1: Create directory and symlink**

```bash
mkdir -p autoresearch-trading
cd autoresearch-trading
ln -s ../data data
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "autoresearch-trading"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.2",
    "gymnasium>=1.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=15.0",
]

[build-system]
requires = ["setuptools>=65"]
build-backend = "setuptools.build_meta"
```

**Step 3: Create .python-version**

```
3.12
```

**Step 4: Install dependencies**

Run: `cd autoresearch-trading && uv sync`
Expected: Dependencies install successfully, torch with MPS support.

**Step 5: Verify MPS**

Run: `cd autoresearch-trading && uv run python -c "import torch; print(torch.backends.mps.is_available())"`
Expected: `True`

**Step 6: Commit**

```bash
git add autoresearch-trading/pyproject.toml autoresearch-trading/.python-version
git commit -m "feat: scaffold autoresearch-trading project"
```

---

### Task 2: Data Loading & Feature Engineering (prepare.py — Part 1)

**Files:**
- Create: `autoresearch-trading/prepare.py`

**Step 1: Write data loading functions**

Write `prepare.py` with these functions (no tests — this is a fixed research script like autoresearch's prepare.py):

- `load_trades(symbol, start_date, end_date)` → reads Parquet files from `data/trades/symbol={sym}/date={date}/*.parquet`, returns DataFrame sorted by ts_ms. Reuse the Parquet discovery pattern from `signal-engine/scripts/run_backtest_parquet.py:discover_parquet_files()`.

- `load_orderbook(symbol, start_date, end_date)` → reads from `data/orderbook/symbol={sym}/date={date}/*.parquet`, returns DataFrame sorted by ts_ms. Parse bids/asks lists into structured arrays.

- `load_funding(symbol, start_date, end_date)` → reads from `data/funding/symbol={sym}/date={date}/*.parquet`, returns DataFrame sorted by ts_ms.

Key columns per source:
- Trades: ts_ms, symbol, side (open_long/open_short/close_long/close_short), price, qty
- Orderbook: ts_ms, symbol, bids (list of {price, qty}), asks (list of {price, qty})
- Funding: ts_ms, symbol, rate, interval_sec

**Step 2: Write feature computation**

Add function `compute_features(trades_df, orderbook_df, funding_df, trade_batch=100)` that:

1. Groups trades into batches of `trade_batch` trades
2. For each batch computes:
   - `vwap`: volume-weighted average price
   - `returns`: log(vwap / prev_vwap)
   - `net_volume`: buy_volume - sell_volume (using normalize_side logic: open_long/close_short = buy, open_short/close_long = sell)
   - `trade_count`: number of trades in batch
   - `buy_ratio`: fraction of buy trades
   - `cvd_delta`: net_volume cumulative change over batch
   - `tfi`: (buy_vol - sell_vol) / (buy_vol + sell_vol)
   - `large_trade_count`: trades where qty > 95th percentile
3. For each batch, finds the most recent orderbook snapshot and extracts:
   - `bid_depth_total`: sum of qty across bid levels
   - `ask_depth_total`: sum of qty across ask levels
   - `imbalance`: (bid_depth - ask_depth) / (bid_depth + ask_depth)
   - `spread_bps`: (best_ask - best_bid) / mid_price * 10000
   - `level_{1..5}_bid_vol`, `level_{1..5}_ask_vol`: per-level volumes
4. For each batch, finds the most recent funding rate and extracts:
   - `funding_rate`: raw rate
   - `funding_rate_change`: rate - previous rate
5. Returns a numpy array of shape (num_batches, num_features) and a corresponding array of timestamps and prices.

**Step 3: Write normalization and caching**

Add functions:
- `normalize_features(features, window=1000)` → rolling z-score normalization. For each feature column, subtract rolling mean and divide by rolling std (with min_periods=100, fill initial NaNs with 0).
- `cache_features(symbol, features, timestamps, prices, cache_dir)` → save to `.npz` file
- `load_cached(symbol, cache_dir)` → load from `.npz` if exists and newer than source data

**Step 4: Write the main data preparation flow**

Add `prepare_data(symbols=["BTC"], trade_batch=100)` that:
1. Defines date splits:
   - TRAIN_END = "2026-01-23" (first 100 days from Oct 16)
   - VAL_END = "2026-02-17" (next 25 days)
   - TEST_END = "2026-03-09" (last 20 days)
2. For each symbol: load → compute features → normalize → cache
3. Also computes BTC cross-asset features for non-BTC symbols
4. Returns dict of {symbol: {train: (features, timestamps, prices), val: ..., test: ...}}

**Step 5: Run and verify**

Run: `cd autoresearch-trading && uv run python -c "from prepare import prepare_data; d = prepare_data(['BTC']); print(d['BTC']['train'][0].shape)"`
Expected: Prints shape like `(N, num_features)` where N is number of trade batches in training period.

**Step 6: Commit**

```bash
git add autoresearch-trading/prepare.py
git commit -m "feat: data loading and feature engineering for autoresearch-trading"
```

---

### Task 3: Gym Environment (prepare.py — Part 2)

**Files:**
- Modify: `autoresearch-trading/prepare.py`

**Step 1: Write TradingEnv class**

Add to prepare.py a Gymnasium environment:

```python
class TradingEnv(gymnasium.Env):
    """Event-driven trading environment.

    Observations: window of normalized features (window_size, num_features)
    Actions: 0=flat, 1=long, 2=short
    Reward: computed externally by train.py's compute_reward()
    """
```

Constructor takes:
- `features`: numpy array (num_steps, num_features)
- `prices`: numpy array (num_steps,) — VWAP at each step
- `window_size`: int (default 50) — how many past steps in observation
- `fee_bps`: float (default 5) — transaction cost in basis points

`reset()`:
- Pick random start index (between window_size and len-1000)
- Set position to flat, cash to 1.0
- Return observation (features[start-window_size:start])

`step(action)`:
- Update position based on action (0=flat, 1=long, 2=short)
- If position changed, apply transaction cost (fee_bps)
- Compute raw step P&L based on price change and current position
- Track: unrealized_pnl, realized_pnl, peak_equity, drawdown, hold_duration, trade_count
- Return observation, 0.0 (placeholder reward), done, truncated, info_dict
- `info_dict` contains all raw values: step_pnl, position, equity, drawdown, trade_count, hold_duration — train.py uses these to compute its own reward
- Episode ends after 2000 steps or end of data

**Step 2: Write evaluate() function**

```python
def evaluate(env_test, policy_fn, min_trades=50, max_drawdown=0.20):
    """Run policy on test env, return val_sharpe.

    policy_fn: callable(obs) -> action
    Returns sharpe ratio, or 0.0 if guardrails violated.
    """
```

- Runs policy through entire test set (no random start, from beginning)
- Collects per-step returns
- Computes Sharpe = mean(returns) / std(returns) * sqrt(steps_per_day)
- If trade_count < min_trades or max_drawdown_seen > max_drawdown: return 0.0
- Prints `val_sharpe: {sharpe:.6f}` and `num_trades: {n}` and `max_drawdown: {dd:.4f}`

**Step 3: Write make_env() helper**

```python
def make_env(symbol="BTC", split="train", window_size=50, trade_batch=100):
    """Create a TradingEnv for the given symbol and data split."""
```

Loads cached data, returns TradingEnv instance.

**Step 4: Add constants at top of file**

```python
# === CONSTANTS (do not modify) ===
TRAIN_BUDGET_SECONDS = 300  # 5-minute training budget
TRAIN_END = "2026-01-23"
VAL_END = "2026-02-17"
TEST_END = "2026-03-09"
DEFAULT_SYMBOLS = ["BTC"]
FEE_BPS = 5  # Taker fee in basis points
```

**Step 5: Run full pipeline test**

Run: `cd autoresearch-trading && uv run python -c "
from prepare import make_env
env = make_env('BTC', 'train')
obs, info = env.reset()
print(f'Obs shape: {obs.shape}')
obs, reward, done, trunc, info = env.step(1)
print(f'Step info keys: {list(info.keys())}')
print(f'Step pnl: {info[\"step_pnl\"]:.6f}')
"`
Expected: Obs shape (50, N), info contains step_pnl, position, equity, etc.

**Step 6: Commit**

```bash
git add autoresearch-trading/prepare.py
git commit -m "feat: Gym trading environment and evaluation function"
```

---

### Task 4: Baseline train.py

**Files:**
- Create: `autoresearch-trading/train.py`

**Step 1: Write the starting train.py**

Single-file CleanRL-style PPO implementation. Sections clearly marked for the agent:

```python
#!/usr/bin/env python3
"""RL trading agent. This file is modified by the autoresearch agent."""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from prepare import make_env, evaluate, TRAIN_BUDGET_SECONDS

# === HYPERPARAMETERS (agent tunes these) ===
ALGO = "PPO"
HIDDEN_DIM = 128
NUM_LAYERS = 2
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
NUM_STEPS = 256        # Steps per rollout
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
WINDOW_SIZE = 50
TRADE_BATCH = 100
LAMBDA_VOL = 0.5
LAMBDA_DRAW = 1.0
SYMBOL = "BTC"

# === DEVICE ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === REWARD FUNCTION (agent redesigns this) ===
def compute_reward(info, reward_state):
    """Compute reward from environment info dict.

    info contains: step_pnl, position, equity, drawdown, trade_count, hold_duration
    reward_state is a mutable dict for tracking rolling statistics.
    """
    pnl = info["step_pnl"]

    # Track rolling P&L std for volatility penalty
    reward_state.setdefault("pnl_history", [])
    reward_state["pnl_history"].append(pnl)
    if len(reward_state["pnl_history"]) > 100:
        reward_state["pnl_history"] = reward_state["pnl_history"][-100:]

    vol = np.std(reward_state["pnl_history"]) if len(reward_state["pnl_history"]) > 10 else 0
    dd = info["drawdown"]

    reward = pnl - LAMBDA_VOL * vol - LAMBDA_DRAW * dd
    return reward

# === NETWORK (agent redesigns this) ===
class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions=3):
        super().__init__()
        flat_dim = obs_shape[0] * obs_shape[1]  # window_size * num_features

        layers = [nn.Linear(flat_dim, HIDDEN_DIM), nn.ReLU()]
        for _ in range(NUM_LAYERS - 1):
            layers.extend([nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()])
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(HIDDEN_DIM, n_actions)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

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

# === TRAINING LOOP (agent can rewrite entirely) ===
def train():
    env = make_env(SYMBOL, "train", window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH)
    obs_shape = env.observation_space.shape

    policy = PolicyNetwork(obs_shape).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    reward_state = {}

    start_time = time.time()
    total_steps = 0
    num_updates = 0

    while (time.time() - start_time) < TRAIN_BUDGET_SECONDS:
        # Collect rollout
        batch_obs = []
        batch_actions = []
        batch_logprobs = []
        batch_rewards = []
        batch_values = []
        batch_dones = []

        for step in range(NUM_STEPS):
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs.unsqueeze(0))

            next_obs, _, done, truncated, info = env.step(action.item())
            reward = compute_reward(info, reward_state)

            batch_obs.append(obs)
            batch_actions.append(action)
            batch_logprobs.append(logprob)
            batch_rewards.append(reward)
            batch_values.append(value)
            batch_dones.append(done or truncated)

            if done or truncated:
                next_obs, _ = env.reset()
                reward_state = {}

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            total_steps += 1

        # Compute advantages (GAE)
        with torch.no_grad():
            _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0))

        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        values = torch.stack(batch_values)
        dones = torch.tensor(batch_dones, dtype=torch.float32, device=device)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * lastgaelam

        returns = advantages + values

        # PPO update
        b_obs = torch.stack(batch_obs)
        b_actions = torch.stack(batch_actions)
        b_logprobs = torch.stack(batch_logprobs)
        b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        b_returns = returns

        batch_size = NUM_STEPS
        minibatch_size = batch_size // NUM_MINIBATCHES

        for epoch in range(UPDATE_EPOCHS):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = policy.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                ratio = (new_logprob - b_logprobs[mb_idx]).exp()
                surr1 = ratio * b_advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_advantages[mb_idx]

                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((new_value - b_returns[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        num_updates += 1

    training_seconds = time.time() - start_time

    # === EVALUATION ===
    env_test = make_env(SYMBOL, "test", window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH)

    def policy_fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = policy(obs_t)
            return logits.argmax(dim=-1).item()

    val_sharpe = evaluate(env_test, policy_fn)

    print("---")
    print(f"val_sharpe: {val_sharpe:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_steps: {total_steps}")
    print(f"num_updates: {num_updates}")

if __name__ == "__main__":
    train()
```

**Step 2: Run baseline**

Run: `cd autoresearch-trading && uv run train.py`
Expected: Runs for ~5 minutes, prints `val_sharpe: X.XXXX` (likely near 0 for untrained baseline).

**Step 3: Commit**

```bash
git add autoresearch-trading/train.py
git commit -m "feat: baseline PPO train.py for autoresearch-trading"
```

---

### Task 5: program.md

**Files:**
- Create: `autoresearch-trading/program.md`

**Step 1: Write program.md**

Adapt autoresearch-mlx's program.md for trading RL domain. Include:

1. **Project description**: Autonomous RL research for DEX perpetual futures trading
2. **Setup section**: Create branch, read files, verify data cached, run baseline, init results.tsv
3. **Domain context**: What the data is (tick trades, 10-level orderbook, funding rates for 25 crypto symbols), what features mean (CVD, TFI, OFI, imbalance, spread), what market regimes look like
4. **What you CAN do**: Modify train.py only — algorithm, architecture, reward, hyperparams, state representation
5. **What you CANNOT do**: Modify prepare.py, install packages, change evaluation
6. **Goal**: Maximize val_sharpe. Guardrails: min 50 trades, max 20% drawdown, else score = 0.0
7. **Research hints**: Risk-aware PPO, Decision Transformer, CNN+Attention orderbook encoder, funding rate as sentiment, adversarial noise, regime conditioning
8. **Output format**: Same as autoresearch-mlx (grep val_sharpe from run.log)
9. **Logging**: results.tsv with columns: commit, val_sharpe, num_trades, status, description
10. **Experiment loop**: Identical to autoresearch-mlx (modify → commit → run → grep → keep/discard → repeat)
11. **NEVER STOP**: Run indefinitely, don't ask the human

**Step 2: Commit**

```bash
git add autoresearch-trading/program.md
git commit -m "feat: program.md agent instructions for autoresearch-trading"
```

---

### Task 6: End-to-End Smoke Test

**Files:**
- No new files

**Step 1: Full pipeline test**

Run the full pipeline from scratch to verify everything works:

```bash
cd autoresearch-trading
uv sync
uv run prepare.py          # Should cache features
uv run train.py             # Should train 5 min, print val_sharpe
```

Expected: No crashes, val_sharpe printed at end.

**Step 2: Simulate one autoresearch iteration**

Manually simulate what the agent will do:
```bash
cd autoresearch-trading
git checkout -b autoresearch/smoke-test
# Modify a hyperparam in train.py (e.g., LEARNING_RATE = 1e-3)
uv run train.py > run.log 2>&1
grep "^val_sharpe:" run.log
# Verify score prints
git checkout main
git branch -d autoresearch/smoke-test
```

**Step 3: Final commit**

```bash
git add -A autoresearch-trading/
git commit -m "feat: autoresearch-trading ready for autonomous RL research"
```

---

### Task 7: Launch

**Not code — instructions for the user:**

1. Open a new Claude Code session in `autoresearch-trading/`
2. Prompt: "Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first."
3. Walk away. Check results.tsv in the morning.
