#!/usr/bin/env python3
"""RL trading agent — multi-symbol CUDA training."""

import io
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from prepare import DEFAULT_SYMBOLS, TRAIN_BUDGET_SECONDS, evaluate, make_env

# === HYPERPARAMETERS (agent tunes these) ===
ALGO = "PPO"
HIDDEN_DIM = 256
NUM_LAYERS = 3
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
NUM_STEPS = 512  # Steps per rollout
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
WINDOW_SIZE = 50
TRADE_BATCH = 100
LAMBDA_VOL = 0.5
LAMBDA_DRAW = 1.0
SYMBOLS = DEFAULT_SYMBOLS  # All 25 symbols

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# === REWARD FUNCTION (agent redesigns this) ===
def compute_reward(info, reward_state):
    """Sortino-style reward: only penalize downside volatility.

    info contains: step_pnl, position, equity, drawdown, trade_count, hold_duration
    reward_state is a mutable dict for tracking rolling statistics.
    """
    pnl = info["step_pnl"]

    # Track rolling downside deviation (Sortino-style)
    reward_state.setdefault("pnl_history", [])
    reward_state["pnl_history"].append(pnl)
    if len(reward_state["pnl_history"]) > 100:
        reward_state["pnl_history"] = reward_state["pnl_history"][-100:]

    if len(reward_state["pnl_history"]) > 10:
        negatives = [p for p in reward_state["pnl_history"] if p < 0]
        downside_vol = np.std(negatives) if len(negatives) > 2 else 0
    else:
        downside_vol = 0

    dd = info["drawdown"]

    reward = pnl - LAMBDA_VOL * downside_vol - LAMBDA_DRAW * dd
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
    # Create training envs for all symbols + compute sampling weights
    print(f"Loading {len(SYMBOLS)} symbol environments...")
    train_envs = {}
    env_weights = {}
    for sym in SYMBOLS:
        try:
            env = make_env(
                sym, "train", window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH
            )
            train_envs[sym] = env
            env_weights[sym] = env.num_steps
            print(f"  {sym}: {env.num_steps} steps")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")

    active_symbols = list(train_envs.keys())
    weights = np.array([env_weights[s] for s in active_symbols], dtype=np.float64)
    weights /= weights.sum()
    print(f"Training on {len(active_symbols)} symbols")

    # Initialize policy from first env's obs shape
    obs_shape = train_envs[active_symbols[0]].observation_space.shape
    policy = PolicyNetwork(obs_shape).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # Initialize all envs
    env_obs = {}
    env_reward_states = {}
    for sym in active_symbols:
        obs, _ = train_envs[sym].reset()
        env_obs[sym] = torch.tensor(obs, dtype=torch.float32, device=device)
        env_reward_states[sym] = {}

    start_time = time.time()
    total_steps = 0
    num_updates = 0

    while (time.time() - start_time) < TRAIN_BUDGET_SECONDS:
        # Sample a symbol for this rollout (proportional to data size)
        sym = np.random.choice(active_symbols, p=weights)
        env = train_envs[sym]
        obs = env_obs[sym]
        reward_state = env_reward_states[sym]

        # Collect rollout
        batch_obs = []
        batch_actions = []
        batch_logprobs = []
        batch_rewards = []
        batch_values = []
        batch_dones = []

        for step in range(NUM_STEPS):
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    obs.unsqueeze(0)
                )

            next_obs, _, done, truncated, info = env.step(action.item())
            reward = compute_reward(info, reward_state)

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

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            total_steps += 1

        # Save env state for next rollout on this symbol
        env_obs[sym] = obs

        # Compute advantages (GAE)
        with torch.no_grad():
            _, _, _, next_value = policy.get_action_and_value(obs.unsqueeze(0))
            next_value = next_value.squeeze()

        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        values = torch.stack(batch_values).squeeze(-1)
        dones = torch.tensor(batch_dones, dtype=torch.float32, device=device)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = (
                delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * lastgaelam
            )

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
                surr2 = (
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                    * b_advantages[mb_idx]
                )

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

    # === MULTI-SYMBOL EVALUATION ===
    def policy_fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = policy(obs_t)
            return logits.argmax(dim=-1).item()

    print("---")
    print("=== PER-SYMBOL EVALUATION ===")

    passing_sharpes = []
    total_trades_all = 0
    worst_drawdown = 0.0

    for sym in SYMBOLS:
        try:
            env_test = make_env(
                sym, "test", window_size=WINDOW_SIZE, trade_batch=TRADE_BATCH
            )
            # Capture evaluate() output to avoid cluttering stdout
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            sharpe = evaluate(env_test, policy_fn)
            sys.stdout = old_stdout
            eval_output = captured.getvalue()

            # Parse num_trades and max_drawdown from evaluate() output
            trades = 0
            max_dd = 0.0
            for line in eval_output.strip().split("\n"):
                if line.startswith("num_trades:"):
                    trades = int(line.split()[1])
                elif line.startswith("max_drawdown:"):
                    max_dd = float(line.split()[1])

            status = "PASS" if sharpe > 0 else "FAIL"
            print(
                f"  {sym}: sharpe={sharpe:.4f} trades={trades} dd={max_dd:.4f} [{status}]"
            )

            if sharpe > 0:
                passing_sharpes.append(sharpe)
            total_trades_all += trades
            worst_drawdown = max(worst_drawdown, max_dd)
        except Exception as e:
            print(f"  {sym}: ERROR ({e})")

    # === PORTFOLIO SUMMARY ===
    portfolio_sharpe = np.mean(passing_sharpes) if passing_sharpes else 0.0

    print("---")
    print("=== PORTFOLIO SUMMARY ===")
    print(f"symbols_passing: {len(passing_sharpes)}/{len(SYMBOLS)}")
    print(f"val_sharpe: {portfolio_sharpe:.6f}")
    print(f"num_trades: {total_trades_all}")
    print(f"max_drawdown: {worst_drawdown:.4f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_steps: {total_steps}")
    print(f"num_updates: {num_updates}")


if __name__ == "__main__":
    train()
