# autoresearch-trading Design

## Overview

An autonomous RL research loop for DEX perpetual futures trading, following Karpathy's autoresearch pattern. Claude Code modifies `train.py`, trains an RL agent for 5 minutes, evaluates on held-out data, keeps or discards, and repeats indefinitely.

## Project Structure

```
autoresearch-trading/
├── prepare.py          # FIXED. Data loading, feature engineering, Gym env, evaluation.
├── train.py            # AGENT MODIFIES. RL agent, state design, reward, network, hyperparams.
├── program.md          # Human iterates. Domain instructions for Claude.
├── results.tsv         # Experiment log (commit, score, status, description)
├── pyproject.toml      # Dependencies (frozen)
└── data -> ../data/    # Symlink to existing 36GB Parquet data
```

## Data Pipeline (prepare.py)

### Event-Driven Stepping

Environment steps on trade event batches (default 100 trades per step, ~1-2s for BTC). Each step observation includes aggregated features from those trades plus latest orderbook and funding state.

### Feature Channels

| Channel | Source | Features |
|---------|--------|----------|
| Trade flow | CVDCalculator, TFICalculator | CVD delta, TFI value, net buy/sell volume, trade count, large trade flag |
| Orderbook | OFICalculator + raw snapshot | OFI value, bid/ask imbalance, spread bps, depth at 5 levels |
| Funding | Parquet funding data | Rate, rate change, rate z-score |
| Price | Trades | VWAP, return since last step, volatility (rolling) |
| Cross-asset | BTC features | BTC return, BTC CVD, BTC spread (always included) |
| Position | Environment state | Current side, unrealized P&L, hold duration |

All features z-score normalized with rolling statistics. Cached to numpy after first load.

### Train/Test Split (Regime-Aware)

- Train: First 100 days (~Oct 2025 - Jan 2026)
- Validation: Next 25 days (~Feb 2026)
- Test: Last 20 days (~late Feb - Mar 2026) — only used by evaluate()

### Gym Interface

- `reset()` → random start in training window
- `step(action)` → returns (obs, raw_info, done, truncated, info)
- Reward computed by train.py's compute_reward(), not prepare.py
- Transaction costs: fixed taker fee applied on entry/exit
- Episode ends after configurable steps or end of window

## Agent-Modified File (train.py)

CleanRL-style single file with marked sections:

1. **Hyperparameters**: ALGO, HIDDEN_DIM, NUM_LAYERS, LR, GAMMA, WINDOW_SIZE, LAMBDA_VOL, LAMBDA_DRAW, TRADE_BATCH
2. **Reward function** (compute_reward): Starting formula PnL - λ_vol * rolling_std - λ_draw * drawdown - tx_costs
3. **Network architecture** (PolicyNetwork): Starting MLP, upgradeable to CNN+Attention, LSTM, Transformer
4. **Training loop**: CleanRL PPO, 5-min wall-clock budget, prints val_sharpe at end

## Agent Instructions (program.md)

- Domain context explaining signals, market regimes, data structure
- Constraints: only modify train.py, no new packages, no eval changes
- Guardrails: min 50 trades, max 20% drawdown, else score = 0
- Research hints from deep research findings
- NEVER STOP: run indefinitely

## Dependencies

- torch >= 2.2 (MPS backend)
- gymnasium >= 1.0
- numpy, pandas, pyarrow

No SB3. Agent writes own training loops.

## Runtime

- ~10 experiments/hour on MacBook Pro
- ~80 experiments overnight
- MPS for network ops, CPU for environment
- First prepare.py run: ~5-10 min. Subsequent: ~10s (cached).

## Metric

Single number: out-of-sample Sharpe ratio on held-out test window.

## Guardrails

- Min 50 trades (prevents "never trade" degenerate solutions)
- Max 20% drawdown (prevents reckless strategies)
- Both violated → score reported as 0.0
