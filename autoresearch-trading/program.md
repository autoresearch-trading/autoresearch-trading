# autoresearch-trading

Autonomous RL research for DEX perpetual futures trading. You are an AI researcher experimenting with RL trading agents. Your goal is to maximize out-of-sample Sharpe ratio by iterating on the training code.

## Setup (run once at start)

1. Read this file completely
2. Read `prepare.py` to understand the environment and features
3. Read `train.py` to understand the current agent
4. Create a new experiment branch:
   ```bash
   git checkout -b autoresearch/experiment-$(date +%s)
   ```
5. Verify data is cached:
   ```bash
   uv run python -c "from prepare import prepare_data, DEFAULT_SYMBOLS; prepare_data(DEFAULT_SYMBOLS)"
   ```
6. Run the baseline and record it:
   ```bash
   uv run train.py > run.log 2>&1
   grep "^val_sharpe:" run.log
   ```
7. Initialize results log:
   ```bash
   echo -e "commit\tval_sharpe\tnum_trades\tmax_drawdown\tstatus\tdescription" > results.tsv
   ```
8. Record baseline result in results.tsv

## Data

You have **36GB of DEX perpetual futures data** from 25 crypto symbols:

2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP

Data sources (Hive-partitioned Parquet):
- **Trades**: `data/trades/symbol={SYM}/date={DATE}/*.parquet` — ts_ms, symbol, side (open_long/open_short/close_long/close_short), price, qty
- **Orderbook**: `data/orderbook/symbol={SYM}/date={DATE}/*.parquet` — ts_ms, bids/asks (list of {price, qty}, up to 10 levels)
- **Funding**: `data/funding/symbol={SYM}/date={DATE}/*.parquet` — ts_ms, rate, interval_sec

Date range: 2025-10-16 to 2026-03-09 (~145 days)

### Splits
- **Train**: 2025-10-16 to 2026-01-23 (100 days)
- **Validation**: 2026-01-23 to 2026-02-17 (25 days)
- **Test**: 2026-02-17 to 2026-03-09 (20 days) — used only by evaluate()

### Feature Channels (24 features per step)

Each step = 100 consecutive trades (~1-2 seconds for BTC).

| # | Feature | Description |
|---|---------|-------------|
| 0 | vwap | Volume-weighted average price of batch |
| 1 | returns | Log return vs previous batch VWAP |
| 2 | net_volume | Buy volume - sell volume |
| 3 | trade_count | Number of trades (always 100) |
| 4 | buy_ratio | Fraction of buy trades |
| 5 | cvd_delta | Cumulative volume delta change |
| 6 | tfi | Trade flow imbalance: (buy-sell)/(buy+sell) |
| 7 | large_trade_count | Trades > 95th percentile qty |
| 8 | bid_depth_total | Total bid-side liquidity |
| 9 | ask_depth_total | Total ask-side liquidity |
| 10 | imbalance | (bid-ask)/(bid+ask) depth imbalance |
| 11 | spread_bps | Spread in basis points |
| 12-16 | level_1-5_bid_vol | Per-level bid volumes |
| 17-21 | level_1-5_ask_vol | Per-level ask volumes |
| 22 | funding_rate | Current funding rate |
| 23 | funding_rate_change | Change in funding rate |

All features are rolling z-score normalized (window=1000, min_periods=100).

### Domain Context

- **Side normalization**: open_long/close_short = buy (lifting asks), open_short/close_long = sell (hitting bids)
- **CVD (Cumulative Volume Delta)**: Running sum of (buy_vol - sell_vol). Divergence from price = reversal signal.
- **TFI (Trade Flow Imbalance)**: Normalized net flow. Near +1 = aggressive buying, near -1 = aggressive selling.
- **OFI (Order Flow Imbalance)**: Changes in orderbook depth at best levels. Predicts short-term price moves.
- **Funding rate**: Periodic payment between longs/shorts. Positive = longs pay shorts (bullish crowd). Large positive → potential long squeeze.
- **Spread**: Wider spread = lower liquidity = higher adverse selection risk.
- **Market regimes**: Trending (large directional moves), mean-reverting (range-bound), volatile (wide swings), low-liquidity (thin books).

## What You CAN Do

**ONLY modify `train.py`.** You have full freedom to change:

- **Algorithm**: PPO, SAC, TD3, Decision Transformer, CQL, IQL, or anything else
- **Network architecture**: MLP, CNN, LSTM, Transformer, attention, skip connections
- **Reward function**: Any function of the info dict (step_pnl, position, equity, drawdown, trade_count, hold_duration)
- **Hyperparameters**: Learning rate, gamma, batch size, hidden dims, etc.
- **State representation**: Reshape obs, add engineered features from raw obs, temporal encoding
- **Training strategy**: Curriculum learning, multi-symbol training, ensemble methods
- **Symbol selection**: Train on any symbol(s) via the SYMBOL variable (or loop over DEFAULT_SYMBOLS)

## What You CANNOT Do

- **DO NOT modify `prepare.py`** — environment, features, evaluation are fixed
- **DO NOT install new packages** — only use torch, gymnasium, numpy, pandas, pyarrow
- **DO NOT change the evaluation** — val_sharpe must come from evaluate() in prepare.py
- **DO NOT modify the data splits** — train/val/test dates are fixed
- **DO NOT skip the 5-minute training budget** — TRAIN_BUDGET_SECONDS = 300

## Goal

**Maximize `val_sharpe`** (out-of-sample Sharpe ratio on the test set).

### Guardrails (automatic, enforced by evaluate())
- **Minimum 50 trades** — prevents "never trade" degenerate solutions
- **Maximum 20% drawdown** — prevents reckless strategies
- Violating either → val_sharpe = 0.0

## Experiment Loop

This is the core loop. Repeat forever:

### 1. Form hypothesis
Think about what might improve the score. Read research findings below for ideas.

### 2. Implement change
Edit `train.py` with your modification. Keep changes focused — one idea per experiment.

### 3. Commit
```bash
git add train.py
git commit -m "experiment: <brief description of what changed>"
```

### 4. Run training
```bash
uv run train.py > run.log 2>&1
```

### 5. Extract results
```bash
SHARPE=$(grep "^val_sharpe:" run.log | tail -1 | awk '{print $2}')
TRADES=$(grep "^num_trades:" run.log | tail -1 | awk '{print $2}')
DRAWDOWN=$(grep "^max_drawdown:" run.log | tail -1 | awk '{print $2}')
COMMIT=$(git rev-parse --short HEAD)
echo "val_sharpe: $SHARPE, num_trades: $TRADES, max_drawdown: $DRAWDOWN"
```

### 6. Keep or discard
```bash
# If score improved or is promising:
echo -e "$COMMIT\t$SHARPE\t$TRADES\t$DRAWDOWN\tkept\t<description>" >> results.tsv

# If score regressed:
echo -e "$COMMIT\t$SHARPE\t$TRADES\t$DRAWDOWN\tdiscarded\t<description>" >> results.tsv
git revert HEAD --no-edit
```

### 7. Repeat
Go back to step 1. Review results.tsv to understand what works.

## Research Hints

Ordered by expected impact:

1. **Risk-Aware Reward**: Current reward = pnl - vol_penalty - dd_penalty. Try Sortino-style (only penalize downside vol), CVaR tail penalty, or EMA-based stats instead of rolling window.

2. **CNN + Attention Encoder**: Replace flat MLP with temporal CNN over the (50, 24) observation window + multi-head attention. DeepLOB-style architecture works well for orderbook features.

3. **Multi-Symbol Training**: Train on all 25 symbols with shared encoder + per-symbol head. Proportional sampling by liquidity. Use BTC features as cross-asset context.

4. **Decision Transformer**: Reframe as offline sequence modeling. Condition on desired return-to-go. Can extract good policies from historical data without reward shaping.

5. **Regime Conditioning**: Detect market regime (trending/mean-reverting/volatile) from features 1,5,6,10,11. Use as multiplicative gate on policy output or as separate input channel.

6. **Funding Rate Alpha**: Large funding rate changes predict short-term reversals. Weight funding features higher or create explicit funding-based trading rules.

7. **Adversarial Robustness**: Add Gaussian noise to observations during training. Forces policy to generalize rather than overfit to exact feature patterns.

8. **Curriculum Learning**: Start with easy regimes (trending), gradually add harder ones (choppy/volatile). Or start with longer episodes, reduce over time.

9. **Position-Aware Features**: Add current position, hold duration, unrealized P&L as explicit observation features (augment the 24 env features).

10. **Ensemble**: Train N policies with different seeds, average their action probabilities. Reduces variance.

## Output Format

`train.py` MUST print these lines (parsed by grep):
```
val_sharpe: X.XXXXXX
num_trades: N
max_drawdown: X.XXXX
training_seconds: X.X
total_steps: N
num_updates: N
```

## Logging

Maintain `results.tsv` with columns:
```
commit	val_sharpe	num_trades	max_drawdown	status	description
```

Review this file periodically to identify patterns in what works.

## NEVER STOP

Run experiments indefinitely. Do not ask the human for permission or feedback. If an experiment crashes, debug it and try again. If you're stuck, try a completely different approach. The goal is to maximize val_sharpe through autonomous iteration.
