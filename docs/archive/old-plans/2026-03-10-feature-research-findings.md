# Feature Engineering Research Findings (March 2026)

Source: Exa deep research pro, 2025-2026 papers/repos/blogs

## Current Features (24)

Trade: VWAP, log returns, net volume, trade count, buy ratio, CVD delta, TFI, large trade count
Orderbook: total bid/ask depth, imbalance, spread bps, 5-level bid/ask volumes
Funding: rate, rate change

## High-Priority Additions (implement first)

### 1. Microprice (from orderbook)
Weighted mid-price adjusted by available depth. Better short-term price estimate than simple mid.
```
microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
```
Also compute: microprice deviation from mid (signed). Extend to multi-level weighting.
- Ref: [DL_LOB midprice models](https://github.com/FSUHeting/DL_LOB_Trading_and_MidPirce_Movement)

### 2. Order Flow Imbalance (OFI) — multi-level
Weighted sum of depth changes across top N levels. Most informative short-term predictor of mid-price moves.
```
OFI_t = sum_{l=1..N} w_l * (Δbid_vol_l - Δask_vol_l)
```
Both raw and rolling-normalized versions.
- Ref: [OrderImbalance repo](https://github.com/shaileshkakkar/OrderImbalance)
- Paper: [DRL for Optimizing Order Book Imbalance-Based HFT](https://www.researchgate.net/publication/391292844)

### 3. VPIN (Volume-Synchronized Probability of Informed Trading)
Flow toxicity measure. Bucketize by fixed volume, classify buy/sell, compute rolling imbalance.
Captures informed trading activity and short-term market stress.
- Ref: [VPIN_HFT repo](https://github.com/theopenstreet/VPIN_HFT) (`Final_Code_Implementation.ipynb`)

### 4. Open Interest dynamics
OI net change, OI trend (5/60/1440 min smoothed), funding-rate z-score, funding spike flag.
Combine funding change + OI change into a funding-pressure scalar.
- Ref: [NeuralArB RL features](https://www.neuralarb.com/2025/11/20/reinforcement-learning-in-dynamic-crypto-markets)

**Derivation: No direct OI feed, but funding rate reflects long/short imbalance (OI proxy). Combine funding rate changes + volume surges to estimate OI expansion/contraction. Funding-pressure scalar = funding_rate_change * volume_delta.**

### 5. Liquidation cascade score
Aggregate large liquidation counts / gross USD per short windows, percentile normalize.
Directional cascade score: net sell-liquidations - net buy-liquidations.
- Ref: [py-liquidation-map](https://github.com/aoki-h-jp/py-liquidation-map)

**Derivation: No direct liquidation feed, but liquidations manifest as clusters of large market orders during sharp price moves. Detect via: large_trade_count spikes + directional price acceleration + elevated volume. Directional cascade score = net large sells - net large buys during detected spike windows.**

## Medium-Priority Additions

### 6. Multi-horizon realized volatility
Realized vol at 1m, 5m, 1h horizons. Use as both input and regime signal.
Volatility-adjusted returns as additional feature.
- Ref: [Risk-Aware DQN paper](https://www.mdpi.com/2227-7390/13/18/3012)

### 7. Hurst exponent
R/S or DFA on rolling windows (512-4096 samples). Persistence measure.
H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk.
- Ref: [Hurst fractal market studies](https://www.aimspress.com/article/doi/10.3934/DSFE.2026004)

### 8. Kyle's lambda (price impact)
Regress short-horizon mid-price changes on signed volume in rolling windows.
Captures price impact per unit of order flow.

### 9. Cross-asset features
- Rolling BTC correlation (Pearson on log-returns)
- Lead-lag: past BTC returns as predictor for altcoin moves
- Cross-symbol order-flow aggregates
- Ref: [yakub268/algo-trading-platform](https://github.com/yakub268/algo-trading-platform)

### 10. Trade arrival rate
Trades per second / per volume unit. Volume-clocked aggregation for nonstationary markets.
- Ref: [Talos TCA](https://www.talos.com/insights/execution-insights-through-transaction-cost-analysis-tca-benchmarks-and-slippage)

## Normalization Improvements

Current: Rolling z-score (window=1000, min_periods=100)

Recommended upgrades:
1. **Robust scaling** (median/IQR) for tail-prone features (large trade count, liquidations)
2. **Quantile transforms** for heavy-skew features (volumes, USD values)
3. **Power transforms** (Yeo-Johnson) for variance stabilization
4. **Group-adaptive normalization (DAGBN)** for regime-shifting features
   - Ref: [DAGBN paper](https://ieeexplore.ieee.org/document/9596155)
   - Ref: [rl_trading repo](https://github.com/Jiawen006/rl_trading)

## What We Can Implement With Current Data

| Feature | Source Data | Have It? |
|---------|-----------|----------|
| Microprice | orderbook (bids/asks) | YES |
| OFI (multi-level) | orderbook (level changes) | YES |
| VPIN | trades (volume bucketing) | YES |
| Multi-horizon realized vol | trades (returns) | YES |
| Hurst exponent | trades (returns) | YES |
| Kyle's lambda | trades + orderbook | YES |
| Trade arrival rate | trades (timestamps) | YES |
| Cross-asset features | trades (multi-symbol) | YES |
| Better normalization | all features | YES |
| OI dynamics | funding + trades (derived) | YES — proxy via funding rate changes + volume |
| Liquidation cascades | trades (derived) | YES — proxy via large trade clusters + price acceleration |
| Long/short ratio | funding + trades (derived) | YES — proxy via funding rate + buy ratio + CVD |

## Key Repos

| Repo | Feature |
|------|---------|
| [OrderImbalance](https://github.com/shaileshkakkar/OrderImbalance) | OFI implementation |
| [VPIN_HFT](https://github.com/theopenstreet/VPIN_HFT) | VPIN flow toxicity |
| [DL_LOB](https://github.com/FSUHeting/DL_LOB_Trading_and_MidPirce_Movement) | Microprice + LOB models |
| [py-liquidation-map](https://github.com/aoki-h-jp/py-liquidation-map) | Liquidation heatmaps |
| [FineFT](https://github.com/qinmoelei/FineFT_code_space) | Ensemble RL + VAE OOD |
| [rl_trading](https://github.com/Jiawen006/rl_trading) | DAGBN normalization |
