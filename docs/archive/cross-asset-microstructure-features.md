---
title: Cross-Asset Microstructure Features
topics: [microstructure, cross-symbol, features, lead-lag, regime]
sources:
  - docs/research/cross-asset-microstructure-features.md
last_updated: 2026-04-03
---

# Cross-Asset Microstructure Features

## What It Is

Cross-asset microstructure features exploit the statistical relationships between different crypto assets at tick scale to improve directional prediction. The core finding from recent literature (Bieganowski & Slepaczuk 2026, Asia-Pacific Financial Markets 2026) is that BTC Granger-causes altcoin price movements with a lag inversely proportional to trade count -- low-liquidity assets like FARTCOIN, PENGU, and KBONK show the greatest price delay. The transmission is regime-asymmetric: BTC shocks propagate faster and more completely during Bear/Crash regimes than Bull.

Beyond lead-lag returns, the research identifies three additional signal categories: (1) level-1 queue depletion rate (how fast top-of-book volume is consumed), which Cont, Kukanov & Stoikov (2015) showed is the strongest single-tick predictor in a limit order book; (2) VPIN acceleration (rate of change of flow toxicity), which Salehi et al. (2026) found precedes large Bitcoin price moves by 15-60 minutes; and (3) cross-symbol average funding rate as a market-wide leverage regime indicator.

## Relevance to Our Project

The tape-reading spec uses 17 per-symbol features with no cross-asset information. For the 23 non-BTC/ETH symbols in our universe, lagged BTC/ETH returns could provide leading-indicator context that the model currently lacks entirely. The research validates that our existing per-symbol features (OFI via [cum_ofi](cum-ofi.md), spread, VWAP-deviation) are universally stable across assets -- the same SHAP importance rankings appear regardless of market cap. This supports the spec's "universal microstructure patterns" thesis. The [kyle_lambda](kyle-lambda.md) and [book_walk](book-walk.md) features already capture some cross-asset dynamics implicitly through orderbook state, but explicit BTC return features would be additive for illiquid symbols.

## Key Findings

- BTC lagged returns (`r_BTC_lag1`) are the highest-priority new feature, particularly for low-liquidity symbols where price delay is greatest
- ETH adds marginal information over BTC alone for DeFi-adjacent mid-caps (AAVE, UNI, LINK) but not for meme coins
- Level-1 queue depletion rate outperforms multi-level [orderbook imbalance](orderbook-alignment.md) as a one-tick predictor
- VPIN acceleration (change in flow toxicity over time) is more predictive than VPIN level alone
- Cross-symbol funding rate mean serves as a cheap regime indicator for market-wide deleveraging
- Symbol embeddings (dim=8) show theoretical appeal but likely redundant given our rolling normalization scheme
- GNN architectures are impractical with 25 symbols x 160 days of data -- insufficient for message passing
- Low-frequency signals (funding, regime) risk polluting tick-scale features if not architecturally separated
- All literature uses CEX data; DEX orderbooks (synthetic AMM curves) may transfer less cleanly

## Related Concepts

- [kyle-lambda](kyle-lambda.md) -- per-snapshot price impact, related to permanent impact decomposition
- [cum-ofi](cum-ofi.md) -- order flow imbalance, the per-symbol version of what cross-asset OFI extends
- [orderbook-alignment](orderbook-alignment.md) -- queue depletion rate depends on proper OB snapshot alignment
- [effort-vs-result](effort-vs-result.md) -- captures similar intuition to Hasbrouck permanent impact but at event level
- [climax-score](climax-score.md) -- detects unusual activity that may coincide with cross-asset cascade events
