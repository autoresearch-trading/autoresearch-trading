# Research: Cross-Asset Features & Advanced Microstructure Signals for Crypto Perpetual Futures

*Date: 2026-03-30 | Context: 25-symbol DEX perp futures, tick-scale MLP classifier, fee_mult=11×*

---

## Summary

The literature strongly validates our current per-symbol microstructure features (OFI, VWAP-dev, spread, VPIN) as the dominant predictors — they are universally stable across crypto assets regardless of market cap. The highest-confidence new signals are: **(1)** BTC/ETH lagged returns as cross-symbol leading indicators (most valuable for our low-liquidity symbols like FARTCOIN, PENGU, KBONK), **(2)** time-varying information content of large trades (extending VPIN toward Hasbrouck-style permanent price impact), and **(3)** funding rate cross-exchange divergence as a regime-conditioning signal. Symbol embeddings and GNN architectures show theoretical appeal but impose high data overhead for marginal gain over our flat MLP at tick scale.

---

## Findings by Topic

---

### 1. Cross-Symbol Features: BTC/ETH as Leading Indicators

**Finding 1.1 — BTC Granger-causes ALTs at high frequency, with lag inversely proportional to trade count.**
A 2026 study using 1-minute Binance data across 369 crypto pairs confirms Granger causality from BTC to altcoins using VAR(3) models. The key finding: *low-trade-count assets show the greatest price delay* (Immediate Sensitivity Indicator negatively correlated with log-trade-count at p < 0.05 in both Bull and Bear regimes). For BTC-dominant shocks, the effect dissipates beyond 3 lags (~3 minutes at 1-min frequency, or ~3–30 batch-steps at our 100-trade granularity depending on symbol). A LightGBM classifier on `[r_BTC(t-1), r_ALT(t-1)]` produced significant out-of-sample alpha during Bull, Sideways, and Crash regimes.
→ [Price Transmission from Bitcoin to Altcoins](https://link.springer.com/article/10.1007/s10690-026-09589-z) (Asia-Pacific Financial Markets, 2026)

**Finding 1.2 — Correlation regime matters: transmission strength differs dramatically across Bull/Bear/Crash.**
The VAR analysis finds that impulse responses to a BTC shock are regime-asymmetric: BTC shocks are transmitted faster and more completely in *Bear/Crash* regimes than Bull, particularly to small-cap ALTs. This suggests a **BTC-return × regime-flag interaction feature** rather than a flat BTC-return feature.
→ Same paper; also confirmed in GMM-VAR regime analysis (DSFE 2025).

**Finding 1.3 — ETH adds marginal information over BTC alone for mid-cap assets.**
The VAR ordering places BTC → ETH → LTC → small-caps. ETH return (lagged) is a significant predictor of medium-cap ALTs (AAVE, UNI, LINK in our symbol set) beyond what BTC alone captures, because ETH leads the DeFi-adjacent assets. For meme coins (FARTCOIN, PENGU, KBONK) neither BTC nor ETH lagged returns explain much — these respond primarily to self-reinforcing tape patterns.
→ Bieganowski & Ślepaczuk, arXiv:2602.00776 (Jan 2026); IMF cross-crypto spillover literature.

**Practical Assessment for our system:**
- **Implementable now:** `r_BTC_lag1`, `r_ETH_lag1` (returns of BTC/ETH from previous step at 100-trade granularity) are cheap to compute — 2 new features
- **Regime flag:** `btc_5min_trend_zscore` (BTC 300-trade return normalized by recent vol) — 1 feature  
- **Selectivity:** Most value for low-liquidity symbols (FARTCOIN, PENGU, KBONK, XPL, WLFI). For BTC/ETH themselves, these are self-referential (trivial).
- **Risk:** At 100-trade granularity for BTC (~1–2s), the 1-min lag found in papers maps to ~30–120 of our steps — the signal may already be captured by our `r_500`/`r_2800` features for high-cap symbols. Worth testing on low-liquidity symbols specifically.

---

### 2. Learned Symbol Embeddings & Market-Regime Representations

**Finding 2.1 — Symbol embeddings encode microstructure characteristics that improve generalization across assets.**
The LOBench benchmark (arXiv:2505.02139, May 2025) demonstrates that **learned LOB representations are more transferable across assets** than task-specific features. The key finding: a shared encoder pre-trained across many assets outperforms per-asset models on smaller-data assets, because embeddings capture structural properties (liquidity tier, typical spread width, volatility regime) that are asset-invariant inputs to prediction.

**Finding 2.2 — Trading GNN (arXiv:2504.07923) frames the multi-asset prediction problem as graph message passing.**
Graph edges encode cross-asset correlation, with edge weights updated dynamically. Performance on stock prediction tasks shows 10–20% Sharpe improvement over per-asset baselines. The key mechanism: **global market information flows through high-degree hub nodes** (BTC/ETH in crypto), which is exactly the lead-lag structure found in Finding 1.1.
→ Wu, arXiv:2504.07923 (Apr 2025)

**Finding 2.3 — Cross-asset pattern stability means a SINGLE universal feature set outperforms symbol-specific tuning.**
The Jan 2026 Bieganowski & Ślepaczuk paper (arXiv:2602.00776) is the most directly relevant result: the same engineered features (OFI, spread, VWAP-dev) show **remarkably similar SHAP importance rankings AND functional shapes** across assets spanning 100× in market cap. This means symbol embeddings in a joint model would primarily encode liquidity scale — something our hybrid z-score + robust normalization already approximately handles.
→ arXiv:2602.00776

**Practical Assessment:**
- **Symbol embedding (lookup table, dim=8, trained end-to-end):** Low cost to implement — append to flat input. Encodes latent liquidity/vol regime per symbol. Likely captures what our per-symbol cache normalization already handles implicitly, but worth a quick ablation.
- **GNN architecture:** High implementation cost; our T47 lesson shows temporal architectures need much more data. Multi-asset GNN has the same problem: 25 symbols × 145 days × sparse graphs = insufficient data for message passing. **Skip for now.**
- **Regime embedding (online HMM or viterbi):** Pre-compute a 3-state HMM on BTC 1-hour returns → {trending-up, trending-down, ranging} → encode as one-hot or continuous soft assignment. This regime label has been shown in multiple papers to dramatically change the feature-return relationship. **Medium priority.**

---

### 3. Alternative Microstructure Signals Beyond VPIN

**Finding 3.1 — Time-varying information content (Hasbrouck-Campigli) captures when trades are informative.**
Campigli, Bormetti & Lillo (arXiv:2212.12687, revised 2023) extend Hasbrouck's information share to a **time-varying setting using a Kalman-filter state-space model**. The key signal is the *permanent price impact* component of a trade — the fraction of a market order that becomes a lasting price change vs. a transient bounce. This is more granular than VPIN because it tracks *informativeness per trade* rather than per volume bucket. In their empirical tests on Italian market data, the time-varying information share predicts adverse selection episodes 2–5 steps ahead.

**Finding 3.2 — Queue Imbalance is the best one-tick predictor in a LOB.**
Cont, Kukanov & Stoikov (arXiv:1512.03492, 2015; still the dominant citation) showed that **top-of-book queue imbalance** (bid volume at best bid / (bid + ask volume at best)) explains more of the next mid-price movement than any other single LOB feature. In crypto perpetuals with tight spreads, this is equivalent to our `weighted_imbalance_5lvl` but computed *only* at level 1 — the rest of the book adds noise. The queue *depletion rate* (how fast top-of-book volume is consumed per unit time) is an even stronger signal for imminent price impact.
→ arXiv:1512.03492; also validated in arXiv:2602.00776 feature rankings

**Finding 3.3 — Large trade informativeness: block trade adverse selection in crypto.**
A 2024 paper in *The British Accounting Review* finds that **delayed block trades in crypto carry greater information content** than immediate fills — implying that large orders split into smaller pieces retain directional information across time. This directly motivates our existing `large_trade_share` and `large_buy_share`/`large_sell_share` features, but suggests adding: the *time between large trades* (inter-arrival of block-sized orders) as a measure of institutional accumulation pace.

**Finding 3.4 — Flow toxicity beyond VPIN: the "Bitcoin wild moves" paper uses order flow toxicity to explain jump episodes.**
Salehi et al. (*Research in International Business and Finance*, 2026) document that order flow toxicity — measured via VPIN and its extensions — precedes large price moves in Bitcoin by 15–60 minutes. The strongest precursor signal is not VPIN level but **VPIN acceleration** (d(VPIN)/dt), i.e., how quickly toxicity is rising. This is a direct analogue to our `tfi_acceleration` feature but for flow toxicity rather than directional flow.

**New signals to consider (ranked by ease of implementation):**
| Signal | Formula sketch | Priority |
|--------|----------------|----------|
| Level-1 queue depletion rate | `Δ(bid_vol_L1) / Δt` per step | High |
| VPIN acceleration | `VPIN(t) - VPIN(t-k)` (already have VPIN) | High |
| Inter-arrival of large trades | `steps_since_last_large_trade` | Medium |
| Permanent price impact (Hasbrouck) | Requires VAR on (mid, sign, vol); expensive | Low |
| Bid-ask bounce vs permanent component | Glosten-Harris decomposition | Low |

---

### 4. Volatility Surface / Funding Rate Curve Features

**Finding 4.1 — Funding rate IS a mean-reversion signal at multi-hour horizons, but the intraday edge is small.**
He, Manela, Ross & von Wachter (arXiv:2212.06888, "Fundamentals of Perpetual Futures") establish that the funding rate anchors the perp price to spot through no-arbitrage. Their key empirical finding: **extreme funding rates (>0.1% per 8h) predict perp price mean-reversion toward spot within 30–120 minutes**. However, at our tick scale (1–2s per step), the mean-reversion signal has negligible precision — it's diluted by microstructure noise. We already have `funding_zscore` and `funding_rate_raw` in our feature set; the paper validates that these are correct but suggests they operate at lower frequency.

**Finding 4.2 — Cross-exchange funding rate divergence is a stronger signal than same-exchange rate.**
Chance & Joshi (2025, "New Limits to Arbitrage: Evidence from Crypto Perpetual Futures Markets") document that during stress events (Terra collapse, FTX collapse, SVB), funding rates *diverge* across exchanges before converging. The divergence itself is a signal of arbitrage capacity exhaustion, which precedes directional moves. For our DEX context: if DEX funding diverges from CEX funding, the direction of the spread predicts the next hourly move.

**Finding 4.3 — Implied volatility from options predicts realized volatility asymmetrically.**
Deterministic modelling of crypto IV (Springer *Financial Innovation*, 2024) confirms that crypto implied volatility is useful as a vol-of-vol proxy, but the signal is low-frequency (daily/hourly). At tick scale, realized vol (`realvol_10`, `bipower_var_20`) dominates over IV. The vol surface adds no marginal value at our resolution.

**Finding 4.4 — Funding rate TERM STRUCTURE across multiple perps is a cross-symbol regime indicator.**
The Kim & Park paper (arXiv:2506.08573) on funding rate design notes that when multiple symbols simultaneously flip to negative funding, it signals a broad deleveraging regime. A **cross-symbol average funding** feature (`mean_funding_across_25_symbols`) would be a market-wide regime indicator. This is cheap to compute given we already load funding data.

**Practical Assessment:**
- **Already have:** `funding_zscore`, `funding_rate_raw` — good, but low frequency
- **New — cheap:** `mean_funding_zscore_cross_symbol` (average funding z-score across all 25 symbols at each step) — 1 feature encoding market-wide leverage regime
- **New — medium:** `funding_btc_minus_self` (BTC funding - own-symbol funding) as relative crowdedness signal
- **Skip:** IV surface, options-derived features — not available on DEX and low-frequency anyway

---

### 5. On-Chain Signals: Liquidations, Open Interest, Funding Divergences

**Finding 5.1 — Liquidation cascade data has demonstrably strong predictive power at 5–30 minute horizons.**
The practitioner literature (XT Blog, Gate Crypto Wiki, multiple cefi data providers) documents that large liquidations ($10M+ in 1 hour) reliably precede continuation moves: long liquidations → further price drops (stop-cascade), short liquidations → further rallies. The key feature is not raw liquidation size but **liquidation acceleration** and **cross-symbol co-occurrence** (when 10+ symbols liquidate simultaneously, it signals a systemic event).

**Finding 5.2 — Open interest changes have opposite predictive value in different regimes.**
Rising OI during a rally = more longs → bullish continuation. Rising OI during a decline = more shorts → potential short squeeze reversal. The signal is **OI change rate × direction**, not OI level. For DEX perpetuals, this data may be available via on-chain indexers depending on the exchange.

**Finding 5.3 — Funding rate divergence (DEX vs. CEX) is a structural alpha signal for DEX-specific trading.**
When DEX funding deviates from CEX counterpart funding, it signals either: (a) DEX users are positioned differently (often more retail/directional), or (b) arbitrage is blocked. This creates persistent mispricings that CEX arbitrageurs eventually correct. The direction of correction is predictable.

**Finding 5.4 — On-chain data latency is the critical barrier for tick-scale use.**
Most on-chain liquidation data from protocols (dYdX, GMX, Hyperliquid) is available with 1–10 block latency (≈2–12 seconds on modern L2s). This is borderline for our system (~1–2s per step for BTC). For slower symbols (PENGU, FARTCOIN), 12s latency is within 5–10 steps — potentially usable.

**Practical Assessment:**
- **Cross-symbol liquidation flag:** Binary flag = 1 if total liquidations across all symbols in past 10 minutes exceeded 1σ threshold. Requires external data feed (Coinalyze, Velo, or protocol events). **High value, medium cost.**
- **OI change rate:** `Δ(open_interest) / open_interest` over past N steps. Requires OI data per step. **Medium value, high cost if not already in data.**
- **DEX vs. CEX funding spread:** Requires CEX funding feed as reference. **Medium value, medium cost.**
- **Current verdict:** These signals require data sources not in current Parquet pipeline. Worth a data acquisition experiment before feature implementation.

---

## Ranked Implementation Roadmap

| Rank | Feature | Type | Cost | Est. Value | Risk |
|------|---------|------|------|------------|------|
| 1 | `r_BTC_lag1`, `r_ETH_lag1` (cross-symbol returns) | Cross-symbol | Low: already have BTC/ETH data | High for low-liq symbols | Noise for BTC/ETH themselves |
| 2 | Level-1 queue depletion rate | Microstructure | Low: from orderbook | High: strongest 1-tick predictor | Correlated with existing OFI |
| 3 | `vpin_acceleration` (ΔVPIN) | Microstructure | Trivial: have VPIN | Medium: new dimension | Collinear with delta_TFI |
| 4 | `mean_funding_zscore_cross_symbol` | Cross-symbol | Low: have funding data | Medium: regime indicator | Low-freq signal |
| 5 | Symbol embedding (lookup, dim=8) | Architecture | Low: 8 params/symbol | Small: normalization handles it | Minimal at tick scale |
| 6 | BTC regime flag (HMM 3-state) | Cross-symbol | Medium: daily HMM run | Medium: amplifies cross-symbol | Stale if regime shifts intraday |
| 7 | `inter_arrival_large_trades` | Microstructure | Low: from trade data | Medium: institutional pace | Sparse for low-liq symbols |
| 8 | Cross-symbol liquidation flag | On-chain | High: new data pipeline | High: cascade predictor | Latency constraint |
| 9 | OI change rate | On-chain | High: new data pipeline | Medium | Latency constraint |

---

## Cautions for Fee-Constrained Tick-Scale System

1. **Cross-symbol features add input dimension → risk of overfitting.** With ~28K test steps/symbol and window=50, we already push the MLP capacity. Adding BTC/ETH features effectively doubles input width for 23 non-BTC/ETH symbols. Recommend: test on SUBSET of 5 low-liquidity symbols first.

2. **Low-frequency signals (funding, regime, OI) pollute tick-scale features.** These signals repeat for hundreds of steps — the MLP will amplify them relative to faster-changing features. Use explicit multi-timescale architecture (or just pass them as context scalars, not in the window).

3. **BTC/ETH features are trivially self-predicting for BTC/ETH themselves.** Mask these features to zero for BTC and ETH symbols or train separate models.

4. **"One change at a time" principle (Key Discovery #2).** The ranking above suggests implementing r_BTC_lag1/r_ETH_lag1 first, running a full evaluation, then proceeding. Multiple simultaneous additions have historically caused regressions.

5. **Paper-to-DEX gap:** Most microstructure papers use CEX data (Binance, Bybit). DEX order books are synthetic/virtual AMM curves with different imbalance dynamics. Our existing features are empirically tuned on DEX data — cross-asset features from the literature may transfer less cleanly.

---

## Sources

### Kept
- **Bieganowski & Ślepaczuk, arXiv:2602.00776** (Jan 2026) — Direct: cross-asset crypto microstructure, OFI/spread/VWAP universality, CatBoost+GMADL on Binance Futures perps. Most directly relevant paper in this search.
- **Price Transmission BTC→ALTs** (Asia-Pacific Financial Markets, 2026) — Direct: BTC lead-lag, VAR, trading strategy. Novel: regime-specific, liquidity-dependent lag.
- **Campigli, Bormetti & Lillo, arXiv:2212.12687** (2022, revised 2023) — Authoritative: time-varying information content, permanent price impact. Extends VPIN framework.
- **Cont, Kukanov & Stoikov, arXiv:1512.03492** (2015) — Seminal: queue imbalance as one-tick predictor. Still dominant citation.
- **He, Manela, Ross & von Wachter, arXiv:2212.06888** (2022/2024) — Authoritative: perpetual futures fundamentals, funding rate no-arbitrage. Foundation paper.
- **Chance & Joshi 2025** ("New Limits to Arbitrage") — Practical: funding rate divergence cross-exchange as signal.
- **Wu, arXiv:2504.07923** (Apr 2025) — GNN multi-asset; less relevant at our scale but useful architecture reference.
- **Zhong et al., arXiv:2505.02139** (May 2025) — LOBench: LOB representation learning; motivates symbol embedding experiments.
- **Kim & Park, arXiv:2506.08573** (Jun 2025) — Funding rate design; cross-symbol funding convergence signal.
- **Salehi et al.** (*RIBAF* 2026) — Order flow toxicity → Bitcoin price jumps; validates VPIN acceleration idea.

### Dropped
- **arXiv:2205.00974** (Cross Crypto Relationship Mining for BTC prediction) — Uses longer-term graph features (news, social); not microstructure.
- **arXiv:2507.22409** (Multi-scale volatility spillovers) — Daily frequency, too low-resolution.
- **VolGAN / implied vol papers** — Options-based features not available on DEX and daily frequency.
- **Regime-Specific HMM papers (preprints.org 2026)** — Bitcoin-only, coarse-grained regime. Useful pattern but no new features directly.
- **GNN arbitrage detection** — Different problem (arbitrage path detection vs. direction prediction).

---

## Gaps & Next Steps

1. **Tick-scale BTC lead-lag for DEX data specifically:** All papers use CEX (Binance 1-min). At our 100-trade batches on a DEX, the lag structure may differ due to: AMM liquidity, different arbitrageur population, oracle-based pricing. **Experiment: compute cross-correlation between BTC step returns and lagged ALT returns in our Parquet data to measure actual lag distribution.**

2. **Queue depletion rate from our orderbook data:** We have `weighted_imbalance_5lvl` but not the *rate of change* of level-1 queue. Check if `Δ(bid_vol_L1)` is computable from our snapshot-based orderbook Parquet.

3. **On-chain liquidation/OI data availability:** Current pipeline has no on-chain event stream. Evaluate: does our exchange (Pacifica DEX) expose liquidation events via API? If yes, this is a high-value addition.

4. **Symbol embedding ablation:** Simple experiment — add `symbol_id` as integer input (or 8-dim lookup) to existing architecture. Expected effect: small but interesting for symbols with unusual dynamics (FARTCOIN, WLFI).

5. **GMADL vs focal loss:** The 2602.00776 paper uses GMADL (direction-aware loss weighted by return magnitude). We use focal loss. Direct comparison warranted — both target the same problem (imbalanced directional classification) but GMADL has stronger theoretical grounding via return magnitude weighting.
