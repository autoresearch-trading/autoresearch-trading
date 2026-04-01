# DEX Perpetual Futures Alpha Sources: A Comparative Analysis

**Date:** 2026-03-31  
**Context:** Non-HFT directional trading on DEX perpetual futures (Hyperliquid/Pacifica). Current system uses 13 microstructure features with a supervised MLP classifier, holding positions 1200+ steps (~minutes to hours). Sortino 0.353, 9/23 symbols passing.

---

## Executive Summary

Eight distinct alpha signal families exist for DEX perpetual futures beyond pure tick-level microstructure. Of these, **funding rate dynamics**, **liquidation/OI concentration**, and **cross-venue flow** have the strongest evidence and are most actionable for a non-HFT supervised classifier. Whale tracking and blockchain metrics are promising but data-heavy. Sentiment and pure technicals are weak at the timescales that matter.

The key strategic insight: **your current microstructure features capture the fastest-decaying alpha. The biggest untapped sources are structural signals unique to DEX perps — funding rates, liquidation cascades, and open interest asymmetries — which decay on minutes-to-hours timescales and are directly complementary to what you already have.**

---

## Signal Family Comparison

### 1. Microstructure (What You Already Have) ✅

**Signals:** OFI, VPIN, microprice deviation, trade arrival rate, directional conviction, Hawkes branching, etc.

**Evidence strength:** Strong. Bieganowski & Slepaczuk (2026, arXiv:2602.00776) document stable cross-asset LOB microstructure patterns across crypto assets spanning orders of magnitude in market cap. SHAP analysis shows order flow imbalance, bid-ask spreads, and VWAP-to-mid deviations are the universally dominant predictive features. This is exactly what your 13 features capture.

**Decay:** Very fast — signal at lag 0, decays by lag 1 (your T47 confirmed this). MLP learns nonlinear trajectory patterns in window=50 that extend useful life.

**Your status:** Fully exploited. All 13 features and hyperparameters are exhaustively swept. This is your local optimum.

**Relevance to your setup:** You ARE this. The question is what to stack on top.

---

### 2. Funding Rate Dynamics 🔥 HIGH PRIORITY

**Signals:**
- Funding rate level (extreme positive/negative = crowding)
- Funding rate velocity (rate of change)  
- Cross-venue funding differentials
- Basis (perp price − spot price) and basis velocity
- Funding rate mean-reversion tendency

**Evidence strength:** Strong — both academic and practitioner.
- Chen et al. (2024, arXiv:2402.03953) show funding rates are the core price-anchoring mechanism in perps. On oracle-priced DEXs (GMX, GNS), they find traders are pure price-takers and exhibit **asymmetric overreaction to positive news** (more long accumulation), measurable via funding/OI asymmetry.
- Monarq Asset Management (2026) identifies funding rate arbitrage as the dominant non-directional strategy on DEX perps, noting that funding "transforms market structure into opportunity."
- PerpFinder's practitioner guide documents funding rate mean-reversion: extreme rates (>0.1%/8h) "almost always revert toward their long-term average," creating directional signals.
- Multiple sources confirm funding rate extremes predict short-term reversals because overleveraged positions force liquidations.

**Decay:** Slow — minutes to hours. Funding settles on fixed schedules (1h or 8h depending on venue), so signals persist across many of your 100-trade steps.

**How it fits your system:**
- **Direct feature candidates:** `funding_rate`, `funding_rate_velocity`, `basis_pct`, `funding_rate_zscore_24h`
- These would be slow-moving compared to your microstructure features — exactly the kind of regime context your MLP might use to gate directional bets
- Your T42 showed funding is negligible as a *cost* (0.16% of fee barrier), but that doesn't mean it's useless as a *signal*. The funding rate captures crowd positioning, not transaction costs.

**Risk:** You already proved `include_funding=True` loading is slow (200K+ parquet files per symbol). Would need an efficient funding rate feature pipeline, not raw data loading.

**Actionability:** ⭐⭐⭐⭐⭐ — Directly testable with Pacifica data. Most promising new signal family.

---

### 3. Liquidation Cascades & Open Interest Concentration 🔥 HIGH PRIORITY

**Signals:**
- Open interest level and velocity (OI growth rate)
- Long/short OI ratio (asymmetry)
- Liquidation volume (recent liquidations as fraction of OI)
- Liquidation index: LIQ_long vs LIQ_short (from Chen et al.)
- OI concentration at specific price levels ("liquidation maps")

**Evidence strength:** Strong.
- Chen et al. (2024) provide the most rigorous treatment: on CEXs (Binance, Bybit), unexpected OI changes negatively correlate with volatility (market depth effect via Kyle's model). On oracle-priced DEXs (GMX, GNS), liquidation of long positions has a significantly positive coefficient with volatility, while short liquidation is insignificant — confirming asymmetric overreaction.
- Soska et al. (2021, BitMEX study) first documented correlation between liquidation volume and price volatility.
- The Hyperliquid whale study (Liu, 2026) shows 61.5% of whale trades are profitable, and gradient boosting achieves 89.64% accuracy predicting whale trade outcomes when using 10-trade historical windows — suggesting OI/liquidation patterns are genuinely predictive.

**Decay:** Medium — liquidation cascades unfold over minutes. OI shifts persist for hours.

**How it fits your system:**
- **Feature candidates:** `oi_growth_rate`, `long_short_oi_ratio`, `recent_liquidation_volume_normalized`, `oi_concentration_near_price`
- These capture *structural fragility* — when the market is positioned for a squeeze. Complementary to microstructure which captures *flow dynamics*.
- The asymmetric liquidation finding (longs liquidated more during volatility) could improve your classifier's handling of drawdown-prone regimes.

**Data availability:** Depends on Pacifica API. OI and liquidation data may be available as separate feeds. Check if Pacifica provides aggregate OI or liquidation data per symbol.

**Actionability:** ⭐⭐⭐⭐ — High if data is available. The asymmetric liquidation effect alone could be a powerful regime feature.

---

### 4. Cross-Venue & Cross-Asset Flow 🟡 MEDIUM PRIORITY

**Signals:**
- CEX-DEX price spread for same asset
- Cross-symbol momentum (BTC leads alts)
- Cross-symbol correlation regime (rho changing = regime shift)
- Bitcoin dominance as regime indicator

**Evidence strength:** Moderate.
- Alexander et al. (2020) found BitMEX derivatives lead price discovery over spot — implying CEX perp prices can predict DEX perp movements.
- The "Foresight/Leverage" system (Knox, 2026) uses `trend_alignment` across multiple timeframes as a key factor, finding that counter-trend trades should be harder to trigger.
- Zhang Wei (2025, arXiv:2508.02356) integrates BTC dominance and cross-market data (S&P 500) into a multi-head CNN for crypto direction prediction, achieving profit factor 1.15.
- Your own T45 found cross-symbol correlation rho=0.28 at tick level, suggesting moderate but real cross-asset information.

**Decay:** Medium to slow. Cross-venue spreads persist for seconds to minutes. Cross-asset regime shifts persist for hours to days.

**How it fits your system:**
- **Feature candidates:** `btc_return_5m` (as cross-asset lead), `cross_symbol_correlation_regime`, `cex_dex_basis_spread`
- The BTC lead signal could be valuable for alt-coin prediction — add BTC price momentum as a feature for non-BTC symbols.
- Your existing 25 symbols already implicitly capture some cross-asset info via multi-symbol training, but explicit cross-symbol features could help.

**Risk:** Cross-venue data requires additional data sources beyond Pacifica. Cross-symbol features within Pacifica data are more feasible.

**Actionability:** ⭐⭐⭐ — Cross-symbol features (e.g., BTC momentum as feature for alts) are immediately testable. Cross-venue requires new data.

---

### 5. Whale/Large Trader Tracking 🟡 MEDIUM PRIORITY

**Signals:**
- Large trade detection (trade size > N × median)
- Whale position changes (on-chain wallet monitoring)
- Copy-trading signals from top performers
- Account value concentration

**Evidence strength:** Moderate, from a single but detailed study.
- Liu (2026, Hyperliquid whale study) analyzed 7,716 whale trade fills from 43 addresses. Key findings:
  - Following whales with account value ≥ $50M yields **98.60% win rate and +12.00% PnL** over 77 days
  - Gradient boosting achieves 89.64% accuracy predicting whale trade outcomes with 10-trade history window
  - Historical context (lagged features) improves loss-class accuracy by 4.52%
  - However: sample is small (77 days, 43 whales), likely survivorship-biased, and win rates at lower thresholds are much weaker

**Decay:** Medium — whale positions unfold over minutes to hours.

**How it fits your system:**
- **Feature candidates:** `large_trade_imbalance` (fraction of volume from trades > 10× median), `whale_trade_direction`  
- You already capture some of this via `directional_conviction` and `vpin`, which measure informed flow. Explicit large-trade detection could add incremental value.
- On-chain wallet tracking requires blockchain data, not available from Pacifica.

**Actionability:** ⭐⭐⭐ — Large trade detection within your existing trade data is easy to implement. Full whale tracking needs blockchain data.

---

### 6. Blockchain/On-Chain Metrics 🟠 LOW-MEDIUM PRIORITY

**Signals:**
- Hash rate ribbons (SMA-30 vs SMA-60 crossover)
- Mining difficulty changes
- Cost per transaction (CPTRA)  
- Number of wallet users (MWNUS)
- Network transaction volume

**Evidence strength:** Moderate but long-horizon.
- King, Dale & Amigó (2024, arXiv:2403.00770) test 21 blockchain metrics as Bitcoin trading indicators. Key findings:
  - MWNUS (wallet users) has highest Chatterjee correlation with BTC price (0.987)
  - Blockchain ribbons generate 56-72% winning trades on long signals
  - Short signals are universally weak across all blockchain metrics
  - Adjusted CPTRA (cost per transaction) is the strongest predictor in Random Forest models
  - LSTM with preprocessed blockchain features achieves MASE 0.74 (10-day horizon)

**Decay:** Very slow — days to weeks. These are macro regime signals, not trade-level features.

**How it fits your system:**
- These operate on completely different timescales than your 100-trade steps. They're useful as daily regime features, not per-step features.
- Could be useful as a training-time filter (e.g., train only on "healthy network" periods) or as a slow-moving context feature updated daily.

**Actionability:** ⭐⭐ — Wrong timescale for your classifier. Would need architectural changes (separate regime model feeding into trade model). Not recommended as a next step.

---

### 7. Volatility Regime Detection 🟡 MEDIUM PRIORITY

**Signals:**
- Hurst exponent (H > 0.6 = trending, H < 0.4 = mean-reverting)
- Realized volatility regime (Parkinson or Garman-Klass)
- Volatility-of-volatility (you already have `vol_of_vol`)
- Funding rate regime (extreme = volatile, moderate = stable)

**Evidence strength:** Moderate.
- Vadim's production scalping engine (vadim.blog) uses Hurst exponent to gate between mean-reversion (OU Z-Score when H<0.4) and momentum (OFI when H>0.6). "The dead zone around 0.5 is the most dangerous."
- Qian et al. (2022) show incorporating regime-switching into Bitcoin volatility models "greatly improves predictive accuracy, particularly during turbulent periods."
- The PERP Prediction framework (Liu, 2026) explicitly notes "when volatility spikes, simple signals degrade" — achieving only 50% accuracy at 10-minute horizons.

**Decay:** Medium — regimes persist for minutes to hours.

**How it fits your system:**
- You already have `vol_of_vol` (feature #5). Adding a Hurst exponent or explicit regime classifier could help your MLP gate between momentum and mean-reversion strategies.
- **Feature candidate:** `hurst_exponent_50` (computed over same 50-step window)
- Your triple barrier labeling already implicitly captures some regime sensitivity, but an explicit regime feature could sharpen classification.

**Actionability:** ⭐⭐⭐ — Hurst exponent is computationally cheap and complementary to your existing features. Good incremental experiment.

---

### 8. Sentiment & News Flow 🔴 LOW PRIORITY

**Signals:**
- Social media sentiment (Twitter/X, Reddit)
- News event detection
- Google Trends / search volume
- GDELT sentiment index

**Evidence strength:** Weak at your timescale.
- The PERP Prediction framework using momentum + mean-reversion + volatility achieves only **50% accuracy** at 10-minute horizons — "barely better than random."
- Knox's Leverage bot uses `news_sentiment` with max 10 points out of 100 — explicitly described as "a sanity check, not a signal generator."
- Multiple academic studies (Chu et al., 2020; Madan et al., 2015) find 50-55% directional accuracy at short horizons, even with sentiment.
- Zhang Wei (2025) includes GDELT sentiment but as one of many inputs to a multi-head CNN; marginal contribution unclear.

**Decay:** Variable — breaking news matters for seconds, but sentiment persistence is low-quality and noisy at sub-hour timescales.

**How it fits your system:** Poorly. Your trade-batch steps (1-2 seconds per step) are far faster than sentiment updates (15-minute intervals at best). Sentiment is a daily/hourly regime feature, not a trade feature.

**Actionability:** ⭐ — Not recommended. Wrong timescale, weak signal, complex data pipeline required.

---

## Comparison Matrix

| Signal Family | Evidence | Decay Rate | Data Available? | Fits Your Timescale? | Complementary? | Priority |
|---|---|---|---|---|---|---|
| **Funding Rate** | Strong | Slow (hours) | Yes (Pacifica) | ✅ Yes (regime) | ✅ Highly | 🔥 **#1** |
| **Liquidation/OI** | Strong | Medium (mins) | Maybe (check API) | ✅ Yes | ✅ Highly | 🔥 **#2** |
| **Cross-Asset** | Moderate | Medium | Partially | ✅ Yes | ✅ Yes | 🟡 **#3** |
| **Whale Detection** | Moderate | Medium | Partially | ⚠️ Somewhat | ⚠️ Partial overlap | 🟡 **#4** |
| **Vol Regime** | Moderate | Medium | Yes (compute) | ✅ Yes | ✅ Yes | 🟡 **#5** |
| **Blockchain Metrics** | Moderate | Very slow | No (new source) | ❌ Wrong timescale | ⚠️ Orthogonal | 🟠 **#6** |
| **Sentiment** | Weak | Variable | No (new source) | ❌ Too slow | ❌ Marginal | 🔴 **#7** |

---

## Recommended Next Steps (Ranked)

### 1. Funding Rate Features (Highest Expected Value)
Add 2-3 funding rate features to your v11b feature set. The funding rate captures **crowd positioning** — when it's extreme, the market is structurally fragile for a reversal. This is exactly the kind of slow-moving regime context that could help your MLP make better directional calls.

**Concrete experiment:** Add `funding_rate_zscore` and `basis_pct` as features #13 and #14. This requires solving the Pacifica funding data loading problem efficiently (cache aggressively, perhaps precompute per-symbol daily features).

### 2. Liquidation/OI Asymmetry Features
If Pacifica provides OI or liquidation data, add `long_short_oi_ratio` and `oi_growth_rate`. The Chen et al. finding that long liquidations have a significantly positive volatility coefficient while short liquidations don't is directly tradeable.

### 3. Cross-Symbol Lead Feature
Add BTC's recent return (last N steps) as a feature for all non-BTC symbols. This exploits the well-documented BTC-leads-alts phenomenon without requiring external data.

### 4. Hurst Exponent
Cheap to compute, orthogonal to existing features, and backed by practitioner evidence. Add as feature #15 or #16 after the above experiments.

---

## Key Insight: DEX-Specific Alpha vs. Generic Crypto Alpha

The most underexploited alpha sources are **structurally unique to DEX perpetuals**:

1. **Transparent liquidation mechanics** — on DEXs, liquidation logic is observable and auditable (Monarq, 2026). This makes liquidation prediction more feasible than on CEXs.
2. **Funding rate transparency** — all funding rate calculations and parameters are on-chain and verifiable.
3. **Uninformed trader overreaction** — Chen et al. (2024) specifically document that DEX oracle-pricing model traders act as "pure price takers" who "overreact more to positive news than negative, leading to increased propensity for long position accumulation." This asymmetry is a systematic, exploitable behavioral pattern.
4. **Liquidity provider dynamics** — the relationship between LP behavior, OI, and volatility creates feedback loops unique to DEXs that don't exist on CEXs.

Your current microstructure features are excellent but "timescale-local." The next edge likely comes from **stacking structural DEX signals (funding, liquidation, OI) on top of your microstructure base.**

---

## Sources

### Academic Papers
- Bieganowski & Slepaczuk (2026). "Explainable Patterns in Cryptocurrency Microstructure." arXiv:2602.00776. https://arxiv.org/abs/2602.00776
- Chen, Ma & Nie (2024). "Exploring the Impact: How Decentralized Exchange Designs Shape Traders' Behavior on Perpetual Future Contracts." arXiv:2402.03953. https://arxiv.org/abs/2402.03953
- King, Dale & Amigó (2024). "Blockchain Metrics and Indicators in Cryptocurrency Trading." arXiv:2403.00770. https://arxiv.org/abs/2403.00770
- Zhang Wei (2025). "Neural Network-Based Algorithmic Trading Systems: Multi-Timeframe Analysis and High-Frequency Execution in Cryptocurrency Markets." arXiv:2508.02356. https://arxiv.org/abs/2508.02356
- Liu (2026). "Combining Simulation and Machine Learning Analysis of Whale Trading on Hyperliquid." https://medium.com/@gwrx2005/combining-simulation-and-machine-learning-analysis-of-whale-trading-on-hyperliquid-93f10d96941b
- Liu (2026). "The PERP Prediction Framework: Signals, Simulation, and Forecast Analysis." https://medium.com/@gwrx2005/the-perp-prediction-framework-signals-simulation-and-forecast-analysis-9e69edf8b55e

### Practitioner Sources
- Vadim (2026). "ML Features Powering a Crypto Scalping Engine." https://vadim.blog/ml-features-crypto-scalping-research-papers
- Knox (2026). "Leverage: Porting the Foresight Signal Stack to Crypto Perpetuals." https://www.jeremyknox.ai/blog/leverage-perpetuals-futures-bot
- PerpFinder (2026). "Advanced Perpetual Futures Trading Strategies for DeFi." https://perpfinder.com/guide/advanced-perp-trading-strategies
- Monarq Asset Management (2026). "Perp DEXs in 2025: The Shift From Subsidies to Market Structure." https://medium.com/@Monarq_Mgmt/perp-dexs-in-2025-the-shift-from-subsidies-to-market-structure-68a1138f4c10
