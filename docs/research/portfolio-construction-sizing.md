# Research: Portfolio Construction & Position Sizing for Multi-Symbol DEX Perpetual Futures

*Date: 2026-03-30 | Context: 23 symbols, binary classifier (flat/long/short), Sortino=0.353, 9/23 passing*

---

## Summary

Equal-weight, binary-position portfolios systematically leave alpha on the table and concentrate drawdown risk in correlated assets. For a multi-symbol DEX perp system, the highest-leverage interventions are (1) **meta-labeling / confidence weighting** (reduces low-quality trades, improves both Sortino and passing count simultaneously), (2) **a market-regime kill switch** (BTC trend filter prevents correlated 20%+ drawdowns that fail the guardrail), and (3) **rolling performance-based symbol selection** (focus on symbols with demonstrated recent edge). Risk parity and HRP are excellent portfolio-level tools but require passing symbols to work on first — they are stage-2 improvements.

---

## Findings

### 1. Kelly Criterion: Optimal Position Sizing from Edge and Variance

**Core formula:** `f* = (p·b - q) / b` where `p` = win rate, `q = 1-p`, `b` = win/loss ratio.

**For a classifier:** The bet size is derived from the model's probability estimate:
```
f_kelly = (p̂ - (1-p̂)/odds) = (p̂·(1+b) - 1) / b
```
At 55% win rate and 1.71 profit factor (current v11b), full Kelly ≈ 20–30% of capital per trade — far too aggressive. **Half-Kelly or Quarter-Kelly** retains ~75% of optimal log-growth while cutting drawdown roughly in half.

**Caveat for crypto:** Kelly assumes i.i.d. bets. With min_hold=1200 steps and correlated crypto assets, full Kelly dramatically overstates the safe leverage. Recommended fraction: **0.25×Kelly** for live trading.

**Impact on {Sortino / passing}:** Kelly primarily helps Sortino (larger positions on high-edge signals). It doesn't directly help passing count and can hurt it if miscalibrated (overcrowding on bad symbols). Best applied *after* meta-labeling to avoid sizing up on misclassified signals.

*Key references: Kelly (1956) original; practical treatment in Thorp (2008); crypto application with half/quarter fractions well-documented in practitioner literature.*

---

### 2. Per-Symbol Confidence Weighting (Meta-Labeling)

**Core idea (López de Prado, 2018, *Advances in Financial Machine Learning*, Ch. 10):** The primary classifier gives direction. A secondary binary model — or simply the classifier's output probability — determines *size*. This separates the question "which direction?" from "how confident are we?".

**Two practical approaches for our system:**

**A. Direct probability sizing:**
```python
softmax = F.softmax(logits / temperature, dim=-1)
max_prob = softmax.max(dim=-1).values          # ∈ [1/3, 1.0] for 3-class
confidence = (max_prob - 1/3) / (2/3)          # normalize to [0, 1]
position_size = confidence ** alpha             # alpha ∈ [0.5, 2.0] — tune
```
With temperature calibration (T=1.5–2.0) to produce well-spread probabilities, this acts as a continuous position multiplier.

**B. Meta-label model:** Train a *second* MLP on the same features that predicts whether the primary model's triple-barrier label was correct (binary: right/wrong). The meta-model's output probability becomes the position size scalar. López de Prado shows this improves precision while maintaining recall — exactly what the 20% drawdown guardrail demands.

**Impact on {Sortino / passing}:** ✅✅ **Both improve.** Low-confidence signals produce smaller positions → smaller drawdowns → more symbols stay under 20% DD. High-confidence signals get full sizing → higher Sortino on passing symbols. This is the highest-priority intervention given the current architecture.

*References: López de Prado (2018) Ch. 10 "Bet Sizing"; Meta-Labeling Wikipedia; Joubert (2022) "Meta-Labeling: Theory and Framework" SSRN 4032018.*

---

### 3. Risk Parity / Volatility Targeting

**Core finding (Moreira & Muir, 2017, *Journal of Finance* 72(4):1611–1644):** Managed portfolios that scale position size inversely with realized variance produce large alphas and higher Sharpe ratios across all major factor portfolios. The mechanism: volatility clusters, so high-vol periods predict future high-vol and lower risk-adjusted returns.

**Inverse-volatility position sizing:**
```python
σ_realized = rolling_std(returns, window=N)    # N = 50–200 steps
σ_target = target_vol                          # e.g., 2% per step
w_vol = σ_target / σ_realized                 # position scale ∈ (0, max_leverage]
w_vol = clip(w_vol, 0.1, 2.0)                 # cap the range
```

**Per-symbol application:** Each symbol has its own realized vol series. Symbols with low recent volatility get larger positions; high-vol symbols get smaller ones. This is equivalent to equalizing risk contribution across symbols.

**For the drawdown guardrail:** When a symbol's realized vol spikes (e.g., during news events), the position is automatically reduced → drawdown is bounded. This directly helps the "<=20% drawdown" guardrail.

**Baltas & Kosowski (2017, *JFE*):** Explicitly show volatility-scaled positions improve risk-adjusted returns in multi-asset momentum portfolios. Their "sigma-scaling" can directly be ported to our per-step position sizing.

**Cederburg et al. (2020, *JFE* 138:95–117):** Critical follow-up showing benefits are *strategy-specific* — work best for momentum/trend strategies (which ours is), less clear for value. Not a universal free lunch.

**Impact on {Sortino / passing}:** ✅ Sortino (tighter risk on high-vol symbols). ✅ Passing count (drawdown guardrail is harder to breach). Moderate implementation complexity — realized vol is already computable from the features in prepare.py.

---

### 4. Regime-Conditional Trading: BTC Trend Filter

**Key finding (Starkiller Capital, Drogen et al., 2022):** A top-quintile cross-sectional crypto momentum strategy with a simple 5/50 EMA BTC trend filter improved annualized return from 37.8% to **93.3%** and reduced max drawdown from 75% to **45%**.

**Why it works in crypto:** Crypto assets show 0.80+ pairwise correlation during drawdowns. A portfolio of 23 long positions is effectively **one big BTC-beta bet**. When BTC enters a sustained downtrend, all 23 symbols are likely to decline together, blowing through the 20% drawdown limit across the entire portfolio simultaneously.

**Practical implementation options:**

**Option A — Hard regime filter (binary):**
```python
btc_trend = rolling_return(btc_price, window=500)  # ~5–10 days at our freq
if btc_trend < -0.05:  # BTC down >5% recently
    force_flat_all_symbols()
```

**Option B — Soft regime weighting (continuous):**
```python
regime_score = sigmoid(btc_trend / vol_btc)        # ∈ [0, 1]
position *= regime_score                            # scale down in bad regimes
```

**Option C — Regime as feature:** Add BTC rolling return features to the per-symbol model. The model learns to use BTC regime as a context signal. This is the cleanest integration with the existing MLP architecture.

**RegimeFolio (Zhang et al., arXiv:2510.14986):** ML-based regime detection with sectoral portfolio reweighting in dynamic markets. More sophisticated but same principle.

**Impact on {Sortino / passing}:** ✅✅ Both. The regime filter is the most direct intervention to prevent correlated mass-failure (the key reason why 14/23 symbols *fail*). During bad regimes, forcing flat avoids the drawdown breaches that disqualify symbols.

---

### 5. Correlation-Aware Portfolio: Reducing Crowded Exposure

**The core problem:** Our 23 DEX perp symbols are structurally highly correlated (crypto-wide beta ≈ 0.7–0.9 for most alts). During stress events, correlation spikes toward 1.0. A portfolio of 23 simultaneous long positions is not diversified — it's a leveraged BTC ETF with extra fees.

**Quantifying correlation exposure:**
```python
# Weekly check: what fraction of P&L moves in the same direction on a 10% BTC drop?
beta_i = cov(symbol_returns, btc_returns) / var(btc_returns)
net_delta = sum(position_i * beta_i for all i)
# Professional threshold: if net_delta > 30% of capital, hedge or reduce
```

**Correlation-aware sizing:** When correlations between active positions are high:
```python
corr_penalty = mean_pairwise_correlation(active_symbols)   # ∈ [-1, 1]
position_scale = 1 / (1 + α * corr_penalty)               # α ∈ [1, 3]
```

**From PerpFinder practitioner framework:** Portfolio heat limit of 30–50% of capital deployed. Net directional exposure capped at 30% of total capital. "True diversification requires structural differences, not just different tickers."

**For short positions specifically:** The AdaptiveTrend paper uses a higher Sharpe threshold for shorts (1.7 vs 1.3 for longs) because of elevated correlation risk and positive crypto drift. This is relevant for our flat/long/short classifier — short signals in a bull regime should require higher confidence.

**Impact on {Sortino / passing}:** ✅ Passing count (less correlated drawdowns). Neutral to slightly negative on Sortino if over-applied. A BTC regime filter (§4) already captures most of this effect more directly.

---

### 6. Dynamic Symbol Selection: Rolling Sharpe/Sortino Filter

**Core strategy (AdaptiveTrend, Nguyen 2025, arXiv:2602.11708):** Monthly rebalancing with a Sharpe-based filter. Assets are selected only if their rolling prior-month Sharpe ≥ γ (1.3 for longs, 1.7 for shorts). This alone contributed meaningfully to the strategy's Sharpe of 2.41 vs benchmarks at 1.0–1.5.

**Cross-sectional momentum evidence (Starkiller Capital, 2022):** Strong cross-sectional momentum in crypto: top quintile by 30-day return outperforms by 37.8% annualized (full sample). The effect is robust to subsample selection (20 random universe subsamples).

**For our system — Rolling performance filter:**
```python
# Monthly or rolling evaluation
for symbol in universe:
    val_sortino = evaluate(make_env(symbol, 'val', ...), policy_fn)
    recent_sortino[symbol] = val_sortino

# Select top-k or above threshold for next period
active_symbols = [s for s in universe if recent_sortino[s] > 0.0]
# or: top_k = sorted(universe, key=recent_sortino)[:k]
```

**Key constraint:** Must use the *validation* set for the rolling filter, never the test set, to avoid lookahead bias.

**Why this helps passing count:** Symbols with negative recent Sortino are likely to continue underperforming (momentum effect). Excluding them reduces the denominator (fewer symbols traded) but raises the fraction that pass guardrails. If 14/23 symbols are chronic non-passers due to structural reasons (thin liquidity, low microstructure predictability), dropping them from the active universe makes the portfolio healthier.

**Impact on {Sortino / passing}:** ✅✅ Both. Directly — by only trading symbols with proven recent edge. The trade-off: fewer positions = less diversification = higher portfolio volatility. Size: start by pruning obvious losers (< −0.1 rolling Sortino) rather than aggressive top-k selection.

---

### 7. Hierarchical Risk Parity (HRP)

**Origin (López de Prado, 2016, *Journal of Portfolio Management* 42(4):59–69):** HRP uses hierarchical clustering on the correlation matrix + recursive bisection to allocate weights. Avoids the instability of Markowitz mean-variance (no matrix inversion needed). Consistently outperforms equal-weight and mean-variance OOS in financial data.

**Three-step algorithm:**
1. **Tree clustering:** Build a dendrogram from the correlation distance matrix `d_ij = √(0.5·(1 - ρ_ij))` using linkage (ward, single, average).
2. **Quasi-diagonalization:** Reorder the correlation matrix so similar assets are adjacent.
3. **Recursive bisection:** Allocate weights by splitting the tree, assigning each sub-cluster a weight proportional to its inverse variance.

```python
from sklearn.cluster import linkage
# Or use riskfolio-lib, PyPortfolioOpt, or mlfinlab
hrp_weights = hrp_allocate(returns_df)  # returns dict of symbol → weight
```

**Fast HRP (Springer 2026, Annals of Operations Research):** New algorithms reduce computational complexity to O(N log N) from O(N²). Practical for 23 symbols.

**Practical limitation for our system:** HRP optimizes *capital allocation across passing symbols*. It does **not** improve per-symbol directional accuracy. With only 9/23 passing, HRP tells us how to split capital among those 9 — valuable, but it can't fix the 14 failing ones.

**HRP + Volatility Targeting combo:**
```python
# Use HRP weights as baseline, then scale by inv-vol
hrp_w = hrp_allocate(active_returns)
vol_scale = σ_target / σ_i_realized
final_w = hrp_w * vol_scale
final_w /= final_w.sum()  # renormalize
```

**Impact on {Sortino / passing}:** ✅ Portfolio Sortino (better risk distribution). ✗ Does not improve per-symbol passing rate. **Stage-2 tool** — maximize value after §2–4 are implemented.

*References: López de Prado (2016) JPM; Antonov, Lipton & López de Prado (2025, Risk.net); Fast HRP (2026, Annals of OR).*

---

### 8. Practical Path: From Flat/Long/Short Classifier to Portfolio Allocation

**The canonical pipeline (López de Prado paradigm):**

```
Primary Model (direction)     →  flat / long / short signal
         ↓
Meta-Model (sizing)           →  confidence scalar ∈ [0, 1]
         ↓
Volatility Scaling            →  σ_target / σ_realized
         ↓
Regime Filter                 →  BTC regime gate (0 or 1)
         ↓
Symbol Selection Filter       →  active / inactive flag
         ↓
HRP/Risk Parity Weights       →  capital allocation across symbols
         ↓
Portfolio Heat Cap             →  max total notional / capital
```

**Concretely for our system:**

```python
def get_position_size(symbol, logits, step_info):
    # 1. Direction from primary model (already have this)
    action = logits.argmax()               # 0=flat, 1=long, 2=short

    # 2. Confidence weighting
    probs = softmax(logits)
    conf = (probs.max() - 1/3) / (2/3)    # normalize
    
    # 3. Volatility scaling
    σ_target = 0.02                        # target 2% per-step vol
    σ_realized = step_info['realized_vol']
    vol_scale = min(σ_target / σ_realized, 2.0)
    
    # 4. Regime filter (BTC trend)
    if step_info['btc_trend_30d'] < -0.05:
        return 0  # forced flat in bear regime
    
    # 5. Symbol activity filter  
    if step_info['rolling_sortino_val'] < -0.1:
        return 0  # chronically underperforming symbol
    
    # 6. Final position size
    raw_size = conf * vol_scale
    return action * raw_size  # positive = long, negative = short
```

**Key practitioner insights from reviewed sources:**
- AdaptiveTrend: higher Sharpe threshold for shorts (1.7) vs longs (1.3) — asymmetric alpha quality
- PerpFinder: portfolio heat ≤ 40–50% of capital; net directional exposure ≤ 30%
- López de Prado: separate the *direction decision* from the *sizing decision* — they have different information requirements

---

## Impact Matrix: Which Methods Help Both Sortino AND Passing Count?

| Method | ↑ Sortino | ↑ Passing Count | Complexity | Priority |
|--------|-----------|----------------|------------|----------|
| Meta-labeling / confidence weighting | ✅✅ | ✅✅ | Low | **1st** |
| BTC/market regime kill switch | ✅ | ✅✅ | Low | **2nd** |
| Dynamic symbol selection (rolling filter) | ✅✅ | ✅✅ | Low | **3rd** |
| Volatility targeting (inv-vol scaling) | ✅ | ✅ | Medium | 4th |
| Kelly criterion (fractional, calibrated) | ✅ | ✗ | Medium | 5th |
| Correlation-aware sizing | ✅ | ✅ | Medium | 6th |
| HRP allocation | ✅ (portfolio) | ✗ | High | Stage 2 |
| Full Markowitz MVO | ✗ | ✗ | High | Avoid |

---

## Sources

**Kept:**
- López de Prado (2018) *Advances in Financial Machine Learning*, Ch. 10 & 16 — primary reference for bet sizing and HRP; canonical ML+finance framework
- Moreira & Muir (2017) "Volatility-Managed Portfolios" *JoF* 72(4) — seminal evidence for inv-vol scaling; directly applicable
- Nguyen (2025) "Systematic Trend-Following with Adaptive Portfolio Construction" arXiv:2602.11708 — crypto-specific with ablation study; rolling Sharpe filter + asymmetric allocation
- Starkiller Capital / Drogen, Hoffstein & Otte (2022) "Cross-Sectional Momentum in Cryptocurrency Markets" — practical evidence for BTC regime overlay and symbol selection in crypto perps
- PerpFinder (2025) "Perpetual Futures Risk Management" — practitioner framework for heat limits, correlation management, Kelly application
- López de Prado (2016) "Building diversified portfolios that outperform OOS" *JPM* 42(4) — original HRP paper
- Antonov, Lipton & López de Prado (2025) "Overcoming Markowitz's instability with HRP" *Risk.net* — recent theoretical validation of HRP
- Fast HRP (2026) *Annals of Operations Research* — algorithmic improvements, O(N log N)
- Joubert (2022) "Meta-Labeling: Theory and Framework" SSRN 4032018 — meta-labeling formalization
- Baltas & Kosowski (2017) "Demystifying TSMOM: Volatility estimators, trading rules, pairwise correlations" *JFE* — volatility scaling for momentum multi-asset

**Dropped:**
- RegimeFolio (arXiv:2510.14986) — LLM-based regime detection; interesting but overkill for our setup
- DeMiguel et al. (2009) "Optimal vs. naive diversification" *RFS* — makes case for 1/N; relevant but we're already there
- Various blog posts (investinglayers, medium) — practitioner summaries but no original evidence

---

## Gaps

**Not fully answered:**
1. **Exact calibration of confidence thresholds:** What probability cutoff produces the best trade-off between signal volume and quality for our specific fee structure (fee_mult=11.0)? Requires ablation on val set.
2. **Optimal vol targeting window:** What lookback window for realized vol minimizes the lag/responsiveness trade-off at our trade frequency (~100 trades = 1–2 seconds)? Literature suggests 20–50 periods for high-freq; our "1 step = 100 trades" means ~20–50 is plausible.
3. **DEX-specific regime indicators:** BTC trend filter is well-evidenced for CEX. For DEX perps, funding rates and liquidity depth might be better regime indicators. The current feature set includes `funding_zscore` — worth testing as a regime gate.
4. **Meta-model training data:** With only ~145 days of data, training a separate meta-model may overfit. The direct confidence weighting approach (§2A) is more data-efficient and should be tried first.
5. **HRP with non-Gaussian crypto returns:** Standard HRP uses variance-based distances. For fat-tailed crypto, a CVaR-based HRP (Conditional VaR linkage) might be more appropriate. See riskfolio-lib implementation.

**Suggested next experiments (ordered by expected ROI):**
1. Add softmax confidence multiplier to position sizing (single parameter `alpha`, tune on val)
2. Add BTC rolling return as a regime gate (force flat when BTC_30d_return < threshold)
3. Implement rolling val-Sortino symbol filter; drop symbols with rolling Sortino < 0
4. Implement per-symbol realized-vol scaling with σ_target ≈ 1–2% per step
5. Apply HRP weighting across the passing symbol set
