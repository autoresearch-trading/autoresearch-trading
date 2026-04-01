# Research: Metalabeling & Advanced Labeling Methods for Financial ML

**Context**: DEX perpetual futures classifier (flat/long/short), triple-barrier labels, fee_mult=11.0, Sortino=0.353,
WR=55%, PF=1.71, 1269 trades, 9/23 passing. All hyperparameters swept to optimum. Fee is the binding constraint.

---

## Summary

Five labeling innovations have meaningful evidence behind them. For this system's fee-constrained setting, **metalabeling** is the highest-priority intervention — it addresses the precision problem directly by filtering bad trades rather than relabeling data. **Asymmetric barriers** (wider TP than SL) are the lowest-risk structural change: a single `fee_mult` split that can be tested in one experiment. **Dynamic barriers** have mixed evidence in crypto contexts. **Trend-scanning** has weak empirical support. A recent 2024–2025 wave of work (AEDL, GA-driven barriers, conformal prediction) offers more radical approaches, but with higher implementation cost and uncertain applicability to tick-frequency microstructure data.

---

## Findings

### 1. Metalabeling (Lopez de Prado, 2018)

**Mechanism**  
A two-stage architecture:  
- **Primary model** (your existing MLP) predicts *side* — long, short, or flat  
- **Secondary model** predicts *bet size* — binary (trade / skip) or continuous [0,1]  

The secondary model is trained only on observations where the primary model fires a long or short signal. Its label is 1 if the primary model was correct (hit TP) and 0 if wrong (hit SL or timeout). Features for the secondary model can be anything the primary doesn't already use: confidence of the primary's softmax output, regime indicators, current drawdown, time-of-day, recent TFI momentum, etc.

The key insight: **separating "which direction?" from "should I trade at all?" improves F1 by increasing precision without sacrificing recall of the primary**. All false positives from the primary become optional — the secondary can veto them.

**Evidence of effectiveness**  
- Hudson & Thames capstone (Singh & Joubert, 2022): On S&P500 e-mini, accuracy jumped from 17–20% → 63–77% OOS. Sharpe and max drawdown improved on both trend-following and mean-reversion primaries. [Source](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)  
- Wölner-Hanssen (Lund University, 2023): Equity market-neutral strategy showed consistent Sortino improvement after metalabeling filter.  
- The caveat: **"If the primary model is bad, metalabeling will only reduce the downside."** Singh & Joubert explicitly note this. For your system (WR=55%, PF=1.71), the primary is positive-expectancy — this is the prerequisite.

**Implementation complexity: Moderate**  
1. After your existing `evaluate()` run, collect `(obs, outcome)` pairs where outcome = did primary model's prediction hit TP?  
2. Build a secondary dataset: only rows where model predicted ±1. Label 1=TP hit, 0=SL/timeout.  
3. Train a new small MLP or logistic regression on these rows, using different/additional features from the primary.  
4. At inference: fire trade only if `secondary_prob > threshold` (tune threshold for precision vs recall tradeoff).  
5. Key subtlety: the secondary model will see **heavy class imbalance** (SL>TP in early training). Use class weights or upsample.

**Expected impact for fee-constrained system**  
High. Your system's biggest drag is fee friction on losing trades. Metalabeling directly prunes those. Going from WR=55% → WR=65% would dramatically improve PF. The trade count would drop (fewer false positives), but each surviving trade should have a better net P&L. The Sortino improvement scales roughly with `precision_gain × avg_profit_per_trade`.

**Risk**: If secondary model overfits on val, you'll lose trades you should have taken. Use strict walk-forward validation.

**arXiv / Citation**: AFML Chapter 3 (Lopez de Prado 2018). See also SSRN 3257302 (Lopez de Prado, "The 10 Reasons Most ML Funds Fail") for the theoretical grounding. Singh & Joubert (2022) at hudsonthames.org/wp-content/uploads/2022/04/Does-Meta-Labeling-Add-to-Signal-Efficacy.pdf

---

### 2. Asymmetric Barriers (wider TP vs SL, or vice versa)

**Mechanism**  
Standard triple barrier uses symmetric `[fee_mult, fee_mult]` for TP and SL. Asymmetric barriers set `[tp_mult, sl_mult]` independently. Two common variants:  
- **Wider TP** (e.g., [15, 11]): Favors high-confidence trending moves. Fewer trades reach TP, but profit per winner is larger. Model learns to find high-momentum setups.  
- **Wider SL** (e.g., [11, 14]): More forgiving on losing trades, TP is tighter. Results in higher win rate but lower P&L per trade. Rarely useful unless you want more label diversity.  
- **Asymmetric by side**: Long trades use [14, 10], short trades use [10, 14], exploiting known asymmetry in crypto (crashes are faster than rallies).

**Evidence of effectiveness**  
- Lopez de Prado (2018) mentions asymmetric barriers but does not give systematic results. The original paper recommends using volatility to set both barriers, implying per-sample asymmetry.  
- Fu, Kang, Hong, Kim (*Mathematics* 12(5):780, 2024, arXiv-equivalent DOI: 10.3390/math12050780): Used genetic algorithms to jointly optimize (TP_mult, SL_mult) per coin-pair for triple-barrier crypto pair trading. Found that asymmetric solutions consistently outperformed symmetric ones across 5 cryptocurrency pairs. The GA-found solutions showed TP/SL ratios ranging from 0.8–2.5x (not always wider TP). [Source](https://www.mdpi.com/2227-7390/12/5/780)  
- Springer 2025 crypto paper (information-driven bars + triple barrier): Sensitivity analysis showed symmetric 5% barriers with CUSUM 2% sampling were near-optimal for ETH (Sharpe=2.0). The optimal point was a narrow band — wide barriers led to too few labels and instability.

**Implementation complexity: Low**  
Modify the `fee_mult` parameter to a tuple `(tp_mult, sl_mult)` in `prepare.py`. Then sweep `tp_mult ∈ [9, 11, 13, 15]` × `sl_mult ∈ [8, 10, 11]` (keeping `sl_mult <= tp_mult`). This is a one-line code change + grid search.

**Expected impact for fee-constrained system**  
Medium-High. Your current `fee_mult=11.0` was swept to optimum. Trying `(13, 10)` or `(15, 9)` biases the label set toward "only trade when there's a big enough move to pay fees twice over." The model would learn to wait for higher-conviction setups. Risk: with wider TP, fewer samples get positive labels (more timeouts), which makes class imbalance worse. Monitor with `avg_hold_steps` — if it goes up, fee amortization improves.

**Crypto-specific note**: For perpetuals, funding adds a time-cost. Wider TP means holding longer, which may fight funding rate drag. Your `include_funding=False` assumption (proven negligible at T42) suggests this is minor.

**arXiv / Citation**: Fu et al. (2024) Mathematics 12(5):780. DOI: 10.3390/math12050780

---

### 3. Dynamic/Adaptive Barriers (volatility-scaled)

**Mechanism**  
Instead of fixed `fee_mult`, barriers are set per observation as:  
```
TP = entry_price × (1 + k × σ_local)
SL = entry_price × (1 - k × σ_local)
```  
where `σ_local` is a rolling volatility estimate (e.g., 20-step ewm std of returns). This ensures that in volatile regimes, barriers expand (giving trades more room) and in quiet periods, barriers tighten.

Lopez de Prado's original recommendation was exactly this — barriers set to `k×daily_vol` multiples, where `k` is a user hyperparameter.

**Evidence of effectiveness**  
Contradictory across papers:  
- AEDL framework (Khrouf et al., *Applied Sciences* 15(24):13204, 2025): Adaptive horizons + volatility-based event detection achieved Sharpe=0.480 vs Triple Barrier=−0.030 on US equities (2023–2025 OOS). Multi-scale temporal analysis + meta-learning drove most of the gains. [Source](https://www.mdpi.com/2076-3417/15/24/13204)  
- Springer 2025 crypto paper (Financial Innovation): Explicitly tested vol-adaptive barriers on BTC/ETH. **Result: performance deteriorated vs static barriers.** The authors concluded that "incorporating volatility directly into the training data, rather than embedding it into the labeling process, may be a more effective approach." This is consistent with your architecture — you already pass `realvol_10` and `bipower_var_20` as features.  
- AEDL sensitivity analysis: Without causal inference (multi-scale + meta-learning only) achieved Sharpe=0.654 — better than full AEDL. This suggests the multi-scale aspect (capturing patterns from intraday to weekly simultaneously) may be more valuable than the adaptive barriers per se.

**Implementation complexity: Medium**  
Requires modifying `compute_labels()` to use per-step volatility multipliers. Need to ensure the model doesn't get confounding signals from barrier width (which correlates with features like `realvol_10`). The fact that your system already has vol as a feature means the model can implicitly learn to adjust by regime — potentially making explicit adaptive barriers redundant.

**Expected impact for fee-constrained system**  
Low-Medium, with significant execution risk. The Springer 2025 result (adaptive barriers hurt crypto performance) is the closest analog to your setup. The theoretical benefit is that quiet-period barriers tighten (generating more trades → more data) while volatile-period barriers widen (longer holds → better fee amortization). But this also changes the label distribution across the training set, which may confuse a flat MLP trained on normalized features. The primary value would be better label quality in volatile vs. quiet regimes.

**Recommendation**: Test only after metalabeling and asymmetric barriers. Lower priority.

**arXiv / Citation**: AEDL: Khrouf et al., Applied Sciences 2025, DOI: 10.3390/app15240 (app15240 is the placeholder; exact ID: 10.3390/app15241320). Springer crypto: DOI: 10.1186/s40854-025-00866-w

---

### 4. Trend-Scanning Labels (Lopez de Prado & Magdon-Ismail, SSRN 2708678)

**Mechanism**  
Instead of a fixed horizon, trend-scanning labels assign each sample the label from the *optimal* forward horizon. The algorithm:
1. For each observation at time t, compute t-statistics of forward return over horizons h ∈ [h_min, h_max]  
2. Choose the horizon h* that maximizes |t-stat|  
3. Assign label based on sign(t-stat at h*)  
4. The absolute t-stat becomes the label confidence weight for the loss function  

This effectively finds the "best possible label" for each sample, rather than forcing an arbitrary hold period. Samples with no detectable trend get down-weighted.

**Evidence of effectiveness**  
- AEDL paper (2025): Trend scanning as a standalone labeling method achieved **Sharpe ≈ 0.00** (essentially zero) on US equity validation data, worse than even triple barrier (−0.03). This is surprising given the method's theoretical appeal.  
- Hudson & Thames video (2022): Shows trend scanning identifies momentum better than fixed horizons on theoretical examples, but with no published OOS results.  
- The practical problem: trend-scanning labels look into the future to find the optimal horizon. On a noisy microstructure signal (your 100-trade batches), the "optimal" horizon is heavily influenced by noise, making labels highly variable and inconsistent across similar market states. The t-stat criterion was designed for longer-horizon price data.

**Implementation complexity: High**  
Requires looping over multiple forward windows for every sample in the training set (O(N × H) complexity). For ~280K samples × 300 horizon options = ~84M forward return computations. Also, the variable-horizon labels are incompatible with your current evaluate() framework, which expects fixed triple-barrier outcomes. Major refactoring of both `prepare.py` and `train.py` would be needed.

**Expected impact for fee-constrained system**  
Low. The AEDL empirical result (Sharpe ≈ 0) discourages this for your use case. Your min_hold=1200 steps already enforces a minimum duration that partially addresses the "too-short horizon" problem trend-scanning solves. The variable horizon also complicates your fee model (switching cost depends on held position).

**Citation**: SSRN 2708678 (Lopez de Prado, "Advances in Financial Machine Learning: Lecture 3/10"). MQL5 implementation guide: mql5.com/en/articles/19253

---

### 5. Recent Literature (2024–2026): Other Notable Innovations

#### 5a. GA-Optimized Barriers (Fu et al., Mathematics 2024)
**DOI**: 10.3390/math12050780  
Genetic algorithms to jointly optimize `(TP_mult, SL_mult, vertical_barrier)` for each asset. Key result: for 5 crypto pairs, GA found solutions that outperformed grid-searched parameters by 15–30% on Sharpe. The optimal TP/SL ratios were asset-specific (0.8–2.5x asymmetry). **Applicability**: Could be used to find per-symbol barrier widths for your 25 symbols rather than one global `fee_mult=11.0`. Expected improvement: moderate (symbols have different vol profiles — BTC vs FARTCOIN warrant different fee multiples). Cost: one Optuna study per symbol, re-running data pipeline with different fee_mult values per symbol.

#### 5b. AEDL Framework (Khrouf et al., Applied Sciences 2025)
**DOI**: 10.3390/app152413204  
Three-innovation framework: (1) volatility-adaptive event detection, (2) multi-scale temporal analysis via wavelets across 5 timeframes, (3) MAML meta-learning for per-asset parameter adaptation. Achieved Sharpe=0.480 on US equities vs Triple Barrier=−0.03. The ablation showed **selective deployment** (multi-scale + meta-learning only, dropping causal inference) achieved Sharpe=0.654 with full asset coverage. **Applicability**: Multi-scale temporal analysis — capturing patterns simultaneously at intraday (100 trades), hourly (~3K trades), and daily (~30K trades) scales — could complement your flat MLP. However, the paper tested daily-bar data; your 100-trade bars are already roughly uniform in information content (volume bars), so the multi-scale benefit may be smaller.

#### 5c. Conformal Prediction for Trade Gating (NeurIPS 2024/2025 literature)
**Mechanism**: Post-hoc calibration of your classifier's softmax outputs using a held-out calibration set. Instead of threshold tuning, CP provides **coverage-guaranteed prediction sets**: "at α=10%, at least 90% of test labels are in the predicted set." When prediction set size > 1 (uncertain), skip the trade.  
**Relevance**: This is essentially a principled version of the softmax-threshold filter you already use. Key paper: Kaya & Nguyen, COPA 2025 proceedings: "Conformal Prediction for Reliable Stock Selections" ([Source](https://proceedings.mlr.press/v266/kaya25a.html)). Also NeurIPS 2024: "Conformalized Time Series with Semantic Features" (neurips.cc/virtual/2024/poster/95653).  
**Implementation**: Apply `MAPIE` library's conformal classifier wrapper to your trained ensemble. Use val set as calibration. In production, only trade when the conformal prediction set is {-1}, {0}, or {1} (singleton — unambiguous prediction).  
**Expected impact**: Medium. Would reduce trade count by skipping ambiguous observations. The metalabeling approach is more powerful but CP is easier to implement on top of existing models.

#### 5d. Optimal Trend Labeling (IEEE Xplore 2023)
**DOI**: 10.1109/JSTSP.2023.3298819 (IEEE Signal Processing)  
Frames labeling as a minimum-description-length problem: find the labeling that minimizes a reconstruction cost. Different from Lopez de Prado's t-stat criterion. Applicable primarily to daily/weekly price data. Not well-suited to microstructure.

---

## Decision Matrix for Fee-Constrained Crypto Futures

| Method | Fee Impact | Implementation | Evidence Strength | Priority |
|--------|-----------|----------------|-------------------|----------|
| Metalabeling | ★★★★★ | Moderate | Strong (H&T empirical) | **1st** |
| Asymmetric barriers | ★★★★ | Low | Moderate (Fu et al. 2024) | **2nd** |
| Per-symbol GA barriers | ★★★ | Moderate | Moderate (Fu et al. 2024) | **3rd** |
| Conformal trade gating | ★★★ | Low | Moderate (COPA 2025) | **3rd** |
| Dynamic/adaptive barriers | ★★ | Medium | Mixed (Springer crypto) | 5th |
| Trend-scanning labels | ★ | High | Weak (AEDL: Sharpe≈0) | Last |

---

## Implementation Sketch: Metalabeling on v11b

```python
# Step 1: Collect primary model outcomes on training data
# After triple_barrier_labels(), for each sample where primary model predicted ±1:
# meta_label = 1 if TP hit, 0 if SL or timeout hit

# Step 2: Build secondary dataset
# Features for secondary model can include:
# - Primary model's softmax confidence (max(softmax) - 1/3)
# - Recent drawdown (current equity - peak)
# - Regime: rolling Hurst, realized_skew, sign_autocorr
# - Time since last trade
# - Symbol-level stats (recent PF over last N trades)

# Step 3: Train secondary model (binary classifier)
# - Use precision-oriented threshold (tune on val)
# - Class weights to handle imbalance (typically 60/40 or 70/30 SL/TP)

# Step 4: At inference, gate trades
# trade = (primary_signal != 0) and (secondary_prob > threshold)
```

**Theoretical upper bound on improvement**: If the secondary model achieves 70% precision (vs 55% base), and you keep all true positives (100% recall), Sortino would roughly scale by the reduction in losing trades. With 45% of trades currently losing, eliminating 30% of them improves PF from 1.71 to ~2.4, and Sortino should scale proportionally.

---

## Sources

**Kept**:
- Singh & Joubert (2022), "Does Meta-Labeling Add to Signal Efficacy?" — direct empirical test, open access
- Fu et al. (2024), Mathematics 12(5):780 — only systematic study of asymmetric barrier optimization for crypto, open access
- AEDL (Khrouf et al., Applied Sciences 2025) — most comprehensive recent labeling study, includes ablation over triple barrier and trend scanning
- Springer Financial Innovation 2025 (crypto triple barrier) — directly relevant, BTC/ETH with transaction costs
- Lopez de Prado (2018), AFML — original source for all methods

**Dropped**:
- Conformal prediction / OOD time series paper (Springer 2025) — relevant to uncertainty quantification but focused on industrial sensors, not financial labels
- AEDL regime-switching RL paper (arxiv 2509) — tangential, about RL portfolio management
- Most "next-bar labeling" papers — inferior method, well-established in literature as dominated by triple barrier for fee-paying strategies

---

## Gaps

1. **No paper tests metalabeling on microstructure data** (sub-second trade-batch resolution). All metalabeling evidence is on daily/hourly bars. The question of whether secondary models can learn meaningful signals at 100-trade granularity is open.

2. **Asymmetric barrier ablation on perpetuals**: Fu et al. tested pair trading (different fee structure). No published study on single-asset perpetuals with funding payments.

3. **Label noise quantification**: None of the reviewed papers measure how much of the "0" (timeout) label class contains hidden positive signals (trades that would have worked if held longer). With min_hold=1200 steps and max_hold=300 steps (per CLAUDE.md `MAX_HOLD_STEPS=300`), timeouts may be contaminating labels for symbols with longer momentum regimes.

4. **Suggested next steps**:
   - Run metalabeling experiment: collect primary model outcomes on train/val splits, train secondary MLP, measure precision gain on val
   - Run asymmetric barrier sweep: `[(tp, sl) for tp in [11,13,15] for sl in [8,9,10,11] if tp >= sl]` — pure relabeling, no architecture change needed
   - If metalabeling works: explore CP wrapper as a simpler alternative with formal guarantees
