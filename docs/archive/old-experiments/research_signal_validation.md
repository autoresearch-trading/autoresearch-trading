# Signal Validation Research — v11

> Compiled from 3 parallel research agents: Exa deep research (pro), academic paper search (15 papers), and code context search (30+ implementations).

## The Common Thread

Every expert perspective — quant PM, microstructure academic, ML practitioner, crypto specialist, risk manager, statistician — converges on the same diagnosis:

**We are over-engineering signal extraction and under-investing in knowing whether any signal exists.**

The highest-leverage work is not more features or better architecture. It is:

1. Feature ablation (which of 17 features actually contribute?)
2. Statistical rigor (walk-forward validation, confidence intervals, bootstrap)
3. Risk management (position sizing, drawdown limits)
4. Domain-specific signals (funding rate depth, liquidation data)

---

## 1. Feature Ablation / Selection

### Problem
With 17 features and ~64 trades per symbol, we are in a high-dimensional-relative-to-samples regime. Every unnecessary feature adds noise and overfitting risk.

### Recommended Pipeline
**(A) Coarse filtering** (LASSO/L1) --> **(B) Stability checks** (SHAP + resampling) --> **(C) Wrapper selection** inside purged CV

### Key Methods

| Method | Library | Best For |
|--------|---------|----------|
| SHAP FeatureAblation | `captum` (PyTorch-native) | Our flat MLP — ablate entire features across all 50 window positions |
| SHAP-Select | `shap` + regression on SHAP values | Captures nonlinear interactions, robust to noise |
| Boruta | `boruta_py` | All-relevant selection (compares vs shadow features) |
| Permutation Importance | `sklearn` | Quick sanity check, but biased by correlated features |

### Captum Implementation (for our DirectionClassifier)
```python
from captum.attr import FeatureAblation
# Create feature mask grouping all window positions of same feature
# Ablate each of 17 features across all 50 window positions simultaneously
# Run for all 3 target classes (flat/long/short)
# Average across 5 ensemble seeds for stability
```

### Actionable Next Step
Run SHAP/ablation on v11 to identify which of the 8 new features (MLOFI, VWAP dev, spread, Amihud, Roll, arrival rate, r_20) actually contribute. The v10 finding (9 features = 91% of v5 Sortino) suggests high redundancy is likely.

---

## 2. Walk-Forward Validation

### Problem
Our single fixed train/val/test split (100d/25d/36d) gives ONE point estimate. With iterative tuning (Optuna, manual experiments), we are implicitly selecting on the test set.

### CPCV > Walk-Forward
The literature (Arian et al. 2024, SSRN:4686376) shows **Combinatorial Purged Cross-Validation (CPCV)** dominates walk-forward:
- Lower Probability of Backtest Overfitting (PBO)
- Superior Deflated Sharpe Ratio (DSR)
- Multiple backtest paths instead of one

### Key Libraries

| Library | Features | Install |
|---------|----------|---------|
| `skfolio` | CombinatorialPurgedCV, WalkForward, purging, embargo | `pip install skfolio` |
| `ml4t-diagnostic` | CPCV + DSR + PBO + SHAP importance + trade diagnostics | `pip install ml4t-diagnostic` |

### Critical for Our Setup
Triple Barrier labels span up to `MAX_HOLD_STEPS=300`. We MUST:
1. **Purge**: Remove training observations whose labels overlap with test period (purge_size=300)
2. **Embargo**: Buffer after each test period (embargo_size=100)

```python
from skfolio.model_selection import CombinatorialPurgedCV
cv = CombinatorialPurgedCV(
    n_folds=6, n_test_folds=2,
    purged_size=300,   # MAX_HOLD_STEPS
    embargo_size=100,  # safety buffer
)
```

---

## 3. Statistical Significance with Limited Trades

### The Hard Truth
With 64 trades per symbol over 36 days:
- **Binomial test**: Need ~99 trades to detect 60% win rate at 95% confidence (our own T29)
- **Sharpe inference**: With T=64, SE(Sharpe) ~ 0.13, so a Sharpe of 0.3 has 95% CI roughly [-0.5, 1.1]
- **Effective sample size**: With min_hold=800 and serial correlation, effective N is even lower

### Bootstrap Confidence Intervals (Priority #1)

```python
from arch.bootstrap import StationaryBootstrap, optimal_block_length
# StationaryBootstrap handles serial correlation via block resampling
# BCa method best for small samples
# Block length should match average hold period (~100-300 steps)
```

### Multiple Testing Correction (25 symbols)

| Method | Type | When to Use |
|--------|------|-------------|
| Bonferroni | FWER | Zero false positives needed |
| Holm step-down | FWER | Default conservative choice |
| **Benjamini-Hochberg** | **FDR** | **Recommended: evaluating many assets, some false positives OK** |
| Romano-Wolf bootstrap | FWER | Accounts for cross-symbol correlation |

```python
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(symbol_pvalues, method='fdr_bh', alpha=0.10)
```

### Deflated Sharpe Ratio

```python
from deflated_sharpe import deflated_sharpe_ratio, min_backtest_length
dsr, p_value = deflated_sharpe_ratio(
    observed_sr=0.34,    # best Sharpe
    num_trials=250,      # ALL configs tried across ALL experiments
    num_obs=28000,       # steps in test set
    skewness=-0.5,
    kurtosis=5.0,
)
# DSR < 0.5: noise. DSR > 0.8: moderate evidence. DSR > 0.95: strong.
```

---

## 4. Position Sizing & Drawdown Management

### Current Problem
- Always full position (no sizing)
- Binary 20% drawdown halt (all or nothing)
- Max DD of 43% in v11 baseline

### Fractional Kelly (Half-Kelly recommended)
Given small edge and high uncertainty, use f=0.5:
```python
kelly_fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
position_size = 0.5 * kelly_fraction  # half-Kelly
```

### Volatility Targeting
Scale each symbol's position inversely to realized vol:
```python
position_size_i = (target_vol * portfolio_value) / asset_vol_i
```

### Drawdown Ladder (replace binary halt)
| Drawdown | Action |
|----------|--------|
| 5% | Reduce exposure 25% |
| 10% | Reduce exposure 50% |
| 15% | Reduce 75%, freeze new positions |
| 20% | Halt (existing guardrail) |

---

## 5. DEX-Specific Signals We're Missing

| Signal | Source | Priority |
|--------|--------|----------|
| Funding rate regime/term structure | Existing data (under-used) | HIGH |
| Cross-venue funding spread (DEX vs CEX) | Requires CEX feed | HIGH |
| Open Interest regime cycles | Pacifica (if available) | MEDIUM |
| Liquidation cascade clustering | On-chain events | MEDIUM |
| Oracle latency (CEX-DEX price lag) | Requires CEX feed | LOW |
| AMM LP flow (deposits - withdrawals) | On-chain | LOW |

Our `funding_zscore` and `funding_rate_raw` barely scratch the surface. Funding rates are THE dominant signal in perps markets.

---

## 6. Overfitting Diagnostics

### Three-Layer Defense

| Layer | Tool | What It Catches |
|-------|------|-----------------|
| DSR | `deflated-sharpe` | Multiple testing bias from hyperparameter search |
| PBO | `pypbo` | Probability that best in-sample config fails OOS |
| CPCV | `skfolio` / `ml4t-diagnostic` | Single-path dependence, temporal leakage |

### PBO Implementation
```python
import pypbo as pbo
# Build returns matrix: rows=time, columns=strategy variants
# S=8 slices for CSCV
result = pbo.pbo(returns_matrix, S=8, metric_func=sharpe, threshold=0)
# PBO > 0.3 => substantial overfitting risk
```

---

## Priority Implementation Order

| # | Action | Library | Effort | Impact |
|---|--------|---------|--------|--------|
| 1 | Bootstrap CI on Sortino per symbol | `arch` | ~20 lines | Reality check: are 64 trades significant? |
| 2 | Deflated Sharpe Ratio | `deflated-sharpe` | ~10 lines | Gate against noise from experiment search |
| 3 | Feature ablation on v11 | `captum` | ~50 lines | Which of 17 features matter? |
| 4 | BH correction across 25 symbols | `statsmodels` | ~5 lines | Honest "passing" count |
| 5 | Walk-forward with CPCV | `skfolio` | ~100 lines | Eliminate data snooping |
| 6 | Drawdown ladder | Custom | ~20 lines | Graceful risk management |
| 7 | PBO after Optuna | `pypbo` | ~50 lines | Detect hyperparameter overfitting |
| 8 | Deeper funding rate features | Existing data | ~30 lines | Exploit under-used domain signal |

## Key Libraries to Install
```bash
pip install arch deflated-sharpe captum shap skfolio ml4t-diagnostic pypbo statsmodels
```

## Additional Findings from Academic Literature

### Walk-Forward: Deep et al. (2025) is a Wake-Up Call
"Interpretable Hypothesis-Driven Trading" (arXiv:2512.12924) ran 34 independent walk-forward test periods over 10 years on 100 US equities. Result: Sharpe 0.33, **p-value 0.34 (not significant)**. With effect size d=0.17, they'd need **~540 folds for 80% power**. This directly parallels our 64-trade problem.

### GT-Score as Alternative Objective
Sheppert (2026, MDPI JRFM 19(1):60) proposes GT-Score = (mean_return * ln(z_score) * R-squared) / downside_deviation. The "generalization ratio" (OOS/IS performance) is a better overfitting diagnostic than raw Sharpe. Their GT-Score achieves 0.365 generalization ratio vs 0.117-0.188 for standard objectives. **We should compute train/test Sortino ratio as a generalization check.**

### Bias-Corrected Feature Selection (BFSA)
Jukl & Lansky (2026, MDPI) found that standard feature selection embeds directional bias. Their BFSA adds a directional-balance penalty. **We should check BiasDev = (N_long - N_short) / N_total per symbol.** If |BiasDev| > 0.3, the model learned a directional bias, not timing skill.

### Harvey's t > 3.0 Rule
Harvey, Liu & Zhu (2016) argue new factors need t-statistics > 3.0 (not 2.0) given extensive data mining. 82% of 452 anomalies fail under multiple-testing adjustment. **Our effective number of independent tests is ~5-8 (not 25) due to crypto cross-correlation.**

### Unified Validation Pipeline (Njoroge, 2026)
Three-layer defense: CPCV (temporal leakage) + V-in-V (researcher degrees of freedom) + CSCV (PBO audit). **Priority: CPCV is non-negotiable for financial time series. Standard k-fold is structurally wrong.**

---

## Complete Paper Index

| # | Title | Authors/Year | URL |
|---|-------|-------------|-----|
| 1 | A Rigorous Walk-Forward Validation Framework | Deep et al. (2025) | arxiv:2512.12924 |
| 2 | Unified Validation Pipeline Against Backtest Overfitting | Njoroge (2026) | mql5.com/en/articles/21603 |
| 3 | The GT-Score: Robust Objective for Reducing Overfitting | Sheppert (2026) | mdpi.com/1911-8074/19/1/60 |
| 4 | Bias-Corrected Feature Selection for FX Trading | Jukl & Lansky (2026) | mdpi.com/3042-5042/3/1/6 |
| 5 | Backtest Overfitting in the ML Era (CPCV comparison) | Arian et al. (2024) | SSRN:4686376 |
| 6 | Multi-period Portfolio Selection (PhD thesis) | Nystrup (2025) | orbit.dtu.dk |
| 7 | CI Construction with Monte Carlo & Backtesting | Fraszka-Sobczyk (2025) | Springer |
| 8 | Advances in Financial Machine Learning | Lopez de Prado (2018) | Book |
| 9 | The Deflated Sharpe Ratio | Bailey & LdP (2014) | JPM |
| 10 | Probability of Backtest Overfitting | Bailey et al. (2016) | JCF |
| 11 | ...and the Cross-Section of Expected Returns | Harvey et al. (2016) | RFS |
| 12 | Data-Snooping and the Bootstrap | Sullivan et al. (1999) | JF |
| 13 | Empirical Asset Pricing via ML | Gu, Kelly & Xiu (2020) | RFS |
| 14 | The Statistics of Sharpe Ratios | Lo (2002) | FAJ |
| 15 | shap-select: SHAP-based feature selection | arXiv:2410.06815 | arXiv |
