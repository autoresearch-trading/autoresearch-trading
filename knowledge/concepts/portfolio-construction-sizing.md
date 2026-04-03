---
title: Portfolio Construction & Position Sizing
topics: [portfolio, position-sizing, risk-management, regime-filtering, kelly-criterion]
sources:
  - docs/research/portfolio-construction-sizing.md
last_updated: 2026-04-03
---

# Portfolio Construction & Position Sizing

## What It Is

Portfolio construction and position sizing research addresses how to translate a per-symbol directional classifier into a multi-asset portfolio with controlled risk. The literature identifies a canonical pipeline: direction prediction, then confidence-based sizing, then volatility scaling, then regime filtering, then symbol selection, then cross-asset allocation (HRP/risk parity). The key insight from Lopez de Prado (2018) is that separating the direction decision from the sizing decision improves both precision and risk-adjusted returns, since these have different information requirements.

For crypto specifically, the dominant risk factor is BTC beta -- during drawdowns, pairwise correlations spike toward 1.0, making a portfolio of 23 simultaneous positions effectively a single leveraged BTC bet. Starkiller Capital (Drogen et al. 2022) showed that adding a simple 5/50 EMA BTC trend filter to a cross-sectional crypto momentum strategy improved annualized returns from 37.8% to 93.3% while cutting max drawdown from 75% to 45%.

The three highest-priority interventions are: (1) confidence weighting via softmax calibration or metalabeling to reduce low-quality trades, (2) a BTC regime kill switch to prevent correlated mass drawdowns, and (3) rolling performance-based symbol selection to drop chronically unprofitable symbols.

## Relevance to Our Project

The tape-reading spec currently treats all symbols equally and all predictions as binary (up/down at each horizon). Portfolio-level improvements are downstream of the CNN training pipeline but directly affect whether the model's edge survives transaction costs. The research shows that inverse-volatility scaling (Moreira & Muir 2017) and regime filtering are the most robust methods for multi-asset momentum strategies -- which is what our system effectively is. Hierarchical Risk Parity (Lopez de Prado 2016) is a stage-2 tool for allocating capital across passing symbols but cannot fix per-symbol accuracy. The Kelly criterion at full size is far too aggressive for correlated crypto (0.25x Kelly recommended).

## Key Findings

- Confidence weighting (softmax probability as position size scalar) improves both Sortino and passing count simultaneously -- highest priority
- BTC regime filter prevents correlated 20%+ drawdowns across all symbols; a hard filter (force flat when BTC 30d return < -5%) is simplest
- Rolling Sortino-based symbol selection (AdaptiveTrend, Nguyen 2025) drops chronic losers, raising the fraction of symbols that pass guardrails
- Inverse-volatility position sizing equalizes risk contribution per symbol and bounds drawdown during vol spikes
- Full Kelly criterion is dangerous with correlated crypto assets; quarter-Kelly retains ~75% of optimal log-growth at half the drawdown
- HRP avoids Markowitz instability (no matrix inversion) but only helps allocate across already-passing symbols
- Higher Sharpe thresholds for short signals (1.7 vs 1.3 for longs) reflect asymmetric alpha quality in crypto
- Portfolio heat should be capped at 40-50% of capital deployed, with net directional exposure under 30%
- All sizing improvements should be applied after metalabeling to avoid sizing up on misclassified signals

## Related Concepts

- [kyle-lambda](kyle-lambda.md) -- price impact estimation relevant to position sizing limits
- [cum-ofi](cum-ofi.md) -- order flow imbalance as a potential regime indicator alongside BTC trend
- [climax-score](climax-score.md) -- extreme climax scores across symbols could signal regime shifts warranting position reduction
- [effort-vs-result](effort-vs-result.md) -- trade quality metric usable as a confidence weighting input
