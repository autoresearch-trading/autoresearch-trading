---
title: Labeling Methods for Financial ML
topics: [labeling, metalabeling, triple-barrier, position-sizing, training]
sources:
  - docs/research/labeling-methods-research.md
last_updated: 2026-04-03
---

# Labeling Methods for Financial ML

## What It Is

Labeling methods determine how forward returns are converted into training targets for financial classifiers. The standard approach -- triple-barrier labeling (Lopez de Prado 2018) -- assigns each sample a label based on whether price first hits a take-profit barrier, stop-loss barrier, or a time expiry. The research evaluates five innovations beyond vanilla triple-barrier: metalabeling, asymmetric barriers, dynamic/adaptive barriers, trend-scanning labels, and conformal prediction gating.

Metalabeling is the highest-impact method. It introduces a two-stage architecture: the primary model predicts direction (long/short/flat), and a secondary model predicts whether to act on that signal (trade/skip). The secondary model is trained on outcomes of the primary's predictions -- label 1 if the primary was correct, 0 if wrong. Singh & Joubert (2022) showed accuracy improvements from 17-20% to 63-77% OOS on S&P500 e-mini futures. The key prerequisite is that the primary model must have positive expectancy, which ours does (WR=55%, PF=1.71).

Asymmetric barriers -- setting take-profit and stop-loss at different multiples -- are the lowest-risk structural change. Fu et al. (2024) found GA-optimized asymmetric solutions consistently outperformed symmetric ones across crypto pairs, with optimal TP/SL ratios ranging from 0.8x to 2.5x.

## Relevance to Our Project

The tape-reading model uses binary direction labels at 4 forward horizons (10/50/100/500 events). While triple-barrier labeling is not directly used in the current spec, the metalabeling concept translates directly: a secondary model (or confidence threshold) could filter the primary CNN's predictions, reducing false positives that erode returns after fees. The research on dynamic barriers found they actually hurt crypto performance (Springer 2025), validating the spec's choice of fixed label horizons. Trend-scanning produced Sharpe of approximately 0 in the AEDL study -- not viable for our setting.

## Key Findings

- Metalabeling is the highest-priority labeling intervention for fee-constrained systems -- it prunes losing trades rather than relabeling data
- Asymmetric barriers (wider TP than SL) bias the label set toward high-conviction setups; sweep `tp_mult x sl_mult` grid as a low-cost experiment
- Dynamic/volatility-adaptive barriers deteriorated performance on crypto specifically (Springer 2025) -- the model handles vol better as a feature than as a label modifier
- Trend-scanning labels achieved Sharpe of approximately 0 in the most comprehensive study (AEDL 2025) -- not recommended
- Conformal prediction gating is a principled alternative to softmax thresholding but less powerful than full metalabeling
- No paper has tested metalabeling on microstructure data (sub-second resolution) -- this is an open research question
- Per-symbol GA-optimized barriers could replace a single global fee_mult, since BTC and FARTCOIN have very different volatility profiles

## Related Concepts

- [effort-vs-result](effort-vs-result.md) -- captures trade quality signal that a meta-model could use as a secondary feature
- [climax-score](climax-score.md) -- high climax scores may correlate with metalabel=1 (correct primary predictions during unusual activity)
- [order-event-grouping](order-event-grouping.md) -- label quality depends on correct event dedup before computing forward returns
