---
title: "v11 MLP — Prior Architecture Reference"
date: 2026-03-25
status: archived
result: partial-success
note: Reference point from main branch. Not a target — tape-reading is a different approach.
sources:
  - docs/experiments/2026-03-25-v11/
  - results.tsv
last_updated: 2026-04-03
---

# Prior Architecture Reference: v11 MLP (main branch)

## Hypothesis

Flat MLP with 13 handcrafted features per 100-trade batch can predict short-term
direction for DEX perpetual futures.

## Setup

- Architecture: flat MLP (not sequential)
- Features: 13 summary statistics per 100-trade batch
- Symbols: 23 (BTC, ETH, SOL, etc.)
- Metric: Sortino ratio (bug-fixed in v11 — historical values were ~1.49x too low)

## Result

- **Sortino: 0.353** (single-window)
- **Walk-forward mean: 0.261** (across 4 folds)
- Profitable on 9/23 symbols
- Every incremental change (sweeps, ablations) made it worse

## What We Learned

1. 100-trade batching destroys tape signals — summary stats lose sequence info
2. The flat MLP ceiling appears to be Sortino ~0.35
3. Feature-return correlations drop 37.6% when trades are shuffled within batches
   — sequence order matters, but the MLP can't use it
4. This motivated the tape-reading branch: sequential model on raw order events

## Sweeps Attempted (All Worse)

- Fee multiplier sweep (2026-03-25)
- Min hold sweep (2026-03-26)
- LR sweep (2026-03-27)
- Batch size sweep (2026-03-28)
- Confidence thresholds (2026-03-28)
- Asymmetric barriers (2026-03-28)

## Verdict

v11 is the best MLP result achievable. Further improvement requires a
fundamentally different architecture (sequential model on raw trade data).
