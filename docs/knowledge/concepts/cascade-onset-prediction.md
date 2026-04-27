---
title: Cascade-Onset Prediction
topics: [labels, microstructure, falsifiability, pacifica-unique]
sources:
  - docs/experiments/goal-a-feasibility/cascade_precursor_real.md
  - docs/experiments/goal-a-feasibility/cascade_precursor_oos.md
  - docs/experiments/goal-a-v2/cascade_adapter_validator_report.md
  - docs/experiments/goal-a-v2-program-end-state.md
last_updated: 2026-04-27
---

# Cascade-Onset Prediction

## What It Is

A binary classification task on Pacifica perp DEX: predict whether at least
one liquidation-cause fill (`cause IN ('market_liquidation',
'backstop_liquidation')`) will occur in `(anchor_ts, ts_at(anchor + H)]` for
horizon H. Uses the Pacifica-unique `cause` flag (only present from
2026-04-01 onward; no public dataset has it).

## Our Implementation

- Label function: `_real_cascade_label_with_event_ts` in
  `scripts/cascade_precursor_probe.py`.
- Primary horizon: H500 (~500 events ≈ 5-30 minutes depending on symbol
  liquidity).
- Base rate: ~6% positives at H500 on the merged Apr 1-26 dataset.
- Feature axes used (council-4 phenomenology): the cascade signature is a
  **liquidity-depletion ramp punctuated by a Composite Operator exit** —
  rolling mean of `kyle_lambda`, `cum_ofi_5`, decline in `is_open`, then
  a `climax_score` spike at the right edge.
- Evaluation: pooled cross-symbol AUC under 5-fold day-blocked CV with
  600-event embargo, day-clustered bootstrap CIs, paired comparison via
  shared day-resamples.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-27 AM | Adopt cascade-onset as Goal-A v2 task | v1 direction prediction fee-blocked; cascade-onset uses Pacifica-unique fields | state.md PM commit `9bb29e3` |
| 2026-04-27 PM | Retire OOS=0.778 as reference baseline | Single-shot OOS biased low; unified-CV AUC=0.8373 is the proper baseline | [Phase 0 result](../experiments/phase0-random-init-probe.md) |
| 2026-04-27 PM-late | Close v2 program | Encoder cannot extract beyond hand features + Maker's Dilemma kills economics | [v2 closure decision](../decisions/v2-program-closure.md) |

## What Works

- **Flat-LR on 83-dim hand features**: pooled AUC = 0.8373 [0.8087, 0.8652]
  on merged Apr 1-26, day-clustered. Per-symbol concentration:
  SUI/AVAX/PENGU/XRP carry the bulk of the signal; BTC/HYPE/ETH at chance.
- **Cross-day generalization**: in-sample AUC=0.815 → OOS AUC=0.778 →
  unified-CV AUC=0.8373 (the unified number is the most honest because it
  avoids the single-realization bias of the OOS run).

## What Doesn't Work

- **Encoder retrain (random-init or non-linear adapter)**: see
  [manifold-deficiency](manifold-deficiency.md). Both Phase 0 and Phase 1
  paired deltas vs flat-LR are firmly below zero (CIs do not cross).
- **Direct trading under maker fees**: top-1% precision OOS = 25.4%; under
  E[realized|filled] = -7.89bp adverse selection, post-fee envelope is
  consumed. Universe median breakeven 0.311; 0/300 cells in (0.50, 0.55).
- **Direct trading under taker fees**: `headroom_table.csv` — 0/300
  survivors at 55% accuracy. Direction-given-cascade is unpredictable
  (cascade direction LR AUC=0.441, marginal-long net-negative).

## Gotchas

1. **`cause` column doesn't exist before April 2026.** Pre-April raw
   parquet must not be loaded into the cascade label pipeline.
2. **April 14-26 holdout was DELIBERATELY consumed on 2026-04-27** (gotcha
   #17). No untouched cascade-labeled OOS remains until ≥30 days of fresh
   data accrue.
3. **Day-clustered bootstrap is mandatory.** Per-window bootstrap
   underestimates uncertainty when cascades cluster intraday. Resample 26
   days with replacement, take all windows of sampled days.
4. **Paired bootstrap on (model_A − model_B) delta** uses ONE RNG seed
   shared across both AUCs per iteration. Two separate bootstraps over the
   same days does NOT have the same statistical power.

## Related Concepts

- [Manifold Deficiency](manifold-deficiency.md)
- [Bootstrap Methodology](bootstrap-methodology.md)
- [V2 Program Closure](../decisions/v2-program-closure.md)
