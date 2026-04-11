---
title: Wyckoff Self-Labels
topics: [wyckoff, contrastive-learning, evaluation, labels]
sources:
  - docs/council-reviews/2026-04-10-round5-council-4-self-labels.md
  - docs/council-reviews/repr-learning-synthesis.md
last_updated: 2026-04-10
---

# Wyckoff Self-Labels

## What They Are

Computable market-state labels derived from the 17 input features using causal
rolling thresholds. No human annotation needed. They serve three purposes:
1. Contrastive pair construction (same-state windows as positives)
2. Probing evaluation (can frozen embeddings predict tape states?)
3. Cluster validation (do embedding clusters correspond to market states?)

## Label Definitions and Expected Firing Rates

| Label | Rule (simplified) | Expected Rate | Contrastive Use |
|-------|-------------------|---------------|-----------------|
| Absorption | mean(effort_vs_result) > 1.5 AND std(log_return) < 0.5σ AND volume elevated, BOTH halves of window | 5-10% | Yes — primary anchor |
| Buying Climax | max(climax_score[-10:]) > 2.5 AND positive spike AND prior uptrend | 0.5-1.5% | No — probe only |
| Selling Climax | max(climax_score[-10:]) > 2.5 AND negative spike AND prior downtrend | 0.5-1.5% | No — probe only |
| Spring | Negative spike + high effort_vs_result + is_open > 0.5 + recovery | 0.3-1.0% | No — probe only |
| Informed Flow | kyle_lambda > rolling_90th pct AND persistent cum_ofi_5 AND log_spread < rolling_50th pct | 5-10% | Yes |
| Stress | log_spread > rolling_90th pct AND abs(depth_ratio) > rolling_90th pct | 3-8% | Yes |

All thresholds are rolling per-symbol (causal, no lookahead).

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-10 | Climax/spring → probe only, not contrastive | 0.3-1.5% firing rate produces 1-4 positive pairs per batch=256 — unstable NT-Xent gradients | round5-council-4 |
| 2026-04-10 | Informed flow threshold tightened to 90th pct | Original 75th pct fires on 10-20% of windows — too common for meaningful contrastive separation | round5-council-4 |
| 2026-04-10 | Informed flow requires low spread | Prevents thin-book illiquidity from firing the label (Council-3: lambda confounds information + illiquidity) | round5-council-3 |
| 2026-04-10 | Absorption requires both halves > 1.5 | Single-half definition creates positional inconsistency — onset vs middle vs end of absorption labeled differently | round5-council-4 |
| 2026-04-10 | Labels are NOT mutually exclusive | Overlapping labels (e.g., absorption + informed flow = "informed absorption") are meaningful compound states | round5-council-4 |

## Missing States (Identified, Not Yet Implemented)

- **Post-Climax Redistribution**: flat log_return, declining is_open after climax spike (HIGH priority)
- **Liquidation Cascade**: climax + stress + partial recovery; validatable with April `cause` field (HIGH priority)
- **Funding Rate Arb Flow**: persistent cum_ofi_5 + low kyle_lambda + elevated is_open (MEDIUM)
- **Pre-Markup Compression**: below-median volume, narrow spread, low climax_score (MEDIUM)

Adding 2-3 labels reduces the unlabeled "nothing" majority from ~60% to ~40-45%.

## The "Nothing" Class

~60% of windows show no distinctive pattern. Strategy: let MEM handle these
(contribute to reconstruction task, not contrastive loss). If symbol identity
probe > 20% after pretraining, apply unsupervised k-means subdivision.

## Gotchas

1. Rolling percentile thresholds must be causal — global statistics are lookahead.
2. At batch=256, expect only 1-4 climax windows per batch — not enough for stable contrastive.
3. The `rolling_std` reference in absorption must be specified: 1000-event rolling σ of log_return.
4. Every spring contains absorption at its low point — labels are nested, not orthogonal.

## Related Concepts

- [Climax Score](climax-score.md) — feature 8, source of climax labels
- [Effort vs Result](effort-vs-result.md) — feature 7, source of absorption labels
- [Kyle Lambda](kyle-lambda.md) — feature 16, source of informed flow labels
