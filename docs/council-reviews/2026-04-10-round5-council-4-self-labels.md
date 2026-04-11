# Council-4 Round 5: Self-Label Calibration Review

**Reviewer:** Council-4 (Richard Wyckoff — Tape Reading)
**Date:** 2026-04-10
**Round:** 5 — Pre-implementation stress test

## Summary

The self-label system is structurally sound but has two execution-level problems: climax and spring labels are too rare for contrastive pair construction, and absorption labels need temporal anchoring.

## 1. Label Distribution — Expected Firing Rates

| Label | Estimated Rate | Contrastive Viability |
|-------|---------------|----------------------|
| Absorption | 5-10% | Good |
| Buying Climax | 0.5-1.5% | Too rare — probe only |
| Selling Climax | 0.5-1.5% | Too rare — probe only |
| Spring | 0.3-1.0% | Too rare — probe only |
| Informed Flow | 10-20% | Too common — tighten threshold |
| Stress | 3-8% | Good |

**Critical finding:** At batch=256, expect only 1-4 climax windows per batch. NT-Xent with 1-4 positive pairs produces unstable gradients. Use climax/spring as probe labels only.

**Remedy for informed flow:** Tighten `kyle_lambda > rolling_90th_pct` (not 75th) to reduce firing rate to ~5-10%.

## 2. Mutual Exclusivity

Labels are NOT mutually exclusive by design — and this is a feature:
- **Absorption + Informed Flow** = "informed absorption" (Composite Operator accumulating)
- **Stress + Climax** = liquidation cascade (distinct from orderly climax)
- **Absorption ⊃ Spring** (every spring contains absorption at the low)

Recommendation: treat overlapping labels as compound states for probing evaluation.

## 3. Temporal Coherence

Absorption definition uses `mean(effort_vs_result[-100:]) > 1.5` — only the LAST 100 events. This creates positional inconsistency:
- Window A (onset): absorption building → label may not fire
- Window B (middle): full absorption → label fires
- Window C (end): absorption resolving → label may not fire

**Fix:** Add "sustained absorption" variant: BOTH halves must satisfy `effort_vs_result > 1.5`. Use this for contrastive pairs.

## 4. Missing States

| Missing State | Feature Signature | Priority |
|---|---|---|
| Post-Climax Redistribution | Flat log_return, declining is_open, widening time_delta after climax_score spike | HIGH |
| Pre-Markup Compression | Below-median volume, narrow spread, low climax_score | MEDIUM |
| Liquidation Cascade | climax + stress + partial recovery (cause=market_liquidation in April+) | HIGH |
| Funding Rate Arb Flow | Persistent cum_ofi_5 + low kyle_lambda + elevated is_open | MEDIUM |

Adding 2-3 labels reduces the unlabeled majority from ~60% to ~40-45%.

## 5. The "Nothing" Class

Recommendation: **Approach 1 for initial build** — unlabeled windows contribute only to MEM, not contrastive. If symbol identity probe shows >20% after pretraining, apply unsupervised k-means subdivision (Approach 3).

## 6. DEX-Specific Patterns

- **Backstop intervention** (cause=backstop_liquidation): cascade termination signal, reversal probability increases
- **Funding-driven OI expansion**: is_open elevated + price stable = position building before resolution
- **Mark vs Oracle divergence**: inter-market positioning signal (April+ prices table, future work)
- **Funding settlement flow**: periodic mechanical flow that is NOT information-driven (no funding-time feature currently)

## Priority Actions

1. Use climax/spring as probe labels only, NOT for contrastive pairs
2. Tighten informed flow to 90th pct kyle_lambda
3. Add "sustained absorption" variant (both halves > 1.5)
4. Add liquidation cascade and post-climax redistribution labels
5. Leave "nothing" class unlabeled in contrastive; handle via MEM
