---
title: Effort vs Result
topics: [wyckoff, features, tape-reading]
sources:
  - docs/council-reviews/2026-04-02-council-4-tape-viability.md
  - docs/council-reviews/round3-synthesis.md (M7)
last_updated: 2026-04-03
---

# Effort vs Result

## What It Is

Wyckoff's effort vs result principle: compare volume (effort) to price movement
(result). When effort is high but result is small, someone is absorbing flow —
a large patient participant is present. When effort is low but result is large,
the move lacks conviction and may reverse.

## Our Implementation (Feature 7)

```
effort_vs_result = clip(log_total_qty - log(abs(return) + 1e-6), -5, 5)
```

- **High value (e.g. +4):** Large volume, small price move = absorption
- **Low value (e.g. -3):** Small volume, large price move = breakout/low conviction
- Uses median-normalized log_total_qty (feature 2), NOT raw log(qty)
- Epsilon = 1e-6 (not 1e-4 — too coarse for BTC tick-level returns)
- Clipped to [-5, 5] to prevent explosion

## Why It Works Without Direction

The key Wyckoff insight: effort vs result is inherently unsigned. Whether 10,000
BTC of buying or selling hit the book, if price barely moved, absorption occurred.
The absorber's direction is ambiguous even in the original Wyckoff framework —
when price holds on high volume, either buyers absorbed sellers OR sellers absorbed
buyers. Both mean: a large patient participant is present.

This makes effort_vs_result robust to the 59% direction ambiguity in pre-April data.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-02 | Use median-normalized log_total_qty | Raw log(qty) not comparable cross-symbol | round3-synthesis.md M7 |
| 2026-04-02 | Epsilon = 1e-6, not 1e-4 | BTC tick-level returns are tiny; 1e-4 is too coarse | round3-synthesis.md M7 |
| 2026-04-02 | Clip to [-5, 5] | Prevents NaN/inf explosion | CLAUDE.md gotcha 5 |

## Gotchas

1. Must use median-normalized qty, not raw — easy to forget since the formula
   references log_total_qty which looks like raw
2. Epsilon 1e-6 not 1e-4 — BTC-specific precision requirement
3. Without clipping, log(1e-6) = -13.8 which dominates other features

## Related Concepts

- [Order Event Grouping](order-event-grouping.md) — source of total_qty
- [Kyle Lambda](kyle-lambda.md) — different price impact measure, requires direction
- [Climax Score](climax-score.md) — complementary Wyckoff signal (volume + return extremes)
