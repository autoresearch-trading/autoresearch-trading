---
title: Climax Score
topics: [wyckoff, features, tape-reading]
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Climax Score

## What It Is

A Wyckoff climax is a moment of extreme volume AND extreme price movement
occurring simultaneously -- the exhaustion point of a trend where the dominant
side capitulates. Selling climaxes mark potential bottoms; buying climaxes mark
potential tops. The conjunction (AND, not OR) is critical: extreme volume alone
could be normal institutional activity, and extreme price movement alone could be
a thin-book gap. Both together signal capitulation.

The climax score quantifies how climactic an order event is as a continuous
intensity measure, rather than a binary yes/no flag.

## Our Implementation (Feature 8)

```
climax_score = clip(min(qty_zscore, return_zscore), 0, 5)
```

- **qty_zscore:** `(log_total_qty - rolling_mean) / rolling_std` over 1000 events
- **return_zscore:** `(abs(log_return) - rolling_mean) / rolling_std` over 1000 events
- **min()** enforces the AND gate: both must be extreme for a high score
- **clip(0, 5)** maps normal events to 0 (either z-score negative = min is negative = clipped to 0), only nonzero when both are simultaneously elevated
- Rolling 1000-event window is causal (no lookahead)

The continuous formulation (Council-4 recommendation, adopted over original binary
flag) preserves intensity and clustering information. During a genuine selling
climax, multiple consecutive events will have high scores -- the model sees the
intensity ramp, not just a single spike.

## Key Decisions

| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|
| 2026-04-01 | min() not product/geometric mean | Product allows compensation: huge volume + tiny return gives moderate score. min() requires BOTH extreme. | council-4, round3-synthesis |
| 2026-04-01 | Continuous score not binary flag | Binary loses intensity and clustering information; continuous provides gradient signal | council-4 |
| 2026-04-01 | Rolling 1000-event sigma, not global | Global sigma is lookahead. Rolling prevents quiet-session sigma from creating false positives at session transitions. | council-5 |
| 2026-04-02 | Guard rolling_std > 1e-10 | std=0 possible at day open or illiquid symbols; produces NaN z-scores | data-eng-13 |

## Gotchas

1. Rolling std can be zero at day open or for illiquid symbols -- guard with
   `np.where(rolling_std > 1e-10, (x - mean) / std, 0.0)`.
2. The 1000-event window is count-based, not time-based. During quiet sessions
   1000 events may span 30-60 minutes; during crashes, 10 minutes. This means
   the threshold for "extreme" adapts to recent regime but with variable time
   horizon. Council-4 flagged this as a source of false positives at session
   transitions (quiet -> active).
3. Pre-warm rolling buffers from prior day to avoid masking day-open events
   (committed decision M6 in round3-synthesis).
4. First calendar day per symbol: mask first 1000 events from training.

## Related Concepts

- [Effort vs Result](effort-vs-result.md) -- complementary Wyckoff signal (absorption detection)
- [Order Event Grouping](order-event-grouping.md) -- source of total_qty and log_return
