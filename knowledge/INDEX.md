# Knowledge Base Index

Auto-maintained by the compile-knowledge skill. Do not edit manually.

Last compiled: 2026-04-03

## Concepts

- [Kyle Lambda](concepts/kyle-lambda.md) — price impact per unit signed flow; per-snapshot regime indicator, not per-event
- [Order Event Grouping](concepts/order-event-grouping.md) — same-timestamp trades = one order; dedup required before grouping
- [Effort vs Result](concepts/effort-vs-result.md) — Wyckoff absorption detection; works without direction, robust to 59% ambiguity

## Decisions

- [Drop is_buy](decisions/drop-is-buy.md) — removed from features: 59% ambiguous, half-life 1, distributional discontinuity
- [Per-Snapshot Kyle Lambda](decisions/per-snapshot-kyle-lambda.md) — 50 snapshots (~20 min) not 50 events; fixes 10x variance inflation

## Experiments

- [v11 MLP Baseline](experiments/v11-baseline.md) — Sortino=0.353, walk-forward=0.261; MLP ceiling reached, motivated tape-reading branch
