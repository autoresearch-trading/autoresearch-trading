# Knowledge Base Index

Auto-maintained by the compile-knowledge skill. Do not edit manually.

Last compiled: 2026-04-10

## Concepts

- [Book Walk](concepts/book-walk.md) — spread-normalized order aggressiveness (feature 6); unsigned, renamed from price_impact
- [Climax Score](concepts/climax-score.md) — Wyckoff climax detection via min(z_qty, z_return); rolling 1000-event sigma
- [Cross-Asset Microstructure](concepts/cross-asset-microstructure-features.md) — BTC lead-lag, cross-symbol OFI, queue depletion
- [Cumulative OFI](concepts/cum-ofi.md) — piecewise Cont 2014 OFI over 5 snapshots (~120s); naive formula has wrong sign
- [Effort vs Result](concepts/effort-vs-result.md) — Wyckoff absorption detection; works without direction, robust to 59% ambiguity
- [Kyle Lambda](concepts/kyle-lambda.md) — price impact per unit signed flow; per-snapshot regime indicator, not per-event
- [Labeling Methods](concepts/labeling-methods.md) — metalabeling, asymmetric barriers, conformal prediction gating
- [Order Event Grouping](concepts/order-event-grouping.md) — same-timestamp trades = one order; dedup required before grouping
- [Orderbook Alignment](concepts/orderbook-alignment.md) — 24s snapshot cadence, np.searchsorted alignment, staleness implications
- [Portfolio Construction](concepts/portfolio-construction-sizing.md) — confidence weighting, BTC regime filter, inverse-vol, Kelly criterion

## Decisions

- [April Hold-Out Window](decisions/april-holdout-window.md) — April 14+ untouched; March test set contaminated by 20+ experiments
- [cum_ofi 5 Not 20](decisions/cum-ofi-5-not-20.md) — 5 snapshots (~120s) matches prediction horizon per Cont 2014 principle
- [Drop is_buy](decisions/drop-is-buy.md) — removed from features: 59% ambiguous, half-life 1, distributional discontinuity
- [Notional Not Raw Qty](decisions/notional-not-raw-qty.md) — depth_ratio, kyle_lambda, cum_ofi use qty*price for cross-symbol comparability
- [OB Cadence 24s](decisions/ob-cadence-24s.md) — measured ~24s not ~3s; cascading impact on kyle_lambda, cum_ofi window sizes
- [Per-Snapshot Kyle Lambda](decisions/per-snapshot-kyle-lambda.md) — 50 snapshots (~20 min) not 50 events; fixes 10x variance inflation
- [Pivot to Representation Learning](decisions/pivot-to-representation-learning.md) — from supervised Sortino to self-supervised MEM+contrastive; MLP ceiling reached

## Experiments

- [v11 MLP Baseline](experiments/v11-baseline.md) — Sortino=0.353, walk-forward=0.261; MLP ceiling reached, motivated tape-reading branch
