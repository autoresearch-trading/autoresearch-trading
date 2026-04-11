# Knowledge Base Index

Auto-maintained by the compile-knowledge skill. Do not edit manually.

Last compiled: 2026-04-10

## Concepts

- [Book Walk](concepts/book-walk.md) — spread-normalized order aggressiveness (feature 6); unsigned
- [Climax Score](concepts/climax-score.md) — Wyckoff climax via min(z_qty, z_return); rolling 1000-event sigma
- [Contrastive Learning](concepts/contrastive-learning.md) — SimCLR NT-Xent on global embeddings; τ=0.5→0.3, asymmetric augmentation
- [Cumulative OFI](concepts/cum-ofi.md) — piecewise Cont 2014 OFI over 5 snapshots (~120s); naive formula has wrong sign
- [Effort vs Result](concepts/effort-vs-result.md) — Wyckoff absorption detection; works without direction, robust to 59% ambiguity
- [Gate 0 PCA Baseline](concepts/gate0-baseline.md) — PCA + logistic regression reference; StandardScaler required, sweep n
- [Kyle Lambda](concepts/kyle-lambda.md) — price impact per unit signed flow; per-snapshot, four-quadrant interaction map
- [Masked Event Modeling](concepts/mem-pretraining.md) — primary pretraining objective; 20-event blocks, BN-before-masking
- [Order Event Grouping](concepts/order-event-grouping.md) — same-timestamp trades = one order; dedup required before grouping
- [Orderbook Alignment](concepts/orderbook-alignment.md) — 24s snapshot cadence, np.searchsorted alignment, staleness implications
- [Wyckoff Self-Labels](concepts/self-labels.md) — computable market state labels; firing rates, contrastive viability, missing states

## Decisions

- [April Hold-Out Window](decisions/april-holdout-window.md) — April 14+ untouched; March test set contaminated
- [cum_ofi 5 Not 20](decisions/cum-ofi-5-not-20.md) — 5 snapshots (~120s) matches prediction horizon per Cont 2014
- [Drop is_buy](decisions/drop-is-buy.md) — removed: 59% ambiguous, half-life 1, distributional discontinuity
- [Liquid Symbol Sub-Gate](decisions/liquid-symbol-subgate.md) — 10+/15 liquid symbols must pass 51.4% to prevent memecoin gaming
- [MEM Block Size 20](decisions/mem-block-size-20.md) — 20-event blocks not 5; RF=253 makes small gaps trivially solvable
- [Notional Not Raw Qty](decisions/notional-not-raw-qty.md) — depth_ratio, kyle_lambda, cum_ofi use qty×price for cross-symbol comparability
- [NT-Xent Temperature](decisions/ntxent-temperature.md) — τ=0.5→0.3; ImageNet default 0.1 too cold for financial data
- [OB Cadence 24s](decisions/ob-cadence-24s.md) — measured ~24s not ~3s; cascading impact on kyle_lambda, cum_ofi
- [Per-Snapshot Kyle Lambda](decisions/per-snapshot-kyle-lambda.md) — 50 snapshots (~20 min) not 50 events; fixes 10x variance inflation
- [Pivot to Representation Learning](decisions/pivot-to-representation-learning.md) — from supervised Sortino to self-supervised MEM+contrastive

## Experiments

- [v11 MLP Baseline](experiments/v11-baseline.md) — Sortino=0.353, walk-forward=0.261; MLP ceiling reached, motivated pivot
