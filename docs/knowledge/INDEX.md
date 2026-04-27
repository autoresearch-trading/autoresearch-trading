# Knowledge Base Index

Auto-maintained by the compile-knowledge skill. Do not edit manually.

Last compiled: 2026-04-27 (program freeze)

## Concepts

- [Book Walk](concepts/book-walk.md) — spread-normalized order aggressiveness (feature 6); unsigned
- [Bootstrap Methodology](concepts/bootstrap-methodology.md) — 1000-resample CI + N=50 shuffled null + class-prior; required on small-n probes
- [Climax Score](concepts/climax-score.md) — Wyckoff climax via min(z_qty, z_return); rolling 1000-event sigma; max 0.256 ≪ pre-reg threshold 3.0
- [Contrastive Learning](concepts/contrastive-learning.md) — SimCLR NT-Xent; τ=0.5→0.3; measured cross-symbol delta +0.037 on step3-r2
- [Cross-Symbol Invariance](concepts/cross-symbol-invariance.md) — SimCLR cluster geometry; +0.037 delta vs +0.1 threshold; 4× weaker than symbol-ID
- [Cumulative OFI](concepts/cum-ofi.md) — piecewise Cont 2014 OFI over 5 snapshots (~120s); naive formula has wrong sign
- [Effort vs Result](concepts/effort-vs-result.md) — Wyckoff absorption detection; rolling-median normalized return term
- [Gate 0 Baseline Grid](concepts/gate0-baseline.md) — 4-baseline publishing grid (PCA, RP, Majority, Shuffled); NOT a threshold-gate
- [Kyle Lambda](concepts/kyle-lambda.md) — price impact per unit signed flow; real trade-attributed in cache layer, OB-proxy in feature module
- [Masked Event Modeling](concepts/mem-pretraining.md) — primary pretraining objective; 20-event blocks; mask-first-then-encode (identity-task bug fixed)
- [Order Event Grouping](concepts/order-event-grouping.md) — same-timestamp trades = one order; dedup required before grouping
- [Orderbook Alignment](concepts/orderbook-alignment.md) — 24s snapshot cadence, np.searchsorted alignment, staleness implications
- [Session-of-Day Leakage](concepts/session-of-day-leakage.md) — `_last` leak pruned 2026-04-23 (FLAT_DIM 85→83); encoder mitigation via timing-noise σ=0.10
- [Underpowered Single-Symbol Probe](concepts/underpowered-single-symbol-probe.md) — stride=50 1-mo 1-sym CI ~0.09-0.12 > encoder's ~1-2pp signal; 1/20 surrogate rate = chance
- [Wyckoff Self-Labels](concepts/self-labels.md) — computable market state labels; firing rates, contrastive viability, missing states

## Decisions

- [Abort Criterion Taxonomy (Class A/B)](decisions/abort-criterion-taxonomy.md) — 1 Class A bug = STOP-and-redesign; 3 Class B before process review (binding 2026-04-26)
- [Amendment Budget Clause](decisions/amendment-budget-clause.md) — 3rd binding-gate amendment without new experiment requires out-of-band review
- [April Hold-Out Window](decisions/april-holdout-window.md) — April 14+ untouched; March test set contaminated
- [Balanced Accuracy at ALL Horizons](decisions/balanced-accuracy-all-horizons.md) — removes H10/H50/H100 raw-accuracy carve-out; raw is gameable
- [Calibrated Interpretation — Per-Symbol-Clustered](decisions/calibrated-interpretation-per-symbol-clustered.md) — adopts +1pp + per-symbol-clustering claim grounded in cohesion + RankMe + symbol-ID; council-1 QA approved
- [cum_ofi 5 Not 20](decisions/cum-ofi-5-not-20.md) — 5 snapshots (~120s) matches prediction horizon per Cont 2014
- [Drop is_buy](decisions/drop-is-buy.md) — removed: 59% ambiguous, half-life 1, distributional discontinuity
- [Gate 1 Thresholds Revised (SUPERSEDED)](decisions/gate1-thresholds-revised.md) — original 4-condition structure; superseded by Feb+Mar H500 window
- [Gate 1 Window Amended — Feb+Mar H500](decisions/gate1-window-amended-feb-mar-h500.md) — April→Feb+Mar, H100→H500; PASSES with +1.9-2.3pp over flat
- [Gate 3 Retired to Informational](decisions/gate3-retired-to-informational.md) — three evidence lines retire binding pass/fail; re-activation requires n≥2000 + delta ≥+0.10
- [Gate 4 Rewrite for Coherence](decisions/gate4-rewrite-for-coherence.md) — old "months 5-6" nonexistent; rewritten as Oct-Nov vs Dec-Jan on Feb+Mar fold
- [Horizon Selection Rule](decisions/horizon-selection-rule.md) — primary horizon = shortest where PCA+LR ≥ 0.505; blocks encoder-side horizon shopping
- [Liquid Symbol Sub-Gate](decisions/liquid-symbol-subgate.md) — 10+/15 liquid symbols must pass 51.4% to prevent memecoin gaming
- [Matched-Density Definition](decisions/matched-density-definition.md) — held-out windows-per-sym-per-day within 0.7-1.3× training, stride=200
- [MEM Block Size 20](decisions/mem-block-size-20.md) — 20-event blocks not 5; RF=253 makes small gaps trivially solvable
- [MPS Substitute for H100](decisions/mps-substitute-for-h100.md) — Apple Silicon MPS at batch 256 = $0 vs H100 $6; 3× ratio doesn't justify cloud spend for <5M params
- [Notional Not Raw Qty](decisions/notional-not-raw-qty.md) — depth_ratio, kyle_lambda, cum_ofi use qty×price for cross-symbol comparability
- [NT-Xent Temperature](decisions/ntxent-temperature.md) — τ=0.5→0.3; ImageNet default 0.1 too cold for financial data
- [OB Cadence 24s](decisions/ob-cadence-24s.md) — measured ~24s not ~3s; cascading impact on kyle_lambda, cum_ofi
- [OB Level Zero-Fill](decisions/ob-level-zero-fill.md) — `np.zeros`, not `np.full(nan)`; missing levels = zero liquidity, not unknown
- [Path A Program Closure](decisions/path-a-program-closure.md) — close + publish on 2026-04-27; Goal-A abandoned for this stack; reusable pipeline + methodology + features carry over
- [Path D — Drop Battery](decisions/path-d-drop-battery.md) — multi-probe battery dropped without measurement (climax threshold 30× empirical max); anti-amnesia disclosure pattern
- [Per-Snapshot Kyle Lambda](decisions/per-snapshot-kyle-lambda.md) — 50 snapshots (~20 min) not 50 events; fixes 10x variance inflation
- [Pivot to Representation Learning](decisions/pivot-to-representation-learning.md) — from supervised Sortino to self-supervised MEM+contrastive
- [SimCLR Augmentations Strengthened](decisions/simclr-augmentations-strengthened.md) — jitter ±10→±25; new σ=0.10 timing-feature noise for session decorrelation
- [Symbol-ID Probe Reframed Aspirational](decisions/symbol-id-probe-reframed-aspirational.md) — <20% target goal for future universality-targeting runs; other diagnostics remain binding

## Experiments

- [Cache NaN Contamination](experiments/cache-nan-contamination.md) — 117 shards with NaN traced to `np.full(nan)` in OB level expansion; fixed → 0 critical
- [Cluster Cohesion Diagnostic](experiments/cluster-cohesion-diagnostic.md) — 6 anchors, cross-sym same-hour delta +0.037, symbol-ID 0.934: unearned universality
- [Gate 0 4-Baseline Grid](experiments/gate0-4baseline-grid.md) — PCA ≈ Majority ≈ RP on balanced acc; pipeline clean; raw-vs-balanced inflated H10 up to 9.9pp
- [Gate 1 Pass — Feb+Mar H500](experiments/gate1-pass-feb-mar-h500.md) — +3.03/+3.12pp vs Majority; 15/24, 17/24 clear 51.4%; hour probe clean
- [Gate 2 Fine-Tuning FAIL](experiments/gate2-finetune-fail.md) — fine-tuned CNN -1.7pp vs flat-LR; 3 abort-criterion bugs en route; per-symbol geometry breaks fine-tuning
- [Gate 3 AVAX Triage](experiments/gate3-avax-triage.md) — bootstrap CIs overlap 4/4 cells; in-sample LINK+LTC fails identically: EXONERATED
- [Gate 4 Temporal Stability PASS](experiments/gate4-temporal-stability-pass.md) — Oct-Nov vs Dec-Jan probes both eval'd on Feb+Mar; <3pp drop on 19/24; +0.6pp mean drift
- [Multi-Probe Battery Path D](experiments/multi-probe-battery-path-d.md) — climax label fires 0% on held-out (max 0.256 ≪ threshold 3.0); dropped without measurement
- [Per-Symbol Surrogate Sweep](experiments/per-symbol-surrogate-sweep.md) — 5 in-sample symbols, 1/20 CI separations = chance rate, validates reframe
- [Step 3 Run-0 Collapse Diagnosis](experiments/step3-run0-collapse-diagnosis.md) — three probe bugs + MEM identity-task bug invalidated run-0 signal
- [Tape-State Diagnostic Off-Ramp](experiments/tape-state-diagnostic-off-ramp.md) — c-4 designed, c-5 rejected pre-commit; ~80% confirm-or-abort odds; soft adjudication wins
- [v11 MLP Baseline](experiments/v11-baseline.md) — Sortino=0.353, walk-forward=0.261; MLP ceiling reached, motivated pivot
- [v11 Prior Architecture Reference](experiments/v11-prior-architecture-reference.md) — main-branch v11 architecture summary for context
