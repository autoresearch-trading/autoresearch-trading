# Council-6 (DL Architect) — Gate 0 Impact on Pretraining Design

**Date:** 2026-04-15
**Reviewer:** council-6

## 1. Does the finding confirm or challenge the pretraining thesis?

**Confirms the flat-feature dead end; leaves the CNN hypothesis unproven.** PCA vs RP gap at H100 = 0.58pp — within symbol-level noise. PCA's marginal wins are concentrated on low-liquidity names (2Z, CRV, WLFI, FARTCOIN, XPL) at short horizons, consistent with autocorrelated thin-book flow, not Wyckoff structure.

Aggregating a 200-event window to 85 summary statistics destroys sequential signal the CNN is designed to exploit. Flat aggregation being weak is consistent with the thesis but does NOT rule out a null: the CNN could equally learn nothing if microstructure signal is non-linear in ways 1 H100-day budget cannot capture. Honest framing: flat aggregation is demonstrated weak; CNN hypothesis remains unproven.

## 2. Session-of-day leak in raw sequence

Real and underweighted. Three channels:

1. **`prev_seq_time_span`** — wall-clock span of prior window correlates with UTC session via inter-event gap rate.
2. **`time_delta`** — inter-event spacing drops sharply at UTC 0/8/16 session opens (BTC UTC-16 open: gaps of 0.3s; overnight: 3–5s).
3. **`log_spread`** — spreads systematically widen at session boundaries.

**MEM does not protect against this:** session patterns repeat day-to-day, so MEM reconstruction improves for free when the encoder learns session identity.

**SimCLR window jitter of ±10 events is insufficient:** at BTC's ~20K events/day, 10 events = 0.05% of a day — far too small to decorrelate session identity.

## 3. Required diagnostics BEFORE the first pretraining run

1. **Hour-of-day linear probe on frozen embeddings.** 24-class LR on 256-dim embedding predicting UTC hour of window center. Target: <10% accuracy (above 1/24 = 4.2% chance = session info; below 10% = clean). Structurally analogous to existing symbol-identity probe.

2. **Hour-of-day stratified accuracy at Gate 1.** Report linear-probe accuracy separately per UTC session (Asia 0–8, Europe 8–16, US 16–24). If >2pp variance across sessions, representation is session-conditional. Acceptable: <1.5pp.

3. **Cluster coloring by hour-of-day** in existing cluster analysis. Make explicit, not optional.

## 4. Pretraining recipe adjustments

Two targeted changes, no model-size or objective-structure impact:

### (a) Increase SimCLR window jitter: ±10 → ±25 events

At BTC's median 1.5s inter-event gap, ±25 events = ±37 seconds (crosses minor session micro-boundaries). At illiquid alts (56–68 min per 200-event window), ±25 events shifts window center by ~10 minutes — meaningful. Strictly semantics-preserving.

### (b) Do NOT exclude `prev_seq_time_span` from MEM

Its role is local event-rate rhythm (stress regimes = rapid bursts; thin markets = long gaps) — a genuine microstructure signal. Instead, **add `time_delta` and `prev_seq_time_span` to the augmentation target list** for noise injection: jitter σ=0.10 (5× baseline). Forces the encoder to rely on relative rhythms, not absolute session-indicative magnitudes.

### Not recommended: hour-of-day adversarial head

Adds gradient conflict in an already-tight compute budget. Adversarial heads are known to destabilize training when the primary task is weak. Probe-at-evaluation is strictly preferable.

## 5. Gate 1 threshold revision

51.4% is now confirmed as noise floor, not a meaningful margin. Council-1 round-5 already flagged that at T=50 experiments, Holm–Bonferroni puts the realistic threshold at 52.0%.

**Revised Gate 1, both conditions required:**

1. CNN linear probe > 51.4% on 15+/25 symbols (existing).
2. CNN linear probe beats RP-control on the same April 1–13 fold by **≥1.0pp mean** across all symbols passing (1). (Strengthened from 0.5pp.)

Rationale: RP H100 mean = 50.33%. CNN at 51.5% beats RP by 1.17pp — real signal. CNN at 51.0% beats RP by 0.67pp on a dataset where symbol noise is easily ±1pp. **The ≥1.0pp margin against RP is the actual discriminating condition**; the absolute 51.4% is retained as sanity.

## Summary

PCA≈RP confirms flat aggregation is weak and supports proceeding with pretraining, but reveals the 51.4% threshold is not meaningfully above noise. Revisions required:

1. Gate 1: CNN must beat RP-control by ≥1.0pp, not 0.5pp above PCA.
2. Mandatory hour-of-day probe on frozen embeddings (<10% accuracy).
3. Hour-of-day stratified accuracy at Gate 1 (<1.5pp cross-session variance).
4. SimCLR jitter ±10 → ±25 events.
5. Add σ=0.10 augmentation noise to `time_delta` and `prev_seq_time_span`.
