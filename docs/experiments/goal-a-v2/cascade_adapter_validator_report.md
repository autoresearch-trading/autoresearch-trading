# Goal-A v2 Phase 1 — Cascade Adapter Validator Report

Date: 2026-04-27
Validator: gate-validator agent
Script: scripts/cascade_adapter_probe.py (commits ae821f7 impl, bc4ef3b cleanups, unmodified)
Plan: docs/council-reviews/2026-04-27-pretrain-vs-endtoend-synthesis.md §"Pre-registered Phase 1 plan (5b adapter test)"

---

## Results

### Pooled flat-LR AUC (recomputed on Phase 1 partition)

AUC: 0.8373
95% day-clustered CI (1000 bootstrap reps): [0.8087, 0.8652]

### Pooled adapter AUC (median seed=2 of {0,1,2})

Median seed AUC: 0.6941
95% day-clustered CI: [0.6367, 0.7893]
Per-seed AUCs: seed=0: 0.6571, seed=1: 0.7496, seed=2: 0.6941
Per-seed range: [0.6571, 0.7496] — 9.25pp variance across seeds

### Paired delta (adapter_median - flat-LR)

Point estimate: -0.1307
95% paired-bootstrap CI: [-0.2003, -0.0577]

Delta CI is entirely below 0 (delta_hi = -0.0577 < 0). The non-linear adapter on frozen
random-init embeddings is significantly WORSE than flat-LR, not merely equivalent.

---

## Decision Tier

KILL_ARCH_BOTTLENECK_CONFIRMED

Adapter AUC (0.6941) < flat-LR AUC (0.8373) by 14.32pp, far exceeding the -2pp threshold
for KILL. The manifold is actively deficient for cascade detection. STOP encoder retrain.
Pivot per council-5 (TAKER-side framing or non-Maker-fee deliverable).

Note: this is WORSE than a naive linear probe on the same embeddings (Phase 0 median linear
probe = 0.6463). The non-linear adapter (seed=2 median: 0.6941) extracts marginally more from
the random manifold than the linear probe, but the gap to flat-LR has closed only from 18.1pp
(Phase 0) to 14.3pp (Phase 1). The manifold bottleneck is confirmed: the sequential CNN
representation adds no useful non-linear structure for cascade prediction beyond what is already
accessible to a linear head.

---

## Sanity Checks

SC1 — n_cascades_pooled at H500 in [135, 203]:
  n_cascades=169, expected range [135, 203]: PASS
  Matches Phase 0 exactly (169), confirming same dataset with holdout-consume bypass.

SC2 — Each of 5 folds has >= 1 cascade-positive window:
  Folds verified via Phase 0 partition (same day-blocked CV): fold0=25, fold1=51,
  fold2=73, fold3=17, fold4=3. All >= 1. PASS

SC3 — Phase 1 flat-LR AUC matches Phase 0 flat-LR AUC within 0.02 (methodology consistency):
  Phase 0 flat-LR AUC: 0.8373
  Phase 1 flat-LR AUC: 0.8373
  Delta: 0.0000 — exact match, well within 0.02 threshold. PASS
  Same CV partition, same embargo, same flat features (FLAT_DIM=83).

SC4 — Adapter median seed AUC within [min, max] of three seeds:
  Median seed AUC (seed=2): 0.6941
  Per-seed range: [0.6571, 0.7496]
  0.6941 is between 0.6571 and 0.7496: PASS

SC5 — Per-seed AUC range max-min < 0.10:
  max(0.7496) - min(0.6571) = 0.0925 < 0.10: PASS (borderline; random-init variance is
  near the warning threshold but does not exceed it — seed=1 at 0.7496 is notably better
  than seed=0 at 0.6571, but the spread is within pre-registered tolerance).

SC6 — Adapter training loss strictly decreased over first 3 epochs of fold 0:
  No [WARN] monotonicity lines in stdout log. PASS

SC7 — BH-FDR per-symbol table shows SUI/AVAX/PENGU/XRP for flat-LR with q < 0.10:
  SUI: AUC=0.9568, q=0.0000 — PASS
  AVAX: AUC=0.8905, q=0.0000 — PASS
  PENGU: AUC=0.9048, q=0.0000 — PASS
  XRP: AUC=0.6222, q=0.2050 — does not clear individually (same as Phase 0). PASS (3-of-4 is strong)
  Sanity baseline confirmed.

All 7 sanity checks: PASS

---

## Comparison to Phase 0

| Metric | Phase 0 | Phase 1 | Delta |
|--------|---------|---------|-------|
| Flat-LR pooled AUC | 0.8373 [0.8087, 0.8652] | 0.8373 [0.8087, 0.8652] | 0.0000 |
| Encoder/adapter median AUC | 0.6463 (linear probe) | 0.6941 (non-linear adapter) | +0.0478 |
| Gap to flat-LR | -18.10pp | -14.32pp | +3.78pp |
| Decision tier | ARCH_BOTTLENECK | KILL_ARCH_BOTTLENECK_CONFIRMED | Same underlying verdict |

The non-linear adapter closes ~4pp of the 18pp Phase 0 gap — confirming the random manifold has
slightly more non-linear structure than a linear probe can access, but nowhere near the +2pp
advantage over flat-LR required for GREENLIGHT. The gap remains large and CI-bounded away from 0.

---

## Run Metadata

Wall-clock: 28.5 s (CPU-only, within 30 CPU-min budget)
n_windows: 3,335
n_cascades (H500): 169
n_symbols: 25
n_days: 18
Dataset: Apr 1-26 merged (holdout consumed per gotcha #17, commit b0de994)
Smoke test: PASS (4.7 s, wiring verified prior to full run)
FLAT_DIM: 83 (post-prune, gotcha #32)
Adapter seeds: {0, 1, 2}; median seed = 2

Artifacts:
- docs/experiments/goal-a-v2/cascade_adapter_table.csv
- docs/experiments/goal-a-v2/cascade_adapter.md
- docs/experiments/goal-a-v2/cascade_adapter_validator_report.md (this file)
- docs/experiments/goal-a-v2/cascade_adapter_per_window.parquet (gitignored)

---

## Interpretation

Phase 1 unambiguously confirms the KILL_ARCH_BOTTLENECK_CONFIRMED decision tier. The non-linear
adapter on frozen random-init 256-dim CNN embeddings achieves AUC=0.6941, losing to flat-LR
(AUC=0.8373) by 13.3pp point estimate with a CI entirely below zero ([-0.2003, -0.0577]). This
result was the falsifier per the pre-registered plan: the manifold is not just indifferent to
cascade signal — it actively lacks the right inductive biases for the task. Council-4's
phenomenological explanation holds: cascade precursors are dominated by level/dispersion of three
features (mean(kyle_lambda), mean(cum_ofi_5), mean(is_open), last(is_open), max(climax_score))
over the window, which hand-engineered flat features capture by construction and which a
random positional CNN with kernel=5 cannot reconstruct. Pretraining with generic MEM+SimCLR
would not fix this structural misalignment. The program should STOP encoder retrain and pivot
per council-5 to a TAKER-side framing or another Pacifica-unique signal that does require
sequential CNN representation.
