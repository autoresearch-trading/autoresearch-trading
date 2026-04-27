# Goal-A v2 Phase 0 — Random-Init Probe Validator Report

Date: 2026-04-27
Validator: researcher-10 (gate-validator agent)
Script: scripts/random_init_probe.py (commit 694d14c, unmodified)

---

## Results

### Pooled flat-LR AUC

AUC: 0.8373
95% day-clustered CI (1000 bootstrap reps, 26 days resampled): [0.8087, 0.8652]

Consistent with in-sample result from feasibility chain (AUC=0.815 on Apr 1-13 only). The full Apr 1-26 merge adds the OOS period and yields a slightly higher pooled AUC.

### Pooled encoder AUC (median seed)

Median seed (seed=1) AUC: 0.6463
95% day-clustered CI: [0.5802, 0.7246]
Per-seed range: seed=0: 0.6952, seed=1: 0.6463, seed=2: 0.6330
Min-max spread: [0.6330, 0.6952] — 6.2pp variance across seeds, within expected random-init noise.

### Paired delta (encoder_median - flat-LR)

Point estimate: -0.1812
95% paired-bootstrap CI: [-0.2594, -0.1063]

Delta CI entirely excludes 0 (delta_hi = -0.1063 < 0). The encoder is significantly worse than flat-LR, not merely equivalent.

---

## Decision Tier

ARCH_BOTTLENECK

Encoder AUC (0.6463) < flat-LR AUC (0.8373) by 18.1pp, far exceeding the -2pp threshold. The flat 83-dim hand-engineered features substantially outperform random 256-dim CNN embeddings on cascade prediction at H500. Architecture is the bottleneck for linear extraction of cascade signal; pretraining is required before the encoder can compete.

Per the pre-registered decision tree (plan §Decision Logic, council protocol §Decision tree): decide between MEM-only pretrain or end-to-end fine-tune with strong regularization.

---

## Sanity Checks

SC1 — n_cascades at H500 in expected range (~169 +/- 20%):
  n_cascades=169, expected range [135, 203]: PASS
  Note: 169 matches the state.md reference exactly, confirming the holdout-consume bypass (commit b0de994) took effect and the full Apr 1-26 dataset is present.

SC2 — All 5 folds have >= 1 cascade-positive window:
  fold 0: n_pos=25 PASS
  fold 1: n_pos=51 PASS
  fold 2: n_pos=73 PASS
  fold 3: n_pos=17 PASS
  fold 4: n_pos=3 PASS
  No fold has 0 positives; per-fold AUC is defined for all folds.

SC3 — SUI/AVAX/PENGU/XRP in BH-FDR table, at least one clears q < 0.10 for flat-LR:
  All four present. SUI (q=0.000), AVAX (q=0.000), PENGU (q=0.000) all clear q < 0.10 for flat-LR. PASS
  Note: XRP q=0.205 (does not clear individually), but the 3-of-4 clearance is strong sanity confirmation.

SC4 — Median-encoder-seed AUC is within [min, max] of three seeds:
  0.6463 is between 0.6330 (min) and 0.6952 (max): PASS

SC5 — Paired-bootstrap delta CI width is reasonable vs per-model CI half-widths:
  Observed delta CI half-width: 0.0766
  Expected upper bound (2 x max per-model half-width): 0.1444
  Ratio: 0.53 — delta CI is NARROWER than the bound, consistent with pairing reducing variance. PASS

All 4 required sanity checks (plan Step 4): PASS

---

## Per-Symbol Summary (flat-LR AUC, BH-FDR q)

Symbols with BH-FDR q < 0.10 for flat-LR (cascade-predictable at alpha=0.10):
- AVAX: AUC=0.891, q=0.000
- ENA: AUC=0.680, q=0.000
- PENGU: AUC=0.905, q=0.000
- SUI: AUC=0.957, q=0.000
- ETH: AUC=0.689, q=0.002
- HYPE: AUC=0.573, q=0.051 (marginal)

14 symbols have 0 cascades in the evaluation window (2Z, ASTER, CRV, DOGE, KBONK, KPEPE, LDO, LINK, LTC, UNI, WLFI, XPL and others) — per-symbol AUC is undefined for these; only the pooled AUC is meaningful for the program decision.

---

## Run Metadata

Wall-clock: 27.2 s (CPU-only, within 30 CPU-min budget)
n_windows: 3,335
n_cascades (H500): 169
n_symbols: 25
n_days: 18 (some symbol-days missing from cache; 26 calendar days Apr 1-26)
Dataset: Apr 1-26 merged (holdout consumed per gotcha #17, commit b0de994)
FLAT_DIM: 83 (post-prune, gotcha #32)

Artifacts:
- docs/experiments/goal-a-v2/random_init_probe_table.csv
- docs/experiments/goal-a-v2/random_init_probe.md
- docs/experiments/goal-a-v2/random_init_probe_validator_report.md (this file)
- docs/experiments/goal-a-v2/random_init_probe_per_window.parquet (gitignored)
