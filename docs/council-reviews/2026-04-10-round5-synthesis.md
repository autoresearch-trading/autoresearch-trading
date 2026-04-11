# Round 5 Synthesis: Pre-Implementation Stress Test

**Date:** 2026-04-10
**Reviewers:** All 6 council members
**Purpose:** Identify implementation risks, mechanical gaps, and edge cases before writing code

---

## Executive Summary

Round 5 identified **4 blocking issues** that must be resolved before pretraining, **6 high-priority fixes** for the spec, and **8 implementation guidelines** for the data pipeline. The spec's framework is structurally sound — no architectural changes needed — but the mechanical details determine success or failure.

---

## Blocking Issues (Must Fix Before First Training Run)

### 1. MEM Block Size: 5 → 20 Events (Council-6)
With RF=253 covering the entire 200-event window, 5-event masked blocks are trivially reconstructible by local interpolation. The model learns to copy neighbors, not to understand tape structure. **Increase to 20-event blocks, 20% masking rate.**

### 2. Mask Token Replacement Order (Council-6)
BatchNorm must run on the FULL, UNMASKED input. Masked positions are then replaced with zero in BN-normalized space. If BN runs after masking, running statistics are contaminated by artificial zeros, producing systematic miscalibration at inference.

### 3. NT-Xent Temperature τ (Council-6)
Currently unspecified. ImageNet default τ=0.1 is too cold for financial data — pushes apart genuinely similar market states using spurious features. **Set τ=0.5, anneal to 0.3 by epoch 10.**

### 4. cum_ofi_5 Piecewise Formula Validation (Council-2, Council-5)
The naive delta-notional formula gets the sign wrong in 60-80% of BTC snapshot pairs during trending markets. Must validate by checking sign-correlation with subsequent mid-price movement on 5 days of BTC trending data. **2-hour validation, blocks all feature caching.**

---

## High-Priority Fixes (Should Fix Before First Run)

| # | Fix | Source | Impact |
|---|---|---|---|
| 5 | Augmentation noise σ=0.02 → 0.05/0.15 | Council-6 | Views too similar → trivial contrastive |
| 6 | Loss weight annealing (0.90/0.10 → 0.60/0.40) | Council-6 | MEM/contrastive convergence asymmetry |
| 7 | StandardScaler before PCA baseline | Council-1 | Gate 0 reference dominated by time_delta scale |
| 8 | PCA n sweep {20,50,100,200} not fixed 50 | Council-1 | Baseline may be artificially weak |
| 9 | Start trial_log.csv before first experiment | Council-1 | DSR computation requires full trial history |
| 10 | Climax/spring labels → probe only, not contrastive | Council-4 | 0.3-1.5% firing rate → unstable NT-Xent gradients |

---

## Implementation Guidelines (For Data Pipeline)

| # | Guideline | Source |
|---|---|---|
| 11 | Signed flow: net per snapshot period, not per event | Council-3 |
| 12 | imbalance_L5: 1/k harmonic weights, not exponential | Council-2 |
| 13 | delta_imbalance_L1: carry-forward (repeated), not spike-and-zero | Council-2 |
| 14 | kyle_lambda: winsorize at rolling 99th pct per symbol | Council-2 |
| 15 | depth_ratio: validate epsilon 1e-6 on KBONK; clip(-10, 10) | Council-2, Council-5 |
| 16 | Informed flow label: add log_spread < 50th pct condition | Council-3 |
| 17 | Absorption label: require both halves > 1.5 for contrastive | Council-4 |
| 18 | Exclude N_test < 500 symbols from 15/25 gate | Council-1 |

---

## New Spec Additions

### Hyperparameters to Add
- NT-Xent temperature: τ=0.5 → 0.3 (linear anneal, epochs 1-10)
- MEM block size: 20 events (was 5)
- MEM masking rate: 20% (was 15%)
- Augmentation noise: σ=0.05 (trade), 0.15 (OB), 0 (discrete)
- Feature dropout: p=0.10 (was 0.05)
- Gradient clipping: max_norm=1.0
- Loss annealing: mem_weight = max(0.50, 0.90 - epoch × 0.02)

### Monitoring to Add
- Effective rank of embeddings (flag if < 20 at epoch 5)
- Symbol probe every 5 epochs (stop if > 30%)
- Per-feature MEM reconstruction MSE (detect interpolation shortcut)

### Checkpointing
- Save best_mem AND best_probe separately
- Primary for fine-tuning: best_probe
- Early stopping: probe not improved for 10 epochs

### Gate Refinements
- Add liquid-symbol sub-gate: 10+/15 liquid symbols must pass 51.4%
- Exclude N_test < 500 symbols from 15/25 count
- Random encoder: 5 seeds, report mean ± std

---

## Dissenting Opinions / Open Questions

1. **Cross-symbol contrastive pairs:** Council-6 says defer to run 2. Council-4 notes they matter for preventing symbol encoding. Resolution: defer but monitor symbol probe.

2. **"Nothing" class treatment:** Council-4 says let MEM handle it. Council-5 notes this means 60% of contrastive pairs have no supervision. Resolution: add 2-3 labels to reduce unlabeled to ~40%.

3. **BN stat freezing:** Council-5 suggests freezing BN stats after epoch 5. Council-6 did not raise this. Resolution: monitor MEM loss stability; freeze only if loss spikes from stat convergence.

4. **51.4% threshold at T=50:** Council-1 argues realistic threshold is 52.0% given development experiments. Resolution: report both; April 14+ uses T=1.

---

## Pre-Implementation Checklist (Ordered)

1. ☐ Start trial_log.csv
2. ☐ Run 6-point pipeline validation on BTC × 5 days
3. ☐ Validate cum_ofi_5 sign on trending period
4. ☐ Validate kyle_lambda non-zero rate on April data
5. ☐ Validate depth_ratio epsilon on KBONK
6. ☐ Build and validate .npz cache (3-5 hours local)
7. ☐ Run Gate 0 baseline (PCA sweep + random encoder × 5 seeds)
8. ☐ Upload cache to RunPod
9. ☐ Pretrain with corrected spec parameters

---

## Source Reviews

- Council-1: `docs/council-reviews/2026-04-10-round5-council-1-gate0-methodology.md`
- Council-2: `docs/council-reviews/2026-04-10-round5-council-2-ob-implementation.md`
- Council-3: `docs/council-reviews/2026-04-10-round5-council-3-information-regimes.md`
- Council-4: `docs/council-reviews/2026-04-10-round5-council-4-self-labels.md`
- Council-5: `docs/council-reviews/2026-04-10-round5-council-5-impl-risks.md`
- Council-6: `docs/council-reviews/2026-04-10-round5-council-6-pretraining-mechanics.md`
