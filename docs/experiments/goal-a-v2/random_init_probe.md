# Goal-A v2 Phase 0 — Random-Init Encoder Linear Probe

Date range: 2026-04-01 → 2026-04-26 (merged Apr 1-26, holdout consumed per gotcha #17).
Symbols: 25 | Days: 18 | Windows: 3,335 | Cascades (H500): 169 | Base rate: 0.0507

## Pooled OOF AUC (5-fold day-blocked CV, 600-event embargo)

| Model | Pooled AUC | 95% CI (day-clustered, 1000 reps) |
|---|---|---|
| Flat-LR (FLAT_DIM=83) | 0.8373 | [0.8087, 0.8652] |
| Random-init encoder LR (median seed=1) | 0.6463 | min-max across seeds: [0.6330, 0.6952] |

### Per-seed encoder pooled AUC

| Seed | Pooled AUC | 95% CI |
|---|---|---|
| 0 | 0.6952 | [0.6604, 0.7493] |
| 1 | 0.6463 | [0.5802, 0.7246] |
| 2 | 0.6330 | [0.5782, 0.7166] |

## Paired delta (encoder_median − flat-LR)

Delta point estimate: **-0.1812** | 95% paired-bootstrap CI: [-0.2594, -0.1063]

## Decision tier (per plan §Decision Logic)

**ARCH_BOTTLENECK** — Encoder < flat-LR by > 2pp.  Architecture is the bottleneck for linear extraction.  Decide between MEM-only pretrain or end-to-end with strong regularization.

## Per-symbol AUC (BH-FDR adjusted, q=0.10 cut)

| Symbol | n_win | n_casc | AUC_flat | q_flat | AUC_enc | q_enc |
|---|---|---|---|---|---|---|
| AAVE | 111 | 11 | 0.4564 | 0.6316 | 0.4882 | 0.9287 |
| AVAX | 109 | 4 | 0.8905 | 0.0000 | 0.2524 | 1.0000 |
| BNB | 103 | 5 | 0.6551 | 0.1861 | 0.6898 | 0.0492 |
| BTC | 218 | 37 | 0.5864 | 0.1526 | 0.5255 | 0.6688 |
| ENA | 130 | 6 | 0.6801 | 0.0000 | 0.2513 | 1.0000 |
| ETH | 195 | 39 | 0.6889 | 0.0022 | 0.4869 | 0.9287 |
| HYPE | 235 | 26 | 0.5732 | 0.0513 | 0.5902 | 0.6688 |
| PENGU | 108 | 3 | 0.9048 | 0.0000 | 0.2095 | 1.0000 |
| SOL | 155 | 21 | 0.6016 | 0.1133 | 0.5736 | 0.5589 |
| SUI | 114 | 4 | 0.9568 | 0.0000 | 0.4091 | 1.0000 |
| XRP | 114 | 9 | 0.6222 | 0.2050 | 0.6000 | 0.6688 |

## Methodology notes

* 5-fold day-blocked CV partition (contiguous, ordered by date).  Each day appears in exactly one fold.
* 600-event embargo at fold boundaries (events, not windows); applied to the LAST events of fold k-1's last day and FIRST events of fold k+1's first day when training fold k.
* Random-init encoder: TapeEncoder(EncoderConfig()) with input BatchNorm1d.track_running_stats=False (gotcha #18).  Eval-mode forward uses batch statistics (256-window batches).
* Encoder seeds: (0, 1, 2).  Median seed binds the report; min-max across seeds bounds random-init variance (council-5 cap).
* Day-clustered bootstrap: 1000 iterations, resample 26 days with replacement.  Paired delta uses the SAME seeded RNG so identical day samples produce both AUCs in lockstep.
* BH-FDR via scipy.stats.false_discovery_control across per-symbol p-values (one-sided H0: AUC ≤ 0.5).

_Pipeline ran in 27.2 s.  CPU-only.  Merged Apr 1-26 dataset; holdout consumed in commit b0de994._
