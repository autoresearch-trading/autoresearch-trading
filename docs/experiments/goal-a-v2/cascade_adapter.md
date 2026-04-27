# Goal-A v2 Phase 1 — Cascade adapter probe vs flat-LR

## Protocol

Date range: 2026-04-01 → 2026-04-26 (merged Apr 1-26, holdout consumed per gotcha #17).  Same 5-fold day-blocked CV with 600-event embargo as Phase 0.

Adapter: `Linear(256→64) + ReLU + Dropout(0.2) + Linear(64→1)` trained on FROZEN random-init `TapeEncoder` 256-dim global embeddings.  BCEWithLogitsLoss(pos_weight=15.7), AdamW(lr=0.001, wd=0.001), no LR schedule, max 50 epochs, batch 256, early-stop on val-AUC patience=5.

Symbols: 25 | Days: 18 | Windows: 3,335 | Cascades (H500): 169 | Base rate: 0.0507

## Pooled OOF AUC

| Model | Pooled AUC | 95% CI (day-clustered, 1000 reps) |
|---|---|---|
| Flat-LR (FLAT_DIM=83) | 0.8373 | [0.8087, 0.8652] |
| Adapter on random-init enc (median seed=2) | 0.6941 | min-max across 3 seeds: [0.6571, 0.7496] |

### Per-seed adapter pooled AUC

| Seed | Pooled AUC | 95% CI |
|---|---|---|
| 0 | 0.6571 | [0.5914, 0.7773] |
| 1 | 0.7496 | [0.7122, 0.8010] |
| 2 | 0.6941 | [0.6367, 0.7893] |

## Paired delta (adapter_median − flat-LR)

Delta point estimate: **-0.1307** | 95% paired-bootstrap CI: [-0.2003, -0.0577]

## Decision tier (council-6 pre-registered)

**KILL_ARCH_BOTTLENECK_CONFIRMED** — adapter < flat-LR by > 2pp.  Manifold is actively deficient for cascade detection.  STOP encoder retrain; pivot per council-5 (TAKER-side framing or non-Maker-fee deliverable).

## Per-symbol AUC (BH-FDR adjusted)

| Symbol | n_win | n_casc | AUC_flat | q_flat | AUC_adapter | q_adapter |
|---|---|---|---|---|---|---|
| AAVE | 111 | 11 | 0.4564 | 0.6316 | 0.5355 | 0.5011 |
| AVAX | 109 | 4 | 0.8905 | 0.0000 | 0.4214 | 0.7924 |
| BNB | 103 | 5 | 0.6551 | 0.1861 | 0.6490 | 0.4785 |
| BTC | 218 | 37 | 0.5864 | 0.1526 | 0.5344 | 0.4785 |
| ENA | 130 | 6 | 0.6801 | 0.0000 | 0.1841 | 1.0000 |
| ETH | 195 | 39 | 0.6889 | 0.0022 | 0.5069 | 0.5304 |
| HYPE | 235 | 26 | 0.5732 | 0.0513 | 0.5320 | 0.4981 |
| PENGU | 108 | 3 | 0.9048 | 0.0000 | 0.5619 | 0.1836 |
| SOL | 155 | 21 | 0.6016 | 0.1133 | 0.6365 | 0.0332 |
| SUI | 114 | 4 | 0.9568 | 0.0000 | 0.6273 | 0.0668 |
| XRP | 114 | 9 | 0.6222 | 0.2050 | 0.5651 | 0.4785 |

## Methodology notes

* Encoder is frozen random-init `TapeEncoder(EncoderConfig())` with input BatchNorm1d.track_running_stats=False (gotcha #18).  Embeddings are extracted under `torch.no_grad()`; no gradients flow into the encoder.
* 3 encoder seeds.  Median seed binds the report; min-max bounds random-init variance.
* Adapter: ~16K params; weight init He/Kaiming for fc1, Xavier for fc2, zero biases.  Per-fold StandardScaler fit on training rows only.
* Day-clustered bootstrap: 1000 iterations.  Paired delta uses the SAME seeded RNG so identical day samples produce both AUCs in lockstep.

_Pipeline ran in 28.5 s.  CPU-only.  Smoke mode: False._
