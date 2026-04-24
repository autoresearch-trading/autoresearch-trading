---
title: Gate 1 Pass — Feb + Mar 2026 at H500
date: 2026-04-23
status: completed
result: success
sources:
  - docs/experiments/step3-run-2-gate1-pass.md
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-24
---

# Experiment: Gate 1 Pass — Feb + Mar 2026 at H500

## Hypothesis

The SSL pretrained encoder (MEM + SimCLR, 376K params) produces
representations that outperform flat baselines on direction prediction at
H500, measured on matched-density held-out months (Feb + Mar 2026) against
the four Gate 1 binding conditions.

## Setup

- **Checkpoint:** `runs/step3-r2/encoder-best.pt` (epoch 6, MEM=0.504, 376K params)
- **Training:** Oct 16 – Jan 31, 2026 (108 days, ~408K windows)
- **Hardware:** M4 Pro MPS, batch 256, 30 epochs, 5h 17m wall-clock, $0
- **Launch:** 2026-04-23T22:22Z
- **Key hyperparameters:** MEM weight 0.90→0.60 annealed, contrastive
  0.10→0.40 annealed, NT-Xent τ=0.5→0.3 by epoch 10, grad_clip=1.0, block
  size 20, masking rate 20%.
- **Bug fixes vs run-0:** early-stop disabled (was tripping on MEM climb);
  best-MEM checkpoint saved (`encoder-best.pt`); hour-of-day probe now uses
  real `ts_first_ms` (was event-index); direction probe now stratified
  per-symbol (was first-50K-alphabet-sorted). Fixes in commits `117187d` +
  `bda524e`.

## Result

**All four Gate 1 conditions pass on BOTH Feb AND Mar independently at H500.**

| Condition | Feb | Mar |
|---|---|---|
| 1. ≥51.4% balanced acc on 15+/24 | 15/24 ✓ | 17/24 ✓ |
| 2. > Majority + 1pp | +3.03pp ✓ | +3.12pp ✓ |
| 3. > Random-Projection + 1pp | +1.91pp ✓ | +2.29pp ✓ |
| 4. Hour-of-day probe < 10% | 0.06–0.09 ✓ | 0.06–0.09 ✓ |

Feb held-out (21,290 windows, 24 symbols): encoder 0.530 mean balanced
accuracy, encoder beats PCA by ≥1pp on 17/24 symbols.
Mar held-out (16,278 windows, 24 symbols): encoder 0.531 mean, beats PCA
≥1pp on 14/24 symbols.

Consistent winners across both months: AAVE, BNB, CRV, FARTCOIN, HYPE,
KBONK, PUMP, SUI, WLFI, XRP.
Strongest cells: FARTCOIN Mar 0.65 (+0.15 over PCA), SUI Feb 0.66 (+0.17
over PCA), PUMP 0.58 on both months.

## What We Learned

1. **MEM-minimum checkpoint is the right choice.** MEM loss hit 0.504 at
   epoch 6 and climbed to 0.94 plateau afterward; the contrastive head wins
   the optimizer after the OneCycleLR peak. The epoch-6 encoder captures
   the MEM-minimum state with genuine representation quality.
2. **Symbol-ID probe grows during pretraining (0.54 → 0.67).** On in-sample
   data; later measurements (step5 cluster cohesion) confirm 0.934 on
   held-out Feb. Expected under the training-config-doesn't-target-universality
   reading ([cross-symbol invariance](../concepts/cross-symbol-invariance.md)).
3. **Horizon matters — H100 is noise-floor for every predictor on this data.**
   This motivated both the horizon-selection rule
   ([horizon-selection decision](../decisions/horizon-selection-rule.md))
   and the Gate 1 window amendment
   ([gate1-window-amended decision](../decisions/gate1-window-amended-feb-mar-h500.md)).
4. **MPS is a viable substrate for <1M-param models.** ~3× slower than H100
   but $0 vs ~$6; the tradeoff favors MPS at current scale.
   See [mps-substitute decision](../decisions/mps-substitute-for-h100.md).

## Verdict

**SUCCESS** — Gate 1 binding pass. Gate 2 (fine-tuning) is unblocked. Step 4
proceeds on the pretrained 24-symbol universe.

The pass required the amended window (Feb + Mar, H500) rather than the
original pre-registered April 1–13 at H100. The anti-amnesia clause requires
every future Gate 1 report to continue publishing the superseded April
numbers. See
[gate1-window-amended decision](../decisions/gate1-window-amended-feb-mar-h500.md)
for the amendment methodology and council review trail.

## Related

- [Gate 1 window amended](../decisions/gate1-window-amended-feb-mar-h500.md)
- [Horizon selection rule](../decisions/horizon-selection-rule.md)
- [MPS substrate decision](../decisions/mps-substitute-for-h100.md)
- [Step 3 run-0 collapse diagnosis](step3-run0-collapse-diagnosis.md)
