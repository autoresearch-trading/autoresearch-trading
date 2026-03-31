# Experiment: Alpha Threshold

## Hypothesis
The 0.5 alpha threshold is too strict for our 3-class imbalanced problem. The ensemble has NEVER formed in recent runs (alpha consistently ~0.48). Disabling the threshold (always use the full 5-seed ensemble) may improve results because logit averaging reduces variance even when per-model directional accuracy is below 50%.

## Eval Definition

**Control:** Sortino=0.353, Passing=9/23 (single best model, alpha=0.477 triggered fallback)

**Success criteria (ALL must pass):**
- [ ] Sortino >= 0.353
- [ ] Passing >= 9/23

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300
- [ ] Passing < 5/23

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: Force ensemble vs single best
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 1 | alpha threshold=0.5 (default) | control (single model fallback) |
| 2 | alpha threshold disabled (always ensemble) | test: does ensemble help? |

## Decision Logic
- If ensemble > single: the threshold was hurting us — lower or remove it
- If ensemble < single: Theorem 10 is correct — the threshold protects us
- If ensemble ~ single (within 0.02 score): threshold is irrelevant at this alpha level

## Budget
2 runs × ~10 min = ~20 min (control already done, reuse result)
