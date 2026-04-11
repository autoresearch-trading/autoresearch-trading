# Experiment: 10-seed ensemble

## Hypothesis
Increasing the ensemble from 5 to 10 seeds will reduce prediction variance and improve Sortino. The logit-sum ensemble averages out per-seed noise — more seeds should produce a more stable signal. This is zero-risk: no architecture, feature, or labeling change.

## Eval Definition

**Control:** Sortino=0.353, Passing=9/23, Trades=1269 (5 seeds)

**Success criteria (ALL must pass):**
- [ ] Sortino >= 0.353 (at least match baseline)
- [ ] Passing >= 9/23 (at least match baseline)
- [ ] Ensemble alpha > 0.5 (model validity)

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300
- [ ] Passing < 5/23
- [ ] Ensemble alpha < 0.5

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: 10 seeds vs 5 seeds
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | FINAL_SEEDS=5 (no change) | control |
| 2   | FINAL_SEEDS=10 | test: double ensemble |

## Decision Logic
- If 10 seeds > 5 seeds on score: KEEP
- If 10 seeds ~ 5 seeds (within 0.02 score): INVESTIGATE (marginal, not worth 2x compute)
- If 10 seeds < 5 seeds: DISCARD (shouldn't happen, but would indicate overfitting to seed 0-4)

## Budget
2 runs: ~10 min (control) + ~20 min (10 seeds) = ~30 min total

## Gotchas
- FINAL_BUDGET is 300s total, divided by n_seeds. With 10 seeds, each gets 30s instead of 60s. This might matter — shorter training per seed could hurt.
- The ensemble validity check (alpha > 0.5) averages training accuracy across seeds. More seeds with less training each could push alpha below threshold.
