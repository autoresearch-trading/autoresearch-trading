# Experiment: Dropout

## Hypothesis
Dropout will improve generalization and may push training accuracy above the 0.5 alpha threshold, enabling the 5-seed ensemble to actually form. Currently the model has zero regularization (wd=0), and the ensemble has never been active (alpha consistently < 0.5).

## Eval Definition

**Control:** Sortino=0.353, Passing=9/23, alpha=0.477

**Success criteria (ALL must pass):**
- [ ] Sortino >= 0.353
- [ ] Passing >= 9/23
- [ ] Ensemble alpha > 0.5 (the real goal)

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300
- [ ] Passing < 5/23

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: Dropout rate sweep
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 1 | dropout=0.0 | control |
| 2 | dropout=0.1 | light dropout |
| 3 | dropout=0.2 | moderate dropout |

Dropout added after each ReLU in the MLP trunk (between hidden layers).
NOT applied after the final head layer.

## Decision Logic
- Compare all 3 on score AND alpha
- If any run achieves alpha > 0.5 AND score >= control: strong win
- If score improves but alpha still < 0.5: modest win (better single model)
- Best score wins if multiple pass

## Budget
3 runs × ~10 min = ~30 min

## Gotchas
- Dropout is only active during training (model.train()), automatically disabled during eval (model.eval()). PyTorch handles this.
- The dropout parameter needs to be added to BEST_PARAMS and passed through the model constructor.
