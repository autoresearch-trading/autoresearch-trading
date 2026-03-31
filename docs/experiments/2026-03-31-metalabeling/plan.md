# Experiment: Metalabeling

## Hypothesis
A secondary binary classifier that gates the primary model's directional predictions will improve Sortino by filtering false positives (losing trades). The primary model fires both winners and losers — a meta-model can learn patterns that distinguish the two.

## Eval Definition

**Control:** Sortino=0.353, Passing=9/23, Trades=1269, WR=55%, PF=1.71

**Success criteria (ALL must pass):**
- [ ] Sortino >= 0.353
- [ ] Win rate > 55% (the whole point is precision improvement)

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300
- [ ] Trades < 200 (too aggressive filtering)

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Implementation

### Architecture
1. Train primary DirectionClassifier as-is (5 seeds, pick best)
2. Run primary model over training data to get predictions
3. For each directional prediction (long/short, not flat):
   - Label = 1 if the trade was profitable (hit TP)
   - Label = 0 if the trade lost (hit SL or timeout with loss)
4. Train a small binary MLP (meta-model) on: primary model's softmax probs (3) + original features (window×13 flattened stats) → binary (trade/skip)
5. At eval: primary predicts direction, meta-model gates execution

### Meta-model features
- Primary model's 3-class softmax probabilities (confidence signal)
- Max softmax probability (how confident is the primary?)
- Temporal mean and std of features (same 26 stats the primary uses)
Total: 3 + 1 + 26 = 30 features for the meta-model

### Meta-model architecture
Small MLP: 30 → 32 → 1 (sigmoid). Binary cross-entropy loss.
Much smaller than primary (30 input features vs 676).

## Phases

### Phase 1: Metalabeling vs baseline
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 1 | no metalabeling | control |
| 2 | metalabeling with threshold=0.5 | test |

### Phase 2: Threshold sweep (if Phase 1 shows promise)
Sweep meta-model threshold: 0.3, 0.5, 0.7

## Budget
2-4 runs × ~12 min = ~30-50 min

## Gotchas
- Meta-model must be trained on DIFFERENT data than primary model's training set to avoid leaking. Use validation set predictions.
- Class imbalance in meta-labels: ~55% win, ~45% lose. Manageable.
- Risk of overfitting meta-model to primary model's quirks on val set. Keep meta-model small.
