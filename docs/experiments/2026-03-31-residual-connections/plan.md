# Experiment: Residual Connections

## Hypothesis
Adding skip connections (`x = x + layer(x)`) to the MLP trunk will improve gradient flow and training accuracy without adding parameters. Gorishniy et al. (NeurIPS 2021) show ResNet-MLP consistently beats plain MLP on tabular benchmarks. This may push alpha above 0.5 and enable the ensemble.

## Eval Definition

**Control:** Sortino=0.353, Passing=9/23, alpha=0.477

**Success criteria (ALL must pass):**
- [ ] Sortino >= 0.353
- [ ] Passing >= 9/23

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300
- [ ] Passing < 5/23

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: Residual vs plain MLP
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 1 | plain MLP (current) | control |
| 2 | residual MLP (`x = x + block(x)`) | test |

## Implementation
In DirectionClassifier, change the trunk from a flat Sequential to a residual structure.
Since hidden_dim is the same across all layers (64→64→64), skip connections are dimensionally valid.
The first layer (676→64) cannot have a skip connection (dimension mismatch). Only layers 2-3 get residuals.

## Budget
2 runs × ~10 min = ~20 min (control already done)
