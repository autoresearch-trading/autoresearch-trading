# Experiment: GCE Loss

## Hypothesis
GCE loss (q=0.7) down-weights high-loss (noisy) samples, complementary to focal which down-weights easy samples. Triple barrier labels are inherently noisy (noise from fee estimation, spread variation, microstructure randomness). GCE's noise robustness may improve the model. Fair comparison requires lr re-sweep (CLAUDE.md gotcha #9).

## Eval Definition

**Control:** Sortino=0.353, Passing=9/23 (focal loss, lr=1e-3)

**Success criteria (ALL must pass):**
- [ ] Sortino >= 0.353
- [ ] Passing >= 9/23

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300 at best lr
- [ ] Passing < 5/23 at best lr

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: GCE lr sweep
Since switching loss functions requires lr re-tuning, sweep lr for GCE:
| Run | Config delta | Purpose |
|-----|-------------|---------|
| 1 | focal, lr=1e-3 | control (reuse baseline) |
| 2 | GCE q=0.7, lr=1e-3 | same lr as focal |
| 3 | GCE q=0.7, lr=3e-4 | lower lr |
| 4 | GCE q=0.7, lr=3e-3 | higher lr |

## Decision Logic
- Best GCE score across lr sweep vs control
- If best GCE > control: KEEP GCE at that lr
- If best GCE < control: DISCARD

## Budget
4 runs × ~10 min = ~40 min (control reused)

## Implementation
GCE loss: L_q = (1 - p_y^q) / q where p_y = softmax(logits)[true_class]
Apply class_weights and sample_w same as focal.
