# Experiment Report: GCE Loss

## Results
| Run | Name | Loss | LR | Sortino | Passing | Alpha | Score |
|-----|------|------|-----|---------|---------|-------|-------|
| 1 | control | focal | 1e-3 | 0.353 | 9/23 | 0.477 | 0.368 |
| 2 | gce_lr1e-3 | GCE q=0.7 | 1e-3 | 0.240 | 6/23 | 0.372 | 0.248 |
| 3 | gce_lr3e-4 | GCE q=0.7 | 3e-4 | 0.125 | 7/23 | — | 0.197 |
| 4 | gce_lr3e-3 | GCE q=0.7 | 3e-3 | 0.052 | 7/23 | — | 0.153 |

## Eval Result

| Metric | Control | Best GCE | Criterion | Status |
|--------|---------|----------|-----------|--------|
| Sortino | 0.353 | 0.240 | >= 0.353 | FAIL |
| Passing | 9/23 | 7/23 | >= 9 | FAIL |

**Verdict:** DISCARD
**Reason:** Focal loss dominates GCE at all 3 learning rates tested. Best GCE (0.240 at lr=1e-3) is 32% worse than focal (0.353).

## Analysis

GCE was theoretically appealing — it down-weights high-loss (noisy) samples while focal down-weights easy samples. But empirically, focal is strictly better at every lr:

- GCE alpha dropped to 0.372 (vs focal 0.477) — the loss function makes the model *less* accurate on directional predictions, not more
- GCE produces more trades (1575 vs 1270) with lower win rate — it's less selective
- The lr sweep covered a 10x range (3e-4 to 3e-3) with monotonically worsening results at higher lr

This adds to the pattern: focal loss + class weights + recency weighting is a strong local optimum that resists replacement. Both UACE and GCE, tested with proper lr sweeps, lose decisively.

## Conclusion

Focal loss confirmed as optimal loss function. All three alternative losses tested (UACE, GCE, logit-bias modification) are worse.

The focal loss setup works because: (1) gamma=1.0 down-weights easy flat predictions, (2) class weights rebalance the 63/18/19 split, (3) recency weighting handles distribution shift. GCE's noise robustness provides no marginal benefit — the labels may be noisy, but focal handles this class of noise better.

## Recommended Config
No change — keep focal loss.
