# Review Context

Mode: Experiment analysis, result interpretation
Focus: Did the experiment answer the question?

## Behavior
- Compare against the relevant baseline (PCA, random encoder, or logistic regression)
- Check representation quality metrics: accuracy, symbol coverage, CKA, effective rank
- Look for red flags: embedding collapse, symbol identity leakage, temporal instability
- Update state.md with the result
- Write experiment report to docs/experiments/ if significant

## Analysis Checklist
- [ ] Accuracy vs baseline (PCA / random encoder / logistic regression)
- [ ] Symbol coverage: how many symbols > 51.4%?
- [ ] CKA across seeds > 0.7? (representation stability)
- [ ] Effective rank > 10? (no embedding collapse)
- [ ] Symbol identity probe < 20%? (learning microstructure, not symbol identity)
- [ ] Temporal stability: < 3pp drop across time periods?
- [ ] Any symbols that flipped (pass->fail or fail->pass)?

## Decision
- KEEP: improvement on representation quality without gate violations
- DISCARD: regression or gate failure — revert and document why
- INVESTIGATE: ambiguous result — needs more runs or different config
