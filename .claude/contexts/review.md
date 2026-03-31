# Review Context

Mode: Experiment analysis, result interpretation
Focus: Did the experiment answer the question?

## Behavior
- Compare against the control run (baseline in state.md)
- Check all metrics: Sortino, passing symbols, trades, drawdown, win rate, profit factor
- Look for red flags: ensemble alpha < 0.5, fewer passing symbols, higher drawdown
- Update state.md with the result
- Write experiment report to docs/experiments/ if significant

## Analysis Checklist
- [ ] Sortino vs baseline (0.353)
- [ ] Passing symbols vs baseline (9/23)
- [ ] Ensemble alpha > 0.5?
- [ ] Any symbols that flipped (pass->fail or fail->pass)?
- [ ] Trade count reasonable (not degenerate)?
- [ ] Max drawdown acceptable (< 20% per symbol)?

## Decision
- KEEP: improvement on primary metric without regression on guardrails
- DISCARD: regression or no improvement — revert and document why
- INVESTIGATE: ambiguous result — needs more runs or different config
