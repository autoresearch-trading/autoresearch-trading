---
name: experiment-eval
description: Define and grade experiments for active Pacifica full-fidelity paper-trading research: post-cost PnL, Sortino, drawdown, trade count, concentration, and causal robustness.
---

# Experiment Eval Protocol — active full-fidelity branch

Define success before running experiments. Grade results after.

## Required eval dimensions

Every trading or overlay experiment must specify:

- data window and number of distinct days,
- dynamic universe snapshot source or explicit symbols,
- causal feature/threshold construction,
- fees, slippage/spread, funding, and adverse-selection assumptions,
- net PnL,
- Sortino,
- max drawdown,
- number of trades and active days,
- symbol/day/event concentration,
- baseline comparison,
- robustness split by day and symbol,
- what would falsify the idea.

## Diagnostic sample rule

If the experiment uses only 1-2 local full-fidelity days, it is diagnostic only. Do not claim edge and do not tune thresholds from it.

## Output

Before run: write explicit success criteria.
After run: classify as KEEP / DISCARD / INVESTIGATE / INSUFFICIENT_SAMPLE with evidence.
