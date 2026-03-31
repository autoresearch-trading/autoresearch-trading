---
name: experiment-eval
description: >
  Define pass/fail criteria before running experiments and grade results after.
  Use when designing experiments in the autoresearch loop, or when asked to
  "define eval", "set success criteria", "grade this result", "did it work".
---

# Experiment Eval Protocol

Define what success looks like BEFORE running an experiment. Grade the result AFTER.

## When to Use

- Before any autoresearch experiment (integrated into the Design step)
- When the user asks "what would success look like?"
- After an experiment completes, to decide KEEP/DISCARD/INVESTIGATE

## Pre-Experiment: Define Eval

Before running, write an eval block in the experiment plan (docs/experiments/):

```
## Eval Definition

**Hypothesis:** [what we expect to happen and why]

**Control:** [baseline values from state.md]
- Sortino: 0.353
- Passing: 9/23
- Trades: 1269

**Success criteria (ALL must pass):**
- [ ] Sortino >= [threshold] (primary metric)
- [ ] Passing >= [threshold] (coverage)
- [ ] Ensemble alpha > 0.5 (model validity)
- [ ] No guardrail violations (DD < 20% per symbol)

**Failure indicators (ANY triggers DISCARD):**
- [ ] Sortino < 0.300 (clear regression)
- [ ] Passing < 5/23 (coverage collapse)
- [ ] Ensemble alpha < 0.5 (training instability)

**Ambiguity zone (triggers INVESTIGATE):**
- [ ] Sortino between [control-0.05, control] (noise vs real regression)
- [ ] Passing changes by 1-2 symbols (sampling variance)
```

## Post-Experiment: Grade Result

After the run completes, grade against the eval definition:

1. **Extract metrics** from PORTFOLIO SUMMARY output
2. **Check each success criterion** — all must pass for KEEP
3. **Check failure indicators** — any trigger means DISCARD
4. **If neither** — result is in the ambiguity zone, INVESTIGATE

### Grading output format

```
## Eval Result

**Experiment:** [name]
**Commit:** [hash]

| Metric | Control | Result | Criterion | Status |
|--------|---------|--------|-----------|--------|
| Sortino | 0.353 | X.XXX | >= 0.353 | PASS/FAIL |
| Passing | 9/23 | N/23 | >= 9 | PASS/FAIL |
| Alpha | >0.5 | X.XXX | > 0.5 | PASS/FAIL |
| Max DD | <20% | X.X% | < 20% | PASS/FAIL |

**Verdict:** KEEP / DISCARD / INVESTIGATE
**Reason:** [one sentence]
```

## Integration with Autoresearch

The autoresearch skill's Design step (section 3) should include an eval definition.
The Conclude step (section 5) should grade against it.

This replaces the informal "did it go up?" check with a structured decision framework.
