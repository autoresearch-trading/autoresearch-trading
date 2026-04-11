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

**Control:** [baseline values — PCA or random encoder baseline]

**Success criteria (ALL must pass):**
- [ ] Accuracy at primary horizon >= [threshold] on 15+/25 symbols
- [ ] Beat baseline by >= [margin] pp (representation quality)
- [ ] CKA across seeds > 0.7 (representation stability)
- [ ] No embedding collapse (effective rank > 10)

**Failure indicators (ANY triggers DISCARD):**
- [ ] Accuracy < 50.5% mean across symbols (no signal)
- [ ] Fewer than 10/25 symbols above 51% (not universal)
- [ ] Symbol identity probe > 30% (memorizing symbols, not microstructure)
- [ ] Embedding collapse (effective rank < 5)

**Ambiguity zone (triggers INVESTIGATE):**
- [ ] Accuracy between [baseline, baseline+0.5pp] (noise vs real improvement)
- [ ] Symbol coverage changes by 1-2 symbols (sampling variance)
```

## Evaluation Gates (pre-registered)

These are the project's formal go/no-go gates. All experiments must be evaluated
against the relevant gate:

| Gate | Threshold | What It Tests |
|------|-----------|---------------|
| 0 | PCA + random encoder baselines | Reference (no pass/fail) |
| 1 | Linear probe > 51.4% on 15+/25 symbols | Frozen representation quality |
| 2 | Fine-tuned > logistic regression by >= 0.5pp | Value of pretraining |
| 3 | AVAX (held out) > 51.4% | Universality |
| 4 | Temporal stability < 3pp drop | Robustness |

## Post-Experiment: Grade Result

After the run completes, grade against the eval definition:

1. **Extract metrics** from probing task / evaluation output
2. **Check each success criterion** — all must pass for KEEP
3. **Check failure indicators** — any trigger means DISCARD
4. **If neither** — result is in the ambiguity zone, INVESTIGATE

### Grading output format

```
## Eval Result

**Experiment:** [name]
**Commit:** [hash]
**Gate:** [which gate this evaluates against]

| Metric | Baseline | Result | Criterion | Status |
|--------|----------|--------|-----------|--------|
| Accuracy (mean) | XX.X% | XX.X% | >= 51.4% | PASS/FAIL |
| Symbols > 51% | N/25 | N/25 | >= 15 | PASS/FAIL |
| CKA | N/A | X.XX | > 0.7 | PASS/FAIL |
| Eff. rank | N/A | XX | > 10 | PASS/FAIL |

**Verdict:** KEEP / DISCARD / INVESTIGATE
**Reason:** [one sentence]
```

## Integration with Autoresearch

The autoresearch skill's Design step (section 3) should include an eval definition.
The Conclude step (section 5) should grade against it.

This replaces the informal "did it improve?" check with a structured decision framework
aligned with the pre-registered evaluation gates.
