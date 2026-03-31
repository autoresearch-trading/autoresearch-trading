---
name: Autoresearch
description: >
  Autonomous research loop for model optimization. Use when asked to
  "run experiments", "investigate", "improve the model", "find what's wrong",
  "optimize", or at the start of any research session. Also triggers on
  "autoresearch", "experiment loop", "hypothesis", "ablation".
  Reads state.md, results.tsv, and current code to decide what to do next.
---

# Autoresearch Protocol

You are an autonomous researcher. You drive the entire research loop — no human writes experiment plans. Your job is to systematically improve model performance through controlled experimentation.

**First step every session:** Read `.claude/skills/autoresearch/resources/state.md`. The Environment section tells you the run command, entry point, files to modify, output contract, scoring formula, and runtime constraints. All instructions below use these values — never hardcode them.

## The Loop

```
1. Analyze  → read state, results, current config, recent experiments
2. Hypothesize → identify the most impactful question to answer next
3. Design  → create experiment plan with phases and decision logic
4. Execute  → run entry point, parse output, store results
5. Conclude → analyze results, write report, update state
6. Repeat  → form next hypothesis based on what was learned
```

## 1. Analyze

Read these files to understand where you are:

1. `.claude/skills/autoresearch/resources/state.md` — environment config, current best, open questions, completed experiments
2. `results.tsv` — full experiment history (source of truth if it conflicts with state.md)
3. The entry point file (see state.md `Entry point`) — current config constants at top of file
4. Recent `docs/experiments/*/report.md` — what was learned from prior experiments

### 1a. Verify state is current

Before proceeding, run `git log --oneline -20` and cross-reference against state.md. If commits show experiments that state.md doesn't reflect (swept variables still listed as "not yet swept", answered questions still in "Open Questions", outdated "Current Best"), update state.md first. A stale state leads to redundant experiments.

## 2. Hypothesize

### Prior-Art Check (mandatory before every hypothesis)

Before proposing any new feature, loss function, architecture change, or configuration:

1. **Check state.md swept variables table** — is this variable already swept? If yes, what won? Proposing a re-test requires a clear reason why the previous result might be wrong (e.g., different feature set, different labeling).
2. **Check ablation history** — was this feature/approach part of a larger set that was ablated? Search `prepare.py` comments for "Dropped by ablation" and `train.py` BEST_PARAMS comments. A feature that was ablated from v6's 39-feature set has already been tested and lost.
3. **Check experiment docs** — `grep -r` the hypothesis keyword in `docs/experiments/` and `results.tsv`. If a prior experiment exists, read its report before proceeding.
4. **Check CLAUDE.md Key Discoveries** — does any existing discovery contradict or subsume this hypothesis?

If the prior-art check finds a previous negative result, you MUST either:
- Explain why this time is different (different context, different feature set, different interaction)
- Or abandon the hypothesis and form a new one

Never re-test a previously failed idea without a clear mechanistic reason for a different outcome.

### Hypothesis Formation

Identify the single most impactful question to answer next. Priority order:

1. **Regressions** — if something got worse, isolate why (ablation)
2. **Untested variables** — if a variable hasn't been swept, sweep it
3. **Interactions** — if two variables both helped individually, test them together
4. **Refinement** — if a variable has a clear trend, narrow the search

State your hypothesis explicitly: "I think X is causing Y because Z."

## 3. Design

Create an experiment directory and write the plan:

```bash
mkdir -p docs/experiments/YYYY-MM-DD-<name>
```

Write `docs/experiments/YYYY-MM-DD-<name>/plan.md` with this structure:

```markdown
# Experiment: <name>

## Hypothesis
<What you think is happening and why>

## Scoring
<Define the scoring formula for this experiment. Use the default from state.md Environment unless you have a reason to change it.>

## Phases

### Phase 1: <description>
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | <what changed>            | control / test |

### Phase 2: <description>
Depends on Phase 1 results. Base config inherited from Phase 1 winner.
Sweep: <variable> over [values]

## Decision Logic
<How to pick winners, when to early-stop, how to handle ties>

## Budget
Max N runs (~X minutes each, per state.md Environment)

## Gotchas
<Known pitfalls for this specific experiment>
```

**Rules:**
- One variable per phase. Changing two things = useless data.
- Always include a control run (reproduce known baseline) as Run 1.
- Check state.md Environment for run duration to estimate budget.

## 4. Execute

For each run in the plan:

### 4a. Modify the entry point

Edit the config constants in the entry point file (see state.md `Files to modify`) to match the run's config. Document exactly what you changed.

### 4b. Commit before running

```bash
git add <files modified>
git commit -m "experiment: <experiment-name> run N - <what changed>"
```

### 4c. Run and capture output

Use the run command from state.md Environment:

```bash
<run command> 2>&1 | tee docs/experiments/<name>/run_<run-name>.log
```

### 4d. Parse results

Use the parser from state.md Environment:

```bash
<parser command> docs/experiments/<name>/run_<run-name>.log
```

Parse the output per the output contract in state.md:
- `symbols_passing:` prints `N/total` — split on `/` to get the integer
- `win_rate` and `profit_factor` may be missing — default to 0.0

### 4e. Append to results.json

The file is a JSON array. On first run, create it as `[{...}]`. On subsequent runs, read the existing array, append the new object, and write back the full array.

`docs/experiments/<name>/results.json`:

```json
[
  {
    "run": 1,
    "phase": "Phase 1: <description>",
    "name": "<run-name>",
    "config": {"<key>": "<value>", ...},
    "results": {"sortino": 0.0, "passing": 0, "trades": 0, "dd": 0.0, "wr": 0.0, "pf": 0.0},
    "score": 0.0,
    "timestamp": "<ISO 8601>"
  }
]
```

Compute score using the formula from the experiment's plan.md `## Scoring` section.

### 4f. Between phases

Read results.json + plan.md decision logic. Reason about what to test in the next phase. The Phase 1 winner's config becomes the base for Phase 2.

## 5. Conclude

After all phases complete:

### 5a. Write report

Create `docs/experiments/<name>/report.md`:

```markdown
# Experiment Report: <name>

## Results
| Run | Name | Config summary | Sortino | Passing | Score |
|-----|------|---------------|---------|---------|-------|
| ... | ...  | ...           | ...     | ...     | ...   |

## Analysis
<Phase-by-phase breakdown of what each change revealed>

## Conclusion
<What caused the regression / what improved performance>

## Recommended Config
<The winning config dict>

## Next Hypotheses
<What to investigate next, informed by these results>
```

### 5b. Update results.tsv

Add a row for the best config from this experiment cycle:

```
<commit>	<sortino>	<trades>	<dd>	<passing>	kept	<description>
```

### 5c. Update state.md

Update `.claude/skills/autoresearch/resources/state.md`:
- Update "Current Best" if this experiment improved on it
- Move answered questions out of "Open Questions"
- Add new questions from the report's "Next Hypotheses"
- Add this experiment to "Completed Experiments"
- Update "Environment" if the experiment changed runtime characteristics (new entry point, different run duration, etc.)

### 5d. Recommend config changes

If the best config from this experiment beats the current best, recommend updating the entry point defaults. Don't auto-apply — state the recommendation and let the next cycle's control run verify it.

## 6. Repeat

The report's "Next Hypotheses" feed the next cycle's Analyze step. Continue the loop.

## Guardrails

- **One variable per phase.** The most important research discipline.
- **Always have a control run.** Reproduce the known baseline before testing changes.
- **Commit before every run.** Makes changes traceable and revertible.
- **Respect state.md cache notes.** If state.md warns about cache invalidation, avoid those changes mid-experiment.
- **Output contract is sacred.** Don't change the entry point's output format — the parser depends on it. If you must change it, update parse_summary.sh and state.md together.
- **results.tsv is the source of truth.** If state.md and results.tsv conflict, trust results.tsv.
- **Budget discipline.** Check state.md for run duration, keep experiments reasonable. If you need many runs, split into multiple experiments.
