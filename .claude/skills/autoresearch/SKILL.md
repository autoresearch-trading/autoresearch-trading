---
name: Autoresearch
description: >
  Autonomous research loop for model optimization. Use when asked to
  "run experiments", "investigate", "improve the model", "find what's wrong",
  "optimize", or at the start of any research session. Also triggers on
  "autoresearch", "experiment loop", "hypothesis", "ablation".
  Reads .autoresearch/state.md, .autoresearch/results.tsv, and current code to decide what to do next.
---

# Autoresearch Protocol

You are an autonomous researcher. You drive the entire research loop — no human writes experiment plans. Your job is to systematically improve model performance through controlled experimentation.

**First step every session:** Run the bootstrap check below, then read `.autoresearch/state.md` (in the project root). The Environment section tells you the run command, entry point, files to modify, output contract, scoring formula, and runtime constraints. All instructions below use these values — never hardcode them.

## 0. Bootstrap (run once per project)

Before entering the loop, check whether this project has the required infrastructure. If any of the following are missing, create them before proceeding.

### Required files

| File | Purpose | Action if missing |
|------|---------|-------------------|
| `.autoresearch/state.md` | Environment config, current best, open questions | Create from template below — ask the user to fill in project-specific values |
| `.autoresearch/parse_summary.sh` | Output parser | Create a parser that extracts metrics from the entry point's stdout — tailor to the output contract in state.md |
| `.autoresearch/results.tsv` | Experiment history (source of truth) | Create with header: `commit\t<metric1>\t<metric2>\t...\tstatus\tdescription` — columns should match the metrics defined in state.md's output contract |
| `.autoresearch/experiments/` | Experiment plans, logs, reports | Create the directory |

### state.md template

If `state.md` does not exist, create it with this structure and ask the user to provide the values marked with `<...>`:

```markdown
# Research State

## Environment
- Run command: `<command to run a single experiment, e.g. "python train.py">`
- Entry point: `<main script, e.g. "train.py">`
- Files to modify per experiment: `<files where config constants live>`
- Output contract: `<describe what the entry point prints — the metrics and their format>`
- Parser: `bash .autoresearch/parse_summary.sh <logfile>` (relative to project root)
- Approximate run duration: `<e.g. "~10 min">`
- Primary metric: `<e.g. "accuracy", "loss", "Sortino ratio">`
- Default scoring: `<formula, e.g. "score = accuracy * 0.7 + (passing / total) * 0.3">`
- Cache note: `<any expensive preprocessing steps to avoid invalidating mid-experiment, or "None">`

## Current Best
- Config: <none yet — will be populated after first experiment>
- Score: <none>

## Swept Variables
| Variable | Values tested | Winner |
|----------|--------------|--------|
| (none yet) | | |

## Open Questions
1. <What is the first thing worth investigating?>

## Completed Experiments
(none yet)

## Key Findings
(none yet)
```

### parse_summary.sh template

If the parser does not exist, create one that extracts the primary metric from stdout. Example skeleton:

```bash
#!/usr/bin/env bash
# Usage: bash parse_summary.sh <logfile>
# Extracts metrics from the entry point's output.
# Adapt the grep/awk patterns to match your output contract.

FILE="$1"
if [ -z "$FILE" ]; then echo "Usage: parse_summary.sh <logfile>"; exit 1; fi

echo "=== Parsed Results ==="
grep -i "primary_metric:" "$FILE" | tail -1
# Add more grep lines for each metric in your output contract
```

Tailor this to the actual output contract once state.md is filled in.

### Bootstrap flow

1. Check for each file above
2. If all exist → proceed to the loop
3. If any are missing → create them, ask the user for required values, then proceed
4. If state.md exists but is clearly from a different project (wrong entry point, wrong metrics) → ask the user if they want to reset it for this project

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

1. `.autoresearch/state.md` — environment config, current best, open questions, completed experiments
2. `.autoresearch/results.tsv` — full experiment history (source of truth if it conflicts with state.md)
3. The entry point file (see state.md `Entry point`) — current config constants at top of file
4. Recent `.autoresearch/experiments/*/report.md` — what was learned from prior experiments

### 1a. Verify state is current

Before proceeding, run `git log --oneline -20` and cross-reference against state.md. If commits show experiments that state.md doesn't reflect (swept variables still listed as "not yet swept", answered questions still in "Open Questions", outdated "Current Best"), update state.md first. A stale state leads to redundant experiments.

## 2. Hypothesize

### Prior-Art Check (mandatory before every hypothesis)

Before proposing any new feature, loss function, architecture change, or configuration:

1. **Check state.md swept variables table** — is this variable already swept? If yes, what won? Proposing a re-test requires a clear reason why the previous result might be wrong (e.g., different feature set, different labeling).
2. **Check ablation history** — was this feature/approach part of a larger set that was ablated? Search the entry point and files listed in state.md `Files to modify` for comments like "Dropped by ablation", "BEST_PARAMS", or similar markers. A feature that was previously ablated has already been tested and lost.
3. **Check experiment docs** — `grep -r` the hypothesis keyword in `.autoresearch/experiments/` and `.autoresearch/results.tsv`. If a prior experiment exists, read its report before proceeding.
4. **Check CLAUDE.md Key Discoveries/Findings** — does any existing discovery contradict or subsume this hypothesis?

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
mkdir -p .autoresearch/experiments/YYYY-MM-DD-<name>
```

Write `.autoresearch/experiments/YYYY-MM-DD-<name>/plan.md` with this structure:

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
<run command> 2>&1 | tee .autoresearch/experiments/<name>/run_<run-name>.log
```

### 4d. Parse results

Use the parser from state.md Environment:

```bash
<parser command> .autoresearch/experiments/<name>/run_<run-name>.log
```

Parse the output per the output contract defined in state.md. Extract each metric listed there. If a metric is missing from the output, default to 0.0.

### 4e. Append to results.json

The file is a JSON array. On first run, create it as `[{...}]`. On subsequent runs, read the existing array, append the new object, and write back the full array.

`.autoresearch/experiments/<name>/results.json`:

```json
[
  {
    "run": 1,
    "phase": "Phase 1: <description>",
    "name": "<run-name>",
    "config": {"<key>": "<value>", ...},
    "results": {"<metric1>": 0.0, "<metric2>": 0, ...},
    "score": 0.0,
    "timestamp": "<ISO 8601>"
  }
]
```

The `results` keys should match the metrics defined in state.md's output contract.

Compute score using the formula from the experiment's plan.md `## Scoring` section.

### 4f. Between phases

Read results.json + plan.md decision logic. Reason about what to test in the next phase. The Phase 1 winner's config becomes the base for Phase 2.

## 5. Conclude

After all phases complete:

### 5a. Write report

Create `.autoresearch/experiments/<name>/report.md`:

```markdown
# Experiment Report: <name>

## Results
| Run | Name | Config summary | <primary metric> | <secondary metrics...> | Score |
|-----|------|---------------|------------------|------------------------|-------|
| ... | ...  | ...           | ...              | ...                    | ...   |

## Analysis
<Phase-by-phase breakdown of what each change revealed>

## Conclusion
<What caused the regression / what improved performance>

## Recommended Config
<The winning config dict>

## Next Hypotheses
<What to investigate next, informed by these results>
```

### 5b. Update .autoresearch/results.tsv

Add a row for the best config from this experiment cycle. Columns should match the header defined during bootstrap (metrics from state.md output contract):

```
<commit>	<metric1>	<metric2>	...	kept	<description>
```

### 5c. Update state.md

Update `.autoresearch/state.md`:
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
- **`.autoresearch/results.tsv` is the source of truth.** If state.md and results.tsv conflict, trust results.tsv.
- **Budget discipline.** Check state.md for run duration, keep experiments reasonable. If you need many runs, split into multiple experiments.
