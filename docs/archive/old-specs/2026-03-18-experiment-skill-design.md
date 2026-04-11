# Autoresearch Skill — Design Spec

## Goal

A Claude Code skill that IS the research loop. Claude autonomously analyzes current state, forms hypotheses, designs experiments, runs them, draws conclusions, and repeats. Replaces `program.md` as the primary way research gets done.

## Core Loop

```
1. Analyze  → read results.tsv, current config, recent experiments
2. Hypothesize → identify the most impactful question to answer next
3. Design  → create experiment phases with configs and decision logic
4. Execute  → run train.py with each config, store results
5. Conclude → analyze results, write report, update config if improved
6. Repeat  → form next hypothesis based on what was learned
```

Claude drives the entire loop. No human writes experiment plans — Claude generates them as artifacts of its reasoning.

## Scoring

Each experiment plan defines its own scoring formula. For the current trading research:

```
score = mean_sortino * 0.6 + (passing / 25) * 0.4
```

Note: `passing` already accounts for the 20% drawdown guardrail — symbols exceeding max_drawdown are excluded by `eval_policy`, so the score implicitly penalizes high-drawdown configs via lower passing counts.

## Experiment Artifacts

Claude produces experiment artifacts in `docs/experiments/`. Each experiment is a folder:

```
docs/experiments/YYYY-MM-DD-<name>/
├── plan.md          — hypothesis, phases, decision logic (Claude-written)
├── results.json     — accumulated results (append-only)
└── report.md        — conclusions, recommended config (Claude-written)
```

### plan.md (Claude-generated)

Before running any experiments, Claude writes the plan:

```markdown
# Experiment: <name>

## Hypothesis
<What Claude thinks is happening and why>

## Scoring
<formula>

## Phases

### Phase 1: <description>
| Run | Config | Purpose |
|-----|--------|---------|
| 1   | ...    | ...     |

### Phase 2: <description>
Depends on Phase 1 results. Base config inherited from winner.
Sweep: <variable> over [values]

## Decision Logic
<How to pick winners, when to early-stop, how to handle ambiguity>

## Gotchas
<Known pitfalls for this specific experiment>
```

### results.json

```json
[
  {
    "run": 1,
    "phase": "Phase 1: Isolate variables",
    "name": "v5-sanity",
    "config": {"labeling": "fixed", "min_hold": 800, "fee_mult": 1.5, "features": 31},
    "results": {"sortino": 0.225, "passing": 17, "trades": 890, "dd": 0.35, "wr": 0.0, "pf": 0.0},
    "score": 0.407,
    "timestamp": "2026-03-18T14:30:00"
  }
]
```

Notes:
- `config` documents what was changed in train.py for this run (descriptive metadata, not CLI args)
- `passing` is an integer (parsed from `symbols_passing: 17/25` by splitting on `/`)
- `wr` and `pf` default to 0.0 when not printed by train.py

### report.md (Claude-generated)

After all phases complete:

```markdown
# Experiment Report: <name>

## Results
| Run | Name | Config summary | Sortino | Passing | Score |
|-----|------|---------------|---------|---------|-------|

## Analysis
<Phase-by-phase breakdown of what each ablation revealed>

## Conclusion
<What caused the regression / what improved performance>

## Recommended Config
<config dict to use going forward>

## Next Hypotheses
<What to investigate next, informed by these results>
```

## Skill Structure

```
.claude/skills/autoresearch/
├── SKILL.md                  — the research protocol
└── resources/
    ├── parse_summary.sh      — extract PORTFOLIO SUMMARY → key=value
    └── state.md              — current research state (Claude updates this)
```

### SKILL.md

```yaml
---
name: Autoresearch
description: >
  Autonomous research loop for trading model optimization. Use when asked to
  "run experiments", "investigate", "improve the model", "find what's wrong",
  "optimize", or at the start of any research session. Also triggers on
  "autoresearch", "experiment loop", "hypothesis", "ablation".
  Reads results.tsv and current train.py config to decide what to do next.
---
```

The skill body contains:

1. **The research loop** — analyze → hypothesize → design → execute → conclude → repeat
2. **How to analyze** — read results.tsv, train.py config, docs/experiments/ history
3. **How to hypothesize** — identify the highest-leverage question (isolate variables, sweep parameters, test architecture changes)
4. **How to design experiments** — write plan.md with phases, configs, decision logic. One variable per phase. Score formula per plan.
5. **How to execute** — run `uv run python train.py {args}`, parse output, store in results.json
6. **How to conclude** — compare scores, write report.md, update results.tsv, recommend config changes
7. **How to repeat** — the report's "Next Hypotheses" feeds the next cycle's analysis
8. **Guardrails** — max budget per cycle (e.g., 12 runs / ~2 hours), don't change multiple variables at once, always have a control run

### resources/state.md

A living document Claude reads at the start of each session and updates after each experiment cycle:

```markdown
# Research State

## Current Best
- Config: {lr=1e-3, hdim=256, nlayers=2, fee_mult=1.5, min_hold=800, labeling=fixed, features=31}
- Score: 0.426 (sortino=0.230, passing=18/25)
- Commit: wd5e4

## Open Questions
- Do tape reading features (31→39) help or hurt?
- Does Triple Barrier labeling improve over fixed-horizon?
- What is the optimal min_hold for the new setup?

## Completed Experiments
- docs/experiments/2026-03-18-v6-ablation/ — in progress
```

This is the skill's memory across sessions — Claude reads it to know where it left off and what to investigate next.

### resources/parse_summary.sh

```bash
#!/bin/bash
# Extract PORTFOLIO SUMMARY fields from train.py log as key=value pairs
# Usage: bash parse_summary.sh run_v5-sanity.log
#
# Parsing notes:
# - symbols_passing prints "3/25" — split on "/" to get integer
# - win_rate and profit_factor are conditionally printed (only when wr > 0)
#   treat missing values as 0.0
grep -E "^(sortino|symbols_passing|num_trades|max_drawdown|win_rate|profit_factor):" "$1"
```

## Research Protocol Details

### Analyze

Claude reads:
- `resources/state.md` — where we are, open questions
- `results.tsv` — full experiment history
- `train.py` — current config constants
- Recent `docs/experiments/*/report.md` — what was learned

### Hypothesize

Claude identifies the most impactful question. Priority:
1. **Regressions** — if something got worse, isolate why (ablation)
2. **Untested variables** — if a variable hasn't been swept, sweep it
3. **Interactions** — if two variables both helped individually, test them together
4. **Refinement** — if a variable has a clear trend, narrow the search

### Design

Claude writes `docs/experiments/YYYY-MM-DD-<name>/plan.md` with:
- Clear hypothesis
- Phases that change one variable at a time
- Scoring formula
- Decision logic between phases
- Budget (max N runs)

### Execute

For each experiment in each phase:
```bash
uv run python train.py {args} 2>&1 | tee docs/experiments/<name>/run_{exp_name}.log
```

Parse PORTFOLIO SUMMARY, compute score, append to results.json.

Between phases: read results.json + plan.md decision logic, reason about what to test next.

### Conclude

- Write `docs/experiments/<name>/report.md`
- Update `results.tsv` with the best config from this cycle
- Update `resources/state.md` with new current best and open questions
- If the best config improved on the previous best, recommend updating train.py defaults

## train.py contract

The skill does NOT require specific CLI args. Instead:

- **Output contract**: train.py prints a PORTFOLIO SUMMARY block with `sortino:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines. The skill parses these.
- **Modification**: When Claude needs to test a hypothesis that requires changing train.py (e.g., adding a labeling method, masking features, changing min_hold), Claude modifies train.py directly as part of the experiment. This is what a researcher does — the code is the experiment.
- **Rollback**: Claude commits before each experiment and can revert if a change breaks things. Each experiment's plan.md documents what was changed.

## File Changes

| File | Change |
|------|--------|
| `.claude/skills/autoresearch/SKILL.md` | New. Research loop protocol. |
| `.claude/skills/autoresearch/resources/parse_summary.sh` | New. Deterministic result parser. |
| `.claude/skills/autoresearch/resources/state.md` | New. Living research state document. |
| `program.md` | Remove. Unique content merged into CLAUDE.md. |
| `CLAUDE.md` | Update: (1) remove `program.md` from Structure section, (2) remove program.md reference from Workflow section, (3) merge Key Discoveries from program.md into Gotchas or new section, (4) rename `val_sharpe` to `sortino` in results.tsv column docs. |
| `results.tsv` | Rename header `val_sharpe` → `sortino`. |

## Gotchas

1. **Cache rebuild time.** Bumping `_FEATURE_VERSION` in prepare.py invalidates all caches. First train.py invocation per symbol/split rebuilds caches (~2 min/symbol, ~30 min total). Avoid changes to prepare.py mid-experiment.

2. **PORTFOLIO SUMMARY is the contract.** The skill parses train.py stdout for `sortino:`, `symbols_passing:`, etc. If Claude modifies train.py's output format, the parse_summary.sh helper breaks. Keep the output format stable.

3. **Commit before each experiment.** Claude modifies train.py to test hypotheses. Always commit before running so the change is traceable and revertible. Each experiment in results.tsv links to a commit.

4. **One variable per phase.** The most important research discipline. Changing two things at once makes attribution impossible. Claude should resist the temptation to "also try X while we're at it."

5. **results.tsv is the source of truth.** state.md is a summary for fast context loading. If they conflict, results.tsv wins.

6. **results.tsv column name mismatch.** The existing header says `val_sharpe` but the values are Sortino ratios. Rename to `sortino` when merging program.md content into CLAUDE.md.

## Success Criteria

- Claude autonomously identifies the right experiments to run
- Experiment plans are well-reasoned (clear hypothesis, one variable per phase)
- Results are stored persistently and inform future sessions
- Reports clearly attribute outcomes to specific variables
- The first cycle (v6 ablation) completes in ~2 hours and identifies the regression cause
- The skill replaces program.md as the primary research workflow
