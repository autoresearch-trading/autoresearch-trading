# Experiment Skill — Design Spec

## Goal

A reusable Claude Code skill for running structured, multi-phase experiments on `train.py`. Defines a general protocol for hypothesis testing: load an experiment plan, run configs, store results, reason between phases, produce a report.

## Core Concepts

### Experiment Plan

A structured document (JSON phases + markdown decision logic) that defines:
- What to test (configs)
- How to score results
- How to decide what to test next

Plans live in `.claude/skills/experiment/plans/`. Each plan is a folder:

```
plans/v6-ablation/
├── plan.md          — hypothesis, context, decision logic (natural language)
├── phases.json      — structured phase/config definitions
└── results.json     — populated during execution (append-only)
```

### Phase

A group of experiments that test one variable. After a phase completes, Claude reads results, applies the decision logic from `plan.md`, and configures the next phase.

### Config

A single train.py invocation defined by CLI args. Each config has a name and a dict of arguments.

## Plan Schema

### phases.json

```json
{
  "scoring": "sortino * 0.6 + (passing / 25) * 0.4",
  "train_command": "uv run python train.py",
  "phases": [
    {
      "name": "Phase 1: Isolate variables",
      "experiments": [
        {
          "name": "v5-sanity",
          "args": {
            "--labeling": "fixed",
            "--forward-horizon": 800,
            "--min-hold": 800,
            "--fee-mult": 1.5,
            "--feature-mask": 31
          }
        },
        {
          "name": "v6-features",
          "args": {
            "--labeling": "fixed",
            "--forward-horizon": 800,
            "--min-hold": 800,
            "--fee-mult": 1.5
          }
        }
      ]
    },
    {
      "name": "Phase 2: Sweep min_hold",
      "template": {
        "args": "inherit_from_phase1_winner"
      },
      "sweep": {
        "--min-hold": [500, 300, 100]
      }
    }
  ]
}
```

Key fields:
- **scoring**: formula evaluated against result fields (sortino, passing, trades, dd, wr, pf). Claude evaluates this as a Python expression.
- **train_command**: how to invoke train.py
- **phases[].experiments**: explicit configs (Phase 1 style — fully specified)
- **phases[].sweep**: parameterized configs (Phase 2+ style — Claude fills in the base from prior results and sweeps one variable)
- **phases[].template.args = "inherit_from_phase1_winner"**: tells Claude to use the best config from a prior phase as the base

### plan.md

Natural language document that Claude reads for:
- **Hypothesis**: what we're testing and why
- **Decision logic**: how to pick winners between phases, when to early-stop, how to handle ambiguous results
- **Gotchas**: known pitfalls for this specific experiment

Example excerpt:
```markdown
## Decision Logic

After Phase 1, pick features and labeling independently:
- If Run 2 scores > Run 1 by 0.02+, keep 39 features. Otherwise use 31.
- If Run 3 scores > Run 2 by 0.02+, keep Triple Barrier. Otherwise use fixed-horizon.
- If Run 3 scored poorly, note the min_hold/labeling mismatch — Phase 2 may reveal Triple Barrier's true value.

After Phase 2, take the min_hold with highest score.

Early stop: if any phase has a clear winner (gap > 0.1), skip remaining runs in that phase.
```

## Skill Structure

```
.claude/skills/experiment/
├── SKILL.md                  — skill definition, protocol, instructions
├── resources/
│   ├── parse_summary.sh      — extract PORTFOLIO SUMMARY → key=value
│   └── report_template.md    — template for final report
└── plans/                    — experiment plans (one folder each)
    └── v6-ablation/
        ├── plan.md
        ├── phases.json
        └── results.json
```

### SKILL.md

```yaml
---
name: Experiment Runner
description: >
  Run structured multi-phase experiments on train.py to compare model configs,
  isolate regressions, and find optimal hyperparameters. Use when asked to
  "run experiment", "run ablation", "test hypothesis", "compare configs",
  "sweep hyperparameters", or when investigating why a model change helped or hurt.
  Also use when a plan exists in plans/ and needs executing.
---
```

The skill body contains:
1. **Protocol**: how to execute an experiment plan (load → run → store → reason → repeat)
2. **Result parsing**: use `resources/parse_summary.sh` or grep PORTFOLIO SUMMARY
3. **Result storage**: append to `plans/<name>/results.json`
4. **Scoring**: evaluate the plan's scoring formula against each result
5. **Phase transitions**: read `plan.md` decision logic, read accumulated results, configure next phase
6. **Report generation**: after all phases, write report from template

### Protocol (what SKILL.md tells Claude to do)

```
1. Load the experiment plan (plan.md + phases.json)
2. For each phase:
   a. Read phases.json to get configs (or generate from sweep + inherited base)
   b. For each experiment in the phase:
      - Run: {train_command} {args} 2>&1 | tee run_{name}.log
      - Parse PORTFOLIO SUMMARY from output
      - Compute score using the plan's scoring formula
      - Append result to results.json
      - Print: "  {name}: score={score:.3f} sortino={sortino:.3f} passing={passing}/25"
   c. Read plan.md decision logic
   d. Read accumulated results.json
   e. Reason: which config won? What does this mean for the next phase?
   f. Print phase summary with analysis
3. After all phases:
   - Pick overall best config
   - Write report to docs/ablation-report.md (or docs/{plan-name}-report.md)
   - Print recommended config and key findings
```

### results.json format

```json
[
  {
    "run": 1,
    "phase": "Phase 1: Isolate variables",
    "name": "v5-sanity",
    "args": {"--labeling": "fixed", "--min-hold": 800, "--fee-mult": 1.5, "--feature-mask": 31},
    "results": {"sortino": 0.225, "passing": 17, "trades": 890, "dd": 0.35, "wr": 0.0, "pf": 0.0},
    "score": 0.407,
    "timestamp": "2026-03-18T14:30:00"
  }
]
```

### parse_summary.sh

```bash
#!/bin/bash
# Extract PORTFOLIO SUMMARY fields from train.py log
# Usage: bash parse_summary.sh run_v5-sanity.log
grep -E "^(sortino|symbols_passing|num_trades|max_drawdown|win_rate|profit_factor):" "$1"
```

## train.py prerequisite

The skill assumes train.py accepts CLI args that override its module-level config. Required args:

| Arg | Type | Purpose |
|-----|------|---------|
| `--labeling` | `{triple, fixed}` | Labeling method |
| `--forward-horizon` | int | Steps for fixed-horizon labeling |
| `--min-hold` | int | Override MIN_HOLD |
| `--fee-mult` | float | Override BEST_PARAMS["fee_mult"] |
| `--feature-mask` | int | Use only first N features (zero the rest) |
| `--n-classes` | `{2, 3}` | Number of output classes |

Implementation of these args is a prerequisite, not part of this spec. The implementation plan will handle it.

## File Changes

| File | Change |
|------|--------|
| `.claude/skills/experiment/SKILL.md` | New. Skill definition with protocol. |
| `.claude/skills/experiment/resources/parse_summary.sh` | New. Deterministic result parser. |
| `.claude/skills/experiment/resources/report_template.md` | New. Report template. |
| `.claude/skills/experiment/plans/v6-ablation/plan.md` | New. First experiment plan (decision logic). |
| `.claude/skills/experiment/plans/v6-ablation/phases.json` | New. Phase/config definitions. |
| `train.py` | Add CLI args, fixed-horizon labeling, feature masking, 2-class support. |

## Success Criteria

- Skill can execute any well-formed experiment plan autonomously
- Results are stored persistently and readable between phases
- Claude reasons about results (not just mechanically picking highest score)
- Reports clearly attribute outcomes to specific variables
- The v6-ablation plan completes in ~2 hours and identifies the regression cause
