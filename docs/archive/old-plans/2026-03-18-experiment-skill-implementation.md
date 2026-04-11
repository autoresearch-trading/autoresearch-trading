# Autoresearch Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Claude Code skill that drives the autonomous research loop — replacing program.md with a persistent, structured experiment workflow.

**Architecture:** A `.claude/skills/autoresearch/` directory containing the skill protocol (SKILL.md), a shell parser for train.py output (parse_summary.sh), and a living research state document (state.md). CLAUDE.md and results.tsv get housekeeping updates to align with the new workflow.

**Tech Stack:** Claude Code skills (YAML frontmatter + markdown), Bash, existing train.py/results.tsv

---

### Task 1: Create parse_summary.sh

**Files:**
- Create: `.claude/skills/autoresearch/resources/parse_summary.sh`

This is the only testable script in the plan. It extracts PORTFOLIO SUMMARY fields from train.py log output.

- [ ] **Step 1: Create the parser script**

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

- [ ] **Step 2: Verify parser against real output format**

Create a small test log and verify parsing. The fields to match come from train.py lines 451-460:

```bash
cat > /tmp/test_summary.log << 'EOF'
=== PORTFOLIO SUMMARY ===
symbols_passing: 18/25
sortino: 0.229896
num_trades: 923
max_drawdown: 0.3672
win_rate: 0.4500
profit_factor: 1.2300
training_seconds: 300.0
total_steps: 500000
num_updates: 9800
EOF

bash .claude/skills/autoresearch/resources/parse_summary.sh /tmp/test_summary.log
```

Expected output:
```
symbols_passing: 18/25
sortino: 0.229896
num_trades: 923
max_drawdown: 0.3672
win_rate: 0.4500
profit_factor: 1.2300
```

Note: `training_seconds`, `total_steps`, `num_updates` should NOT appear (not in the grep pattern).

- [ ] **Step 3: Also verify with missing win_rate/profit_factor**

```bash
cat > /tmp/test_no_wr.log << 'EOF'
=== PORTFOLIO SUMMARY ===
symbols_passing: 3/25
sortino: -0.003032
num_trades: 7635
max_drawdown: 0.4742
training_seconds: 300.0
total_steps: 500000
num_updates: 9800
EOF

bash .claude/skills/autoresearch/resources/parse_summary.sh /tmp/test_no_wr.log
```

Expected output (no win_rate/profit_factor lines):
```
symbols_passing: 3/25
sortino: -0.003032
num_trades: 7635
max_drawdown: 0.4742
```

- [ ] **Step 4: Make executable and commit**

```bash
chmod +x .claude/skills/autoresearch/resources/parse_summary.sh
git add .claude/skills/autoresearch/resources/parse_summary.sh
git commit -m "feat: add parse_summary.sh for autoresearch skill"
```

---

### Task 2: Create state.md (initial research state)

**Files:**
- Create: `.claude/skills/autoresearch/resources/state.md`

This is the skill's cross-session memory. Initialize it from current results.tsv data and open questions from the spec.

- [ ] **Step 1: Create state.md with current best config and open questions**

```markdown
# Research State

## Environment
- Run command: `uv run python train.py`
- Entry point: `train.py`
- Files to modify per experiment: `train.py` (config constants at top of file)
- Output contract: PORTFOLIO SUMMARY block with `sortino:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines
- Parser: `bash .claude/skills/autoresearch/resources/parse_summary.sh <logfile>`
- Symbols: 25
- Approximate run duration: ~10 min (local M4 CPU, 25 symbols × 5 seeds × 25 epochs)
- Primary metric: Sortino ratio
- Default scoring: `score = mean_sortino * 0.6 + (passing / total_symbols) * 0.4`
- Cache note: Changing prepare.py invalidates all .cache/*.npz (~30 min rebuild). Avoid mid-experiment.

## Current Best
- Config: {lr=1e-3, hdim=256, nlayers=2, wd=5e-4, fee_mult=1.5, min_hold=800, labeling=fixed_horizon, forward_horizon=800, features=31, seeds=5, epochs=25}
- Score: 0.426 (sortino=0.230, passing=18/25)
- Commit: wd5e4

## Latest (v6 tape reading pivot)
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, wd=5e-4, fee_mult=10.0, min_hold=300, labeling=triple_barrier, max_hold=300, features=39, window=50, seeds=5, epochs=25}
- Score: 0.162 (sortino=0.057, passing=8/25)
- Commit: c4f55fb
- Note: Regressed badly from v5. Cause not yet isolated — multiple variables changed simultaneously.
- Note: train.py currently has min_hold=500 (changed after c4f55fb, untested).

## Open Questions
1. What caused the v5→v6 regression? Candidates: new features (31→39), labeling change (fixed→triple barrier), min_hold change (800→300), fee_mult change (1.5→10.0). Need ablation to isolate.
2. Do tape reading features (31→39) help or hurt in isolation?
3. Does Triple Barrier labeling improve over fixed-horizon, holding everything else equal?
4. What is the optimal min_hold for the current setup?
5. Is fee_mult=10.0 too aggressive (barriers too wide)?

## Completed Experiments
(none yet — prior results tracked in results.tsv, pre-skill)
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/autoresearch/resources/state.md
git commit -m "feat: add initial research state for autoresearch skill"
```

---

### Task 3: Create SKILL.md (the research protocol)

**Files:**
- Create: `.claude/skills/autoresearch/SKILL.md`

This is the core of the skill. It contains the full research protocol Claude follows when the skill is invoked.

- [ ] **Step 1: Create SKILL.md**

The skill file must have YAML frontmatter (name, description) followed by the protocol body. The description field controls when Claude Code invokes the skill — make it comprehensive.

```markdown
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

## 2. Hypothesize

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
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/autoresearch/SKILL.md
git commit -m "feat: add autoresearch skill protocol"
```

---

### Task 4: Rename results.tsv header column

**Files:**
- Modify: `results.tsv:1` (header line only)

The header currently says `val_sharpe` but the values are Sortino ratios (see spec Gotcha #6).

- [ ] **Step 1: Rename the column**

Change line 1 of `results.tsv` from:
```
commit	val_sharpe	num_trades	max_drawdown	symbols_passing	status	description
```
to:
```
commit	sortino	num_trades	max_drawdown	symbols_passing	status	description
```

Only the header changes. Data rows are untouched.

- [ ] **Step 2: Verify data rows are intact**

```bash
wc -l results.tsv
cat results.tsv
```

Expected: 10 lines (1 header + 9 data rows), all data rows unchanged.

- [ ] **Step 3: Commit**

```bash
git add results.tsv
git commit -m "fix: rename val_sharpe → sortino in results.tsv header"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Four changes per the spec:
1. Remove `program.md` from Structure section
2. Replace program.md workflow with autoresearch skill workflow
3. Merge Key Discoveries from program.md into a new section
4. Update results.tsv column docs (val_sharpe → sortino)

- [ ] **Step 1: Remove program.md from Structure**

In the Structure section (around line 14), remove the line:
```
program.md              — Experiment loop instructions
```

And add the skill directory:
```
.claude/skills/autoresearch/
  SKILL.md              — Autonomous research loop protocol
  resources/
    parse_summary.sh    — Extract PORTFOLIO SUMMARY → key=value
    state.md            — Current research state (updated each cycle)
docs/experiments/       — Experiment plans, logs, results, reports
```

- [ ] **Step 2: Replace Workflow section**

Replace the current Workflow section (lines 113-119) with:

```markdown
## Workflow

Two distinct modes of work:

1. **Superpowers workflow** (spec → plan → execute) — for structural changes: new features in prepare.py, architecture pivots, eval metric changes, anything that needs design review before code. Specs go in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`.

2. **Autoresearch skill** — for autonomous experimentation. Claude reads current state, forms hypotheses, designs experiments, runs them, draws conclusions, and repeats. Invoked by asking Claude to "run experiments", "investigate", "improve the model", etc. See `.claude/skills/autoresearch/SKILL.md`.

The handoff: execute the superpowers plan to build new infrastructure, then autoresearch takes over for tuning and experimentation.
```

- [ ] **Step 3: Add Key Discoveries section**

After the Gotchas section, add a new section with the unique discoveries from program.md that aren't already captured in Gotchas:

```markdown
## Key Discoveries

1. **Fee structure is the binding constraint** — alpha exists but is thin per trade. Barrier width (fee_mult) is the most sensitive parameter.
2. **One change at a time** — multiple arch/config changes simultaneously = regression. Ablation is the only way to attribute improvements.
3. **MLP beats XGBoost** (18/25 vs 8/25) — temporal pattern extraction from windowed features matters.
4. **Recency weighting helps** — decay=1.0, recent samples ~2.7x weight of oldest.
```

Items 3 (full-test eval) and 5 (v7 attention) from program.md are already in Gotchas #6 and #7.

- [ ] **Step 4: Update results.tsv column reference**

In the Conventions section (around line 125), the line currently says:
```
- **Experiment tracking**: `results.tsv` (commit, sortino, trades, dd, passing, status, description)
```

This already says `sortino` — no change needed. Verify this is correct.

Also check the Evaluation section mentions Sortino (it does). No changes needed there.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "chore: update CLAUDE.md for autoresearch skill workflow"
```

---

### Task 6: Delete program.md

**Files:**
- Delete: `program.md`

All unique content from program.md has been merged into CLAUDE.md (Key Discoveries) and the autoresearch skill (state.md, SKILL.md). Safe to remove.

- [ ] **Step 1: Verify all unique content has been captured**

Quick mental checklist before deleting:
- Setup instructions → CLAUDE.md already has this (Key Exports, Structure)
- Data/splits → CLAUDE.md already has this
- Current Approach → CLAUDE.md already has this (Architecture section)
- v5 Baseline → CLAUDE.md already has this
- What You CAN Modify → CLAUDE.md Workflow section covers this
- Goal/Guardrails → CLAUDE.md Evaluation section
- Experiment Loop → autoresearch SKILL.md
- Output Format → autoresearch SKILL.md parse contract
- Logging → CLAUDE.md Conventions
- Key Discoveries → CLAUDE.md new Key Discoveries section (Task 5)

- [ ] **Step 2: Delete the file**

```bash
git rm program.md
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove program.md, replaced by autoresearch skill"
```

---

### Task 7: Verify the skill works end-to-end

**Files:** (none modified)

Quick smoke test that the skill directory is properly structured and the parser works.

- [ ] **Step 1: Verify skill file structure**

```bash
find .claude/skills/autoresearch -type f | sort
```

Expected:
```
.claude/skills/autoresearch/SKILL.md
.claude/skills/autoresearch/resources/parse_summary.sh
.claude/skills/autoresearch/resources/state.md
```

- [ ] **Step 2: Verify SKILL.md has valid frontmatter**

```bash
head -6 .claude/skills/autoresearch/SKILL.md
```

Expected: YAML frontmatter with `name:` and `description:` fields between `---` delimiters.

- [ ] **Step 3: Verify parse_summary.sh is executable**

```bash
ls -la .claude/skills/autoresearch/resources/parse_summary.sh
```

Expected: `-rwxr-xr-x` permissions.

- [ ] **Step 4: Verify program.md is gone and CLAUDE.md no longer references it**

```bash
test ! -f program.md && echo "program.md deleted OK"
grep -c "program.md" CLAUDE.md
```

Expected: "program.md deleted OK" and count = 0.

- [ ] **Step 5: Verify results.tsv header is updated**

```bash
head -1 results.tsv
```

Expected: `commit	sortino	num_trades	max_drawdown	symbols_passing	status	description`

- [ ] **Step 6: Verify docs/experiments/ doesn't exist yet (created on first run)**

```bash
test ! -d docs/experiments && echo "docs/experiments will be created on first experiment"
```

Expected: confirmation message. The skill creates this directory on first invocation.
