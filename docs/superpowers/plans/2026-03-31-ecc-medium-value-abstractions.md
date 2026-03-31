# ECC Medium-Value Abstractions Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Abstract 4 medium-value patterns from everything-claude-code into our project: dynamic contexts for mode switching, a research-before-coding skill, an experiment eval harness, and a prior-art check integrated into the autoresearch skill.

**Architecture:** Each abstraction is a self-contained file or small skill directory. No shared dependencies between them. Adapted for our ML research workflow (not generic SaaS dev). All files live under `.claude/` or `docs/`.

**Tech Stack:** Markdown (skills, contexts), Bash (hook scripts), Python (eval grader)

**Source repo:** `/tmp/everything-claude-code` (cloned for reference, not a dependency)

---

## Scope

4 independent items, in priority order:

1. **Prior-art check in autoresearch** — add a "check what was already tested" step to the autoresearch skill (prevents the r_btc_lag1 mistake where we re-tested ablated features)
2. **Dynamic contexts** — mode-switching files for research vs dev vs review sessions
3. **Experiment eval harness** — formalized pre/post eval definitions for experiments (we have `results.tsv` but no structured pass/fail criteria)
4. **Research-before-coding skill** — adapted search-first workflow for ML research (check papers, existing features, ablation history before implementing)

Items 5-8 from the ECC repo (continuous-learning-v2, instinct system, observe.sh) are excluded — they require hooks infrastructure we don't have and the instinct extraction depends on a background Haiku agent. Revisit after the simpler items prove their value.

---

### Task 1: Prior-Art Check in Autoresearch Skill

**Files:**
- Modify: `.claude/skills/autoresearch/SKILL.md`
- Reference: `.claude/skills/autoresearch/resources/state.md` (read-only, already exists)

This is the highest-impact change. The r_btc_lag1 experiment failed because we didn't check that funding_zscore and Hurst were already in the 39-feature v6 set and were ablated out. The autoresearch skill's "Hypothesize" step needs a mandatory prior-art check.

- [ ] **Step 1: Read the current autoresearch skill**

Read `.claude/skills/autoresearch/SKILL.md` in full to understand the current Hypothesize step.

- [ ] **Step 2: Add prior-art check to the Hypothesize step**

In `.claude/skills/autoresearch/SKILL.md`, find the `## 2. Hypothesize` section and add a prior-art substep before forming the hypothesis. Insert after the section header and before any existing content:

```markdown
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
```

- [ ] **Step 3: Verify the edit is well-formed**

Run: `head -80 .claude/skills/autoresearch/SKILL.md`
Expected: The prior-art check appears inside section 2, before the hypothesis formation instructions.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/autoresearch/SKILL.md
git commit -m "feat: add prior-art check to autoresearch hypothesis step"
```

---

### Task 2: Dynamic Contexts for Mode Switching

**Files:**
- Create: `.claude/contexts/research.md`
- Create: `.claude/contexts/dev.md`
- Create: `.claude/contexts/review.md`

These are mode-switching files adapted from ECC's `contexts/` directory. The user can inject them via `@.claude/contexts/research.md` at the start of a session to set Claude's posture. Adapted for our ML research workflow.

- [ ] **Step 1: Create the contexts directory**

Run: `mkdir -p .claude/contexts`

- [ ] **Step 2: Create research.md**

Write `.claude/contexts/research.md`:

```markdown
# Research Context

Mode: Exploration, hypothesis formation, literature review
Focus: Understanding before implementing

## Behavior
- Read state.md, results.tsv, and experiment history before proposing anything
- Check prior art (ablation history, swept variables) before forming hypotheses
- Document findings as you go — write to docs/experiments/ or docs/research/
- Do not modify prepare.py or train.py until the research phase concludes with a clear hypothesis

## Research Process
1. Understand the question — what metric are we trying to move?
2. Check what's been tried — state.md swept variables, results.tsv, docs/experiments/
3. Explore literature — arxiv, practitioner blogs, existing research docs
4. Form hypothesis with mechanistic reasoning
5. Design experiment with control run before executing

## Tools to favor
- Read for understanding code and experiment history
- Grep for finding prior results and ablation comments
- WebSearch/WebFetch for papers and external evidence
- Agent(Explore) for codebase questions

## Output
Findings and hypothesis first, implementation plan second. Never code.
```

- [ ] **Step 3: Create dev.md**

Write `.claude/contexts/dev.md`:

```markdown
# Development Context

Mode: Active implementation
Focus: One change at a time, test before and after

## Behavior
- Make the smallest possible change that tests the hypothesis
- Run `uv run pytest tests/ -x -q` after any prepare.py change
- Run `uv run python train.py` for the actual experiment
- Commit before every experiment run
- Parse output with `.claude/skills/autoresearch/resources/parse_summary.sh`

## Priorities
1. Don't break existing tests
2. Make the change
3. Run the experiment
4. Record results in results.tsv

## Tools to favor
- Edit for targeted code changes
- Bash for running tests and training
- Read for verifying changes before running
```

- [ ] **Step 4: Create review.md**

Write `.claude/contexts/review.md`:

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add .claude/contexts/
git commit -m "feat: add dynamic contexts for research/dev/review mode switching"
```

---

### Task 3: Experiment Eval Harness

**Files:**
- Create: `.claude/skills/experiment-eval/SKILL.md`

Adapted from ECC's eval-harness. Instead of capability/regression evals for software features, this defines pre/post eval criteria for ML experiments. It formalizes what we do informally: define success criteria before running, then grade the result.

- [ ] **Step 1: Create the skill directory**

Run: `mkdir -p .claude/skills/experiment-eval`

- [ ] **Step 2: Write the skill**

Write `.claude/skills/experiment-eval/SKILL.md`:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/experiment-eval/
git commit -m "feat: add experiment eval harness skill for structured experiment grading"
```

---

### Task 4: Research-Before-Coding Skill

**Files:**
- Create: `.claude/skills/research-first/SKILL.md`

Adapted from ECC's search-first. Instead of searching npm/PyPI, this searches our own experiment history, ablation results, research docs, and academic literature before implementing a new feature or architecture change.

- [ ] **Step 1: Create the skill directory**

Run: `mkdir -p .claude/skills/research-first`

- [ ] **Step 2: Write the skill**

Write `.claude/skills/research-first/SKILL.md`:

```markdown
---
name: research-first
description: >
  Research before implementing. Check experiment history, ablation results,
  and literature before writing code. Use when proposing new features,
  architecture changes, or training methodology changes. Triggers on
  "should we try", "what about", "add feature", "new approach".
---

# Research-First Protocol

Before implementing any new feature, loss function, architecture change, or training method:
**search for prior art in our own project and in the literature.**

## When to Use

- Before adding a new feature to prepare.py
- Before changing the loss function or training loop in train.py
- Before proposing an architecture modification
- Whenever someone says "what about X?" or "should we try Y?"

## Research Checklist

### Step 1: Check Internal History (mandatory)

```bash
# Was this feature/approach already in the v6 39-feature set?
grep -n "FEATURE_NAME\|feature_name" prepare.py

# Was it tested and ablated?
grep -rn "FEATURE_NAME\|feature_name" prepare.py train.py | grep -i "drop\|ablat\|hurt\|remove"

# Is it in the swept variables table?
grep "VARIABLE_NAME" .claude/skills/autoresearch/resources/state.md

# Any prior experiment?
grep -ri "KEYWORD" docs/experiments/ results.tsv
```

### Step 2: Check Research Docs

```bash
# Do we have research on this topic?
grep -ri "KEYWORD" docs/research/ docs/superpowers/specs/
```

If a research doc exists, read it before proceeding. It may contain findings that inform or invalidate the idea.

### Step 3: Check Literature (if novel idea)

If steps 1-2 found nothing, search for academic evidence:
- arXiv for recent papers on the method
- Practitioner blogs for implementation experience
- The outputs/ directory for any prior deep research

### Step 4: Decision

| Finding | Action |
|---------|--------|
| Previously tested, failed | **STOP** unless you have a mechanistic reason for a different outcome |
| Previously tested, succeeded but was dropped | **INVESTIGATE** why it was dropped (ablation? interaction effect?) |
| In research docs, recommended | **PROCEED** with the recommended approach |
| In research docs, not recommended | **STOP** and read the reasoning |
| Novel, no prior art | **PROCEED** but design a controlled experiment |
| Literature says it works in similar domains | **PROCEED** with adapted implementation |
| Literature says it doesn't work at this timescale | **STOP** |

## Anti-Patterns

- **Implementing without checking:** Writing code for a feature that was already ablated
- **Ignoring negative results:** Re-testing a failed approach without a new reason
- **Literature without context:** Adopting a method from equity markets without checking if it applies to DEX perps at 100-trade batches
- **Over-researching:** Spending hours on literature when a 10-minute experiment would answer the question
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/research-first/
git commit -m "feat: add research-first skill for prior-art checking before implementation"
```

---

## Verification

After all 4 tasks are complete:

- [ ] **Check skill discovery:** Run `ls .claude/skills/` and verify `autoresearch/`, `experiment-eval/`, `research-first/` all exist
- [ ] **Check contexts:** Run `ls .claude/contexts/` and verify `research.md`, `dev.md`, `review.md` exist
- [ ] **Check autoresearch has prior-art check:** `grep "Prior-Art Check" .claude/skills/autoresearch/SKILL.md`
- [ ] **Run tests to confirm no regressions:** `uv run pytest tests/ -x -q`
