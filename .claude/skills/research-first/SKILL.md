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
