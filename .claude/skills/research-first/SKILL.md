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

- Before adding a new feature or changing the feature pipeline
- Before changing pretraining objectives (MEM, contrastive) or fine-tuning strategy
- Before proposing an architecture modification
- Whenever someone says "what about X?" or "should we try Y?"

## Research Checklist

### Step 1: Check Internal History (mandatory)

```bash
# Check the knowledge base first
grep -ri "KEYWORD" knowledge/

# Was this approach discussed in council reviews?
grep -ri "KEYWORD" docs/council-reviews/

# Any prior experiment?
grep -ri "KEYWORD" docs/experiments/

# Is it in the spec?
grep -ri "KEYWORD" docs/superpowers/specs/
```

### Step 2: Check Research Docs

```bash
# Do we have research on this topic?
grep -ri "KEYWORD" docs/research/ outputs/
```

If a research doc exists, read it before proceeding. It may contain findings that inform or invalidate the idea.

### Step 3: Check Literature (if novel idea)

If steps 1-2 found nothing, search for academic evidence:
- arXiv for recent papers on the method (MEM, contrastive learning, microstructure)
- Practitioner blogs for implementation experience
- Reference implementations (DeepLOB, etc.)

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

- **Implementing without checking:** Writing code for an approach that was already discussed and rejected by the council
- **Ignoring negative results:** Re-testing a failed approach without a new reason
- **Literature without context:** Adopting a method from equity markets without checking if it applies to DEX perps with 24s OB cadence
- **Over-researching:** Spending hours on literature when a 10-minute experiment would answer the question
- **Skipping the knowledge base:** The wiki at `knowledge/` has distilled council findings — check it before going to raw sources
