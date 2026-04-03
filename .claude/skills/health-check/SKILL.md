---
name: health-check
description: Cross-reference spec, CLAUDE.md, code, memory, and knowledge base for inconsistencies. Use periodically or before major implementation steps. Triggers on "health check", "lint knowledge", "check consistency".
---

# Health Check

You are running a consistency audit across the project's documentation, code,
and knowledge base. The goal is to find drift, contradictions, and gaps.

## Process

### 1. Load the three sources of truth

Read these files:
- `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md` (spec)
- `CLAUDE.md` (conventions and gotchas)
- `knowledge/INDEX.md` → then read each linked article

### 2. Cross-reference checks

For each check, record PASS or FAIL with details.

**Spec vs CLAUDE.md:**
- Does CLAUDE.md's feature list match the spec's 17 features?
- Do CLAUDE.md's gotchas reflect all spec constraints?
- Does the architecture description match?

**Spec vs Knowledge Base:**
- Does every feature in the spec have a concept article? List missing ones.
- Does every major decision in the spec have a decision record?
- Are there knowledge articles that contradict the spec?

**Knowledge Base vs Code:**
- For each concept article mentioning a feature: grep for the feature name in
  `prepare.py` and `train.py`. Does the implementation match the article?
- For each decision: is it reflected in current code?

**Staleness:**
- Are any knowledge articles' `last_updated` dates more than 7 days old while
  their source files have been modified more recently? (Check with git log)
- Are any CLAUDE.md gotchas no longer relevant?

**Internal Consistency:**
- Do cross-links in knowledge articles point to articles that exist?
- Are there duplicate articles covering the same topic?
- Do decision statuses make sense? (e.g., "accepted" decision that was later
  superseded should be "superseded")

### 3. Suggest improvements

For each FAIL:
- What is inconsistent
- Which source is likely correct (prefer: code > spec > knowledge > CLAUDE.md)
- Suggested fix

Also suggest:
- New concept articles for topics discussed in council reviews but not yet in
  the knowledge base
- New experiment articles for completed experiments not yet summarized
- Interesting connections between existing articles

### 4. Output

Write the report to `knowledge/HEALTH_CHECK.md`:

```markdown
# Knowledge Base Health Check

**Date:** <today>
**Status:** X pass / Y fail / Z suggestions

## Failures
### F1: <description>
...

## Suggestions
### S1: <description>
...
```

Return a 1-2 sentence summary to the caller.
