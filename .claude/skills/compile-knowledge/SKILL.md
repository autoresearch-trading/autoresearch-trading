---
name: compile-knowledge
description: Compile council reviews, experiment results, and research into the knowledge base wiki. Use after council rounds, experiment analysis, or research sessions. Triggers on "compile knowledge", "update wiki", "update knowledge base".
---

# Compile Knowledge

You are compiling raw sources (council reviews, experiments, research docs) into
the `knowledge/` wiki. The wiki is the canonical place agents look for accumulated
project wisdom. Source files stay untouched — you distill them into wiki articles.

## Process

### 1. Read current state

Read `knowledge/INDEX.md` to see what articles exist and when they were last compiled.

### 2. Scan for new or updated sources

Check these directories for files newer than the last compilation date:

```
docs/council-reviews/*.md
docs/experiments/*/
docs/research/*.md
outputs/*.md
```

For each source file, check if its content is already reflected in existing
knowledge articles (look at the `sources` field in article frontmatter).

### 3. For each new/updated source, extract and file

**Council reviews** produce:
- **Concept updates** — new understanding of a feature or technique
- **Decision records** — what was decided, why, alternatives considered

**Experiment results** produce:
- **Experiment summaries** — hypothesis, result, verdict, what we learned

**Research docs** produce:
- **Concept articles** — new topics or deeper understanding of existing ones

### 4. Create or update articles

**For new concepts:** Create `knowledge/concepts/<slug>.md` using this schema:

```markdown
---
title: <Title>
topics: [<topic1>, <topic2>]
sources:
  - <path to source file>
last_updated: <today's date>
---

# <Title>

## What It Is
<1-3 paragraphs explaining the concept>

## Our Implementation
<How we use it — feature number, computation, parameters>

## Key Decisions
| Date | Decision | Rationale | Source |
|------|----------|-----------|--------|

## Gotchas
<Numbered list of pitfalls>

## Related Concepts
<Links to other knowledge articles>
```

**For new decisions:** Create `knowledge/decisions/<slug>.md`:

```markdown
---
title: <Decision Title>
date: <decision date>
status: accepted|rejected|superseded
decided_by: <who>
sources:
  - <path to source>
last_updated: <today's date>
---

# Decision: <Title>

## What Was Decided
## Why
## Alternatives Considered
## Impact
```

**For new experiments:** Create `knowledge/experiments/<slug>.md`:

```markdown
---
title: <Experiment Name>
date: <experiment date>
status: completed|in-progress|abandoned
result: success|partial-success|failure|inconclusive
sources:
  - <path to source>
last_updated: <today's date>
---

# Experiment: <Title>

## Hypothesis
## Setup
## Result
## What We Learned
## Verdict
```

**For updates to existing articles:** Edit the article to incorporate new
information. Update the `sources` list and `last_updated` date.

### 5. Rebuild INDEX.md

Rewrite `knowledge/INDEX.md` with one-line summaries of all articles, grouped
by type (Concepts, Decisions, Experiments). Each line: link + dash + brief hook
(under 100 chars after the dash). Update the "Last compiled" date.

### 6. Report

Output a summary: how many articles created, updated, and the new INDEX.md
contents. Return a 1-2 sentence summary to the caller.

## Guidelines

- **Distill, don't copy.** Wiki articles should be shorter and more actionable
  than the raw sources. A 2000-word council review becomes a 400-word article.
- **Cross-link aggressively.** If article A mentions a concept covered in article
  B, add a relative link.
- **Preserve dissent.** If council members disagreed, note the dissenting view.
- **Date decisions.** Every entry in a Key Decisions table gets a date.
- **Slug convention:** lowercase, hyphens, no dates (e.g., `kyle-lambda.md`,
  not `2026-04-02-kyle-lambda.md`). Dates go in frontmatter.
