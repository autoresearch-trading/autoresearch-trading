---
title: Amendment Budget — Third Binding-Gate Amendment Requires Council Review
date: 2026-04-24
status: accepted
decided_by: council-5 + lead-0 (spec amendment v2)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
  - docs/council-reviews/council-5-amendment-2026-04-24.md
last_updated: 2026-04-24
---

# Decision: Amendment Budget Clause

## What Was Decided

The spec has been amended twice:
1. Council round 6 on **2026-04-15** — Gate 0 into a 4-baseline publishing
   grid; Gate 1 strengthened to 4 dual-control conditions.
2. Gate 1 + Gate 3 amendment on **2026-04-24** (commits `b1f4065` + `9c91f85`).

**A third binding-gate amendment without a new pre-registered experiment —
i.e., without new evidence collected against pre-dispatched council
thresholds — requires out-of-band council-1 + council-5 review before being
committed to this spec.**

Amendments that update notes, typos, commit-hash references, diagnostic
status, or the Representation Quality Metrics' aspirational targets are **not
binding-gate amendments** and do not consume amendment budget.

## Why

Council-5 identified a pattern: each individual amendment in v1 and v2 was
defensible with its own evidence trail, but the *accumulated* spec after two
amendments is significantly different from the original 2026-04-10
pre-registration. The risk is gradual drift — a sequence of individually-
defensible amendments that collectively erode the spec's falsifiability.

The amendment-budget clause is a firebreak. It does not prohibit future
amendments; it requires that the third one be reviewed specifically for
"this amendment plus the prior two don't collectively break the spec."

## What Counts as a Binding-Gate Amendment

- Changing a pass/fail threshold on Gate 0, 1, 2, 3, or 4.
- Changing the held-out window, held-out symbol, held-out horizon.
- Changing the primary metric (balanced accuracy, raw accuracy, etc.).
- Adding or retiring a binding condition within any gate.
- Changing the training-vs-held-out split definition.

## What Doesn't Count

- Updating commit hashes in references.
- Adding new experiments to the Step 6 interpretation phase (not binding).
- Fixing typos or clarifying ambiguous language without changing thresholds.
- Updating aspirational diagnostic targets (CKA, symbol-ID, Wyckoff probes)
  when the diagnostic is explicitly marked aspirational.
- Rewriting a gate for coherence when the underlying test structure is
  preserved (e.g., [Gate 4 rewrite](gate4-rewrite-for-coherence.md), which
  preserved the 3pp / 10-symbol thresholds).

## Alternatives Considered

1. **Cap total amendments at 2.** Rejected — can be defensible to amend on new
   pre-registered evidence; a hard cap would force silent non-amendment (the
   worse outcome).
2. **Require any amendment (even typo-level) to go through council.** Rejected
   — creates friction for clearly-non-substantive changes and would slow
   research cadence.
3. **Leave unchanged (no budget clause).** Rejected per council-5 — the two
   prior amendments were already a tell; a third without friction is the
   failure mode.

## Impact

- Spec "Amendment Budget" section added in v2 (commit `9c91f85`).
- The clause is self-enforcing: any future amender must cite either (a) new
  pre-registered experimental evidence against pre-dispatched thresholds, or
  (b) council-1 + council-5 out-of-band review.

## Related

- [Gate 1 window amended](gate1-window-amended-feb-mar-h500.md)
- [Gate 3 retired to informational](gate3-retired-to-informational.md)
