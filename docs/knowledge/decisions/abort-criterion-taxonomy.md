---
title: Class A vs Class B Abort Criterion Taxonomy
date: 2026-04-26
status: accepted
decided_by: lead-0 + council-5 + council-6 (post-Phase-B-third-strike)
sources:
  - lead-0.md (binding from 2026-04-26)
  - docs/experiments/step4-phase-b-third-strike-postmortem.md
  - docs/council-reviews/2026-04-26-step4-phase-a-abort-triage.md
  - docs/council-reviews/2026-04-26-step4-phase-b-cka-abort-triage.md
last_updated: 2026-04-27
---

# Decision: Class A vs Class B Abort Criterion Taxonomy

## What Was Decided

Going forward, every abort criterion in any pre-registered training run
must be classified A or B before launch:

- **Class A — binding-science abort:** the pre-committed success criteria
  themselves are at stake. Threshold misspecification = the success
  criterion was wrong. **Limit: one Class A bug per run** before
  STOP-and-redesign.
- **Class B — redundant guard:** a path-property check that's already
  covered by a binding success criterion (e.g., a mid-run rate-of-change
  check redundant with an end-of-run absolute check). **Limit: three
  Class B bugs per run** before mandatory process review.

## Pre-launch Obligations

For any future training run:

1. Every abort criterion classified A or B before launch
2. Every Class B criterion explicitly states which Class A criterion
   subsumes it
3. Any Class A threshold derived from an lr-schedule-dependent quantity
   (CKA, BCE, embed_std, gradient norm) must be checked against a
   deterministic trajectory simulation under the lr schedule — if
   simulation says the threshold cannot be reached, the threshold is
   wrong and must be re-derived BEFORE launch
4. For `BCE × init_factor` thresholds: pre-derive via
   `required_β = 0.5 + sqrt((1 − factor) · log(2) / 2)` against
   measured Gate-1-style baselines

## Audit Trail Rule

If a Class B bug fires, document in a postmortem committed BEFORE
downstream evaluation reads numbers; retired guards must be published
alongside downstream results (anti-amnesia).

## Why

The Step 4 Phase B "three-bug day" (2026-04-26) produced three abort
criteria that fired during a single run:

- **Bug #1 (Class A):** AM Phase A `BCE > 0.95×init` required β=0.632
  from a frozen encoder Gate 1 baseline measured at 0.514. The threshold
  was unreachable under any reasonable training.
- **Bug #2 (Class A):** PM Phase B `CKA > 0.95 after epoch 8` demanded
  ~3× faster encoder rotation than the OneCycleLR schedule supports.
  Mathematically incompatible with the lr schedule.
- **Bug #3 (Class B):** Phase B `max(ΔCKA over last 3) < 0.005` fired
  during scheduled OneCycleLR cosine cooldown (a guaranteed phenomenon).
  Subsumed by the end-of-Phase-B CKA<0.95 upper bound, which the run
  passed.

Without the taxonomy, all three bugs would count equally toward the
"too many bugs, redesign required" decision. With the taxonomy, two
Class A bugs are over the limit and require redesign; one Class B is
not, but counts toward the 3-strike cumulative limit.

## Alternatives Considered

- **Hard-stop at any abort-criterion bug:** Rejected. Class B bugs are
  inevitable when guarding rate-of-change in lr-scheduled training; a
  hard stop on the first such bug would force re-pre-registration on
  every run.
- **No taxonomy, just count:** Rejected after Phase B. Counting Bug #3
  the same as Bugs #1+#2 would have triggered a STOP-and-redesign for
  what was an avoidable redundant guard, not a science-criterion error.

## Impact

- The lead-0.md binding rules were updated 2026-04-26.
- All future training runs (any future re-pretrain or fine-tuning
  attempt) must classify abort criteria A/B before launch.
- The Path D drop of multi-probe battery and the Path A program closure
  both use this taxonomy implicitly: c-5's pre-commit binding on the
  declined tape-state diagnostic is essentially Class A for design
  validity.

## Related

- [Gate 2 Fine-Tuning FAIL](../experiments/gate2-finetune-fail.md) — the run that produced the three bugs
