---
title: Kyle Lambda Per-Snapshot Not Per-Event
date: 2026-04-02
status: accepted
decided_by: Council-3 (Albert Kyle), confirmed by data-eng-13
sources:
  - docs/council-reviews/2026-04-02-council-3-kyle-lambda.md
  - docs/council-reviews/2026-04-02-spec-review-data-eng-13.md
last_updated: 2026-04-03
---

# Decision: Kyle Lambda Per-Snapshot Not Per-Event

## What Was Decided

Compute kyle_lambda over 50 consecutive OB snapshots (~20 min at 24s cadence),
not over 50 consecutive order events. Forward-fill the result to events.

## Why

OB snapshots arrive every ~24s. In a 50-event window, only ~5 events coincide
with a new snapshot (non-zero delta_mid). The other 45 events contribute nothing
to the numerator but inflate the denominator, creating a statistical illusion of
precision. Council-3 showed variance is inflated ~10x relative to true effective
sample size.

Per-snapshot: 50 real observations, each with delta_mid != 0.
Per-event: 50 nominal observations, ~5 effective, ~10x variance inflation.

## Alternatives Considered

- **Per-event with larger window (500 events):** More effective observations but
  covers ~45 min — too long for a meaningful regime indicator.
- **Per-event with variance guard only:** Still noisy, just clipped. Underlying
  estimate quality doesn't improve.

## Impact

Kyle lambda becomes a "regime indicator per snapshot" rather than a per-event
feature. Updates ~every 24s. Model sees the same lambda value for clusters of
events between snapshots. This is correct — the market maker's repricing behavior
doesn't change between snapshots.
