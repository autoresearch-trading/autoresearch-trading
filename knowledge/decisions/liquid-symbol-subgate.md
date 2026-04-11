---
title: Liquid Symbol Sub-Gate for Gate 1
date: 2026-04-10
status: accepted
decided_by: council-1, council-5
sources:
  - docs/council-reviews/2026-04-10-round5-council-1-gate0-methodology.md
  - docs/council-reviews/2026-04-10-round5-council-5-impl-risks.md
last_updated: 2026-04-10
---

# Decision: Liquid Symbol Sub-Gate for Gate 1

## What Was Decided

Add sub-criterion to Gate 1: **10+/15 liquid symbols must individually exceed
51.4%**, in addition to the existing 15/25 overall requirement.

Also: exclude symbols with N_test < 500 evaluation windows from the 15/25 count.

## Why

The 15/25 gate can be passed by gaming low-volume memecoins where 51.4% has no
statistical power (N=325 windows → critical accuracy is 56.2% at 80% power).
A model that fails on BTC/ETH/SOL but passes on 15 memecoins has not learned
universal microstructure.

Liquid symbols (15): BTC, ETH, SOL, BNB, LINK, LTC, AAVE, UNI, DOGE, AVAX,
SUI, XRP, ENA, LDO, HYPE.

## Alternatives Considered

- Per-symbol adaptive thresholds based on N_test: correct but complex
- Symbol-cluster coverage (5-6 correlation groups): informative diagnostic but
  not a clean gate criterion

## Impact

Pre-register before any pretraining touches April data. Adds one line to the
Gate 1 evaluation code.
