---
title: cum_ofi Window Is 5 Snapshots Not 20
date: 2026-04-02
status: accepted
decided_by: Council-2 (Rama Cont)
sources:
  - docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md
last_updated: 2026-04-10
---

# Decision: cum_ofi Window Is 5 Snapshots Not 20

## What Was Decided

Set `cum_ofi` to accumulate over 5 OB snapshots (~120s at 24s cadence), not the
originally specified 20 snapshots. Rename the feature from `cum_ofi_20` to
`cum_ofi_5`. After the linear baseline validates signal, sweep {3, 5, 10}.

## Why

The OB cadence is ~24s, not ~3s (see [OB cadence decision](ob-cadence-24s.md)).
At 24s cadence, 20 snapshots covers ~480 seconds (~8 minutes) -- 8x longer than
the spec intended (~60s). This violates the Cont et al. (2014) matching
principle: OFI lookback should roughly match the prediction horizon.

The primary prediction horizon is 100 events forward, which corresponds to ~300
seconds (~5 min) at measured BTC event rates. An OFI lookback of ~120s (5
snapshots) brackets the primary horizon from below. This slight asymmetry is
correct: accumulated order flow over the past 2 minutes predicting whether price
continues in that direction over the next 5 minutes is the forward-looking
version of Cont's same-direction test.

Horizon-specific analysis from Council-2:
- 10 events (~30s): 2-3 snapshots appropriate
- 50 events (~150s): 5 snapshots appropriate
- 100 events (~300s): 5-10 snapshots appropriate
- 500 events (~1500s): 20 snapshots may suit this horizon only

Since horizon-100 is the primary metric, optimize for it.

## Alternatives Considered

- **Keep 20 snapshots:** Covers ~8 min, longer than primary horizon. Wrong
  economic regime for the feature at this cadence.
- **Use 10 snapshots (~240s):** Closer to primary horizon. Valid but Council-2
  recommended starting with 5 and sweeping upward.
- **Per-horizon OFI windows:** Different cum_ofi window for each prediction head.
  Rejected for prototype -- adds complexity. Revisit if 500-event head shows
  strong signal.

## Impact

Feature renamed `cum_ofi_5` throughout spec and CLAUDE.md. The variable name in
code remains `cum_ofi` with window as a hyperparameter for sweeping. The 5-
snapshot default gives ~120s of OFI history -- more aligned with Cont (2014)
original 1-minute test window than the inadvertent 8-minute window from 20
snapshots at 24s cadence.
