---
title: Cache NaN Contamination — First Full Build
date: 2026-04-15
status: completed
result: bug-found-and-fixed
sources:
  - docs/experiments/cache-nan-investigation.md
last_updated: 2026-04-15
---

# Experiment: Cache NaN Contamination — First Full Build

## Hypothesis

The Task 7 cache pipeline (commit `ec1ea5d`) produces NaN-free feature shards
when run on the full corpus. Smoke test on AAVE 2025-10-16 was clean.

## Setup

- Build: `scripts/build_cache.py` with default date range (2025-10-16 to
  2026-03-31), all 25 symbols, April hold-out hard-gated.
- Output: `data/cache/*.npz`, 4175 attempted / 4003 built / 172 skipped (raw
  data missing for specific symbol-days).
- Wall-clock: 852s (~14 min).
- Validator: `scripts/validate_cache.py` — 8 modular checks.

## Result

**Validator flagged 117 shards with NaN contamination** (879 cells total) across
~20 symbols. Distribution:
- Feb 2–6 2026 cluster (suspected raw-data incident).
- Early October edges (first-day-of-coverage artifacts).
- Scattered individual days.

**One warning**: `KBONK__2026-02-02` had `log_spread = 0.074 > 0` — real extreme
dislocation (ask=0.01067 vs bid=0.0032, 3× ratio), not a code bug.

## Root cause (analyst-9)

`tape/io_parquet.py::expand_ob_levels()` initialized the 10-level flattened
arrays with `np.full(shape, np.nan)`. When a raw OB snapshot had fewer than
10 levels on a side, unfilled slots remained NaN and propagated to four features:

- `depth_ratio` (722 cells) — 10-level sum.
- `imbalance_L5` (141 cells) — 5-level sum.
- `cum_ofi_5` (11 cells) — L1 price/qty usage.
- `delta_imbalance_L1` (5 cells) — cross-snapshot diff.

## Fix (builder-8)

Change `np.full(shape, np.nan)` → `np.zeros(shape)`. Missing levels semantically
= zero liquidity at that depth. Downstream guards (`max(x, 1e-6)`) handle zero
cleanly. Commit `95ca60c`.

## Rebuild

- Wall-clock: 900s (~15 min).
- 4003 shards × 0 critical validator failures.
- 4 residual warnings — all real extreme-dislocation events on illiquid
  symbols on specific dates (not code bugs).
- 289/289 tests passing.

## What We Learned

1. **Single-symbol smoke tests are insufficient.** AAVE on 2025-10-16 didn't
   exhibit the bug because its OB always had 10 levels. Cross-corpus validation
   is required before claiming a cache pipeline is NaN-free.
2. **Raw OB shape is non-uniform.** Thin-book symbols and specific incidents
   produce <10-level snapshots; code must handle this gracefully.
3. **`np.full(np.nan)` is a footgun for feature pipelines.** Any downstream
   aggregation that doesn't explicitly filter NaN will contaminate.
4. **The validator paid for itself on its first real run.** Without it, we'd
   have trained on 117 contaminated shards and chased phantom accuracy
   regressions for days.

## Verdict

**Bug found and fixed.** Cache is production-clean at commit `95ca60c`.
Regression test added to `tests/tape/test_io_parquet.py`.

## Related

- [OB Level Zero-Fill decision](../decisions/ob-level-zero-fill.md)
- [Orderbook Alignment](../concepts/orderbook-alignment.md)
