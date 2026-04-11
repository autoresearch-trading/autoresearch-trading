# Walk-Forward Validation

> **Goal:** Test if the Sortino=0.353 edge is stable across 4 independent out-of-sample windows or overfit to one test period.

## Design

**Method:** Rolling window, 80-day train / 20-day test / 20-day step forward.

**Folds:**

| Fold | Train Start | Train End | Test Start | Test End |
|------|------------|-----------|------------|----------|
| 1 | 2025-10-16 | 2026-01-03 | 2026-01-04 | 2026-01-23 |
| 2 | 2025-11-05 | 2026-01-23 | 2026-01-24 | 2026-02-12 |
| 3 | 2025-11-25 | 2026-02-12 | 2026-02-13 | 2026-03-04 |
| 4 | 2025-12-15 | 2026-03-04 | 2026-03-05 | 2026-03-25 |

**Architecture:** Standalone `scripts/walk_forward.py`.

1. Imports `full_run`, `eval_policy`, `make_ensemble_policy` from train.py
2. Modifies `make_env` to accept explicit date ranges via a new `date_override` parameter
3. For each fold: patches the date range, runs full training + eval, captures results
4. Reports per-fold and aggregate metrics

**Changes to prepare.py:** Add `train_start`/`train_end`/`test_start`/`test_end` optional params to `make_env()` that override the hardcoded split lookup when provided.

**Output format:**
```
Fold  Train Period          Test Period          Sortino  Passing  Trades  DD
1     Oct 16 - Jan 03       Jan 04 - Jan 23      0.XXX    X/23     XXX    X.XX
2     Nov 05 - Jan 23       Jan 24 - Feb 12      0.XXX    X/23     XXX    X.XX
3     Nov 25 - Feb 12       Feb 13 - Mar 04      0.XXX    X/23     XXX    X.XX
4     Dec 15 - Mar 04       Mar 05 - Mar 25      0.XXX    X/23     XXX    X.XX
──────────────────────────────────────────────────────────────────────────────
Mean                                              0.XXX    X/23     XXX
Std                                               0.XXX
Min                                               0.XXX
```

**Success criteria:**
- Mean Sortino > 0 across all 4 folds → edge is real
- All folds positive → edge is consistent
- 1+ folds negative → edge is regime-dependent
- High variance across folds → edge is fragile

**Cache:** Each fold hits different date ranges. Feature caching keyed on `(symbol, start, end, trade_batch, version)` so folds get their own cache entries. First run will be slow (~30 min/fold × 4 = ~2 hours total).

**Params:** Uses current best: `{lr=1e-3, hdim=64, nlayers=3, batch_size=256, fee_mult=11.0, r_min=0.0, wd=0.0, seeds=5, budget=300s}`
