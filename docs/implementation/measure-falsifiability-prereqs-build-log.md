# Build Log: measure_falsifiability_prereqs.py

**Date:** 2026-04-14
**Runtime:** 684.3s (11.4 min) on local CPU

## Files

- Script: `scripts/measure_falsifiability_prereqs.py`
- Report: `docs/experiments/step0-falsifiability-prereqs.md`
- JSON: `docs/experiments/step0-falsifiability-prereqs.json`

## Test Run Results

All five measurements completed successfully.

| Gate | Measurement | Verdict |
|------|-------------|---------|
| 1 | Stress firing (BTC+ETH × 3 dates) | PASS (0.725%–2.819%) |
| 2 | Informed flow firing (BTC+ETH × 3 dates) | PASS (8.80%–13.75%) |
| 3 | Climax date-diversity (all 25 symbols) | PASS (2σ: 25/25 ≥15 dates) |
| 4 | Spring threshold recalibration | PASS (σ=3.0 → ≤8% on 25/25) |
| 5 | Feature autocorr at lag 5 | RECALIBRATE (prev_seq_time_span r>0.8 on all 3; kyle_lambda r>0.8 on BTC) |

## Pyright

0 errors, 0 warnings after fixing typing annotations for pandas rolling chain returns.

## Key numeric results

- Stress: mean 1.626% across BTC+ETH × 3 dates (range 0.725%–2.819%)
- Informed flow: range 8.80%–13.75%
- Climax: minimum 113 dates (LDO at 3σ), all symbols well above 15-date threshold
- Spring sigma=3.0: BTC=6.14%, ETH=5.49%, SOL=3.78% — all under 8%
- Autocorr r>0.8: prev_seq_time_span (BTC 0.907, ETH 0.917, SOL 0.940); kyle_lambda (BTC only: 0.812)
