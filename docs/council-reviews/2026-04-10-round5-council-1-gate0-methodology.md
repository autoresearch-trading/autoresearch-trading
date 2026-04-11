# Council-1 Round 5: Gate 0 Methodology Review

**Reviewer:** Council-1 (Lopez de Prado — Financial ML Methodology)
**Date:** 2026-04-10
**Round:** 5 — Pre-implementation stress test

## Summary

Gate 0 has four defects that would produce a misleading reference: missing feature standardization, undefined PCA fitting scope, unjustified n=50, and insufficient power for low-volume symbols. The trial log must begin before the first experiment.

## 1. PCA Baseline Setup — Four Defects

### 1.1 Missing Standardization Before PCA
Without StandardScaler, PCA is dominated by features with largest variance (time_delta, kyle_lambda). First 10 components capture timing variation, not market state. **Critical fix: StandardScaler fit on training data only.**

### 1.2 PCA Fitting Scope
- PCA fit: all pre-April data (unsupervised — no label leakage)
- Logistic regression fit: training fold only (supervised)
- C sweep: temporal inner split (last 20% of pre-April by date)

### 1.3 Fixed n=50 Unjustified
Sweep n ∈ {20, 50, 100, 200}. Report 95%-variance-threshold n. Gate 0 reference = best-performing n (make baseline as strong as possible).

### 1.4 Low-Volume Symbol Power Failure
At N=325 windows (low-volume symbol, 13 days), critical accuracy = 56.2% to distinguish from 50% at 80% power. The 51.4% threshold has zero power.

**Fix:** Exclude symbols with N_test < 500 from 15/25 gate.

## 2. Walk-Forward for Baselines

**Recommended clean setup:**
- Training: Oct 16, 2025 – Mar 4, 2026 (~140 days, excluding contaminated Mar 5-25)
- Validation (C sweep): Feb 19 – Mar 4 (inner temporal split)
- Test: April 1-13 (same as Gates 1-4)

Single train/test split, not full walk-forward. Gate 0 is a reference measurement, not a deployment simulation.

## 3. Sample Size and Base Rate

**Per-symbol April 1-13 windows (stride=200):**
- BTC: ~1,820 windows — adequate
- Low-volume: ~65-325 windows — insufficient for 51.4% threshold

**Base rate non-stationarity:** Training base rate may be 52-53% (bull market); April may be 50% (flat). Step 0 must report per-symbol, per-period base rates. Gap > 1pp = regime warning.

## 4. Multiple Testing: Is 15/25 Calibrated?

Under null, expected false positives: ~3.3 symbols (mostly low-volume with weak power). P(15+/25) under null ≈ negligible.

**Danger:** Model passes by gaming low-volume memecoins, failing liquid symbols.

**Fix:** Add sub-gate: **10+/15 liquid symbols must individually exceed 51.4%.**

## 5. Random Encoder Baseline

- Run with **5 seeds**, report mean ± std
- Use `model.eval()` for BatchNorm running statistics
- Expected accuracy: 50.5%-51.0%
- If random encoder > 52%: investigate BatchNorm leakage

## 6. Trial Counting for DSR

**Pre-registration dramatically reduces T:**
- PCA sweep: ~16 trials
- Random encoder: 5 trials
- Pretraining probes: ~6 trials
- Total T ≈ 37-67 (vs T=1,600 on main branch)

**At T=50, realistic threshold = 52.0%** (Holm-Bonferroni corrected).

**April 14+ is the only clean T=1 evaluation.** Everything on April 1-13 is development.

**trial_log.csv must start NOW.** Columns: date, experiment_id, stage, hypothesis, config_hash, metric_primary, n_symbols_passing, test_window, notes.

## Priority Fixes

| Finding | Severity | Timing |
|---|---|---|
| Add StandardScaler before PCA | Critical | Before any baseline runs |
| Start trial_log.csv | Critical | Immediately |
| Sweep PCA n, not fixed 50 | High | Before Gate 0 |
| Exclude N<500 symbols from gate | High | Before Gate 1 |
| Random encoder: 5 seeds | Medium | Before Gate 0 |
| Add liquid-symbol sub-gate | Medium | Before Gate 1 |
