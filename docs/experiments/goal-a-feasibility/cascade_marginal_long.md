# Goal-A cascade-precursor marginal-long + precision sweep

**Question.** If we go always-long whenever the stage-2 cascade-onset model fires (top 1%, top 0.5%, top 0.1% of windows by predicted probability), is the strategy tradeable net of cost?  Cascades on this universe are 76.7% long-biased at H500 — overwhelmingly forced-short squeezes — so a marginal-long bet may dominate a conditional-direction predictor that we already showed cannot do better than majority-class (LR direction AUC = 0.441 [0.329, 0.551]).

**Protocol.** Reanalysis on existing per-window predictions (`cascade_precursor_real_per_window.parquet`).  No retraining.  Per-day top-pct quantile gating (within each held-out April day, no future leakage).  Day-clustered bootstrap CIs.  Costs from per_window slip at size_usd = 10000 + 4.0bp/side taker fee, both legs.  H500 only (only horizon where the stage-2 LR clears the shuffled-baseline).

**Hard constraints.** April 14+ untouched.  Sample size honest: at top 0.1% across 7 April-diagnostic days, n_triggers may be ~7 (roughly 1/day) — wide CIs, dominated by 1-2 trades on bad days.

## 0. Universe baseline (no filtering)

- Universe: **1803** H500 windows across **7** April-diagnostic days (73 real cascades).
- P(positive forward return | universe) = **0.5752** (NOT cascade-conditional — shows whether April was a uniformly long-biased market).
- P(positive forward return | real_cascade) = **0.7671** (the 76% figure reported in the cascade-direction writeup; reproduced here for consistency check).

## 1. Marginal-long headline (pooled, per precision cutoff)

| top % | n_trig | trig/day | precision | dir_acc_long | mean_pnl_bps (95% CI) | median_pnl_bps (95% CI) | cost_avg | headroom | gross/day (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| 1.000% | 22 | 3.14 | 0.227 | 0.500 | -20.30 [-69.81, +26.13] | +1.27 [-45.71, +17.79] | 11.31 | -31.61 | -99.36 [-262.49, +45.49] |
| 0.500% | 14 | 2.00 | 0.286 | 0.500 | -4.39 [-49.41, +31.48] | +2.64 [-13.96, +21.24] | 9.91 | -14.30 | -28.61 [-115.59, +43.65] |
| 0.100% | 7 | 1.00 | 0.429 | 0.429 | -23.83 [-100.82, +32.25] | -7.37 [-45.82, +17.79] | 9.63 | -33.46 | -33.46 [-112.83, +22.57] |

## 2. Marginal-long vs LR-direction strategy (same precision cells)

LR-direction = 'go long if `direction_pred_proba_lr > 0.55`, short if < 0.45, skip otherwise'.  Only rows with an LR prediction (cascade-likely subset = top 5% of pred_proba) get a position.  Both columns use the same per-day-quantile filter from §1; LR-direction is naturally a *subset* of marginal-long triggers (windows that fall in BOTH top-pct AND top-5% AND have confident LR direction).

| top % | marginal_long gross/day | LR-direction gross/day | n_lr_predicted | which is better? |
|---|---|---|---|---|
| 1.000% | -99.36 | -50.73 | 5 | LR-direction better by +48.63 bps/day |
| 0.500% | -28.61 | -47.69 | 4 | marginal-long better by +19.09 bps/day |
| 0.100% | -33.46 | -48.59 | 3 | marginal-long better by +15.13 bps/day |

## 3. Marginal asymmetry stability across precision cells

Does the long-bias hold up at tighter precision cells, or dilute back toward 0.50?  If P(positive | triggered) drops toward 0.50 at top 0.1%, the 0.77 cascade-conditional asymmetry was driven by selection effects — the *prediction-flagged* windows are not the same population as *realized cascades*.

| top % | n_trig | precision (real cascade frac) | marginal P(positive | triggered) |
|---|---|---|---|
| 1.000% | 22 | 0.227 | 0.500 |
| 0.500% | 14 | 0.286 | 0.500 |
| 0.100% | 7 | 0.429 | 0.429 |

Reference: cascade-conditional asymmetry P(positive | real_cascade) = **0.767**; universe baseline P(positive) = **0.575**.

## 4. Per-symbol breakdown at best precision cell (top 0.500%)

Symbols with at least 3 triggers at this cutoff only — anything fewer is sample-size noise.

| symbol | n_trig | precision | dir_acc_long | mean_pnl_bps | cost | headroom | gross/day |
|---|---|---|---|---|---|---|---|
| BTC | 5 | 0.200 | 0.600 | +16.63 | 8.83 | +7.80 | +5.57 |
| ETH | 5 | 0.600 | 0.400 | -21.04 | 8.55 | -29.59 | -21.14 |

## 5. Verdict

**NOT TRADEABLE pre-encoder.**  Best cell (top 0.500%) yields **-28.61 bps/day** — net of cost, every precision cell is unprofitable.  The 0.77 cascade-conditional long-bias does NOT carry through to the prediction-flagged subset because the LR's top-1% precision is only ~28%; the other ~72% of triggers are non-cascade windows with ~50/50 directional symmetry that drag the mean back to zero.

## 6. Methodological flags

* **Sample size at top 0.1%.** Across 7 April-diagnostic days, n_triggers at top 0.1% is ~7 (≈1/day).  Day-clustered bootstrap captures cluster noise but cannot rescue inference from n=7 — the CI widths reported above honestly reflect this.

* **Per-day quantile vs global quantile.** Per-day quantile keeps stride day-by-day (no future leakage), but on slow days it lowers the trigger threshold, admitting low-confidence windows.  A global quantile would give a tighter cutoff but leak the future distribution.  Per-day is the conservative, leakage-safe choice.

* **Cost size selection.** Used per_window slip at size_usd = 10000 (median bucket).  At 100K size cost would roughly double; at 1K size cost would halve.  The headroom math is sensitive to the size assumption — a strategy that's marginal at 10K may be tradeable at 1K (lower fees) or untradeable at 100K (higher slippage).

* **Universe vs cascade-conditional asymmetry.** The 0.77 long-bias is ON REAL CASCADES.  The prediction-flagged subset has precision ~28% at top 1%, ~43% at top 0.1% — the OTHER 57-72% of triggered windows are non-cascade windows.  If those non-cascade windows have P(positive) ≈ 0.50, the marginal-long bet on the flagged subset ends up close to 0.50 even though the cascade-conditional bias is 0.77.  The §3 table shows this dilution directly.

* **Single-trade-domination at top 0.1%.** The day-clustered bootstrap resamples DAYS, so a single big PnL on one day appears in roughly k/(k+1) ≈ 87% of bootstrap draws.  The CIs at top 0.1% are dominated by which 1-2 trades fall on which day, not by 7-day ensemble averaging.  Read the top 0.1% row with this in mind.

---

_Wall-clock: 4.3 s.  n_boot = 1000.  CPU-only.  No April 14+ data touched._

