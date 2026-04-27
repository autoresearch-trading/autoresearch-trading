# Council-4 Review — Multi-Probe Calibration (C1, C3, C4)

**Date:** 2026-04-26 (PM, post Gate 4 PASS)
**Subject:** `docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md` (commit `6fab561`)
**Verdict:** Path D, with a narrowed-scope concession from me.

## 1. Did I check the empirical scale before specifying 2.5 / 3.0?

**No. I imported the thresholds from a different scale assumption and did not measure.** I specified `climax_score > 2.5 / 3.0` reading the feature as "z-score-like, clipped to [0, 5]" and assumed the upper end of that range was reachable in tail events. I did not run the distribution check on training shards before ratification. The implementation in `tape/features_trade.py:121-125` takes `clip(min(z_qty, z_ret), 0, 5)` with **rolling-1000 σ floored at 0.1** on log_total_qty and on |log_return|. The MIN operator is the killer: for an event to score high, BOTH the qty z-score AND the |return| z-score must be high simultaneously. Step0's per-event rates (`buying_climax_freq` 0.0002–0.0021, `stress_freq` ~0.0000) were sitting in `docs/experiments/step0-data-validation.json` and would have told me the empirical p99 was nowhere near 2.5 if I had looked. I did not look. That is my error and the user should record it.

The same goes for the absorption third-criterion: I wrote `std(log_return[-100:]) < 0.5 * rolling_std_log_return` without resolving what `rolling_std_log_return` meant in a window-level setting where no longer baseline is in the tensor. The within-window resolution is the only honest one and it makes the criterion a 4×-variance-collapse detector — phenomenologically a "post-spike calm" predicate, not "absorption." That is over-specification on my part.

## 2. Single replacement probe I would back

**Yes — exactly one, and it survives all four criteria.** It is **axis recovery on `effort_vs_result`**, lifted from my own original Test 2 in `2026-04-26-post-gate2-strategic-c4.md`:

**Probe:** Train a linear regression on the frozen 256-d encoder embeddings to predict `mean(effort_vs_result over last 100 events of the window)` as a continuous target. Report Spearman ρ on Feb+Mar held-out windows per symbol.

**Pre-registerable now (training-period calibration):** Compute the same probe on Oct–Jan training-period windows (frozen encoder). Take the per-symbol Spearman ρ distribution and set the pass threshold as **median Spearman ρ on training-period − 0.05** (one-time, recorded in commit before Feb+Mar evaluation). This calibrates "is the encoder representing this axis" to the encoder's own training-period level, not to a magic number I would otherwise pull out of the air.

**Phenomenologically rich:** `effort_vs_result` is the master Wyckoff signal — absorption (high) vs ease-of-movement (low) with `is_open` and `climax_score` as the other two load-bearing features. If the encoder cannot linearly recover the per-window mean of this axis on held-out months, it is not representing tape state. If it can, the +1pp Gate 1 result was not a pure direction prior.

**Empirically operational:** `effort_vs_result` is dense, continuous, clipped to [-5, 5] with non-trivial spread in every symbol's training shards (no zero-fire risk).

**Falsifiable on Feb+Mar held-out:** PASS = Spearman ρ ≥ (training-period median − 0.05) on **≥ 14/24 symbols**. FAIL otherwise. Bootstrap CI for information only, point estimate is the test (matching the pre-reg's discipline).

This is **one** probe, regression not classification (sidesteps the entire label-positive-rate disaster), and the threshold derives from training-period data so council-5 cannot charge "calibrated on test."

## 3. Recommendation

**I concede Path D as the primary recommendation.** The pre-reg I co-authored is broken: the absorption label is unreachable, climax/stress labels are 30× off scale, C4's seed cannot fire. Amending now (Paths B/C) compounds falsifiability cost — every amendment looks more like reverse-engineering passable thresholds. Gate 2 FAIL + Gate 4 PASS is the publishable end-state.

If the user wants ONE additional phenomenology probe added before publication, the `effort_vs_result` axis-recovery regression above is the only one I will sign with my name and a training-period-derived threshold. Anything else, including my own Test 1 (Wyckoff-state k-NN retrieval), inherits the same label-sparsity disease and should not run.

**Paths I reject:** B (test-set-fitted quantiles — council-5 will eat it), C (step0-port — only absorption survives, climaxes and stress remain dead, and the X aggregation choices are post-hoc).

## Summary

(1) Admission of error: I specified climax thresholds 2.5/3.0 without measuring the empirical p99 of `min(z_qty, z_ret)` on training shards; the recorded step0 per-event rates would have told me the labels were 30× off scale and I did not consult them. The absorption low-vol-ratio criterion was also over-specified — its within-window-only operationalization makes it a 4×-variance-collapse detector that fires <1% of windows. (2) The only replacement probe I will sign is `effort_vs_result` axis-recovery regression on frozen 256-d embeddings, with PASS threshold derived from Oct–Jan training-period Spearman ρ median minus 0.05, evaluated on ≥14/24 symbols on Feb+Mar held-out — sidesteps the label-positive-rate disaster by being a regression not a classifier and pre-calibrates against training-period data council-5 cannot charge as "test fitting." (3) Path D (skip battery, declare Gate 2 FAIL + Gate 4 PASS the publishable end-state) endorsed; if user wants one more probe, run the axis-recovery probe; reject Paths B and C.
