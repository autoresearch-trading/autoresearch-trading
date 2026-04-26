# Post-Gate-2 Pre-Registration — Multi-Probe Battery + Stop Conditions

**Status: DRAFT — AWAITING USER RATIFICATION.** Once user signs (commit message containing `ratify-post-gate2-prereg`), this document becomes binding.

**Date:** 2026-04-26 (PM)
**Trigger:** Step 4 Gate 2 FAILED (`docs/experiments/step4-gate2-finetune.md`, commit `f795e48`).
**Council inputs:**
- `docs/council-reviews/2026-04-26-post-gate2-strategic-c4.md` (phenomenology — Wyckoff retrieval + axis recovery)
- `docs/council-reviews/2026-04-26-post-gate2-strategic-c5.md` (skeptic — multi-probe battery + stop conditions)
- `docs/council-reviews/2026-04-26-post-gate2-strategic-c6.md` (DL architect — Gate 4 prediction)

## Why this pre-registration is binding

Council-5 was explicit: *"Once Gate 4 runs, signing the stop conditions will be post-hoc. Sign them now."*

The +1pp Gate 1 signal is small enough that confirmation bias on the next probes will be very strong. When the Wyckoff absorption probe comes back at +1.8pp (just under the +2pp threshold), the temptation will be enormous to argue "+1.8pp is statistically significant given the bootstrap CI, surely this counts." It does not count. **The threshold is +2pp; either it clears or it doesn't.**

This document removes lead-0 + user discretion at the moment they will most want to exercise it.

## Pre-committed Multi-Probe Battery

After ratification, the program produces a **POSITIVE** result iff:

### Gate 4 PASSES (binding precondition)

Per spec section "Evaluation Gates":
- Train probe on Oct-Nov 2025 windows (frozen encoder).
- Train separate probe on Dec-Jan 2026 windows (frozen encoder).
- Both probes evaluated on Feb-Mar 2026 (matched-density per spec amendment v2).
- **PASS:** balanced accuracy at H500 drops by < 3pp on > 14/24 symbols between the two probes.
- **FAIL:** drop ≥ 3pp on > 10/24 symbols.

### AND ≥2 of the following 4 multi-probe conditions hold (Feb+Mar held-out)

**Condition 1 — Wyckoff absorption probe.**
- Construct binary label per window:
  ```
  is_absorption = (mean(effort_vs_result[-100:]) > 1.5) AND
                  (std(log_return[-100:]) < 0.5 * rolling_std_log_return) AND
                  (mean(log_total_qty[-100:]) > 0.5)
  ```
- Train logistic regression on frozen 256-dim encoder embeddings.
- **PASS:** balanced accuracy > majority+2pp on **12+/24 symbols** (point estimate, bootstrap CI lower bound published but not the test).
- Computed on Feb+Mar held-out windows only.

**Condition 2 — CKA seed-stability.**
- Train a fresh seed-1 encoder with identical config on a 50% subsample of training data (~3h MPS or ~$3 H100).
- Compute CKA between Run-2 epoch-6 (`runs/step3-r2/encoder-best.pt`) and the new seed-1 epoch-6 checkpoint on a fixed 1024-window held-out batch.
- **PASS:** CKA ≥ 0.75.

**Condition 3 — Cluster purity for Wyckoff states.**
- k-means with k=16 on frozen encoder embeddings of held-out Feb+Mar windows.
- For each of {is_absorption, is_buying_climax, is_selling_climax, is_stressed} compute the **maximum** cluster-conditional label rate.
- **PASS:** at least 2 of 4 labels have max cluster-conditional rate ≥ 40% (vs ≤10% null expectation given roughly-balanced labels and 16 clusters).

**Condition 4 — Embedding trajectory test.**
- Manually identify ≥10 climax events on held-out Feb+Mar data using `climax_score > 3.0` as the seed criterion (then human-validate as actual phase transitions).
- For each event window at time t, compute encoder distance ‖embed(t) − embed(t-50)‖ and compare to the within-symbol-day distance distribution.
- **PASS:** ≥ 7/10 events show distance jump > 2σ above the within-symbol-day mean.

## Pre-committed Program Stop Conditions

### STOP — write negative result if any of:

**Stop A:** Gate 4 fails (>3pp drop on >10/24 symbols at H500).
- Implication: encoder is non-stationary across training period; no fine-tuning or re-pretraining at this scale fixes data-level non-stationarity.
- Action: write up *"Tape SSL on DEX perpetual data: the encoder learned period-specific features; the Gate 1 +1pp linear-probe margin reflects regime-conditional artifacts, not transferable structure. Negative result."*

**Stop B:** Gate 4 passes BUT fewer than 2 of the 4 multi-probe conditions hold.
- Implication: encoder is at a stable +1pp ceiling that extracts only direction-flavored signal; not amplifiable by fine-tuning AND not phenomenologically rich.
- Action: write up *"Self-supervised pretraining on DEX perpetual tape achieves +1pp linearly-extractable direction signal at H500, stable across two held-out months but not amplifiable by supervised fine-tuning. Negative result with respect to a tradeable edge."*

### CONTINUE if:

**Continue A:** Gate 4 passes AND ≥2 of 4 multi-probe conditions hold.
- This is NOT a green light to ship.
- This IS a green light to write up the diagnostics, then design ONE follow-up experiment with explicit pre-registered thresholds.
- No re-pretrain without council-1 + council-5 sign-off and amendment-budget consumption.

**Continue B (shipping criteria, NOT yet ratified for this run):** any tradeable claim requires
- positive Sortino across ≥10 symbols on April 14+ untouched data
- AFTER fees
- AFTER Deflated Sharpe Ratio adjustment for number of probes run
- This is Phase 2 territory per spec line 415-421 and has NOT been pre-committed in this document.

## Bootstrap CI handling (council-5 explicit demand)

For every probe in the multi-probe battery:
- 1000-resample bootstrap 95% CI (percentile method, matching the established `scripts/avax_gate3_probe.py` protocol).
- Report point estimate, 95% CI, and bootstrap-null comparison.
- **The TEST is the point estimate against the threshold.** The CI lower bound is published for information only.
- A point estimate of +1.8pp does NOT pass a +2pp threshold even if the CI lower bound is +1.5pp.

## Anti-amnesia clause (mirroring spec amendment v2 §"Amendment-budget clause")

This pre-registration does not vanish from the record once measured. Future writeups citing the multi-probe battery results MUST publish:
- The thresholds as written here
- The point estimates measured
- The pre-registered stop condition that was triggered (or that wasn't)
- A reference to this file by path

## Trial-count log

Probes computed under this pre-registration:
1. Gate 4 (frozen encoder, Oct-Nov-trained vs Dec-Jan-trained, evaluated on Feb+Mar)
2. Wyckoff absorption probe (Condition 1)
3. CKA seed-stability (Condition 2; requires fresh seed-1 pretraining run)
4. k-means cluster purity (Condition 3)
5. Embedding trajectory test (Condition 4)

Each probe is run ONCE on each of {Feb 2026, Mar 2026} where applicable. No re-runs. Trial count = 5 probes.

## What this pre-registration does NOT authorize

- **Re-pretrain with widened LIQUID_CONTRASTIVE_SYMBOLS (council-4 path C).** Council-5 was explicit: re-pretrain consumes amendment budget (per 2026-04-24 amendment-budget clause) and is an "unearned universality" gambit unless preceded by a measured failure that the new config addresses. The multi-probe battery may produce that measured failure; if so, a re-pretrain pre-registration would be a separate document.
- **Architecture surgery on the head/trunk (council-4 explicitly rejects).** Gate 2 already showed lr=5e-5 fine-tuning destroys representation; head-only variations don't address that.
- **Retrying Phase B with different hyperparameters or temporal val split (council-6 (D)).** Pre-committed against in `docs/experiments/step4-phase-b-third-strike-postmortem.md`.

## Ratification

To ratify this pre-registration as binding:

```
git commit -m "spec: ratify-post-gate2-prereg

Sign-off on docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md
as binding. Multi-probe battery + program stop conditions are now
pre-committed."
```

Until ratified, **NO NEW MEASUREMENT** in the multi-probe battery should run. Gate 4 in particular must wait until ratification — once it runs, signing this document becomes post-hoc and the falsifiability discipline is broken.
