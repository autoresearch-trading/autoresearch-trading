# Post-Gate-2 Pre-Registration — Multi-Probe Battery + Stop Conditions

**Status: RATIFIED 2026-04-26 PM.** This document is binding. Amendments: Condition 3 swapped from cluster-purity ≥40% to ARI ≥ 0.05 (noise-robustness against heuristic label noise); execution sequence added (Gate 4 → C1+C3+C4 → C2 only if needed) to minimize wasted compute on likely negative outcomes.

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

**Condition 3 — Cluster–Wyckoff alignment (amended 2026-04-26 PM).**
- k-means with k=16 on frozen encoder embeddings of held-out Feb+Mar windows.
- For each of {is_absorption, is_buying_climax, is_selling_climax, is_stressed}:
  - Compute **adjusted Rand index (ARI)** between the binary Wyckoff labels and the 16 cluster assignments (treating clusters as a 16-class partition, Wyckoff labels as a 2-class partition).
  - ARI is noise-robust against the heuristic nature of the spec's self-labels and against marginal label rate variation across symbols.
  - Pure-random clusters yield ARI ≈ 0; perfect alignment yields ARI = 1.
- **PASS:** at least 2 of 4 Wyckoff labels have **ARI ≥ 0.05** (small but non-zero alignment — accounts for label noise while requiring measurable cluster–state correspondence).
- *Amendment rationale:* the original threshold (max cluster-conditional rate ≥ 40%) was too tight for k=16 on labels with 10–30% marginal rates and meaningful heuristic noise — even a healthy encoder could fail it from label noise alone. ARI ≥ 0.05 is the noise-robust analogue: at chance the value is ~0; at any modest concentration it exceeds 0.05; at perfect alignment it reaches 1.0.

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
4. Cluster–Wyckoff ARI (Condition 3; amended)
5. Embedding trajectory test (Condition 4)

Each probe is run ONCE on each of {Feb 2026, Mar 2026} where applicable. No re-runs. Trial count = 5 probes.

## Execution sequence (binding)

To minimize wasted compute on a likely negative result, probes are run in this order with stop checks:

**Step 1 — Gate 4 first** (~2h CPU, $0). Decisive in both directions.
- If Gate 4 fails → Stop A → write the "encoder non-stationary" negative result. Do NOT run Conditions 1–4.
- If Gate 4 passes → proceed to Step 2.

**Step 2 — Conditions 1, 3, 4** (~4–6h on existing encoder, $0). All on `runs/step3-r2/encoder-best.pt`.
- If 0 of {C1, C3, C4} pass → Stop B → write the "+1pp ceiling, not phenomenologically rich" negative result. Do NOT run Condition 2.
- If 1 of {C1, C3, C4} pass → defer Stop B; proceed to Step 3 (Condition 2 may yet provide the second passing condition).
- If ≥2 of {C1, C3, C4} pass → battery passes; Condition 2 (CKA seed-stability) becomes optional information rather than gating. Skip Step 3 unless reproducibility is needed for the writeup.

**Step 3 — Condition 2 only if Step 2 returns exactly 1 pass** (~3h MPS or ~$3 H100). Fresh seed-1 pretraining on 50% subsample.
- If C2 passes → battery passes (1 + 1 = 2 of 4).
- If C2 fails → Stop B → write negative result.

**Total expected compute, by outcome:**
- Gate 4 fails: ~2h, $0.
- Gate 4 passes, all of C1/C3/C4 fail: ~6–8h, $0.
- Gate 4 passes, one of C1/C3/C4 passes (C2 needed): ~9–11h, ~$3.
- Gate 4 passes, ≥2 of C1/C3/C4 pass: ~6–8h, $0.

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
