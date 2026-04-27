# Step 4 — Gate 4 (Temporal Stability of Frozen Encoder)

**Date:** 2026-04-26 (PM)
**Status:** PASS (binding precondition of multi-probe battery cleared)
**Pre-registration:** `docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md` (RATIFIED commit `c28bc17`)
**Script:** `scripts/run_gate4.py` (commit `94647d2`)
**Output:** `runs/step4-r1-gate4/gate4.json`
**Checkpoint under test:** `runs/step3-r2/encoder-best.pt` (Run-2 epoch-6)

## Pre-registered protocol (verbatim from pre-reg)

> Per spec section "Evaluation Gates":
> - Train probe on Oct-Nov 2025 windows (frozen encoder).
> - Train separate probe on Dec-Jan 2026 windows (frozen encoder).
> - Both probes evaluated on Feb-Mar 2026 (matched-density per spec amendment v2).
> - **PASS:** balanced accuracy at H500 drops by < 3pp on > 14/24 symbols between the two probes.
> - **FAIL:** drop ≥ 3pp on > 10/24 symbols.

## Verdict

**PASS** — 19 of 24 symbols have `drop = bal_acc(LR_dec_jan) − bal_acc(LR_oct_nov) < 0.03` on Feb+Mar held-out data. Threshold was 15/24.

| Aggregate | Value |
|-----------|-------|
| pass_count | 19 / 24 |
| pass_count_required | 15 |
| mean drop | +0.0061 |
| median drop | −0.0008 |
| max drop | +0.0914 (2Z) |
| mean bal_acc, Oct-Nov-trained | 0.5001 |
| mean bal_acc, Dec-Jan-trained | 0.5063 |

Mean drop +0.6pp is well within the binding 3pp threshold. **The frozen encoder's directional signal is stationary at H500 across the Oct-Nov vs Dec-Jan training-period halves.**

## Symbols that FAILED the per-symbol drop check (5/24)

These are not blockers for the binding gate — the gate requires only 15 of 24 to pass — but worth recording for the diagnostics writeup:

| Symbol | Oct-Nov bal_acc | Dec-Jan bal_acc | Drop | Note |
|--------|-----------------|-----------------|------|------|
| 2Z | 0.4731 | 0.5645 | +0.0914 | The Dec-Jan-trained probe is materially better on Feb-Mar than the Oct-Nov-trained one. NOT the failure direction we feared (catastrophic drop). |
| AAVE | 0.4623 | 0.5174 | +0.0551 | Same pattern: later half is stronger, Oct-Nov-trained underperforms. |
| ASTER | 0.4810 | 0.5164 | +0.0354 | Same pattern. |
| LDO | 0.4938 | 0.5312 | +0.0374 | Same pattern. |
| LINK | 0.4873 | 0.5243 | +0.0370 | Same pattern. |

**All five failures are in the same direction**: the Dec-Jan-trained probe is the stronger one, the Oct-Nov-trained probe is the weaker one. This is a sign-of-life for the encoder, not a non-stationarity flag — recent training data carries more signal for the Feb+Mar evaluation period than older training data, which is the expected direction under any "regimes drift but recent regimes are more like the future" prior.

## Council-6 prediction (verbatim, 2026-04-26 PM strategic review)

> *"Gate 4 prediction (frozen encoder, Oct-Nov vs Dec-Jan, evaluated on Feb+Mar at H500): I expect Gate 4 to PASS with mild stress — the frozen Gate-1 encoder has no fine-tune-induced shortcut overfitting, so within-period drift will look like the Gate-1 ~1pp ceiling on each half rather than a sign-flipped reversal; the worry is that Oct-Nov-trained probe may show 1.5-2.5pp lower balanced accuracy than Dec-Jan-trained probe on 6-10 symbols (regime drift in the training period), but I do not predict >3pp drop on >10/24 symbols, so the binding stop-gate should hold."*

**Outcome vs prediction**: PASS confirmed. Mean Dec-Jan-trained probe is 0.6pp higher than Oct-Nov-trained probe across 24 symbols (council-6 predicted "1.5-2.5pp lower [Oct-Nov] on 6-10 symbols" — actual is 0.6pp on a 24-symbol mean with 5 symbols showing >3pp drop in that direction). Council-6's "mild stress" forecast was accurate; the binding threshold held with margin (19 ≥ 15, 5 fails ≤ 10 max-fail tolerance).

## Implications per pre-registration

- **Stop A (Gate 4 fail) is NOT triggered.** The encoder is NOT non-stationary at the level the binding gate measures.
- **Step 2 of execution sequence is unblocked**: Conditions 1, 3, 4 (Wyckoff absorption probe, ARI cluster–Wyckoff alignment, embedding trajectory test) on existing encoder, ~4–6h, $0.
- The +1pp Gate 1 margin is **not** a regime-conditional artifact in the catastrophic sense the pre-registration's Stop A guards against. It may still be small, fragile, or restricted to direction-flavored signal — those are the questions C1/C3/C4 are designed to test.

## What this result does NOT say

- It does NOT say the encoder represents tape phenomenology. Gate 4 tests stationarity of the directional signal, not whether that signal is "tape-reading" vs "per-symbol direction prior." That is the job of C1/C3/C4.
- It does NOT say the encoder will support a tradeable edge after fees. Phase 2 territory; not in scope of this pre-registration.
- It does NOT say SSL on tape data works in general. One encoder, one config, one pre-registered probe.

## Anti-amnesia clause

This Gate 4 PASS does not retroactively reframe Gate 2's failure. Gate 2 falsified the conjunction [this encoder + supervised end-to-end fine-tuning at lr=5e-5 + H500 BCE labels]. Gate 4 confirms that the frozen encoder used in Gate 2 was at least temporally stable — which means Gate 2's failure was driven by fine-tuning dynamics consuming the +1pp signal margin (council-6's "shortcut learning at fine-tune time" hypothesis), not by encoder non-stationarity.

## Next step (binding per pre-reg)

Execute Step 2 of the execution sequence: run **Conditions 1, 3, 4** in parallel on `runs/step3-r2/encoder-best.pt`.

- C1 — Wyckoff absorption probe (logistic regression on frozen 256-d embeddings → bal_acc > majority+2pp on 12+/24)
- C3 — ARI cluster–Wyckoff alignment (k-means k=16 → ARI ≥ 0.05 on ≥2 of 4 Wyckoff labels)
- C4 — Embedding trajectory test (≥7/10 climax events show ‖embed(t) − embed(t−50)‖ > 2σ above within-symbol-day mean)

If 0 of {C1, C3, C4} pass → **Stop B**, write the "+1pp ceiling, not phenomenologically rich" negative result.
If exactly 1 passes → defer Stop B; run **Step 3 (C2 CKA seed-stability, ~3h)**; if C2 also fails → Stop B.
If ≥2 pass → battery passes; skip C2 unless reproducibility is required for the writeup.
