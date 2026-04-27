# Step 4 / Program End-State — Publishable Writeup

**Date:** 2026-04-26 (PM)
**Status:** Program reaches publishable end-state under the post-Gate-2 pre-registration. Multi-probe phenomenology battery (C1, C3, C4) was discovered to be operationally non-runnable on the held-out distribution and is **dropped without measurement** per Path D consensus from council-1, council-4, and council-5.

## One-paragraph headline

We trained a 376K-parameter dilated-CNN encoder via MEM + SimCLR self-supervised pretraining on 25 crypto perpetual-futures symbols × 161 days of DEX trade-and-orderbook data (~641K windows at stride 50, no direction labels used). On Feb+Mar 2026 held-out evaluation: a frozen-encoder linear probe achieves **+1pp balanced-accuracy at H500** above majority-class and random-projection controls (Gate 1 PASS, 2026-04-23). Supervised end-to-end fine-tuning at lr=5e-5 with horizon-weighted BCE **destroys** that signal (Gate 2 FAIL, 2026-04-26 AM, mean H500 bal_acc 0.4947 vs flat-LR 0.5115; -1.7pp on held-out). Two probes trained on disjoint training-period halves (Oct-Nov vs Dec-Jan), both evaluated on Feb+Mar at H500, show **<3pp drop on 19/24 symbols** (Gate 4 PASS, 2026-04-26 PM) — the +1pp directional signal is **stable** across training-period halves, not a regime-conditional artifact. The phenomenological multi-probe battery (Wyckoff absorption, climax, stress retrieval) intended to test whether the encoder represents tape state beyond direction prior was **discovered to be operationally non-runnable** because the pre-registered window-level labels fire on 0% of held-out windows; per council unanimous Path D, the battery is dropped without measurement and the writeup does not claim phenomenological richness either way. **Final claim:** the encoder produces a linearly-extractable +1pp directional signal at H500 that is temporally stable across training-period halves but is not amplifiable by supervised end-to-end fine-tuning. No tradeable claim made.

## Gates and outcomes (binding)

| Gate | Date | Verdict | File |
|------|------|---------|------|
| Gate 0 (4-baseline grid, noise floor) | 2026-04-15 | Published — pipeline-clean (shuffled labels at 0.500±0.003) | `docs/experiments/step0-falsifiability-prereqs.md` |
| Gate 1 (linear probe on frozen embeddings) | 2026-04-23 | **PASS** — +1.0–1.9pp on 15+/25 symbols, hour-of-day probe <10% | (run-2 results in encoder commit) |
| Gate 2 (fine-tuned CNN vs flat-LR) | 2026-04-26 AM | **FAIL** — -1.7pp at H500, all 3 binding criteria failed both held-out months | `docs/experiments/step4-gate2-finetune.md` |
| Gate 3 (held-out symbol AVAX) | (deferred — gate 2 failed) | not run | — |
| Gate 4 (temporal stability, two-half probes) | 2026-04-26 PM | **PASS** — 19/24 symbols, mean drop +0.6pp | `docs/experiments/step4-gate4-temporal-stability.md` |
| Multi-probe battery (C1, C3, C4) | 2026-04-26 PM | **DROPPED without measurement** — labels operationally non-runnable on held-out distribution | `docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md` + three council reviews |

## What the program established

1. **A small dilated-CNN trained with MEM + SimCLR on ~641K windows × 25 symbols produces frozen embeddings whose direction at H500 is linearly extractable above majority-class and random-projection controls by approximately 1pp.** (Gate 1 PASS.)

2. **That signal is temporally stable.** Probes trained on Oct-Nov 2025 windows and on Dec-Jan 2026 windows, both evaluated on Feb+Mar 2026 at H500, agree within 3pp on 19 of 24 non-AVAX symbols. The 5 per-symbol failures are all in the direction "Dec-Jan-trained probe stronger" (mean Dec-Jan bal_acc 0.5063 vs Oct-Nov 0.5001) — a sign-of-life pattern (recent training data carries more signal for evaluation period), not encoder non-stationarity. (Gate 4 PASS.)

3. **Supervised end-to-end fine-tuning at lr=5e-5 on H10/H50/H100/H500 BCE labels with the spec-prescribed protocol consumes the +1pp signal margin and inverts it on held-out months.** Phase B 13 epochs of joint encoder+head training reduced held-out balanced accuracy to 0.4947 (1.7pp below flat-LR baseline, all 3 binding criteria failed both held-out months). The diagnostic pattern is regression to 0.50: liquid symbols where flat-LR was high (SUI 0.626→0.477, LTC 0.595→0.458, 2Z 0.633→0.499) lost 7–15pp; illiquid alts where flat-LR was below 0.5 gained 6–9pp toward chance. Council-6's diagnosis (2026-04-26 PM): shortcut learning at fine-tune time — Phase B fit day-conditional structure visible in the in-distribution random val split, then collapsed to per-symbol-and-hour priors on novel days. Not catastrophic forgetting (CKA drift mild), not classical overfitting (no train/val gap), not pure distribution shift (Feb+Mar windows statistically similar). (Gate 2 FAIL.)

## What the program did NOT establish

1. **Whether the encoder represents tape phenomenology beyond direction.** The pre-registered multi-probe battery (C1: Wyckoff absorption probe; C3: ARI cluster–Wyckoff alignment; C4: embedding trajectory at climax events) was designed to answer this. We discovered before any encoder forward pass that the binding label thresholds (`climax_score > 2.5/3.0`; absorption's `std(log_return[-100:]) < 0.5 * rolling_std_log_return`) fire on 0% of held-out windows across all 24 non-AVAX symbols. Empirical max of `climax_score` is 0.256 (PENGU); the threshold of 3.0 is ~30× the empirical maximum. Step0_validate's per-event labels (different operationalization) fire at 3–11% absorption per event, confirming the underlying phenomenology is at least partially recoverable with proper window-level operationalization. Council-4 admitted the threshold was specified without checking the empirical scale of `min(z_qty, z_ret)` after rolling-1000 σ + MIN operator. Per unanimous council Path D, the battery is dropped without measurement. **Phenomenological richness is neither confirmed nor falsified by this program.**

2. **Whether the +1pp signal supports a tradeable edge after fees.** Phase 2 territory per spec line 415–421; not in scope. Any tradeable claim requires positive Sortino across ≥10 symbols on April 14+ untouched data, after fees, after Deflated Sharpe Ratio adjustment for number of probes run. The DSR adjustment factor for this program would be N≥3 (Gate 1, Gate 2, Gate 4 — three independent probes against the same temporal slice) per council-1's accounting; even the Gate 1 +1pp result becomes statistically marginal at PSR ≥ 0.95 with N=3.

3. **Whether SSL on tape data works in general.** One encoder, one config, one pre-registered set of probes. The negative result on fine-tuning narrows the hypothesis class (lr=5e-5 with horizon-weighted BCE is a poor downstream protocol) but does not falsify SSL on tape data as a research direction.

## Key diagnostic observations

- **Cluster-cohesion delta = +0.037 (cross-symbol delta), well below the +0.10 universality threshold.** Symbol-identity probe = 0.934. The encoder's representation space is per-symbol, not universal. (Reported 2026-04-24.) This is consistent with both (a) "encoder learned per-symbol direction priors" and (b) "encoder learned per-symbol tape geometries that don't share representation space." The multi-probe battery was intended to discriminate these; it could not be run.
- **Effective rank ~215, embedding std 0.769, CKA drift 0.061 across fine-tune.** No collapse, no catastrophic forgetting. The encoder geometry is well-conditioned; the issue is what it encodes, not how it encodes.
- **Hour-of-day probe at Gate 1 was <10% with cross-session variance <1.5pp.** The pre-pretraining session-of-day mitigation (timing-noise σ=0.10 on `time_delta` and `prev_seq_time_span`, plus pruning `_last` features from the 85-dim flat baseline to FLAT_DIM=83) worked.

## What this means for the original research question

The spec asked whether self-supervised pretraining on DEX perpetual tape data produces meaningful representations — measured downstream by direction prediction, with multi-probe phenomenology as the secondary validator.

The honest answer this program supports: **YES on direction (linearly, +1pp, temporally stable), and UNDETERMINED on phenomenology (the secondary validator was operationally undefinable).** This is a narrower positive claim than the spec's intent and does not establish a tradeable edge. It also leaves the most interesting question — does the encoder actually read tape state? — open.

## Open questions and what would close them

If the user wants to extend the program toward an operational tape-reading claim, the following would be the **methodologically clean** path forward (not authorized by this writeup; would require a new pre-registration consuming amendment budget):

1. **Properly pre-registered phenomenology probe with split-data calibration.** Calibrate window-level Wyckoff labels on April-1-13 already-touched data (or on Oct-Jan training-period data); freeze label spec; evaluate frozen-encoder probes on April-14+ untouched data. Either council-4's `effort_vs_result` axis-recovery regression or council-5's step0-port absorption classifier would be defensible candidates.
2. **One amended probe at council-1's binding accounting** — DSR effective N=4, threshold raised to +2.5pp/12+ or +2pp/14+, full amendment-history disclosure.
3. **Phase 2 backtest on April 14+ untouched data** — positive Sortino across ≥10 symbols, after fees, after DSR adjustment for N≥3 probes already run.

None of these are in scope of the current pre-registration. They are noted here for orientation only.

## Audit trail (binding)

This writeup is consistent with:
- `docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md` (RATIFIED commit `c28bc17`)
- `docs/experiments/step4-multi-probe-c1c3c4-calibration-issue.md` (calibration discovery, commit `6fab561`)
- `docs/council-reviews/2026-04-26-multi-probe-calibration-c1.md` (Path D endorsement on methodology grounds)
- `docs/council-reviews/2026-04-26-multi-probe-calibration-c4.md` (Path D endorsement, c4 admits over-specification error in original threshold spec)
- `docs/council-reviews/2026-04-26-multi-probe-calibration-c5.md` (Path D endorsement, 0%-fire discovery is feasibility check)
- `docs/experiments/step4-gate2-finetune.md` (Gate 2 FAIL, 2026-04-26 AM)
- `docs/experiments/step4-gate4-temporal-stability.md` (Gate 4 PASS, 2026-04-26 PM)

## Anti-amnesia clause

Future writeups citing this program's results MUST disclose:
1. Gate 2 FAIL on the spec-prescribed fine-tuning protocol
2. Gate 4 PASS on temporal stability of the frozen encoder
3. Multi-probe battery dropped without measurement, with reference to commit `6fab561` and the three council reviews
4. The DSR effective N=3 (Gates 1, 2, 4) for any statistical-significance claim on the +1pp Gate 1 result

The Gate 1 +1pp result is small enough that publishing it as headline without these disclosures is misleading. The headline of any external writeup MUST be *"+1pp linearly-extractable direction signal at H500, stable across training-period halves but not amplifiable by supervised fine-tuning, with phenomenological richness not tested due to operational label calibration failure"* — not *"SSL learns tape representations on DEX perpetual data."*

## End-state

The program ends here under the ratified pre-registration. No further measurement is authorized without a new pre-registration document and council-1 + council-5 sign-off.
