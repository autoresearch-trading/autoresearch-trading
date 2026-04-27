# Step 4 / Program End-State — Publishable Writeup

**Date:** 2026-04-26 (PM)
**Status:** Program reaches publishable end-state under the post-Gate-2 pre-registration. Multi-probe phenomenology battery (C1, C3, C4) was discovered to be operationally non-runnable on the held-out distribution and is **dropped without measurement** per Path D consensus from council-1, council-4, and council-5.

## One-paragraph headline (revised per council-6 reframe, 2026-04-26 PM)

We trained a 376K-parameter dilated-CNN encoder via MEM + SimCLR self-supervised pretraining on 25 crypto perpetual-futures symbols × 161 days of DEX trade-and-orderbook data (~641K windows at stride 50, no direction labels used). The encoder produced a **per-symbol-clustered, well-conditioned 256-d representation** (pooled RankMe 64.2/256, per-symbol RankMe median 41.4, embed_std 0.825 on Feb+Mar held-out; symbol-ID probe 0.934 on training-period; cross-symbol cluster-cohesion delta +0.037 vs +0.10 universality threshold). Within that representation, a frozen-encoder linear probe achieves **+1pp balanced-accuracy at H500** above majority-class and random-projection controls on Feb+Mar held-out (Gate 1 PASS, 2026-04-23). Supervised end-to-end fine-tuning at lr=5e-5 with horizon-weighted BCE **destroys** that signal — held-out mean H500 bal_acc 0.4947 vs flat-LR 0.5115 (-1.7pp; Gate 2 FAIL, 2026-04-26 AM) — because the per-symbol representation geometry has no shared trunk for fine-tuning to specialize, so gradient signal collapses to per-symbol-and-hour day-conditional shortcuts (CKA drift to Phase B = 0.061, mild; not catastrophic forgetting). Two probes trained on disjoint training-period halves (Oct-Nov vs Dec-Jan), both evaluated on Feb+Mar at H500, show **<3pp drop on 19/24 symbols** (Gate 4 PASS, 2026-04-26 PM) — the +1pp directional signal is **temporally stable** across training-period halves, not a regime-conditional artifact. The phenomenological multi-probe battery (Wyckoff absorption, climax, stress retrieval) intended to test whether the encoder represents tape state beyond direction prior was **discovered to be operationally non-runnable** because the pre-registered window-level labels fire on 0% of held-out windows; per council unanimous Path D (c-1, c-4, c-5), the battery is dropped without measurement and the writeup makes no phenomenological claim. **Final claim:** the encoder produces a per-symbol-clustered representation within which a linearly-extractable +1pp directional signal at H500 is temporally stable across training-period halves but is not amplifiable by supervised end-to-end fine-tuning. The per-symbol representation geometry is the causal mechanism for the fine-tuning failure. No tradeable claim made.

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

- **Cluster-cohesion delta = +0.037 (cross-symbol delta), well below the +0.10 universality threshold.** Symbol-identity probe = 0.934. The encoder's representation space is per-symbol, not universal. (Reported 2026-04-24.) Per council-2's pre-publication review, this is **expected** under symbol-specific OFI coefficients (Cont-de Larrard 2013) — KBONK and BTC live in different microstructure universes (3 orders of magnitude in queue depth, 1–2 in tick-to-spread). A representation that collapsed them into shared geometry would encode *less* OFI structure, not more. This is consistent with "encoder learned per-symbol tape geometries with weak shared geometry from the universal sign-of-flow predicate," but does not falsify "tape-reading"; it characterizes the representation as symbol-conditional, not universal.
- **RankMe (Garrido et al. 2023) on Feb+Mar held-out: pooled 64.2 of 256 (~25% of full dimensionality), per-symbol mean 42.7, median 41.4, min 29.7 (XPL), max 58.1 (HYPE).** Embedding matrix is well-conditioned; not collapsed. Per-symbol RankMe is lower than pooled because each symbol uses fewer directions and the pool spans a larger union — consistent with the per-symbol-clustered geometry above.
- **Embedding std 0.825 on Feb+Mar held-out** (well above the 0.05 collapse threshold from `tape/concepts/contrastive-learning.md`).
- **CKA drift to the Phase B fine-tuned checkpoint = 0.061** (formalized scalar). Mild; not catastrophic forgetting. The encoder's representation geometry is preserved through fine-tuning, but the +1pp signal margin is consumed by shortcut-learning at fine-tune time (council-6's diagnosis, 2026-04-26 PM).
- **Hour-of-day probe at Gate 1 was <10% with cross-session variance <1.5pp.** The pre-pretraining session-of-day mitigation (timing-noise σ=0.10 on `time_delta` and `prev_seq_time_span`, plus pruning `_last` features from the 85-dim flat baseline to FLAT_DIM=83) worked.

### Per-horizon Gate-1 probe accuracy on April 1-13 (descriptive disclosure per council-3, c-3 review 2026-04-26 PM)

Reported as point estimates; **NO pass/fail threshold attached** — this is descriptive context for the Gate 1 +1pp at H500. Operationalization differs from the binding Gate 1 result in two ways: (a) stride=50 not stride=200 (overlapping windows, inflated n; iid assumption violated), (b) April 1-13 not Feb+Mar (different temporal slice, which is the spec's diagnostic-carve-out month, see CLAUDE.md gotcha #17). Per council-3, this disclosure is zero-amendment-budget because it re-uses the same encoder + same probe procedure, decomposed across horizons as a single SSL representation property.

| Horizon | Encoder LR mean bal_acc | Encoder LR > 0.510 count | PCA LR mean | RP LR mean | Shuffled-PCA LR mean (noise) |
|---|---|---|---|---|---|
| H10  | 0.534 | 13/24 | 0.497 | 0.509 | 0.505 |
| H50  | 0.528 | 13/24 | 0.515 | 0.513 | 0.494 |
| H100 | 0.545 | 17/24 | 0.523 | 0.508 | 0.483 |
| H500 | 0.648 | 22/23 | 0.596 | 0.634 | 0.500 |

**Reading per council-3's Kyle-lens framing:** H500 does not stand alone — the encoder is above the shuffled-noise floor at every horizon (H10 +2.9pp, H50 +3.4pp, H100 +6.2pp, H500 +14.8pp). That rules out council-3's worst-case diagnosis ("H500-only would mean slow-drift residual not Kyle-shaped informed-flow detection"). It does NOT, however, prove informed-flow detection — the H500 inflation on April 1-13 (mean 0.648 vs Feb+Mar's mean ~0.514) suggests a genuine directional drift in April that any reasonable predictor (PCA 0.596, RP 0.634, encoder 0.648) picks up linearly. The encoder's H500 margin over RP is +1.4pp; the encoder's margin over Shuffled is +14.8pp but that gap is driven by April's directional momentum, not encoder-specific informational content. **The cleanest reading: the encoder represents short-horizon directional structure (H10/H50/H100 above shuffled noise by 3-6pp), and on April 1-13 also picks up a slow-drift directional momentum at H500.** This is consistent with c-3's "consistent with informed-flow representation but not proof of it" framing.

## What this means for the original research question

The spec asked whether self-supervised pretraining on DEX perpetual tape data produces meaningful representations — measured downstream by direction prediction, with multi-probe phenomenology as the secondary validator.

The honest answer this program supports: **YES on direction (linearly, +1pp, temporally stable), and UNDETERMINED on phenomenology (the secondary validator was operationally undefinable).** This is a narrower positive claim than the spec's intent and does not establish a tradeable edge.

### Calibrated interpretation of the representation (added 2026-04-27)

The interpretation question — does the encoder read tape volume-price phenomenology, or does it read per-symbol direction priors? — is adjudicated **softly** by the existing diagnostics and **not** by a new probe.

**Reading from existing measurements:**

| Diagnostic | Value | Interpretation |
|---|---|---|
| Symbol-identity probe (6 liquid anchors) | 0.934 bal_acc | Encoder is nearly symbol-separable |
| Same-symbol-diff-hour vs cross-symbol-diff-hour cosine delta | +0.139 | Strong per-symbol clustering inside the embedding |
| Cross-symbol-same-hour vs cross-symbol-diff-hour cosine delta (the SimCLR-trained-for axis) | +0.037 | SimCLR alignment signal is **3.8× weaker** than the per-symbol clustering |
| Per-symbol RankMe (median 41.4) vs pooled RankMe (64.2) | ratio 0.65 | Each symbol uses fewer directions than the pool — consistent with per-symbol-clustered geometry |

**Calibrated claim (replaces prior "undetermined" framing):** The encoder produces a **per-symbol-clustered representation with linearly-extractable directional signal** (+1pp at H500, temporally stable). The +0.037 cross-symbol same-hour cosine delta indicates a weak shared geometry — most plausibly the universal sign-of-flow predicate (consistent with council-2's Cont-de Larrard symbol-specific OFI framing). The +0.139 same-symbol delta and 0.934 symbol-ID probe indicate the dominant geometric structure is per-symbol clustering, not Wyckoff-phenomenological tape state.

**What this rules out:** A strong "encoder reads universal tape phenomenology that transfers across symbols" claim. The cross-symbol invariance the spec hoped for was not earned.

**What this does NOT rule out:** Per-symbol tape-reading. The encoder may represent tape state *within each symbol's local geometry*, with the +1pp linear-probe signal being the in-symbol-readable component. This residual question is consistent with the program's own evidence and would require a different evaluation rubric to adjudicate.

**Diagnostic considered and declined (anti-amnesia disclosure):** A bucketed-cohesion paired probe (effort_vs_result × is_open tertiles, encoder vs restricted-PCA cosine within tape-state cells) was designed by council-4 and reviewed by council-5 on 2026-04-27. Council-5 rejected the design as proposed pre-commit on falsifiability grounds (symbol-confound abort risk ~30-40%, multiple-comparisons inflation, marginal power for effects below +0.05) unless council-4 committed to a binding pre-registration with: chi-square symbol-confound abort, 5/9-buckets-with-high-effort-row pass rule, (symbol, date)-blocked bootstrap with 99.44% Bonferroni-corrected CI, and pre-published per-bucket effective block count. Council-5's recommendation: take the off-ramp because the diagnostic had ~80% probability of confirming what existing evidence already showed or aborting on confound grounds, with only ~15% probability of meaningful new information. Lead-0 took the off-ramp on EV grounds. Reviews preserved at:
- `docs/council-reviews/2026-04-27-tape-state-paired-probe-c4-design.md`
- `docs/council-reviews/2026-04-27-tape-state-paired-probe-c5-falsifiability.md`

The interpretation framing above does NOT depend on running this diagnostic; it is grounded in the cohesion + RankMe + symbol-ID evidence already collected. If a future program revisits this question on a different stack (different objective, different data, different architecture), c-4's bucket fix and c-5's falsifiability spine are the starting point.

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
- `docs/council-reviews/2026-04-26-prepublication-strengthening-c2.md` (publish-as-is from OFI lens; per-symbol cohesion is expected)
- `docs/council-reviews/2026-04-26-prepublication-strengthening-c3.md` (publish + per-horizon table on April 1-13)
- `docs/council-reviews/2026-04-26-prepublication-strengthening-c6.md` (publish + RankMe + headline reframe)
- `docs/council-reviews/2026-04-27-tape-state-paired-probe-c4-design.md` (diagnostic design, declined for execution; preserved for institutional memory)
- `docs/council-reviews/2026-04-27-tape-state-paired-probe-c5-falsifiability.md` (diagnostic falsifiability rejection + off-ramp recommendation)
- `runs/step4-r1-perhorizon/` — per-horizon table (`h{10,50,100,500}-april-stride50.json`) and RankMe (`rankme-feb-mar.json`) artifacts

## Anti-amnesia clause

Future writeups citing this program's results MUST disclose:
1. Gate 2 FAIL on the spec-prescribed fine-tuning protocol
2. Gate 4 PASS on temporal stability of the frozen encoder
3. Multi-probe battery dropped without measurement, with reference to commit `6fab561` and the three council reviews
4. The DSR effective N=3 (Gates 1, 2, 4) for any statistical-significance claim on the +1pp Gate 1 result
5. Tape-state-paired-probe diagnostic considered 2026-04-27 and declined pre-commit on falsifiability grounds (council-5 review); calibrated interpretation in §"Calibrated interpretation" rests on existing cohesion + RankMe + symbol-ID evidence, not on a new measurement

The Gate 1 +1pp result is small enough that publishing it as headline without these disclosures is misleading. The headline of any external writeup MUST be *"+1pp linearly-extractable direction signal at H500 within a per-symbol-clustered representation, stable across training-period halves but not amplifiable by supervised fine-tuning, with phenomenological richness untested due to operational label calibration failure"* — not *"SSL learns tape representations on DEX perpetual data."*

## End-state

The program ends here under the ratified pre-registration. No further measurement is authorized without a new pre-registration document and council-1 + council-5 sign-off.
