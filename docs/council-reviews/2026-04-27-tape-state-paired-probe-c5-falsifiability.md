# Council-5 — Tape-State Paired Probe Falsifiability Review

**Date:** 2026-04-27
**Diagnostic under review:** Bucket eval windows by (effort_vs_result tertile × climax_score tertile), compare encoder vs PCA mean cosine on cross_symbol_diff_hour pairs within bucket.
**Verdict:** **REJECT AS PROPOSED.** The diagnostic is not unfalsifiable in principle, but as currently specified it carries three independent failure modes that each reduce it to "another +0.037 vs +0.10" inconclusive read. I will not sign off on running it without the modifications below. **There is also an honest off-ramp:** Path A can close cleanly with the interpretation question marked unresolved, and that is in my view the higher-EV outcome unless council-4 can commit to a tightened design that survives sections 1-5 below.

> **Disposition (lead-0, 2026-04-27):** Off-ramp taken. Diagnostic NOT executed. Writeup interpretation tightened to a calibrated claim grounded in existing cohesion data (+0.139 symbol-ID delta, +0.037 SimCLR delta, 0.934 symbol-ID probe). c-5's binding rules and c-4's bucket fix are documented for institutional memory; future programs revisiting this question should start here.

---

## 1. Multiple-comparisons rule

**Setup.** 9 buckets (3×3), one encoder-vs-PCA-cosine-delta test per bucket. The buckets are NOT independent — adjacent tertile cells share window populations under any continuous label, and `effort_vs_result` and `climax_score` are correlated by construction (both contain `log_total_qty` / `z_qty` in the numerator path). Effective number of tests is between 4 and 6, not 9.

**Binding rule for this diagnostic.**
- **Per-bucket alpha:** 0.05 / 9 = **0.0056** (Bonferroni on raw bucket count, conservative).
- **Family-wise pass criterion:** **5/9 buckets** clear the corrected threshold AND **at least 2 of those 5 are in the high-`effort_vs_result` row** (top tertile across all climax columns) — this is the cell where the Wyckoff "absorption / effort exceeds result" hypothesis predicts the strongest signal. A pass concentrated in low-`effort_vs_result` cells is NOT a tape-reading pass; it's volume-magnitude-conditional symbol clustering.
- **Why 5/9, not 9/9:** Gate 1 cleared on 15+/24 symbols, not 24/24, with margin +1.0–1.9pp at the noise edge. Demanding 9/9 here would be inconsistent with the Gate 1 pass standard. 5/9 with the high-`effort_vs_result` concentration constraint is the analog of "majority of buckets, in the buckets that matter."
- **Failure mode this guards against:** "encoder beats PCA in 6/9 buckets but all 6 are low-volume cells" — that's tape-irrelevant (low-volume cells are dominated by inactive periods where any reasonable encoder differentiates noise structure).

**The +1.9-2.3pp Gate 1 margin context.** Gate 1's margin came at the noise edge; that is precisely the reason the per-bucket threshold here cannot be loose. If a bucket-delta below +0.05 counts as "pass," we are weaker than Gate 1 was, applied to a softer test. **The per-bucket binding effect-size threshold must be ≥ +0.05 (see section 4 for full pass language).**

---

## 2. Symbol-confound test (the load-bearing concern)

**Stated correctly:** if illiquid alts populate one tertile and liquid majors populate another, "within-bucket cross-symbol cosine" conditions on symbol set, not on tape state. The diagnostic then re-measures the +0.139 symbol-ID delta inside a confounded subspace and reports it as evidence of tape reading.

**Pre-run binding test (must pass on training-period data BEFORE encoder forward pass on eval windows).**

For each of the 9 (eff_vs_result tertile × climax tertile) buckets, count windows per symbol across the 6 SimCLR anchors (BTC, ETH, SOL, BNB, LINK, LTC):

1. **Per-bucket symbol distribution computed** as 6-vector of counts.
2. **Uniformity test:** chi-square against uniform (expected 1/6 each). The bucket **passes** if p > 0.05 (cannot reject uniformity). The bucket **fails** if any single symbol's share exceeds **35%** OR any symbol's share falls below **5%** (deterministic guardrails on top of the chi-square — the chi-square alone can pass when n is small and miss heavy concentration).
3. **Family-wise rule:** if **3 or more of the 9 buckets fail uniformity**, the diagnostic is **aborted as confounded** before the cosine measurement runs. Reporting partial results across only the uniform buckets is not allowed (post-hoc bucket selection).

**What I expect to happen.** I would put 60% odds that the diagonal cells (low-effort × low-climax, high-effort × high-climax) fail this test — those are the cells where illiquid alts vs liquid majors structurally separate, because rolling-1000-event normalization is *within-symbol* and the joint distribution of normalized-effort × normalized-climax is symbol-shaped (some symbols spend most time in the high-climax-low-effort cell, others rarely visit it). If 3+ buckets fail, the diagnostic ends here and the interpretation question stays unresolved.

**This is the abort condition I most want council-4 to acknowledge.** If they cannot commit to running this confound test FIRST and aborting on its failure, I reject the diagnostic.

---

## 3. Bootstrap CI design

**Parametric SE of cosine at n=5000 pairs is ~0.001 — yes, and that number is a lie.** The pairs are not independent. Each window participates in O(n) pairs; symbol membership is shared across pairs; same-hour and same-date pairs share market state. Effective sample size for a within-bucket cross-symbol cosine mean is closer to **n_unique_(symbol, date) tuples**, which at 168 Feb shards × 6 symbols ≈ 1000 tuples per bucket at uniform distribution, and far fewer in skewed buckets.

**Required design: hierarchical block bootstrap, blocking by (symbol, date).**

- **Block unit:** `(symbol, date)` tuple. This is the minimal unit at which "different market state" can be claimed — within a (symbol, date), windows share regime; across them they don't.
- **Resampling procedure:** For each bucket, resample (symbol, date) blocks WITH replacement to the original count of unique blocks, then enumerate all within-block windows assigned to that bucket, then sample 5000 cross-symbol-diff-hour pairs from the resampled population. **Both the block resample and the pair sample inside the resampled population must be redrawn each bootstrap iteration** — bootstrapping only at the pair level after fixing blocks underestimates variance.
- **Iterations:** 1000 minimum, 2000 preferred.
- **Reported statistic:** percentile-method 95% CI on (encoder cosine − PCA cosine) per bucket. Not the basic-bootstrap CI (asymmetric distributions are likely under skewed bucket symbol mixes).
- **Tie-in to multiple-comparisons:** the per-bucket 95% CI must EXCLUDE ZERO at the Bonferroni-corrected α=0.0056 (i.e., 99.44% CI excludes zero) for that bucket to count toward the 5/9 pass.

**Why not block by hour?** Hour is the protected variable in the cohesion design (cross-symbol-DIFF-hour pairs). Blocking by hour would resample the structure we're conditioning on. **Why not block by date alone?** Cross-symbol confound — different symbols on the same date share macro tape state, so dates aren't independent across symbols. Block by `(symbol, date)` is the cleanest unit.

---

## 4. Pre-commit binding threshold (the critical section)

**This is binding, written before any cosine is computed, and not amendable post-hoc.**

> **PASS condition (all four must hold):**
>
> 1. **Symbol-confound test passes** (≤2 of 9 buckets fail the section-2 uniformity test).
> 2. **Per-bucket effect size:** at least **5 of the 9 buckets** show (encoder cosine − PCA cosine) ≥ **+0.05** as point estimate.
> 3. **Per-bucket significance:** in those same ≥5 buckets, the (symbol, date)-blocked bootstrap **99.44% CI** (Bonferroni-corrected from 95%) excludes zero.
> 4. **Where-it-matters constraint:** at least **2 of the ≥5 passing buckets** are in the high-`effort_vs_result` tertile row.
>
> **FAIL or INCONCLUSIVE condition (any one suffices):**
>
> - 3+ buckets fail the symbol-confound test → ABORT, interpretation question unresolved (not a fail of the encoder; a fail of the diagnostic design).
> - Fewer than 5 buckets clear point estimate +0.05 → INCONCLUSIVE; encoder reads symbol priors in this measurement design.
> - 5+ clear point estimate but the high-`effort_vs_result` concentration constraint fails → INCONCLUSIVE; the directionality of the effect is wrong for a Wyckoff/effort-vs-result reading.
> - 5+ clear point estimate AND concentration but CIs don't exclude zero at corrected α → INCONCLUSIVE on power grounds (see section 5); n_pairs needs to be larger or the diagnostic abandoned.

**Binding language requirement.** Council-4 must commit to the above (or a single named alternative pre-commit) **in a writeup committed to git BEFORE the cosine measurement is computed**. If council-4's proposed threshold contains the phrase "depends on bucket symbol composition" or "we'll know it when we see it," it is unfalsifiable and **I reject the diagnostic outright**.

**Abort path if council-4's threshold is itself unfalsifiable.** Path A closes with this paragraph appended to `docs/experiments/step4-program-end-state.md`:

> *The unresolved interpretation question — does the encoder read tape volume-price phenomenology or per-symbol direction priors — was not adjudicated. A bucketed-cohesion diagnostic was considered but rejected pre-commit on falsifiability grounds (council-5 review 2026-04-27): the diagnostic could not be specified with a binding pass/fail threshold that survived the symbol-confound and multiple-comparisons constraints. The interpretation question is left open and any future writeup citing the +1pp Gate 1 result must disclose it as such.*

**This is a clean Path A close.** The +1pp Gate 1 PASS, +0.037 cohesion delta, and 0.934 symbol-ID probe stand on their own as a coherent "per-symbol-clustered representation with linearly-extractable directional signal" claim. We do NOT need to adjudicate tape-vs-priors to publish that claim honestly.

---

## 5. Power check

**Setup.** Assume the true within-bucket encoder-PCA cosine delta is +0.05 (modest but real). Per-pair cosine std under the existing measurement is ~0.147 (cross_symbol_diff_hour population std from `step5-cluster-cohesion.md`). With effective n at the (symbol, date) block level — call it 500 unique blocks per bucket as an optimistic estimate — the SE on the bucket-mean delta under (symbol, date)-blocked bootstrap is approximately 0.147 / √500 ≈ **0.0066**.

**Detection at +0.05 effect size, Bonferroni-corrected α = 0.0056 (z ≈ 2.77):**
- Required margin to clear: 2.77 × 0.0066 ≈ **0.018**.
- Effect of +0.05 is comfortably above this. Power at +0.05 effect with 500 effective blocks is **≥ 0.95** — adequate.

**At +0.03 effect size:**
- 0.03 / 0.0066 ≈ z = 4.5 — but this assumes the parametric SE estimate. Under realistic block bootstrap with skewed bucket symbol mixes, effective blocks per bucket may be 100–200, not 500. Re-running: SE ≈ 0.147 / √150 ≈ 0.012; z = 0.03 / 0.012 ≈ 2.5 — **just below** the Bonferroni-corrected threshold. **Power at +0.03 is marginal (≈0.5–0.7).**

**Verdict on power.** At the council-4 spec's likely n (5000 pairs per bucket from a population that may have only 100–500 unique (symbol, date) blocks per bucket), the diagnostic has good power for +0.05 effect and **inadequate power for +0.03**. This is acceptable IF the binding threshold is ≥ +0.05 (per section 4). It is **not acceptable** if council-4 wants to claim "tape reading" from a +0.03 effect.

**Pre-run requirement:** council-4 publishes the per-bucket effective block count (unique `(symbol, date)` tuples) **before** running the cosine measurement. If any pass-eligible bucket has fewer than 100 effective blocks, that bucket cannot count toward the 5/9 — either expand the eval set (more shards) or the bucket is dropped from the family.

---

## 6. Cleaner diagnostic? — yes, one alternative I'd accept

The bucketed cosine design has the symbol-confound problem baked in. Here is an alternative that is more falsifiable per analyst-9 half-day budget, no retraining:

**Alternative: per-symbol partial-correlation regression of cosine on tape-state distance, with symbol fixed effects.**

For pairs (i, j) with i, j drawn from the 6 liquid anchors:

```
cos_encoder(i, j) ~ β_tape · |effort_vs_result_i − effort_vs_result_j|
                  + γ_tape · |climax_score_i − climax_score_j|
                  + α_{symbol_i, symbol_j}        (pair-level fixed effect)
                  + δ_{date_i, date_j}            (pair-level date fixed effect)
                  + ε
```

Same regression on `cos_pca`. Compare β_tape, γ_tape coefficients between encoder and PCA.

**Why this is cleaner:**
1. **Symbol identity absorbed into FE** — the +0.139 symbol-ID delta becomes a nuisance parameter, not the load-bearing measurement. Council-2's "symbol-specific OFI coefficients are expected" point is fully accommodated.
2. **Single statistical test** — one β_tape comparison, no multiple-comparisons inflation, no bucket gerrymandering.
3. **Continuous tape-state distance** — no tertile cutoffs, so confound from rolling-1000-event normalization being symbol-shaped is partly absorbed by symbol-pair FEs.
4. **Pre-commit threshold is simpler:** `β_tape_encoder − β_tape_pca` < some negative number (more negative β = closer pairs at smaller tape distance) by ≥ X SE, with cluster-robust SEs at (symbol, date) pair level.

**Pre-commit threshold for the alternative.**
> **PASS:** (β_tape_encoder − β_tape_pca) more negative than (β_pca point estimate − 2 × cluster-robust SE), AND the same direction holds for γ_climax. Both coefficients reported with cluster-robust SEs at the (symbol, date) pair level. Single test, single pass/fail.

**Cost vs benefit.** Same compute as the bucketed design (cosine matrix on the same eval windows), simpler statistics, no multiple-comparisons trap, no symbol-confound bucket gerrymandering. **If council-4 will not commit to the section-4 binding threshold for the bucketed design, I propose this alternative as the single-experiment substitute.**

---

## 7. Final stance

**On the bucketed diagnostic as proposed:** REJECT unless council-4 commits IN WRITING (in a git-committed pre-commit doc, before cosine computation) to:
1. The 5/9 + high-effort-row pass rule (section 1).
2. The chi-square + 35%/5% guardrails symbol-confound abort test (section 2).
3. The (symbol, date)-blocked bootstrap with 99.44% CI exclusion of zero (section 3).
4. The +0.05 minimum per-bucket point estimate (section 4).
5. The pre-publication disclosure of per-bucket effective block count (section 5).

**On the regression alternative:** acceptable as substitute under the section-6 pre-commit.

**On the off-ramp:** the cleanest outcome for Path A is to publish with the interpretation question marked unresolved and the +1pp directional claim narrowly framed (which the end-state writeup already does at lines 60–62 and 91–97). Running an underpowered or confounded diagnostic and getting "INCONCLUSIVE" is strictly worse than not running it — it adds a measurement artifact future writeups must disclose under the anti-amnesia clause, with no information gain.

**My recommendation to lead-0:** unless council-4 agrees to one of the two binding pre-commits in sections 4 or 6, take the off-ramp. The end-state writeup is already honest about what the program established and didn't.
