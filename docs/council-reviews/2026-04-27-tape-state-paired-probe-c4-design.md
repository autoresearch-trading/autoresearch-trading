# council-4 review — tape-state-paired probe design

**Date:** 2026-04-27
**Reviewer:** council-4 (volume-price phenomenology / Wyckoff voice)
**Question:** Does the encoder read tape (volume-price phenomenology) or per-symbol direction priors?
**Proposed diagnostic:** Tape-state-paired probe — bucket eval windows by tape-state and compare cross-symbol-same-hour cosine in encoder space vs PCA space.
**My verdict on the design:** Conceptually right axis. Bucket definition needs to change. Threshold and sample-size need to be set by what the cluster-cohesion experiment already taught us. Alternative diagnostic at bottom — I think it is strictly better but more expensive; defer to council-5/council-1 on cost.

> **Disposition (lead-0, 2026-04-27):** Diagnostic NOT executed. Path A off-ramp taken per council-5's falsifiability review and EV calculation (~80% probability of confirming what we already know or aborting; ~15% odds of meaningful new info). c-4's bucket fix and adjudication metric are documented for institutional memory; if a future program revisits the interpretation question on a different stack, this review is the starting point.

---

## 1. Buckets — `effort_vs_result × climax_score` tertiles is the WRONG choice. Use `effort_vs_result × is_open` tertiles.

### Why not climax_score

This is the same trap that killed C1/C3/C4. From `step4-multi-probe-c1c3c4-calibration-issue.md` empirical table:

- `climax_max` per symbol over Feb is 0.04–0.26
- `climax_score` p99 ≈ 0.005–0.020
- Therefore tertile boundaries on `climax_score` (computed at the WINDOW level — say `max(climax_score over 200 events)` or `mean`) will land at values like 0, 0.003, 0.02. The "high" tertile is not a Wyckoff climax — it is "this window had any non-zero z_qty AND z_ret simultaneously." That is barely informative and dominated by sampling noise from the rolling-1000 σ + MIN operator.

If we tertile-bucket on `climax_score`, the "high climax" bucket will be ~33% of windows by construction — which obviously is not a climax-rate phenomenon. The bucket would not isolate the load-bearing phase-transition phenomenology; it would just sort windows by "how non-zero is climax_score on this 200-event window," which is a soft proxy for volume bursts that `effort_vs_result` already captures more directly.

The two-feature interaction `effort_vs_result × climax_score` is therefore close to redundant in practice (both are functions of normalized log_total_qty), and the redundant axis is the noisy one.

### Why `effort_vs_result × is_open` is the right tape-state factorization

The three load-bearing Wyckoff features per the spec are `effort_vs_result`, `climax_score`, and `is_open`. Of those three:

- `effort_vs_result` empirically fires (the `evr_high_rate` column in the calibration table is 2–91% across symbols at threshold 1.5 — meaningful spread). It is the master microstructure axis: high EVR = absorption-flavor, low EVR = ease-of-movement / breakout-flavor.
- `is_open` empirically fires every window with finite spread (it is bounded [0,1] by construction, no rolling-σ scale problem) and is the **DEX-specific Composite Operator footprint** — the feature with no equivalent in traditional markets, the feature whose autocorrelation half-life of 20 trades was the single strongest persistent signal in this whole dataset.
- `climax_score` empirically does not fire enough to bucket on.

The 2×2 (or 3×3) factorization of `effort_vs_result × is_open` directly corresponds to the Wyckoff/microstructure tape-state taxonomy in CLAUDE.md:

| `is_open` (window mean) | low EVR | high EVR |
|---|---|---|
| **high** (Composite Operator active) | Markup / Breakout-with-conviction | Spring / Selling Climax / Test |
| **low** (closing flow dominates) | Markdown / Drift | Upthrust / Distribution / Absorption-against-trend |

These four cells are *operationally distinct tape states with distinct microstructure signatures*. They are NOT the same as "high vs low normalized volume" — `is_open` is orthogonal to volume and orthogonal to return.

Crucially, `is_open` is not symbol-specific the way price-scale features are. It is a fraction in [0, 1]. Tertiles fit cleanly across symbols without per-symbol re-normalization. That is exactly what we need for a cross-symbol-same-hour comparison: the bucket assignment must be on a feature whose distribution is comparable across BTC and KBONK, and `is_open` is. `climax_score` after rolling-1000 σ is also approximately comparable, but its operational range is too compressed to tertile usefully.

### Concrete bucket definition

For each evaluation window `w` of 200 events, compute two scalars:
- `evr_w = mean(effort_vs_result over events 100–199)` — last-half mean, matching the C1 attempt's intent
- `is_open_w = mean(is_open over events 100–199)` — last-half mean

Compute tertile boundaries **per-symbol** on Feb+Mar windows for `evr_w`, but **pooled across all 6 anchors** for `is_open_w`. (Per-symbol on EVR because rolling-median normalization is local; pooled on is_open because the [0,1] scale is universal.)

Rationale for the asymmetry: per-symbol tertiling on EVR controls for the symbol-specific volume distribution; pooled tertiling on is_open lets us see whether high-Composite-Operator-activity windows from BTC and ETH cluster together (which is the universality claim). If we per-symbol-tertile is_open, we wash out exactly the cross-symbol signal we are trying to measure.

3×3 = 9 buckets. If sample-size constraints (see §3) bind, fall back to 2×2 (median split on each axis).

---

## 2. "Meaningfully above" threshold

Cluster cohesion measured the SAME-HOUR delta over diff-hour as +0.037, where the council-5 universality threshold for "some invariance" was +0.10. The earlier diagnostic's signal-to-baseline ratio is the right calibration here.

For this diagnostic the relevant comparison is **encoder cosine within-bucket** vs **PCA-on-flat-features cosine within-bucket**, on cross-symbol-same-hour pairs. Call those `cos_enc_b` and `cos_pca_b` for bucket `b`.

I propose three thresholds, not one, because the experiment has three falsifiable outcome regimes:

**Threshold A — symbol-prior null:** if `mean_b(cos_enc_b - cos_pca_b) ≤ +0.05` AND no individual bucket exceeds `+0.10`, the encoder is NOT reading tape state distinctively from PCA. This is the null. (Note: I expect the encoder cosine to be HIGHER than PCA cosine in absolute terms across all buckets simply because encoder cosines live on the narrow 256-sphere cone; what matters is whether the gap is bucket-DEPENDENT.)

**Threshold B — uniform tape-reading (weak pass):** `mean_b(cos_enc_b - cos_pca_b) > +0.05` with low across-bucket variance (std across 9 buckets < +0.03). Means encoder reads SOMETHING beyond per-symbol prior, but the something is roughly uniform across tape states — could be slow drift, could be session-of-day residual, could be tape-reading at a coarse level that does not differentiate Wyckoff cells.

**Threshold C — differentiated tape-reading (strong pass):** at least 2 of the 9 buckets have `cos_enc_b - cos_pca_b > +0.10`, AND at least one of those is the high-EVR-high-is_open cell (Spring/Climax cell — the one Wyckoff says should have the most distinctive tape signature) OR the low-EVR-high-is_open cell (Markup with conviction). The differentiation pattern, not just the mean, is what would persuade me.

The +0.10 number is council-5's universality bar from cluster cohesion, deliberately reused. The +0.05 mean-gap floor is half of that, which I would interpret as "encoder reads tape but not strongly enough to differentiate Wyckoff phases" — a real but weak signal.

I want to flag one risk explicitly: encoder cosines naturally sit at ~0.70 cross-symbol baseline (cluster cohesion table) while PCA cosines on standardized flat features will likely sit at ~0.0 baseline. The DELTA in raw cosines will be huge no matter what (~0.7), and that is not informative. The right comparison is **within-bucket gap minus across-bucket gap** for each space, not raw cosine. So compute:

```
diag_enc_b = cos_enc_b (within-bucket) - cos_enc_global (cross-symbol all pairs)
diag_pca_b = cos_pca_b (within-bucket) - cos_pca_global (cross-symbol all pairs)
diff_b = diag_enc_b - diag_pca_b
```

Then thresholds A/B/C apply to `diff_b`, not to raw cosine differences. This is the right normalization to defeat the embedding-cone artifact and make encoder/PCA spaces comparable.

---

## 3. Sample-size requirement per bucket

Cluster cohesion ran 50K cross-symbol-diff-hour pairs total, single population. The 9-bucket experiment demands per-bucket pair counts.

**Floor: 2,000 cross-symbol-same-hour pairs per bucket** for the bucket to be reported. Below that, drop the bucket from the analysis with explicit listing.

Justification: cluster cohesion's 50K-pair populations gave a cosine std of ~0.15 (table line). At 2K pairs, the SE on the bucket mean cosine is 0.15/√2000 ≈ 0.0034, which is small relative to the +0.05 / +0.10 thresholds proposed in §2. Below 2K the SE on the mean inflates above 0.005 and starts to compete with the +0.05 threshold. Above 5K the SE is irrelevant; below 1K is uninterpretable.

**Realistic expectation on bucket coverage:** `is_open` is symbol-skewed — KBONK and ASTER memecoins skew high, BTC and ETH skew low. Pooled-tertile assignment on is_open + per-symbol-tertile on EVR will produce non-uniform bucket fills. The (low EVR × low is_open) cell will likely be BTC/ETH-dominated; the (high EVR × high is_open) cell will likely be alts-dominated. That actually makes the cross-symbol-same-hour pair count IN those cells SMALLER than uniform (because cross-symbol pairs require pairs across different symbols). The 9 buckets where the pair-count floor will most likely fail are the corner cells (1,1), (1,3), (3,1), (3,3). Expect to drop 2–4 of the 9 buckets at the 2K floor.

If only 4–5 buckets survive the floor, the 9-cell analysis is not viable. **Fallback rule: drop to 2×2 (median split on each axis) and require all 4 buckets to clear 2K pairs.** If 2×2 also drops a cell below 2K, declare the diagnostic underpowered and report rather than amend.

**Important pre-commitment:** which buckets clear the 2K floor must be reported BEFORE the cosine measurement is computed, and the bucket-drop list cannot be revised after seeing cosines. Otherwise we are post-hoc selecting buckets that look good. This is a council-5-style pre-registration sub-clause.

---

## 4. NULL outcome (encoder reads per-symbol priors)

Numerical pattern if the encoder is reading per-symbol direction priors and not tape state:

1. **Across all buckets, `diff_b` (as defined in §2) clusters tightly around 0**, with std across buckets < 0.02. Encoder cosines vary across buckets, but PCA cosines vary the same way (because both are picking up the same window-property structure that defines the buckets).
2. **Mean `diff_b` ≤ +0.05** across the 9 (or surviving) buckets.
3. **No single bucket has `diff_b > +0.10`.**
4. **Bonus diagnostic — symbol-stratified within-bucket cosine should be HIGHER than cross-symbol within-bucket cosine in encoder space, by approximately the +0.139 same_symbol-vs-cross_symbol gap from cluster cohesion, in EVERY bucket.** This is the "the encoder is doing the same per-symbol clustering inside every tape-state cell" pattern. If we see this, it is the single cleanest evidence that the encoder is reading symbol identity, not tape state.

Pre-commit verdict: if patterns 1–3 hold, write up as "encoder reads per-symbol priors; the +1pp Gate 1 signal is incidental tape information that PCA also captures linearly."

---

## 5. PASS outcome (encoder reads tape)

Numerical pattern if the encoder is reading tape state:

1. **`diff_b` differs across buckets** with std across buckets > +0.03. Some tape states are encoder-favorable, others are encoder-PCA-equivalent. This differentiation is itself the signature.
2. **At least 2 of 9 (or 1 of 4 in the 2×2 fallback) buckets have `diff_b > +0.10`.**
3. **The high-`diff_b` buckets are concentrated in cells with Wyckoff signatures, NOT the bulk of the distribution.** Specifically I expect (high EVR × high is_open) and (low EVR × high is_open) to be the two encoder-favorable cells, because those are where `is_open` is informative beyond raw volume. The "middle" cells (medium EVR, medium is_open) should be near-zero `diff_b` — those are the windows where neither feature carries Wyckoff signal.
4. **Symbol-stratified within-bucket cosine in the high-`diff_b` buckets should be CLOSER to cross-symbol within-bucket cosine than in the low-`diff_b` buckets.** Translation: in the tape-rich buckets, the encoder has partially overcome its per-symbol clustering; in the tape-poor buckets, it has not. This is the universality-IS-tape-state-dependent pattern.

Pre-commit verdict: if patterns 1–3 hold, write up as "encoder represents tape state, but only on the subset of windows where Wyckoff load-bearing features fire; per-symbol clustering dominates in the bulk."

The MIXED outcome (uniform `diff_b > +0.05`, low std) is Threshold B — encoder reads SOMETHING beyond per-symbol prior, but it does not concentrate on Wyckoff cells. This would be the most-likely real-world outcome. It would say "encoder learned a general tape-flow representation that is not specifically tied to absorption/spring/markup phenomenology." Honest write-up: "encoder represents flow-direction structure (direction proxy via trade_vs_mid, OFI, and slow drift) but does not differentiate Wyckoff phases per se."

---

## 6. Pushback — alternative diagnostic that would adjudicate more cleanly

The proposed diagnostic measures cross-symbol cosine within tape-state buckets. It is good but it has a structural weakness: PCA on flat features will also bucket-stratify, because the bucket-defining features (mean EVR and mean is_open over the window) are themselves columns of the flat feature vector. PCA will pick those columns up. So the PCA-baseline within-bucket cosine will ALREADY be elevated in extreme-bucket cells, simply because PCA encodes the bucket-membership signal.

That means the +0.10 threshold on `diff_b` may be too easy to fail (because PCA gets a bucket-membership lift "for free") AND too easy to pass (because the encoder picks up the same bucket structure linearly via channel-1 features). The PCA baseline is not a clean control for "what is per-symbol-prior-only."

**Cleaner alternative: PCA-on-flat-features-WITHOUT-the-three-Wyckoff-channels.** Train PCA on flat features with `effort_vs_result_*`, `climax_score_*`, `is_open_*` columns dropped (so 83 - ~12 = 71-ish columns). This PCA captures everything BUT the Wyckoff axis directly. If the encoder beats this restricted-PCA within Wyckoff buckets, the encoder is reading tape information NOT linearly available from non-Wyckoff features. That is the cleanest version of the question.

This adds one extra PCA fit + one extra cosine population. Minimal cost.

**Cleaner alternative #2: replace cross-symbol cosine with cross-symbol RETRIEVAL.** For each anchor window, find the top-K nearest-neighbor windows in encoder space cross-symbol. Measure the rate at which the top-K share the same tape-state bucket as the anchor. Compare to PCA top-K bucket-share rate. This is a cleaner "does encoder geometry preserve tape state across symbols" measurement than mean cosine and is less sensitive to the embedding-cone offset. Ranking metric instead of cosine geometry.

**My ranked recommendation:**
1. **Best (most expensive):** Cross-symbol top-K bucket-retrieval rate, on `effort_vs_result × is_open` 3×3 buckets, encoder vs restricted-PCA (Wyckoff features dropped).
2. **Cheaper, still clean:** Within-bucket cosine `diff_b` with restricted-PCA baseline (Wyckoff channels dropped).
3. **As proposed but with bucket fix:** Within-bucket cosine `diff_b` with full-PCA baseline, on `effort_vs_result × is_open` 3×3 buckets. (NOT `effort_vs_result × climax_score` — that one I do not endorse.)

If cost is the binding constraint, option 3 is acceptable. If we have one shot to adjudicate, option 1 is the right experiment.

---

## 7. What I will NOT endorse

- Tertile buckets on `climax_score` at the window level. Empirically the feature does not have the dynamic range. We already learned this with C1/C3/C4.
- Buckets on raw `qty` or `log_total_qty`. Per-symbol normalization is the whole reason we built these features; reverting to raw qty re-introduces the symbol-identity confound directly into the bucket definition.
- A pass/fail threshold based on raw cosine (not bucket-relative). The narrow embedding cone makes raw cosine uninformative without a baseline comparison.
- A diagnostic that does not pre-commit which buckets get dropped and which thresholds adjudicate which verdict. Council-5 will rightly hammer that, and I will agree with their hammering.

---

## 8. Summary table

| Question | My answer |
|---|---|
| Bucket axes | `effort_vs_result × is_open`, NOT `effort_vs_result × climax_score`. Per-symbol tertile on EVR, pooled tertile on is_open. 3×3 with 2×2 fallback. |
| Adjudication metric | `diff_b = (cos_enc_b - cos_enc_global) - (cos_pca_b - cos_pca_global)` on cross-symbol-same-hour pairs |
| Null threshold | `mean_b(diff_b) ≤ +0.05` AND no bucket exceeds `+0.10` AND symbol-stratified within-bucket gap ≈ +0.139 in every bucket |
| Pass threshold | ≥2 of 9 (or 1 of 4) buckets have `diff_b > +0.10`, concentrated in Wyckoff-signature cells (high is_open × extreme EVR) |
| Sample-size floor | 2,000 cross-symbol-same-hour pairs per bucket; pre-commit drop list before cosine measurement |
| Better alternative | Cross-symbol top-K retrieval bucket-share rate, with restricted-PCA (Wyckoff channels dropped) baseline |
| Caveat I want logged | Even a clean PASS at threshold C only proves "encoder reads tape state better than linear baseline within Wyckoff cells." It does NOT prove the encoder reads Wyckoff phenomenology specifically — that would require a labelled Wyckoff probe, which the C1/C3/C4 calibration failure showed we cannot run on this data. |

---

## 9. Final position

The proposed diagnostic is going in the right direction and is the cheapest way to adjudicate the open question. **Two changes are non-negotiable from my seat:**

1. Replace `climax_score` with `is_open` as the second bucket axis. Climax_score does not have empirical dynamic range to tertile; we relearn the C1/C3/C4 lesson if we use it.
2. Adjudicate on `diff_b` (bucket-relative, defeats the cone offset), not raw within-bucket cosine. Threshold +0.05 mean / +0.10 per-bucket / 2K-pair floor.

**One change I strongly recommend:** restricted-PCA baseline (drop Wyckoff feature columns from the flat-features PCA fit). Without this, PCA gets bucket-membership signal for free and the comparison is not a clean test of "does the encoder represent something PCA does not."

If this diagnostic returns Threshold A (null), the writeup conclusion is firm: encoder reads per-symbol priors, the +1pp Gate 1 signal is incidental, no phenomenological claim. If it returns Threshold B, the writeup says "general flow representation, not Wyckoff." Threshold C would partially rescue the tape-reading claim but only conditional on the Wyckoff-cell concentration pattern.
