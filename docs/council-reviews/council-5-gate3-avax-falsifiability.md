# Council-5 Review: Gate 3 AVAX Falsifiability

**Date:** 2026-04-24
**Reviewer:** council-5 (critical skeptic)
**Inputs:**
- `docs/experiments/step5-gate3-avax-probe.md`
- `docs/experiments/step3-run-2-gate1-pass.md`
- `docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md`

## 1. Is the stride=50 Gate 3 failure a clean falsifier, or does small-n kill the conclusion?

**Verdict:** It is **not** a clean falsifier of "universal tape features," but it **is** a clean falsifier of the specific claim "the encoder transfers to AVAX at a level meeting the pre-registered 51.4% H100 threshold." Be precise about which claim is being tested.

The stride-flip argument in the writeup is directionally correct but the writeup's own binomial SE math is slightly generous in its interpretation:

- **stride=200, n≈120 test, 60 per class:** binomial SE on balanced accuracy ≈ sqrt(0.25/60) ≈ 0.065. 95% CI ≈ ±0.127. Encoder 0.575 is 1.15σ from 0.5 — *not* a pass by any statistical standard once you account for that we ran 6 cells (3 months × 2 horizons) without Bonferroni. Under mild multiple-testing correction, the Feb H100 stride-200 number was never evidence of transfer; it was a lucky cell.
- **stride=50, n≈472 test:** binomial SE ≈ 0.032. 95% CI ≈ ±0.064. Encoder 0.531 is ~1σ above 0.5; PCA 0.548 is ~1.5σ. Both are weak individually; what's load-bearing is the *relative* ordering (encoder < PCA) across 4 of 4 stride-50 cells.

Crucially, stride-50 overlapping windows do NOT give independent observations. With stride=50 and window=200, overlap is 75%. For strongly autocorrelated binary labels at H100/H500 on AVAX, the effective sample size is meaningfully lower than raw n — plausibly 1.5–2× the stride=200 count, not 4×. The writeup's "sqrt(4) = 2× independent" heuristic is in the right ballpark. **So stride=50 is better than stride=200, but it is not a definitive binomial-clean falsifier either.**

**What stride=50 *does* falsify cleanly:** the stride=200 Feb H100 "pass" at 0.575. That number does not replicate at higher density on the same data. The writeup is correct to distrust it.

**What stride=50 *does not* falsify at high confidence:** that encoder balanced accuracy on AVAX is literally below 0.500. It is consistent with AVAX accuracy being ~0.51–0.53 — at or just above chance, but without the +1.9–2.3pp separation seen on the pretrained universe. That is a **weak transfer signal at best, not a clean win and not a clean zero.**

**Bottom line on (1):** The stride flip is not "proof stride=50 is also noise with sign reversed." It is proof that the point estimate at n=120 was inside its own error bar. The stride=50 result is a noisy but real signal that the encoder does not beat PCA on AVAX, which is the relevant inequality for Gate 3's "transfer" claim. But neither result is tight enough to say "the encoder learned nothing on AVAX" — the honest statement is "the encoder did not learn enough useful features on AVAX to beat PCA+LR, against a pre-registered binding threshold, on n=1 held-out symbol."

## 2. Measurement artifacts — is shuffled_pca_lr drifting from 0.500 pathological?

**Verdict:** Not pathological, but the writeup should report bootstrap CIs before claiming the probe is clean. The Apr H500 0.700 is a yellow flag that deserves a 30-minute investigation.

Under the null hypothesis (labels shuffled), balanced accuracy is distributed with mean ≈ 0.500 and SE determined by the smaller test class count per class:

- **Feb H500 stride=200, shuffled=0.566, n≈120:** SE ≈ 0.065. z = 1.02. Two-sided p ≈ 0.31. Within noise for a single cell.
- **Feb H500 stride=50, shuffled=0.526, n≈472:** SE ≈ 0.032. z = 0.81. Fine.
- **Mar H500 stride=200, shuffled=0.540, n≈97:** SE ≈ 0.072. z = 0.56. Fine.
- **Mar H500 stride=50, shuffled=0.484:** z = -0.50. Fine.
- **Apr H500 stride=50, shuffled=0.700, n≈60:** SE ≈ 0.09. z = 2.22. **Two-sided p ≈ 0.026.** With 16 shuffled cells total reported across the run, family-wise you'd expect one at this z purely by chance, but this one is outlier-ish.

**Diagnosis on Apr 0.700:** at n=60 with unbalanced classes at H500 (AVAX April direction labels are likely imbalanced at the 500-event horizon because the sample window is short and drift dominates), `balanced_accuracy_score` becomes high-variance. One minority-class correct prediction swings balanced accuracy by ~0.05–0.10. With a single shuffle seed (`seed=0`), any one draw is noisy.

**Required fix:** run `shuffled_pca_lr` with N=50 shuffles per cell and report mean ± 2σ. If the null band tightens around 0.500, pipeline is clean. If shuffled stays >0.55 on Apr across shuffles, there's a class-imbalance or leakage issue that needs tracing. Also report class prior on test fold for every cell — if AVAX April H500 is 75/25 imbalanced, the whole cell's effective info content is tiny and should be marked as such.

**Compared to Gate 1's shuffled control at 0.504 on Mar (16K windows):** that's the pipeline-clean bar. The Gate 3 shuffled drift is explainable by sample size, not a pathology — but it should be *shown* to be so, not asserted.

**But:** the fact that shuffled PCA+LR hit 0.566 on Feb H500 stride=200 is a caution that the encoder's 0.575 Feb H100 stride=200 "pass" should have been suspect from day one. A probe where the null distribution has a 95% CI of ±0.13 cannot confirm a 0.08 lift.

## 3. Experiments that would convert "inconclusive signal" to "binding falsifier" (ranked by cost-value)

**Rank 1 — Bootstrap CIs on every cell. Cost: 1 hour.** Do this before any other experiment. Running the existing probe with 1,000 bootstrap resamples of the test set gives proper per-cell 95% CIs, which will tell you definitively whether encoder 0.514 on Mar stride=50 is "above 0.514" or "indistinguishable from chance." I suspect every H100 cell's CI includes 0.500, which cleanly validates "Gate 3 fails under pre-registered threshold at the stated confidence level." Without this, the writeup is making eyeball claims about significance.

**Rank 2 — Per-symbol transfer sweep (swap AVAX for other symbols individually). Cost: 2–3 hours.** This is the single highest-information experiment. Run `avax_gate3_probe.py` logic with each of {ASTER, LDO, DOGE, PENGU, UNI} held out *retrospectively* — i.e., pretend that symbol was the held-out one, train a probe on only its held-out-month windows, check encoder vs PCA. If **all 5 surrogates also fail**, the encoder did not learn universal features — kill the universality claim, reframe Gate 3 as informational. If **3/5 pass**, AVAX is an adversarial example and the spec pre-registration got unlucky on symbol choice. If **1/5 passes**, the encoder has weak generalization but to specific symbol-classes. This is the only experiment that disambiguates "AVAX is weird" from "encoder doesn't generalize."

*Caveat: these symbols' embeddings were seen during pretraining, so this is not a true held-out test — it tests whether the fitted LR transfers when trained only on the surrogate symbol's labels. It's informative about LR-on-out-of-distribution-label-fit, not about pure representation universality. That caveat matters and must be in the writeup.*

**Rank 3 — Measure cross-symbol SimCLR cluster tightness on the 6 liquid anchors. Cost: 2 hours.** The "did SimCLR actually learn cross-symbol invariance" question is a prior for (1) vs (2). If cosine similarity between same-hour, same-date embeddings from BTC/ETH/SOL/BNB/LINK/LTC is high (e.g., mean cos > 0.6) and they cluster together in UMAP, the encoder did learn some cross-symbol invariance within the trained set — and AVAX's failure is then a genuine generalization gap. If cosine similarity is ~0 (anchors don't cluster together), cross-symbol SimCLR never kicked in, and AVAX failure is *expected* — the universality claim was unearned from the start. This is the single cleanest test of whether the encoder was ever positioned to generalize.

**Rank 4 — Final-epoch checkpoint probe. Cost: 30 min.** Useful but unlikely to change the verdict. Council-6's prior says MEM minimum = representation purity. I agree. If final-epoch transfers better than best-MEM, that would be a surprising finding but also means MEM is not measuring what we thought it was, which is its own problem. Low expected value — run it as a side check, don't build the case around it.

**Priority order:**
1. Bootstrap CIs on current results (1 hr) — **required** before any spec amendment.
2. Cluster tightness on 6 liquid anchors (2 hr) — tells you whether transfer was possible in principle.
3. Per-symbol surrogate sweep (3 hr) — converts n=1 to n=5 on the transfer claim.
4. Final-epoch probe (30 min) — cheap sanity.

Total cost: ~6.5 hours. Do not proceed to spec amendment without (1) at minimum. Do not proceed to Step 4 without (2).

## 4. Should you amend the spec now or do triage first?

**Verdict:** Triage first. Specifically, bootstrap CIs + cluster tightness check (~3 hours total) before touching the spec. Do NOT amend Gate 3 based on the current writeup alone.

**Why not amend now:** The writeup's conclusion ("Gate 3 fails at stride=50") is probably right, but the statistical rigor backing it is thin. If you amend the spec to "Option C: reframe Gate 3 as informational" and later someone points out that bootstrapping shows encoder CIs overlap PCA CIs on every cell (which is my prediction), the amendment reads as retroactive rationalization rather than principled response. That is exactly the p-hacking pattern council-5 exists to prevent. **Amending a pre-registered gate after it fails, on the basis of a visually-assessed small-n result, is the classic move of "my model didn't hit the bar, so I moved the bar."** The whole point of pre-registration is that you do NOT do this.

**Why triage is cheap and exonerates or condemns:** Bootstrap CIs take an hour. If they show encoder CI is clearly below PCA CI on every cell, the amendment becomes "Gate 3 failed at pre-registered threshold with 95% CI non-overlap — reframe per council-agreed Option C." That reads as science. If they show encoder and PCA CIs mostly overlap, the amendment becomes "Gate 3 is underpowered at n=1 symbol with ~480 test windows; spec requires broader held-out set." That also reads as science. Either way you earn the amendment.

**The cluster-tightness check is the deeper point:** If the 6-liquid-anchor SimCLR cluster cohesion is weak (mean same-hour cross-symbol cos < 0.3), then **AVAX failure was overdetermined from the start** and the spec's Gate 3 formulation was unfair to this training config — cross-symbol SimCLR on only 6 anchors out of 24 training symbols was never going to produce universal features robust enough to transfer to an unseen symbol. That's a design finding, not a model-quality finding, and it belongs in the amendment rationale.

**Most-honest outcome:** After the two triage experiments, you will probably land at a version of Option C ("representations capture meaningful tape features on the pretrained universe; universal-across-symbols is a stronger claim this training run was not designed to test"), but grounded in measurements rather than eyeballed tables. This is the cheap, falsifiable, non-p-hacking path.

**Do NOT proceed to Step 4 fine-tuning without resolving this.** Fine-tuning on a representation that doesn't transfer cross-symbol is not a bug per se (the fine-tuned model lives on the pretrained universe), but you need to know *why* Gate 3 failed before you amend "universal" claims out of the spec. The scientific cost of doing fine-tuning while the transfer story is ambiguous is that Step 4's "it works" becomes indistinguishable from "it works because each symbol has its own learned features" — which is the H1 hypothesis in your writeup and which the current evidence cannot distinguish from H2/H3.

## Red flags noted

- **Single-seed shuffled control.** `seed=0` for the shuffled_pca_lr predictor means the 0.700 on Apr H500 is one draw from a noisy distribution. Running N=50 shuffles per cell is cheap and mandatory for reporting a clean null.
- **No per-cell bootstrap CIs in the writeup.** Reporting 6+ point estimates with no error bars and claiming "encoder ahead on 0/6 cells" as evidence of failure is the same epistemology that produced the original stride=200 "pass." It happens to land on the right answer here, but the method is wrong.
- **Symbol-ID probe grew 0.54 → 0.67 during training** (per Gate 1 writeup). **This is a smoking gun for H1 (symbol-specific features).** The encoder progressively learned to distinguish symbols during pretraining. Cross-symbol transfer was being actively eroded by the training objective. The Gate 3 failure is *consistent* with this trajectory and should have been predictable from the Step 3 run-2 monitoring.
- **Class-imbalance reporting missing.** For every cell, the writeup should state the test-fold class prior. Balanced accuracy on a 75/25 split with n=60 is unstable — it's not reporting the same quantity as balanced accuracy on a 50/50 split with n=472.

## TL;DR

The stride=50 Gate 3 failure on AVAX is **directionally correct but statistically thin** — it falsifies the stride=200 "pass" cleanly, but by itself it does not establish non-transfer with tight CIs, and the ascending symbol-ID probe (0.54→0.67) during pretraining already predicted this failure via H1 (symbol-specific features). **Run bootstrap CIs + 6-liquid-anchor cluster cohesion (~3 hr) before amending the spec** — amending a pre-registered gate based on eyeballed small-n tables is precisely the p-hacking pattern the gate structure exists to prevent, and the triage is cheap enough that there is no defensible reason to skip it.
