# Council Review: Financial ML Methodology (Lopez de Prado)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Reviewer:** Council-1 (Financial ML Methodology — AFML, 2018)
**Date:** 2026-04-01

---

## 1. Multiple Testing Risk: 4 Horizons x 25 Symbols = 100 Hypothesis Tests

This is the most serious methodological risk in the spec and it is not acknowledged anywhere in the document.

The spec evaluates 4 prediction horizons (10, 50, 100, 500 events) across 25 symbols simultaneously and uses accuracy > 52% as the success threshold. The success criterion is stated as "accuracy > 52% consistently across 20+ symbols at any horizon." The phrase "at any horizon" is where the multiple testing problem lives.

When you test 4 horizons independently and declare success if any one of them clears 52%, you are running 4 correlated hypothesis tests per symbol and 100 tests total. Under no-signal conditions, the probability that at least one horizon exceeds the threshold by chance is substantially higher than the single-test probability. For 4 independent tests at alpha=0.05, the family-wise error rate is 1 - (0.95)^4 = 18.5%. With 25 symbols this compounds further.

**Required corrections:**

1. **Deflated Sharpe Ratio (DSR)** must be computed once accuracy translates to a Sortino/Sharpe estimate in Phase 2. DSR adjusts for the number of trials N, the skewness and kurtosis of returns, and the non-normality of the backtest distribution. Without it, the reported Sharpe in Phase 2 is optimistically biased by an unknown amount.

2. **Bonferroni or Holm correction** must be applied to the 4-horizon accuracy tests. If testing at alpha=0.05 significance, each individual test must clear alpha/4 = 0.0125. Equivalently, the 52% threshold must be recalibrated: what sample size N gives a two-sided binomial test significant at alpha=0.0125 with power 0.80?

3. **Track trial count** from the first experiment forward. The spec already notes the Mar 5-25 window "has been used for 20+ experiments" on the main branch. This means the test set is already contaminated with at least 20 trials. Every accuracy report against that window is biased upward. The fresh April data suggestion is correct and must be treated as the definitive, single-use test set — not a running leaderboard.

4. **The multi-task loss exacerbates this.** The model is trained on the sum of cross-entropies across all 4 horizons. This means the model is implicitly tested on all 4 horizons simultaneously during training. The hyperparameter search will gravitate toward configurations that perform on the easiest horizon, and that horizon's reported accuracy will be upward biased.

**Recommendation:** Designate one primary horizon before any training begins. Report all four, but power the statistical test and the go/no-go decision on the pre-designated horizon only. The other three are exploratory and must be labelled as such.

---

## 2. Walk-Forward Validation Design: 120d Train / 20d Test

The single walk-forward split described (train on first 120 days, test on next 20 days, rotate forward) is a step in the right direction but has structural weaknesses.

**What is missing:**

**a) Number of folds is unspecified.** With 160 days total, a 120d/20d sliding window produces at most 2 non-overlapping folds (days 1-120 train + 121-140 test; days 21-140 train + 141-160 test). Two folds provide almost no statistical power. The current v11b baseline ran 4-fold walk-forward and found std=0.220 on a mean of 0.261 — that variance, across only 4 folds, demonstrates how little confidence a 2-fold setup yields.

**b) No embargo zone.** When the training window ends on day 120 and the test window starts on day 121, the last sequences in training and the first sequences in test share label-lookahead overlap. A 200-event window ending at the last training timestamp will have its label computed from events that extend into the test period. Specifically, at the 500-event forward horizon, a sequence sampled near the train/test boundary will peek 500 events — up to 30-60 seconds — beyond that boundary. The spec must impose an embargo of at least max_horizon events (500 events, ~1 minute of BTC tape) between the end of training data and the start of test data.

**c) CPCV is superior.** Combinatorial Purged Cross-Validation (AFML Chapter 12) generates all C(T, k) combinations of k test folds from T available periods, runs them in parallel, and produces a distribution of backtest paths rather than a single point estimate. With 160 days and k=20-day folds, CPCV generates substantially more paths than walk-forward, giving a tighter confidence interval on the mean accuracy and making the DSR calculation tractable. The computational overhead is justified given the H100 is already in use.

**Recommendation:** Implement at minimum a 4-fold purged walk-forward with 500-event embargo zones at each fold boundary. CPCV with k=4 or k=5 folds is strongly preferred for the final evaluation.

---

## 3. Information-Driven Bars: Are Non-Overlapping 200-Event Windows Optimal?

The spec uses non-overlapping 200-event windows as the sampling unit. This is a fixed-count bar — a known suboptimal choice from an information theory standpoint.

**The problem with fixed-count event bars:**

200 consecutive order events during a BTC liquidation cascade carry dramatically more information than 200 events during a Sunday afternoon consolidation. The label distribution (up/down) is not stationary across these two regimes. A model trained on a mixture of high-information and low-information samples will be dominated by the low-information periods (which are far more common) and will underfit the high-information periods — precisely where edge exists.

**Dollar bars are more appropriate here.** Each bar represents a fixed dollar volume of trading activity (e.g., $10M notional for BTC). Dollar bars equalize the information content per sample because the market microstructure literature (Kyle 1985, Admati-Pfleiderer 1988) establishes that price discovery is driven by dollar volume, not event count. A $10M bar during a quiet period contains ~3000 trades; a $10M bar during a crash contains ~50 trades. The crash bar is information-dense; the quiet bar is noise-dense. Fixed-event bars treat them identically.

**Practical mitigation given the spec's design:** The spec's use of order events (not raw trades) already partially addresses this — grouping same-timestamp fills into one event removes pure matching-engine noise. The `seq_time_span` feature (feature 10) provides the model with a signal about whether it is in a fast or slow tape, which is a reasonable runtime approximation. However, this is a post-hoc correction, not a solution to the sampling problem.

**Recommendation:** At minimum, after the baseline CNN is trained with fixed-event bars, run one experiment with dollar bars at a notional threshold calibrated so the median bar count per day equals ~300 (matching the current sample count). This is a controlled comparison. If dollar bars show higher accuracy or lower variance across symbols, adopt them as the default.

---

## 4. Statistical Rigor of the 52% Accuracy Threshold

The 52% success criterion is presented without statistical justification. This is insufficient.

**Required analysis before any experiment begins:**

Given the expected number of samples per symbol per test fold (20 days × 300 samples/day = 6,000 samples), the 95% confidence interval around 50% accuracy under the null hypothesis (no signal) is:

50% ± 1.96 × sqrt(0.5 × 0.5 / 6000) = 50% ± 1.27%

This means 52% exceeds the null by approximately 1.57 standard errors. The corresponding p-value (one-sided) is approximately 0.058 — just above the conventional 0.05 threshold. **52% on a single 20-day test fold is not statistically significant at alpha=0.05.**

To achieve significance at alpha=0.05 with 80% power, the required sample size for detecting a 2% effect (50% vs. 52%) is approximately:

N = (z_alpha + z_beta)^2 × p(1-p) / delta^2 = (1.645 + 0.842)^2 × 0.25 / 0.0004 ≈ 3,880

6,000 samples per fold is sufficient for a single symbol at a single horizon — barely. But once Bonferroni correction is applied across 4 horizons (requiring alpha=0.0125), the critical value rises and 52% on 6,000 samples does not clear the bar. The spec needs to state the minimum detectable effect at the corrected significance level, or increase the test fold size.

**The 53% "best horizon" target** has better statistical footing but still requires reporting the PSR (Probabilistic Sharpe Ratio) when converting to trading performance. PSR < 0.95 means insufficient confidence in the Sharpe estimate, regardless of what accuracy the model achieves.

**Recommendation:** Compute and report PSR alongside Sortino in Phase 2. State the minimum detectable effect and required sample size in the spec before any training begins. Report exact N per symbol per fold so readers can compute their own confidence intervals.

---

## 5. Phase 0 Base Rate Validation Methodology

The Phase 0 methodology is correct in spirit but has a gap: it computes the base rate without controlling for the label construction boundary.

**The issue:** When computing "next-100-event direction" for a sample ending at position i, the label uses events [i+1, i+100]. If samples are drawn non-overlappingly every 200 events, the label horizon (100 events) is half the sample window. This means consecutive non-overlapping samples share no label overlap — good. But if sequence length is swept to {100, 200, 500}, a 500-event window with a 100-event label means the last 100 events of the input window and the label window are nearly contemporaneous. Check whether any of the 16 features computed on the input window use information from the label period.

Specifically, `is_climax` uses a rolling 1000-event sigma. At the end of a 200-event input window, the rolling sigma is computed from the 800 events preceding the window plus the 200 events in the window. For the last events in the window, the rolling sigma is fine. But for the 500-event label horizon: if `is_climax` is recomputed for any label-period events, it may include sigma estimates from events after the label timestamp. This is not currently a risk (the label is scalar, not a sequence of features) but becomes a risk if gradient attribution (Step 4) retroactively analyzes model attention on label-period events.

**The base rate itself:** The spec's threshold of 50 ± 0.5% as the noise criterion is reasonable. However, mean absolute return < 2 bps as the secondary criterion requires clarification: is this 2 bps per order event (in which case it's extremely small — BTC tick is ~$0.10 on $68K = 0.15 bps) or 2 bps over the full 100-event horizon? The spec must specify units explicitly.

**Recommendation:** Clarify the 2 bps threshold (per event vs. per horizon). Add a check in `test_label_signal.py` that verifies label computation does not use any data from the label period itself.

---

## 6. Additional Methodological Red Flags

**a) Logistic regression on 3,200 features with 6,000 samples is underdetermined.**
The Phase 0.5 linear baseline flattens (200, 16) to 3,200 features and fits logistic regression per horizon. With 6,000 training samples and 3,200 features, the feature-to-sample ratio is 0.53 — the model is barely identified. Logistic regression will overfit without strong regularization (L2/L1). The spec must use `LogisticRegression(C=0.01)` or equivalent; the default `C=1.0` in sklearn will overfit on this problem. Report training vs. test accuracy separately, not just test accuracy.

**b) The training loss sums cross-entropies across 4 horizons equally.**
The 10-event and 500-event horizons carry fundamentally different amounts of signal. Summing them with equal weight means the noisiest horizon (10-event) may dominate gradient updates by providing large, random loss gradients. Consider ablating with weighted horizon loss or training separate models per horizon for the baseline comparison.

**c) No mention of label overlap between adjacent training samples.**
The spec draws 300 non-overlapping samples per day (200-event windows). Adjacent windows do not overlap in input. But the labels for window i and window i+1 overlap: window i's 500-event label horizon extends 500 events forward, while window i+1 starts 200 events later. The label for window i covers events [200, 700] and the label for window i+1 covers events [400, 900]. Events [400, 700] appear in both labels. This serial correlation in labels inflates the effective sample size and makes the accuracy confidence intervals too narrow. Use purging: exclude from training any sample whose label period overlaps with a test sample's input window.

**d) The "universal model" claim requires out-of-symbol testing.**
The spec trains on all 25 symbols jointly and claims to learn universal patterns. But the success criterion (accuracy > 52% on 20+ symbols) is measured on the same 25 symbols that were in training — just at a later time period. A truly universal model should be tested on a held-out symbol (e.g., exclude FARTCOIN entirely from training, test exclusively on FARTCOIN). Without this, the model may be learning symbol-specific distributional features that generalize across time but not across assets.

---

## Summary

The spec is methodologically ambitious and the sequential design (Phase 0 → linear baseline → neural network) is sound in structure. The critical gaps are: (1) no multiple testing correction for 100 simultaneous hypothesis tests, (2) insufficient statistical power of the 52% threshold after Bonferroni correction, (3) missing embargo zones in walk-forward splits allowing label-lookahead at fold boundaries, (4) no plan for Deflated Sharpe Ratio in Phase 2, and (5) logistic regression baseline will overfit with default regularization on 3200 features/6000 samples. Fix these before running any experiment or the results will not be interpretable.
