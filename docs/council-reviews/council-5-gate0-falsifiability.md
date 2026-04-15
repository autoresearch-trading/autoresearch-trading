# Council-5 (Practitioner Skeptic) — Gate 0 Falsifiability Review

**Date:** 2026-04-15
**Reviewer:** council-5

## Smoking gun — already in the JSON

The high raw H10 accuracies are majority-class artifacts. Balanced accuracy reveals the truth:

| Symbol | H10 raw | H10 balanced | Gap |
|--------|---------|--------------|-----|
| 2Z | 0.6208 | 0.5222 | **9.86pp** |
| CRV | 0.6159 | 0.5278 | 8.81pp |
| WLFI | 0.5945 | 0.5133 | 8.12pp |
| XPL | 0.5796 | 0.5166 | 6.30pp |

Every symbol inflating the "14/25 above 51.4%" count at H10 is doing so via label skew. Balanced accuracy for these sits at 51–53% — indistinguishable from a majority-class predictor. **The Gate 0 summary reports raw accuracy at H10/H50/H100, which masks the imbalance entirely.**

## RP vs PCA at the per-symbol level

PCA's mean advantage at H10 = +0.0003 (coin flip). RP *beats* PCA outright on **13/25 symbols at H10** (AAVE, AVAX, CRV, DOGE, ENA, HYPE, KPEPE, LDO, LINK, PENGU, SUI, UNI, WLFI). PCA's marginal aggregate win at H50/H100 (+0.58–0.65pp) is within symbol-level noise and not anchored to a paired statistical test.

## Leakage vectors

### 1. Session-of-day via `_last` statistic block

Flat features 68–84 are last-event values of all 17 features. Two specific leaks:

- `time_delta_last` — inter-event gap at session opens differs systematically from overnight.
- `prev_seq_time_span_last` — for illiquid symbols (2Z: ~56 min per 200-event window), encodes approximate UTC time directly.

Both PCA and RP ingest these identically, explaining the equivalence: both operate in the same session-leaking feature space.

### 2. Embargo adequacy for illiquid symbols

600-event embargo at stride=200 = 3 window-steps = ~3 hours of dead zone on 2Z. Label autocorrelation at H10 (10 events forward) could extend across this for slow-moving symbols. The embargo is sized for BTC, not for 2Z.

## Three null-hypothesis experiments, ranked

1. **Shuffled labels** (run first). True null-hypothesis test: shuffle direction labels within each fold, run the pipeline. If accuracy > 50.5%, there is structural leakage from label construction, window alignment, or fold construction. If this fails, (2) and (3) are uninterpretable.

2. **Majority-class predictor** (run second). Predict training-fold majority class universally. If majority-class matches PCA, Gate 0 is meaningless. Given the balanced-accuracy data, it almost certainly will match on 2Z, CRV, WLFI, XPL.

3. **Per-fold base rate audit** (run third). For each fold-pair (train, test), report positive-class rates and test-fold base rate drift. Diagnostic of mechanism, not gate-setter.

## CNN risk

The CNN sees `time_delta` and `prev_seq_time_span` at **every event**, not just as summaries. Session-of-day signal is denser in the raw sequence than in flat aggregates. If illiquid symbols have morning directional bias, the CNN finds it faster and more confidently than LR. **The primary falsifiability risk: the CNN clears Gate 1 by learning session-of-day patterns, not tape microstructure.**

## Required spec amendments

1. **Gate 0 must use balanced accuracy at ALL horizons, not just H500.** Data already exists in the JSON.
2. **Add majority-class predictor as a mandatory co-baseline.** Gate 0 PCA must exceed majority-class by ≥1pp on 15+ symbols to be meaningful.
3. **Gate 1 threshold reframed as a delta above majority class**, not an absolute 51.4%. Symbols where majority class already scores 55–62% cannot be meaningfully thresholded at 51.4%.
4. **Session-of-day confound check before pretraining.** Train LR with only hour-of-day (4-hour bins) as the feature. If this beats flat-feature PCA on ≥5 symbols, remove `time_delta_last` and `prev_seq_time_span_last` from the 85-dim vector and audit the CNN for session-of-day exploitation.
5. **Gate 2 (CNN vs LR on same features) remains sound** — both models share the same session-of-day exposure, so the delta between them is still meaningful.

## Risk rating

Without these amendments, Gate 0's reference bar is partially a majority-class artifact. The CNN will clear that bar without learning tape microstructure. **This is the most direct falsifiable failure mode: the project declares success at Gates 1-2 because it learned session-of-day, not Wyckoff.**
