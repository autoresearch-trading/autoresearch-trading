# Council Review — Market Microstructure Theory (Council-3)

**Reviewer:** Kyle (market microstructure theory)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Date:** 2026-04-01

---

## Summary Judgment

The spec demonstrates serious theoretical grounding. The feature set correctly targets the informed/uninformed distinction at the order-event level, and the is_open framing as a Composite Operator proxy is exactly right. Two gaps are significant: (1) price_impact as defined is not Kyle's lambda — it measures order aggressiveness, not the market maker's updating coefficient — and an explicit lambda estimate is absent from the sequential feature stream; (2) treating all 25 symbols as exchangeable suppresses the cross-symbol variance in information environments that theory predicts should be large. Both gaps are addressable without redesigning the pipeline.

---

## 1. Feature Set: Informed vs. Uninformed Trading

The 16-feature design correctly distinguishes the two trader populations at the order level.

**What the spec gets right:**

- `log_total_qty` + `price_impact` together approximate the Kyle (1985) sufficient statistic for informed order size. Large quantity that walks the book is the canonical informed trader signature — they cannot hide size when information is strong and the deadline is near.
- `is_open` (fraction of fills that are position-opens) is theoretically the single most discriminating feature. In the Kyle multi-period model, informed traders open positions early and accumulate; noise traders open and close with roughly equal frequency. A sustained sequence of high `is_open` values is the sequential signature of the Composite Operator.
- `time_delta` captures urgency, which maps directly to information half-life. As an informed trader's private information approaches resolution, the optimal strategy shifts from patient accumulation to aggressive execution. Shrinking `time_delta` in a directional sequence is the multi-period Kyle model's acceleration pattern.
- `effort_vs_result` (Wyckoff absorption) is theoretically dual to adverse selection: high volume with low price impact means someone is absorbing the flow — a large patient counterparty (the informed side) is absorbing aggressive uninformed flow.
- `delta_imbalance_L1` is the Cont (2014) result — the change in order book imbalance is more predictive than the level because it captures the market maker's updating. This is the book-side manifestation of Kyle's lambda in action.

**What is underspecified:**

- The spec has no explicit feature for the ratio of permanent to transitory price impact. In the Hasbrouck (1991) framework, informed trades produce permanent impact (the price does not mean-revert); noise trades produce transitory impact (the spread eventually closes). None of the 16 features directly captures this ratio. The model must learn this decomposition implicitly from `log_return` and `delta_imbalance_L1` across the event window.
- `is_climax` by construction is a rare joint tail event — near-zero most of the time, which contributes little gradient signal. Consider replacing with a continuous version: `climax_score = min(qty_zscore, return_zscore)` using the same rolling 1000-event window already specified. This preserves the theoretical content while providing a continuous gradient signal.

---

## 2. Kyle's Lambda: price_impact Is the Wrong Proxy

This is the most significant theoretical gap.

**What price_impact measures:** `(last_fill_price - first_fill_price) / mid` measures how far a single order walked the book — order aggressiveness within one event. This is a property of the order itself, not of the market's pricing function.

**What Kyle's lambda measures:** Lambda (λ) is the market maker's price updating coefficient — the expected price change per unit of net signed order flow, estimated from the time series of price changes and signed quantities. Formally: `dp_t = λ * x_t + noise`, where x_t is net signed order flow. Lambda is a property of the information environment at a point in time, not of any individual order.

**Why this matters:** Lambda is the adversarial awareness measure. High lambda means the market is currently populated by informed flow — market makers have widened their pricing response because they are learning from order flow. Low lambda means the tape is currently noise-dominated. This is a regime indicator, not an order-level feature.

**Recommendation:** Add an explicit rolling lambda estimate as feature 17. Compute over a rolling window of the last 50 order events: `λ_rolling = Cov(Δprice, signed_qty) / Var(signed_qty)`. This is already implemented in `prepare.py` at lines 816-827 as `kyle_lambda` for the batch-level pipeline — port that computation to the event-level pipeline. Without an explicit lambda feature, the model must infer the information regime from indirect signals. Providing it directly is strictly theory-aligned and will accelerate convergence on regime-detection patterns.

---

## 3. is_open: Theoretical Significance for Detecting Informed Flow

The spec correctly identifies `is_open` as critical but understates the precise theoretical mechanism.

In the Kyle multi-period model, the informed trader has a known liquidation horizon T. The optimal strategy is to trade at a rate proportional to the remaining information advantage. This means position opening concentrates at the beginning of the informed trader's horizon — when private information is fresh, the informed trader opens aggressively. Noise traders open and close with roughly equal probability at any horizon, as their position changes are driven by portfolio rebalancing.

Therefore a sequence of consecutive order events with high `is_open` fractions, particularly accompanied by directional consistency in `is_buy`, is a direct temporal signature of the multi-period informed trader. The spec's 200-event window capturing ~10 half-lives of `is_open` persistence is theoretically sufficient to distinguish sustained accumulation from noise.

**One addition theory recommends:** Include the running cumulative imbalance of `is_open` direction (open-long minus open-short) as an explicit feature. The cumulative signed open position builds up precisely when the Composite Operator is active. This is the integral of the informed flow signal and gives the model a direct measure of net informed directional commitment over the window, which the per-event fraction alone does not provide.

---

## 4. Multi-Horizon Prediction and Information Revelation Timescales

The 4-horizon design (10, 50, 100, 500 events) is theoretically well-motivated but the multi-task loss weighting requires adjustment.

**Horizon interpretation:** The 10-event (~0.5-1 sec) horizon is dominated by immediate book physics — the order IS the price change. The 500-event (~30-60 sec) horizon tests whether the model detects the onset of an informed trading episode before it is fully priced in. The 100-event (~5-10 sec) horizon is the most theoretically interesting, falling within the typical information half-life window for crypto markets.

**The multi-task loss concern:** Summing binary cross-entropy equally across all 4 horizons gives equal weight to the 10-event prediction (mostly deterministic, dominated by current momentum) and the 500-event prediction (requires genuine information detection). Equal weighting will cause the model to over-optimize the easy short-horizon prediction at the cost of the valuable long-horizon prediction.

**Recommendation:** Use asymmetric loss weighting that reflects the theoretical signal-to-noise gradient: `L = 0.10 * L_10 + 0.20 * L_50 + 0.35 * L_100 + 0.35 * L_500`. This can be treated as a hyperparameter if the linear baseline reveals different horizon signal strengths.

---

## 5. Economic Significance of 52% Accuracy After Costs

**The average-accuracy framing is the wrong question.** Kyle's model implies that informed flow is concentrated in time — a bimodal accuracy distribution is the expected outcome, not uniform 52%. If the model achieves 60% accuracy on 20% of events and 50% on 80%, trading only on high-confidence signals produces economically significant alpha even if average accuracy is only 52%.

**The binding constraint is fee coverage.** For the current fee model (fee_mult=11.0, ~11 basis points round-trip), 52% average accuracy on binary direction requires that winning trades have sufficient magnitude. If mean absolute return at the 100-event horizon is 3 bps and the fee is 11 bps, 52% accuracy produces negative expected value. If mean absolute return during informed episodes is 20 bps, 52% accuracy is highly profitable.

**Recommendation:** Phase 0 label validation should report the conditional distribution — top decile and top quartile of absolute returns at each horizon — not just the mean. If the top decile of 100-event returns exceeds 15 bps, a model achieving 54% accuracy on those events is economically viable regardless of overall accuracy. This framing also connects the accuracy target to the fee structure explicitly, which the current spec does not do.

---

## 6. Symbol Exchangeability: A Theoretical Concern

The spec trains on all 25 symbols jointly with no per-symbol indicator, on the premise that the model should learn universal microstructure patterns.

**The theoretical case is strong:** Kyle's mechanism is universal. The features are designed to be relative (log-normalized quantities, spread-normalized prices), which should make them comparable across symbols.

**The theoretical concern is real:** Information asymmetry is not uniform. BTC and ETH are the most liquid, most efficiently priced crypto markets — information edges are thin and quickly arbitraged. FARTCOIN and PUMP are highly illiquid — lambda is orders of magnitude larger. A model trained jointly is learning two different information regimes under the same architecture. The `log_total_qty` normalization by `median_event_qty` helps, but the dynamics of information revelation at different lambda regimes cannot be fully captured by quantity normalization alone.

**Practical recommendations:**
- Adding the rolling lambda estimate (recommended above in section 2) partially solves this problem — it gives the model direct information about which information regime it is currently operating in.
- Evaluate per-cluster accuracy separately: liquid symbols (BTC, ETH, SOL), mid-tier (AVAX, LINK, UNI), and low-liquidity (FARTCOIN, PUMP, XPL). If accuracy varies systematically by cluster, the exchangeability assumption fails.
- The per-symbol accuracy variance metric in Phase 1 will expose this. Variance above 3% across symbols is a sign that the model is learning symbol-specific characteristics rather than universal microstructure patterns.

---

## 7. Minor Issues

**Feature count discrepancy:** The Input Representation section header states "14 features from 2 data sources" but describes 10 trade features + 6 orderbook features = 16. All subsequent references correctly use 16. The header is a stale value from a prior draft.

**Linear baseline discrepancy:** Phase 0.5 states "200×12=2800" features for the logistic regression flattening. The correct count is 200×16=3200.

**GlobalAvgPool and temporal ordering:** The CNN architecture uses GlobalAvgPool to compress the 200-event sequence to a fixed-length vector. This averages away the temporal ordering information that distinguishes accelerating informed flow (shrinking time_delta, increasing is_open late in the window) from decelerating flow. For a first pass this is acceptable. If the CNN shows signal, replacing GlobalAvgPool with a learned attention-weighted pool over time positions would better capture the Kyle model's acceleration pattern. This is a natural upgrade path before committing to the full Transformer architecture.

---

## Conclusion

The spec is theoretically sound and ready to implement with two recommended additions: (1) an explicit rolling Kyle lambda estimate as feature 17, ported from the existing `prepare.py` implementation at lines 816-827; (2) asymmetric multi-task loss weighting that up-weights the 100-event and 500-event horizons. The is_open feature is correctly designed and theoretically central. The multi-horizon prediction framework aligns well with information revelation timescales. The symbol exchangeability assumption is a real risk — the rolling lambda estimate will partially mitigate it, and the per-symbol accuracy variance metric in Phase 1 will expose it if it remains a problem.
