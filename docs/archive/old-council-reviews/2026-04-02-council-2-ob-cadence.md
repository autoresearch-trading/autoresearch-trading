# OB Cadence Impact Analysis: Order Flow Features at 24s Snapshot Resolution

**Reviewer:** Council-2 (Rama Cont — Order Flow Microstructure)
**Date:** 2026-04-02
**Triggered by:** data-eng-13 finding that OB snapshot cadence is ~24s median, not ~3s as spec assumed
**Reference:** `docs/council-reviews/2026-04-02-spec-review-data-eng-13.md`

---

## Executive Summary

The 24s snapshot cadence does not fundamentally undermine the orderbook feature set, but it does require three concrete changes: (1) the Cont 2014 OFI formula must use the piecewise price-level-change implementation, not the naive delta-notional; (2) `cum_ofi_20` should be reduced to 5 snapshots (~120s lookback) as a default with a note to sweep 3/5/10; (3) `kyle_lambda` should be retained with its variance guard but its interpretation must shift from \"event-by-event regime indicator\" to \"snapshot-triggered regime indicator\" - the feature is meaningful but sparse. The static book features (`log_spread`, `imbalance_L1/L5`, `depth_ratio`, `trade_vs_mid`, `book_walk`) operate correctly under staleness as long as the model is given an implicit staleness signal, which `delta_imbalance_L1 == 0` already provides. The spec's training augmentation (OB feature dropout with p=0.15) also partially addresses this - the model is already being trained to predict without fresh book data.

---

## Point 1: `cum_ofi_20` Window Size

**Recommendation: Reduce to 5 snapshots. Sweep {3, 5, 10} after baseline.**

### The Problem

At ~24s median cadence, 20 snapshots covers approximately 480 seconds (~8 minutes). The spec intended roughly 60 seconds of OFI lookback. The economic meaning has shifted by 8x.

The key question is: what is the right OFI lookback window relative to the prediction horizons?

The primary prediction horizon is 100 events forward. At the measured BTC event rate, 100 events corresponds to roughly 300 seconds (~5 minutes) based on the data engineer's finding that a 200-event window spans ~605s median (so 100 events ~ 302s). The secondary prediction horizon (500 events) corresponds to roughly 25 minutes.

### Theoretical Framework

In Cont et al. (2014), OFI was tested with a 1-minute lookback window against 1-minute price changes. The predictive relationship was strongest when the OFI lookback roughly matched the prediction horizon. The logic is that accumulated order flow imbalance over a period of length T predicts price movement over the same period T - the market maker is forced to update prices to clear the accumulated directional pressure.

At our prediction horizons:
- 10 events ~ 30s: OFI lookback of ~60s (2-3 snapshots at 24s) is appropriate
- 50 events ~ 150s: OFI lookback of ~120s (5 snapshots) is appropriate
- 100 events ~ 300s: OFI lookback of ~120-240s (5-10 snapshots) is appropriate
- 500 events ~ 1500s: OFI lookback of ~480s (20 snapshots) may actually be appropriate for this horizon only

The primary horizon (100 events, ~300s) argues for 5-10 snapshots. The current 20-snapshot window at 480s is longer than the primary prediction horizon, which is backwards - it violates the Cont et al. matching principle.

### Recommendation

**Set `cum_ofi_20` to `cum_ofi_5` (5 snapshots, ~120s lookback) as the implementation default.**

This provides ~120s of OFI history, which brackets the primary 100-event prediction horizon of ~300s from below. The slight asymmetry (lookback shorter than horizon) is correct: OFI accumulated over the past 2 minutes predicting whether price continues in that direction over the next 5 minutes is the forward-looking version of Cont's same-direction test.

The feature name in the spec should be updated to `cum_ofi_5` but the variable name in code can remain `cum_ofi` with the window as a hyperparameter. After the linear baseline validates signal existence, sweep {3, 5, 10} snapshots. Do not sweep 20 - that is the wrong economic regime for this feature at this cadence.

A note on the 500-event horizon: if the model shows strong signal at 500 events, revisiting a longer OFI lookback (10-20 snapshots) for that head specifically would be worth exploring. But since horizon-100 is the primary metric, optimize for it.

---

## Point 2: `kyle_lambda` Viability at 24s Snapshot Resolution

**Recommendation: Retain with modified interpretation. Add staleness count as secondary signal. Do NOT replace with cumulative order flow - that is already captured by `cum_ofi_5`.**

### The Data Engineer's Concern

At ~10.6 events per snapshot, a rolling 50-event window contains only ~4-5 non-zero Δmid values (9.4% of events have non-zero Δmid because that is when a new snapshot arrives). The concern is that `Cov(Δmid, signed_notional)` estimated from 4-5 effective observations is statistically unreliable.

### My Assessment: The Concern Is Real But Manageable

The data engineer is correct that the estimator is sparse. Let me quantify:

With n=5 effective observations (non-zero Δmid pairs), the standard error of the OLS slope estimator is:
```
SE(lambda) = sqrt(Var(epsilon) / (n * Var(signed_notional)))
```

At n=5, the 95% confidence interval on lambda is approximately ±2 standard errors. This is wide. However, the feature is still doing work: even a noisy estimate of whether the market is in a high-lambda or low-lambda regime has predictive content. A constantly-zero lambda (would occur if the variance guard triggers, i.e., `Var(Δmid) < 1e-20`) carries the information \"book hasn't moved in the last 50 events\" which is itself a regime signal.

**The critical guard:** The spec already specifies `where(var > 1e-20, cov/var, 0.0)` and the data engineer confirms this is in the existing `prepare.py`. This guard will trigger for the majority of rolling windows when fewer than 2 distinct Δmid values occur. The feature will be zero (stale book) approximately 60-70% of the time after the guard, and only active when the book has moved 2+ times in the last 50 events.

### What kyle_lambda Actually Measures in This Setting

At 24s snapshots, `kyle_lambda` is no longer a per-event regime indicator. It becomes: **how responsive was price movement to signed order flow over the last ~470 seconds (50 events at current rate), measured at the granularity of book updates?**

This is actually a valid and distinct signal from `cum_ofi_5`. They capture different things:
- `cum_ofi_5` asks: \"what was the net order flow imbalance at L1 over the last ~120s?\"
- `kyle_lambda` asks: \"over the last ~470s, when book prices moved, was that movement correlated with net signed order flow?\"

A high `kyle_lambda` with near-zero `cum_ofi_5` would indicate: the market has been responding to informed flow historically, but there is no current directional OFI pressure - useful for asymmetric confidence adjustment. A near-zero `kyle_lambda` with high `cum_ofi_5` would indicate: OFI is building but the market maker hasn't started updating yet - this is early-stage accumulation.

### Implementation Note

The existing `prepare.py` implementation (lines 816-827) should be ported with minimal changes. The key difference: use `Δmid` between consecutive events (which is 0 whenever two events share the same snapshot) rather than between consecutive snapshots. This is what the spec already specifies. Do NOT recompute on a per-snapshot basis - that would conflate the lambda estimate with a pure OFI measure.

The guard `where(Var(Δmid) > 1e-20, Cov/Var, 0)` is essential. When this triggers (zero output), the model is receiving a valid piece of information: \"the book has not moved recently.\" The model can learn to condition on this.

### Against Replacement

Several replacement options have been discussed informally:
- Replace with \"signed trade imbalance over 50 events\" - this duplicates `cum_ofi_5` at the trade level. Not adding information.
- Replace with \"book update frequency\" - this is interesting but is already implicit in `delta_imbalance_L1` (if `delta != 0`, a book update just arrived).
- Drop entirely and add a 19th feature - the spec is at 18 features deliberately. Losing one orderbook feature means adding something else, and there is nothing clearly better than a noisy regime indicator in the current set.

**Verdict: Retain `kyle_lambda` as specified. Accept that it will be sparse and occasionally unreliable. The model's BatchNorm input layer will handle scale, and the CNN's dilated architecture will learn to downweight sparse features naturally through its weight learning.**

---

## Point 3: `delta_imbalance_L1` Informativeness at ~90% Zero Rate

**Recommendation: Keep as-is. The carry-forward is correct. The sparsity is a feature, not a bug.**

### Analysis

The spec already acknowledges sparsity from the 3s snapshot assumption (\"~95% zero\"). At 24s, the actual sparsity is ~90.6% zero (not worse, because events per snapshot = 10.6, so 1/10.6 = 9.4% non-zero). The carry-forward approach means the non-zero events are exactly when a new snapshot arrived.

The carry-forward logic is: every event between two snapshots carries the delta from the most recent book change. So when snapshot k arrives with `imbalance_L1_k != imbalance_L1_{k-1}`, all 10.6 events between snapshots k-1 and k will have the same `delta_imbalance_L1 = imbalance_L1_k - imbalance_L1_{k-1}`.

This is more informative than it looks: the model sees a constant non-zero delta for ~10 consecutive events after each book change. The sequence pattern is:
```
... [0, 0, 0, ..., 0] [delta_k, delta_k, ..., delta_k] [0, 0, ..., 0] ...
```

where the block of `delta_k` values spans the 10-event window following each snapshot. The CNN's dilated receptive field (RF=253) can absolutely learn to interpret this block structure. In fact, the block pattern is more learnable than a single non-zero spike would be.

### What the Model Learns

The model learns: \"when I see a sustained block of identical non-zero delta_imbalance_L1 values for ~10 events, a book update just arrived, and its direction tells me whether bid-side or ask-side pressure was refreshed.\" Empirically in Cont et al. (2014), the sign of `delta_imbalance_L1` at the most recent book update is one of the strongest predictors at the 1-5 event horizon. The model has this information.

### Day Boundary Pre-Warm

The spec already handles this (pre-warm from prior day's last snapshot, or 0.0 for first calendar day). No change needed.

---

## Point 4: `book_walk` (Feature 7) and `trade_vs_mid` (Feature 15) Under 24s Staleness

**Recommendation: Keep both as-is. The staleness is bounded and the spec's training augmentation mitigates it.**

### Staleness Quantification

At ~24s max inter-snapshot interval on BTC, the mid-price can drift. For BTC trading around 111,000, a 24s period of high volatility might move the price by 0.05-0.3% (~$55-$330). This is material.

For `book_walk`: `abs(last_fill - first_fill) / max(spread, 1e-8 * mid)`. The spread in the denominator is from the stale snapshot. If the spread has widened (as it typically does during volatile moves), the stale spread will underestimate the current spread, causing `book_walk` to be upward biased. In the worst case (volatile trending market), the stale spread is 50% narrower than the current spread, meaning `book_walk` reads as twice as aggressive as it actually is.

For `trade_vs_mid`: `(event_vwap - mid) / max(spread, 1e-8 * mid)`. If the mid has trended away from the stale snapshot mid by $100 in a bull run, an event that filled at the actual ask will appear to have filled far above the stale mid. This shifts the feature value upward for buys (appears aggressive) and downward for sells during uptrends.

### Why This Does Not Require a Fix

1. **The signal is still present.** Despite staleness, the relative ordering of `book_walk` values and `trade_vs_mid` values is preserved. A large `book_walk` on a stale snapshot still means a relatively aggressive order. The staleness adds noise but does not flip the sign of the signal.

2. **The model receives an implicit staleness indicator.** `delta_imbalance_L1 == 0` is a reliable indicator that the snapshot is stale. The CNN can learn \"when delta_imbalance is zero for many consecutive events, discount book_walk and trade_vs_mid.\" This is exactly the kind of conditional pattern a dilated CNN excels at.

3. **Training augmentation already addresses this.** The spec includes \"orderbook feature dropout: zero features 11-18 with p=0.15.\" This trains the model to predict under complete book absence, which is more extreme than staleness. A model trained to handle full book dropout will be robust to book staleness.

4. **Cross-symbol consistency is preserved.** Both features are normalized by spread, so the staleness affects them consistently across symbols. No cross-symbol bias is introduced.

### One Targeted Improvement

If the implementation allows it (without adding a 19th feature), the integer count of events since the last snapshot update could be encoded implicitly by appending `events_since_last_ob_update = log(n + 1)` as a 19th feature. This would let the model precisely discount stale book features. However, this is an enhancement, not a requirement. The current 18-feature set handles staleness adequately for the prototype stage.

---

## Point 5: Cont 2014 OFI Formula — Piecewise Implementation Required

**Recommendation: BLOCKING. The piecewise price-level-change formula must be implemented. The naive delta-notional formula is materially wrong at 24s cadence.**

### Why the Naive Formula Fails

The naive implementation the spec describes:
```
OFI_t = (bid_notional_L1_t - bid_notional_L1_{t-1}) - (ask_notional_L1_t - ask_notional_L1_{t-1})
```

This assumes the best bid and best ask prices are the same between consecutive snapshots. At 24s intervals, this assumption fails frequently. The data engineer observed a concrete example: `111235 → 111166` for the best bid between two snapshots.

When the best bid falls from 111235 to 111166:
- The old bid quantity at 111235 is no longer at the best bid. It has effectively been withdrawn (either cancelled or the market moved through it).
- The new best bid at 111166 has its own quantity.

The naive formula computes `new_qty * new_price - old_qty * old_price`, which could be positive (spuriously suggesting buying pressure) even when the bid has dropped (selling pressure signal). This is a sign-error in the OFI during trending markets - precisely when OFI has the most predictive value.

### The Correct Cont 2014 Piecewise Formula

As correctly stated by the data engineer:

**For bid side:**
- If `best_bid_price_t > best_bid_price_{t-1}`: `delta_bid = +bid_notional_L1_t` (price moved up - new buyers appeared at a higher level; interpret as full positive contribution)
- If `best_bid_price_t == best_bid_price_{t-1}`: `delta_bid = bid_notional_L1_t - bid_notional_L1_{t-1}` (same price, normal delta)
- If `best_bid_price_t < best_bid_price_{t-1}`: `delta_bid = -bid_notional_L1_{t-1}` (price moved down - the old bid was consumed/cancelled; full negative contribution)

**For ask side (mirror logic):**
- If `best_ask_price_t < best_ask_price_{t-1}`: `delta_ask = +ask_notional_L1_t` (asks compressed - sellers moved in)
- If `best_ask_price_t == best_ask_price_{t-1}`: `delta_ask = ask_notional_L1_t - ask_notional_L1_{t-1}`
- If `best_ask_price_t > best_ask_price_{t-1}`: `delta_ask = -ask_notional_L1_{t-1}` (asks withdrew)

**OFI_t = delta_bid - delta_ask**

At 3s cadence, price levels changed infrequently (maybe 10-20% of snapshots), making the naive formula approximately correct. At 24s cadence, price levels change in the majority of snapshots. The naive formula is not a good approximation here.

### Frequency of Price-Level Changes at 24s Cadence

Based on the data engineer's observation (111235 → 111166 in one snapshot), and given BTC's volatility (~$100/hour typical), a 24s interval will on average see the best bid/ask move by approximately $67. Given BTC ticks at ~$1 for this price level, the best bid changes in most 24s windows during active trading. I would estimate 60-80% of snapshot pairs will have a best bid/ask price change during normal market conditions, rising to near 100% during trending periods.

This means the naive formula has the wrong sign for OFI in the majority of snapshots during trending markets. Since `cum_ofi_5` is a sum of 5 consecutive OFI values, and trending markets are exactly when OFI is most predictive, using the naive formula would produce a `cum_ofi_5` that is anti-correlated with price direction during trends - the opposite of what the feature should measure.

**This is a correctness-level error that will cause the feature to actively hurt model performance on trending symbols (BTC, ETH, SOL in trending sessions).** It must be fixed before any feature computation begins.

### Implementation

The piecewise formula requires storing `best_bid_price_{t-1}` and `best_ask_price_{t-1}` alongside the notional values. This adds ~2 floats per snapshot to the running state, which is negligible. The implementation:

```python
def compute_ofi(best_bid_price_prev, best_bid_notional_prev,
                best_ask_price_prev, best_ask_notional_prev,
                best_bid_price_curr, best_bid_notional_curr,
                best_ask_price_curr, best_ask_notional_curr):
    # Bid side
    if best_bid_price_curr > best_bid_price_prev:
        delta_bid = best_bid_notional_curr
    elif best_bid_price_curr == best_bid_price_prev:
        delta_bid = best_bid_notional_curr - best_bid_notional_prev
    else:  # price fell
        delta_bid = -best_bid_notional_prev
    
    # Ask side
    if best_ask_price_curr < best_ask_price_prev:
        delta_ask = best_ask_notional_curr
    elif best_ask_price_curr == best_ask_price_prev:
        delta_ask = best_ask_notional_curr - best_ask_notional_prev
    else:  # price rose
        delta_ask = -best_ask_notional_prev
    
    return delta_bid - delta_ask
```

Vectorize across all N snapshots per day using numpy comparisons. This is a straightforward vectorized operation.

---

## Section 6: Does 24s OB Cadence Fundamentally Undermine Orderbook Features?

**Short answer: No. But the feature set must acknowledge its temporal resolution.**

### The Longer Answer

The orderbook features in this spec occupy a specific role: they provide the market context in which the order event occurred, not a real-time book feed. At 24s cadence, they provide a coarse but still valuable market context:

- **Level and shape of the book** (features 11-14: `log_spread`, `imbalance_L1`, `imbalance_L5`, `depth_ratio`) reflect market structure that evolves slowly (minutes to hours). A 24s snapshot cadence is adequate for these - the spread regime (tight vs. wide) and the book shape (bid-heavy vs. ask-heavy) are persistent features. These features are correct and informative.

- **Dynamic features** (`delta_imbalance_L1`, `cum_ofi_5`) capture book motion. At 24s, they capture motion at 24s resolution. This is coarser than tick-by-tick OFI but still captures the dominant low-frequency order flow pressure. In Cont et al. (2014), OFI was measured at 1-minute resolution and was predictive. Our 24s is better than 1 minute.

- **Trade-vs-book features** (`book_walk`, `trade_vs_mid`) are affected by staleness but the effect is bounded and partially mitigated by training augmentation.

- **Regime indicator** (`kyle_lambda`) is sparse but still carries regime information.

### What the 24s Cadence Cannot Provide

The spec's original 3s assumption allowed for microstructure features at the tick-by-tick level of resolution: seeing individual quote updates, tracking quote stuffing, detecting iceberg refills. None of this is available at 24s. However, these features were never in the spec to begin with. The spec's book features were always coarse-grained contextual features, not tick-by-tick quote surveillance.

The model's primary signal comes from the trade sequence (features 1-10), not the orderbook. The orderbook provides context. Degraded context quality at 24s vs 3s affects a secondary signal. The model can function - and in fact, the training augmentation (OB dropout at p=0.15) explicitly trains the model to predict without book context at all.

### A Novel Point: 24s Cadence May Filter Noise

At 3s cadence, the book snapshots would have included a substantial amount of high-frequency quote flickering - market makers adjusting quotes dozens of times per minute in response to individual trades. This noise would appear in `delta_imbalance_L1` and `cum_ofi_5`. At 24s, quote flickering is averaged out and only sustained directional book changes appear. This may actually improve the signal quality of the book features by eliminating the micro-noise layer. The L1 imbalance at a 24s snapshot reflects a genuine 24s-sustained market maker positioning, not a quote flash.

---

## Summary Table

| Feature | Recommendation | Severity | Change Required |
|---------|---------------|----------|----------------|
| `cum_ofi_20` (F18) | Reduce to `cum_ofi_5` | High | Window: 20 → 5 snapshots; update name |
| `kyle_lambda` (F17) | Retain with guard | Medium | No code change; update spec interpretation |
| `delta_imbalance_L1` (F16) | Keep as-is | Low | No change; sparsity is informative |
| `trade_vs_mid` (F15) | Keep as-is | Low | No change; staleness bounded |
| `book_walk` (F7) | Keep as-is | Low | No change; staleness bounded |
| OFI formula (F18) | Piecewise Cont 2014 | **BLOCKING** | Must implement price-level-change cases |

---

## Appendix: Empirical Prediction on Feature Utility

Given these constraints, my prediction for feature utility at the primary 100-event horizon:

1. **Most informative OB features (likely):** `imbalance_L1`, `cum_ofi_5`, `delta_imbalance_L1` - these carry the core Cont (2014) signal
2. **Contextual but informative:** `log_spread`, `depth_ratio`, `imbalance_L5` - book shape carries regime information
3. **Noisy but present:** `kyle_lambda`, `trade_vs_mid`, `book_walk` - staleness and sparsity add noise but signal exists
4. **Feature interaction worth monitoring:** the correlation between `delta_imbalance_L1 == 0` and the quality of `book_walk`/`trade_vs_mid` - the CNN should learn this implicitly

The gradient-based attribution in Step 4 should be run on the OB features specifically to verify this ordering after training.

---

*Council-2 sign-off: The 24s snapshot cadence is a significant constraint but not a fatal one. Fix the OFI formula, reduce the cum_ofi window to 5, and proceed. The primary signal in this model is in the trade sequence, and the book provides coarse but valid context at 24s resolution.*


