# Council-3 Review: Kyle's Lambda — Statistical Viability Under Real Data Constraints

**Reviewer:** Council-3 (Albert Kyle — price impact theory, informed/uninformed trading)
**Date:** 2026-04-02
**Issue:** Feature 17 (`kyle_lambda`) faces two compounding problems identified by data-eng-13: (1) OB snapshots arrive every ~24s, not ~3s, leaving only ~5 non-zero Δmid values per 50-event window; (2) 59% of pre-April events have ambiguous direction, making signed_notional near-zero for those events.

---

## 1. Is kyle_lambda Theoretically Viable With ~5 Non-Zero Δmid Observations Per Window?

### The estimator's statistical basis

The rolling OLS estimator for Kyle's lambda is:

```
λ̂ = Cov(Δmid, signed_notional) / Var(signed_notional)
```

Under standard OLS assumptions, the variance of this estimator is:

```
Var(λ̂) = σ²_ε / (n × Var(signed_notional))
```

where σ²_ε is the residual variance of the price-impact regression and n is the number of effective observations. With a 50-event window where only ~5 events produce non-zero Δmid, the effective sample size for the regression is n_eff ≈ 5, not 50.

This is the critical distinction. The formula uses all 50 observations in the covariance computation, but 45 of them contribute zero to the numerator (Cov contribution is Δmid × signed_notional, and Δmid = 0 for shared-snapshot events). The 45 zero-Δmid observations do reduce Var(signed_notional) in the denominator, which actually makes the estimate look more precise than it is. This is a statistical illusion: we are dividing a covariance estimated from 5 informative observations by a variance estimated from 50 observations.

The effective degrees of freedom for hypothesis testing on λ̂ are approximately 5, not 48. A t-statistic of λ̂/SE(λ̂) has critical value ~2.78 at 5% significance (two-tailed, df=5) versus ~2.01 at df=48. The estimate is substantially noisier and requires larger signal magnitude for statistical significance.

**Practical consequence:** The variance of λ̂ is inflated by approximately 50/5 = 10x relative to what it would be if all 50 Δmid values were non-zero. This means the estimate oscillates wildly between snapshots, contributing noise rather than signal to feature 17.

### Minimum non-zero observations for a meaningful estimate

From the OLS variance formula and the requirement that λ̂ have a meaningful signal-to-noise ratio, I establish the minimum as follows. For the feature to carry predictive information rather than noise, we need:

```
E[λ]² / Var(λ̂) ≥ 1    (signal exceeds estimation noise)
```

If the true λ is on the order of 1e-6 (typical for crypto perps in USD-notional terms), and σ²_ε ≈ Var(Δmid) (roughly, if R² of the price-impact regression is ~0.3), then we need approximately:

```
n_eff ≥ σ²_ε / (E[λ]² × Var(signed_notional)) ≈ 10-20 observations
```

This is not a precise bound but a practical rule of thumb: **fewer than 10 non-zero Δmid observations renders the estimate unreliable.** With 24-second snapshots and 10.6 events per snapshot, a 50-event window contains ~4.7 non-zero Δmid values on average. This is below the minimum. The estimate is essentially noise.

**Verdict on the 50-event window: not viable as currently specified.** The guard condition `where(var > 1e-20, cov/var, 0)` correctly prevents division by zero, but it does not rescue the estimate from being statistically meaningless at small effective sample sizes. The feature will output near-zero values most of the time (when all 50 events share 4-5 snapshots, Cov is tiny and noisy), occasionally spiking unpredictably when the 5 snapshot-change events happen to be correlated with signed notional. This is not regime detection — it is snapshot timing coincidence.

---

## 2. Should the Rolling Window Be Increased? Per-Event vs. Per-Snapshot Computation?

### Option A: Increase window to 200 events

A 200-event window would contain approximately 200/10.6 ≈ 19 non-zero Δmid values. This reaches the minimum threshold of ~10-20 identified above. The estimate becomes marginally viable but still has effective df ≈ 19, giving SE roughly 3x larger than an ideal estimator with all observations informative.

**Problem with the 200-event window:** The model's input sequence IS 200 events. Rolling lambda over the full input window means feature 17 at event t=200 summarizes the entire window's price impact regime. This creates a subtle coupling: the label (direction at 100 events forward from the last event) is partially predicted by a feature that summarizes the same 200-event window. This is not technically lookahead (the feature uses only events in the window, not future events), but it means the model can use lambda computed from near-future events (events 150-200) to predict the outcome of event 201. That is the correct use, but it is important to verify that the rolling computation is causal at every event within the window, not just at the window boundary.

**Problem with rolling 200 at event-level:** When computing features for the 50th event within a sequence, the 200-event rolling lambda uses the 50 most recent events, not 200. Lambda at event 1 of the sequence has approximately 0 effective observations (all from prior windows). Lambda only becomes meaningful at events 200+, which means the first ~200 events of each symbol-day have degraded lambda estimates regardless of window choice. This is handled by `min_periods` in the rolling computation but results in systematic feature degradation at sequence starts.

**Recommendation for Option A:** If window is increased to 200, this partially addresses Issue 1. It does not address Issue 2 (direction ambiguity).

### Option B: Compute lambda per-snapshot rather than per-event

The theoretically cleaner approach is to redefine what lambda measures in this data environment. Instead of `Cov(Δmid_per_event, signed_notional_per_event)`, compute:

```
λ_snapshot = Cov(Δmid_per_snapshot, cum_signed_notional_per_snapshot) / Var(cum_signed_notional_per_snapshot)
```

where:
- Δmid_per_snapshot = change in midpoint from one OB snapshot to the next (always non-zero or zero by construction)
- cum_signed_notional_per_snapshot = sum of all signed notional across all events between two consecutive snapshots

With 24-second snapshots, a 50-snapshot window covers 20 minutes of market time and contains exactly 50 non-zero Δmid observations. This is the theoretically correct form of the Kyle lambda estimator — it aggregates order flow over the period between price updates, which is the natural unit in a market where prices update discretely (each snapshot).

This approach completely solves Issue 1. Every observation in the 50-snapshot rolling window has a non-zero Δmid (the snapshot changed). The estimator has full effective degrees of freedom.

**Counterargument:** The 50-snapshot window covers ~20 minutes. Lambda measured over 20 minutes is a much slower regime indicator than lambda measured over ~75 seconds (50 events × ~1.5 sec/event). The economic interpretation shifts: you are detecting information regimes at the 20-minute timescale, not the 75-second timescale. For predicting direction at 100 events forward (~5 minutes), a 20-minute lambda lookback is lagged but still useful — information regimes tend to persist.

**My recommendation is Option B (per-snapshot computation) with a 50-snapshot window (≈20 minutes).** This produces a statistically valid estimate at the cost of temporal resolution. The feature is then forward-filled to all events between snapshots, exactly as `delta_imbalance_L1` and `cum_ofi_20` already are. Implementation is consistent with the existing OB-feature pipeline.

---

## 3. Is Signed Notional Computable Given 59% Direction Ambiguity?

### The ambiguity problem in detail

For pre-April data, 59% of events contain both long-side and short-side fills at the same millisecond. This occurs because the exchange records both counterparties of each match. The data-eng-13 analysis confirmed that 75.7% of these mixed events have exact buy/sell quantity equality, making notional imbalance useless for direction recovery.

If we use `is_buy = 0.5` for ambiguous events (the fallback in Section 2b of data-eng-13), then:

```
signed_notional = (is_buy - 0.5) × 2 × total_notional
```

For ambiguous events: `is_buy = 0.5` → `signed_notional = 0`.

This means 59% of events contribute zero to both the numerator (Cov) and denominator (Var) of the lambda estimator. The effective sample size for lambda is now reduced to 41% of events. Combined with the snapshot-cadence problem (only 9.4% of events have non-zero Δmid), the effective sample size for the event-level lambda estimator is:

```
n_eff = 50 events × 0.094 (non-zero Δmid) × 0.41 (non-ambiguous direction) ≈ 1.9
```

**With fewer than 2 effective observations in a 50-event window, the event-level kyle_lambda estimate is statistically meaningless for 150 days of pre-April data.** The `where(var > 1e-20)` guard will fire on the majority of windows, returning 0. Feature 17 will be a near-constant zero for 95%+ of training data.

### Impact on feature utility

A feature that is nearly always zero provides no gradient signal during training. The model will learn to ignore it. This wastes one of 18 feature channels and adds noise at the rare windows where the guard does NOT fire (spurious non-zero values from coincidental snapshot timing).

This is not a recoverable situation for the event-level formulation on pre-April data. The direction ambiguity is fundamental to the exchange's data format and cannot be resolved without `event_type`.

### Does April data rescue this?

From April 1 onward, `event_type = 'fulfill_taker'` provides unambiguous direction. With clean direction:
- Ambiguous events: ~0% (all events have a recoverable taker leg)
- Effective sample size for event-level lambda: 50 × 0.094 ≈ 4.7

Still only ~5 effective observations due to the snapshot cadence problem. The direction fix alone is insufficient.

**For the per-snapshot formulation (Option B above):**
- Each snapshot aggregates all fills between two OB updates
- For mixed events within a snapshot period, sum all `signed_notional` contributions — buys offset sells
- The net signed notional per snapshot period is what matters for lambda estimation, not per-event direction
- If direction is 50% ambiguous within a snapshot period, the net signed notional will be zero, but this is a real signal (balanced flow = no directional pressure), not a data artifact
- With April data, the taker-side aggregation per snapshot period gives a clean cumulative signed notional

**Conclusion on Issue 2:** Event-level signed notional with 59% ambiguity renders the event-level lambda estimator useless for pre-April data and borderline for April data. The per-snapshot formulation with aggregated signed notional is robust to individual-event direction ambiguity because balanced intra-snapshot flow correctly produces near-zero signed notional — which IS the market signal.

---

## 4. Alternative Price Impact Formulations That Work Without Clean Direction

### Option 1: Unsigned Amihud-style illiquidity

```
amihud_50 = rolling_mean(|Δmid| / total_notional, 50 events)
```

This is already implemented in `prepare.py` at lines 362-369 as `amihud_illiq`. It measures price impact per dollar of TOTAL volume (not signed volume). It does not require direction at all.

**Theoretical interpretation:** Amihud (2002) illiquidity measures the price movement generated per unit of trading activity, without conditioning on direction. High Amihud = market moves a lot per dollar traded = illiquid/informed regime. Low Amihud = market barely moves per dollar traded = liquid/noise-dominated regime.

**Limitation vs. Kyle's lambda:** Amihud is unsigned — it cannot distinguish whether a given notional amount of buying caused the price to rise (expected under Kyle) or whether high volume was accompanied by high price volatility regardless of direction (realized volatility effect). Lambda is theoretically superior because it captures the directional price impact asymmetry that identifies informed traders. But Amihud works without direction data and is substantially more stable than the rolling OLS lambda estimator.

**Recommendation for pre-April data:** Amihud-style rolling illiquidity is a reliable fallback that captures the broad information-intensity regime without requiring direction.

### Option 2: Realized spread and price impact decomposition (Huang-Stoll 1997)

This approach decomposes the spread into adverse selection and inventory components:

```
adverse_selection = sign(trade) × (mid_t+k - mid_t) / spread
realized_spread = sign(trade) × (price_t - mid_t+k) / spread
```

The problem: requires clean direction AND a forward mid price (lookahead). Not suitable for the causal pipeline.

### Option 3: Variance ratio approach

```
var_ratio = Var(Δmid over k events) / (k × Var(Δmid over 1 event))
```

Under a random walk (pure noise), this ratio = 1. Under informed trading with autocorrelated order flow, this ratio > 1 (momentum) or < 1 (mean-reversion). This is direction-free and does not require signed notional. It is the Lo-MacKinlay (1988) variance ratio adapted to order events.

**Limitation:** This is a feature of mid-price dynamics, not of order flow. It captures the outcome of informed trading (persistent price drift) rather than the cause (directional order flow). It will be highly correlated with `log_return` autocorrelation in the window.

### Option 4: Relative signed notional from `is_open` (no ambiguity)

This is specific to this dataset and theoretically motivated by Kyle's multi-period model:

```
directional_open_pressure = (fraction_open_long - fraction_open_short) × total_notional_in_window
```

`is_open` gives the fraction of fills that are position-opens. Combined with the side of those opens, this gives the directional commitment of the opening flow — which is the informed trader signal without requiring taker/maker resolution. An event with `is_open = 0.8` and `side = open_long` (or buy-side majority) is almost certainly an informed buyer opening a position. This does not depend on `event_type` — it only requires the `side` field (open_long vs. close_long vs. open_short vs. close_short).

The computation requires separating the `is_open` numerator by direction:

```
net_open_notional = (notional_open_long - notional_open_short)
```

This is computable from the existing data without `event_type` because `open_long` and `close_long` are distinct fields — the direction of the OPENING trade is unambiguous even in mixed events. A `close_short` (which is a buy) is still identifiable as a close regardless of which counterparty it matches with.

**This is theoretically the strongest feature of the set for detecting informed flow** — it directly measures the net directional commitment of position-opening trades, which is the Kyle multi-period model's primary prediction.

### Option 5: Per-snapshot kyle_lambda (my preferred recommendation)

Already described in Section 2, Option B. Restate for completeness:

```
Δmid_j = mid at snapshot j - mid at snapshot j-1        (50 snapshots, always non-zero or truly zero)
cum_sn_j = Σ signed_notional across all events between snapshot j-1 and snapshot j
λ_snapshot_50 = Cov(Δmid_j, cum_sn_j) / Var(cum_sn_j)   over rolling 50 snapshots
```

This resolves both issues: all 50 observations have potentially non-zero Δmid, and directional ambiguity within a snapshot period is resolved by aggregation (balanced intra-period flow → near-zero cum_sn_j, which correctly represents noise trading).

For pre-April ambiguous events: when computing `cum_sn_j` per snapshot period, events where direction is ambiguous (both long and short sides present) contribute near-zero signed notional after netting. This is the correct representation — a snapshot period containing balanced matched flow has low directional pressure, which is what cum_sn_j = 0 represents.

---

## 5. Recommendation

### Do not keep kyle_lambda as currently specified

The event-level formulation (`rolling 50-event Cov(Δmid, signed_notional) / Var(signed_notional)`) is broken in two independent ways on this dataset:

1. With ~5 effective Δmid observations per 50-event window, the estimator has 10x inflated variance. The feature will be dominated by snapshot-timing noise.
2. With 59% direction ambiguity pre-April, the effective sample size drops further to ~2 observations. The feature will be near-constant zero for 95%+ of training data.

Both issues cause the same outcome: feature 17 provides no gradient signal during training and the model learns to ignore it.

### Replace with per-snapshot kyle_lambda (50-snapshot window)

**New specification for feature 17:**

```
kyle_lambda: Cov(Δmid_snapshot, cum_signed_notional_snapshot) / Var(cum_signed_notional_snapshot)
             over rolling 50 OB snapshots (~20 minutes at 24s cadence)
             Forward-filled to all events between snapshots.
             Guard: where(var > 1e-20, cov/var, 0.0)
```

Where:
- `Δmid_snapshot` = midpoint change between consecutive OB snapshots
- `cum_signed_notional_snapshot` = sum of (is_buy - 0.5) × 2 × qty × price across all events between consecutive snapshots. For April data: use taker leg. For pre-April data: net the long and short sides within each snapshot period.
- Forward-fill: all events within a snapshot period share the same `kyle_lambda` value (the lambda estimated at the snapshot boundary)

This gives 50 fully informative observations, effective df = 48, and a stable regime indicator at the 20-minute timescale.

**Economic interpretation (preserved):** High λ_snapshot = market makers are updating prices strongly in response to order flow → informed regime. Low λ_snapshot = prices barely move per unit of flow → noise regime. The 20-minute timescale is appropriate for the 100-500 event prediction horizon (which covers 5-25 minutes at measured event rates).

### Add net_open_notional as a supplementary feature

If there is capacity for a 19th feature, the `net_open_notional` measure described in Section 4, Option 4 is theoretically the strongest direct measure of informed flow in this dataset. It does not suffer from either of the kyle_lambda problems and is computable without `event_type`. However, the spec is at 18 features and the architecture is designed for that count — do not expand the feature set until the baseline is validated.

### Implementation priority

1. Change the rolling computation from per-event to per-snapshot in `tape_dataset.py`.
2. Use the same infrastructure already in place for `cum_ofi_20` and `delta_imbalance_L1` (per-snapshot computation, forward-fill to events).
3. For the signed notional computation per snapshot period, net the `open_long`/`close_long` vs. `open_short`/`close_short` sides rather than requiring taker identification.
4. Keep the `where(var > 1e-20, cov/var, 0)` guard from `prepare.py` lines 822-826.
5. The warm-up period for this feature is 50 snapshots ≈ 20 minutes per symbol-day. The first 20 minutes of each day will have degraded lambda estimates. Do not mask these from training (the model should learn that low-confidence lambda = use other features), but do pre-warm by loading the last 50 snapshots of the prior day.

### What to report in Step 0

Before committing to this redesign, Step 0 should compute and report:

- Fraction of 50-event windows with fewer than 5 distinct Δmid values (expect > 90% under current spec)
- Variance of the event-level lambda estimate over the full dataset (expect very high relative to mean)
- Correlation of event-level lambda (current) with per-snapshot lambda (proposed) over April data where direction is available
- Whether the per-snapshot lambda has any predictive correlation with the 100-event label (even a 0.5% correlation is useful signal)

This is a minimal empirical check that can be done locally in ~15 minutes on 1 symbol, 1 day.

---

## Summary

The 50-event event-level kyle_lambda estimator has effective sample size ~2 under real data conditions (24s OB cadence + 59% direction ambiguity). This renders it statistically meaningless for pre-April data and borderline for April data. The fix is to redefine the computation unit as OB snapshots rather than order events: `Cov(Δmid_snapshot, cum_signed_notional_snapshot) / Var(cum_signed_notional_snapshot)` over rolling 50 snapshots. This fully resolves the sample-size problem, partially resolves the direction ambiguity (intra-snapshot netting), and produces a regime indicator at the 20-minute timescale appropriate for 5-25 minute prediction horizons. The theoretical content of the feature — Kyle's price-updating coefficient as an information regime indicator — is preserved.

The Amihud illiquidity measure (unsigned, already in `prepare.py`) is a viable fallback if the per-snapshot lambda implementation is deferred, though it loses the directional asymmetry that makes lambda theoretically superior.


