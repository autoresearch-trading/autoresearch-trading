# Council Review: Practitioner Quant — Tape Reading Direct Spec

**Reviewer:** Council-5 (Practitioner Quant — the skeptic)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Date:** 2026-04-01

---

## Critical Issues (Must Fix Before Building)

### 1. `seq_time_span` is Hard Lookahead Bias

Feature 10 is defined as:

```
seq_time_span: log(last_ts - first_ts + 1) for the full 200-event window
```

This feature is constant within a sample, but computing it requires knowing `last_ts` — the timestamp of the **200th event** in the sequence. When computing features for event 1 of the sequence, the model is given information about the future (specifically, how far into the future the 200-event window extends).

The stated justification — "tells the model whether 200 events covers 0.3 sec (crash) or 30 sec (quiet)" — is valid information, but it must be derived from **past** context, not the current window. Correct formulation: use a rolling estimate of tape speed derived from the previous N events before this sequence, not from the current window itself.

The "Lopez de Prado fix" label attached to this feature does not make it safe. Lopez de Prado's time-bar normalization uses the elapsed time of the *current bar* which is known at bar *close*, not at bar *open*. This feature as written is evaluated at the start of the sequence using information from the end.

**Fix:** Replace with `prev_seq_time_span` — the `seq_time_span` of the immediately preceding 200-event window. This is always available and captures tape speed from past context.

### 2. `median_event_qty` Normalization — Scope Undefined, Likely Leaks Future

Feature 2 is:

```
log_total_qty: log(total_qty / median_event_qty)
```

`median_event_qty` is not defined in the spec. If computed as a global statistic over the full training set (or the full day), it uses information from future events to normalize past events. This is a classic expanding-statistics lookahead.

The fix is already in the codebase's existing normalization approach: use a rolling median over the preceding 1000 events. The spec must explicitly state this. "median_event_qty" should be renamed to something like `rolling_median_event_qty_1000` to make the causal structure unambiguous.

**Additional concern:** this normalization window restarts at each day file boundary. The first 1000 events of each day will use an under-informed median. This is not lookahead, but it is a systematic bias — events early in the day will have noisier features. This should be handled by either: (a) pre-warming the rolling window from the end of the prior day, or (b) masking the first 1000 events of each day from training samples.

### 3. `depth_ratio` Has a Log-of-Zero Trap

Feature 14 is:

```
depth_ratio: log(total_bid_qty / total_ask_qty)
```

No epsilon is specified. When one side of the book is empty — which happens during flash crashes, around liquidations, and for illiquid symbols (several in the 25-symbol universe) — this produces `log(0)` = `-inf`. The spec added `+ 1e-10` to `log_spread` but missed `depth_ratio`.

**Fix:** `log((total_bid_qty + 1e-8) / (total_ask_qty + 1e-8))`. The 1e-8 should be in native qty units (not notional), so verify the scale — for a symbol like KPEPE trading at 0.00002 USD, minimum qty could be in the thousands per trade, so 1e-8 would be negligible. A safer guard is `log(max(total_bid_qty, 1e-6) / max(total_ask_qty, 1e-6))`.

### 4. `trade_vs_mid` Divides by Spread

Feature 15 is:

```
trade_vs_mid: (event_vwap - mid) / spread
```

The spread is not guaranteed to be positive. For BTC at $70K, the minimum tick is $0.10 — a spread of 1 tick on a $70K mid is 0.14 bps. Under high-frequency conditions, the best-bid and best-ask can appear equal in snapshot data (due to rounding in snapshot precision), producing spread = 0. This gives inf or NaN. The existing `prepare.py` code handles spread divisions with explicit guards — the new spec must do the same.

**Fix:** `clip((event_vwap - mid) / max(spread, 1e-8 * mid), -5, 5)`.

### 5. Same-Timestamp = Same-Order Assumption Is Unverified

The entire Order Event Grouping section rests on:

> "Trades with the same timestamp are fragments of one order being filled across price levels."

This is a reasonable heuristic for some exchanges but is **not validated against actual Pacifica data**. In practice, a DEX matching engine may process multiple independent taker orders within the same millisecond, particularly during high-activity periods. Grouping them as one event would:

- Inflate `num_fills` for what are actually separate orders
- Conflate opposing-side fills if a buy and sell happen at the same ms
- Produce misleading `price_impact` (the fills are from different orders, not one order walking the book)

The `cause` and `event_type` fields added on April 1 are specifically designed to disambiguate this — but the spec acknowledges historical data (Oct 2025–Mar 2025) does not have these fields. The assumption must be empirically validated on the historical data before it drives the entire pipeline design.

**Fix before Step 1:** Add a data validation script that computes the distribution of (a) fill count per timestamp, (b) same-timestamp fills with mixed sides, and (c) same-timestamp fills spanning more price levels than a normal single order would. If >5% of "events" have mixed sides, the assumption is broken.

---

## Moderate Issues (Should Address)

### 6. `delta_imbalance_L1` Produces Spurious Spikes at Day Boundaries

Feature 16 requires `prev_event_imbalance_L1`. When data is loaded day-by-day from Parquet files, the first event of each new day has no previous event — the delta will be 0 (or computed against a stale value from the previous iteration). This creates a systematic artifact: the model sees an artificial zero-delta at every day boundary, which is not a real market signal.

The first event of each day should either be masked out of training samples or pre-warmed from the last event of the prior day. This is the same issue that affects multi-level OFI (`mlofi`) in `prepare.py` — see `_compute_orderbook_features` where `prev_bid_depths = None` at the start of each call.

### 7. BatchNorm Degenerates at Inference Time with Batch Size 1

The architecture uses `BatchNorm1d(16)` as the input normalization layer. At training time with batch_size=256, this works correctly. At inference time with a single sequence (the production use case), batch normalization with a single sample produces NaN output because variance of a single sample is undefined.

PyTorch handles this via `model.eval()` which uses running statistics instead of batch statistics. The spec must explicitly call out: (a) that inference always uses `model.eval()`, and (b) that the running statistics are accumulated during training. A model deployed without `eval()` will produce garbage on single-sample inference.

### 8. Internal Inconsistency in Linear Baseline Feature Count

Phase 0.5 states:

> "Logistic regression on flattened (200×12=2800) features"

But the spec defines 16 features per event. The correct flattened size is 200×16=3200. This 12 appears to be a copy-paste error from an earlier spec revision. The logistic regression must use all 16 features for the baseline to be comparable to the neural network.

### 9. Logistic Regression Regularization Must Be Specified

The linear baseline uses sklearn's LogisticRegression with default C=1.0. With 3200 features and ~1.2M samples, the result is moderately sensitive to the regularization strength. More importantly, a model with C=1.0 and C=0.001 can differ by several accuracy percentage points. The spec should state: sweep C over {0.001, 0.01, 0.1, 1.0}, report the best per-horizon. Otherwise the go/no-go decision is made against an arbitrarily regularized baseline.

---

## Overfitting Assessment

65K parameters, 1.2M nominal training samples gives a ratio of ~18 samples/parameter — superficially healthy. However, the effective independent sample count is much lower due to temporal autocorrelation:

- 25 symbols × 160 days × 300 samples/day = 1.2M
- But within a symbol-day, the 300 windows are drawn from ~50-70K events. Adjacent samples share 195 out of 200 events (if non-overlapping, they share 0, but the spec says "random offset" which implies overlapping samples are possible)
- The underlying independent units are approximately symbol-days: 25 × 160 = 4000
- At 300 samples per symbol-day with rolling windows: information is not independent

The practical implication: cross-validation that respects this structure is walk-forward by time (which the spec does correctly). But if training samples are shuffled across symbols and dates without blocking by symbol-day, then a model can learn symbol-day-specific patterns and appear to generalize when it's actually overfitting to the specific symbol-days in training. The 52% accuracy target is low enough to be achievable via genuine signal, but monitor per-symbol-day accuracy variance closely.

---

## Compute Reality Check

The "2-3 hours" estimate for local CPU data pipeline assumes efficient processing. The actual bottleneck is the per-event orderbook alignment. The existing `_compute_orderbook_features` in `prepare.py` uses a Python for-loop over batches (line 994) — the new pipeline will have the same structure at 40-50x more iterations (50K events per symbol-day vs 1400 batches). At ~10K events/second for Python-level OB alignment, 25 symbols × 160 days × 50K events = 2 × 10^8 events → ~6 hours for the alignment step alone.

**Fix:** Vectorize the OB alignment using `np.searchsorted` over all events at once (as a batch), not in a Python for-loop. This matches what `prepare.py` already does for the funding alignment (line 530-531) and should be applied to the OB alignment too.

---

## What the Spec Gets Right

- Mandatory Phase 0 label validation before any model building — this is correct and disciplined.
- Mandatory linear baseline with explicit go/no-go threshold — this is the right approach.
- Rolling σ for `is_climax` rather than global σ — catches the obvious lookahead.
- `effort_vs_result` clipped to [-5, 5] — catches the numerical stability issue.
- Walk-forward validation with explicit skepticism about the Mar 5-25 window.
- Small model (65K params) appropriate for the data scale.
- The "success looks like universal signal across all 25 symbols" framing is correctly conservative.

---

## Priority Order for Fixes

1. **`seq_time_span` lookahead** — invalidates all results, fix before any code
2. **`depth_ratio` and `trade_vs_mid` epsilon guards** — silent NaN/inf corruption
3. **`median_event_qty` scope** — define as rolling 1000 explicitly
4. **Same-timestamp assumption** — validate empirically before pipeline build
5. **`delta_imbalance_L1` day boundary** — systematic artifact, moderate impact
6. **BatchNorm eval mode** — document explicitly for production path
7. **Linear baseline feature count** — correct the 12→16 error
8. **Vectorize OB alignment** — compute estimate will be wrong by 3x otherwise
