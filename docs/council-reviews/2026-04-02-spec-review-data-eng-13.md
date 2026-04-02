# Data Engineering Review: Tape Reading Direct Spec
**Reviewer:** data-eng-13  
**Date:** 2026-04-02  
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`  
**Status:** Conditional PROCEED — 3 blocking issues, 9 implementation flags

---

## Executive Summary

The pipeline is buildable. The spec is architecturally sound and the raw parquet schemas are sufficient for all 18 features. However, three findings from actual data inspection require resolution before the pipeline can be trusted: (1) the orderbook snapshot cadence is ~24 seconds, not ~3 seconds — every staleness assumption in the spec is 8x off; (2) 59% of raw order events have both buy and sell fills at the same timestamp, and the pre-April data has no `event_type` field to resolve aggressor direction; (3) 30-74% of trade rows are exact duplicates that must be deduplicated before any feature computation. None of these are fatal to the project but all three require explicit handling decisions before `tape_dataset.py` is written.

---

## 1. Data Pipeline Feasibility

**Overall: Feasible.** The parquet schemas are sufficient. DuckDB glob loading works correctly. The pipeline can be built as described.

### Confirmed schema (from actual data inspection)

**Trades** (`data/trades/symbol={SYM}/date={DATE}/*.parquet`):
- Columns: `ts_ms` (int64), `symbol` (str), `trade_id` (str, **always empty**), `side` (str), `qty` (float64), `price` (float64), `recv_ms` (int64), `date` (datetime)
- **Missing:** `cause` and `event_type` fields do not exist in pre-April data. These are only available from April 1 onward. The spec acknowledges this but it affects the direction-assignment problem described in Section 2.
- **`trade_id` is always empty string** — deduplication cannot use this field.

**Orderbook** (`data/orderbook/symbol={SYM}/date={DATE}/*.parquet`):
- Columns: `ts_ms` (int64), `symbol` (str), `bids` (numpy array of dicts), `asks` (numpy array of dicts), `recv_ms` (int64), `agg_level` (Int32, always NA), `date` (datetime)
- **10 levels per side, not 25.** The spec's CLAUDE.md says "25 levels" but all sampled symbols (BTC, FARTCOIN, XPL) have exactly 10 bid levels and 10 ask levels. This affects `imbalance_L5` (needs L1-5, so it still works) and `depth_ratio` (uses total depth, still works), but `imbalance_L5` should be described as "up to 5 of available levels" not "L1:5 of 25."
- Each level is a Python dict `{'price': float, 'qty': float}` — not a struct or flat columns. Parsing cost at scale must be factored in (see Section 7).
- `agg_level` is always NA — ignore this field.

**Funding** (`data/funding/symbol={SYM}/date={DATE}/*.parquet`):
- Columns: `ts_ms` (int64), `symbol` (str), `rate` (float64), `interval_sec` (int64), `recv_ms` (int64), `date` (datetime)
- Funding is not used in the 18 features. The spec correctly excludes it from the feature set. No action needed.

### Missing fields that would require workarounds
- `event_type` (taker/maker) — not available pre-April. See Section 2.
- `trade_id` — always empty; cannot use for deduplication.

---

## 2. Order Event Grouping — BLOCKING ISSUE

**The same-timestamp grouping assumption is complicated by two findings.**

### Finding 2a: Matched-pair duplication (30-74% of rows)

Inspecting BTC (2025-10-16): 30% of raw rows are exact duplicates on `(ts_ms, side, qty, price)`. For XPL: 74%.

This is not data corruption. It is the exchange reporting both counterparties of each fill as separate rows in the same parquet files. Each fill appears as: one `open_long`/`close_long` row for the buyer, and one `open_short`/`close_short` row for the seller, at identical `ts_ms`, `qty`, `price`.

**Required action:** Deduplicate with `df.drop_duplicates(subset=['ts_ms', 'side', 'qty', 'price'])` before grouping. This should happen in the data loading step, not feature computation. After dedup, BTC goes from 74,370 to ~51,979 rows (70% retention).

Validation: `assert len(df) < len(raw_df)` after dedup.

### Finding 2b: Mixed-direction events (59% of events after dedup)

After deduplication, 59.2% of unique-timestamp groups on BTC still contain both buy-direction rows (open_long, close_long) and sell-direction rows (open_short, close_short) at the same millisecond.

**Root cause:** This is an exchange mechanic. When a taker buy order is matched against multiple maker sell resting orders, the exchange records both the taker's fills and the maker's fills at the same timestamp. The 59% figure reflects high fill frequency — in a liquid market, most timestamps involve matching between a taker and one or more makers.

**Impact on `is_buy` (feature 3):** For 59% of events, the direction cannot be read directly from the side field using `open_long/close_long = buy`. You need to identify which leg is the aggressor (taker).

**Available methods, in priority order:**
1. **Post-April data:** Use `event_type == 'fulfill_taker'` to identify the aggressor. Clean and unambiguous.
2. **Pre-April data:** The buy/sell quantity is nearly balanced in mixed events (75.7% of sampled events have exact buy/sell qty equality). This means the taker direction cannot be recovered from notional imbalance alone. The price level provides a weak signal (trade at ask = buyer aggressed; trade at bid = seller aggressed), but without a contemporaneous mid-price this is approximate.
3. **Fallback for pre-April mixed events:** Mark `is_buy = 0.5` (ambiguous). The model will need to learn from the other 17 features for these events.

**Recommended implementation:** The validate_event_grouping.py script (Step 0) should quantify what fraction of events have recoverable direction. The pipeline should implement the taker/maker split for April+ data and apply the price-vs-bid-ask heuristic for earlier data, with `is_buy` clamped to [0, 1].

**The spec's 5% threshold for mixed-side events is not achievable.** The actual rate is 59%. The spec should be updated: this is an exchange-level artifact, not a data quality problem with the grouping heuristic. Mixed-direction events are real and common; the pipeline must handle them, not treat them as anomalies.

### Finding 2c: Fill count distribution

After dedup, BTC event fill counts range from 1 to 53, with the modal count at 2. XPL has events with up to 79 fills (before dedup this is due to duplicated matched pairs). The `num_fills = log(fill_count)` feature will work correctly after dedup, but Step 0 validation should assert that no event has > 100 fills after dedup as a sanity check.

### Finding 2d: `is_open` feature

`is_open` is "fraction of fills that are opens." After dedup, each group may contain `open_long`, `close_long`, `open_short`, `close_short` rows. For a pure buy event (only long-side rows), `is_open = n_open_long / (n_open_long + n_close_long)`. For a mixed event, the definition is ambiguous. Recommendation: compute `is_open` from the taker side only when identifiable; otherwise average across all fills.

---

## 3. Orderbook Alignment

### `np.searchsorted` correctness

`np.searchsorted(ob_ts, event_ts, side='right') - 1` is correct for "nearest prior snapshot." Verified:

```
event_ts=100 (== first snapshot) -> index 0 (correct: snapshot at 100 is prior)
event_ts=150 -> index 0 (correct: uses snapshot at 100)
event_ts=400 (== last snapshot) -> index 3 (correct)
event_ts=500 (> last) -> index 3 (correct: carry last snapshot forward)
```

The approach is causally clean: `side='right'` means equal timestamps return the snapshot at that exact time, which is acceptable (the book state was known at that ms).

### BLOCKING: Edge case — events before first snapshot

`np.searchsorted` returns index -1 for any event earlier than `ob_ts[0]`. On BTC 2025-10-16, the first trade arrives at `ts=1760614446343` and the first orderbook snapshot is at `ts=1760614502801` — a 56-second gap. All events in this gap would get index -1.

**Required handling:**
```python
ob_idx = np.searchsorted(ob_ts, event_ts, side='right') - 1
no_prior_ob = ob_idx < 0
# Option A: zero-fill orderbook features for these events
ob_features[no_prior_ob] = 0.0
# Option B: drop these events from training samples (they appear at sequence start only)
```

Option A is preferred — a zero-fill is honest and the rolling statistics will handle warm-up. Assert after alignment: `assert np.all(ob_ts[ob_idx[~no_prior_ob]] <= event_ts[~no_prior_ob])`.

### BLOCKING: Snapshot cadence is ~24 seconds, not ~3 seconds

The spec claims orderbook snapshots arrive "every ~3 seconds." Measured cadence (BTC 2025-10-16): **median ~24 seconds**, mean ~24 seconds, max 59 seconds. This is 8x slower than the spec states.

**Implications:**
- `delta_imbalance_L1` (feature 16): the spec says it will be "~95% zero due to 3s snapshot cadence vs ~50ms event rate." At 24s intervals with 10.6 events per snapshot, the actual zero fraction is approximately 90.6% (events/snapshot ratio: 10.6, so 1/10.6 = 9.4% of events coincide with a new snapshot). The carry-forward approach is still correct.
- `cum_ofi_20` (feature 18): "20 book snapshots" now covers ~480 seconds (~8 minutes) of market time, not ~60 seconds. This substantially changes the economic interpretation of the feature. The rolling window should be reconsidered: 20 snapshots at 24s = 480s lookback vs. the spec's implied ~60s.
- Orderbook staleness for `book_walk` (feature 7): events use `mid` from the most recent snapshot, which could be up to 24 seconds old. During volatile periods this mid-price will be meaningfully stale. No fix needed, but the model should learn to discount book features when `delta_imbalance_L1 == 0` (book is stale).
- The `kyle_lambda` rolling 50-event window uses `Δmid` between events, not between snapshots — this is unaffected by snapshot cadence.

**Recommendation:** Update spec to say ~24s cadence. Consider whether `cum_ofi_20` window should be 5 snapshots (~120s) instead of 20 (~480s) to match intended timescale.

---

## 4. Feature Computation Concerns

### Feature 1: `log_return`
- **Edge case:** `vwap = 0` is impossible for real trades (price > 0). Safe.
- **Edge case:** `prev_vwap = 0` cannot occur after the first event. First event: use 0.0 or skip. Recommendation: set `log_return = 0.0` for the first event of each day.
- **Edge case:** Log of negative ratio impossible since both vwaps are positive.
- No concerns at scale.

### Feature 2: `log_total_qty`
- Rolling 1000-event median is correct. Use `pandas.Series.rolling(1000, min_periods=1).median()` — `min_periods=1` handles the warm-up period honestly (early events use partial windows).
- **Pre-warms:** The spec says pre-warm from end of prior day. This requires loading the prior day's events to extract the last 1000 qty values. Adds ~1 DuckDB query per symbol-day. Implement as a `_load_warm_state(symbol, date)` helper that returns `{rolling_qty_buffer: array(1000), rolling_return_buffer: array(1000)}`.
- **First calendar day (2025-10-16):** Mask first 1000 events from training. With ~19K events per day, this is 5% of day-1 events — acceptable.

### Feature 3: `is_buy`
- Ambiguous for 59% of events. See Section 2. Must use taker-identification logic, not raw side field.

### Feature 4: `is_open`
- Computed from fill sides within the grouped event. After dedup, each fill has a clear side (open_long/close_long/open_short/close_short). Fraction of opens = `(n_open_long + n_open_short) / total_fills` for the taker leg.
- If direction is ambiguous, use all fills. Reasonable approximation.

### Feature 5: `time_delta`
- `log(ts - prev_ts + 1)` — the +1 prevents log(0) when two events have identical timestamps (after grouping, impossible by construction since we're grouping same-ts trades). Safe.
- For the first event of each day: prev_ts is from prior day. Use the actual prior event timestamp if pre-warming; otherwise set `time_delta = log(1) = 0`.

### Feature 6: `num_fills`
- `log(fill_count)` where fill_count >= 1. Always safe.

### Feature 7: `book_walk`
- `abs(last_fill_price - first_fill_price) / max(spread, 1e-8 * mid)`
- **Spread from which snapshot?** Uses the nearest prior orderbook snapshot. The snap could be up to 24 seconds old, meaning spread may not reflect current market. Acceptable — it's the best available.
- **Single-fill events:** `last_fill == first_fill`, so `book_walk = 0`. Correct.
- **Zero-spread guard:** `max(spread, 1e-8 * mid)` — confirmed necessary. With 10-level books, spread at L1 should always be positive, but illiquid symbols (XPL) can have wide or zero spreads during thin periods. Guard is correct.

### Feature 8: `effort_vs_result`
- `clip(log_total_qty - log(|return| + 1e-6), -5, 5)`
- `log_total_qty` is the already-normalized value from feature 2, not raw `log(qty)`. The spec is explicit. This is correct.
- `|return| = 0` is common for events where vwap equals prev vwap. With epsilon=1e-6, `log(1e-6) = -13.8`. This makes `effort_vs_result = log_total_qty - (-13.8) = log_total_qty + 13.8`, then clipped to 5. So any zero-return event gets clipped to 5 (maximum effort, no result = absorption). This is the intended interpretation.
- Clip to [-5, 5]: prevents explosion. Correct.

### Feature 9: `climax_score`
- `clip(min(qty_zscore, return_zscore), 0, 5)` with rolling 1000-event sigma.
- `rolling_std` can be 0 if all events in the window have identical qty/return (unlikely but possible at day open or for illiquid symbols). **Guard required:** `np.where(rolling_std > 1e-10, (x - rolling_mean) / rolling_std, 0.0)`.
- `min(qty_zscore, return_zscore)` can be negative (normal event). The `clip(..., 0, 5)` maps all normal events to 0, which is correct — climax score is only nonzero for simultaneous extreme qty AND extreme return.
- Same pre-warm requirement as feature 2 (shares the rolling window state).

### Feature 10: `prev_seq_time_span`
- `log(last_ts - first_ts + 1)` for the PREVIOUS 200-event window.
- For events in the first 200-event window of a day: no prior window exists. Options: (a) use 0.0 (safest — tells model "no prior window known"), (b) use the prior day's last-window time span (requires pre-warm).
- **Recommendation:** Use 0.0 for the first 200 events per day. Do not pre-warm this feature across day boundaries — day-open tape speed is genuinely different from prior-day close and the prior span would be misleading.
- At sequence boundaries within a day (events 0-199, 200-399, etc.): each sequence at position i (within a 200-event window) needs the same `prev_seq_time_span`. The value is constant within a non-overlapping window and changes at window boundaries. This is correct and efficient to compute.

### Feature 11: `log_spread`
- `log((ask - bid) / mid + 1e-10)`
- The `+ 1e-10` prevents log(0) for zero spread. However, `(ask - bid) / mid + 1e-10` where spread=0 gives log(1e-10) = -23.0. This is a large negative value that will stand out from normal spreads. Consider using `log(max((ask - bid) / mid, 1e-10))` instead for cleaner semantics.
- Mid = (ask + bid) / 2. Unambiguous from 10-level book.

### Feature 12: `imbalance_L1`
- `(bid_notional_L1 - ask_notional_L1) / (bid_notional_L1 + ask_notional_L1)`
- **Edge case:** `bid_notional_L1 + ask_notional_L1 = 0` (both sides empty). Extremely unlikely for L1 but should be guarded: `np.where(denom > 0, num / denom, 0.0)`.
- Notional = qty * price. With 10-level book, L1 is always the `bids[0]` and `asks[0]` dict elements. Simple to extract.

### Feature 13: `imbalance_L5`
- Inverse-level weights: 1.0, 0.5, 0.33, 0.25, 0.2 for levels 1-5.
- **Actual book has only 10 levels, not 25.** Since we're using L1-5 of 10, this works correctly. No change needed.
- **Edge case:** Some snapshots may have fewer than 5 levels (extremely thin book). Guard: use `min(5, len(bids))` levels for the sum, normalize weights accordingly.

### Feature 14: `depth_ratio`
- `log(max(total_bid_notional, 1e-6) / max(total_ask_notional, 1e-6))`
- Total notional = sum over all 10 bid levels and all 10 ask levels.
- Epsilon guard is essential — confirmed in data: XPL can have very lopsided books.
- Parsing cost: summing 10 dicts per snapshot per side. At 1,857 snapshots/day × 25 symbols = 46,425 snapshots/day, this is 46K dict accesses per day, easily vectorized.

### Feature 15: `trade_vs_mid`
- `clip((event_vwap - mid) / max(spread, 1e-8 * mid), -5, 5)`
- Mid and spread from nearest prior snapshot. Same staleness concern as feature 7.
- When `event_vwap` is outside the bid-ask spread (order that walked the book past the visible levels), this value exceeds 1.0 before clipping. The clip to [-5, 5] handles this correctly.

### Feature 16: `delta_imbalance_L1`
- `imbalance_L1 - prev_event_imbalance_L1`
- Carry-forward between snapshots: correct. At 10.6 events per snapshot, ~90.6% of events will have `delta = 0` (book hasn't changed).
- Day boundary pre-warm: load last snapshot from prior day. This is a single-row read and is worthwhile to avoid masking day-open events.
- **First calendar day:** Mask first event (set `delta_imbalance_L1 = 0`). Correct.

### Feature 17: `kyle_lambda`
- `Cov(Δmid, signed_notional) / Var(signed_notional)` over rolling 50 events.
- **Δmid** = change in LOB midpoint between consecutive events (not between snapshots). Since mid is carried forward from the last snapshot, Δmid will be 0 for most consecutive events (10.6 events share the same snapshot). Only ~9.4% of consecutive event pairs will show a non-zero Δmid. This makes `Var(Δmid)` very small and makes `Cov(Δmid, signed_notional)` noisy.

  **This is a real concern.** The rolling 50-event window will often contain only 4-5 non-zero Δmid values. `kyle_lambda` will be numerically unstable for large stretches. The existing `prepare.py` implementation (lines 816-827) guards with `rolling_var > 1e-20`, which is the right approach. Carry that guard forward.

- `Var(signed_notional) = 0` when all 50 events in the window are the same size (very unlikely for real data, more likely at sequence start). Guard with `np.where(var > 1e-20, cov/var, 0.0)`.
- The existing `prepare.py` implementation can be ported directly (lines 816-827).

### Feature 18: `cum_ofi_20`
- Rolling sum of OFI over last 20 book snapshots, normalized by rolling 20-snapshot total notional.
- **OFI at each snapshot (Cont 2014):** `OFI_t = (bid_notional_L1_t - bid_notional_L1_{t-1}) - (ask_notional_L1_t - ask_notional_L1_{t-1})`. Note: if the best bid price changes between snapshots, this is not simply delta-qty × price. At 24s intervals, the best bid price does change frequently (observed: 111235 → 111166 in one snapshot). The correct Cont 2014 formula applies:
  - If best_bid_price_t > best_bid_price_{t-1}: `delta_bid = +bid_notional_L1_t` (full new quantity)
  - If best_bid_price_t = best_bid_price_{t-1}: `delta_bid = bid_notional_L1_t - bid_notional_L1_{t-1}`
  - If best_bid_price_t < best_bid_price_{t-1}: `delta_bid = -bid_notional_L1_{t-1}`
  - Mirror for ask side.

  This piecewise formula is not described in the spec. The spec says "notional delta-bid minus notional delta-ask at L1" which is the simplified version assuming price levels don't change. **The price-change case must be implemented correctly** or OFI will have sign errors during trending markets.

- Normalization: divide by rolling 20-snapshot total notional (sum of all bid + ask notional across 20 snapshots). Guard: `np.where(norm > 1e-10, cum_ofi / norm, 0.0)`.
- Forward-fill to events: same carry-forward as delta_imbalance_L1.

---

## 5. Rolling Statistics at Boundaries

### Day boundaries (all rolling features)
- **Correct approach:** Pre-warm from the last N events of the prior day. Load prior-day events, extract final state of rolling buffers, pass to current-day computation.
- **Implementation:** Cache rolling warm state alongside features. Add to `.npz` cache: `{rolling_qty_buffer, rolling_return_buffer, rolling_std_buffer}`.
- **First calendar day (2025-10-16):** No prior day. Mask first 1000 events from training samples for features 2, 9. Set `time_delta = 0` for first event.

### Within-day sequence start (feature 10: `prev_seq_time_span`)
- The first 200-event window of each day has no prior window. Use 0.0. Do not pre-warm from prior day — day-open tape speed differs and would confuse the model.

### Symbol boundaries
- Rolling windows are per-symbol-day. No cross-symbol contamination is possible since we process one symbol-day at a time.

### Cold start at sequence position 0
- `log_return` (feature 1): first event has no prior vwap. Set to 0.0.
- `delta_imbalance_L1` (feature 16): first event has no prior imbalance. Pre-warm from prior day's last snapshot, or 0.0 for first calendar day.
- All rolling windows: use `min_periods=1` for pandas rolling functions to avoid NaN before window fills.

---

## 6. Caching Strategy

### Cache key
The `.npz` file should be keyed by: `{symbol}_{date}_{feature_hash}.npz` where `feature_hash` is a short hash of the feature spec version (e.g., first 8 chars of SHA256 of a version string like `"v1-18features"`). This allows cache invalidation when the feature set changes without requiring a separate versioning file.

**Do not** key by raw data file contents — the files are immutable once collected and checking their hash adds latency.

### Cache contents
Each `.npz` should contain:
- `features`: `float32` array shape `(n_events, 18)` — the full day's features
- `labels`: `float32` array shape `(n_events, 4)` — precomputed direction labels for all 4 horizons
- `timestamps`: `int64` array shape `(n_events,)` — event timestamps for alignment and debugging
- `prices`: `float32` array shape `(n_events,)` — vwap per event, for label validation
- `warm_state`: dict with rolling buffer state for pre-warming the next day

### Incremental updates
Per the feedback in MEMORY.md: migrate caches when adding features, never rebuild from raw parquet. The feature_hash in the cache key enables this: when the feature spec changes, the old cache is a miss and only that symbol-day is recomputed. The `warm_state` dependency means re-computation should proceed in date order per symbol.

### Cache location
`.cache/tape/{symbol}/{date}_{feature_hash}.npz` — creates one directory per symbol, each with up to 160 files.

---

## 7. Memory and Performance Concerns

### Actual dataset sizes (measured, not estimated)

| Metric | Spec Estimate | Measured (BTC, full day) |
|--------|-------------|--------------------------|
| Raw trades/day | ~140K | ~143K (full day), ~74K (partial first day) |
| Order events/day | 50-70K | ~28K (BTC 2025-10-18) — 3.8-8.6x grouping ratio |
| 200-event samples/day | ~300 | ~141 (BTC) — less active symbols fewer |
| Total training samples | ~1.2M | **~390K-560K** (depending on symbol activity) |
| Per symbol-day cache | not specified | ~1.4 MB (float32, 18 features × ~19K events) |
| Total cache (uncompressed) | not specified | ~5.7 GB |

The spec's "1.2M samples" estimate is optimistic. The actual sample count is likely 400K-600K. This is still adequate for training but changes batch size and epoch length calculations.

**Important:** All 25 symbols produce ~74K raw trades/day regardless of liquidity (FARTCOIN, XPL, BTC all yield ~74K). This strongly suggests a per-day collection cap in the Pacifica API. The grouping ratio varies widely: BTC/ETH/SOL ~4x, DOGE/AVAX ~8x. Illiquid symbols have higher grouping ratios (more matched pairs, fewer unique timestamps).

### Memory during processing
- Per symbol-day: 143K trades × ~8 columns × 8 bytes = ~9 MB in-memory. Safe.
- Orderbook: 1,857 snapshots × 10 levels × 2 sides × 2 fields × 8 bytes = ~0.6 MB. Safe.
- Feature computation: one symbol-day at a time, write to .npz, move on. Peak memory < 100 MB per symbol-day. No problem on any machine.

### Orderbook parsing performance concern
The orderbook `bids` and `asks` columns are arrays of Python dicts, not structured numpy arrays. Parsing 10 dicts per snapshot × 2 sides × 1,857 snapshots = 37,140 dict accesses per day. At 25 symbols × 160 days = 4,000 symbol-days, this is 148M dict accesses total. This is not a bottleneck (each dict access is fast) but should be vectorized:

```python
# Vectorize: extract all L1-L10 prices and qtys into flat arrays
def parse_ob_levels(ob_array, n_levels=10):
    prices = np.array([[lv['price'] for lv in snap[:n_levels]] for snap in ob_array])
    qtys = np.array([[lv['qty'] for lv in snap[:n_levels]] for snap in ob_array])
    return prices, qtys  # shape (n_snapshots, n_levels)
```

This produces numpy arrays and enables fully vectorized feature computation for all 8 orderbook features.

### 200-event window time span
Measured on BTC (full day, Oct 18): median 200-event window = **605 seconds (~10 minutes)**. This is much longer than "~5-10 seconds" implied by the spec's label description. The labels themselves are at 10/50/100/500 events forward, which at BTC's event rate corresponds to roughly 30s / 2.5min / 5min / 25min. The spec's "very short term, ~0.5-1 sec" for 10-event horizon is wrong for order events — it applies to raw trades but not grouped events.

---

## 8. Data Quality Risks

### Required validation before any training

**Step 0 must run these checks:**

1. **Duplicate detection:** For each symbol, assert that `drop_duplicates(subset=['ts_ms','side','qty','price'])` reduces row count by 20-75%. If no duplicates are found, the loading logic is wrong.

2. **Mixed-direction events:** Assert that mixed-direction events exist (they will, 50-60%). Assert that after handling, `is_buy` has no NaN values. Report the fraction of events where direction was recovered via `event_type` (April data) vs. approximation (pre-April).

3. **Orderbook level count:** Assert `len(bids) == 10` and `len(asks) == 10` for all snapshots. If fewer levels appear, the imbalance and depth features need length guards. Flag any snapshot with fewer than 2 levels on either side.

4. **Feature finiteness:** `assert np.all(np.isfinite(features))` after computing all 18 features. NaN sources to watch:
   - `kyle_lambda` when `Var(signed_notional) == 0`
   - `log_return` at day start
   - `cum_ofi_20` before 20 snapshots have accumulated

5. **Causal alignment:** `assert np.all(ob_ts[aligned_idx[valid]] <= event_ts[valid])` where `valid = aligned_idx >= 0`. Events with no prior snapshot should be explicitly tracked, not silently dropped.

6. **Grouping reduction:** `assert len(events) < len(raw_trades_after_dedup)` — grouping should reduce row count further (not all timestamps will be unique after dedup).

7. **Label sanity:** For each horizon h, assert that label[i] = `int(price[i+h] > price[i])` does not look ahead into unavailable events. At the end of each day, the last h events have no label — use `np.nan` and exclude from training. Assert `sum(isnan(labels))` matches expected boundary counts.

8. **Snapshot cadence sanity:** Assert that median OB interval is in [15s, 35s]. If it falls outside this range, a data fetch issue may have occurred.

9. **Same-symbol-day row count sanity:** Assert each symbol-day has at least 500 order events after dedup and grouping. Days with fewer events are incomplete and should be excluded (they produce fewer than 2-3 training samples and distort rolling windows).

---

## 9. Implementation Recommendations

### `tape_dataset.py` critical path

1. Load trades via DuckDB glob, sort by `ts_ms`.
2. **Deduplicate** on `(ts_ms, side, qty, price)`.
3. Group by `ts_ms` — each group is one order event.
4. Assign direction: for April+ data use `event_type`; for pre-April use price-vs-book heuristic.
5. Compute per-event trade features (1-10) in a single vectorized pass using `pandas.groupby().agg()`.
6. Load and parse orderbook: extract `bids`/`asks` into numpy arrays of shape `(n_snapshots, 10)` for prices and qtys.
7. Align events to snapshots: `ob_idx = np.searchsorted(ob_ts, event_ts, side='right') - 1`, clip to `[-1, len(ob_ts)-1]`.
8. Compute orderbook features (11-18) for each snapshot, then use `ob_idx` to select the right snapshot for each event.
9. Write `(features, labels, timestamps, prices, warm_state)` to `.cache/tape/{symbol}/{date}_{hash}.npz`.

### `PyTorch Dataset` key behaviors
- `__getitem__(idx)` should return `(seq, label)` of shapes `(200, 18)` and `(4,)`.
- Pre-load all `.npz` caches at `__init__` into a manifest. Do not open files in `__getitem__` — use memory-mapped arrays.
- Randomize first-window offset per epoch by storing an epoch-seed offset in the dataset.
- Embargo: at fold boundaries, exclude events within 600 events of the boundary from both sides.
- Handle day boundaries: do not stitch event windows across day boundaries (different trading sessions, rolling state resets). Each sample must be from within a single symbol-day.

---

## Summary of Blocking Issues

| # | Issue | Severity | Required Action |
|---|-------|----------|----------------|
| B1 | OB cadence is 24s not 3s | High | Update spec, reconsider `cum_ofi_20` window size |
| B2 | 59% mixed-direction events, no `event_type` pre-April | High | Implement price-vs-book heuristic or accept `is_buy=0.5` for ambiguous events |
| B3 | 30-74% duplicate rows; `trade_id` always empty | High | Dedup on `(ts_ms, side, qty, price)` before grouping |

## Summary of Implementation Flags

| # | Issue | Action |
|---|-------|--------|
| F1 | Events before first OB snapshot | Zero-fill OB features, guard `ob_idx >= 0` |
| F2 | OB has 10 levels not 25 | Update spec; no code change needed |
| F3 | `kyle_lambda` near-zero Δmid due to 24s snaps | Port `prepare.py` guard: `where(var > 1e-20, cov/var, 0)` |
| F4 | Cont 2014 OFI price-change case | Implement piecewise OFI formula, not naive delta-notional |
| F5 | `cum_ofi_20` covers ~8 min not ~1 min | Decide on intended lookback; consider 5-snapshot window |
| F6 | `climax_score` rolling std can be zero | Guard: `where(std > 1e-10, (x-mean)/std, 0)` |
| F7 | `prev_seq_time_span` at day boundary | Use 0.0 for first 200 events; do not pre-warm across days |
| F8 | Training sample count is ~400-560K not 1.2M | Adjust epoch/batch planning accordingly |
| F9 | Day-boundary `delta_imbalance_L1` pre-warm | Load last OB snapshot from prior day; 0.0 for first calendar day |
