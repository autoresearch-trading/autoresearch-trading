# Council-5 Review: Dedup Strategy and Direction Signal — Tape Reading Pipeline

**Reviewer:** Council-5 (Practitioner Quant)
**Date:** 2026-04-02
**Scope:** Dedup of raw trade rows; mixed-direction order events; `is_buy` feature viability

---

## Background

The data engineer has identified two structural data issues that must be resolved before the pipeline can produce valid features. Both stem from the same root cause: Pacifica reports both sides of each fill as separate rows. The dedup issue is about removing the double-reporting. The mixed-direction issue is about what remains after dedup — a genuine data ambiguity in pre-April data.

---

## Issue 1: Dedup Strategy

### The exchange's double-reporting structure

Each matched trade generates two rows at identical `ts_ms`, `qty`, `price`:
- One row from the buyer's perspective: `side ∈ {open_long, close_long}`
- One row from the seller's perspective: `side ∈ {open_short, close_short}`

The `trade_id` field is always empty, so it cannot be used to pair rows. The 30-74% duplication rate reported is consistent with this: ~50% would be the theoretical baseline if every trade appears exactly twice (buyer row + seller row). The observed range of 30-74% suggests some rows are already deduplicated or that some fills are single-sided (e.g., liquidations where one party has no row, or data delivery anomalies).

### Evaluating the proposed dedup: `drop_duplicates(subset=['ts_ms', 'side', 'qty', 'price'])`

**This dedup key is wrong.** The proposed key retains `side` in the subset. Consider what the raw data looks like for one fill:

```
ts_ms=100, price=68686, qty=0.001, side=open_long   ← buyer row
ts_ms=100, price=68686, qty=0.001, side=open_short  ← seller row
```

These two rows have different `side` values. `drop_duplicates(subset=['ts_ms', 'side', 'qty', 'price'])` will retain BOTH rows because they differ on `side`. The duplication is entirely NOT removed by this key. You would end up with the full double-counted dataset.

The dedup key was presumably intended to catch exact duplicates within one side's perspective (two identical rows from the buyer), not the structural buyer/seller pairing. That is a different problem.

**The correct dedup for the structural pairing problem** must recognize that for each fill, one of the two rows is the \"canonical\" event and one is redundant. There are two approaches, each with different trade-offs:

### Approach A: Keep one row per fill, deduplicate on `(ts_ms, qty, price)` without side

```python
df.drop_duplicates(subset=['ts_ms', 'qty', 'price'], keep='first')
```

This removes one row from each buyer/seller pair, keeping whichever appeared first in the file. The retained row has a `side` that reflects one party's perspective — either the taker or the maker, depending on file ordering.

Risk: If two genuinely different fills happened at exactly the same millisecond, same price, and same quantity — two separate market participants each buying 0.001 BTC at 68686 in the same ms — this dedup incorrectly merges them into one event. For liquid symbols (BTC, ETH) during active periods, this collision probability is non-trivial. For illiquid symbols (KPEPE, FARTCOIN) it is very low.

A practical check: after dedup, count the number of timestamps where more than one row remains with the same `(qty, price)`. On liquid symbols, these represent genuine same-price collisions (multiple fills at the same level). On illiquid symbols, any such \"double\" is suspicious.

### Approach B: Keep taker rows only (April+ data), or reconstruct aggressor direction

For April+ data, `event_type == 'fulfill_taker'` unambiguously identifies the aggressor's row. Keeping only taker rows is cleaner than deduplicating by quantity matching — it uses the semantically correct field.

```python
# April+ data: keep only taker fills
df = df[df['event_type'] == 'fulfill_taker']
```

This is strictly better than dedup for April data. Each row is now exactly one taker-initiated fill with the correct aggressor direction encoded in `side`.

For pre-April data without `event_type`, Approach A is the fallback.

### What the dedup does NOT solve

After dedup (by any method), the remaining rows for a given timestamp may still represent multiple independent fills from different participants — the same-timestamp collision problem noted above. Dedup removes the buyer/seller double-reporting; it does not resolve multiple takers hitting the book at the same ms. Both problems exist independently and must be handled separately:

1. **Buyer/seller double-reporting:** Resolved by dedup (Approach A) or taker filter (Approach B)
2. **Multiple independent fills at same timestamp:** Not a dedup problem — these are genuine distinct events that happen to share a timestamp. The order event grouping step must decide whether to group them (assumes one large order) or keep them separate (multiple orders)

### Required validation after dedup

After applying the dedup, run these checks before building any features:

1. **Duplication rate by symbol:** Compute `(raw_rows - dedup_rows) / raw_rows` per symbol per day. Expect ~50% for clean double-reporting. If a symbol consistently shows <30%, investigate — may have different reporting conventions. If >55%, there are genuine row-level duplicates beyond the buyer/seller pairing.

2. **Residual same-price collisions:** After dedup on `(ts_ms, qty, price)`, count timestamps with more than one remaining row. On BTC during active hours, expect occasional collisions. If >10% of timestamps have multiple residual rows on an illiquid symbol, the dedup logic is wrong.

3. **Side distribution after dedup:** The retained rows should be roughly 50% long-side (`open_long`, `close_long`) and 50% short-side (`open_short`, `close_short`) — one side chosen per fill by the `keep='first'` logic. If the split is heavily skewed (>70/30), the file ordering is systematically biased toward one side and the dedup is pulling signal from one counterparty preferentially.

4. **`num_fills` sanity check:** After grouping same-timestamp rows into order events, compare the pre-dedup and post-dedup fill counts per event. If pre-dedup shows consistent 2x multiples of post-dedup counts, the double-reporting hypothesis is confirmed.

5. **Cross-symbol consistency:** Run the dedup check on 5 symbols representing different liquidity tiers (BTC, SOL, DOGE, LDO, FARTCOIN) for 5 different dates each. Dedup rates should be consistent within each symbol and across dates. Large day-to-day variance suggests partial dedup had already been applied upstream on some days.

---

## Issue 2: Mixed Buy/Sell Fills After Dedup

### What the 59% figure means

After dedup (by whatever method), 59% of unique-timestamp groups contain both long-side and short-side rows. This is the core ambiguity: at a given millisecond, both a buyer-initiated and a seller-initiated fill are recorded. There is no clean single-direction aggressor for those timestamps.

The cause is clear: the matching engine stamps both taker fills and maker fills with the same timestamp when they occur within the same matching cycle. The `event_type` field introduced April 1 exists precisely to solve this — `fulfill_taker` marks the aggressor.

### The four options ranked

**Option 2 (April+ data, `event_type == 'fulfill_taker'`)** is not really an \"option\" — it is the correct behavior for April data regardless of which pre-April strategy is chosen. Use it. No ranking needed. The only risk: on the first day of April data (April 1), verify that `event_type` is populated for all fills, not just some. If there are rows with null or empty `event_type`, the field is not fully reliable and needs a fallback.

For pre-April data, the three options are:

**Rank 1: Option B — `is_buy = 0.5` for ambiguous events**

This is the correct choice for pre-April data, with one modification described below.

The argument against it — \"59% of events have ambiguous direction, so the feature is useless\" — is backward. Setting `is_buy = 0.5` is not injecting noise; it is accurately encoding the model's uncertainty. The model will learn that `is_buy = 0.5` is an uninformative state and weight the other 17 features more heavily. This is a learned adaptation, not a handicap.

Critically, `is_buy = 0.5` does not contaminate any other feature. `log_return`, `book_walk`, `effort_vs_result`, `climax_score`, `trade_vs_mid`, and all 8 orderbook features are computed from price and quantity, not from side direction. The 17 features that do not depend on direction are fully valid for pre-April data. The model has 17 clean channels and 1 uninformative channel per pre-April event. That is not a death sentence.

What matters: `is_buy = 0.5` must be applied consistently. Do not try to \"fix\" 59% of events with heuristics and leave 41% clean — that creates a mixed-quality feature with unpredictable behavior. For pre-April data, set `is_buy = fraction_of_long_fills_in_the_event_group`. For a pure-ambiguous event (equal long and short fills), this is 0.5. For an unambiguous long event (all fills are open_long or close_long), this is 1.0. For an event that is 2/3 long fills and 1/3 short fills, this is 0.67. This fractional encoding is strictly more informative than the binary 0.5 floor.

**Modification:** do not treat `is_buy` as binary at all. It already encodes \"fraction of fills that are buys\" in the fractional form. The only structural change needed is: for pre-April data, use `is_buy = long_fill_count / total_fill_count` rather than attempting to assign a binary direction. This is an honest representation of what is known.

**Rank 2: Option A — price-vs-book heuristic**

The heuristic (trade at ask price = buyer aggressed) is the standard approach in the academic microstructure literature (Lee-Ready algorithm). It is not lookahead bias — the book snapshot used must be the nearest prior snapshot (already specified in the pipeline), not the current state.

However, it has two problems specific to this dataset:

First, the book snapshots arrive every ~3 seconds. A fill at `ts_ms = T` uses the book snapshot from up to 3 seconds prior. If the book moved significantly between the snapshot and the fill (which is common during fast markets), the classification is wrong. On liquid symbols during volatile periods, the false-positive rate for Lee-Ready with 3-second stale books can exceed 30%.

Second, on a DEX perpetual, the \"ask\" is not as clearly defined as on a central limit order book. If the snapshot shows `best_ask = 68686.00` and the fill happens at `68686.00`, was the taker a buyer? Maybe — but it could also be a maker who placed a limit order at that price and got lifted. Without `event_type`, the heuristic introduces a systematic error that concentrates precisely in the high-volatility, high-activity events that are most informative for the model.

The heuristic is acceptable as a secondary validation — use it to check whether `is_buy = fraction_of_long_fills` is correlated with `price_vs_mid` in the expected direction. If the two signals are positively correlated (long-side-heavy events tend to execute at ask, short-side-heavy at bid), the fraction encoding is capturing real information even for pre-April data.

Do not use the heuristic as the primary `is_buy` encoding. It introduces false precision that the `fraction_of_long_fills` approach avoids.

**Rank 3 (last): Option C — drop pre-April data**

13 days of April 1-13 data = approximately 13 × 25 × 300 samples = ~97,500 training samples across all symbols. After walk-forward embargo, usable training may be 8-10 days. At 300 samples/day, that is 2,000-3,000 training samples per symbol. A 94K-parameter CNN is dramatically overparameterized relative to 2,000 samples per symbol.

The sample-to-parameter ratio drops from ~18:1 (with all 160 days) to approximately 0.6:1 (with 13 days). This is not a model; it is a random-initialization artifact. Any reported accuracy on this sample size is statistically meaningless — the Bonferroni-corrected significance threshold at N=2,000 requires >53% to clear the null band, and the variance will be enormous.

Additionally, April 14+ is the designated untouched hold-out (irrevocable as of April 2, 2026). April 1-13 is the development validation set. If the pre-April data is discarded, the development validation set becomes the training set, the untouched hold-out has nothing to compare against, and the entire evaluation framework collapses.

Option C is not a valid choice. It should be removed from consideration.

---

## Issue 3: Is `is_buy` Worth Keeping?

Yes, but with an honest implementation.

The theoretical case for `is_buy` comes from the Kyle multi-period model: informed traders accumulate in one direction. Sustained sequences of `is_buy = 1.0` (genuine unambiguous buys) are the temporal signature of accumulation. The `is_open` autocorrelation half-life of 20 events implies that the side direction persists across an accumulation episode — a 200-event window covering 10 half-lives is well-positioned to detect this.

The empirical counter-evidence from the current dataset is `is_buy` having a half-life of 1. This means the buy/sell direction in the raw trade data alternates too fast for persistence-based signals to survive. That finding is consistent with the 59% mixed-direction rate — if 59% of events have genuinely ambiguous direction, the autocorrelation of the correctly-classified 41% is diluted by the noise of the mislabeled 59%.

The question is whether the April data, where `event_type` provides clean taker direction, shows different autocorrelation for `is_buy`. If the April data shows half-life > 5 events for `is_buy`, the half-life of 1 in pre-April data was an artifact of the direction ambiguity, and clean `is_buy` is a strong feature. If April data also shows half-life of 1, then `is_buy` has no persistence and is genuinely useless as a stand-alone feature.

**Recommended action:** compute `is_buy` autocorrelation on April 1-13 data using `event_type == 'fulfill_taker'` for clean direction. Compare to the pre-April half-life of 1. If the April autocorrelation is substantially higher, keep `is_buy` and accept the degraded quality in pre-April data. If April also shows half-life ~1, drop `is_buy` and replace with `trade_vs_mid` (feature 15), which captures the same information (did the fill happen at bid or ask) without requiring a clean direction field. `trade_vs_mid` is sign-sensitive: positive values indicate buy-side execution (above mid), negative indicate sell-side — it is a continuous, price-derived proxy for aggressor direction that is computable from price and book alone.

**If retaining `is_buy`:** for pre-April data, use `fraction_of_long_fills` as described above. For April+ data, use `1 if event_type == 'fulfill_taker' and side in {open_long, close_long} else 0 if event_type == 'fulfill_taker' and side in {open_short, close_short} else 0.5`. The `else 0.5` handles any null `event_type` rows gracefully.

---

## Overall Pipeline Recommendation

### Step 1: Dedup

```
Pre-April data:
  1. Drop duplicate rows where (ts_ms, qty, price) are identical — keep='first'
  2. Flag days where dedup_rate is outside [40%, 60%] for manual inspection
  3. After grouping by ts_ms: compute is_buy = long_fill_count / total_fill_count

April+ data:
  1. Filter to event_type == 'fulfill_taker' — this is both the dedup AND the direction assignment
  2. Check that event_type is non-null for >99% of rows before using this filter
  3. After grouping by ts_ms: is_buy = 1 if side in {open_long, close_long} else 0
```

### Step 2: Validation gates before proceeding to feature computation

Run `validate_dedup.py` with these checks:

- Per-symbol dedup rate: expect 45-55% row reduction. Flag outliers.
- Residual same-`(qty, price)` duplicates at same timestamp: should be <5% of timestamps on illiquid symbols, potentially higher on BTC/ETH/SOL.
- Post-dedup side distribution: expect ~50/50 long/short within 5%.
- `is_buy` distribution: what fraction of events are 0.0, 0.5, 1.0 exactly? For pre-April data, 0.5 fraction should be roughly consistent with the 59% ambiguity rate. If it is much lower, the event grouping is absorbing the ambiguity.
- April data: verify `event_type` non-null rate per symbol per date. Require >99% before using taker filter.

### Step 3: Handling the two data periods as distinct populations

The clearest approach is to treat pre-April and April data as separate data streams with different feature quality, joined at training time:

- Pre-April (Oct 2025 - Mar 2025): `is_buy` = fractional direction, noisy, 159 trading days
- April (Apr 1-13): `is_buy` = clean binary direction, 13 trading days

Add a boolean feature `has_clean_direction` (1 for April data, 0 for pre-April). This allows the model to learn different reliance on `is_buy` across data periods without any architectural changes. The model receives an honest signal about data quality rather than having the ambiguity hidden.

Alternatively — and slightly cleaner — add a `direction_confidence` feature that is `|is_buy - 0.5| * 2` (ranges 0 to 1, where 1 = fully unambiguous, 0 = fully ambiguous). This is a continuous quality indicator derivable from the data itself without a flag.

### Step 4: Accept the pre-April limitation and proceed

The pre-April `is_buy` ambiguity is a real limitation, not a catastrophic one. The pipeline has 17 other features that are fully valid. The model trained on pre-April data will learn to predict direction from price dynamics, book state, quantity patterns, and flow signals — all of which are clean. The `is_buy` feature will provide a weak, noisy secondary signal. A model that relies heavily on `is_buy` is building on sand; a model that uses it as one of 18 inputs is using it appropriately.

The correct framing: this is a signal quality problem, not a lookahead problem. There is no leakage. The direction ambiguity does not contaminate the label or any other feature. It simply degrades one feature's signal-to-noise ratio for 90% of the training data. Accept this and proceed.

---

## Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Dedup on `(ts_ms, side, qty, price)` leaves double-counting intact | Critical | Confirmed | Use `(ts_ms, qty, price)` without side |
| Same-price collision on liquid symbols causes under-dedup | Moderate | Moderate (BTC/ETH) | Validate residual duplicates; accept for prototype |
| April `event_type` not fully populated | Blocking for April strategy | Low | Validate before using taker filter |
| `is_buy = 0.5` for 59% of events reduces signal quality | Moderate | Confirmed | Accept; use `trade_vs_mid` as correlated backup |
| Dropping pre-April data makes model untrainable | Critical | Certain | Do not drop pre-April data |
| Lee-Ready heuristic with 3s stale book introduces systematic error | Moderate | High during volatility | Do not use as primary encoding |
| Inconsistent dedup between pre-April and April data creates training distribution shift | Moderate | Moderate | Document the regime clearly; use `direction_confidence` feature |

---

## Summary

The proposed dedup key `(ts_ms, side, qty, price)` is incorrect — it retains `side` in the key and therefore does NOT remove the buyer/seller double-reporting. The correct key is `(ts_ms, qty, price)` without `side`, or alternatively filter to `event_type == 'fulfill_taker'` for April+ data. For the direction ambiguity in pre-April data, `is_buy = fraction_of_long_fills` is the correct encoding — honest about uncertainty, non-contaminating, and allows the model to learn appropriate reliance on this feature. Dropping pre-April data is not viable at 13 training days; the Lee-Ready heuristic introduces false precision with 3-second stale books. The pipeline should proceed with 18 features, acknowledging that `is_buy` is a degraded but valid signal for pre-April data, and should validate the dedup strategy empirically before computing any downstream features.


