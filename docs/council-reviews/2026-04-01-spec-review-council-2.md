# Council Review 2: Order Flow Microstructure (Rama Cont)

**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Reviewer:** Council-2 (Order Flow and LOB Microstructure)
**Date:** 2026-04-01

---

## 1. Order Event Grouping — Same-Timestamp Assumption

The spec groups same-timestamp trades as fragments of a single order. This is reasonable for a first pass but has two structural problems.

**Multi-participant collision.** At millisecond resolution in a liquid perpetuals venue, multiple independent participants can and do trade at identical timestamps. The matching engine serializes fills but may stamp them with the same millisecond tick. Grouping them conflates what are genuinely distinct informed flow events. The distribution of fills-per-event should be inspected: if you regularly see groups of 10+ fills at identical timestamps, suspect multi-participant collision rather than single-order walking.

**Iceberg orders.** Iceberg (hidden quantity) orders present the opposite problem: a large resting order that refreshes its visible quantity repeatedly will generate many successive fills at the same price level, often at consecutive (not identical) timestamps a few milliseconds apart. The same-timestamp grouping criterion will split an iceberg into many single-fill events, losing the accumulated size signal entirely. A better heuristic adds a temporal tolerance window (e.g., group fills within 5ms at the same price level if same side) rather than requiring exact timestamp equality.

Neither problem is fatal at this stage, but the spec should at minimum monitor `fills_per_event` distribution and flag the 99th percentile. If it exceeds ~5 regularly, the grouping logic needs revision before training.

---

## 2. Orderbook Features: Specification Quality

**`log_spread` (feature 11).** Well-specified. Spread normalization is implicit since we are operating per-symbol and log compresses the scale. However, spreading the spread against mid (i.e., `spread_bps = (ask - bid) / mid`) is more interpretable across price regimes and is already used in the existing `prepare.py`. Log of absolute spread in dollar terms conflates price-level changes with genuine liquidity changes. Recommend `log((best_ask - best_bid) / mid)` instead of `log(best_ask - best_bid + 1e-10)`.

**`imbalance_L1` (feature 12).** The spec uses raw quantity at Level 1. The existing codebase uses notional (qty * price) for cross-symbol invariance. For a multi-symbol model this distinction matters: a 1 BTC bid and a 100 SOL bid have different dollar weights but may appear equal under raw-qty imbalance. Adopt notional-based imbalance consistent with the existing `prepare.py` implementation.

**`imbalance_L5` (feature 13).** The notation `bid_qty_L1:5` is ambiguous — it must mean levels 1 through 5, not just level 1 repeated. More importantly, flat summing across levels ignores that Level 1 is 5-10x more predictive than Level 5. The existing codebase uses inverse-level weighting (1.0, 0.5, 0.33, 0.25, 0.2) which is closer to optimal. The spec should adopt this weighting, otherwise imbalance_L5 duplicates noise from deep levels that contain no short-horizon predictive signal.

**`depth_ratio` (feature 14).** This feature (`log(total_bid / total_ask)`) captures deep book asymmetry, which research shows is largely uninformative at the 1-10 event horizon. It may be useful at longer horizons (50-500 events) but adds noise at the short end. Consider making it optional or replacing with `log_bid_depth` and `log_ask_depth` separately, which gives the model more flexibility.

**`trade_vs_mid` (feature 15).** Defined as `(event_vwap - mid) / spread`. This is correctly dimensioned as a fraction of spread (typically in [-0.5, +0.5] for normal fills, but walking orders can exceed 1.0). This is a good feature: it captures how deeply the order crossed into the other side's territory, which is a strong signal of urgency and information content. The implementation must guard against zero-spread events (use `max(spread, tick_size)`).

---

## 3. OFI Representation — Is delta_imbalance_L1 Sufficient?

The spec includes only `delta_imbalance_L1` (the change in L1 imbalance between consecutive book snapshots). This is correct directionally — delta is more predictive than level — but the implementation has a critical sampling mismatch.

The book snapshots arrive every ~3 seconds. Order events arrive every ~10-50ms in active markets. This means the overwhelming majority of events (perhaps 95-99%) will see `delta_imbalance_L1 = 0` because the book has not been re-snapshotted since the prior event. The feature is structurally sparse and the non-zero values cluster at the exact timestamps when a new snapshot arrived — which itself may introduce an aliasing artifact (the model learns "when the book updates, do X").

**Recommended fix:** Replace the raw snapshot-delta with a per-sequence-step interpolated or last-known-change value. Specifically, compute delta as `imbalance_at_current_snapshot - imbalance_at_previous_snapshot`, carry this value forward for every event between those two snapshots, then reset when the next snapshot arrives. This way every event in a "book-update window" shares the same delta, which at least is informative about the direction of the most recent book change.

**Missing: cumulative OFI.** The standard Cont et al. (2014) OFI is a cumulative sum of signed book-quantity changes across multiple book updates. The spec computes a one-step delta but does not provide a rolling-window cumulative OFI. At the 50-100 event horizon, a rolling cumulative OFI over the past 20 book updates is substantially more predictive than the instantaneous delta. This is a gap that should be addressed. Add `cum_ofi_20` (sum of OFI changes over the last 20 book snapshots, forward-filled to events).

---

## 4. Orderbook Alignment Methodology

The spec specifies `np.searchsorted` for nearest-prior snapshot alignment, which is correct and consistent with the existing `prepare.py`. The 3-second snapshot cadence creates step-function book features, which is acknowledged as a limitation.

One implementation detail: when the first order event of the day precedes the first book snapshot of the day (common in the first few seconds after session open), `searchsorted` returns index 0, which may be a snapshot from the prior day or an uninitialized state. The code must handle this boundary condition explicitly, falling back to `imbalance_L1 = 0` and `log_spread = nan` with masking.

---

## 5. Price Impact Feature (Feature 7)

The spec defines `price_impact = (last_fill_price - first_fill_price) / mid`. This is a reasonable intra-order price impact proxy. However, this quantity is zero for the vast majority of events (most orders fill at a single price level). The non-zero tail is where the information content lives.

A better formulation is `(last_fill_price - first_fill_price) / (best_ask - best_bid)`, i.e., normalized by the spread rather than the mid. This converts the raw dollar walk into units of spread-widths, which is directly comparable across symbols and regimes. An order that walks 2 spreads on BTC and 2 spreads on DOGE has experienced comparable levels of book consumption regardless of their dollar prices.

Also note: for sell orders, the price impact as defined will be negative (price decrements with each fill). The spec does not indicate whether to take absolute value or preserve sign. Preserving sign is correct since the existing `is_buy` feature already encodes direction; `price_impact` should be an unsigned measure of aggressiveness. Recommend `abs(last_fill_price - first_fill_price) / spread`.

---

## 6. Missing: Explicit Permanent vs. Transitory Price Impact

The spec conflates intra-order price impact (feature 7) with realized price impact, but makes no attempt to decompose permanent vs. transitory components. In the research literature (Almgren-Chriss, Cont-Kukanov), the key signal is: does the price revert after the order completes (transitory = market-making noise) or does it stick (permanent = informed flow)?

This could be approximated without forward-looking information by tracking `mid_after_event - mid_before_event` relative to `intra_order_walk`. If `mid` moves less than the walk, there was transitory impact. Adding a feature for the ratio of consecutive-event mid-price change to same-event walk would add a powerful information-content signal.

---

## 7. Summary Assessment

The orderbook feature set is mostly well-conceived. The critical gaps are: (1) cumulative OFI is missing and should be added as `cum_ofi_20`; (2) `delta_imbalance_L1` will be structurally sparse due to the 3s snapshot cadence and needs the carry-forward fix; (3) imbalance features should use notional rather than raw quantity for cross-symbol comparability; (4) `price_impact` should be spread-normalized and unsigned. The same-timestamp grouping heuristic is sufficient for a prototype but requires iceberg-order monitoring before production. These are addressable issues, not blockers.
