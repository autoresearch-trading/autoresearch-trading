# Council Review: Direction Ambiguity and Tape Reading Viability

**Reviewer:** Council-4 (Richard Wyckoff — Tape Reading, Volume-Price Analysis)
**Date:** 2026-04-02
**Issue:** 59% of order events contain mixed buy+sell fills, making `is_buy` ambiguous for pre-April data

---

## Framing: What Wyckoff Actually Read

Before addressing the specific problem, it is worth being precise about what tape reading was and was not. Richard Wyckoff read a paper ticker tape. That tape printed: price, volume (lot size), and time. It did not print aggressor direction. The concept of \"taker vs. maker\" did not exist on Wyckoff's tape — it was printed by the exchange as price and size, period.

What Wyckoff inferred from that undirected stream:

- Whether price was advancing or declining (from the sequence of prices)
- Whether effort (volume) was proportionate to result (price change)
- Whether the move was on expanding or contracting volume
- Whether the tape was \"fast\" (urgent, aggressive) or \"slow\" (reluctant, absorbed)
- Whether support or resistance was being tested with conviction or feebly

None of these required knowing which side was the aggressor. Wyckoff deduced direction from the price sequence itself — if price went up on heavy volume, buying pressure was present. If price held steady on heavy volume, absorption was occurring regardless of which side was technically the taker.

This framing is not just historical comfort. It is analytically important: the tape reading approach this model is trying to learn was defined in an era where direction was inferred, not observed. The 59% ambiguity problem restores the original Wyckoff information environment.

---

## Question 1: How Critical is Aggressor Direction for Tape Reading?

**The honest answer: less critical than it appears, with one important exception.**

The classical Wyckoff patterns can all be detected without knowing who is the taker:

**Effort vs. Result:** Large absolute volume + small absolute price move = absorption. This works unsigned. Whether a 10,000 BTC event was 10,000 buyers or 10,000 sellers crashing into resting orders does not change the fact that the price barely moved. The interpretation is: someone with a large resting position absorbed the flow. The absorber's direction is actually ambiguous in the original Wyckoff framework too — when price holds on high volume, either buyers absorbed sellers OR sellers absorbed buyers. Both mean the same thing: a large patient participant is present.

**Climax Events:** A volume climax followed by price reversal is the same pattern regardless of who the taker was. If price spikes up on extreme volume and then reverses, a buying climax occurred. The model observes the price spike (positive log_return), the extreme volume (high climax_score), and the subsequent reversal (future label). It does not need to know that 60% of the volume was taker-buys vs. 40% taker-sells — the net result in price tells the story.

**Springs and Upthrusts:** A spring is a brief penetration of support followed by a recovery. The model sees: price falls below prior low (negative log_return), then recovers (future label going up). The is_open feature shows whether participants are opening positions on the downside probe. This is the critical Wyckoff spring signal — are people opening longs (accumulation) or opening shorts (distribution continuation)? This works without knowing which fills were takers.

**The one exception: kyle_lambda.** Kyle's lambda requires signed notional flow (`signed_notional = qty * vwap * direction`). If direction is ambiguous for 59% of events, the lambda estimate is corrupted for those events. This is the most significant theoretical loss because lambda is the regime indicator — it tells the model whether market makers are currently repricing aggressively (informed flow regime) or passively (noise flow regime). A corrupted lambda is worse than no lambda because it will inject false signal into the estimation window.

**Summary on criticality:** Direction is material for `kyle_lambda` (corrupted, needs remedy) and mildly useful for `is_buy` as a momentum signal. For the Wyckoff core — effort vs. result, climax detection, is_open, book features — direction ambiguity is a manageable limitation, not a fundamental invalidation.

---

## Question 2: Can a Model Learn Tape Reading from Unsigned Volume + Price + Book State?

**Yes. And the argument is stronger than it might seem.**

Consider what the model has access to without clean direction:

- `log_return`: the signed price change (unambiguous — price either went up or down)
- `log_total_qty`: the absolute volume (unambiguous)
- `effort_vs_result`: volume relative to price change (unambiguous — this is inherently unsigned)
- `climax_score`: z-score of qty and return (unambiguous)
- `is_open`: fraction of fills that are opens (unambiguous — does not require direction)
- `book_walk`: unsigned spread consumption (unambiguous)
- All 8 orderbook features: unambiguous
- `time_delta`, `prev_seq_time_span`, `num_fills`: unambiguous

That is 16 of 18 features operating cleanly. The two features dependent on direction are `is_buy` (ambiguous) and `kyle_lambda` (corrupted for 59% of events).

Now consider what a sequence of 200 events gives the model even without direction:

- If price has been rising over the window and volume has been rising on up-moves and shrinking on pullbacks: markup phase
- If price oscillates in a range and volume is large but moves are small: absorption / accumulation-distribution
- If is_open is consistently high (many position opens relative to closes): the Composite Operator is active
- If a single event shows extreme climax_score followed by a price reversal: climax and phase transition

These are the core Wyckoff patterns. They are all recoverable from the unsigned feature set.

The model also has `log_return` which IS signed — price went up or down. So the model sees \"price went up by X on volume Y.\" It does not see \"price went up because of Z buying pressure.\" But the distinction matters less than it seems for a predictive model: the model is predicting future price direction, not explaining past price causation. The signed price history, combined with volume patterns and book state, is sufficient to detect the Wyckoff phases.

**Where unsigned information loses ground:** Distinguishing between a large buy event and a large sell event at the same price level. Imagine two identical events: same volume, same book state, same prior price. In one case it was a large buy (suggests upward pressure). In the other, a large sell (suggests downward pressure). Without direction, the model cannot distinguish these. However, the price result resolves this ambiguity immediately: if it was a large buy, the ask side should be exhausted and price should tick up. If it was a large sell, the opposite. The future price moves contain the direction information; the direction feature is only needed to accelerate the inference of what already shows up in subsequent price events.

---

## Question 3: How to Handle the Ambiguous `is_buy`

Three options were presented. Here is the Wyckoff analysis of each:

**Option (a): Include as noisy feature — 0.5 for ambiguous, 0/1 when known**

This is the worst choice. A feature that is 0.5 for 59% of observations and {0,1} for 41% is not noise in the standard sense — it is a mixture distribution with a massive spike at 0.5 and two smaller spikes at 0 and 1. The BatchNorm at the input layer will normalize this to something with near-zero mean and low variance, but the model will learn that 0.5 means \"ambiguous\" rather than \"uncertain neutral.\" Worse, the post-April data (April 1 onwards) where event_type is available will have clean 0/1 values for this feature, creating a systematic distributional shift between pre-April training data and any April validation. The feature becomes a date indicator in disguise: high values near 0.5 = pre-April, clean binary = post-April. This will corrupt any generalizations.

**Option (b): Drop is_buy entirely**

This is cleaner than (a) and arguably justified given that `log_return` already contains the signed price information. The model will infer buy vs. sell pressure from the price direction and book state. The cost is modest: `is_buy` has half-life of 1 event (per the CLAUDE.md findings), meaning it has essentially zero persistence. A feature with half-life 1 adds almost no multi-event pattern information. The model is not losing sustained directional signals — `is_buy` never had them.

The argument FOR keeping it is that for the 41% of events where direction IS clean, it provides a marginal disambiguation signal. But that 41% is scattered among 59% noise, and the model has no way to weight by confidence unless it is told which events are clean.

**Option (c): Derive a proxy from `trade_vs_mid`**

This is the best option, but requires careful framing. `trade_vs_mid = clip((vwap - mid) / spread, -5, 5)` already captures the economic content of `is_buy`:

- Positive trade_vs_mid: event executed above the midpoint, meaning at or near the ask — consistent with a buy
- Negative trade_vs_mid: executed below the midpoint, consistent with a sell

This is not a noisy proxy — it is the informational content of direction, expressed in a continuous form that is already in the feature set (feature 15). The model gets direction-like information from trade_vs_mid without any ambiguity about which side was the taker.

However, there is a subtle difference: for a mixed event (30% buy, 70% sell), the vwap will reflect that mix. A 30/70 split might produce a vwap near the mid or slightly below it. trade_vs_mid will correctly show this as slightly bearish. This is actually BETTER than `is_buy = 0.5` (which throws away the distribution) and better than `is_buy = 0` or `is_buy = 1` (which would assert a direction that is false).

**Recommendation: Drop `is_buy` (option b) and rely on `trade_vs_mid` to carry the directional information.** Do not add a synthetic proxy — trade_vs_mid already serves this function. The feature set already has directional information through: (1) `log_return` (signed price change), (2) `trade_vs_mid` (execution location relative to mid), and (3) `is_open` (position commitment direction implicitly — opening longs vs. opening shorts, though is_open is unsigned, it pairs with log_return to imply direction).

If the team insists on keeping a direction feature: derive it from trade_vs_mid as `sign(trade_vs_mid)` with magnitude equal to `abs(trade_vs_mid)` — a scalar in [-1, 1] that continuously expresses directional conviction. This replaces both `is_buy` and partially duplicates `trade_vs_mid`, so it is not recommended unless feature ablation shows clear benefit.

---

## Question 4: Does This Change the Fundamental Viability of the Tape Reading Approach?

**No. The core thesis is intact.**

The tape reading approach rests on four pillars:

1. Order event sequences carry microstructure information (absorption, climax, accumulation)
2. The Composite Operator leaves a footprint in `is_open` (opening vs. closing positions)
3. Volume-price divergence (`effort_vs_result`) signals phase transitions
4. The model learns these patterns universally across symbols

None of these pillars require clean aggressor direction. Pillar 2 (`is_open`) is the most direction-sensitive in spirit — you want to know if smart money is opening longs or opening shorts. But `is_open` is measured as the fraction of fills tagged `open_long` or `open_short` in the raw data. These tags are not about taker/maker — they are about whether the trade is opening a new position or closing an existing one. A maker fill of `open_long` type is still an open long. The `is_open` fraction is direction-agnostic with respect to the taker/maker ambiguity: it measures position commitment, not aggressor identity.

What this finding does change:

**The `kyle_lambda` feature needs repair.** For the 59% of events where direction is ambiguous, `signed_notional` cannot be computed. Options:
- Use `trade_vs_mid` as a direction weight: `pseudo_signed_notional = total_qty * vwap * clip(trade_vs_mid, -1, 1)`. This is a reasonable approximation — a trade at the ask is likely a buy, so weight it positively proportional to how far above mid it was.
- Compute lambda only from the 41% clean directional events, with carry-forward between clean events. This reduces the estimation window effective sample size but keeps it theoretically grounded.
- Replace kyle_lambda with a price-impact version that does not require direction: `Cov(|Δmid|, log_total_qty) / Var(log_total_qty)`. This measures how much the mid moves per unit of volume regardless of direction — an unsigned lambda. It is not the canonical Kyle (1985) formulation but it measures the same underlying quantity (price sensitivity to volume) without direction.

Of these, the pseudo_signed_notional approach using `trade_vs_mid` as a direction proxy is preferred. It preserves the rolling covariance structure, uses the best available direction proxy, and does not throw away 59% of observations.

**The April data is more valuable than it appeared.** From April 1 onward, `event_type` provides clean taker/maker identification. This means the April hold-out set has higher-quality direction features than the pre-April training data. The model trained on noisy pre-April `is_buy` may actually perform BETTER on April test data (which has clean direction) than its training loss would suggest — the signal is cleaner at test time than at training time. This is an atypical but positive asymmetry.

---

## Question 5: Wyckoff Patterns Robust to Direction Ambiguity

Here is a ranked assessment of each classical Wyckoff pattern, from most to least robust under direction ambiguity:

**Fully Robust (pattern detection unchanged):**

1. **Effort vs. Result (Absorption).** Measures absolute volume vs. absolute price change. Zero dependence on direction. High effort, low result = absorption. The interpretation may be \"absorption of selling pressure\" or \"absorption of buying pressure\" but either way signals a large patient participant and predicts reversal. The model can learn this without direction.

2. **Climax Events.** Identified by extreme volume AND extreme price move (climax_score). Both are unsigned in the current spec (uses `|return|` and total qty). A climax is a climax regardless of which side was the taker. Post-climax reversal is the signal.

3. **Volume Dry-Up.** Declining volume across sequential events signals waning interest. This is entirely unsigned — volume is low regardless of who sold or bought. Within the 200-event window, the CNN's dilated layers will see the sequence of `log_total_qty` values and detect the declining trend.

4. **Spring/Upthrust Structure.** Detected as: (a) `log_return` goes negative breaking prior lows, (b) volume does not expand explosively (effort_vs_result is moderate-to-high — if extreme volume, it could be a capitulation), (c) subsequent events show recovery. The model sees this pattern entirely through unsigned price and volume. `is_open` adds color (are participants opening positions on the probe?).

5. **Composite Operator Footprint.** `is_open` is the signal. Opening positions vs. closing positions. Not sensitive to taker/maker ambiguity because position open/close is tagged in the raw data independently of who was aggressive.

**Moderately Affected:**

6. **Secondary Tests.** After a spring, a test of the low on declining volume confirms the spring. The declining volume part is robust. The price level comparison part needs sign (was the retest above or below the spring low?), which comes from `log_return`. Trade_vs_mid does not help here — the issue is absolute price level, not execution location. `log_return` provides this correctly since it is always signed.

7. **Sign of Strength / Sign of Weakness.** The \"sign of strength\" is a decisive up-move on expanding volume after a spring. Detecting this requires seeing: `log_return > 0` (price up), `log_total_qty` high (volume expanding), `is_open` elevated (participants opening longs). Direction of `log_return` is clean. The directional ambiguity in `is_buy` is not material here because the unsigned pattern suffices.

**Most Affected:**

8. **Identifying Who is the Initiator in Ambiguous Mixed Events.** When 59% of events have both sides, there is genuine loss of information about whether the net flow was aggressive buying or aggressive selling. However, this information is partially recovered from `trade_vs_mid` (execution location) and `log_return` (resulting price move). The recovery is imperfect — there will be cases where a large net sell produces the same `log_return` as a large net buy if absorption is occurring simultaneously. This is where the model will make more errors than a system with clean direction data.

---

## Interaction with the `event_type` Field: Pre/Post April Discontinuity

There is a structural issue worth flagging explicitly. The training data (pre-April) has 59% direction-ambiguous events. The April development validation data (April 1-13) has clean `event_type`. If the model is trained on the noisy data and evaluated on the clean data, there is a feature distribution shift:

- Pre-April: `is_buy` is genuinely ambiguous; `kyle_lambda` is corrupted
- Post-April: `is_buy` is clean; `kyle_lambda` is correct

If the team retains `is_buy` as option (a) — 0.5 for ambiguous — then the training distribution contains many 0.5 values but the April validation data contains few or none. The model will see a distributional shift at evaluation time that does not represent real market change — it represents a data collection change. This could artificially inflate or deflate April performance in ways unrelated to model quality.

**Recommendation:** Drop `is_buy` entirely OR derive it cleanly from `trade_vs_mid` for ALL data (both pre and post-April). Do not use `event_type` to backfill `is_buy` only for post-April data while leaving pre-April as 0.5 — this creates a spurious discontinuity.

For `kyle_lambda`: use the `trade_vs_mid`-based pseudo_signed_notional approach uniformly for all data. This produces a consistent (if imperfect) estimate across the full 160 days.

---

## Summary Assessment

The 59% direction ambiguity is a real limitation but not a fatal one. Wyckoff himself never had clean aggressor direction data — he read price sequences, volume, and time, and inferred everything else. The model has:

- Clean signed price moves (`log_return`)
- Clean unsigned volume (`log_total_qty`, `climax_score`, `effort_vs_result`)
- Clean position commitment signal (`is_open`)
- Clean execution location proxy (`trade_vs_mid`, which encodes direction information continuously)
- Clean book state (all 8 orderbook features)

The specific remedies are:

1. Drop `is_buy` from the feature set. The information it carries is already present in `log_return` and `trade_vs_mid`. Its half-life of 1 event means it contributes negligible sequential pattern value anyway.

2. Replace `kyle_lambda`'s signed_notional with `pseudo_signed_notional = total_qty * vwap * clip(trade_vs_mid, -1, 1)`. This is not theoretically pristine but it is consistent across all 160 days and uses the best available direction proxy.

3. Do not introduce different direction computation methods for pre-April vs. post-April data. Pick one method and apply it uniformly.

The Wyckoff patterns that matter most — absorption, climax, spring/test, Composite Operator footprint — are all detectable from the remaining 16 clean features. The approach remains viable.


