# Council Review: Tape Reading Spec — Richard Wyckoff Perspective

**Reviewer:** Council-4 (Tape Reading / Wyckoff)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`
**Date:** 2026-04-01

---

## 1. effort_vs_result Formula

The formula `clip(log_qty - log(abs(return) + 1e-4), -5, 5)` is directionally correct but has a structural asymmetry that needs attention.

The Wyckoff principle is: when a large effort (volume) produces a small result (price change), absorption is occurring — smart money is absorbing supply or demand invisibly. The formula captures this: large `log_qty` and near-zero `return` yields a large positive value, flagging absorption.

**The problem is the epsilon floor `1e-4`.** At BTC prices around 68000, a return of `1e-4` (0.01%) corresponds to a 6.8-point move. Order events that truly close at the same VWAP as the previous event (zero return) are not rare — maker fills on a resting order produce exactly this. The `1e-4` floor masks the distinction between:

- A zero-return event (pure absorption, the strongest Wyckoff signal)
- A tiny-return event (near-absorption, still meaningful)

A better approach: use `log(abs(return) + epsilon)` where epsilon is set to the instrument's minimum tick return, not a universal constant. For BTC at 68000 with 0.1 tick increments, the minimum non-zero return is ~0.0000015. Using `1e-4` as the floor effectively treats anything below 6.8 points as zero — this is too coarse for a market that moves 0.1 points per tick.

**Recommendation:** Set epsilon dynamically as `min_tick / current_price` (computed per symbol, per day) or use a fixed value of `1e-6` as a safer universal floor. The clip to `[-5, 5]` is correct and necessary.

The formula also only works correctly if `log_qty` is already normalized (the spec uses `log(total_qty / median_event_qty)`, which is good). An unnormalized quantity would make this a noisy ratio. The normalization is correct.

**Verdict:** Formula is sound in principle, epsilon floor needs tightening. Not a blocking issue but will cause the model to undercount true zero-return absorption events.

---

## 2. is_climax: Rolling 1000-Event Window

The rolling 1000-event window for sigma estimation is a reasonable starting point but has two weaknesses.

**Window size.** At 50K order events per symbol per day, 1000 events is roughly 2% of a day. For slow symbols (e.g., PENGU, LDO), 1000 events might span 30-60 minutes. For fast symbols (BTC during a volatile session), 1000 events might span 10 minutes. The window is not adaptive to regime. During a quiet overnight session, the 1000-event sigma will be low — then when New York opens with normal activity, virtually every event will be flagged as a climax because it exceeds the quiet-session sigma. This creates false positives precisely when the tape is transitioning to normal, not climactic.

**The Wyckoff definition of a climax** requires both extreme volume AND extreme price movement occurring together. The spec captures this conjunction (`qty > 2σ AND |return| > 2σ`), which is correct. Many climax-detection schemes get this wrong by using OR. The AND requirement is right.

**However, the binary flag loses information.** A `1/0` flag tells the model "climax yes/no" but not "how climactic." During a genuine selling climax, there will be multiple consecutive climax events. The model should see the intensity and clustering, not just presence. A continuous score (`z_qty * z_return` clipped to `[0, 5]`) would be more informative than a binary flag and would naturally weight extreme events more heavily.

**Adaptive window proposal:** Use a time-based rolling window (e.g., 30 minutes of events, however many that is) rather than a count-based window. This would make sigma estimates regime-consistent across symbols and sessions.

**Verdict:** The rolling window prevents lookahead bias (correct), but a fixed count window will produce regime-inconsistent thresholds. Recommend either adaptive (time-based) windows or replacing the binary flag with a continuous intensity score.

---

## 3. Wyckoff Phase Captureability with 200 Events

Wyckoff's accumulation phase has a characteristic structure: a trading range forms (price oscillates), volume dries up on rallies within the range (no real selling pressure), then a spring (price briefly breaks below support on high volume) followed by a sign of strength (decisive rally on expanding volume).

**200 events is marginal for capturing full phases.** At ~50K events per day, 200 events covers roughly 8 minutes of average tape. Accumulation ranges on liquid perpetuals typically span 30-120 minutes (based on the price action behavior of BTC, ETH, SOL). The 200-event window will almost always catch the model mid-phase, not at the start of accumulation.

**What 200 events can capture reliably:**
- Springs and upthrusts (sudden probes below/above support/resistance, reversing within 20-50 events)
- Climax events (the spike itself)
- The test of a spring (decreasing volume on a retest — this is exactly what effort_vs_result measures over sequential events)

**What 200 events cannot capture:**
- The full accumulation or distribution range formation (needs 1000+ events)
- The overall character shift — volume drying up over many hours
- Phase transitions (markup beginning, markdown beginning)

**The spec acknowledges this implicitly** by sweeping `{100, 200, 500}`. The 500-event sweep is important — it may capture more complete phase structure. However, 500 events at 200 samples/day gives 30% fewer non-overlapping samples, which hurts training data volume.

**Critical gap:** The spec has no feature representing the rolling mean volume trend within the window. If volume is declining across the 200-event window (drying up), that is a powerful Wyckoff accumulation signal. None of the 16 features explicitly captures this. The model would have to infer it from the sequence of `log_total_qty` values — possible for a CNN or Transformer, but an explicit "volume trend within window" feature would accelerate learning.

---

## 4. Missing Tape Reading Signals

**Volume drying up** is arguably Wyckoff's most important signal and it is not explicitly featured. The spring test is confirmed by decreasing volume on the retest — the model must infer this from the raw `log_total_qty` sequence without guidance.

**Urgency changes** are partially captured by `time_delta`. Increasing time between events = tape slowing = urgency fading. Decreasing time = tape accelerating = urgency building. This is correct but `time_delta` is a per-event feature; the within-window trend of `time_delta` (is the tape speeding up or slowing down?) is not explicitly captured.

**Composite Operator footprint:** `is_open` (fraction of fills that are opens) is correctly identified as the Composite Operator's signal. When smart money accumulates, they appear as open_long buyers. During distribution, they open_short. The spec treats this as a raw per-event feature, which is right — the CNN/Transformer will see sequences of `is_open` events and learn patterns like "sustained open_long buying on declining volume = accumulation."

**Secondary tests** (lower-volume probes to confirm a spring) are capturable within 200 events if the spring and test are both in the window, but the window must be properly positioned. This is a sampling concern, not a feature concern.

---

## 5. Multi-Horizon Label Alignment with Wyckoff Phases

The multi-horizon label (10, 50, 100, 500 events forward) has a good fit with how Wyckoff phases resolve.

- **10-event horizon (~0.5-1 sec):** Too short for any Wyckoff phase to fully resolve. This horizon is noise for phase-based signals but may capture microstructure momentum.
- **50-event horizon (~2-5 sec):** Captures spring reversals and upthrust failures.
- **100-event horizon (~5-10 sec):** Signs of strength / weakness. The impulse move after a spring resolves at this scale.
- **500-event horizon (~30-60 sec):** Markup and markdown beginnings. The most Wyckoff-relevant horizon.

**The 500-event label is the most important for Wyckoff.** The multi-task loss sums all four, which means the 10-event noise horizon gets equal weight to the 500-event signal horizon. This could dilute the gradient from the Wyckoff-relevant signal.

**Recommendation:** Weight the multi-task loss by horizon, giving the 500-event label 2x or 3x the weight of the 10-event label. Or report per-horizon accuracy separately and identify which horizon drives the model's actual trading decisions before moving to Phase 2.

---

## Summary Assessment

The spec is well-constructed from a Wyckoff perspective. The core insight — effort vs. result as absorption detection, is_climax as phase transition marker, is_open as Composite Operator footprint — maps cleanly to classical tape reading principles. The main gaps are: (1) epsilon floor in effort_vs_result is too coarse for tick-precision absorption detection, (2) is_climax binary flag should be a continuous intensity score, (3) no explicit volume-trend-within-window feature to capture "drying up," and (4) the 500-event label should carry higher loss weight than the 10-event label. None of these are blockers — the pipeline as designed will produce a viable signal. These are refinements that sharpen the Wyckoff signal before the model has to discover it implicitly.
