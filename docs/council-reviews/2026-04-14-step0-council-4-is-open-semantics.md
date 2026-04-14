# Council-4 (Phenomenologist) — Step 0 is_open Semantics Review

**Date:** 2026-04-14
**Reviewer:** Council-4 (Tape Phenomenologist — Wyckoff framing, load-bearing features)
**Sources:** `docs/experiments/step0-data-validation.md` + JSON, spec §Order Event Grouping, spec §Self-Labels

---

## Verdict

The corrected mixed-side rate (3-16% vs spec's 59%) is a **documentation error only** — it does **not** weaken the Composite Operator footprint hypothesis, and **no Wyckoff threshold requires recalibration**. The bimodal `is_open` distribution is arguably a cleaner signal than a fractional blur.

## Distributional reality of `is_open`

- **Illiquid symbols** (2Z, CRV, KBONK, etc.): avg_fills_per_event 1.00-1.09 → `is_open` is effectively binary {0, 1}.
- **Mid-tier** (AVAX, BNB, SOL, etc.): avg_fills 1.09-1.84 → mostly {0, 1}, occasional 0.5.
- **Most liquid** (BTC, ETH): avg_fills 2.07-2.67 → values like 0.33, 0.67, 0.75 reachable. BTC Jan-04 has 40.7% multi-fill events, max_fills=28.

This is **effectively ternary** for most events and **approximately continuous** only for BTC/ETH during active regimes.

## Why bimodal `is_open` strengthens, not weakens, the hypothesis

The Wyckoff Composite Operator signal was never about fractional participation — it was about distinguishing **position-opening flow from position-closing flow**. A step function where `is_open=1` means "100% new position entry" and `is_open=0` means "100% position exit" is a **crisper signal** than a continuous blur. Absorption vs distribution clusters should separate more cleanly in embedding space.

**DEX-specific claim still holds:** in traditional L1 trade data, the open/close distinction is unobservable. On this DEX it's present in every event, regardless of mixed-side rate.

## Wyckoff threshold calibration — no changes needed

| Label | Uses `is_open`? | Threshold | Status |
|-------|----------------|-----------|--------|
| Absorption | No (uses evr, log_return std, log_total_qty) | — | Unaffected |
| Buying/Selling Climax | Diagnostic context only, not a hard gate | — | Unaffected |
| Spring | Yes: `is_open_at_min > 0.5` | With bimodal `is_open`, reduces to "was the minimum-price event an open?" | **This is the correct Wyckoff semantic** — at a spring low, the Composite Operator should be opening longs. Threshold intact. |

Observed Step 0 frequencies (with proxy): absorption 2-11%, spring 0.6-12.9%, climaxes 0.01-0.21%. Consistent with Round 5 estimates. The 59% error did not corrupt label frequencies.

## Effort_vs_result and climax_score

**`effort_vs_result`** — `total_qty` is side-agnostic sum of fill sizes. A mixed-side event with 0.5 open-long + 0.5 open-short has the same total_qty as a single-side 1.0 fill. Unaffected.

**`climax_score`** — z-scores over rolling 1000-event window. If multi-fill collapse produces fewer, larger events, the qty_zscore distribution shifts upward, making climax detection **more sensitive**. Net positive.

## Falsifiability under council-5 challenge

**Challenge:** bimodal `is_open` carries 1 bit per event. How does that support a rich Composite Operator footprint?

**Answer:** Richness comes from the **sequence**. A 200-event window = 200 bits. Wyckoff signatures are **temporal patterns**:
- Absorption: sustained `is_open≈1` over 50-100 events at support
- Distribution: sustained `is_open≈0` at resistance
- Spring: `is_open=0` dominant during shakeout, sudden spike to `is_open=1` at low

Falsification test: absorption vs distribution windows must cluster separately in frozen embedding space (Wyckoff label probe accuracy > chance on April 1-13 hold-out).

**Weakened claim (honest):** The original hypothesis that `is_open` would be "continuously varying" is wrong. The actual mechanism is "unambiguous directional declaration per event." This is a cleaner mechanism than the spec implied.

## Step 1 falsification checks

Council-4 adds four checks before training:

1. **Distribution shape of `is_open` per symbol** — confirm bimodal for illiquid, more continuous for BTC/ETH.
2. **Autocorrelation half-life of `is_open`** — CLAUDE.md cites "20 trades"; this may have been measured pre-dedup. Re-measure under corrected pipeline.
3. **Correlation between `is_open=1` and log_return recovery at spring lows** — if absent, spring's `is_open` condition is purely definitional and cannot be probed.
4. **`is_open` asymmetry during climaxes** — buying climax should show elevated `is_open`, selling climax depressed. If both near 0.5, the climax hypothesis for `is_open` fails.

## Required spec edits

Same three text corrections council-2 identified (spec §Order Event Grouping line 47, CLAUDE.md gotchas #3 and #19). **No threshold changes.**

## Summary

The 59% → 3-16% correction is purely textual; `is_open` semantics, Wyckoff thresholds, and all load-bearing features remain valid. Four Step 1 falsification checks added.
