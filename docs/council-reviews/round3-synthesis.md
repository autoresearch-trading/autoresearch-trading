# Round 3 Council Review — Synthesis

**Date:** 2026-04-02
**Reviewers:** Council 1-6 (all)
**Spec:** `docs/superpowers/specs/2026-04-01-tape-reading-direct-spec.md`

---

## Executive Summary

The spec is fundamentally sound. The 18-feature representation, dilated CNN architecture, multi-horizon labels, and walk-forward evaluation are all well-designed. However, Round 3 identified **3 critical bugs**, **5 high-priority fixes**, and **8 medium-priority improvements** across methodology, features, architecture, and numerical safety.

**The single most urgent action:** Designate the April hold-out window (suggest April 14+) before any April data is viewed. This is time-sensitive.

---

## CRITICAL — Must fix before any code runs

### C1. kyle_lambda uses Δvwap instead of Δmid (Council-3)
**Bug:** The spec says `Cov(Δprice, signed_qty) / Var(signed_qty)` where "price" is implicitly vwap. Using Δvwap conflates intra-order book walk (already captured by feature 7) with the market maker's information-driven price update. Biases lambda upward for large orders even under pure noise trading.
**Fix:** Use `Δmid` (from aligned orderbook snapshot) instead of `Δvwap`.

### C2. Go/no-go gate logic is inverted (Council-1)
**Bug:** The AND condition means BOTH base rate ∈ 50±0.5% AND return < 2 bps must be true to stop. A horizon with 50.1% base rate but 2.1 bps return passes.
**Fix:** Change to OR logic. Either condition alone should trigger stop.

### C3. Linear baseline threshold (50.5%) is inside the null band (Council-1)
**Bug:** At N=6,000, the 95% CI under H0 is [48.7%, 51.3%]. The 50.5% gate will pass pure noise ~69% of the time per symbol-horizon. Across 100 tests, it will never trigger.
**Fix:** Raise to ~51.4% (Bonferroni-corrected significance boundary). Gate per-symbol (15+/25 above threshold), not aggregate.

---

## HIGH PRIORITY — Fix before implementation (Step 0/1)

### H1. Three features are not cross-symbol comparable (Council-2)
- `depth_ratio`: uses raw qty, not notional. BTC lots ≠ FARTCOIN lots.
- `kyle_lambda`: uses raw qty in signed_qty. Dimensionally incomparable across price levels.
- `cum_ofi_20`: accumulates raw notional without normalization.

**Fix:** Use notional (qty × price) for depth_ratio and kyle_lambda. Normalize cum_ofi_20 by rolling total notional volume.

### H2. April hold-out window not designated (Council-1)
**Risk:** Every day without designation risks progressive contamination through informal data inspection.
**Fix:** Designate April 14+ as untouched hold-out. April 1-13 available for development validation.

### H3. cum_ofi_20 definition is ambiguous (Council-2)
**Bug:** "Cumulative sum of OFI changes" could mean sum-of-OFIs (correct, Cont 2014) or sum-of-delta-OFIs (telescopes to two-point difference, losing all intermediate info).
**Fix:** Reword to "rolling sum of OFI (Cont 2014) over the last 20 book snapshots."

### H4. Trial count understated ~10-15x (Council-1)
**Bug:** DSR uses T=100, but actual trials include C sweep (400), lr sweep (300), architecture comparison (300), sequence length sweep (300) ≈ 1,600 total.
**Fix:** Implement trial_log.csv from experiment 1. Use total row count as T in DSR.

### H5. Walk-forward "4 folds" is arithmetically inconsistent (Council-1)
**Bug:** 120d train + 20d test × 4 folds = 200 days needed, but only 160 available.
**Fix:** Specify expanding-window design. Honestly acknowledge 2-3 truly independent test periods over 160 days.

---

## MEDIUM PRIORITY — Fix before full training (Step 3)

### M1. Replace pure GlobalAvgPool with concat[GAP, last_position] (Council-6)
The last CNN position has RF=253 — it already sees all 200 events. GAP averages this rich summary with incomplete-RF positions. concat[GAP, last_position] adds only 256 params.

### M2. Add residual connections for layers 3-6 (Council-6)
Same 64-channel throughout — zero extra parameters. Preserves local patterns through the large-dilation layers.

### M3. OneCycleLR with 30% warmup (Council-6)
Critical while BatchNorm statistics stabilize in early training. Replace flat LR sweep with OneCycleLR(max_lr=3e-4).

### M4. Horizon-specific label smoothing (Council-6)
Epsilon: 0.10 / 0.08 / 0.05 / 0.05 for 10/50/100/500 events. Addresses label noise directly. No change to loss function — only targets change.

### M5. Specify stride=200 for training samples (Council-5)
"Random offset" language is ambiguous. Adjacent overlapping windows share 60% label correlation at 500-event horizon. Stride=200 (non-overlapping inputs) is the right default.

### M6. Commit to pre-warming (not masking) rolling windows (Council-5)
Masking loses 4M day-open events and systematically biases against day-open dynamics. Pre-warming from prior day is causally clean. Only mask first calendar day per symbol.

### M7. Clarify effort_vs_result uses log_total_qty (Council-5)
The spec doesn't specify whether `log_qty` in feature 8 is raw or median-normalized. Must be median-normalized (feature 2) for cross-symbol comparability.

### M8. Increase embargo from 500 to 600 events (Council-1)
The 500-event embargo provides zero margin for off-by-one errors from random offset sampling. 600 absorbs edge cases at negligible cost.

---

## LOW PRIORITY — Enhancements for after prototype validates

### L1. Rename feature 7 from `price_impact` to `book_walk` (Council-3)
Eliminates terminological collision with the theoretical concept of price impact.

### L2. kyle_lambda window: 200 events instead of 50 (Council-2)
50 events spans only ~2.5s for liquid symbols — too short for a regime indicator. 200 matches the sequence length.

### L3. Add microprice_dev feature (Council-2)
Already implemented in prepare.py. Zero-cost, high theoretical value. Feature 19 candidate.

### L4. Add book_is_fresh binary flag (Council-2)
Helps CNN discount stale book features during volatility spikes. Feature 20 candidate.

### L5. Add funding_zscore (Council-2)
Already in data. Conditions 100-500 event horizon predictions with persistent 8-hour signal.

### L6. Augmentation: additive noise (5% σ) + OB feature dropout (p=0.15) (Council-6)
OB dropout teaches model to predict from trade features alone when book is stale.

### L7. Shared neck: Linear(128→64)+ReLU before per-horizon heads (Council-6)
Adds 8.4K params. Allows non-linear feature combinations before horizon projection.

### L8. NaN assertion checkpoints at 3 points in prototype (Council-5)
After feature stacking, after first conv, before loss. A single NaN corrupts the entire model.

---

## Explicitly NOT Recommended

| Proposal | Why not | Source |
|----------|---------|--------|
| Focal loss | Labels are ~50/50, not imbalanced. Focal would down-weight the clearest signals. | Council-6 |
| Hierarchical heads (500→100) | 500-event prediction is noisier — conditioning propagates noise into shorter horizons. | Council-6 |
| Time reversal augmentation | Breaks is_buy/is_open semantics. | Council-6 |
| Product/geometric mean for climax_score | Violates "both must be extreme" — allows compensation across dimensions. | Council-4 |
| Signed effort_vs_result | Other signed features (is_buy, log_return, trade_vs_mid) already carry direction. Unsigned is defensible. | Council-4 |

---

## Consensus Validations (confirmed correct by multiple reviewers)

| Feature/Design | Validated by | Status |
|----------------|-------------|--------|
| min(z_qty, z_return) for climax_score | Council-3, 4 | Correct gate operator |
| is_open as Composite Operator footprint | Council-3, 4 | Uniquely powerful for DEX perps |
| 100-event primary horizon | Council-1, 3 | Defensible (beyond bounce, within half-life, below macro) |
| Asymmetric loss weights 0.10/0.20/0.35/0.35 | Council-4, 6 | Correct |
| LayerNorm in conv body | Council-5, 6 | Regime-invariant, correct |
| Hold-out symbol test (FARTCOIN) | Council-1, 3 | Correct diagnostic for exchangeability |
| Rolling 1000-event causal statistics | Council-1, 5 | No lookahead, correct |

---

## Dissenting Opinions / Open Questions

1. **Council-5 says 85K params may overfit** (effective samples ~60K after autocorrelation), while **Council-6 says 94K params is fine** (with augmentation and dropout). The truth will be empirical — monitor train/val gap.

2. **Council-2 suggests 200-event kyle_lambda window**, but **Council-3 accepts 50 events** as "short-term price impact coefficient." The window defines what the feature measures — 50 events = local aggressiveness, 200 events = regime indicator. Start with 50, consider 200 as a sweep parameter.

3. **Council-4 notes the CNN cannot detect springs/upthrusts without reference price levels**, but agrees this is acceptable for v1. Add "price position within recent range" only if Phase 1 underperforms on reversal events.

---

## Action Items for Spec Update (Round 4)

Sorted by implementation order:

1. **Designate April hold-out: April 14+ untouched** (do this now)
2. Fix go/no-go gate: AND → OR logic
3. Fix linear baseline threshold: 50.5% → 51.4%, per-symbol (15+/25)
4. Fix kyle_lambda: Δvwap → Δmid, use signed_notional
5. Fix depth_ratio: raw qty → notional
6. Fix cum_ofi_20: clarify definition, normalize by rolling notional
7. Clarify effort_vs_result uses log_total_qty (median-normalized)
8. Commit to pre-warming (remove "or mask")
9. Specify stride=200 for training samples
10. Increase embargo: 500 → 600 events
11. Walk-forward: specify expanding window, honest fold count
12. Add trial_log.csv requirement
13. Architecture: concat[GAP, last_position], residuals L3-6, OneCycleLR, label smoothing
14. Rename feature 7: price_impact → book_walk
