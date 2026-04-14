# Council-5 (Practitioner Quant) — Step 0 Skeptic Audit

**Date:** 2026-04-14
**Reviewer:** Council-5 (Falsifiability, Implementation Risk)
**Sources:** spec, CLAUDE.md, `docs/experiments/step0-data-validation.md` + JSON

---

## One-sentence verdict

The **AVAX cross-symbol contrastive pair contamination is the only hard correctness bug** that must be fixed before touching data, but the **single most dangerous unverified assumption is H500 base-rate stationarity across the 160-day training window** — if base rates drift > 3pp intra-period, Gate 4 loses its ability to distinguish representation non-stationarity from label distribution shift.

---

## 🔴 Hard bug (one-line fix, blocks Step 1)

**AVAX contamination in cross-symbol contrastive pairs.** Spec §Training Strategy lists AVAX among liquid symbols for contrastive soft positives:

> "Cross-symbol positive pairs: same-date, same-hour windows from different liquid symbols (BTC, ETH, SOL, BNB, **AVAX**, LINK)"

But AVAX is the pre-designated held-out symbol for Gate 3 (cross-symbol transfer). If AVAX windows participate in contrastive pairs during pretraining, Gate 3's falsifiability is destroyed before training starts.

**Fix:** Remove AVAX from the liquid-symbol contrastive pair list. Keep BTC, ETH, SOL, BNB, LINK. Replace AVAX with another liquid symbol if a 6-symbol anchor set is required — candidate: DOGE or LTC (liquid, high-volume, not held-out).

---

## Forensics: how "59% mixed-side" became load-bearing spec text

Someone measured mixed-side rate on dedup-with-side data (or undeduped data) and wrote the result into the spec as expected behavior. The dedup **instruction** in gotchas #3/#19 was correct; the **expected observable consequence** was generated with the wrong key. The spec got validated against a broken intermediate result.

**Process lesson:** Every "expect X%" claim in the spec should be traced to a specific measurement. If it can't be, it's a hypothesis, not a validated fact, and should be flagged as such.

---

## Unverified numeric claims in the spec (priority-ordered)

| # | Claim | Status | Risk |
|---|-------|--------|------|
| 1 | **AVAX excluded from cross-symbol contrastive** | ❌ Spec contradicts itself | Gate 3 invalidated |
| 2 | "~28K events/day on BTC" | ❌ Step 0 measured 2,231–22,971/day (bear regime) | Total window count overstated; compute budget wrong |
| 3 | "200 events ≈ 10 minutes on BTC" | ❌ At observed event rates, 100-143 min | Architecture claim about pattern duration coverage wrong |
| 4 | "~3.5M training windows at stride=50" | ❌ Likely 1-2M | Epoch timing / batch-count estimates off |
| 5 | **H500 base rates drift-explained** | ⚠️ 2Z/CRV/UNI/LDO 8-9pp below 50% — alt-specific collapse, not BTC-drift | Gate 4 can't distinguish representation drift from label drift |
| 6 | Rolling base rate at H500 stationary | ❌ Not measured | Same as #5 |
| 7 | Stress fires 3-8% with full OB | ❌ Not measured — joint 90th pct AND 90th pct condition may fire <1% | Stress probe has no statistical power |
| 8 | Informed flow fires 2-15% | ❌ Not measured with full features | Label may be unusable |
| 9 | Spring rate biologically plausible | ❌ BTC proxy shows 12.2% — too loose, catching trend noise | Spring probe encodes bear-trend bias, not Wyckoff |
| 10 | Climax windows span 15+ dates per symbol | ❌ Not measured | Probe may memorize 2-3 crash events, not structure |
| 11 | 5-event MEM block masking non-trivial | ❌ Not tested against feature autocorrelation | If r>0.8 at lag 5, blocks need 10-15 |
| 12 | OB cadence ~24s | ✅ Validated (all 25 symbols p50 23.67-24.0s) | — |
| 13 | 10 bid + 10 ask levels | ✅ Validated | — |

---

## On H500 base rates — is "bear market" sufficient?

**Directionally yes, quantitatively no.**

Drift model: `base_rate ≈ 0.5 + drift_per_H500 / (2 * sigma_per_H500)`. For BTC's -36% over ~160 days and typical H500 volatility, expected base rate is ~48%. BTC's observed 48.3% is consistent.

**Outliers that are NOT drift-explained:** 2Z 41.0%, CRV 41.6%, UNI 41.9%, LDO 42.5%. These are 8-9pp below 50% — would require 60-80% idiosyncratic drawdowns, plausible for alts but **not verified**. Builder-8 attributed all deviations to BTC drift, which assumes unit beta across 25 symbols (false by construction).

**What this means for training:** For H500 on high-downtrend alts, a "predict down always" model gets 58-59% accuracy. The 51.4% Gate 1 threshold becomes trivially beatable by majority-class prediction. **H500 is not a fair evaluation target unless class-balanced.** The spec's primary Gate 1 horizon (H100) is fine because H100 base rates are close to 50%.

### Minimum falsification checks before Step 1

1. **Per-symbol total return Oct 2025 → Mar 2026** — compare to drift-implied vs observed H500 base rate. Gap >2pp warrants investigation.
2. **Rolling 30-day H500 base rate** for BTC, ETH, SOL, CRV, 2Z. Any >3pp intra-period shift means Gate 4 must use balanced accuracy / F1, not raw accuracy, at H500.
3. **Label-generation day-boundary safety** — verify pipeline enforces "no window crossing day boundaries" at label stage, not just window stage.
4. **Label uses event-VWAP, not last-fill price** — confirm in code.
5. **H500 = 500 order events ahead, not 500 raw rows** — confirm in code.

---

## On Wyckoff labels — hard falsifiability thresholds for Step 1

| Label | Step 1 threshold | Action if violated |
|-------|-----------------|---------------------|
| **Stress** (log_spread > p90 AND |depth_ratio| > p90) | ≥ 0.5% firing on 20+/25 symbols | Recalibrate to p80 or single-feature trigger |
| **Informed flow** (kyle_lambda > p75 AND \|cum_ofi_5\| > p50 AND 3-snapshot sign consistency) | 2-15% firing on 20+/25 symbols | Tighten/loosen kyle_lambda percentile |
| **Spring** (min < -2σ AND evr > 1 AND is_open > 0.5 AND mean[-10:] > 0) | ≤ 8% per symbol | Tighten — BTC/ETH/SOL at 12%+ in proxy indicates trend noise, not Wyckoff |
| **Climax** (binary, both types) | ≥ 15 distinct calendar dates per symbol | Drop from probe, keep for qualitative illustration only |

---

## The single most dangerous unverified assumption (elaborated)

**H500 base-rate stationarity.** If CRV's H500 base rate shifts from 48% in Oct-Nov to 38% in Feb-Mar (plausible under idiosyncratic protocol risk in a crypto bear):

- Pretraining learns representations averaged over two distinct label distributions.
- Gate 4 (temporal stability: <3pp drop months 1-4 vs 5-6) will fail at H500 on 10+ symbols.
- **But the true cause is label distribution shift, not representation quality.**
- The council cannot distinguish "representations degraded" from "labels degraded" without this measurement.

Gate 4 is **not falsifiable as written for H500** until we measure rolling base rates. Required: compute 30-day rolling H500 base rate for BTC, ETH, SOL, CRV, 2Z before Step 1. If >3pp shift, Gate 4 must use balanced accuracy / F1 at H500, not raw accuracy.

---

## Required actions before Step 1

### Must fix (correctness bugs)
1. Remove AVAX from cross-symbol contrastive pair list
2. Correct spec text (3 locations per council-2)

### Must measure (falsifiability prerequisites)
3. Per-symbol total return + per-symbol rolling 30-day H500 base rate
4. Full-OB stress / informed_flow firing rates on 3 dates of BTC+ETH
5. Feature autocorrelation at lag 5 (for MEM block-size calibration)
6. Climax label date-diversity per symbol
7. Recalibrate spring threshold to cap at 8% firing rate

### Should measure (design integrity)
8. Actual events/day distribution per symbol (for window-count and event-duration claims)
9. Label-construction day-boundary and VWAP-vs-last-fill audits

Step 1 plan must fold these in as explicit sub-tasks, not assumed as already done.
