# Goal-A cascade-precursor direction LR — fade vs continuation

**Question.** Conditional on the stage-2 cascade-onset model predicting a cascade is likely, can we predict its direction (long vs short) from the same 83-dim flat baseline + the cascade-onset confidence?  Without direction the cascade-onset AUC=0.817 / top-1% precision=27.8% signal is not tradeable — you cannot take a position.

**Hard constraints.** April 14+ untouched.  H500 only (n=~73 cascades; H100 has n=20, too underpowered).  Sample size honest: with leave-one-day-out CV across 7 April-diagnostic dates the CIs are wide.  If |marginal_p_positive - 0.5| > 0.10, LR may be exploiting the marginal not the conditional — we report a majority-baseline AUC for comparison.

## 1. Marginal direction asymmetry

P(forward_log_return > 0 | real_cascade_h500) = **0.7671** (n_cascades = 73)

|marginal - 0.5| = **0.2671** → asymmetric (LR may exploit marginal — must beat majority baseline)

## 2. Direction LR AUC at H500 on cascade-likely subset

Subset = top-5% by `pred_proba_h500` from stage-2 → 90 windows (16 real cascades).

| metric | value |
|---|---|
| Direction LR AUC (realized direction) | 0.4413 |
| 95% bootstrap CI (lo) | 0.3288 |
| 95% bootstrap CI (hi) | 0.5506 |
| Majority-class baseline AUC | 0.5000 |

**Verdict:** Direction LR AUC CI does not cleanly exclude 0.5 → direction is NOT predictable from this representation.

## 3. Realized vs overshoot direction agreement

Agreement(realized ⇔ overshoot, both binary) = **0.8125**, P(overshoot up) = **0.7500** → continuation-dominated (overshoot persists to horizon end).

## 4. Conditional headroom

| component | value |
|---|---|
| P(top-1% pred_proba_h500) | 0.0100 |
| P(LR confidence > 0.55 \| cascade-likely) | 0.9444 |
| Triggers per day (universe-pooled) | 2.4286 |
| E[\|forward_log_return\| at H500 \| cascade-likely] (bps) | 80.96 |
| Direction accuracy at confidence threshold | 0.4824 |
| Gross per trigger (bps) | -2.86 |
| Cost per trigger (bps; 4bp fee + 1bp slip per side, both legs) | 10.00 |
| Net per trigger (bps) | -12.86 |
| **Per-day expected gross (bps)** | **-31.2250** |

**Headroom verdict:** NOT tradeable (per-day gross is non-positive)

## 5. One-paragraph verdict

The cascade-onset signal (AUC=0.817 at top-1% precision=27.8%) appears to be DIRECTIONLESS at H500 in this representation — the LR cannot predict whether a flagged cascade overshoot goes long or short.  Without direction the strategy reduces to a coin flip with round-trip costs, which is unprofitable.  Until a representation provides direction skill, the cascade-precursor signal is statistically interesting but untradeable.

---

_Wall-clock: 5.7 s._

