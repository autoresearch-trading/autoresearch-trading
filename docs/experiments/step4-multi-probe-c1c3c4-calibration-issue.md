# Multi-Probe Battery (C1, C3, C4) — Pre-Registration Calibration Issue

**Date:** 2026-04-26 (PM, post Gate 4 PASS)
**Status:** BLOCKER — battery as written produces 0/3 trivially due to label calibration mismatch
**Pre-registration referenced:** `docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md` (RATIFIED commit `c28bc17`)

## Summary

Before running any of C1/C3/C4 on encoder embeddings, I implemented the four window-level Wyckoff labels (`is_absorption`, `is_buying_climax`, `is_selling_climax`, `is_stressed`) per the pre-reg's formulae and computed their empirical positive rates on the Feb+Mar held-out feature data **only** (no encoder forward pass yet). On all 24 non-AVAX symbols:

| Label | Empirical positive rate (Feb+Mar held-out) | Pre-reg threshold |
|-------|---------------------------------------------|-------------------|
| `is_absorption` | **0.000 across all 24 symbols** | (none) |
| `is_buying_climax` | **0.000 across all 24 symbols** | (none) |
| `is_selling_climax` | **0.000 across all 24 symbols** | (none) |
| `is_stressed` | **0.000 across all 24 symbols** | (none) |

The C4 climax-event seed (`max(climax_score over last 50 events) > 3.0`) also fires on **0 windows out of all 24 symbols × Feb+Mar shards**. Empirical max climax_score across all symbols' Feb shards is **0.256** (PENGU); typical max is 0.05–0.13. The pre-reg threshold of 3.0 is **roughly 30× larger than the actual empirical maximum**.

This is decisive: running the multi-probe battery as pre-registered will trivially return underpowered/skip on all 4 conditions in C1+C3 and produce zero candidate events for C4. **Stop B (≥0 of 4 conditions pass) would fire by construction, not by encoder phenomenology.**

I did NOT run any encoder forward pass against these labels. The discovery is purely on the label side.

## Root cause

The pre-reg's labels are window-level summaries. Three issues compound:

### 1. `climax_score` empirical scale

`climax_score` is implemented as `clip(min(z_qty, z_ret), 0, 5)` per `tape/features_trade.py` lines 121-125, with rolling-1000 σ on log_total_qty and on |log_return|. The MIN operator combined with the rolling baseline produces an empirical distribution where:

- p99 ≈ 0.005–0.020 across symbols
- max ≈ 0.05–0.26 across symbols × all Feb shards
- The 2.5 / 3.0 thresholds are essentially never reached

Step0's per-event scoring uses the same `climax_score`, and the recorded per-event rates in `docs/experiments/step0-data-validation.json` confirm:

| Label | Per-event rate (step0) |
|-------|------------------------|
| `buying_climax_freq` | 0.0002–0.0021 |
| `selling_climax_freq` | 0.0001–0.0020 |
| `stress_freq` | ~0.0000 universally |

So the climax/stress labels are **already extremely sparse at per-event level** in step0's implementation. The per-window aggregation in the pre-reg kills them entirely.

### 2. Absorption's low-vol-ratio criterion

The pre-reg requires `std(log_return[-100:]) < 0.5 * rolling_std_log_return`. I interpreted `rolling_std_log_return` as `std(log_return)` over the FULL 200-event window (the only within-window baseline available). With this:

- BTC: `frac(std_recent < 0.5 * std_full) = 0.16%`
- Across all 24 symbols × Feb: `frac < 1.1%`

The criterion requires the recent half's variance to be 4× lower than the full window's variance — only happens when the early half of the window had a large vol spike. This is a rare regime by construction.

The other absorption criteria fire at meaningful rates:
- `mean(effort_vs_result[-100:]) > 1.5`: 6–73% per symbol
- `mean(log_total_qty[-100:]) > 0.5`: 4–91% per symbol

The conjunction is killed by the low-vol-ratio third condition.

### 3. Step0's per-event vs pre-reg's per-window operationalization

Step0's `is_absorption` (per-event) fires at **3–11% across symbols** (median ~5% — meaningful). It uses *rolling* statistics computed at each event over a long history (1000 events). The pre-reg adapted this to per-window via `[-100:]` notation, but did NOT specify what `rolling_std_log_return` resolves to in the window-level setting. My within-window interpretation produced a much stricter criterion than step0's per-event one.

## Three paths forward

I considered three operationalizations. None is risk-free vs council-5's pre-registration discipline; the user / council needs to choose.

### Path A — Run as-written, accept 0/3 trivial Stop B

**Action:** Execute C1/C3/C4 unchanged. All conditions return underpowered → 0 of 4 pass → Stop B → write the "+1pp ceiling, not phenomenologically rich" negative result.

**Pro:** Maximally faithful to council-5's "sign before measurement, after is post-hoc" rule. No amendment needed.

**Con:** The conclusion would be **misleading**. Stop B's writeup says "encoder is at +1pp ceiling AND extracts only direction-flavored signal" — but we'd be reporting that without ever testing the encoder against any phenomenological label that actually fires. The 0/3 is a feasibility failure of the labels, not an encoder failure. Calling this a Stop B negative result conflates two distinct claims.

**Council-5 reading:** would still endorse this if asked "does discovering label-distribution sparsity AFTER ratification count as post-hoc?" — strict reading is yes, the ratification covered the operationalization too. But a charitable reading sees this as a **pre-experiment feasibility check** that didn't touch the encoder, analogous to discovering the disease never appears in the population.

### Path B — Amend with empirical-quantile thresholds

**Action:** Re-define the labels as quantile-based:
- `is_absorption = top-5%-of-windows-by-effort_vs_result.mean()-AND-log_total_qty.mean()` (or similar joint quantile)
- `is_buying_climax = top-X%` of windows by `max(climax_score)` AND positive return
- etc.

**Pro:** Guarantees non-zero positive rate; lets the encoder probe actually run; calibration is principled (matches council-4's ~5% intent).

**Con:** Quantile thresholds are computed from the held-out distribution itself → leaks distributional information into the label definition. Council-5 would call this "labels conditioned on test data." Even if the label set is fixed before encoder evaluation, the quantile is fit on Feb+Mar — which is the same window we evaluate on.

### Path C — Amend by porting step0's per-event mechanic to per-window

**Action:** Re-implement labels to mirror step0_validate.py's per-event mechanic:
- Compute per-event labels using rolling-1000 statistics.
- Aggregate to per-window using "≥X of last 100 events fired the label" (e.g., X=20 for absorption to require 20% sustained, X=1 for climax to require any peak event).
- Document the X choices ex-ante.

**Pro:** Best-of-both — stays faithful to council-4's master Wyckoff phenomenology (the per-event labels are the spec's canonical operationalization), but lifts to window-level via a stable mechanic. Non-zero positive rates by construction (step0 confirmed absorption at ~5% per event → with X=20 aggregation, expect ~1-3% positive windows).

**Con:** Still an amendment after ratification. Requires explicit aggregation thresholds (X) which are choices that council-4 didn't make ex-ante. Council-5 would scrutinize the X choices for backdoor calibration.

Climax/stress labels would STILL underfire because the underlying per-event rates are 0–0.2%; no per-window aggregation rescues them. Path C effectively reduces the C3 battery from 4 labels to 1 (only absorption is meaningfully testable).

## Recommendation

I recommend **Path C, with explicit ex-ante aggregation rules and council-1 + council-5 sign-off**, but I am **NOT** taking that step unilaterally — it is a binding amendment.

Specifically:
1. `is_absorption_window = (≥20 of last 100 events satisfy step0's per-event is_absorption criterion)` — gives ~1-5% positive rate per symbol based on step0 data
2. `is_buying_climax_window = (any event in last 50 satisfies step0's per-event is_buying_climax criterion)` — but the per-event rate is 0.04% so even union-aggregation fires <2% of windows; might fail underpowered guard for many symbols
3. `is_selling_climax_window` = ditto, same risk
4. `is_stressed_window` = step0's per-event stress is ~0% across all symbols → CANNOT recover; should be dropped from the battery

Effective C3 battery becomes: 1 strong label (absorption) + 2 marginal labels (climaxes) + 1 dead label (stress). Pre-reg required ARI ≥ 0.05 on ≥2 of 4 labels — with stress dead and climaxes marginal, the achievable bar is "absorption + one of the two climaxes." This is harder than the original spec.

For C4 (embedding trajectory): the seed criterion `climax_score > 3.0` cannot be satisfied empirically. **Either drop C4 entirely, or amend the seed to a percentile-based criterion** (e.g., "top 10 windows ranked by max(climax_score) over last 50 events, requiring climax_score > 95th-percentile-on-symbol-day"). The latter is a Path B-flavored amendment for C4 specifically.

## Decision needed from user

Before any encoder probe runs, please choose:

- **(A)** Run as-written, accept Stop B trivially. Negative result is honest about the literal pre-reg but conceptually mislabels what was tested.
- **(B)** Quantile-based label amendment for all 3 conditions (acknowledges leakage but keeps battery operational).
- **(C)** Step0-port per-event aggregation amendment (best phenomenological fidelity but reduces C3 to ≤2 viable labels and forces a separate C4 seed amendment).
- **(D)** Pause the multi-probe battery; declare Gate 2 + Gate 4 PASS the publishable end-state and write the writeup directly from those (skipping C1/C3/C4 entirely with explicit acknowledgment that the labels were undefinable in the operational data).

My weak preference is (D) — the pre-reg's authors did not measure the empirical scale of their own labels before ratification, and trying to salvage the battery via amendment will compound the falsifiability cost. Gate 2 FAIL + Gate 4 PASS is already a coherent, publishable finding: *the encoder's directional signal is stable across training period halves but not amplifiable by supervised fine-tuning at lr=5e-5*. C1/C3/C4 were intended to falsify or strengthen this; if they cannot be operationally run, that is itself a finding about the limits of the spec's phenomenological labels on this data.

But I will defer to user + council-5 for the call.

## Diagnostic data appended below for reference

Empirical per-window positive rates (Feb 2026, first 5 shards per symbol, stride=200, my pre-reg implementations):

```
symbol   n_windows  absorp_rate  climax_max  evr_high_rate  qty_high_rate  lowvol_rate
2Z             90      0.0000      0.0486         0.567         0.078      0.0000
AAVE          103      0.0000      0.0993         0.194         0.641      0.0000
ASTER         112      0.0000      0.2135         0.438         0.902      0.0000
BNB           235      0.0000      0.0354         0.068         0.289      0.0000
BTC           615      0.0000      0.0551         0.231         0.299      0.0016
CRV            90      0.0000      0.1339         0.533         0.100      0.0111
DOGE          102      0.0000      0.1288         0.196         0.588      0.0000
ENA           141      0.0000      0.0763         0.475         0.674      0.0000
ETH           585      0.0000      0.0806         0.265         0.513      0.0017
FARTCOIN      101      0.0000      0.0811         0.277         0.545      0.0000
HYPE          480      0.0000      0.0972         0.023         0.054      0.0021
KBONK          97      0.0000      0.0798         0.186         0.412      0.0000
KPEPE          99      0.0000      0.1228         0.293         0.737      0.0000
LDO            92      0.0000      0.1265         0.239         0.207      0.0109
LINK          113      0.0000      0.0457         0.301         0.743      0.0000
LTC            93      0.0000      0.0905         0.183         0.290      0.0000
PENGU          94      0.0000      0.2560         0.170         0.489      0.0000
PUMP          121      0.0000      0.0636         0.727         0.884      0.0000
SOL           379      0.0000      0.0615         0.193         0.406      0.0026
SUI           111      0.0000      0.0688         0.351         0.910      0.0000
UNI            93      0.0000      0.1195         0.183         0.323      0.0000
WLFI           98      0.0000      0.1232         0.633         0.561      0.0000
XPL            90      0.0000      0.0894         0.567         0.178      0.0000
XRP           326      0.0000      0.0362         0.061         0.040      0.0000
```

`climax_max` column shows the maximum of `max(climax_score over last 50 events of each window)` across all sampled windows for that symbol. The pre-reg threshold of 3.0 (or 2.5 for buying/selling climax) is **never approached**.

The `lowvol_rate` column shows the rate at which `std(log_return[-100:]) < 0.5 * std(log_return over full window)`. Maximum across all 24 symbols is 1.11%; this is the binding criterion that kills the absorption label.

## What I commit to NOT doing

- Run any encoder forward pass against these labels until calibration is resolved.
- Silently amend the pre-reg.
- Run a Path B/C amendment without explicit user + council-5 sign-off.
- Cherry-pick which labels to test based on pilot positive rates.

## Anti-amnesia clause

Whatever path is chosen here MUST be recorded in this file with the choice, the council sign-off (or absence thereof), and the resulting amended pre-reg. Future writeups citing C1/C3/C4 results MUST reference this calibration-issue document by path and disclose the amendment history.
