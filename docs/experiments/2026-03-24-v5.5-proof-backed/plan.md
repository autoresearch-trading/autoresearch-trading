# v5.5 Proof-Backed Experiment Plan

## Goal
Test the v5.5 hypothesis on the current cached v9 feature set:

- target `Sortino > 0.15`
- target `passing >= 21/25`
- avoid any feature-cache rebuild

Inference from T16 plus the observed v5/v9 gap: v9's `2122` trades are likely to the right of the optimal trade-count cutoff. The plan therefore removes complexity first, then removes the redundant second gate, then tightens frequency only if the simpler stack still overtrades.

## Code-Grounded Constraints

### Safe to change without invalidating cache
- `train.py`
  - model class selection
  - `WINDOW_SIZE`
  - `MIN_HOLD`
  - `MAX_HOLD_STEPS`
  - `BEST_PARAMS["fee_mult"]`
  - `BEST_PARAMS["r_min"]`
  - `BEST_PARAMS["vpin_max_z"]`
- Reason: `make_env()` applies `window_size` and `min_hold` after loading cached features, and `evaluate()` applies `r_min` / `vpin_max_z` as inference-time gates.

### Do not change in this plan
- `prepare.py`
  - `USE_V9`
  - `_FEATURE_VERSION`
  - feature computation / normalization
- `TRADE_BATCH`
- Reason: cache keys are `symbol + start + end + trade_batch + _FEATURE_VERSION`; changing `trade_batch` or feature version triggers rebuilds.

## Fixed Throughout This Plan
- feature set: v9, 5 features
- `TRADE_BATCH = 100`
- `WINDOW_SIZE = 75`
- `MAX_HOLD_STEPS = 300`
- training budget and seed counts unchanged

Rationale:
- Keep `WINDOW_SIZE = 75` fixed so Phase 1 isolates classifier complexity, not context length.
- Keep `MAX_HOLD_STEPS = 300` fixed so Phase 3 isolates label selectivity, not label horizon.

## Baseline
Run `0` is the current control from repo head:

| Run | Delta vs current head | Expected use |
|---|---|---|
| 0 | none | Reproduce the current v9-ish control: Hybrid, `MIN_HOLD=200`, `fee_mult=3.0`, `r_min=0.7`, `vpin_max_z=1.5` |

Expected outcome:
- passing stays high
- Sortino stays near zero
- trade count stays too high relative to v5

## Phases

All runs inherit the prior phase winner. Within each phase, only the listed variable changes.

### Phase 1: Simplify the classifier
Theorem basis:
- `T19 (ApproxGainVsEstimationCost)`: a more complex model only wins when approximation gain beats estimation cost.
- Repo evidence already points the same way: flat MLP outperformed more complex temporal models historically.

| Run | Variable | Config change | Expected outcome |
|---|---|---|---|
| 1 | classifier | `HybridClassifier -> DirectionClassifier` | Sortino up or flat, passing flat/slightly down, runtime drops from about `80m` to about `30m` |

Decision logic:
- Keep `DirectionClassifier` if Sortino improves.
- Also keep it if Sortino is within `0.02` of control and passing stays `>= 20/25`; the runtime reduction makes all later phases cheaper.
- Revert to `HybridClassifier` only if passing collapses or Sortino clearly regresses.

### Phase 2: Remove the redundant second gate
Theorem basis:
- `T17 (GateSortinoSemivariance)`: a gate can increase passing count while decreasing portfolio Sortino.
- `T22 (DualGateDependence)`: if VPIN and Hawkes proxy the same latent toxicity/regime factor, the second gate is redundant.

| Run | Variable | Config change | Expected outcome |
|---|---|---|---|
| 2 | VPIN gate | `vpin_max_z: 1.5 -> 0.0` | Portfolio Sortino improves if the VPIN paradox is active; passing may stay flat or dip slightly |

Decision logic:
- Keep `vpin_max_z = 0.0` if Sortino improves and passing stays `>= 20/25`.
- If Sortino improves but passing slips to `19/25`, still carry the no-VPIN winner into Phase 4 as the higher-quality stack.
- Restore VPIN only if removing it worsens both Sortino and passing.

### Phase 3: Re-establish the trade-count optimum with label selectivity
Theorem basis:
- `T16 (OptimalTopNTrades)`: with decreasing marginal edge, there is a unique optimal trade count.
- `T18 (FrequencyOptimum)`: when edge is heterogeneous, the cutoff logic dominates the "every trade helps" homogeneous-edge result.

Operationalization in current code:
- `fee_mult` changes the triple-barrier threshold used to label long/short samples.
- With `FEE_BPS = 5`, the current threshold is `2 * 5bps * fee_mult = 10bps * fee_mult`.
- So `fee_mult=3.0` means about `30bps`; `4.0` means about `40bps`; `5.0` means about `50bps`.

| Run | Variable | Config change | Expected outcome |
|---|---|---|---|
| 3a | `fee_mult` | `3.0 -> 4.0` | Trades down, PF and win rate up, Sortino up if v9 is overtrading |
| 3b | `fee_mult` | `3.0 -> 5.0` | Stronger trade suppression; more likely to improve Sortino, more likely to hurt passing |

Decision logic:
- Prefer the smallest `fee_mult` that improves Sortino while keeping `passing >= 20/25`.
- If both improve Sortino, prefer higher passing first, then fewer trades.
- If both reduce passing below `20/25`, revert to the Phase 2 winner and use Phase 4 as the recovery lever instead of pushing selectivity harder.

### Phase 4: Shorten min_hold only as a coverage-recovery lever
Theorem basis:
- `T20 (MinHoldSurface)`: under zero drift, `min_hold` does not improve win probability; it only changes achievable frequency.
- Therefore `min_hold` should not be the primary alpha lever. Use it only to recover coverage after Phases 2-3.

| Run | Variable | Config change | Expected outcome |
|---|---|---|---|
| 4 | `MIN_HOLD` | `200 -> 100` | Passing and trade count rise; Sortino only improves if there is real short-horizon drift left after filtering |

Decision logic:
- Run Phase 4 only if the Phase 3 winner has `Sortino >= 0.10` but `passing < 21/25`, or if trade count was cut so hard that the strategy is clearly under-covered.
- Keep `MIN_HOLD=100` only if it restores `passing >= 21/25` without knocking Sortino back below `0.15`.
- Otherwise keep `MIN_HOLD=200`.

### Optional Phase 5: Retune the single surviving Hawkes gate
Theorem basis:
- `T17` implies there is a threshold trade-off: gating helps only while the quality gain beats the `sqrt(phi)` pass-rate penalty.

Use this only after VPIN is removed.

| Run | Trigger | Config change | Expected outcome |
|---|---|---|---|
| 5a | `regime_filter_rate > 0.56` | `r_min: 0.7 -> 0.6` | Less over-filtering, more passing, modest Sortino recovery |
| 5b | `regime_filter_rate < 0.25` and trades still too high | `r_min: 0.7 -> 0.8` | Stronger single-gate frequency control |

Decision logic:
- Run only one branch: `5a` or `5b`, never both in the same pass.
- Keep the new `r_min` only if Sortino improves and passing stays `>= 20/25`.

## Success Criteria

### Primary
- `Sortino > 0.15`
- `passing >= 21/25`

### Secondary
- total trades materially below v9's `2122`
- PF moves back toward v5's `1.59`
- win rate moves back above `55%`

If multiple runs satisfy the primary goal, rank them:
1. higher Sortino
2. higher passing
3. fewer trades

## Budget

### Core plan
- Run `0` control: `1 x ~80m`
- Runs `1`, `2`, `3a`, `3b`: `4 x ~30m`
- Core total: `5 runs`, about `200 minutes` total, about `3h20m`

### Likely full plan
- Add Phase `4`: `+1 x ~30m`
- Full likely total: `6 runs`, about `230 minutes` total, about `3h50m`

### Optional extension
- Add one Hawkes retune run from Phase `5`: `+1 x ~30m`
- Max planned total: `7 runs`, about `260 minutes` total, about `4h20m`

## Files To Modify

### Required
- `docs/experiments/2026-03-24-v5.5-proof-backed/plan.md`
- `train.py`

### Recommended `train.py` changes before running experiments
- add a small `model_kind` switch so `train_one_model()` can instantiate `DirectionClassifier` or `HybridClassifier` without hand-editing the model line each run
- keep experiment knobs in one config block:
  - `MIN_HOLD`
  - `WINDOW_SIZE`
  - `MAX_HOLD_STEPS`
  - `BEST_PARAMS["fee_mult"]`
  - `BEST_PARAMS["r_min"]`
  - `BEST_PARAMS["vpin_max_z"]`
- print the chosen `model_kind` and gate settings in the final summary so run logs are comparable

### Do not modify for this experiment
- `prepare.py`
  - no feature changes
  - no `_FEATURE_VERSION` bump
  - no `TRADE_BATCH` changes

## Recommended Stop Rule
Stop the sequence as soon as one run clears:
- `Sortino > 0.15`
- `passing >= 21/25`
- trades clearly below control

At that point, record the winner and only open a new branch if there is a specific theorem-backed reason to do so.
