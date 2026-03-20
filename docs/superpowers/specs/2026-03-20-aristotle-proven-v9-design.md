# v9 Design: Aristotle-Proven Tape Reading Strategy

**Date**: 2026-03-20
**Status**: Design approved, pending implementation
**Branch**: `autoresearch/v9-hybrid-tcn`

## Motivation

The v8 architecture (46 features, window=50, ~692K params) regressed from v5 baseline (Sortino 0.230 → 0.137). Root cause: dimensionality overfitting — flat MLP can't separate signal from noise across 46 features.

We submitted 10 theorems to Harmonic's Aristotle formal theorem prover and obtained **133 formally verified results in Lean 4** (zero sorry statements). These proofs provide mathematically guaranteed design principles rather than empirical guesswork.

## Key Proven Results

### Features
- **(λ·OFI, TFI·|OFI|) is a sufficient statistic** for forward returns under the Kyle model (Theorem 1). Under Kyle's assumptions, 2 numbers capture all predictive information. **Caveat**: real DEX markets deviate from Kyle (multiple informed traders, no designated MM, funding mechanics). The 46→5 reduction is a hypothesis to be validated empirically at Step 1 of the rollout — if Sortino drops, the Kyle model's assumptions don't transfer and we add features back.
- **VPIN ∈ [0,1]**, quasi-convex, =0 iff balanced flow, =1 iff one-sided (Theorem 7).
- **Hawkes branching ratio n̂ = 1 - 1/√(Var(N)/E[N])** is a consistent estimator, in (0,1) for overdispersed counts (Theorem 8).
- **Avellaneda-Stoikov reservation price deviation** increases in inventory, volatility, and time; spread is always positive and decreasing in arrival intensity (Theorem 7).

### Architecture
- **Window ≥ 75 is the proved minimum** for TCN to extract temporal structure with AR(1) autocorrelation ρ=0.95 and n features (Theorem 5).
- **Hybrid weak dominance proved** (hybrid ≤ min(flat, tcn) risk), but strict dominance is false — if flat MLP already achieves Bayes risk, TCN adds nothing (Theorem 5).
- **Ensemble majority voting requires α > 1/2** to help, not α > 1/3. At α=0.4, ensembling hurts (Theorem 10).
- **v9 model complexity is 16.3% of v8** (375 vs 2300 effective dims, 55K vs 692K params) (Theorem 10).

### Labeling & Barriers
- **α_min(f) = 1/2 + 1/(2f)** is the minimum accuracy for profitability (Theorem 3, 9).
  - f=1.5 → α_min=83.3% (current setup — nearly impossible)
  - f=4.0 → α_min=62.5%
  - f=8.0 → α_min=56.25%
- **Correct Kelly optimal fee_mult**: f_opt = (p_win - p_loss)·(1-c) / ((p_win + p_loss)·c) (Theorem 3).
- **Two-pass Kelly iteration converges** to unique fixed point with geometric rate (Theorem 9, Banach fixed point).
- **Triple Barrier labels are mutually exclusive and exhaustive** when fee_mult·fee > 0 (Theorem 0).

### Regime Gating
- **SNR(r) = SNR_base/√(1-r)** is strictly increasing; unique r_min threshold exists (Theorem 6).
- **Regime gate provably improves accuracy**: E[Φ(SNR(r)) | r ≥ r_min] ≥ E[Φ(SNR(r))] (Theorem 9, monotone upper tail).
- **Gated strategy strictly beats ungated** when α_all < α* but α_regime > α* (Theorem 9).
- **Critical gating fraction**: φ_min = 1/k² — if gating improves returns by 50%, must pass ≥44% of timesteps (Theorem 10).

### Corrections Found
- **Hurst exponent can go negative** when S > R (Theorem 0) — our clip to [0,1] hides a bug.
- **f* formula was wrong** in original conjecture — Aristotle provided the correct derivation (Theorem 3).
- **Hawkes supercritical threshold is α/β=1**, not 0.5 (Theorem 4).
- **Current feature 44 (hawkes_ratio) is NOT the branching ratio** — it's buy/sell imbalance, not self-excitation.
- **Growth threshold ≠ accuracy threshold** — α_min is for expected value, log-growth threshold depends on both f and c (Theorem 9).

## Design

### Feature Engineering (5 features, v9)

Replace all 46 features with 5 formally verified features:

| # | Feature | Formula | Proved By |
|---|---------|---------|-----------|
| 0 | `lambda_ofi` | kyle_lambda × signed_flow | Theorem 1 (sufficient statistic) |
| 1 | `directional_conviction` | TFI × |OFI| | Theorem 1 (sufficient statistic) |
| 2 | `vpin` | rolling 50-batch mean of |TFI| | Theorem 7 (bounds, quasi-convexity) |
| 3 | `hawkes_branching` | 1 - 1/√(Var(N)/E[N]) over rolling 50-batch window | Theorem 8 (consistent estimator) |
| 4 | `reservation_price_dev` | weighted_imbalance_5lvl × realvol² | Theorem 7 (Avellaneda-Stoikov) |

**Intermediate features needed** (computed but not used as model inputs):
- `kyle_lambda`: rolling 50-batch Cov(return, signed_notional) / Var(signed_notional) — same as v8 feature 9
- `signed_notional`: buy_notional - sell_notional per batch (trade-based flow, NOT orderbook OFI)
- `TFI`: (n_buys - n_sells) / (n_buys + n_sells) — same as v8 feature 6
- `realvol`: rolling 10-batch std of returns — same as v8 feature 4
- `weighted_imbalance_5lvl`: orderbook imbalance — same as v8 feature 14
- `trade_counts_per_batch`: number of trades per batch (for Hawkes variance/mean)

**Clarification on signed_flow vs OFI**:
- Feature 0 (`lambda_ofi`): `kyle_lambda × signed_notional` — trade-based signed flow (v8 line 345)
- Feature 1 (`directional_conviction`): `TFI × |signed_notional|` — trade-based flow magnitude
- These are both trade-derived, not orderbook-derived OFI (feature 16 in v8)

**Hawkes branching domain guards**:
- When `Var(N) <= E[N]` (not overdispersed): clamp `hawkes_branching` to 0.0
- When `E[N] = 0` (empty batch): set to 0.0
- Clip final values to [0.0, 0.99] for numerical safety

**Normalization**: All 5 features use rolling z-score (window=1000, min_periods=100). Exception: `reservation_price_dev` (feature 4) uses IQR-based robust scaling since realvol² is heavy-tailed. Clip to [-5, 5].

**Cache**: Bump `_FEATURE_VERSION` from "v8" to "v9".

### Architecture

**Window: 50 → 75**

```
Input: (batch, 75, 5)
         │
    ┌─────┴─────┐
    │            │
  [TCN]      [Flat branch]
    │            │
  Conv1d(5→16, k=5)  Flatten: 75×5 = 375
  Conv1d(16→8, k=3)  + mean(5) + std(5) = 10
    │                 = 385
  AdaptiveAvgPool → 8    │
    │                     │
    └───────┬─────────────┘
            │
    Concatenate: 385 + 8 = 393
            │
    MLP: 393 → hdim → hdim → 3
    ReLU, orthogonal init (MLP), Kaiming (TCN)

Ensemble: 5 seeds, logit sum argmax
~55K parameters (at hdim=128)
```

**Implementation**: Modify `HybridClassifier.__init__` — change Conv1d intermediate channels from 32→16, pool output from 16→8. The `n_feat` input channel is already parameterized via `obs_shape`. Drop `temporal_summaries` (v8 experiment, not carried forward).

**Rationale**:
- 375 flat dims (vs 2300) — 84% reduction in input dimensionality
- 55K params (vs 692K) — P/N ratio drops from 0.28 to 0.022 (Theorem 10)
- TCN gets 75 timesteps (proved minimum for temporal extraction)
- Ensemble of 5 is valid only if per-model α > 0.5 (Theorem 10 correction)

### Labeling

- Triple barrier stays (mutual exclusivity and exhaustiveness proved, Theorem 0)
- `fee_mult` Optuna range: **[1.5, 12.0]** (widened from [1.0, 4.0])
  - At f=1.5: α_min=83.3%. At f=4: α_min=62.5%. At f=8: α_min=56.25%
  - Theory says higher is easier, but empirical history shows high fee_mult underperformed (f=8→Sortino -0.003). Keep f=1.5 accessible since it's the historical best; the feature reduction may change the dynamics.
- `MAX_HOLD_STEPS` stays at 300
- `MIN_HOLD`: 200 (moderate — regime gate handles quality filtering, so MIN_HOLD can be relaxed from current 500)
- `FEE_BPS` stays at 5

### Regime Gate

Post-classifier gate in `evaluate()` and inference:

```python
if raw_hawkes_branching[step] < r_min:
    action = 0  # force flat
```

**Raw feature storage**: `compute_features()` returns both the normalized feature array AND a separate `raw_hawkes_branching` array (un-normalized). This is stored alongside features in the `.npz` cache. The gate operates on raw values, not z-scored values.

- `r_min` is an Optuna hyperparameter, range [0.3, 0.7]
- Proved: gating improves accuracy when SNR is monotone in Hawkes branching (Theorem 9)
- Proved: need φ ≥ 1/k² fraction of steps to pass (Theorem 10)
  - At k=1.5 (50% return improvement): φ_min ≈ 44%
  - At k=2.0 (100% return improvement): φ_min = 25%
- Monitor `regime_filter_rate` — if > 56% filtered, gating may hurt Sortino via √φ penalty

### Training

- Optimizer: AdamW, weight_decay=5e-4
- Loss: Focal loss, gamma=1.0, inverse-frequency class weights (proved to equalize across classes, Theorem 10)
- Recency weighting: decay=1.0 (recent samples ~2.7x weight)
- Epochs: 25
- Ensemble: 5 seeds (valid when α > 0.5; monitor and disable if α < 0.5)
- **Ensemble validity check**: After training each seed, compute directional accuracy on validation set. If mean α < 0.5, fall back to single best model instead of ensemble. Log warning.
- **α_min early stopping**: During training, if validation accuracy < α_min for 5 consecutive epochs, skip remaining seeds (classifier can't profit at current fee_mult)

**Optuna search space**:
| Parameter | Range | Rationale |
|-----------|-------|-----------|
| `lr` | [5e-4, 5e-3] | Same |
| `hdim` | [64, 128, 256] | Smaller model |
| `fee_mult` | [1.5, 12.0] | Theory favors higher, empirical history keeps 1.5 accessible |
| `r_min` | [0.3, 0.7] | Regime gate threshold |
| `batch_size` | [128, 256, 512] | Same |

### Evaluation Diagnostics

New prints in `evaluate()`:
```
alpha_min: {0.5 + 1/(2*fee_mult):.4f}
empirical_accuracy: {directional_correct/directional_total:.4f}
regime_filter_rate: {filtered_steps/total_steps:.4f}
f_opt_kelly: {(p_win-p_loss)*(1-c)/((p_win+p_loss)*c):.4f}
hawkes_branching_mean: {mean_branching:.4f}
```

### Rollout Plan (one change at a time)

| Step | Change | Expected Effect |
|------|--------|----------------|
| 1a | Replace 46 features with 5, keep DirectionClassifier, window=50, fee_mult=1.5 | Isolate feature reduction effect |
| 1b | Switch to smaller HybridClassifier (55K params) | Isolate model size effect |
| 2 | Increase window 50→75 | TCN branch contributes → Sortino ↑ |
| 3 | Increase fee_mult to [4, 12] range | Lower α_min → more profitable trades |
| 4 | Add regime gate (r_min via Optuna) | Higher quality trades → Sortino ↑ |
| 5 | Full Optuna search (hdim, fee_mult, r_min) | Optimal config |

Each step is a separate experiment in `results.tsv`. If any step regresses, stop and investigate before proceeding.

### Testing

1. **Feature sanity** — verify 5 features have correct shapes, no NaNs, expected bounds
2. **Hawkes estimator** — synthetic Hawkes process with known α/β, verify n̂ recovers true branching ratio
3. **Regime gate** — verify gate forces flat when hawkes_branching < r_min
4. **Backward compat** — v8 cache untouched, v9 gets new cache key
5. **Accuracy diagnostic** — verify α_min computation matches proved values

## Proof Artifacts

All Lean 4 proofs are in `proofs/`:
- `theorem0-math-review/` — Sortino, Hurst, Kyle, Amihud, Triple Barrier
- `theorem1-sufficient-statistic/` — (λ·OFI, TFI·|OFI|) sufficiency
- `theorem2-entropy-collapse/` — all claims disproved (counterexamples)
- `theorem3-kelly-barriers/` — corrected f*, α_min=56.25%
- `theorem4-hawkes-predictability/` — variance decay, correlation bounds
- `theorem5-architecture-bounds/` — weak dominance, W_min=75, ensemble
- `theorem6-unified-strategy/` — full strategy specification (18/18 proved)
- `theorem7-avellaneda-vpin/` — A-S properties, VPIN bounds, adverse selection
- `theorem8-hawkes-mle/` — MLE validation, method-of-moments estimator
- `theorem9-regime-kelly/` — gated Kelly optimization, Banach convergence
- `theorem10-training-regime/` — focal loss, capacity bounds, gating trade-off

**Totals**: 133 proved, 12 disproved, 7 corrected, 0 sorry statements.
