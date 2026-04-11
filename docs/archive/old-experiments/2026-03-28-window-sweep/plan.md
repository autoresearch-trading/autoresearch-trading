# Experiment: window_size sweep

## Hypothesis
T47 (formally verified, 8 theorems, 0 sorry) proved that when predictive signal is concentrated at lag 0, the optimal observation window minimizes noise: W* = √(σ_x²/σ_noise²) ≈ 1. Empirically, all 13 features have zero cross-correlation with returns at lag ≥ 1. Window=50 forces the MLP to process 650 input dimensions (50×13) when 130 (10×13) captures all signal. Prediction: window=10 or 20 will outperform window=50.

## Scoring
`score = mean_sortino * 0.6 + (passing / 23) * 0.4`

## Phases

### Phase 1: window_size sweep
| Run | Config delta (vs baseline) | Purpose |
|-----|---------------------------|---------|
| 1   | window_size=50 (baseline) | Control — reproduce current best |
| 2   | window_size=10            | T47 minimum viable window |
| 3   | window_size=20            | Middle ground |

All other params: lr=1e-3, hdim=64, nlayers=3, batch_size=256, fee_mult=11.0, r_min=0.0, wd=0.0, seeds=5, budget=300s

## Decision Logic
- Winner = highest score
- If window=10 or 20 beats 50: T47 confirmed, adopt smaller window
- If window=50 still wins: nonlinear temporal patterns exist that T47's linear analysis missed
- Ties (within 0.02 score): prefer smaller window (less overfitting risk)

## Budget
3 runs × ~30 min = ~90 min total

## Gotchas
- Smaller window = fewer steps for mean/std normalization statistics. May hurt robust-scaled features.
- The MLP's summary statistics (mean + std per feature) become noisier with fewer samples.
- Window change doesn't affect cache — same features, just a different slice.
