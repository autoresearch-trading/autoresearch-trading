# Research State

## Environment
- Run command: `uv run python train.py`
- Entry point: `train.py`
- Files to modify per experiment: `train.py` (config constants at top of file)
- Output contract: PORTFOLIO SUMMARY block with `sortino:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines
- Parser: `bash .claude/skills/autoresearch/resources/parse_summary.sh <logfile>`
- Symbols: 23 (25 minus CRV, XPL excluded)
- Approximate run duration: ~10 min (local M4 CPU, 23 symbols × 5 seeds × 25 epochs)
- Primary metric: Sortino ratio
- Default scoring: `score = mean_sortino * 0.6 + (passing / 23) * 0.4`
- Cache note: Changing prepare.py features invalidates .cache/*.npz (~30 min rebuild). Avoid mid-experiment.

## Current Best
- Config: {lr=1e-3, hdim=64, nlayers=3, batch_size=256, wd=0.0, fee_mult=11.0, min_hold=1200, window_size=50, MAX_HOLD_STEPS=300, labeling=triple_barrier, features=13, seeds=5, epochs=25, logit_bias=0.0, use_uace=False, curriculum_epochs=0}
- Score: 0.368 (sortino=0.353, passing=9/23, trades=1269)
- Walk-forward: mean=0.261, std=0.220, all 4 folds positive
- Commit: ac92256 (wd=0 experiment)

## Swept Variables (all confirmed optimal at current values)
| Variable | Values tested | Winner |
|----------|--------------|--------|
| features | 39→17→13 (ablation) | 13 |
| fee_mult | 3,5,7,8,9,10,11,12.9,15 | 11.0 |
| min_hold | 400,600,800,1200 | 1200 |
| MAX_HOLD | 300,600,1200 | 300 |
| r_min | 0.0,0.24,0.5 | 0.0 |
| epochs | 15,25,35 | 25 |
| lr | 1e-3,2e-3,4.4e-3,8e-3 | 1e-3 |
| weight_decay | 0.0,1e-3,5e-4 | 0.0 |
| window_size | 10,20,50 | 50 |
| batch_size | 128,256,512 | 256 |
| hdim | 64,128,256 | 64 |
| nlayers | 2,3,4 | 3 |
| symbols | 25→23 (CRV/XPL excluded) | 23 |
| logit_bias | 0.0,0.5,1.0 | 0.0 |
| curriculum_epochs | 0,10 | 0 |
| use_uace | False,True (with lr sweep 1e-4 to 3e-3) | False |

## Open Questions
1. Would more data (sync newer dates from Pacifica) improve the model? T46 proved 207 days needed for SE(Sortino)<0.1, we have 160. This is the only untested lever with a strong theoretical basis.

## Completed Experiments (through 2026-03-31)
- Realism improvements (T42-T45): funding negligible, no spread widening, rho=0.28, latency covered
- Walk-forward validation: 4 folds all positive, mean Sortino=0.261, T46 proved variance is sampling noise
- Aristotle T42-T47: 73 theorems, 0 sorry, all formally verified in Lean 4
- T47 signal autocorrelation: signal decays at lag 1, but MLP learns nonlinear temporal patterns (window=50 wins)
- window_size sweep: 50 > 20 > 10
- batch_size sweep: 256 > 128, 512
- hdim sweep: 64 > 128 > 256
- nlayers sweep: 3 > 2, 4
- logit bias: no bias > 0.5 > 1.0
- curriculum learning: no curriculum > 10 warm-up epochs
- UACE loss (with proper lr sweep): focal loss > UACE at all tested lr (best UACE=0.258@3e-4 vs focal=0.353@1e-3)
- SWA (weight averaging): 0.285 vs 0.353 baseline — hurts
- Asymmetric barriers: symmetric 11/11 > tp=15/sl=11 and tp=9/sl=11
- Confidence threshold: 0 > 0.45 > 0.55 (gating hurts)
- r_btc_lag1 cross-asset feature: HURTS (0.343/5pass, alpha<0.5 broke ensemble)
- 10-seed ensemble: DISCARD (alpha<0.5, ensemble never forms, seed count irrelevant)
- Dropout 0.1/0.2: DISCARD (model already regularized by 676→64 bottleneck)
- Alpha threshold removal: DISCARD (Theorem 10 confirmed, single model wins)
- Residual connections: DISCARD (skip connections hurt at 52K params, 0.167 vs 0.353)
- GCE loss (3 lr sweep): DISCARD (focal wins at all lr, best GCE=0.240 vs focal=0.353)
- Metalabeling (t=0.5, 0.7): DISCARD (meta-model can't learn useful gate on 25d val data)
- GELU/SiLU activations: DISCARD (ReLU > GELU 0.282 > SiLU 0.099)
- Adaptive barriers: DISCARD (vol-scaled thresholds hurt badly, 0.058 vs 0.353)

## Key Findings
- Model is at a local optimum on the hyperparameter surface — every variable has been swept
- Focal loss + gamma=1.0 + class weights + recency weighting is a strong training setup that resists "improvements"
- Smaller network generalizes better (64 > 128 > 256 hdim)
- Walk-forward edge is real but modest (Sharpe ~0.18, regime-dependent)
- When testing new training methods, must re-tune at least lr for a fair comparison (AlgoPerf protocol)
