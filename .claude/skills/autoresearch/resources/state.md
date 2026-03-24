# Research State

## Environment
- Run command: `uv run python train.py`
- Entry point: `train.py`
- Files to modify per experiment: `train.py` (config constants at top of file)
- Output contract: PORTFOLIO SUMMARY block with `sortino:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines
- Parser: `bash .claude/skills/autoresearch/resources/parse_summary.sh <logfile>`
- Symbols: 25
- Approximate run duration: ~30 min (DirectionClassifier), ~80 min (HybridClassifier)
- Primary metric: Sortino ratio
- Default scoring: `score = mean_sortino * 0.6 + (passing / total_symbols) * 0.4`
- Cache note: v5 and v9 feature caches both exist. Switching USE_V9 in prepare.py + _FEATURE_VERSION selects which to use. No rebuild needed.

## Current Best
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, fee_mult=1.5, min_hold=800, labeling=triple_barrier, max_hold=300, features=v5 (31), window=50, seeds=5, epochs=25, model=DirectionClassifier, gates=none}
- Score: sortino=+0.177, passing=16/25, PF=1.39, WR=62.7%, trades=900
- Commit: de882f7 (Run 7 of v5.5 experiment)
- Note: Best Sortino in project history with current codebase. 15 of 16 passing symbols have positive Sortino (94% quality rate).

## Previous Best (v5 original)
- Config: same as above but fixed-horizon labeling (forward_horizon=800)
- Score: sortino=0.230, passing=18/25, PF=1.59, WR=59.5%, trades=923
- Note: Original v5 used fixed-horizon labeling; current codebase uses triple barrier. Explains gap.

## v5.5 Experiment Results (2026-03-24, proof-backed)
| Run | Features | Key Config | Sortino | Passing | PF | WR | Trades |
|---|---|---|---|---|---|---|---|
| 0 | v9 (5) | Hybrid+VPIN+regime | -0.018 | 23/25 | 1.35 | 52.6% | 2122 |
| 1 | v9 (5) | DirectionClassifier | +0.007 | 20/25 | 1.33 | 52.0% | 2076 |
| 2 | v9 (5) | + drop VPIN | +0.025 | 21/25 | 1.41 | 53.0% | 2081 |
| 3a | v9 (5) | fee_mult=4.0 | -0.031 | 20/25 | 0.96 | 50.2% | 2154 |
| 3b | v9 (5) | fee_mult=2.0 | +0.026 | 17/25 | 1.44 | 54.8% | 2028 |
| 5 | v9 (5) | r_min=0.5 | +0.017 | 19/25 | 1.34 | 52.6% | 2526 |
| 6 | v5 (31) | v9 config | +0.083 | 11/25 | 1.14 | 52.6% | 2853 |
| **7** | **v5 (31)** | **v5 config** | **+0.177** | **16/25** | **1.39** | **62.7%** | **900** |
| 8 | v5 (31) | min_hold=200 | +0.098 | 10/25 | 1.09 | 49.5% | 3488 |

## Permutation Importance (top 10 of 31 v5 features)
| Rank | Feature | Sortino Drop | Category |
|---|---|---|---|
| 1 | vol_of_vol | +0.301 | Higher-order volatility |
| 2 | utc_hour_linear | +0.279 | Time-of-day |
| 3 | microprice_dev | +0.218 | Orderbook |
| 4 | delta_TFI | +0.210 | Flow |
| 5 | ofi | +0.199 | Orderbook |
| 6 | r_20 | +0.194 | Returns |
| 7 | trade_arrival_rate | +0.191 | Microstructure |
| 8 | volume_spike_ratio | +0.186 | Trade |
| 9 | realvol_10 | +0.185 | Volatility |
| 10 | amihud_illiq_50 | +0.181 | Liquidity |
Bottom 3: sign_autocorr (+0.042), Hurst (+0.048), log_total_depth (+0.067)

## Key Findings
1. **Feature set is the bottleneck** — v5 (31 features) gives 7x better Sortino than v9 (5 features) with same model
2. **DirectionClassifier beats HybridClassifier** — T19 confirmed empirically; TCN adds variance without useful signal
3. **VPIN gate is harmful** — T17 paradox confirmed; removes it improves Sortino, PF, WR
4. **Regime gate (r_min=0.7) is good but needs raw_hawkes** — only available with v9 features
5. **min_hold=800 is critical for quality** — lower min_hold increases trades but kills quality (T20)
6. **v9 missed the top 4 most important features** — vol_of_vol, utc_hour, microprice_dev, delta_TFI
7. **Training on fewer symbols fails** — model needs all 25 for data diversity
8. **23 Aristotle proofs (T0-T22)** back the design decisions with formal verification

## Open Questions
1. Can we build a "v10" feature set with the top ~15 features from permutation importance + regime gate? Would need compute_features_v10() and cache rebuild.
2. Would fixed-horizon labeling (v5 original) recover the gap from 0.177 to 0.230?
3. Can confidence gating (logit margin threshold) improve quality further?
4. Is there a min_hold between 200-800 that improves passing without killing Sortino?

## Completed Experiments
- v5 baseline → v6 tape reading → v9 Aristotle features → v5.5 proof-backed sweep
- Permutation importance analysis on 31 v5 features
- 23 formally verified theorems (T0-T22)
- See results.tsv and docs/experiments/ for full history
