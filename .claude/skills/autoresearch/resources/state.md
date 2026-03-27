# Research State

## Environment
- Run command: `uv run python train.py`
- Entry point: `train.py`
- Files to modify per experiment: `train.py` (config), `prepare.py` (features/metrics)
- Output contract: PORTFOLIO SUMMARY block with `sortino:`, `sharpe:`, `calmar:`, `cvar_95:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines
- Parser: `bash .claude/skills/autoresearch/resources/parse_summary.sh <logfile>`
- Symbols: 25
- Approximate run duration: ~30 min (DirectionClassifier), ~80 min (HybridClassifier)
- Primary metric: Sortino ratio (FIXED in v11 — T26, divides by N not N_neg)
- Default scoring: `score = mean_sortino * 0.6 + (passing / total_symbols) * 0.4`
- Cache note: v5, v9, v10, v11, v11a, v11b caches all exist.
- Data: 161 days synced (2025-10-16 → 2026-03-25, 40GB). TEST_END=2026-03-25 (36 test days).

## Current Best (v11b, honest slippage, 36 test days)
- Config: {lr=4.4e-3, hdim=64, nlayers=3, batch_size=256, fee_mult=11.0, r_min=0.0, min_hold=1200, MAX_HOLD=300, features=v11a (13), window=50, seeds=5, epochs=25}
- Slippage: half_spread (from OB data) + 3 bps impact per side. T40 filter: CRV/XPL excluded.
- Score: sortino=0.303, sharpe=0.212, calmar=20.3, passing=8/23, WR=54.1%, PF=1.70, trades=1224
- Top symbols: BNB (0.647, PF=3.40), SUI (0.408), LINK (0.320), AAVE (0.302)
- T41 maker upper bound: Sortino=0.257 (limit orders would save ~13 bps RT on BTC)

## Prior Best (v10, buggy Sortino, 20 test days)
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, fee_mult=1.5, min_hold=800, features=v10 (9)}
- Score: sortino=+0.230 (BUGGY, true ≈ 0.154), passing=18/25, trades=923
- Note: Not comparable — different Sortino formula, different test period

## v11a Progression
- v11 baseline (17 feat, fee_mult=1.5): Sortino=0.032, 5/25
- v11a ablated (13 feat, fee_mult=1.5): Sortino=0.091, 6/25 (dropped 4 hurting features)
- v11a Optuna (13 feat, fee_mult=12.9): Sortino=0.144, 9/25 (tuned params)

## Key Findings (This Session)
1. **Feature set is the bottleneck** — v5 (31 feat) >> v9 (5 feat) for Sortino; v10 (9 feat) captures 91%
2. **Permutation importance** identified top 4 missing features: vol_of_vol, utc_hour, microprice_dev, delta_TFI
3. **DirectionClassifier beats HybridClassifier** — T19 confirmed (TCN adds variance, not signal)
4. **VPIN gate is harmful** — T17 paradox confirmed empirically
5. **r_min=0.7 validated** but needs raw_hawkes (only in v9+ features)
6. **Sortino formula is WRONG** — T26 proved buggy = correct × √p, ~1.49x understatement
7. **36 trades/symbol is NOT statistically significant** — T29: need ~99 for WR=60%
8. **T25 corrected our assumption** — new symbol must have Sortino ≥ portfolio mean (not just >0) to help
9. **Not a tape reading model** — it's a microstructure-informed direction classifier
10. **No public dataset matches our Pacifica data** — our trade-level DEX perps data is unique
11. **5 core feature families** (OFI, spread, VWAP, volatility, activity) — v11 covers all 5
12. **Academic literature** confirms our features: VPIN best OOS (Easley 2024), Amihud #1 in-sample, multi-level OFI adds significant R²

## Aristotle Proofs (35 total, 0 sorry)
- T0-T15: Original batch (math review, sufficient statistics, Kelly, Hawkes, gates, diversification)
- T16-T22: Experiment-backed (optimal trade count, gate paradox, frequency, complexity, min_hold, loss clusters, dual gate)
- T23-T29: Metrics validation (optimal features k*, drawdown bounds, marginal symbols, Sortino bug, Sharpe-Calmar, VaR/CVaR, statistical significance)
- T30-T35: Feature validation (multi-level OFI, VWAP decomposition, Roll/Amihud, microprice, arrival rate, momentum)

## Swept Variables (cost-adjusted regime)
- fee_mult: 3,5,7,8,9,10,11,12.9,15 → winner 11.0
- epochs: 15,25,35 → winner 25
- min_hold: 400,600,800,1200 → winner 1200
- MAX_HOLD: 300,600,1200 → winner 300 (momentum filter)
- r_min: 0.0,0.24,0.5 → winner 0.0 (no gate, redundant with cost-adjusted barriers)
- features: 17→13 (ablation dropped 4 hurting)
- symbols: 25→23 (T40 excluded CRV/XPL for wide spreads)

## Not Yet Swept
- lr: 4.4e-3 from Optuna, not re-swept in cost-adjusted regime
- weight_decay: 5e-4 hardcoded, never swept
- batch_size: 256 from Optuna, not re-swept
- window_size: 50, never swept in v11

## Open Questions
1. Does lr need re-tuning now that barriers/costs are different?
2. Is weight_decay=5e-4 optimal for the smaller 64-dim network?
3. Walk-forward validation — still not implemented
4. Can we build limit order execution? T41 shows Sortino=0.257 with maker costs

## Completed Experiments
- v5→v6→v9→v5.5→v10→v11→v11a (feature iterations)
- v11a ablation (17→13 features)
- Slippage model (T39 cost-adjusted barriers, T40 symbol filter)
- fee_mult sweep (cost-adjusted), epoch sweep, min_hold sweep, MAX_HOLD sweep, r_min sweep
- T41 maker execution simulation
- See results.tsv and docs/experiments/ for full history
