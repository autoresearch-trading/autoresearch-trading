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
- Cache note: v5, v9, v10, v11 caches all exist.
- Data: 161 days synced (2025-10-16 → 2026-03-25, 40GB). TEST_END=2026-03-25 (36 test days).

## Current Best (v10, buggy Sortino, 20 test days)
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, fee_mult=1.5, min_hold=800, labeling=triple_barrier, max_hold=300, features=v10 (9), window=50, seeds=5, epochs=25, model=DirectionClassifier, gates=none}
- Score: sortino=+0.230 (BUGGY, true ≈ 0.154), passing=18/25, trades=923
- Commit: wd5e4

## v11 Baseline (17 features, corrected Sortino, 36 test days)
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, fee_mult=1.5, min_hold=800, features=v11 (17), window=50, seeds=5, epochs=25}
- Score: sortino=0.032, sharpe=0.023, calmar=2.049, cvar_95=0.002, passing=5/25, WR=54.7%, PF=1.02, trades=1600
- Commit: 23ac443
- Passing symbols: BNB, BTC, ENA, LTC, SOL
- Note: Significant regression — new features untouched by Optuna, 36 test days harder than 20, Sortino fix removes inflation. Needs hyperparameter tuning.

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

## Open Questions
1. v11 baseline regressed (0.032 vs 0.230 buggy) — is it the Sortino fix, 36 test days, or new features hurting?
2. Ablation needed: run v10 (9 features) with corrected Sortino + 36 days to isolate the regression cause
3. Optuna search on v11 — fee_mult and feature selection are likely the levers
4. With 64 trades/symbol (min_hold=800), each symbol has ~64 trades vs 99 needed (T29)
5. Walk-forward validation — should we implement instead of fixed splits?

## Completed Experiments
- v5 baseline → v6 tape → v9 Aristotle → v5.5 proof-backed sweep → v10 top features
- Permutation importance on 31 v5 features
- Per-symbol reliability analysis across 8 runs
- Metrics validation research (academic + Aristotle proofs)
- Feature validation research (12 academic topics + Aristotle proofs)
- See results.tsv and docs/experiments/ for full history
