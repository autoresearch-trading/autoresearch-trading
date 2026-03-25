# Research State

## Environment
- Run command: `uv run python train.py`
- Entry point: `train.py`
- Files to modify per experiment: `train.py` (config), `prepare.py` (features/metrics)
- Output contract: PORTFOLIO SUMMARY block with `sortino:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines
- Parser: `bash .claude/skills/autoresearch/resources/parse_summary.sh <logfile>`
- Symbols: 25
- Approximate run duration: ~30 min (DirectionClassifier), ~80 min (HybridClassifier)
- Primary metric: Sortino ratio (BUGGY — see T26, fix pending for v11)
- Default scoring: `score = mean_sortino * 0.6 + (passing / total_symbols) * 0.4`
- Cache note: v5, v9, v10 caches all exist. v11 cache needs building (~30-45 min).
- Data: 161 days synced (2025-10-16 → 2026-03-25, 40GB). TEST_END needs updating to 2026-03-25.

## Current Best
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, fee_mult=1.5, min_hold=800, labeling=triple_barrier, max_hold=300, features=v5 (31), window=50, seeds=5, epochs=25, model=DirectionClassifier, gates=none}
- Score: sortino=+0.177 (BUGGY, true ≈ 0.264), passing=16/25, PF=1.39, WR=62.7%, trades=900
- Commit: de882f7 (Run 7)

## v10 Best (9 features)
- Config: same as above but features=v10 (9), no gates
- Score: sortino=+0.161 (BUGGY, true ≈ 0.240), passing=16/25, PF=1.34, WR=59.7%, trades=900
- Commit: 6a68f9f (Run 11)
- Note: 9 features capture 91% of v5's 31-feature Sortino

## v11 Plan (NEXT SESSION — implement)
17 features = 9 existing + 8 new, all backed by 35 Aristotle proofs:
- ADD: multi_level_ofi, buy_vwap_dev, sell_vwap_dev, spread_bps, amihud_illiq, roll_measure, trade_arrival_rate, r_20
- FIX: Sortino formula (T26: divides by N not N_neg, ~1.49x correction)
- ADD METRICS: Sharpe, Calmar, CVaR 95%
- EXTEND: TEST_END 2026-03-09 → 2026-03-25 (20 → 36 test days)
- CACHE: bump _FEATURE_VERSION to "v11", rebuild ~30-45 min

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
1. Will v11 (17 features) beat v5 (31 features)? T23 says k*≈10-20 is optimal.
2. With 36 test days, do we reach per-symbol significance? (~65 trades vs 99 needed)
3. Does the Sortino fix change which symbols pass?
4. Can Calmar ratio provide better risk assessment than Sortino alone?
5. Walk-forward validation — should we implement instead of fixed splits?

## Completed Experiments
- v5 baseline → v6 tape → v9 Aristotle → v5.5 proof-backed sweep → v10 top features
- Permutation importance on 31 v5 features
- Per-symbol reliability analysis across 8 runs
- Metrics validation research (academic + Aristotle proofs)
- Feature validation research (12 academic topics + Aristotle proofs)
- See results.tsv and docs/experiments/ for full history
