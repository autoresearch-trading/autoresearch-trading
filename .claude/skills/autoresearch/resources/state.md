# Research State

## Environment
- Run command: `uv run python train.py`
- Entry point: `train.py`
- Files to modify per experiment: `train.py` (config constants at top of file)
- Output contract: PORTFOLIO SUMMARY block with `sortino:`, `symbols_passing:`, `num_trades:`, `max_drawdown:`, `win_rate:`, `profit_factor:` lines
- Parser: `bash .claude/skills/autoresearch/resources/parse_summary.sh <logfile>`
- Symbols: 25
- Approximate run duration: ~10 min (local M4 CPU, 25 symbols × 5 seeds × 25 epochs)
- Primary metric: Sortino ratio
- Default scoring: `score = mean_sortino * 0.6 + (passing / total_symbols) * 0.4`
- Cache note: Changing prepare.py invalidates all .cache/*.npz (~30 min rebuild). Avoid mid-experiment.

## Current Best
- Config: {lr=1e-3, hdim=256, nlayers=2, wd=5e-4, fee_mult=1.5, min_hold=800, labeling=fixed_horizon, forward_horizon=800, features=31, seeds=5, epochs=25}
- Score: 0.426 (sortino=0.230, passing=18/25)
- Commit: wd5e4

## Latest (v6 tape reading pivot)
- Config: {lr=1e-3, hdim=256, nlayers=2, batch_size=256, wd=5e-4, fee_mult=10.0, min_hold=300, labeling=triple_barrier, max_hold=300, features=39, window=50, seeds=5, epochs=25}
- Score: 0.162 (sortino=0.057, passing=8/25)
- Commit: c4f55fb
- Note: Regressed badly from v5. Cause not yet isolated — multiple variables changed simultaneously.
- Note: train.py currently has min_hold=500 (changed after c4f55fb, untested).

## Open Questions
1. What caused the v5→v6 regression? Candidates: new features (31→39), labeling change (fixed→triple barrier), min_hold change (800→300), fee_mult change (1.5→10.0). Need ablation to isolate.
2. Do tape reading features (31→39) help or hurt in isolation?
3. Does Triple Barrier labeling improve over fixed-horizon, holding everything else equal?
4. What is the optimal min_hold for the current setup?
5. Is fee_mult=10.0 too aggressive (barriers too wide)?

## Completed Experiments
(none yet — prior results tracked in results.tsv, pre-skill)
