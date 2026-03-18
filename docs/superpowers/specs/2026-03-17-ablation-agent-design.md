# Ablation Agent — Design Spec

## Goal

Systematically isolate why the v5→v6 tape reading pivot regressed (Sortino 0.230→-0.003) by running controlled single-variable experiments. Outputs a recommended config and analysis report.

## Background

Three variables changed simultaneously between v5 (Sortino=0.230, 18/25) and tape-v1 (Sortino=-0.003, 3/25):

1. **Features**: 31 → 39 (8 tape reading features added)
2. **Labeling**: Fixed-horizon (FORWARD_HORIZON=800) → Triple Barrier (MAX_HOLD_STEPS=300, TP=SL=fee_threshold)
3. **min_hold**: 800 → 100

Without isolating these, we can't attribute the regression or know what to fix.

## Scoring

```
score = mean_sortino * 0.6 + (passing / 25) * 0.4
```

- v5 baseline: 0.230 * 0.6 + 18/25 * 0.4 = 0.138 + 0.288 = **0.426**
- tape-v1: -0.003 * 0.6 + 3/25 * 0.4 = -0.002 + 0.048 = **0.046**

Both Sortino and breadth matter. Note: `passing` already accounts for the 20% drawdown guardrail — symbols exceeding max_drawdown are excluded by `eval_policy`, so the score implicitly penalizes high-drawdown configs via lower passing counts.

## Decision Tree

### Phase 1: Sanity + Isolate (3 runs)

| Run | Features | Labeling | min_hold | fee_mult | Purpose |
|-----|----------|----------|----------|----------|---------|
| 1 | 31 (masked) | Fixed-horizon (800) | 800 | 1.5 | v5 config sanity check |
| 2 | 39 | Fixed-horizon (800) | 800 | 1.5 | Isolate: features |
| 3 | 39 | Triple Barrier (300) | 800 | 1.5 | Isolate: labeling |

Analysis: Compare scores. Run 1 should approximate v5 baseline (~0.43) — it won't match exactly because feature masking zeros out columns 31-38 rather than removing them from normalization, so the first 31 features may differ slightly from the original v5 cache. Run 2 vs Run 1 = feature impact. Run 3 vs Run 2 = labeling impact.

**Note on Run 3:** Triple Barrier labels on a 300-step horizon but min_hold=800 prevents exiting before step 800. This is a known mismatch — the label says "TP hit at step 60" but the model can't act on it until step 800. This is intentional for isolation (we want to test labeling quality independent of trade frequency), but the mismatch may suppress Triple Barrier's advantage. Phase 2 will reveal the true impact when min_hold is reduced.

### Phase 2: Sweep min_hold (3 runs)

Take the best labeling method from Phase 1. Use 39 features (unless Phase 1 shows they hurt).

| Run | min_hold | Purpose |
|-----|----------|---------|
| 4 | 500 | Moderate selectivity |
| 5 | 300 | Tape reading scale |
| 6 | 100 | Maximum frequency |

Analysis: Find optimal min_hold. Expect monotonic trend — higher min_hold = fewer trades = less fee drag.

### Phase 3: Sweep fee_mult (3 runs)

Take best config from Phases 1-2.

| Run | fee_mult | Purpose |
|-----|----------|---------|
| 7 | 5.0 | Tighter barriers |
| 8 | 10.0 | Moderate barriers |
| 9 | 2-class (long/short only) | Remove flat class entirely |

For Run 9 (2-class): relabel flat samples as the direction of the forward return at timeout. If return > 0, long; else short. This tests whether the model performs better when it only needs to pick direction, not decide whether to trade.

### Phase 4: Confirmation (1 run)

| Run | Config | Purpose |
|-----|--------|---------|
| 10 | Best from all phases | Confirm best config |

**Early stopping:** If Phase 1 shows features hurt (Run 2 < Run 1 by > 0.05), Phase 2+ uses 31 features. If a phase shows a clear winner (gap > 0.1), skip remaining runs in that phase.

**Total budget:** 10 runs × ~8 min = ~80 min. Up to 12 runs if no early stopping = ~96 min.

## Implementation: Skill-Based Agent

### Architecture shift: skill, not script

Instead of a rigid Python script (`scripts/ablation_agent.py`) with hardcoded decision logic, the ablation agent is implemented as a **Claude Code skill**. Claude itself is the agent — guided by the skill's protocol, it runs experiments, stores results, reasons about them between phases, and adapts.

**Why:** A Python script hardcodes `if/else` decisions. A skill lets Claude reason about results — e.g., "Run 3 scored 0.38 but had suspiciously few trades, maybe the min_hold/labeling mismatch is suppressing it — I should weight Phase 2 results more heavily." This adaptive reasoning is exactly what Claude is good at and what a script can't do.

### Skill folder structure

```
.claude/skills/ablation/
├── SKILL.md                        — main skill: protocol, decision tree, scoring
├── resources/
│   ├── phase1_configs.json         — pre-defined experiment configs
│   ├── phase2_configs.json
│   ├── phase3_configs.json
│   ├── parse_summary.sh            — deterministic PORTFOLIO SUMMARY → JSON
│   └── report_template.md          — template for final report
└── data/
    └── results.json                — accumulated results (Claude reads between phases)
```

### SKILL.md role

The skill file contains:
- **Trigger**: "Use when running ablation experiments on train.py configs"
- **Protocol**: The 4-phase decision tree from this spec
- **Scoring formula**: `score = sortino * 0.6 + (passing/25) * 0.4`
- **Decision criteria**: How to pick winners between phases (score comparison, early stopping rules)
- **Output requirements**: Results table, per-phase analysis, recommended config
- **Gotchas section** (see below)

### How Claude executes

1. **Read skill** → understands the protocol and phases
2. **Load phase configs** from `resources/phase1_configs.json`
3. **For each experiment**: run `uv run python train.py --labeling fixed --min-hold 800 --fee-mult 1.5 ...`
4. **Parse results** using `resources/parse_summary.sh` (deterministic extraction)
5. **Append to `data/results.json`** — persistent memory across phases
6. **Reason about results** — compare scores, decide next phase's configs
7. **After all phases**: write report from template to `docs/ablation-report.md`

### Progressive disclosure

Claude doesn't load all configs upfront. Each phase's configs are in separate files. After completing Phase 1, Claude reads `data/results.json`, reasons about which labeling/features won, then reads `resources/phase2_configs.json` and adapts the configs based on Phase 1 outcomes. This mirrors the decision tree but with Claude's reasoning in the loop.

### Memory between phases

`data/results.json` accumulates all experiment results:

```json
[
  {
    "run": 1,
    "phase": 1,
    "name": "v5-sanity",
    "config": {"features": 31, "labeling": "fixed", "min_hold": 800, "fee_mult": 1.5, "n_classes": 3},
    "results": {"sortino": 0.225, "passing": 17, "trades": 890, "dd": 0.35, "wr": 0.0, "pf": 0.0},
    "score": 0.407
  }
]
```

Claude reads this between phases to make decisions. This is the "standup log" pattern — Claude reads its own history and knows what changed.

### Deterministic helper: parse_summary.sh

```bash
#!/bin/bash
# Extract PORTFOLIO SUMMARY from train.py stdout into greppable key=value pairs
grep -E "^(sortino|symbols_passing|num_trades|max_drawdown|win_rate|profit_factor):" "$1"
```

Claude uses this for reliable extraction rather than re-parsing stdout each time.

## train.py modifications

**CLI arguments** (added to existing argparse): The skill controls train.py via command-line arguments. Safe (no source modification) and easy to verify.

```python
parser.add_argument("--labeling", choices=["triple", "fixed"], default="triple")
parser.add_argument("--forward-horizon", type=int, default=800)
parser.add_argument("--min-hold", type=int, default=None)  # overrides MIN_HOLD
parser.add_argument("--fee-mult", type=float, default=None)  # overrides BEST_PARAMS
parser.add_argument("--feature-mask", type=int, default=None)  # e.g. 31 = use first 31 only
parser.add_argument("--n-classes", type=int, choices=[2, 3], default=3)
```

When `--min-hold` / `--fee-mult` are provided, they override the module-level constants. When `--feature-mask=31`, features[:,31:] are zeroed after loading in both training and eval.

**Fixed-horizon labeling** (re-added as `make_labeled_dataset_fixed`):

```python
def make_labeled_dataset_fixed(env, horizon, fee_threshold, max_samples=10000):
    """Fixed-horizon labeling (v5 method). Label by forward return at exactly step i+horizon.
    Returns (obs, labels, indices) matching Triple Barrier signature."""
    features = env.features
    prices = env.prices
    n = len(features)
    window = env.window_size

    valid_start = window
    valid_end = n - horizon
    if valid_end <= valid_start:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    all_idx = np.arange(valid_start, valid_end)
    if len(all_idx) > max_samples:
        idx = np.random.choice(all_idx, max_samples, replace=False)
        idx.sort()
    else:
        idx = all_idx

    idx = idx[prices[idx] > 0]
    if len(idx) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    fwd_return = (prices[idx + horizon] - prices[idx]) / prices[idx]
    labels = np.zeros(len(idx), dtype=np.int64)
    labels[fwd_return > fee_threshold] = 1   # long
    labels[fwd_return < -fee_threshold] = 2  # short

    obs = np.array([features[i - window:i] for i in idx], dtype=np.float32)
    return obs, labels, idx
```

**Labeling dispatch** in `train_one_model`:
```python
if args.labeling == "fixed":
    obs, labels, indices = make_labeled_dataset_fixed(env, args.forward_horizon, fee_threshold)
else:
    obs, labels, indices = make_labeled_dataset(env, MAX_HOLD_STEPS, fee_threshold, fee_threshold)
```

**Feature masking** in `train_one_model` and `eval_policy`:
```python
if args.feature_mask is not None:
    X[:, :, args.feature_mask:] = 0.0  # zero out features beyond mask
```

**2-class support** (N_CLASSES=2):
- In `train_one_model`: after labeling, relabel flat (0) samples using sign of forward return at the labeling horizon. Map labels to {0=long, 1=short}. Model outputs 2 classes.
- In `make_ensemble_fn`: when n_classes=2, map model output 0→action 1 (long), 1→action 2 (short). The model never predicts flat — it always has a position.

## Gotchas

1. **Feature masking ≠ v5 features.** Zeroing columns 31-38 is not identical to v5's 31-feature cache. The normalization window statistics differ because 39 features are present during z-score/IQR computation. Run 1 will approximate but not exactly match the v5 baseline.

2. **Cache rebuild on first run.** v6 caches for all symbol/split combos must exist. The first train.py invocation per split will rebuild missing caches (~2 min/symbol). After the first full run, subsequent runs use cache and take ~8 min.

3. **Phase 1 Run 3 mismatch.** Triple Barrier labels on 300-step horizon + min_hold=800 means the model can't exit when the barrier hits. This intentionally isolates labeling quality from trade frequency, but may understate Triple Barrier's true advantage.

4. **2-class relabeling needs the forward return.** For `N_CLASSES=2`, flat samples need the forward return at the labeling horizon to decide long vs short. For Triple Barrier, this is `prices[i + max_hold]`; for fixed-horizon, `prices[i + horizon]`. Both are already available in the labeling functions.

5. **train.py `args` must be accessible in `train_one_model` and `eval_policy`.** Currently these functions don't receive CLI args. Either pass `args` as a parameter, or promote the relevant flags to module-level globals set in `main()`.

## Report format

Written to `docs/ablation-report.md`:

```markdown
# Ablation Report — YYYY-MM-DD

## Results

| Run | Name | Features | Labeling | min_hold | fee_mult | Classes | Sortino | Passing | Trades | Score |
|-----|------|----------|----------|----------|----------|---------|---------|---------|--------|-------|
| 1   | ...  | ...      | ...      | ...      | ...      | ...     | ...     | ...     | ...    | ...   |

## Phase 1: Isolation
- Feature impact: Run 2 vs Run 1 = ...
- Labeling impact: Run 3 vs Run 2 = ...

## Phase 2: min_hold sweep
- Best min_hold: ...

## Phase 3: fee_mult + classes
- Best fee_mult: ...
- 2-class vs 3-class: ...

## Recommended Config
{config dict}

## Suggested Next Steps
- ...
```

## File Changes

| File | Change |
|------|--------|
| train.py | Add CLI args (--labeling, --forward-horizon, --min-hold, --fee-mult, --feature-mask, --n-classes). Re-add fixed-horizon labeling as make_labeled_dataset_fixed(). Add feature masking logic. Add 2-class support with action remapping in make_ensemble_fn. Pass args to train_one_model and eval_policy. |
| .claude/skills/ablation/SKILL.md | New file. Skill definition with protocol, decision tree, scoring, gotchas. |
| .claude/skills/ablation/resources/ | New folder. Phase configs (JSON), parse helper (sh), report template (md). |
| .claude/skills/ablation/data/ | New folder. results.json populated during execution. |

## Success Criteria

- Agent completes all phases without manual intervention
- Produces a clear report attributing the regression to specific variables
- Recommended config scores >= 0.35 (within striking distance of v5 baseline)
- If tape reading features/labeling genuinely don't help, the report says so clearly
