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

## Implementation

### File: `scripts/ablation_agent.py`

Single Python script, ~250 lines. No external dependencies beyond what's already installed.

### train.py modifications

**CLI arguments** (added to existing argparse): The ablation agent controls train.py via command-line arguments, not file patching. This is safe (no source modification) and easy to verify.

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

### ablation_agent.py logic

```python
def run_experiment(config: dict) -> dict:
    """Run train.py with CLI args, parse results."""
    # 1. Build CLI args from config dict
    # 2. subprocess.run(["uv", "run", "python", "train.py", ...args], capture_output=True)
    # 3. Parse PORTFOLIO SUMMARY from stdout
    # 4. Return {sortino, passing, trades, dd, wr, pf, score}

def run_phase(phase_name, experiments):
    """Run a list of experiments, print results table."""
    results = []
    for exp in experiments:
        print(f"[{phase_name}] Running: {exp['name']}...")
        result = run_experiment(exp)
        results.append(result)
        print(f"  score={result['score']:.3f} sortino={result['sortino']:.3f} passing={result['passing']}/25")
    return results

def main():
    # Phase 1
    p1_results = run_phase("Phase 1", [
        {"name": "v5-sanity", "features": 31, "labeling": "fixed", "min_hold": 800, "fee_mult": 1.5},
        {"name": "v6-features", "features": 39, "labeling": "fixed", "min_hold": 800, "fee_mult": 1.5},
        {"name": "triple-barrier", "features": 39, "labeling": "triple", "min_hold": 800, "fee_mult": 1.5},
    ])

    # Decide: which features? which labeling?
    best_features = pick_features(p1_results)
    best_labeling = pick_labeling(p1_results)

    # Phase 2: sweep min_hold
    p2_experiments = [make_exp(best_features, best_labeling, mh, 1.5) for mh in [500, 300, 100]]
    p2_results = run_phase("Phase 2", p2_experiments)
    best_min_hold = pick_best(p2_results, "min_hold")

    # Phase 3: sweep fee_mult + 2-class
    p3_experiments = [
        make_exp(best_features, best_labeling, best_min_hold, fm) for fm in [5.0, 10.0]
    ] + [make_2class_exp(best_features, best_labeling, best_min_hold)]
    p3_results = run_phase("Phase 3", p3_experiments)

    # Phase 4: confirmation
    best_config = pick_overall_best(p1_results + p2_results + p3_results)
    final = run_phase("Phase 4", [best_config])

    # Write report
    write_report(all_results, final)
```

### Report format

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
| train.py | Add CLI args (--labeling, --forward-horizon, --min-hold, --fee-mult, --feature-mask, --n-classes). Re-add fixed-horizon labeling as make_labeled_dataset_fixed(). Add feature masking logic. Add 2-class support with action remapping in make_ensemble_fn. |
| scripts/ablation_agent.py | New file. Experiment runner with decision tree logic. |

## Success Criteria

- Agent completes all phases without manual intervention
- Produces a clear report attributing the regression to specific variables
- Recommended config scores >= 0.35 (within striking distance of v5 baseline)
- If tape reading features/labeling genuinely don't help, the report says so clearly
