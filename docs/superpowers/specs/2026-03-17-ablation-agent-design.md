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

Both Sortino and breadth matter.

## Decision Tree

### Phase 1: Sanity + Isolate (3 runs)

| Run | Features | Labeling | min_hold | fee_mult | Purpose |
|-----|----------|----------|----------|----------|---------|
| 1 | 31 (masked) | Fixed-horizon (800) | 800 | 1.5 | v5 config sanity check |
| 2 | 39 | Fixed-horizon (800) | 800 | 1.5 | Isolate: features |
| 3 | 39 | Triple Barrier (300) | 800 | 1.5 | Isolate: labeling |

Analysis: Compare scores. Run 1 should match v5 baseline (~0.43). Run 2 vs Run 1 = feature impact. Run 3 vs Run 2 = labeling impact.

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

Single Python script, ~200 lines. No external dependencies beyond what's already installed.

### train.py modifications

Add three config flags at the top of train.py (below existing config):

```python
# ── Ablation flags (set by ablation agent) ─────────────────────
USE_TRIPLE_BARRIER = True   # False = fixed-horizon labeling
FORWARD_HORIZON = 800       # only used when USE_TRIPLE_BARRIER=False
FEATURE_MASK = None          # None = all 39, or list of indices to keep
N_CLASSES = 3                # 2 = long/short only (no flat)
```

**USE_TRIPLE_BARRIER**: When False, `make_labeled_dataset` uses the old fixed-horizon logic (re-added as `make_labeled_dataset_fixed`). When True, uses Triple Barrier.

**FEATURE_MASK**: When set to `list(range(31))`, zeros out features 31-38 after loading. Applied in `train_one_model` before training and in `eval_policy` before evaluation. No cache invalidation needed — all 39 features are loaded, then masked.

**N_CLASSES**: When 2, flat labels (0) are relabeled: if forward return at timeout > 0, label as long (1), else short (2 → remapped to 1 for 2-class). Model output is 2 classes, ensemble picks long (mapped to action 1) or short (mapped to action 2).

### ablation_agent.py logic

```python
def run_experiment(config: dict) -> dict:
    """Patch train.py config, run, parse results."""
    # 1. Read train.py
    # 2. Regex-replace config values
    # 3. Write patched train.py
    # 4. subprocess.run(["uv", "run", "python", "train.py"])
    # 5. Parse PORTFOLIO SUMMARY from stdout
    # 6. Restore original train.py
    # 7. Return {sortino, passing, trades, dd, wr, pf, score}

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
| train.py | Add USE_TRIPLE_BARRIER, FORWARD_HORIZON, FEATURE_MASK, N_CLASSES flags. Re-add fixed-horizon labeling as make_labeled_dataset_fixed(). Add feature masking logic. Add 2-class support. |
| scripts/ablation_agent.py | New file. Experiment runner with decision tree logic. |

## Success Criteria

- Agent completes all phases without manual intervention
- Produces a clear report attributing the regression to specific variables
- Recommended config scores >= 0.35 (within striking distance of v5 baseline)
- If tape reading features/labeling genuinely don't help, the report says so clearly
