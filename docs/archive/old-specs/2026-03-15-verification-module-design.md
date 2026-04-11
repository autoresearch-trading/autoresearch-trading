# Verification Module Design

## Purpose

A standalone diagnostic module (`verify.py`) that the autoresearch agent uses at decision points to validate whether the current research direction is sound. Answers two questions:

1. **Is the alpha in the features or the model?** — XGBoost baseline comparison
2. **Is the signal real?** — Label shuffle test

## Architecture

### verify.py (standalone, never modified by autoresearch)

Imports from `prepare.py` (data loading, `make_env`, `evaluate`) and reuses `make_labeled_dataset` and `eval_policy` from `train.py`. Does not modify either file.

Requires `xgboost` pip dependency. On import failure, prints `"ERROR: pip install xgboost"` and exits with code 1.

### Interface

```
python verify.py                  # full diagnostic suite
python verify.py --xgboost-only   # baseline comparison only
python verify.py --shuffle-only   # label shuffle test only
```

### Time Budget

Verification is **exempt from the 5-minute autoresearch budget**. Expected wall-clock times:
- XGBoost only: ~2-3 minutes (data loading + XGBoost training is fast)
- Shuffle only: ~15 minutes (3 MLP seeds x ~5 min each)
- Full suite: ~18 minutes

### MLP Baseline

verify.py **re-runs the MLP** using the current `full_run()` from train.py with `BEST_PARAMS` to get a fresh baseline. No hardcoded values — the comparison always reflects the current state of train.py.

## Diagnostic 1: XGBoost Baseline

### Feature Engineering

Takes the same `(50, 31)` observation windows from `make_labeled_dataset()`, which returns `(obs, labels, sample_indices)` — all three values are used. Computes per-feature summary statistics from the observation windows:

- mean, std, min, max, last per feature
- Result: 155-dimensional flat feature vector per sample

Same labels, same class balancing as the MLP pipeline.

### Recency Weights

Computed identically to the MLP: exponential decay from `sample_indices` using `exp(1.0 * norm_idx)` where `norm_idx` is normalized to [0, 1]. Passed as `sample_weight` to XGBClassifier.fit().

### Model Configuration

- `XGBClassifier` with `objective='multi:softprob'`, 3 classes (flat/long/short)
- Defaults: `n_estimators=500`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`

### Evaluation

Two runs:

1. **Single model** — one XGBClassifier, `random_state=0`
2. **5-seed ensemble** — 5 models with different `random_state` (0-4), sum `predict_proba()` outputs, argmax

Both feed into `eval_policy()` via a policy function with signature `obs -> action` (takes `(50, 31)` numpy array, computes summary stats internally, returns int action 0/1/2).

### Edge Case: Zero Trades

If XGBoost produces 0 trades on a symbol, that symbol fails (same as MLP). If XGBoost produces 0 total trades across all symbols, verdict is `XGBOOST_FAILED` — flagged separately from a normal comparison.

### Output

```
XGBoost single:   sharpe=X.XXX  pass=XX/25  trades=XXX
XGBoost ensemble: sharpe=X.XXX  pass=XX/25  trades=XXX
MLP baseline:     sharpe=X.XXX  pass=XX/25  trades=XXX  (fresh run)
```

### Decision Rules and Verdicts

Concrete thresholds for the baseline comparison:

| Condition | Verdict | Meaning |
|-----------|---------|---------|
| XGBoost ensemble sharpe >= MLP sharpe | `XGBOOST_AHEAD` | Alpha is in features, pivot to prepare.py |
| XGBoost ensemble sharpe >= MLP sharpe * 0.7 | `COMPARABLE` | Features carry most signal, model adds modest value |
| XGBoost ensemble sharpe < MLP sharpe * 0.7 | `MLP_AHEAD` | MLP temporal processing adds significant value |
| XGBoost total trades == 0 | `XGBOOST_FAILED` | XGBoost couldn't produce trades, inconclusive |

## Diagnostic 2: Label Shuffle Test

### How It Works

1. Loads the same training data as `train_one_model()` in train.py
2. Uses `make_labeled_dataset()` which returns `(obs, labels, sample_indices)`
3. Randomly permutes the labels (`np.random.permutation(y)`)
4. Trains the MLP with identical hyperparameters (BEST_PARAMS, `n_epochs` from train.py, focal loss)
5. Evaluates with `eval_policy()`
6. Runs 3 shuffled seeds, reports mean shuffled sharpe

Note: the shuffle test reuses `train_one_model()` from train.py directly (with permuted labels passed in), so the epoch count, optimizer, and loss function are always in sync. This requires adding an optional `labels_override` parameter to `train_one_model()` — the only modification to train.py.

### Decision Rules and Verdicts

| Condition | Verdict | Meaning |
|-----------|---------|---------|
| shuffled_sharpe >= real_sharpe (and real_sharpe > 0) | `SIGNAL_ABSENT` | Model performs same on random labels, no real signal |
| real_sharpe <= 0 | `SIGNAL_ABSENT` | Model is unprofitable, signal question is moot |
| real_sharpe - shuffled_sharpe < 0.05 | `SIGNAL_WEAK` | Signal likely not real or very thin |
| real_sharpe > 2x shuffled_sharpe and delta >= 0.05 | `SIGNAL_ROBUST` | Strong evidence of real signal |
| Otherwise | `SIGNAL_PRESENT` | Signal exists but is thin |

### Output

```
Shuffle test (3 seeds):
  shuffled_sharpe: 0.XXX +/- 0.XXX
  real_sharpe:     0.XXX
  delta:           0.XXX
  verdict:         SIGNAL_ROBUST / SIGNAL_PRESENT / SIGNAL_WEAK / SIGNAL_ABSENT
```

## Autoresearch Integration

### Trigger Rules (added to program.md)

1. **Plateau** — 3 consecutive experiments with no sharpe improvement -> run `python verify.py`
2. **Feature change** — any modification to `compute_features()` in prepare.py -> run `python verify.py --xgboost-only`
3. **Architecture change** — any change to model class in train.py -> run `python verify.py` (full suite)

### Agent Decision Logic

Based on verification verdicts, the autoresearch agent adjusts its next experiment:

| Baseline Verdict | Signal Verdict | Recommendation |
|-----------------|----------------|----------------|
| `XGBOOST_AHEAD` | `SIGNAL_ROBUST` | Pivot to feature engineering in prepare.py |
| `XGBOOST_AHEAD` | `SIGNAL_WEAK/ABSENT` | Investigate data quality before any changes |
| `COMPARABLE` | `SIGNAL_ROBUST` | Try both feature and model improvements |
| `MLP_AHEAD` | `SIGNAL_ROBUST` | Continue model architecture work in train.py |
| `MLP_AHEAD` | `SIGNAL_WEAK/ABSENT` | Investigate data quality before any changes |
| `XGBOOST_FAILED` | any | XGBoost inconclusive, rely on signal verdict only |

The agent reads the stdout summary and acts accordingly. Saved reports provide history.

## Reports

### Directory

`reports/` (gitignored — local diagnostic artifacts)

### JSON Report (`verification_YYYY-MM-DD_HHMMSS.json`)

```json
{
  "timestamp": "2026-03-15T14:30:00",
  "xgboost_single": {"sharpe": 0.12, "passing": 20, "trades": 180, "max_dd": 0.09},
  "xgboost_ensemble": {"sharpe": 0.15, "passing": 22, "trades": 200, "max_dd": 0.08},
  "mlp_baseline": {"sharpe": 0.19, "passing": 25, "trades": 260, "max_dd": 0.076},
  "shuffle_test": {"mean_sharpe": 0.02, "std": 0.01, "n_seeds": 3},
  "verdicts": {
    "baseline_comparison": "MLP_AHEAD",
    "signal_quality": "SIGNAL_ROBUST"
  },
  "recommendation": "Continue model architecture work"
}
```

### Markdown Report (`verification_YYYY-MM-DD_HHMMSS.md`)

Human-readable version with comparison table and verdicts.

## Dependencies

- `xgboost` — new pip dependency (graceful error on import failure)
- All other imports already available (numpy, prepare.py, train.py)

## Files Changed

- **New:** `verify.py` — standalone verification module
- **Modified:** `program.md` — add trigger rules and decision logic for verification
- **Modified:** `.gitignore` — add `reports/` directory
- **Modified:** `train.py` — add optional `labels_override` parameter to `train_one_model()` (for shuffle test)
- **Modified:** `train.py` — fix `make_labeled_dataset()` early return to return 3 values (bug fix)
- **No changes to:** `prepare.py`
