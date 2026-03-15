# Verification Module Design

## Purpose

A standalone diagnostic module (`verify.py`) that the autoresearch agent uses at decision points to validate whether the current research direction is sound. Answers two questions:

1. **Is the alpha in the features or the model?** — XGBoost baseline comparison
2. **Is the signal real?** — Label shuffle test

## Architecture

### verify.py (standalone, never modified by autoresearch)

Imports from `prepare.py` (data loading, `make_env`, `evaluate`) and reuses `make_labeled_dataset` and `eval_policy` from `train.py`. Does not modify either file.

### Interface

```
python verify.py                  # full diagnostic suite
python verify.py --xgboost-only   # baseline comparison only
python verify.py --shuffle-only   # label shuffle test only
```

## Diagnostic 1: XGBoost Baseline

### Feature Engineering

Takes the same `(50, 31)` observation windows from `make_labeled_dataset()`. Computes per-feature summary statistics:

- mean, std, min, max, last per feature
- Result: 155-dimensional flat feature vector per sample

Same labels, same recency weights, same class balancing as the MLP pipeline.

### Model Configuration

- `XGBClassifier` with `objective='multi:softprob'`, 3 classes (flat/long/short)
- Defaults: `n_estimators=500`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`
- `sample_weight` from recency weights to match MLP training

### Evaluation

Two runs:

1. **Single model** — one XGBClassifier, `random_state=0`
2. **5-seed ensemble** — 5 models with different `random_state`, sum `predict_proba()` outputs, argmax

Both feed into `eval_policy()` identically to the MLP. The ensemble policy function takes an observation, computes summary stats, runs all 5 models, sums probabilities, returns argmax action.

### Output

```
XGBoost single:   sharpe=X.XXX  pass=XX/25  trades=XXX
XGBoost ensemble: sharpe=X.XXX  pass=XX/25  trades=XXX
MLP baseline:     sharpe=0.191  pass=25/25  trades=260
```

### Decision Rules

- XGBoost >= MLP: alpha is in features, pivot to feature engineering (prepare.py)
- XGBoost << MLP: MLP temporal processing adds value, continue model architecture work

## Diagnostic 2: Label Shuffle Test

### How It Works

1. Loads the same training data as `train_one_model()` in train.py
2. Randomly permutes the labels (`np.random.permutation(y)`)
3. Trains the MLP with identical hyperparameters (BEST_PARAMS, 28 epochs, focal loss)
4. Evaluates with `eval_policy()`
5. Runs 3 shuffled seeds, reports mean shuffled sharpe

### Decision Rules

- `real_sharpe - shuffled_sharpe < 0.05`: signal likely not real, flag `SIGNAL_WEAK`
- `real_sharpe > 2x shuffled_sharpe`: signal is robust, flag `SIGNAL_ROBUST`
- Otherwise: `SIGNAL_PRESENT` (signal exists but is thin)

### Output

```
Shuffle test (3 seeds):
  shuffled_sharpe: 0.XXX +/- 0.XXX
  real_sharpe:     0.191
  delta:           0.XXX
  verdict:         SIGNAL_ROBUST / SIGNAL_WEAK / SIGNAL_ABSENT
```

## Autoresearch Integration

### Trigger Rules (added to program.md)

1. **Plateau** — 3 consecutive experiments with no sharpe improvement -> run `python verify.py`
2. **Feature change** — any modification to `compute_features()` in prepare.py -> run `python verify.py --xgboost-only`
3. **Architecture change** — any change to model class in train.py -> run `python verify.py` (full suite)

### Agent Decision Logic

Based on verification results, the autoresearch agent adjusts its next experiment:

- XGBoost >= MLP -> pivot to feature engineering in prepare.py
- XGBoost << MLP -> continue model architecture work in train.py
- Shuffle test shows weak/absent signal -> stop and investigate data/features before more experiments
- Shuffle test robust -> continue current direction

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
  "mlp_baseline": {"sharpe": 0.191, "passing": 25, "trades": 260, "max_dd": 0.076},
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

- `xgboost` — new pip dependency
- All other imports already available (numpy, prepare.py, train.py)

## Files Changed

- **New:** `verify.py` — standalone verification module
- **Modified:** `program.md` — add trigger rules and decision logic for verification
- **Modified:** `.gitignore` — add `reports/` directory
- **No changes to:** `prepare.py`, `train.py`
