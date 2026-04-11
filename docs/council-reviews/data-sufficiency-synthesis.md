# Data Sufficiency Assessment — Council Synthesis

**Date:** 2026-04-10
**Question:** Is 40GB / 160 days / 25 symbols sufficient for training a 91K-param dilated 1D CNN targeting 52% accuracy at the 100-event horizon?
**Reviewers:** Council-1 (Lopez de Prado), Council-2 (Rama Cont), Council-5 (Practitioner Quant), Council-6 (DL Researcher)

---

## Executive Summary

**Verdict: MARGINALLY SUFFICIENT for signal detection. INSUFFICIENT for statistical confirmation under full multiple testing correction.**

The data supports building and validating the prototype. It does NOT support claiming a statistically rigorous 2% edge after Holm-Bonferroni at T=1,600 trials. The linear baseline (Step 1.5) is the correct and binding gate before committing CNN compute.

---

## Consensus Findings (all 4 council members agree)

### 1. The raw numbers are deceptive — effective sample count is much lower

| Metric | Nominal | Effective | Source |
|--------|---------|-----------|--------|
| Training windows | 400-560K | 100-250K | Serial correlation in labels (C1, C5, C6) |
| Params-to-samples ratio | 1:5.5 | 1:1.1 to 1:2.7 | After autocorrelation adjustment (C5) |
| Independent OB observations/symbol | 576K snapshots | 960 regime blocks | 4-hour autocorrelation in book state (C2) |
| Clean walk-forward folds | 3-4 | 2 | Mar 5-25 contaminated by 20+ experiments (C1) |

### 2. The binding constraint is SNR, not architecture or parameter count

At 52% accuracy, 48% of labels are effectively noise. The theoretical minimum samples for reliable 2% edge learning is ~576K (C6). We have ~500K nominal, ~100-250K effective. **We are at the threshold, not above it.**

The flat MLP's plateau at 9/23 symbols is more likely a signal-strength ceiling than an architecture limitation (C5). The CNN may reach 12-15/25, but the data cannot support the spec's target of 20+/25 symbols above 51% without a stronger underlying signal.

### 3. Regime diversity is the biggest gap

| Requirement | Available | Gap |
|------------|-----------|-----|
| Regime-robust learning | ~500 days across 3-4 regimes | 3x short (C5) |
| Single-regime learning | ~160 days | Adequate (C2, C5) |
| 100-event microstructure patterns | 160 days | Sufficient (C2) |
| 500-event macro-influenced patterns | 160 days | Marginal (C2, C6) |

160 days in crypto (Oct 2025 – Mar 2026) contains ~3 distinct market regimes. The model will learn the dominant training regime and partially generalize. Systematic regime-specific failures should be expected.

### 4. The linear baseline IS the data sufficiency test

All council members converge: if logistic regression cannot find signal in 15+/25 symbols, no architecture will rescue it. The mandatory Step 1.5 gate answers the data sufficiency question empirically — don't spend H100 compute until it passes.

---

## Key Disagreements

### Optimism spectrum

| Member | Verdict | Rationale |
|--------|---------|-----------|
| **Council-2** (most optimistic) | Sufficient for 100-event on liquid symbols | Correct normalization enables cross-symbol transfer; trade features dominate over stale OB |
| **Council-6** (cautiously optimistic) | Marginally sufficient, fixable | Stride=50 overlapping windows converts "marginal" to "adequate" at zero cost |
| **Council-1** (cautious) | Sufficient for prototyping, insufficient for confirmation | DSR deflation factor 3.3x kills any 52% edge claim at T=1,600 |
| **Council-5** (most pessimistic) | 3x short on regime diversity | CNN will memorize recent regime, not learn universal patterns |

### Cross-symbol pooling

- **Council-2:** Pooling justified for liquid symbols (BTC, ETH, SOL, BNB). NOT justified for memecoins (FARTCOIN, KBONK, KPEPE) without empirical validation.
- **Council-5:** Pooling makes 91K params serve 25 distinct classification problems — tighter than it looks.
- **Council-6:** Pooling acts as data augmentation and is necessary (single-symbol data is too small).

### Overlapping windows

- **Council-6:** Stride=50 during training is safe with 600-event embargo. 4x data multiplication, highest ROI change.
- **Council-1:** Inflates nominal sample count but does NOT increase effective independent samples. Label overlap at 500-event horizon is 60%. Useful for gradient diversity, not for statistical power.
- **Council-5:** Adjacent windows share market context even without input overlap. Effective N improvement is modest.

---

## Actionable Recommendations (priority order)

### Before training (zero compute cost)

1. **Start trial_log.csv NOW** — before the first experiment. Every config tested, including linear baseline sweeps. Non-negotiable for DSR computation. (C1)

2. **Treat Mar 5-25 as training, not test** — it's contaminated by 20+ main-branch experiments. Use only Folds 1-2 for walk-forward + April hold-out for final evaluation. (C1)

3. **Designate the hold-out symbol before any training** — pre-register (e.g., AVAX — mid-tier, not extreme) to prevent post-hoc selection. (C1, C2)

### Architecture tweaks (one-line changes)

4. **Use stride=50 for training DataLoader, stride=200 for val/test** — 4x training samples. The walk-forward embargo protects against leakage. (C6)

5. **Reduce 500-event head weight from 0.35 to 0.20, increase 100-event from 0.35 to 0.50** — the 500-event head is too noisy for equal weight with the primary target. (C6)

6. **Verify kernel_size=5** — RF=253 requires k=5. If k=3 was implemented, RF=127 and the first 73 events are invisible to last_position pooling. (C6)

### Monitoring (add to training loop)

7. **Per-symbol-day accuracy variance** — if std > 8-10%, the model is memorizing day-specific patterns. (C5)

8. **Gradient norm per head** — if h500 gradient > 2x h100 gradient, reduce h500 weight. (C6)

9. **Training accuracy check at epoch 30** — if train acc < 52% at horizon 100, over-regularization is suppressing signal. (C5)

### Experiments to run (after linear baseline passes)

10. **Liquidity-stratified model comparison** — train on top-12 liquid symbols only, compare accuracy on bottom-13 memecoins vs. the all-25 model. Single most informative experiment for the universality hypothesis. (C2)

11. **Masked event pretraining** — self-supervised on all 40GB unlabeled data before fine-tuning. Expected 1-2% accuracy gain. Pursue after baseline validates signal exists. (C6)

---

## The DSR Problem (Council-1's most important finding)

At T=1,600 trials (including all sweeps), the Deflated Sharpe Ratio deflation factor is ~3.3x. An observed Sharpe corresponding to 52% accuracy (~0.5) deflates to DSR ~0.15 — far below the 0.95 threshold.

**Implication:** The project can either:
- (a) Dramatically reduce trial count by pre-committing architecture + hyperparameters before seeing data (reduces T from 1,600 to ~50)
- (b) Accept that statistical confirmation requires a larger edge than 2%
- (c) Use the April hold-out as a single clean out-of-sample test (not subject to DSR deflation because it's a single pre-registered evaluation)

Option (c) is the pragmatic path — the April 14+ hold-out is the only evaluation not contaminated by multiple testing.

---

## Bottom Line for the User

Your 40GB dataset is **enough to build and test the hypothesis**. It is **not enough to prove it** under rigorous multiple testing correction. The practical path:

1. Build the linear baseline (Step 1.5) — this is the real data sufficiency test
2. If it passes, build the CNN with stride=50 overlapping windows
3. Use the April 14+ hold-out as the single clean evaluation
4. If it works there, you have a tradeable signal. If it doesn't, no amount of architecture tuning will fix it — you need more data or different features.

The data is not the bottleneck for exploration and learning. It IS the bottleneck for statistical proof.
