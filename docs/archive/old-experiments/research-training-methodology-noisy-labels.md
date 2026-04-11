# Research: Training Methodology Improvements for Noisy Financial Classification

**Context**: Flat MLP (160K params, v11b), focal loss (γ=1.0) + class weights + recency weighting (decay=1.0),
triple-barrier labels, fee-constrained (fee_mult=11.0). Prior failures: UACE, curriculum learning, logit bias.
Current best: Sortino=0.353 fixed test, walk-forward mean=0.261.

---

## Summary

The most actionable near-term improvements for this system are: **(1) Generalized Cross Entropy (GCE) loss**
as a drop-in focal loss replacement with stronger theoretical noise resistance; **(2) Cleanlab-style confident
learning** to audit and prune the training set before retraining; and **(3) Meta-labeling** as a second-stage
filter that explicitly targets the fee constraint by improving trade precision. Online learning (EWC/replay) and
temporal noise correction (ICLR 2025) are medium-complexity improvements with solid theoretical backing.
Contrastive pretraining, multi-task learning, and knowledge distillation are unlikely to yield returns given
the model/data constraints here.

---

## Findings

### 1. Sample Selection & Cleaning

**1.1 Confident Learning / Cleanlab**

Northcutt et al.'s Confident Learning (JAIR 2021) estimates a *label noise transition matrix* T ∈ ℝ^{C×C}
from predicted out-of-fold probabilities and uses it to identify likely mislabeled samples without requiring
clean reference data. The algorithm: train k-fold cross-validator → collect p̂(ỹ|x) → build joint distribution
of noisy vs. thresholded predicted class → flag samples where ỹ ≠ argmax p̂ in high-confidence regions.

**Practical verdict for this system:**
- Triple-barrier labels have structural noise: the "timeout" outcome is most ambiguous. CL can identify the
  subset of timeout-labeled samples where the model is confidently predicting the opposite direction.
- Cost: requires one k-fold re-training pass (~5× current budget). Prune ~5-15% of data, retrain.
- Gain expectation: **moderate**. Noise rate for triple-barrier at fee_mult=11.0 is not uniform — it peaks
  near the timeout boundary. CL targets exactly those borderline samples.
- [Confident Learning paper](https://arxiv.org/abs/1911.00068) | [cleanlab library](https://github.com/cleanlab/cleanlab)

**1.2 Generalized Cross Entropy (GCE) Loss**

Zhang & Sabuncu (NeurIPS 2018) propose L_q(f(x), y) = (1 − f_y(x)^q) / q, which interpolates between
MAE (q→0, noise-robust but slow to learn) and CE (q=1, fast but noise-sensitive). Key insight: MAE loss
has a *symmetric* property that makes it noise-robust under the assumption that noise is instance-independent
and class-conditional. GCE at q≈0.7 provides the noise robustness of MAE while retaining CE-speed learning.

**Practical verdict:**
- **Drop-in replacement for focal loss.** No lr re-tuning should be as dramatic as switching loss families
  entirely, but do re-tune.
- Focal loss (your current setup) already down-weights easy samples. GCE down-weights high-loss samples
  (likely noisy), which is complementary and arguably more principled for label noise.
- For a *small* model (160K params), GCE > focal on noisy data in Zhang et al.'s ablations. Focal helps when
  classes are imbalanced by hard examples; GCE helps when noise is the issue.
- Symmetric variants (NLNL, NCE+AUL) go further but are harder to tune.
- [GCE paper](https://arxiv.org/abs/1805.07836) | [Normalized loss variants (ICML 2020)](http://proceedings.mlr.press/v119/ma20c/ma20c.pdf)

**1.3 Temporal Label Noise Correction (ICLR 2025)**

Nagaraj et al. (ICLR 2025) formalize *temporal label noise* — where noise rates Q(t) change over time — and
show existing static noise correction methods (cleanlab, VolMinNet) underperform when noise is non-stationary.
They propose jointly learning a time-varying noise matrix Q_ω(t) alongside the classifier via augmented
Lagrangian on a min-volume simplex constraint.

Results: on activity recognition with 30% mixed temporal noise, their Continuous Estimation method reduces
test error from ~29% (ignore noise) to ~4% — a 7× improvement over static correction methods (14%).

**Practical verdict for triple-barrier labels:**
- Triple-barrier noise IS temporally structured: high-volatility regimes produce more timeout labels (noisy);
  directional regimes produce cleaner TP/SL labels. This matches the decay/growth noise patterns in Table 1.
- The method requires sequence-level labels (they assume T time steps per sequence). Your setup has one label
  per window step — it would need to be re-framed.
- **Implementation complexity: high.** Needs augmented Lagrangian, separate Q_ω network. Best treated as a
  research direction, not a quick experiment.
- [ICLR 2025 paper](https://openreview.net/pdf?id=5o0phqAhsP) | [code](https://github.com/sujaynagaraj/TemporalLabelNoise)

---

### 2. Self-Training / Pseudo-Labeling

**2.1 Mean Teacher (EMA-based soft pseudo-labels)**

Tarvainen & Valpola (NeurIPS 2017) maintain an exponential moving average (EMA) "teacher" of model weights
θ_teacher ← α θ_teacher + (1−α) θ_student. The teacher generates soft pseudo-labels on *all* samples; the
student minimizes a consistency loss against teacher predictions alongside supervised loss on labeled data.

For noisy-label settings (not semi-supervised), this becomes: the EMA teacher acts as a noise-smoothed
ensemble, producing softer targets that resist memorizing noisy labels. Brown & Schifferer (arXiv 2109.14563)
show Robust Temporal Ensembling reduces test error by ~3-5% on noisy classification tasks.

**Practical verdict:**
- **Low implementation cost**: add 3 lines (EMA update) + consistency term to existing training loop.
- The EMA teacher is essentially free at inference (same architecture, just averaged weights).
- Risk: if the current model is already at a noise-floor, smoothing predictions may not help further.
- **Recommendation: try first** — lowest risk/effort ratio of any approach here.
- α=0.999, consistency weight λ=0.1 × supervised loss are good starting points.
- [Mean Teacher](https://papers.neurips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf)

**2.2 Self-Iterative Label Refinement (NeurIPS 2025)**

Asano et al. (NeurIPS 2025) propose iterative relabeling: (1) train on noisy labels, (2) use model's
predicted soft labels to replace ambiguous hard labels, (3) retrain. Critically, they handle the *confirmation
bias* problem (model overconfidently relabels its own errors) via an unlabeled learning regularizer.

**Practical verdict:**
- Triple-barrier labels are particularly amenable: "flat/neutral" timeout labels are the most likely to be
  relabeled as long/short after a trained model provides signal.
- Risk: confirmation bias. If your model's signal is weak (Sortino=0.35 is modest), relabeling based on
  model predictions amplifies errors.
- **Recommendation: worth one experiment.** Use high-confidence threshold (top 20% probability mass) for
  relabeling; leave the rest as original labels.
- [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115910) | [arXiv](https://arxiv.org/pdf/2502.12565)

---

### 3. Online / Incremental Learning

**3.1 Deep Incremental Learning for Financial Tabular Data**

Wong & Barahona (arXiv 2303.07925, Imperial College) present a framework specifically for financial tabular
datasets with distribution shifts. Key components:
- **Elastic Weight Consolidation (EWC)**: penalizes changes to "important" weights (measured by Fisher
  information diagonal) when adapting to new data → prevents catastrophic forgetting of old regime patterns.
- **Experience Replay**: maintain a reservoir of old samples; replay a fraction each update.
- **Progressive sliding window**: train on most recent N days, update every K days.

Results: outperforms batch retrain on 5-year financial regression datasets when regimes shift mid-period.

**Practical verdict for this system:**
- Your current setup is batch: train on 100d, eval on test 20d. Concept drift is real over 145 days.
- **Walk-forward already validates this** (T46 theorem) — fold variance is sampling noise, not regime
  shifts. This reduces urgency for online learning but doesn't eliminate it.
- The 2026-03-09 data endpoint is fixed; online learning would shine if you had live deployment.
- **Recommendation: medium priority.** Implement a rolling-window retrain (last 60d instead of 100d fixed)
  first — simpler and addresses the same problem without EWC complexity.
- [arXiv 2303.07925](https://export.arxiv.org/pdf/2303.07925v9.pdf)

---

### 4. Multi-Task Learning

**4.1 Joint Direction + Magnitude + Hold Prediction**

Ong & Herremans (SSRN/arXiv 2306.13661) train a deep MTL model for TSMOM portfolios, jointly predicting
cross-sectional rank, magnitude of return, and signal quality. Main gain: the return-magnitude auxiliary
task provides gradient signal that is correlated with but less noisy than the direction task alone.

Lopez de Prado's **Meta-Labeling** (2018/2024) is a structured MTL variant: train a *primary* direction
model, then train a *secondary* binary model that predicts whether the primary model's bet will succeed.
The secondary model learns to filter noise from the primary signal. Hudson & Thames (2024) validated this
across multiple strategies and found precision improvements of 10-15% with 20-30% trade count reduction.

**Practical verdict:**
- **Meta-labeling is highly relevant to your fee constraint.** Your Sortino is bounded by the fee barrier.
  A secondary "bet quality" classifier that reduces trade count while improving win rate directly attacks
  this constraint.
- Direction: your current classifier predicts flat/long/short. A meta-label model predicts: "given this
  signal predicted [long], will it clear fee_mult=11.0 barrier before timeout?"
- Training labels: use actual trade outcomes from evaluate() to label each signal.
- Cost: one additional small model (same size as current), no changes to primary.
- **Recommendation: high priority experiment.**
- [TSMOM MTL paper](https://export.arxiv.org/pdf/2306.13661v1.pdf)
- [Meta-labeling Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)

**4.2 Auxiliary Return Regression**

Add a regression head to predict the raw forward return (before barrier clipping). This provides denser
gradient signal — every sample has a real return, even if the direction label is timeout/noise. The two
losses share the MLP backbone; only the final head differs.

**Practical verdict:**
- Low implementation complexity (add 1 output head + regression loss term).
- Risk: regression target is also noisy (raw returns have fat tails, outliers dominate gradients).
- Huber loss or rank-IC loss mitigates this.
- [arXiv 2501.09760 — MTL Stock Prediction](https://arxiv.org/abs/2501.09760)

---

### 5. Contrastive Learning

**5.1 Asset Embeddings via Contrastive Learning**

Dolphin, Smyth & Dong (arXiv 2407.18645, July 2024) propose a contrastive framework for financial time
series. Key innovation: positive pairs are subwindows of the *same asset* across time; negative pairs are
different assets. Uses hypothesis testing to reject near-identical windows (avoiding false negatives).
Task: sector classification and portfolio optimization — not direction prediction.

ContraSim (arXiv 2502.16023) extends this to market prediction, learning a similarity space where
"similar market states" cluster together, then using k-NN on learned embeddings for classification.

**Practical verdict:**
- **Mismatch with your problem.** Contrastive methods learn *asset-level* embeddings or *market-state*
  similarity. Your task is *step-level* direction prediction at 1-2 second resolution.
- At tick scale (100-trade windows), the useful contrastive signal would be: "does this microstructure
  pattern (high OFI + spread spike) produce the same outcome as similar past patterns?" This is essentially
  k-NN in feature space — already captured by the MLP's learned representation.
- Pretraining contrastive + fine-tuning for classification requires 2-3× more compute and a careful
  self-supervised pretext task design. Given the MLP is at 160K params, there's limited capacity to
  benefit from pre-trained representations.
- **Recommendation: skip.** The overhead is not justified for a small model on a tick-level task.
- [arXiv 2407.18645](https://arxiv.org/abs/2407.18645) | [ContraSim arXiv 2502.16023](https://arxiv.org/pdf/2502.16023)

---

### 6. Knowledge Distillation

**6.1 Distill Large → Small**

Standard knowledge distillation (Hinton et al. 2015): train a large teacher, use soft logits as targets for
the small student. The soft targets encode inter-class similarity ("this was almost long but became flat"),
which is richer than hard labels for noisy-label settings.

In finance, most distillation work targets NLP (FinBERT → TinyBERT; GPT-4 → FinBERT). For tabular/MLP
trading, Prior Knowledge Distillation (IEEE 2021) uses domain knowledge (trend indicators) as teacher
signals rather than a larger neural model.

**Practical verdict:**
- **No obvious "large teacher" exists for your task.** You could train a large ensemble (500-seed, hdim=256)
  and distill to the current hdim=64 5-seed model, but the ensemble IS already your inference strategy
  (logit sum over 5 seeds). The student would just be learning from itself.
- An interesting variant: distill from a **gradient-boosted ensemble** (XGBoost on same features) trained
  without the fee-structure constraint, then use its soft predictions as auxiliary targets. XGBoost already
  loses to MLP (8/25 vs 18/25) so likely not useful.
- **Recommendation: skip** unless you have a fundamentally stronger model to distill from.

---

### 7. Temporal Ensemble Methods Beyond Simple Logit Averaging

**7.1 Snapshot Ensembling / Stochastic Weight Averaging**

Snapshot Ensembling (Huang et al. 2017): use cyclic learning rate schedules; save model snapshots at each
cycle's minimum. Each snapshot represents a different loss basin. Average predictions at inference.
SWA (Izmailov et al. 2018): average weights along SGD trajectory, not just predictions.

**Practical verdict:**
- Your current setup: 5 random-seed ensembles, logit sum. This already covers *across-seed* diversity.
- Snapshot ensembling adds *within-seed across-LR-cycle* diversity. Could capture different regime
  representations.
- SWA is a single-pass operation (average the last K checkpoint weights). At 25 epochs, save every 5
  epochs and average weights 20-25. This is essentially free.
- **Recommendation: try SWA** — 2 lines of PyTorch (`torch.optim.swa_utils`). No tuning required.
- [SWA](https://arxiv.org/abs/1803.05407) | Gains in noisy settings: typically +1-3% accuracy, consistent.

**7.2 Lookahead Optimizer**

Zhang et al. (2019) maintain a "slow weights" θ_slow updated every k steps as θ_slow += α(θ_fast − θ_slow).
Effectively smooths the optimization trajectory without changing the model. Consistently reduces variance
across seeds on noisy tasks.

**Practical verdict:**
- Drop-in optimizer wrapper. Compatible with Adam. No LR re-tune needed beyond the inner optimizer.
- [Lookahead paper](https://arxiv.org/abs/1907.08610)

---

### 8. Financial ML Training Tricks (2024–2026)

**8.1 Supervised Autoencoders + Triple Barrier (Nov 2024)**

Bieganowski (arXiv 2411.12753): SAE architecture reconstructs features while predicting direction label.
The autoencoder bottleneck forces the representation to capture only information relevant for prediction,
acting as a denoising filter. Tested on BTC/LTC/ETH with triple barrier labels. Key finding:
*moderate* noise augmentation (corrupt 10-20% of training features) significantly boosts Sharpe; excessive
corruption hurts. The SAE with bottleneck size ~50% of input features performs best.

**Practical verdict:**
- Your model already has a bottleneck (676→64). The SAE approach would add a reconstruction branch.
- Feature noise augmentation (Gaussian jitter on input features during training) is trivially implementable
  and may help with generalization. Your 39 features → try N(0, 0.05) jitter per feature per forward pass.
- [arXiv 2411.12753](https://arxiv.org/abs/2411.12753)

**8.2 Alpha Half-Life and Recency Weighting**

Quantitative Finance research (2025-2026) increasingly emphasizes that microstructure alpha decays within
hours to days. Your recency weighting (decay=1.0, ~2.7× recent vs. oldest) is already addressing this.
The "half-life of alpha" literature suggests: the optimal decay should match the *autocorrelation time* of
your features' predictive power, not a fixed constant.

**Recommendation:** Run a rolling IC (information coefficient = Spearman correlation between features and
next step return) by day. Fit an exponential decay to IC vs. lag-in-days. Set `decay` to match that
half-life. If IC decays in ~10 days, `decay = exp(-ln(2)/10)` per day.

**8.3 State Space Models for Microstructure (2026)**

Jonathan Kinlay (March 2026) surveys Mamba-class state space models for HFT. Main finding: at tick scale
(your regime), SSMs do not yet outperform flat MLPs due to insufficient training data and the ultra-short
effective memory needed. SSMs shine at minute+ resolution. **Confirms your v7 attention result.**

---

## Prioritized Recommendations

| Priority | Approach | Effort | Expected Gain | Rationale |
|----------|----------|--------|---------------|-----------|
| 🟢 **1** | **GCE loss** (q=0.7) | Low | Medium | Drop-in replacement; theoretically superior to focal for label noise |
| 🟢 **2** | **Meta-labeling** (secondary bet filter) | Medium | High | Directly attacks fee constraint; precision > recall tradeoff |
| 🟢 **3** | **Mean Teacher / EMA** soft targets | Low | Low–Medium | 3 lines of code; reduces label memorization |
| 🟢 **4** | **SWA** (stochastic weight averaging) | Trivial | Low–Medium | Free weight smoothing; last 5 epoch avg |
| 🟡 **5** | **Feature noise augmentation** (SAE-style jitter) | Low | Low–Medium | N(0,0.05) on features; encourages robustness |
| 🟡 **6** | **Cleanlab audit** + prune noisy samples | Medium | Medium | k-fold pass to identify timeout-boundary mislabels |
| 🟡 **7** | **Self-iterative relabeling** | Medium | Medium | High-confidence relabeling of timeout samples |
| 🟡 **8** | **Rolling-window train** (60d instead of 100d) | Low | Low | Simpler online learning; tests alpha decay hypothesis |
| 🔴 **9** | **Temporal noise correction** (ICLR 2025) | High | Medium–High | Theoretically well-matched but implementation complex |
| 🔴 **10** | **Auxiliary return regression head** | Medium | Low–Medium | More gradient signal but noisy target |
| ⬛ Skip | Contrastive pretraining | High | Low | Mismatch: asset-level vs tick-level; small model |
| ⬛ Skip | Knowledge distillation | High | Low | No strong teacher exists |
| ⬛ Skip | SSM / Attention architectures | High | Low | Already validated failure mode (v7) |

---

## Sources

**Kept:**
- Nagaraj et al. "Learning under Temporal Label Noise" (ICLR 2025) — directly addresses time-varying noise in sequential classification; code available
- Zhang & Sabuncu "Generalized Cross Entropy Loss for Training DNNs with Noisy Labels" (NeurIPS 2018) — canonical paper for noise-robust loss; GCE is the most practical recommendation
- Ma et al. "Normalized Loss Functions for Deep Learning with Noisy Labels" (ICML 2020) — symmetric loss extensions; NCE+AUL variants
- Northcutt et al. "Confident Learning: Estimating Uncertainty in Dataset Labels" (JAIR 2021) — cleanlab algorithm
- Bieganowski "Supervised Autoencoders with Fractionally Differentiated Features and Triple Barrier Labelling" (arXiv 2411.12753) — directly tested on crypto triple-barrier
- Tarvainen & Valpola "Mean Teachers are Better Role Models" (NeurIPS 2017) — EMA teacher
- Asano et al. "Self Iterative Label Refinement" (NeurIPS 2025) — iterative relabeling with confirmation bias protection
- Ong & Herremans "Constructing TSMOM Portfolios with Deep MTL" (arXiv 2306.13661) — MTL for financial trading
- Wong & Barahona "Deep Incremental Learning for Financial Temporal Tabular Datasets" (arXiv 2303.07925) — EWC/replay for financial concept drift
- Dolphin et al. "Contrastive Learning of Asset Embeddings from Financial Time Series" (arXiv 2407.18645) — contrastive for finance; good asset-level results
- Hudson & Thames "Does Meta Labeling Add to Signal Efficacy?" (2024) — empirical validation of meta-labeling
- Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI 2018) — SWA

**Dropped:**
- PLTA-FinBERT / TinyBERT papers — NLP-specific, not applicable to tabular trading
- SSCL-GBM — gradient boosting, already inferior to MLP for this task
- SSM/Mamba papers — architectures already ruled out by v7 experiments

---

## Gaps

1. **Noise rate quantification**: We don't have an empirical estimate of what % of triple-barrier labels at
   fee_mult=11.0 are genuinely noisy (vs. correctly labeled "flat" outcomes). Measuring this would sharpen
   which noise-handling methods are most worth pursuing.

2. **Information coefficient decay**: No measurement of how quickly microstructure features lose predictive
   power over days/weeks. Critical for setting optimal recency decay.

3. **GCE vs focal on financial data**: No paper directly compares GCE against focal loss on triple-barrier
   labeled microstructure data. This is a gap worth filling with one experiment.

4. **Meta-labeling implementation details**: Hudson & Thames show aggregate results but don't address the
   specific challenge of training the secondary classifier when positive labels (winning trades) are rare
   and imbalanced — a real concern at fee_mult=11.0.
