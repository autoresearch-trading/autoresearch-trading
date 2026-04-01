# Research: Neural Network Architectures That Could Beat a Flat MLP for Tick-Scale Financial Classification

**Date:** 2026-03-30  
**Context:** Flat MLP (window=50 × 13 features → 676 input → 3×64 hidden → 3 classes, ~160K params, ensemble of 5 seeds) achieves Sortino=0.353 on test, 0.261 mean walk-forward. Has already beaten attention (0.061), LSTM-based, and XGBoost baselines.

---

## Summary

No single architecture offers a reliable, validated upgrade over a well-tuned small MLP for this problem profile (~100K samples, 13 hand-crafted microstructure features, 50-step window, noisy Triple Barrier labels). The two highest-probability candidates are **ModernTCN** (large-kernel depthwise convolutions, ICLR 2024, empirically #1 across 9 architectures × 12 financial instruments × 918 runs) and **N-HiTS** (hierarchical multi-rate MLP, conceptually aligned with multi-scale microstructure). KAN shows *marginal* accuracy gains on tabular benchmarks but is not validated for sequence data. Mamba/S4 is exciting but systematically untested on tick-level classification with small data. MoE has theoretical appeal but adds complexity without tabular evidence. Neural ODEs solve a problem we don't have (irregular sampling). TabPFN-v2.5 / RealMLP are the best-validated tabular alternatives but treat the window as flat features, losing temporal structure. **The risk of regression is high**—MLP beats attention and LSTM precisely because temporal inductive biases overfit on the available data size.

---

## Findings

### 1. State-Space Models: S4 and Mamba

**Mechanism:** S4 (Gu et al., 2022) learns structured state-space matrices to model long-range sequences with O(N log N) complexity. Mamba (Gu & Dao, 2023) extends this with *selective* state-space dynamics—the model learns *when* to remember vs. forget input, analogous to content-dependent LSTM gating but with global receptive fields and parallel training.

**Financial evidence:** Two papers address financial Mamba specifically. *MambaTS* (Cai et al., 2024; [arXiv:2405.16440](https://arxiv.org/abs/2405.16440)) improves long-term forecasting on ETTh/Weather benchmarks. *CMDMamba* (Frontiers AI, 2025) shows Mamba beats LSTM+attention for daily stock prediction. *S4M* (ICLR 2025, [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/7b2f0758334389b8ad0665a9bd165463-Paper-Conference.pdf)) addresses **missing values** in multivariate forecasting—directly relevant to irregular tick timing. However: none of these papers test at sub-second tick level against a well-tuned MLP baseline with microstructure features. *FSMamba* (Knowledge-Based Systems, 2025) is a dual-expert Mamba+attention hybrid, untested on microstructure.

**Parameter count:** A minimal Mamba block is ~2–3× the params of an equivalent MLP layer due to state dimension expansion. A 3-layer Mamba model matching our 160K budget would have only 1–2 blocks with small state dimension, likely losing selectivity benefits.

**Addresses the key challenge?** Partially. The selectivity mechanism could learn to attend to bursts of correlated trades (buy runs, OFI spikes) within the window. However, our window is only 50 steps—S4/Mamba's advantage is in *very long* sequences (>512 steps). For window=50, the MLP's flat receptive field is already global; there is no temporal bottleneck to solve.

**Verdict:** Low-priority experiment. Theoretical appeal, but no tick-scale evidence and our window length negates the primary advantage. If window is expanded (200–500 steps), revisit.

**Papers:**
- Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Cai et al., "MambaTS" [arXiv:2405.16440](https://arxiv.org/abs/2405.16440) (2024)
- Peng et al., "S4M: S4 for Multivariate Time Series with Missing Values" ICLR 2025

---

### 2. Mixture of Experts (MoE) for Regime Detection

**Mechanism:** MoE routes inputs to specialized sub-networks (experts) via a gating function. In the regime detection framing, different experts would specialize in trending, mean-reverting, or choppy regimes. Sparse MoE (Shazeer et al.) activates only top-K experts per input, keeping inference cost constant.

**Financial evidence:** *Adaptive Market Intelligence: A MoE Framework for Volatility-Sensitive Stock Forecasting* (Vallarino, arXiv:2508.02686, 2025) proposes MoE for stock prediction with volatility-regime routing. *A Dynamic Approach to Stock Price Prediction: RNN and MoE Across Volatility Profiles* ([arXiv:2410.07234](https://arxiv.org/abs/2410.07234), 2024) tests RNN vs. MoE on daily data—finds MoE *does* capture volatility-driven nonlinearities but at the cost of training instability and more data requirements.

**Parameter count concern:** Each expert is a full sub-network. 4 experts × 3-layer 64-wide MLP = ~640K params minimum, 4× our current budget. Sparse MoE doesn't help at this scale.

**Addresses the key challenge?** The core hypothesis is valid: different microstructure regimes (e.g., informed vs. noise trading, opening auction dynamics vs. intraday equilibrium) may need different feature transformations. However, **our current architecture is probably already learning this implicitly**—the 676→64 bottleneck forces the network to find regime-invariant representations, which is why it generalizes. An explicit MoE risks: (1) overfitting individual experts to rare regimes, (2) training instability with only ~100K samples, (3) load-balancing collapse where one expert dominates.

**More practical alternative:** Implicit mixture via LayerNorm + dropout + multi-seed ensemble already provides regime-sensitive averaging. Consider soft routing (e.g., mixture weights as additional outputs) only if we have evidence that regime labels cluster cleanly in our feature space (e.g., via clustering of `tfi_acceleration`, `vol_of_vol`).

**Verdict:** Medium-term experiment after we have stronger evidence of regime separability. Do not implement as primary architecture pivot.

**Papers:**
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated MoE" (2017)
- Vallarino, "A MoE Framework for Volatility-Sensitive Stock Forecasting" [arXiv:2508.02686](https://arxiv.org/pdf/2508.02686) (2025)

---

### 3. Temporal Convolutional Networks (TCN) — Including ModernTCN

**Mechanism:** TCNs apply causal dilated 1D convolutions to capture temporal patterns. Standard TCN (Bai et al., 2018) uses exponentially dilated kernels to grow receptive field without recurrence. **ModernTCN** (Luo & Wang, ICLR 2024) upgrades this: large-kernel *depthwise* convolutions, structural reparameterization at inference, multi-stage downsampling, optional RevIN normalization.

**Financial evidence (918-experiment benchmark):** This is the strongest signal in the literature. The controlled benchmark ([arXiv:2603.16886](https://arxiv.org/html/2603.16886v1)) runs 9 architectures × 12 financial instruments × 2 horizons × 3 seeds under identical HPO. **ModernTCN ranks #1 on 18/24 evaluation points** (75% win rate). PatchTST ranks #2. Architecture explains 99.9% of raw forecast variance vs. 0.01% for seed. ModernTCN's advantages are largest on low-noise markets (forex, equity indices); on high-noise crypto, it ties with PatchTST. LSTM is worst. DLinear (~1K params, linear decomposition) beats LSTM and Autoformer—confirming that inductive bias > raw capacity.

**Critical caveat:** This benchmark uses hourly OHLCV, not microstructure features. The features are **much simpler** than our 13-feature engineered inputs. Our MLP already performs temporal feature extraction from a well-curated feature set. The benchmark's finding that MLP (N-HiTS) and TCN (ModernTCN) beat attention and LSTM is *directionally consistent* with our results (MLP > attention).

**TCN vs. our architecture:** Our MLP flattens the (50, 13) window to 650 and appends per-feature stats (676 total). This is already a form of "feature-time joint encoding." A TCN would process each of the 13 feature channels separately with causal 1D convolutions across the 50 steps. The architectural question is: does explicit *local temporal pattern detection* (short 1D conv filters catching e.g., buy run acceleration over 3–5 steps) add value over the flat MLP's learned combinations?

**Our prior TCN attempt:** A TCN hybrid failed (one of the documented regressions). However, the failure may have been confounded by simultaneously changing other parameters—this is the "one change at a time" gotcha. A clean, minimal TCN matching our 160K budget deserves an isolated test.

**Recommended experiment:** Replace the MLP with a 2–3 layer TCN with kernel size 5–7, stride 1, dilation {1,2,4}, depth 3, ~64 channels, keeping RevIN + per-feature stats appended. Target <200K params. Use identical training config (focal loss, recency weighting, 5 seeds, 25 epochs).

**Papers:**
- Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
- Luo & Wang, "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis" ICLR 2024
- Ahmad Saidd, "A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments" [arXiv:2603.16886](https://arxiv.org/html/2603.16886v1) (2026) ← **primary reference**

---

### 4. Neural ODE / Continuous-Time Models

**Mechanism:** Neural ODEs (Chen et al., NeurIPS 2018) define the hidden state as the solution to an ODE: dh/dt = f(h,t,θ). For irregularly sampled time series, this allows exact interpolation at any point—the model naturally handles variable inter-event times. ODE-RNNs (Rubanova et al., NeurIPS 2019) combine ODE continuous dynamics with discrete jump updates at observation times.

**Financial evidence:** *Neural Jump ODEs* ([arXiv:2006.04727](https://arxiv.org/abs/2006.04727)) is designed for exactly this problem: continuous hidden state that jumps at event times (trades), relevant for unevenly spaced order flow. *U-Former ODE* ([arXiv:2602.11738](https://arxiv.org/html/2602.11738v1)) applies probabilistic ODE forecasting to irregular time series. A Google patent (US12462146B2) explicitly covers Neural ODE for irregularly-sampled time series.

**Relevance to our problem:** Our pipeline already normalizes for irregular timing—each step is exactly 100 consecutive trades (not time-equal intervals). The temporal irregularity is *absorbed at the feature computation stage* (trade arrival rate, VPIN, etc.). A continuous-time model would be solving a problem we've already engineered away. Neural ODEs also require ODE solver calls during training, which is 10–50× slower than standard forward passes.

**Verdict:** Not applicable. Our batch = 100 consecutive trades is the right discretization. Neural ODEs would only add value if we moved to raw event-level processing (each individual trade/quote update as input), which is a much larger architectural change. Skip.

**Papers:**
- Chen et al., "Neural Ordinary Differential Equations" NeurIPS 2018
- Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series" NeurIPS 2019
- Kidger et al., "Neural Controlled Differential Equations for Irregular Time Series" NeurIPS 2020

---

### 5. Kolmogorov-Arnold Networks (KAN)

**Mechanism:** KAN (Liu et al., 2024) replaces fixed activation functions with learnable univariate spline functions on edges rather than nodes. Each edge learns its own nonlinear transformation, making the network more interpretable (the learned functions can be visualized and sometimes identified as known mathematical functions). The theoretical basis is the Kolmogorov-Arnold representation theorem.

**Tabular evidence:** The primary benchmarking study ([arXiv:2406.14529](https://arxiv.org/abs/2406.14529)) tests KAN on standard tabular datasets vs. MLP. **Result: KAN achieves marginal accuracy gains on some datasets but is 10–100× slower to train** than equivalent MLPs due to the spline computation. TabKAN (Springer ML for Computational Science, 2025) and TabKANet (ICML workshop 2024) propose hybrid architectures. A withdrawn paper "Beyond Tree Models: KAN and gMLP for Large-Scale Financial Tabular Data" ([arXiv:2412.02097](https://arxiv.org/abs/2412.02097)) combined KAN with gated MLP for financial data, but was retracted—suggesting the results didn't hold up. *No evidence of KAN being tested on sequential microstructure data.*

**Parameter count:** KAN has comparable parameter count to MLP at small sizes, but the spline computation overhead makes it effectively more expensive. A KAN matching 160K MLP params would have similar capacity but ~5–10× slower training.

**For our use case:** KAN's theoretical advantage is for functions with *known smooth structure* (e.g., physics equations). Financial microstructure signals are noisy and non-smooth—the spline approximation may actually *underperform* the ReLU MLP's piecewise linear approximation which is already well-suited to threshold-like effects (e.g., "if order imbalance > X, predict up"). Additionally, our 5-seed ensemble already provides a form of function smoothing.

**Verdict:** Low priority. Interesting for interpretability (could potentially visualize what each feature's learned transformation looks like), but no evidence of performance gains for financial sequential classification and significant training overhead.

**Papers:**
- Liu et al., "KAN: Kolmogorov-Arnold Networks" [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) (2024)
- Poeta et al., "A Benchmarking Study of KANs on Tabular Data" [arXiv:2406.14529](https://arxiv.org/abs/2406.14529) (2024)
- TabKAN: [link.springer.com/article/10.1007/s44379-025-00042-y](https://link.springer.com/article/10.1007/s44379-025-00042-y) (2025)

---

### 6. Tabular-Specialized Architectures: TabNet, FT-Transformer, RealMLP, TabPFN

**TabNet** (Arik & Pfister, 2021): Uses sequential attention to select a sparse subset of features at each decision step. Interpretable but consistently underperforms gradient boosting and MLP on benchmarks. **Not recommended**—we already have feature selection implicitly via the bottleneck, and sequential attention would conflict with treating the 50×13 window as a temporal sequence.

**FT-Transformer** (Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data," NeurIPS 2021; [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)): Embeds each tabular feature as a token, applies standard Transformer. Significantly outperforms GBDT on some benchmarks (especially high-dimensional data). Relevant to our problem as a *feature-wise* transformer (treating each of 13 features + their temporal statistics as tokens). ~200–500K params at standard config.

**ResNet-like MLP** (same Gorishniy et al. paper): Adding skip connections to our MLP. The NeurIPS 2021 paper shows ResNet-MLP consistently beats plain MLP on tabular benchmarks. Extremely low implementation cost: add one skip connection around each linear layer. If we're looking for the highest-probability quick win, this is it.

**RealMLP** (Holzmüller et al., "Better by Default: Strong Pre-Tuned MLPs," 2025): A heavily tuned MLP with specific normalization, initialization, and regularization choices. On the TabArena benchmark (March 2026), RealMLP beats XGBoost in >75% of head-to-head comparisons. Key architectural choices: (1) layer normalization instead of batch norm, (2) SELU or GeLU activations, (3) careful weight initialization.

**TabPFN-v2.5** (2025): A prior-data fitted network—a transformer trained on millions of synthetic tabular datasets. At inference, conditions on the training set as context. Top performer on TabArena for datasets <150K rows. However: (1) it treats each sample as a flat feature vector, **losing temporal structure entirely**, (2) doesn't scale to our 100K training set as context easily, (3) no trajectory shape learning.

**The Gorishniy et al. LOB finding:** The "Revisiting Deep Learning for Tabular Data" paper's key finding is that *ResNet + FT-Transformer are competitive, but neither clearly beats well-tuned MLP*. The gap narrows with proper hyperparameter tuning. This is consistent with our experience (T47: MLP learns trajectory shapes that linear models miss, but attention-based temporal models don't consistently beat it).

**Verdict for our use case:**
- **ResNet skip connections**: High probability quick win. Add `x = x + linear(x)` residual connections. Essentially free.
- **FT-Transformer on features**: Potentially useful if we treat each of the 13 features as tokens + temporal stats. But our previous attention attempt suggests caution.
- **RealMLP (layer norm, SELU, better init)**: Worth testing one component at a time (we haven't specifically tested LayerNorm vs. BatchNorm).
- **TabPFN**: Not applicable—loses temporal structure.

**Papers:**
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" NeurIPS 2021 [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Holzmüller et al., "Better by Default: Strong Pre-Tuned MLPs" 2025 [arXiv:2407.04491](https://arxiv.org/abs/2407.04491)
- TabArena benchmark: [arxiv.org/abs/2506.16791](https://arxiv.org/abs/2506.16791) (March 2026)

---

### 7. Small Data + Noisy Labels: Architectures and Techniques That Help

This is arguably the most important dimension. Our Sortino variance (std=0.220 on walk-forward, per T46) is dominated by sampling noise, not regime shifts. Labels from Triple Barrier method with fee_mult=11.0 are inherently noisy—many steps are near-boundary.

**Key findings from the literature:**

**(a) Architecture size:** The strongest finding from our own experiments and the tabular benchmarks is that **smaller architectures generalize better under noise**. The TabArena critical difference diagram shows that DLinear (~1K params, linear!) beats LSTM (~172K params). The non-monotonic complexity-performance relationship (Jonckheere-Terpstra test p>0.35, [arXiv:2603.16886](https://arxiv.org/html/2603.16886v1) Section 4.7) means more parameters do NOT reliably help. Our hdim=64 > 128 > 256 finding is consistent across the literature.

**(b) Noisy label learning techniques:** FINE (NeurIPS 2021) and related methods identify *which* samples have noisy labels via eigenvector decomposition of the loss gradient, then reweight them. More practical: our **focal loss + recency weighting** already performs this implicitly—focal loss down-weights easy (high-confidence correct) and hard (near-boundary ambiguous) samples, while recency weighting reduces the influence of stale market regimes.

**(c) Sample selection:** "Curriculum learning" has been tested and failed for us (T47 follow-up). DivideMix (Li et al., 2020) separates clean vs. noisy labels via Gaussian mixture model on loss values—could be adapted to Triple Barrier to filter borderline labels. This is a **data-level technique independent of architecture**.

**(d) TabTransformer with noise:** The ResearchGate comparison ([link](https://www.researchgate.net/figure/Performance-of-TabTransformer-and-MLP-with-noisy-data-For-each-dataset-each-prediction_fig4_347124498)) shows that for noisy tabular data, TabTransformer and MLP perform similarly—confirming no architecture-level advantage of attention for noise robustness.

**(e) Ensemble size:** Our 5-seed ensemble is a strong regularizer. Increasing to 10 seeds would reduce variance by ~√2 at 2× training cost. High expected value for variance-limited performance.

**For small data specifically:** The LOB microstructure guide ([Briola et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)) makes a crucial point: *small-tick (low-liquidity) stocks have MCC near 0 at all horizons—the fundamental signal is too weak*. We have 9/23 passing symbols, and 14 failing ones may simply be uninformative. Architecture cannot fix absent signal.

**Papers:**
- FINE (Kim et al.), "FINE Samples for Learning with Noisy Labels" NeurIPS 2021
- Li et al., "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning" ICLR 2020
- Briola et al., "Deep Limit Order Book Forecasting: A Microstructural Guide" *Quantitative Finance* 2025 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)

---

## Priority-Ranked Experiment Recommendations

| Priority | Experiment | Expected Δ Sortino | Risk | Params | Implementation Cost |
|----------|------------|-------------------|------|--------|---------------------|
| 🟢 **High** | ResNet skip connections on current MLP (add `x = x + layer(x)`) | +0.02–0.05 | Low regression risk | ~160K (unchanged) | 2 lines of code |
| 🟢 **High** | Increase ensemble from 5 to 10 seeds | +0.03–0.07 (variance reduction) | None | Unchanged | 1 line of code |
| 🟡 **Med** | LayerNorm instead of BatchNorm (RealMLP finding) | ±0.02 | Low | Unchanged | 5 min |
| 🟡 **Med** | TCN (ModernTCN-style, small: kernel=5, 2 dilation levels, 64 channels) | +0.0–0.1 | Moderate | ~150K | 1 day |
| 🟡 **Med** | Label filtering: drop samples within ε=fee/2 of barrier boundary | +0.02–0.05 | Low | Unchanged | Feature engineering |
| 🔴 **Low** | Mamba (needs window expansion to ≥200 steps first) | Unknown | High | 200K+ | 2–3 days |
| 🔴 **Low** | KAN (interpretability only, not performance) | 0 (slower) | Medium | Similar | 1 day |
| 🔴 **Skip** | MoE (needs regime labels, more data) | Unknown | High | 640K+ | Week+ |
| 🔴 **Skip** | Neural ODE (solves the wrong problem) | Negative | High | 300K+ | Week+ |
| 🔴 **Skip** | TabPFN (loses temporal structure) | Negative | High | N/A | — |

---

## The Meta-Finding: Why MLP Wins in This Setting

The consistent finding across the controlled 918-experiment benchmark, the tabular DL literature, and our own T47 walk-forward analysis is that:

1. **Inductive bias > capacity**: Architecture choice explains 99.9% of variance ([arXiv:2603.16886](https://arxiv.org/html/2603.16886v1)). Seed explains 0.01%. This validates our 5-seed ensemble strategy.

2. **Smaller networks generalize better on noisy financial data**: Non-monotonic complexity-performance is empirically confirmed. DLinear (1K params, linear) beats Autoformer (438K params) on financial forecasting.

3. **Frequency-domain inductive biases hurt**: TimesNet and Autoformer (FFT-based) fail on financial data—confirming that financial microstructure does NOT have the strong periodicity of electricity/weather. Our ReLU MLP makes no periodicity assumptions.

4. **The bottleneck is alpha, not architecture**: The LOB microstructure paper ([Briola et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)) shows that model accuracy for uninformative stocks (small-tick, illiquid) is near-random regardless of architecture. Our 14 failing symbols may simply lack exploitable signal at this timescale.

5. **Directional accuracy ≠ Sortino**: The 918-experiment benchmark finds MSE-trained models have 50.08% directional accuracy across all architectures—essentially random. Yet our model earns Sortino 0.353. This confirms that our **Triple Barrier labeling + focal loss + fee-aware training** creates directional skill that pure price-regression training cannot. No amount of architectural complexity compensates for bad loss function design.

---

## Sources

### Kept
- **A Controlled Comparison of Deep Learning Architectures** ([arXiv:2603.16886](https://arxiv.org/html/2603.16886v1), 2026) — 918 experiments on financial data, directly comparable. Primary reference for TCN and tabular architecture findings.
- **Deep Limit Order Book Forecasting: A Microstructural Guide** (Briola et al., *Quantitative Finance* 2025, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)) — LOB-specific microstructure → predictability analysis. Key insight: signal strength is structural, not architectural.
- **Revisiting Deep Learning Models for Tabular Data** (Gorishniy et al., NeurIPS 2021, [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)) — FT-Transformer, ResNet-MLP canonical reference.
- **A Benchmarking Study of KANs on Tabular Data** ([arXiv:2406.14529](https://arxiv.org/abs/2406.14529), 2024) — KAN marginal gains, 10-100× training overhead.
- **Is Boosting Still All You Need for Tabular Data?** (M. Clark, 2026; [m-clark.github.io](https://m-clark.github.io/posts/2026-03-01-dl-for-tabular-foundational/)) — March 2026 TabArena state, RealMLP vs. XGB comparison, practical guidance.
- **MambaTS** ([arXiv:2405.16440](https://arxiv.org/abs/2405.16440), 2024) — Mamba for time series, benchmarks on standard datasets.
- **S4M** (ICLR 2025) — S4 for missing-value multivariate forecasting, closest to tick-irregular setting.

### Dropped
- **CMDMamba / Frontiers AI 2025** — Daily stock prediction, not tick-level, no MLP comparison.
- **MoE RNN paper** ([arXiv:2410.07234](https://arxiv.org/abs/2410.07234)) — Daily data, no microstructure features, MoE instability noted.
- **TabKANet** — KAN+Transformer hybrid, no financial validation, conference workshop.
- **Neural Jump ODE** ([arXiv:2006.04727](https://arxiv.org/abs/2006.04727)) — Theoretically relevant but our batching already handles irregular timing.
- **TabPFN papers** — Loses temporal structure entirely, not applicable.

---

## Gaps

1. **No controlled comparison of MLP vs. TCN vs. Mamba at tick-level with engineered microstructure features** (all studies use raw OHLCV or LOB snapshots, not our 39-feature set). The finding that TCN beats MLP on OHLCV may not transfer to already-extracted features.

2. **No study on ensemble size for small financial datasets** under noisy labels. 5 vs. 10 vs. 20 seeds at our exact problem scale is untested.

3. **Label quality intervention literature** (DivideMix, FINE) is entirely in computer vision with image noise, not Triple Barrier financial labels. Adapting these techniques requires domain translation.

4. **Symbol-level performance heterogeneity** (why 9/23 pass and 14 fail) is not explainable from the architecture literature alone—likely a function of per-symbol liquidity and signal-to-noise ratio, consistent with the LOB microstructure paper's tick-size findings.

**Suggested next steps:**
1. Single isolated test: ResNet skip connections on current v11b config.
2. Single isolated test: 10-seed ensemble vs. 5-seed.  
3. Analyze 14 failing symbols' feature distributions vs. passing symbols—if they are fundamentally unlearnable, architecture changes are irrelevant for those symbols.
4. If window is expanded to 200+ steps, Mamba becomes a credible experiment.
