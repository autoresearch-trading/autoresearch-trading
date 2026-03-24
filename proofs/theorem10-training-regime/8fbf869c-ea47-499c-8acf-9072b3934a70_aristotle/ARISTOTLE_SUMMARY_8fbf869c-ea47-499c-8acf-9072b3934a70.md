# Summary of changes
## Summary

All 5 claims have been formalized and proved in Lean 4 across 5 files, with **zero `sorry`** remaining and only standard axioms used. One claim (5b) was found to be **false as stated** and was corrected with a formal counterexample.

### Files

- **`RequestProject/Claim1.lean`** — Focal loss properties (6 theorems)
  - (a) `focal_loss_gamma_zero`: FL with γ=0 equals weighted cross-entropy ✓
  - (b) `focal_loss_ratio_tendsto_zero`: FL/CE ratio → 0 as p_t → 1 when γ > 0 ✓
  - (c) `focal_loss_ratio_at_zero`: FL/CE ratio → 1 as p_t → 0 ✓
  - (d) `grad_factor_at_third`, `grad_factor_at_half`, `grad_factor_third_gt_half`: gradient values at p=1/3 and p=1/2 computed exactly (2+log3 and 1+log2), and the ordering proved ✓

- **`RequestProject/Claim2.lean`** — Class imbalance and optimal weights (4 theorems)
  - (a) `inverse_freq_equal_contribution`: p·(1/p) = 1 ✓
  - (b) `weighted_accuracy_majority`: For K classes with w_c = 1/p_c, majority-class weighted accuracy = 1/K (general) ✓
  - `weighted_accuracy_three_class`: Concrete verification with (0.65, 0.175, 0.175) ✓

- **`RequestProject/Claim3.lean`** — Regime gate trade-off (4 theorems)
  - (b) `gating_threshold`: k·√φ ≥ 1 ↔ φ ≥ 1/k² (the critical gating fraction) ✓
  - (c) `phi_min_k_1_5`: φ_min = 4/9 for k=1.5 ✓
  - `phi_min_k_2`: φ_min = 1/4 for k=2 ✓
  - `phi_min_approx`: 4/9 > 0.44 ✓

- **`RequestProject/Claim4.lean`** — Model capacity bounds (9 theorems)
  - (a) `param_sample_ratio`: P/N > 1 ↔ P > N ✓
  - (b) `v9_ratio_lt_v8_ratio`, `v9_ratio_concrete`: v9 ratio < v8 ratio ✓
  - Exact ratio values: P_v9/N = 0.022, P_v8/N = 0.2768 ✓
  - (c) `effective_complexity_v8` = 2300, `effective_complexity_v9` = 375, ratio < 0.164 ✓

- **`RequestProject/Claim5.lean`** — Ensemble majority voting (5 theorems)
  - ⚠️ **Claim 5(b) is FALSE as stated.** The original claim says α > 1/3 suffices for ensemble to beat individual. This is wrong for majority voting (≥3 of 5).
  - `ensemble_counterexample`: At α = 2/5 > 1/3, ensemble accuracy (992/3125 ≈ 0.317) < individual accuracy (0.4) ✓
  - **Corrected claim**: `ensemble_beats_individual`: For α > **1/2**, ensemble accuracy strictly exceeds α ✓
  - `ensemble_accuracy_at_half`: At α = 1/2, ensemble accuracy = 1/2 exactly (boundary case) ✓
  - `ensemble_accuracy_simplified`: ensembleAccuracy5(α) = 6α⁵ - 15α⁴ + 10α³ ✓

### Key Finding on Claim 5
The Condorcet jury theorem threshold for majority voting is α > 1/2 regardless of the number of classes (not α > 1/K). This is because majority voting requires ≥⌈M/2⌉ models to agree on the correct class, and the count of correct votes follows Binomial(M, α) independently of how errors distribute among wrong classes. The threshold α > 1/K would apply to plurality voting as M → ∞, but not for majority voting with M=5.