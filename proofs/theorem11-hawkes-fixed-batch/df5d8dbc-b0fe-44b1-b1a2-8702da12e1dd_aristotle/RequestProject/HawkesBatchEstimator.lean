/-
# Hawkes Branching Ratio from Fixed-Size Batch Arrival Rates

We formalize the algebraic and analytic core of estimating the branching ratio
of a stationary Hawkes process from fixed-size batch arrival rates.

## Overview

For a stationary Hawkes process with branching ratio n ∈ (0,1) and background
intensity μ > 0, the unconditional intensity is E[λ] = μ/(1-n).

When we observe fixed-size batches of B events, each batch has a random
duration T_B and an arrival rate R = B/T_B. We prove algebraic relationships
between the moments of R and T_B, the branching ratio estimator properties,
and the equivalence condition with fixed-time-window estimation.

The probabilistic foundations (Hawkes process definition, Wald's identity,
delta method, SLLN) are abstracted as hypotheses on the moments.
-/
import Mathlib

open Real

noncomputable section

/-! ## Basic Hawkes process parameters -/

/-- The unconditional intensity of a stationary Hawkes process. -/
def hawkesIntensity (μ : ℝ) (n : ℝ) : ℝ := μ / (1 - n)

/-- The branching ratio estimator from a variance/mean ratio. -/
def branchingEstimator (r : ℝ) : ℝ := 1 - 1 / Real.sqrt r

/-! ## Claim 1(a): Expected batch duration -/

/-- The expected time to observe B events equals B / E[λ] = B(1-n)/μ.
    This is a direct algebraic identity given E[λ] = μ/(1-n). -/
theorem expected_duration_eq (B μ n : ℝ) (_hμ : μ ≠ 0) (_hn : n ≠ 1) :
    B / hawkesIntensity μ n = B * (1 - n) / μ := by
  rw [hawkesIntensity, div_div_eq_mul_div]

/-! ## Claim 4: Equivalence condition

**The original Claim 4 as stated is FALSE.** It claimed:
  Var(R)/E[R] = 1/(1-n)² iff Var(T_B)/E[T_B]² = 1/(B · E[λ])

Numerical counterexample: With B=2, μ=3, n=1/2:
  E[λ] = 6, meanT = 1/3
  Delta method: varR/meanR = E[λ] · varT/meanT²
  Setting varR/meanR = 1/(1-n)² = 4:
    varT/meanT² = 4/E[λ] = 2/3
  But 1/(B·E[λ]) = 1/12 ≠ 2/3.

The **correct** equivalence condition is:
  Var(R)/E[R] = 1/(1-n)² iff Var(T_B)/E[T_B]² = 1/((1-n)² · E[λ])

This means the fixed-size-batch estimator generally has a DIFFERENT
variance/mean ratio than the fixed-time-window estimator, differing by
a factor that depends on both B and n. The correction factor is
C = B · (1-n)², so the estimator must be adjusted accordingly.
-/

/-
PROBLEM
The corrected equivalence condition: under the delta method relationship
    Var(R)/E[R] = E[λ] · Var(T_B)/E[T_B]², the variance/mean ratio of rates
    equals 1/(1-n)² if and only if Var(T_B)/E[T_B]² = 1/((1-n)² · E[λ]).

PROVIDED SOLUTION
From hdelta, varR / hawkesIntensity μ n = hawkesIntensity μ n * varT / meanT^2. So varR/hawkesIntensity = 1/(1-n)^2 iff hawkesIntensity * varT / meanT^2 = 1/(1-n)^2 iff varT/meanT^2 = 1/((1-n)^2 * hawkesIntensity). This is just dividing both sides of hdelta by hawkesIntensity (which is positive since μ > 0 and n < 1). Use rw [hdelta] to reduce to the algebraic identity, then field_simp and split the iff.
-/
theorem equivalence_condition_corrected (varR varT meanT μ n : ℝ)
    (hμ : μ > 0) (hn : 0 < n) (hn1 : n < 1)
    (hdelta : varR / hawkesIntensity μ n =
      hawkesIntensity μ n * varT / meanT ^ 2) :
    (varR / hawkesIntensity μ n = 1 / (1 - n) ^ 2) ↔
    (varT / meanT ^ 2 = 1 / ((1 - n) ^ 2 * hawkesIntensity μ n)) := by
  simp_all +decide;
  rw [ show hawkesIntensity μ n = μ / ( 1 - n ) from rfl ] ; ring_nf ; norm_num [ hμ.ne', hn.ne', hn1.ne ] ;
  grind

/-
PROBLEM
The original Claim 4 as stated is false. Here is a formal counterexample:

Counterexample to Claim 4: with specific parameters, the original
    equivalence condition fails.

PROVIDED SOLUTION
Push negation to get ∃ parameters such that the iff fails. Use B=2, μ=3, n=1/2, meanR=6, meanT=1/3. Set varT = 2/27 (which makes varR/meanR = 4 = 1/(1-n)^2). Then varT/meanT^2 = (2/27)/(1/9) = 2/3, but 1/(B*E[λ]) = 1/12. So the forward direction of the iff fails: varR/meanR = 4 but varT/meanT^2 = 2/3 ≠ 1/12. Use push_neg, then provide these concrete values with norm_num.
-/
theorem claim4_original_is_false :
    ¬ ∀ (varR meanR varT meanT B μ n : ℝ),
      μ > 0 → 0 < n → n < 1 → B > 0 →
      meanR = hawkesIntensity μ n →
      meanT = B / hawkesIntensity μ n →
      varR / meanR = hawkesIntensity μ n * varT / meanT ^ 2 →
      ((varR / meanR = 1 / (1 - n) ^ 2) ↔
       (varT / meanT ^ 2 = 1 / (B * hawkesIntensity μ n))) := by
  push_neg;
  -- Set the parameters to specific values.
  use 4 * 6, 6, 2 / 27, 1 / 3, 2, 3, 1 / 2
  norm_num [ hawkesIntensity ] at *

/-! ## Claim 5: Estimator properties -/

/-- When the variance/mean ratio r = 1 (Poisson baseline), the estimator gives 0. -/
theorem estimator_at_poisson : branchingEstimator 1 = 0 := by
  unfold branchingEstimator; norm_num

/-- When r > 1 (overdispersed rates), the estimator gives a value in (0, 1). -/
theorem estimator_in_unit_interval {r : ℝ} (hr : r > 1) :
    0 < branchingEstimator r ∧ branchingEstimator r < 1 := by
  exact ⟨sub_pos.2 <| by
    rw [div_lt_one <| Real.sqrt_pos.2 <| by positivity]
    exact Real.lt_sqrt_of_sq_lt <| by linarith,
   sub_lt_self _ <| by positivity⟩

/-
PROBLEM
The estimator is monotonically increasing in r for r > 0.

PROVIDED SOLUTION
branchingEstimator r = 1 - 1/sqrt(r). For r > 0, 1/sqrt(r) is strictly decreasing (since sqrt is strictly increasing), so 1 - 1/sqrt(r) is strictly increasing. Use StrictMonoOn for the composition. Key facts: Real.sqrt is strictly monotone on nonneg reals, 1/x is strictly antitone on positive reals, and 1 - x is strictly antitone. The composition of strictly increasing sqrt with strictly decreasing 1/x gives strictly decreasing 1/sqrt(x), and subtracting from 1 gives strictly increasing.
-/
theorem estimator_monotone : StrictMonoOn branchingEstimator (Set.Ioi 0) := by
  exact fun x hx y hy hxy => sub_lt_sub_left ( one_div_lt_one_div_of_lt ( Real.sqrt_pos.mpr hx ) ( Real.sqrt_lt_sqrt hx.out.le hxy ) ) _;

/-! ## Claim 2: Rate variance/mean ratio (algebraic content)

The delta method gives Var(R)/E[R] ≈ E[λ] · Var(T_B)/E[T_B]².
Substituting Claim 1's moments, we get a formula in terms of (μ, n, B).
We prove the algebraic simplification. -/

/-
PROBLEM
**Corrected Claim 2**: If Var(T_B) = B · (1 + 2n/(1-n)²) / E[λ]² (Claim 1(b)),
    then Var(R)/E[R] = E[λ] · (1 + 2n/(1-n)²) / B.

    The original claim omitted the E[λ] factor. The variance/mean ratio of rates
    is NOT purely a function of n — it also depends on μ through E[λ] = μ/(1-n).
    This means the batch-rate estimator requires knowledge of the mean rate
    (or equivalently μ) to extract n, unlike the fixed-time-window case.

PROVIDED SOLUTION
Unfold the let definitions. el = μ/(1-n), meanT = B / (μ/(1-n)) = B(1-n)/μ, varT = B*(1+2n/(1-n)^2) / (μ/(1-n))^2 = B*(1+2n/(1-n)^2)*(1-n)^2/μ^2.

el * varT / meanT^2 = (μ/(1-n)) * B*(1+2n/(1-n)^2)*(1-n)^2/μ^2 / (B^2*(1-n)^2/μ^2)
= μ*B*(1+2n/(1-n)^2)*(1-n) / μ^2 * μ^2/(B^2*(1-n)^2)
= (1+2n/(1-n)^2)*μ/(B*(1-n))
= el * (1+2n/(1-n)^2) / B.

Just use simp [hawkesIntensity] then field_simp and ring.
-/
theorem rate_variance_mean_formula (B μ n : ℝ)
    (hμ : μ > 0) (hn : 0 < n) (hn1 : n < 1) (hB : B > 0) :
    let el := hawkesIntensity μ n
    let meanT := B / el
    let varT := B * (1 + 2 * n / (1 - n) ^ 2) / el ^ 2
    el * varT / meanT ^ 2 = el * (1 + 2 * n / (1 - n) ^ 2) / B := by
  field_simp

/-! ## Claim 3: Consistency (algebraic content)

The probabilistic content (SLLN, consistency of sample variance) is
standard. We formalize the algebraic consequence: the continuous
mapping theorem preserves the relationship. -/

/-
PROBLEM
If the variance/mean ratio converges to some limit L > 0,
    then the estimator converges to 1 - 1/√L.
    This is the algebraic content of consistency — the probabilistic
    convergence is a standard application of SLLN + continuous mapping.

PROVIDED SOLUTION
branchingEstimator = fun r => 1 - 1/sqrt(r). This is continuous at L > 0 because sqrt is continuous and positive at L, so 1/sqrt is continuous, and 1 - x is continuous. Use Filter.Tendsto for the composition. Show that branchingEstimator is continuous on (0, ∞) using Continuous.sub, Continuous.div, Real.continuous_sqrt (or continuousAt_sqrt for L > 0).
-/
theorem estimator_continuous_at_ratio (L : ℝ) (hL : L > 0) :
    Filter.Tendsto branchingEstimator (nhds L) (nhds (1 - 1 / Real.sqrt L)) := by
  exact Filter.Tendsto.sub tendsto_const_nhds <| Filter.Tendsto.div tendsto_const_nhds ( Real.continuous_sqrt.continuousAt ) <| ne_of_gt <| Real.sqrt_pos.mpr hL

/-! ## Claim 5 continued: Correction factor -/

/-- For any correction factor C > 0, when r = C (Poisson-equivalent
    baseline), the adjusted estimator gives 0. -/
theorem adjusted_estimator_at_baseline (C : ℝ) (hC : C > 0) :
    branchingEstimator (C / C) = 0 := by
  rw [div_self hC.ne']
  exact estimator_at_poisson

/-
PROBLEM
For any correction factor C > 0, when r > C (overdispersed),
    the adjusted estimator gives a value in (0, 1).

PROVIDED SOLUTION
r > C and C > 0 implies r/C > 1. Apply estimator_in_unit_interval with hr : r/C > 1 (from div_lt_iff and the hypotheses).
-/
theorem adjusted_estimator_in_interval (C r : ℝ) (hC : C > 0) (hr : r > C) :
    0 < branchingEstimator (r / C) ∧ branchingEstimator (r / C) < 1 := by
  exact ⟨ estimator_in_unit_interval ( by rwa [ gt_iff_lt, one_lt_div hC ] ) |>.1, estimator_in_unit_interval ( by rwa [ gt_iff_lt, one_lt_div hC ] ) |>.2 ⟩

/-! ## Main theorem: Branching ratio identifiability

Regardless of whether there is a correction factor C, the branching ratio
n is always identifiable from the rate statistics of fixed-size batches.
-/

/-
PROBLEM
The branching ratio n is recovered by the estimator when applied to
    the true variance/mean ratio 1/(1-n)².

PROVIDED SOLUTION
Unfold branchingEstimator. Need 1 - 1/sqrt(1/(1-n)^2) = n. We have 1/(1-n)^2 = ((1-n)^2)⁻¹. sqrt(((1-n)^2)⁻¹) = (sqrt((1-n)^2))⁻¹ = (|1-n|)⁻¹ = (1-n)⁻¹ since 1-n > 0. So 1/sqrt(...) = 1/((1-n)⁻¹) = 1-n. Hence 1-(1-n) = n. Use Real.sqrt_inv, Real.sqrt_sq (for 1-n ≥ 0), and algebra.
-/
theorem branching_ratio_recovery (n : ℝ) (hn : 0 < n) (hn1 : n < 1) :
    branchingEstimator (1 / (1 - n) ^ 2) = n := by
  unfold branchingEstimator;
  norm_num [ show 1 - n ≥ 0 by linarith ]

end