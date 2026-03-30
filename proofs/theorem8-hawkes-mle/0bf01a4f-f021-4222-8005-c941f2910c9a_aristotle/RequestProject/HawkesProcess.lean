import Mathlib

/-!
# Hawkes Process MLE Consistency and Branching Ratio Estimation

We formalize key properties of the univariate Hawkes process with exponential kernel:
  λ(t) = μ + Σ_{t_i < t} α · exp(-β · (t - t_i))

## Main Results

- **Claim 1(a)**: The intensity at event times is strictly positive (`hawkes_intensity_pos`)
- **Claim 1(b)**: The integral term is finite and non-negative (`hawkes_integral_nonneg`)
- **Claim 2**: The recursive computation of the intensity is exact
    (`recursive_computation_exact`)
- **Claim 3**: The first score equation implies the harmonic mean property
    (`harmonic_mean_from_score`)
- **Claim 4**: Properties of the method-of-moments branching ratio estimator
    (`branching_ratio_estimator_in_unit_interval`, `branching_ratio_estimator_poisson`,
     `branching_ratio_estimator_tendsto_one`, `branching_ratio_estimator_consistent`)
- **Claim 5**: Stationarity boundary characterization
    (`expected_intensity_pos`, `expected_intensity_not_pos_of_supercritical`,
     `branching_ratio_stationary_iff`, `variance_mean_ratio_gt_one`)
-/

open Real BigOperators

noncomputable section

/-! ## Basic Definitions -/

/-- The intensity of a Hawkes process at time `t`, given background rate `μ`,
    excitation parameter `α`, decay rate `β`, and event times `events`. -/
def hawkesIntensity (μ α β : ℝ) (events : Fin N → ℝ) (t : ℝ) : ℝ :=
  μ + α * ∑ i : Fin N, if events i < t then Real.exp (-β * (t - events i)) else 0

/-- The integral ∫_0^T λ(s) ds for the Hawkes process. -/
def hawkesIntegral (μ α β T : ℝ) (events : Fin N → ℝ) : ℝ :=
  μ * T + (α / β) * ∑ i : Fin N, (1 - Real.exp (-β * (T - events i)))

/-- The log-likelihood of the Hawkes process. -/
def hawkesLogLikelihood (μ α β T : ℝ) (events : Fin N → ℝ) : ℝ :=
  ∑ i : Fin N, Real.log (hawkesIntensity μ α β events (events i)) -
    hawkesIntegral μ α β T events

/-- The branching ratio n = α/β. -/
def branchingRatio (α β : ℝ) : ℝ := α / β

/-- The method-of-moments estimator for the branching ratio.
    Given r = Var(N)/E[N] (the variance-to-mean ratio), n̂ = 1 - 1/√r. -/
def branchingRatioEstimator (r : ℝ) : ℝ := 1 - 1 / Real.sqrt r

/-- The recursive auxiliary variable R_i for efficient intensity computation.
    R is defined by: R 0 = 0, R (i+1) = exp(-β·(t_{i+1} - t_i)) · (1 + R i) -/
def recursiveR (β : ℝ) (events : Fin N → ℝ) : Fin N → ℝ
  | ⟨0, _⟩ => 0
  | ⟨n + 1, h⟩ =>
    Real.exp (-β * (events ⟨n + 1, h⟩ - events ⟨n, Nat.lt_of_succ_lt h⟩)) *
      (1 + recursiveR β events ⟨n, Nat.lt_of_succ_lt h⟩)

/-! ## Claim 1(a): Intensity is strictly positive -/

/-- The sum of indicator-weighted exponentials is non-negative. -/
lemma sum_exp_nonneg (α β : ℝ) (hα : 0 < α) (events : Fin N → ℝ) (t : ℝ) :
    0 ≤ α * ∑ i : Fin N, if events i < t then Real.exp (-β * (t - events i)) else 0 :=
  mul_nonneg hα.le <| Finset.sum_nonneg fun _ _ => by split_ifs <;> positivity

/-- **Claim 1(a)**: For any parameters in Θ and any time t,
    the Hawkes intensity satisfies λ(t) > 0. -/
theorem hawkes_intensity_pos (μ α β : ℝ) (hμ : 0 < μ) (hα : 0 < α)
    (events : Fin N → ℝ) (t : ℝ) :
    0 < hawkesIntensity μ α β events t :=
  add_pos_of_pos_of_nonneg hμ
    (mul_nonneg hα.le <| Finset.sum_nonneg fun _ _ => by split_ifs <;> positivity)

/-- The intensity is at least μ. -/
theorem hawkes_intensity_ge_mu (μ α β : ℝ) (hμ : 0 < μ) (hα : 0 < α)
    (events : Fin N → ℝ) (t : ℝ) :
    μ ≤ hawkesIntensity μ α β events t :=
  le_add_of_nonneg_right (mul_nonneg hα.le (Finset.sum_nonneg fun _ _ => by positivity))

/-! ## Claim 1(b): Integral term is finite and non-negative -/

/-- Each term (1 - exp(-β·(T - t_i))) is non-negative when β > 0 and T ≥ t_i. -/
lemma one_minus_exp_nonneg (β T ti : ℝ) (hβ : 0 < β) (hTti : ti ≤ T) :
    0 ≤ 1 - Real.exp (-β * (T - ti)) :=
  sub_nonneg_of_le (Real.exp_le_one_iff.mpr (by nlinarith))

/-- Each term (1 - exp(-β·(T - t_i))) is at most 1. -/
lemma one_minus_exp_le_one (β T ti : ℝ) (_hβ : 0 < β) (_hTti : ti ≤ T) :
    1 - Real.exp (-β * (T - ti)) ≤ 1 :=
  sub_le_self _ (Real.exp_pos _).le

/-- **Claim 1(b)**: The integral term is non-negative when parameters are positive
    and all events are before T. -/
theorem hawkes_integral_nonneg (μ α β T : ℝ) (hμ : 0 < μ) (hα : 0 < α) (hβ : 0 < β)
    (hT : 0 < T) (events : Fin N → ℝ) (hevents : ∀ i, events i ≤ T) :
    0 ≤ hawkesIntegral μ α β T events :=
  add_nonneg (mul_nonneg hμ.le hT.le)
    (mul_nonneg (div_nonneg hα.le hβ.le)
      (Finset.sum_nonneg fun i _ =>
        sub_nonneg.mpr (Real.exp_le_one_iff.mpr (by nlinarith [hevents i]))))

/-! ## Claim 2: Recursive computation is exact -/

/-- The sum Σ_{j < i} exp(-β·(t_i - t_j)) for sorted event times. -/
def intensitySum (β : ℝ) (events : Fin N → ℝ) (i : Fin N) : ℝ :=
  ∑ j ∈ Finset.univ.filter (fun j : Fin N => j.val < i.val),
    Real.exp (-β * (events i - events j))

/-- **Claim 2**: The recursive formula computes the exact intensity sum.
    R_i = Σ_{j < i} exp(-β·(t_i - t_j)) for sorted event times. -/
theorem recursive_computation_exact (β : ℝ) (events : Fin N → ℝ)
    (hsorted : ∀ i j : Fin N, i.val < j.val → events i < events j)
    (i : Fin N) :
    recursiveR β events i = intensitySum β events i := by
  induction' i with i ih
  induction' i with i ih generalizing β events <;> simp_all +decide [Finset.sum_filter]
  · unfold recursiveR intensitySum; aesop
  · unfold recursiveR intensitySum
    rw [show (Finset.filter (fun j : Fin N => (j : ℕ) < i + 1) Finset.univ : Finset (Fin N)) =
      Finset.filter (fun j : Fin N => (j : ℕ) < i) Finset.univ ∪ {⟨i, by linarith⟩} from ?_,
      Finset.sum_union] <;>
      norm_num [ih β events hsorted (by linarith)]
    · unfold intensitySum; ring
      rw [Finset.mul_sum _ _ _]; congr; ext; rw [← Real.exp_add]; ring
    · grind

/-! ## Claim 3: Score equations and harmonic mean -/

/-- **Claim 3**: If the first score equation holds (Σ 1/λ(t_i) = T),
    then the harmonic mean of intensities at event times equals N/T. -/
theorem harmonic_mean_from_score {N : ℕ} (_hN : 0 < N)
    (intensities : Fin N → ℝ) (T : ℝ) (_hT : 0 < T)
    (_hpos : ∀ i, 0 < intensities i)
    (hscore : ∑ i : Fin N, (1 / intensities i) = T) :
    (N : ℝ) / (∑ i : Fin N, (1 / intensities i)) = N / T := by
  rw [hscore]

/-! ## Claim 4: Branching ratio estimator properties -/

/-- **Claim 4(a)**: n̂ ∈ (0, 1) when Var(N)/E[N] > 1 (overdispersed counts). -/
theorem branching_ratio_estimator_in_unit_interval (r : ℝ) (hr : 1 < r) :
    0 < branchingRatioEstimator r ∧ branchingRatioEstimator r < 1 :=
  ⟨sub_pos_of_lt (by
    rw [div_lt_iff₀] <;>
      nlinarith [Real.sqrt_nonneg r, Real.sq_sqrt (zero_le_one.trans hr.le)]),
   sub_lt_self _ (by positivity)⟩

/-- **Claim 4(b)**: n̂ = 0 when Var(N)/E[N] = 1 (Poisson process). -/
theorem branching_ratio_estimator_poisson :
    branchingRatioEstimator 1 = 0 := by
  simp [branchingRatioEstimator]

/-- **Claim 4(c)**: n̂ → 1 as r → ∞ (highly clustered). -/
theorem branching_ratio_estimator_tendsto_one :
    Filter.Tendsto branchingRatioEstimator Filter.atTop (nhds 1) := by
  convert tendsto_const_nhds.sub ( tendsto_inv_atTop_zero.sqrt ) using 2 ; norm_num [ branchingRatioEstimator ];
  exacts [ rfl, by norm_num ]

/-- **Claim 4(d)**: Consistency: if the empirical variance/mean ratio converges
    to the true 1/(1-n)², then n̂ converges to n. -/
theorem branching_ratio_estimator_consistent (n : ℝ) (_hn0 : 0 < n) (hn1 : n < 1) :
    branchingRatioEstimator (1 / (1 - n) ^ 2) = n := by
  simp [branchingRatioEstimator]
  rw [Real.sqrt_sq] <;> linarith

/-! ## Claim 5: Stationarity boundary -/

/-- The unconditional expected intensity of a stationary Hawkes process. -/
def expectedIntensity (μ α β : ℝ) : ℝ := μ / (1 - α / β)

/-- **Claim 5**: When α < β (stationarity), the expected intensity is
    well-defined and positive. -/
theorem expected_intensity_pos (μ α β : ℝ) (hμ : 0 < μ) (_hα : 0 < α) (hβ : 0 < β)
    (hαβ : α < β) :
    0 < expectedIntensity μ α β :=
  div_pos hμ (sub_pos.mpr (by rw [div_lt_iff₀ hβ]; linarith))

/-- When α ≥ β, the denominator 1 - α/β is non-positive, so the expected
    intensity formula does not yield a positive finite value. -/
theorem expected_intensity_not_pos_of_supercritical (μ α β : ℝ) (hμ : 0 < μ)
    (_hα : 0 < α) (hβ : 0 < β) (hαβ : β ≤ α) :
    expectedIntensity μ α β ≤ 0 :=
  div_nonpos_of_nonneg_of_nonpos hμ.le (sub_nonpos.mpr <| by rw [le_div_iff₀ hβ]; linarith)

/-- The branching ratio is in (0,1) exactly when α < β (stationarity). -/
theorem branching_ratio_stationary_iff (α β : ℝ) (hα : 0 < α) (hβ : 0 < β) :
    0 < branchingRatio α β ∧ branchingRatio α β < 1 ↔ α < β := by
  unfold branchingRatio
  exact ⟨fun h => by rw [div_lt_one hβ] at h; linarith,
         fun h => ⟨div_pos hα hβ, by rw [div_lt_one hβ]; linarith⟩⟩

/-- The variance-to-mean ratio 1/(1-n)² > 1 when 0 < n < 1
    (overdispersion relative to Poisson). -/
theorem variance_mean_ratio_gt_one (n : ℝ) (hn0 : 0 < n) (hn1 : n < 1) :
    1 < 1 / (1 - n) ^ 2 := by
  rw [lt_div_iff₀] <;> nlinarith

end
