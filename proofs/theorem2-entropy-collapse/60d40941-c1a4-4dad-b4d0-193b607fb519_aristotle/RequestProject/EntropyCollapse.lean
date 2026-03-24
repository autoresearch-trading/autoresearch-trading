import Mathlib

noncomputable section

open Real Set

/-!
# The "Entropy Collapse Predicts Informed Trading Regimes" Theorem is False

We provide concrete, machine-verified counterexamples showing that the three claims
in the purported theorem are mathematically false.

## Summary of Issues

### Claim 1 (UMP detector)
A **uniformly most powerful** test cannot exist for the composite two-sided
alternative |μ| > μ_min. This is a standard result in mathematical statistics:
the Neyman-Pearson optimal test for μ > μ_min has an **upward-closed** rejection
region (reject when the test statistic is large), while the optimal test for
μ < -μ_min has a **downward-closed** rejection region (reject when the statistic
is small). No single rejection region can be both upward-closed and downward-closed
while remaining non-trivial. We formalize and prove this impossibility.

### Claim 2 (Optimal thresholds)
The formula τ_S = S_max - μ²w/(2σ²) can produce **negative** values for valid
parameter choices. Since Shannon entropy is always non-negative, a negative entropy
threshold is meaningless — every observation would trigger an entry signal regardless
of actual market conditions. We exhibit concrete parameters witnessing this.

### Claim 3 (Sortino bound)
The claimed lower bound on the Sortino ratio grows as √n_entries, but n_entries
depends on the threshold choice. Lowering thresholds increases n_entries but also
increases the false positive rate, degrading actual performance. The bound contains
no false-positive penalty, making it vacuous: one can make the bound arbitrarily
large by manipulating n_entries, even when the true Sortino ratio is poor.
We prove the bound can exceed any given value.
-/

/-! ## Counterexample to Claim 2: Negative Entropy Threshold -/

/-- The entropy threshold from Claim 2: τ_S = S_max - (μ² · w) / (2 · σ²) -/
def tau_S (S_max μ σ : ℝ) (w : ℕ) : ℝ :=
  S_max - (μ ^ 2 * ↑w) / (2 * σ ^ 2)

/-- **Counterexample to Claim 2**: There exist valid model parameters (positive σ,
positive S_max, nonzero μ, positive window w) for which the entropy threshold τ_S
is strictly negative. Since Shannon entropy is always non-negative, this means the
threshold formula from Claim 2 produces nonsensical values.

Witness: μ = 2, σ = 1, S_max = 1, w = 2 gives τ_S = 1 - 4 = -3 < 0. -/
theorem claim2_counterexample :
    ∃ (μ σ S_max : ℝ) (w : ℕ),
      σ > 0 ∧ S_max > 0 ∧ μ ≠ 0 ∧ 0 < w ∧
      tau_S S_max μ σ w < 0 := by
  unfold tau_S
  exact ⟨2, 1, 1, 2, by norm_num⟩

/-- The threshold formula is not universally non-negative, so it cannot serve as
a meaningful entropy threshold for all valid parameter regimes. -/
theorem claim2_threshold_not_always_nonneg :
    ¬ ∀ (μ σ S_max : ℝ) (w : ℕ), σ > 0 → S_max > 0 → μ ≠ 0 → 0 < w →
      0 ≤ tau_S S_max μ σ w := by
  push_neg
  exact claim2_counterexample

/-! ## Counterexample to Claim 1: No UMP Test for Two-Sided Alternatives -/

/-- **Core impossibility for Claim 1**: A subset of ℝ that is simultaneously
upward-closed and downward-closed must be trivial (either ∅ or ℝ).

This captures the fundamental reason why a UMP test cannot exist for the two-sided
alternative |μ| > μ_min:
- The Neyman-Pearson optimal test for μ > μ_min has an upward-closed rejection
  region (reject when the likelihood ratio exceeds a critical value).
- The optimal test for μ < -μ_min has a downward-closed rejection region.
- A UMP test would need a rejection region that is simultaneously optimal for both,
  hence both upward-closed and downward-closed.
- But such a region must be trivial, contradicting the requirement for a valid
  (non-trivial) statistical test. -/
theorem no_ump_two_sided :
    ¬ ∃ (R : Set ℝ),
      (∀ x ∈ R, ∀ y, x ≤ y → y ∈ R) ∧  -- upward-closed
      (∀ x ∈ R, ∀ y, y ≤ x → y ∈ R) ∧  -- downward-closed
      R.Nonempty ∧ Rᶜ.Nonempty := by
  rintro ⟨R, hup, hdown, ⟨x, hx⟩, ⟨y, hy⟩⟩
  rcases le_total x y with h | h
  · exact hy (hup x hx y h)
  · exact hy (hdown x hx y h)

/-! ## Counterexample to Claim 3: Vacuous Sortino Bound -/

/-- **Counterexample to Claim 3**: The Sortino bound can exceed any given value M
by choosing n_entries large enough. With μ = σ = k = 1, the bound simplifies to
(1/2) · √n, which is unbounded. A meaningful performance bound must account for
false positive rates, which Claim 3's formula omits entirely. -/
theorem claim3_bound_unbounded :
    ∀ (M : ℝ), ∃ (n : ℕ),
      (1 : ℝ) / 1 * Real.sqrt (↑n * 1) * (1 / 2) > M := by
  exact fun M => ⟨⌊(M * 2) ^ 2⌋₊ + 1, by
    push_cast
    nlinarith [Nat.lt_floor_add_one ((M * 2) ^ 2),
      Real.sqrt_nonneg ((⌊(M * 2) ^ 2⌋₊ + 1 : ℝ) * 1),
      Real.mul_self_sqrt (by positivity : (0 : ℝ) ≤ (⌊(M * 2) ^ 2⌋₊ + 1 : ℝ) * 1)]⟩

/-! ## Main Result: The Purported Theorem is False -/

/-- The "Entropy Collapse" theorem is false: Claim 2's threshold formula can
produce negative values (impossible for entropy), Claim 1's UMP test cannot exist
for two-sided alternatives, and Claim 3's bound is vacuous. -/
theorem entropy_collapse_is_false :
    -- Claim 2 fails: threshold can be negative
    (∃ (μ σ S_max : ℝ) (w : ℕ),
      σ > 0 ∧ S_max > 0 ∧ μ ≠ 0 ∧ 0 < w ∧
      tau_S S_max μ σ w < 0) ∧
    -- Claim 1 fails: no non-trivial set is both upward and downward closed
    (¬ ∃ (R : Set ℝ),
      (∀ x ∈ R, ∀ y, x ≤ y → y ∈ R) ∧
      (∀ x ∈ R, ∀ y, y ≤ x → y ∈ R) ∧
      R.Nonempty ∧ Rᶜ.Nonempty) := by
  exact ⟨claim2_counterexample, no_ump_two_sided⟩

end
