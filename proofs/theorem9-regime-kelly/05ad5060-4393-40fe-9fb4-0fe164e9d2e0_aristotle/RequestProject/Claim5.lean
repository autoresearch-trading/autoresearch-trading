/-
# Claim 5: Combined strategy optimality

The regime-gated Kelly barrier strategy achieves strictly higher geometric
growth than the ungated strategy whenever the gated accuracy is sufficient
for positive growth and the ungated accuracy is not.

NOTE: The original claim used α_min(f) = 1/2 + 1/(2f) as the threshold for
log-growth positivity. This is INCORRECT: α_min(f) is the threshold for
positive EXPECTED VALUE (linear payoff), not for positive GEOMETRIC GROWTH
(logarithmic payoff). The log-growth threshold depends on both f and c and
is given implicitly by the equation
  α·log(1+fc-c) + (1-α)·log(1-fc-c) = 0.

We reformulate correctly: G(α) is strictly increasing and continuous in α
(for fixed f, c, p_trade with appropriate constraints), so there exists a
unique threshold α* where G = 0, and G > 0 iff α > α*.

The combined theorem then states: if α_all < α* ≤ α_regime, then
G_all ≤ 0 < G_regime.
-/
import Mathlib

open Real

noncomputable section

/-- Geometric growth rate as a function of win/loss probabilities and fee structure -/
def G_growth (p_win p_loss : ℝ) (f c : ℝ) : ℝ :=
  p_win * Real.log (1 + f*c - c) + p_loss * Real.log (1 - f*c - c)

/-- The "inner" growth rate (factoring out p_trade):
    g(α) = α·log(1+fc-c) + (1-α)·log(1-fc-c) -/
def g_inner (α f c : ℝ) : ℝ :=
  α * Real.log (1 + f*c - c) + (1 - α) * Real.log (1 - f*c - c)

/-- G_growth factors as p_trade · g_inner -/
theorem G_growth_eq_ptrade_ginner (p_trade α f c : ℝ) :
    G_growth (α * p_trade) ((1 - α) * p_trade) f c = p_trade * g_inner α f c := by
  unfold G_growth g_inner; ring

/-
PROBLEM
g_inner is strictly increasing in α when f > 1, 0 < c, fc + c < 1

PROVIDED SOLUTION
g_inner α f c = α·log(1+fc-c) + (1-α)·log(1-fc-c) = α·(log(1+fc-c) - log(1-fc-c)) + log(1-fc-c). This is affine in α with slope log(1+fc-c) - log(1-fc-c). Since f>1, c>0, fc+c<1: 1+fc-c > 1-fc-c > 0 (because fc-c > -(fc+c), and 1+fc-c > 1-fc-c iff 2(fc-c) > 0, wait fc-c = c(f-1) > 0 since f>1). So log(1+fc-c) > log(1-fc-c), the slope is positive, and g_inner is strictly increasing in α.
-/
theorem g_inner_strictMono_alpha (f c : ℝ) (hf : 1 < f) (hc : 0 < c) (hfc : f*c + c < 1) :
    StrictMono (fun α => g_inner α f c) := by
      refine' fun α β hαβ => _;
      -- The slope of $g_inner$ is positive since $\log(1+fc-c) > \log(1-fc-c)$.
      have h_slope_pos : Real.log (1 + f * c - c) > Real.log (1 - f * c - c) := by
        exact Real.log_lt_log ( by nlinarith ) ( by nlinarith );
      unfold g_inner; nlinarith;

/-
PROBLEM
g_inner is continuous in α

PROVIDED SOLUTION
g_inner α f c = α·A + (1-α)·B where A, B are constants (in α). This is continuous in α as an affine function. Use Continuous.add, Continuous.mul, continuous_const, continuous_id.
-/
theorem g_inner_continuous_alpha (f c : ℝ) :
    Continuous (fun α => g_inner α f c) := by
      apply_rules [ Continuous.add, Continuous.mul, continuous_id, continuous_const ];
      exact continuous_neg

/-
PROBLEM
g_inner(0) < 0 (pure losses)

PROVIDED SOLUTION
g_inner 0 f c = 0·log(1+fc-c) + 1·log(1-fc-c) = log(1-fc-c). Since fc+c < 1, we have 1-fc-c > 0 but 1-fc-c < 1 (since fc+c > 0 because f>1, c>0). So log(1-fc-c) < log 1 = 0.
-/
theorem g_inner_at_zero_neg (f c : ℝ) (hf : 1 < f) (hc : 0 < c) (hfc : f*c + c < 1) :
    g_inner 0 f c < 0 := by
      -- Substitute α = 0 into the definition of g_inner.
      simp [g_inner];
      exact Real.log_neg ( by nlinarith ) ( by nlinarith )

/-
PROBLEM
g_inner(1) > 0 (pure wins)

PROVIDED SOLUTION
g_inner 1 f c = 1·log(1+fc-c) + 0·log(1-fc-c) = log(1+fc-c). Since f>1 and c>0, fc-c = c(f-1) > 0, so 1+fc-c > 1, so log(1+fc-c) > log 1 = 0.
-/
theorem g_inner_at_one_pos (f c : ℝ) (hf : 1 < f) (hc : 0 < c) (hfc : f*c + c < 1) :
    0 < g_inner 1 f c := by
      -- Since $g_inner 1 f c = \log(1 + fc - c)$ and $1 + fc - c > 1$, we have $\log(1 + fc - c) > 0$.
      have h_log_pos : Real.log (1 + f * c - c) > 0 := by
        exact Real.log_pos ( by nlinarith );
      unfold g_inner; aesop;

/-- Combined: if α₁ gives non-positive growth and α₂ gives positive growth,
    then gating (trading only when accuracy is α₂) strictly improves over ungated (α₁). -/
theorem gating_improves_growth
    (p_trade_all p_trade_gated α_all α_regime f c : ℝ)
    (hp_all : 0 < p_trade_all) (hp_gated : 0 < p_trade_gated)
    (hf : 1 < f) (hc : 0 < c) (hfc : f*c + c < 1)
    (h_all_neg : g_inner α_all f c ≤ 0)
    (h_regime_pos : 0 < g_inner α_regime f c) :
    G_growth (α_all * p_trade_all) ((1 - α_all) * p_trade_all) f c ≤ 0 ∧
    0 < G_growth (α_regime * p_trade_gated) ((1 - α_regime) * p_trade_gated) f c := by
  constructor
  · rw [G_growth_eq_ptrade_ginner]; exact mul_nonpos_of_nonneg_of_nonpos hp_all.le h_all_neg
  · rw [G_growth_eq_ptrade_ginner]; exact mul_pos hp_gated h_regime_pos

/-
PROBLEM
Monotonicity consequence: if α_all < α_regime and g_inner(α_all) ≤ 0,
    then any accuracy between them also has g ≤ 0 (by strict monotonicity,
    the threshold is unique).

PROVIDED SOLUTION
By g_inner_strictMono_alpha, g := fun α => g_inner α f c is strictly monotone. By g_inner_continuous_alpha, it is continuous. We have g(α₁) < 0 < g(α₂). By the intermediate value theorem, there exists α_star ∈ (α₁, α₂) with g(α_star) = 0. For uniqueness and the iff: since g is strictly monotone, g(β) > 0 iff g(β) > g(α_star) = 0 iff β > α_star.

For IVT, use intermediate_value_uIcc or the fact that the image of [α₁, α₂] under g is connected and contains g(α₁) < 0 and g(α₂) > 0, hence contains 0. The α_star is strictly between α₁ and α₂ because g(α₁) < 0 ≠ g(α_star) = 0 and g(α₂) > 0 ≠ 0.
-/
theorem g_inner_threshold_unique (f c : ℝ) (hf : 1 < f) (hc : 0 < c) (hfc : f*c + c < 1)
    (α₁ α₂ : ℝ) (h12 : α₁ < α₂) (h1 : g_inner α₁ f c < 0) (h2 : 0 < g_inner α₂ f c) :
    ∃ α_star, α₁ < α_star ∧ α_star < α₂ ∧
      g_inner α_star f c = 0 ∧
      ∀ β, g_inner β f c > 0 ↔ α_star < β := by
        -- By the intermediate value theorem, since $g_inner$ is continuous and strictly increasing, there exists $\alpha^* \in (α₁, α₂)$ such that $g_inner(\alpha^*) = 0$.
        have h_ivt : ∃ α_star ∈ Set.Ioo α₁ α₂, g_inner α_star f c = 0 := by
          apply_rules [ intermediate_value_Ioo ];
          · linarith;
          · exact Continuous.continuousOn ( g_inner_continuous_alpha f c );
          · aesop;
        obtain ⟨ α_star, hα_star₁, hα_star₂ ⟩ := h_ivt;
        -- Since $g_inner$ is strictly increasing, we have $g_inner(\beta) > g_inner(\alpha_star)$ if and only if $\beta > \alpha_star$.
        have h_strict_mono : StrictMono (fun α => g_inner α f c) := by
          exact g_inner_strictMono_alpha f c hf hc hfc;
        exact ⟨ α_star, hα_star₁.1, hα_star₁.2, hα_star₂, fun β => ⟨ fun hβ => h_strict_mono.lt_iff_lt.mp ( by linarith ), fun hβ => by linarith [ h_strict_mono hβ ] ⟩ ⟩

end