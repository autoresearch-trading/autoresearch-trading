/-
# Claim 2: Optimal Barrier Width — Growth Rate Product Characterization

Given win/loss probabilities and a fee rate, we prove the product characterization
of positive geometric growth and its equivalence to a win-rate bound.
-/
import Mathlib

open Real

section Claim2

variable {p_win p_loss c f : ℝ}

/-- The geometric growth rate at fee multiplier f with fee rate c. -/
noncomputable def growth_rate (p_win p_loss c f : ℝ) : ℝ :=
  p_win * Real.log (1 + f * c - c) + p_loss * Real.log (1 - f * c - c)

/-- The product form: (1 + f·c - c)^p_win · (1 - f·c - c)^p_loss -/
noncomputable def growth_product (p_win p_loss c f : ℝ) : ℝ :=
  (1 + f * c - c) ^ p_win * (1 - f * c - c) ^ p_loss

/-
PROBLEM
G > 0 if and only if the growth product > 1.
    This is the core product characterization of positive growth.

PROVIDED SOLUTION
growth_rate > 0 iff p_win * log(a) + p_loss * log(b) > 0, where a = 1+f*c-c > 0 and b = 1-f*c-c > 0. Since p_win, p_loss > 0, this equals log(a^p_win * b^p_loss) > 0 (using Real.log_rpow and Real.rpow_natCast or just Real.add_log_le... actually use Real.log_mul and Real.log_rpow). Then log(x) > 0 iff x > 1 for x > 0. The product a^p_win * b^p_loss is positive since a,b > 0 and we use Real.rpow. Actually growth_product uses (·)^(·) which is Real.rpow. Use Real.log_rpow to rewrite p_win * log(a) = log(a^p_win), similarly for p_loss. Then combine to log(product) > 0 iff product > 1 using Real.log_pos and Real.exp_log.
-/
theorem growth_pos_iff_product_gt_one
    (hp_win : p_win > 0) (hp_loss : p_loss > 0)
    (h_win_arg : 1 + f * c - c > 0) (h_loss_arg : 1 - f * c - c > 0) :
    growth_rate p_win p_loss c f > 0 ↔ growth_product p_win p_loss c f > 1 := by
  constructor <;> intro <;> simp_all +decide [ Real.rpow_def_of_nonneg ];
  · unfold growth_rate growth_product at *;
    rw [ Real.rpow_def_of_pos ( by linarith ), Real.rpow_def_of_pos ( by linarith ) ];
    rw [ ← Real.exp_add ] ; norm_num ; linarith;
  · unfold growth_product growth_rate at *;
    rw [ Real.rpow_def_of_pos ( by linarith ), Real.rpow_def_of_pos ( by linarith ) ] at *;
    rw [ ← Real.exp_add ] at * ; norm_num at * ; linarith

/-- The optimal fee multiplier formula -/
noncomputable def f_opt (p_win p_loss c : ℝ) : ℝ :=
  (p_win - p_loss) * (1 - c) / ((p_win + p_loss) * c)

/-
PROBLEM
The win-rate bound: positive growth requires p_win/(p_win+p_loss) to exceed a threshold.
    Specifically, G(f) > 0 requires:
      p_win / (p_win + p_loss) > c·f / (1 + c·f - c)
    This gives a necessary accuracy condition.

PROVIDED SOLUTION
We have G(f) = p_win * log(1+fc-c) + p_loss * log(1-fc-c) > 0. Since 0 < 1-fc-c < 1 (from h_loss_arg and h_fc which gives fc < 1-c so 1-fc-c > 0 and also 1-fc-c < 1), log(1-fc-c) < 0. And 1+fc-c > 1 (since f > 0 and c > 0 gives fc > 0, so 1+fc-c > 1-c > 0, and actually fc > 0 so 1+fc-c > 1 when fc > c, hmm need to be more careful). Actually, we need p_win * log(1+fc-c) > -p_loss * log(1-fc-c) = p_loss * log(1/(1-fc-c)). This implies p_win/(p_win+p_loss) > log(1/(1-fc-c))/(log(1+fc-c) + log(1/(1-fc-c))). The bound p_win/(p_win+p_loss) > cf/(1+cf-c) can be derived via Jensen's inequality or direct calculation. Actually, let's try a direct approach: if G > 0 then the weighted average of log(1+fc-c) and log(1-fc-c) with weights p_win and p_loss is positive. By concavity of log, this is at most log(p_win*(1+fc-c) + p_loss*(1-fc-c))/(p_win+p_loss). Hmm, this is getting complicated. Let me try a different approach.

Actually, let me reconsider. The claim is: G > 0 implies p_win/(p_win+p_loss) > cf/(1+cf-c). Note cf/(1+cf-c) = 1 - (1-c)/(1+cf-c). We want to show p_win/(p_win+p_loss) exceeds this. This might be a direct algebraic consequence. Let me think... if G > 0 then (1+fc-c)^p_win * (1-fc-c)^p_loss > 1. Taking the (p_win+p_loss)-th root and using AM-GM: p_win/(p_win+p_loss)*(1+fc-c) + p_loss/(p_win+p_loss)*(1-fc-c) >= ((1+fc-c)^p_win * (1-fc-c)^p_loss)^(1/(p_win+p_loss)) > 1. So p_win*(1+fc-c) + p_loss*(1-fc-c) > p_win+p_loss, which gives p_win*fc - p_win*c + p_loss*(-fc) - p_loss*c + (doesn't simplify). Wait: p_win*(1+fc-c) + p_loss*(1-fc-c) = (p_win+p_loss) + fc*(p_win-p_loss) - c*(p_win+p_loss). For this to be > p_win+p_loss we need fc*(p_win-p_loss) > c*(p_win+p_loss), i.e., f*(p_win-p_loss) > (p_win+p_loss), hmm that's not the right bound.

This bound might actually be hard or possibly not exactly right. Let me check with specific values. Let p_win=0.6, p_loss=0.4, c=0.01, f=20. Then cf=0.2, 1+cf-c = 1.19, 1-cf-c = 0.79. G = 0.6*log(1.19) + 0.4*log(0.79) = 0.6*0.1740 + 0.4*(-0.2357) = 0.1044 - 0.0943 = 0.0101 > 0. And p_win/(p_win+p_loss) = 0.6. cf/(1+cf-c) = 0.2/1.19 = 0.168. So 0.6 > 0.168. True.

Let me try a harder case. p_win=0.51, p_loss=0.49, c=0.1, f=1.1. cf=0.11, 1+cf-c = 1.01, 1-cf-c = 0.79. G = 0.51*log(1.01) + 0.49*log(0.79) = 0.51*0.00995 + 0.49*(-0.2357) = 0.00507 - 0.1155 = -0.110 < 0. Not applicable.

Let me try: p_win=0.55, p_loss=0.45, c=0.001, f=100. cf=0.1, 1+cf-c = 1.099, 1-cf-c = 0.899. G = 0.55*log(1.099) + 0.45*log(0.899) = 0.55*0.0944 + 0.45*(-0.1065) = 0.0519 - 0.0479 = 0.004 > 0. p_win/(p_win+p_loss) = 0.55. cf/(1+cf-c) = 0.1/1.099 = 0.091. 0.55 > 0.091. True.

The bound seems very loose. Maybe it's easier to prove than I thought. Actually, if G > 0 and 1-fc-c > 0 and 1+fc-c > 0, there might be a simpler path. Let me think about it differently.

If (1+fc-c)^p_win * (1-fc-c)^p_loss > 1, then since 1-fc-c < 1 (because fc > 0 as f > 0 and c > 0), (1-fc-c)^p_loss < 1. So (1+fc-c)^p_win > 1, meaning 1+fc-c > 1, so fc > c, so f > 1. Then... hmm.

Actually I think maybe the right approach is just to give it to the subagent and let it figure out the proof or find issues.
-/
theorem growth_pos_implies_winrate_bound
    (hp_win : p_win ≥ 0) (hp_loss : p_loss ≥ 0) (hsum : p_win + p_loss > 0)
    (hc : 0 < c) (hc1 : c < 1) (hf : f > 0)
    (h_win_arg : 1 + f * c - c > 0) (h_loss_arg : 1 - f * c - c > 0)
    (h_fc : f * c < 1 - c)
    (hG : growth_rate p_win p_loss c f > 0) :
    p_win / (p_win + p_loss) > (c * f) / (1 + c * f - c) := by
  -- By multiplying both sides of the inequality by $(p_win + p_loss)(1 + c * f - c)$, we can eliminate the denominators and simplify the expression.
  have h_mul : p_win * (1 + c * f - c) > c * f * (p_win + p_loss) := by
    -- Since $G(f) > 0$, we have $p_win * \log(1 + f * c - c) > -p_loss * \log(1 - f * c - c)$.
    have h_log_pos : p_win * Real.log (1 + f * c - c) > -p_loss * Real.log (1 - f * c - c) := by
      unfold growth_rate at hG; linarith;
    nlinarith [ Real.log_le_sub_one_of_pos h_win_arg, Real.log_le_sub_one_of_pos h_loss_arg ];
  rw [ gt_iff_lt, div_lt_div_iff₀ ] <;> linarith

end Claim2