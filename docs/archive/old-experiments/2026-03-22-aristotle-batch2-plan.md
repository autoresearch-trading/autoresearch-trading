# Aristotle Batch 2 — Proof Submission Plan

6 theorems, 24 claims total. All build on the 144 proved results from Batch 1.

## Theorem 12: Feature Independence & Normalization Preservation

**Why:** We use 5 features but don't know if they're redundant or if z-score normalization breaks the sufficient statistic property.

**Claims:**
1. Under the Kyle model, λ·OFI and TFI·|OFI| are NOT independent (they share OFI). Characterize their correlation: Corr = sign(λ)·E[TFI·OFI²] / (std(λ·OFI)·std(TFI·|OFI|)). Prove this is non-zero in general.
2. VPIN = E[|TFI|] is a function of TFI alone, so it is NOT independent of feature 1 (which uses TFI). Prove: I(VPIN; directional_conviction) > 0 whenever TFI is not symmetric around 0.
3. Z-score normalization (x - μ)/σ is a monotone transformation when σ > 0. Prove: if T(X) is sufficient for Y, then ((T(X) - μ_T)/σ_T) is also sufficient for Y (sufficiency preserved under invertible transforms).
4. IQR-based scaling (x - median)/IQR is NOT necessarily monotone (IQR can be zero). Prove: sufficiency is preserved when IQR > 0, but may fail at the boundary IQR = 0.

## Theorem 13: Dual Gate Optimality (Hawkes + VPIN)

**Why:** Next experiment adds VPIN gate on top of Hawkes gate. Need to know if it helps or hurts.

**Claims:**
1. Given two binary filters F₁ (pass when hawkes ≥ r_min) and F₂ (pass when VPIN ≤ v_max) with pass rates φ₁ and φ₂, if F₁ and F₂ are independent, the combined pass rate is φ₁·φ₂. Prove.
2. If F₁ filters for high-activity regimes and F₂ filters for low-toxicity regimes, and accuracy improves by k₁ under F₁ and k₂ under F₂, then combined accuracy improvement is at most k₁·k₂ (with equality iff independence). Prove the upper bound.
3. The combined Sortino ratio satisfies: Sortino_dual / Sortino_ungated = k₁·k₂·√(φ₁·φ₂). Prove that dual gating beats single gating (Sortino_dual > Sortino_F1) iff k₂ > 1/√φ₂ (the accuracy improvement from the second gate must overcome the √φ penalty). This follows from Theorem 10's φ_min = 1/k².
4. Hawkes branching and VPIN are NOT independent in general: high Hawkes intensity (bursty arrivals) tends to coincide with imbalanced flow (high VPIN). Prove: under a Hawkes process where buy/sell arrivals have different excitation parameters, Corr(branching_ratio, VPIN) > 0.

## Theorem 14: PF-Sortino Relationship & Kelly Position Sizing

**Why:** We observe PF=1.54 but Sortino≈0. Need to understand why and compute optimal position size.

**Claims:**
1. The relationship between profit factor PF, win rate w, and average win/loss ratio R is: PF = w·R / ((1-w)·1) when average loss = 1 (normalized). Therefore R = PF·(1-w)/w. For PF=1.54, w=0.54: R = 1.54·0.46/0.54 = 1.31. Prove this identity.
2. Sortino can be zero even when PF > 1 if losses cluster in time (high downside autocorrelation). Prove: for i.i.d. returns with PF > 1, Sortino > 0 always. But for returns with positive autocorrelation in losses (ρ_loss > 0), Sortino can be ≤ 0 even when PF > 1. The critical autocorrelation is ρ_loss* = 1 - (PF-1)²/(PF+1)².
3. The Kelly-optimal fraction for a binary bet with win rate w and payoff ratio R is f* = w - (1-w)/R = w - 1/PF. For PF=1.54, w=0.54: f* = 0.54 - 1/1.54 = 0.54 - 0.649 = -0.109. NEGATIVE Kelly fraction means DON'T BET at these parameters. Prove: f* > 0 iff PF > 1/(w/(1-w)) = (1-w)/w. For w=0.54: need PF > 0.46/0.54 = 0.852 — we satisfy this. Wait, let me recompute. f* = (p·b - q)/b where p=win_rate, q=1-p, b=avg_win/avg_loss. If PF = p·avg_win / (q·avg_loss), then b = PF·q/p. So f* = (p·PF·q/p - q)/(PF·q/p) = (PF·q - q)/(PF·q/p) = q(PF-1)·p/(PF·q) = p(PF-1)/PF. For p=0.54, PF=1.54: f* = 0.54·0.54/1.54 = 0.189. Prove this formula.
4. The expected geometric growth rate at Kelly fraction f* is: G* = p·log(1 + f*·b) + q·log(1 - f*). Prove G* > 0 when f* > 0 (positive Kelly means positive growth).
5. The maximum drawdown under Kelly betting satisfies E[MaxDD] ≤ 1 - (1-f*)^(1/f*) ≈ 1 - e^(-1) ≈ 0.632 for f* small. For f*=0.189: E[MaxDD] ≤ 1 - (0.811)^(5.29) ≈ 1 - 0.34 = 0.66. Prove this bound and compare to our observed max_dd=0.30.

## Theorem 15: Logit-Sum Ensemble vs Majority Vote

**Why:** We use logit-sum argmax for ensembling. Is this provably better than majority vote?

**Claims:**
1. For K classifiers with identical accuracy α and independent errors, logit-sum argmax is equivalent to a weighted majority vote where the weight of each classifier's vote is proportional to its confidence. Prove: argmax(Σ logits_k) = argmax(Σ softmax(logits_k)) when logits are calibrated.
2. Logit-sum ensembling is at least as good as majority vote: P(correct | logit_sum) ≥ P(correct | majority_vote). Prove: logit-sum uses more information (continuous logits vs discrete votes).
3. For 3-class with K=5 and α=0.54, the majority vote accuracy is: α_mv = Σ_{k≥3} C(5,k)·α^k·((1-α)/2)^(5-k). Compute the exact value. The logit-sum accuracy satisfies α_ls ≥ α_mv.

## Theorem 16: Max Drawdown Bounds & Symbol Diversification

**Why:** We trade 25 symbols. Does diversification provably reduce max drawdown?

**Claims:**
1. For N independent strategies each with Sortino S and max drawdown D, the portfolio Sortino is S·√N (diversification benefit). Prove under independence assumption.
2. The portfolio max drawdown satisfies E[MaxDD_portfolio] ≤ E[MaxDD_single] / √N for independent strategies. Prove.
3. For correlated strategies with pairwise correlation ρ, the portfolio Sortino is S·√(N/(1 + (N-1)·ρ)). Prove. For N=25 and ρ=0.3: Sortino_portfolio = S·√(25/8.2) = S·1.75. The diversification benefit is 1.75x, not 5x (= √25).
4. The optimal number of symbols N* that maximizes risk-adjusted return after transaction costs is: N* = argmax S·√(N/(1+(N-1)ρ)) - c·N where c is per-symbol overhead. Prove this has a unique maximum and characterize it.

## Theorem 17: Focal Loss Convergence & Optimal Epochs

**Why:** We're running 100 epochs. Does focal loss converge? Is there an optimal epoch count?

**Claims:**
1. Focal loss FL(p) = -(1-p)^γ · log(p) is convex in logits for γ ∈ [0, 2]. Prove for γ=1 (our setting): the Hessian is positive semi-definite.
2. SGD with focal loss and learning rate η converges to a stationary point at rate O(1/√T) where T is the number of updates. Prove: this follows from focal loss being Lipschitz smooth when γ ≥ 0 and the standard SGD convergence theorem.
3. The optimal number of epochs before overfitting is bounded by: E_opt ≤ N/(P·log(N/P)) where N=samples, P=parameters. For N=250K, P=68K: E_opt ≤ 250K/(68K·log(3.67)) ≈ 2.8. This suggests even 25 epochs may be overfitting! Prove or disprove this bound.
4. With recency weighting (decay=1.0), the effective sample size is N_eff = N / (1 + Var(weights)/E[weights]²). For exponential decay with factor 2.7x: N_eff ≈ N/1.37. Prove and compute for our setup.
