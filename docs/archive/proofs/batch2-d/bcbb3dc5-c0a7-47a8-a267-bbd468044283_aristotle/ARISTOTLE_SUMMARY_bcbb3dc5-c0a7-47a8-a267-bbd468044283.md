# Summary of changes
Successfully formalized and proved all four claims about Portfolio Diversification with Correlated Strategies in Lean 4 with Mathlib. The file is at `RequestProject/PortfolioDiversification.lean`.

## Definitions
- **`denominator N ρ`** = 1 + (N-1)·ρ
- **`portfolioVar N σ ρ`** = σ²·(1+(N-1)ρ)/N  
- **`innerRatio N ρ`** = N / (1+(N-1)ρ)
- **`divRatio N ρ`** = √(N/(1+(N-1)ρ)) — the diversification ratio

## Proved Theorems

### Claim 1 (Independence, ρ=0)
- `claim1_independent_variance`: portfolioVar N σ 0 = σ²/N
- `claim1_sortino_scaling`: divRatio N 0 = √N (√N scaling of Sortino)

### Claim 2 (Correlated portfolio special cases)
- `claim2a_rho_zero`: ρ=0 gives √N (matches Claim 1)
- `claim2b_rho_one`: ρ=1 gives 1 (no diversification benefit)
- `claim2c_specific`: N=25, ρ=0.3 gives √(125/41) ≈ 1.746

### Claim 3 (Diversification ratio properties)
- `claim3a_increasing_in_N`: DR is strictly increasing in N for ρ ∈ [0,1)
- `claim3b_decreasing_in_rho`: DR is strictly decreasing in ρ for N > 1
- `claim3c_limit`: DR → 1/√ρ as N → ∞ (asymptotic bound)
- `claim3d_max_value`: 1/√0.3 = √(10/3) ≈ 1.826
- `claim3d_specific`: N=25, ρ=0.3 gives √(125/41) (95.6% of theoretical max)

### Claim 4 (Diminishing marginal returns)
- `claim4_diminishing_returns`: ΔDR = DR(N+1) - DR(N) → 0 as N → ∞

All proofs compile without `sorry`, using only standard axioms (propext, Classical.choice, Quot.sound).