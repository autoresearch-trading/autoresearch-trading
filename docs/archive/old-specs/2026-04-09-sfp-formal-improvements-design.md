# SFP Formal Improvements — Design Spec

**Date:** 2026-04-09
**Project:** Fork of `paradigmxyz/sfp` at `/Dev/non-toxic/sfp/`
**Goal:** Formalize tighter mathematical foundations for Sparse Feature Preservation using Aristotle (Lean 4), implement theory-derived algorithmic improvements, and submit on the SFP leaderboard.

## Background

SFP (Paradigm) prevents catastrophic forgetting in LLM continual learning by preserving activations in a PCA-identified subspace. Their Lean 4 formalization (17 theorems, 0 sorry) has significant gaps: the core hypothesis H1 is an axiom, the forgetting constant C is unformalized, and there are no optimizer dynamics, finite-sample, or multi-task theorems.

We have Aristotle (Lean 4 CLI) with 47+ theorems formalized for microstructure research, and domain experience in information theory / regression that transfers directly to closing these gaps.

## Theorem Roadmap (T1–T13)

### Tier 1 — Bounding C (first deliverable)

**T1: C as regression coefficient**
- Frame the forgetting constant as `C_ℓ = Cov(Δloss, ‖P_W(Δa_ℓ)‖) / Var(‖P_W(Δa_ℓ)‖)` per layer ℓ
- Transfer: identical structure to Kyle's Lambda (Cov(Δprice, flow) / Var(flow))
- Under sub-Gaussian activations with n memory-buffer samples, prove: `|Ĉ_ℓ - C_ℓ| ≤ O(√(log(1/δ)/n))` with probability 1-δ
- Numerical verification: for Qwen2.5-1.5B results (R²=0.933 at top-r), verify the bound is non-vacuous
- Algorithmic output: select probe layers by smallest Ĉ_ℓ (tightest bound) instead of fixed [0.25, 0.5, 0.75]

**T2: Finite-sample W estimation (Davis-Kahan)**
- If eigenvalue gap Δ_k > 0 and ‖Σ̂ - Σ‖ ≤ ε, then sin(θ(Ŵ, W)) ≤ ε/Δ_k
- Required buffer size: n ≥ O(d·log(d) / Δ_k²) for ε/Δ_k < threshold
- Numerical verification: default n=128 with hidden_dim=1536 — is that sufficient?
- Algorithmic output: principled memory buffer sizing

### Tier 2 — Strengthening H1

**T3: Conditional H1 under exponential family**
- If layer activations follow exponential family distribution, P_W(a) is sufficient statistic for old-task loss
- H1 follows as theorem (not axiom) under this distributional assumption
- Algorithmic output: diagnostic test for when H1 is expected to hold

**T4: Ablation importance = causal importance**
- Under monotonicity (removing a direction can only hurt) and independence (directions contribute additively), ablation ranking recovers true causal ordering
- Algorithmic output: conditions to check before trusting ablation-based top-r selection

**T5: PCA vs gradient-SVD optimality**
- PCA is optimal when forgetting is variance-dominated (high activation variance directions carry task info)
- Gradient-SVD is optimal when forgetting is gradient-dominated (loss is sensitive in low-variance directions)
- Formal decision boundary: compare trace(Σ_topk) / trace(Σ) vs ‖∇L_topk‖ / ‖∇L‖
- Algorithmic output: automatic basis selection criterion

### Tier 3 — Optimizer Dynamics

**T6: Adaptive λ from signal decay**
- Under L2 regularization, preserved-subspace drift follows exponential decay
- Optimal schedule: λ(t) = λ₀ · exp(αt) where α depends on learning rate and subspace curvature
- Transfer: directly from T47 (optimal window under exponential signal decay) in autoresearch-trading
- Algorithmic output: λ schedule replacing static λ=0.1

**T7: SGD convergence to small δ**
- For L-smooth, μ-strongly-convex L_new with SFP regularization (λ · L_pres), after K steps with step size η:
  δ ≤ (1 - ημ)^K · δ₀ + (ηλ / μ) · ‖∇L_pres‖²
- Algorithmic output: minimum training steps for target δ

**T8: Bound tightness (lower bound)**
- Construct adversarial example showing forgetting = Ω(√δ) — matching the upper bound
- Shows the √δ scaling in `sfp_reduces_forgetting` is tight
- Algorithmic output: none (theoretical completeness)

### Tier 4 — Extensions

**T9: Multi-task cumulative forgetting**
- Bound total forgetting over K sequential tasks: forgetting_total ≤ Σ_k C_k · √δ_k
- Task ordering matters: optimal order minimizes max(C_k) across transitions
- Algorithmic output: task ordering heuristic

**T10: Multi-layer optimal weighting**
- Optimal λ_ℓ ∝ 1/C_ℓ (invest more regularization where forgetting-to-drift ratio is highest)
- Contrast with current uniform sum
- Algorithmic output: per-layer λ weights

**T11: Projection information loss**
- Rate-distortion bound: I(task; P_W(a)) ≥ I(task; a) - r · log(σ_max/σ_{r+1})
- Quantifies how much task information the top-r projection preserves
- Algorithmic output: minimum r for target information retention

**T12: LoRA confound conditions**
- When LoRA rank r_lora ≤ r_sfp, all weight updates are confined to a subspace of dimension ≤ r_lora
- H1 becomes trivially true (drift is low-dimensional by construction, not by learned structure)
- Full fine-tuning experiments required to validate H1 is genuine
- Algorithmic output: diagnostic flag when LoRA makes SFP results uninterpretable

**T13: Harmonic mean scorer properties**
- Harmonic mean is the unique scoring function (up to monotone transform) where: score=0 iff either retention=0 or plasticity=0, and equal weighting at retention=plasticity
- Game-theoretic: equivalent to Nash bargaining solution between stability and plasticity agents
- Algorithmic output: none (evaluation methodology justification)

## First Deliverable — Information-Theoretic SFP

### Loss function: `information_theoretic_sfp_loss`

Changes from upstream `sfp_loss`:
1. **Layer selection by Ĉ_ℓ** — at setup, estimate C per candidate layer using memory buffer (one forward pass per sample, compute per-layer forgetting-drift regression). Select top-3 layers by smallest C (tightest forgetting bound).
2. **Buffer size by Davis-Kahan** — compute eigenvalue gap from PCA, set n = ceil(d·log(d)/Δ_k²) clamped to [128, 1024].
3. Everything else unchanged: same PCA basis, same ablation importance, same preservation loss formula.

### Setup function: `information_theoretic_sfp_setup`

Extended from `sfp_setup`:
1. Run standard activation collection on memory buffer
2. For each candidate layer (all transformer layers, not just 3 fixed):
   a. Compute PCA basis
   b. Estimate Ĉ_ℓ via OLS on (‖P_W(Δa_ℓ)‖, Δloss) pairs from checkpoint pairs
   c. Compute eigenvalue gap Δ_k for Davis-Kahan buffer bound
3. Select top-3 layers by smallest Ĉ_ℓ
4. Adjust buffer size if Davis-Kahan says 128 is insufficient
5. Proceed with standard SFP (PCA + ablation importance + preservation loss)

Must complete within 120-second wall-clock budget.

### Submission

```python
# submissions/information_theoretic_sfp.py
CONTRIBUTOR = "0xQuinto"
SETUP = "information_theoretic_sfp"

def information_theoretic_sfp_loss(model, batch, **kw) -> Tensor:
    ...
```

### Success criterion

Beat `sfp` baseline on harmonic mean score (2 × retention × plasticity / (retention + plasticity)). Even a small improvement is meaningful — it comes with a proof certificate explaining why.

## Project Structure

```
/Dev/non-toxic/sfp/                    ← fork of paradigmxyz/sfp
├── lean-sfp/                          ← upstream's 17 theorems (untouched)
├── lean-proofs/
│   ├── inputs/                        ← Aristotle input files (T1-T13)
│   ├── results/                       ← Aristotle output (Lean 4 source)
│   └── ROADMAP.md                     ← theorem status tracker
├── sfp/                               ← upstream Python code
├── submissions/
│   └── information_theoretic_sfp.py   ← our loss function
├── scripts/
│   └── estimate_C.py                  ← empirical C estimation tooling
├── .claude/
│   ├── agents/
│   │   └── prover.md                  ← prover agent for this project
│   └── skills/                        ← autoresearch for benchmark iteration
├── CLAUDE.md
└── MEMORY.md
```

## Branch Strategy

- `main` tracks upstream `paradigmxyz/sfp` (set as remote `upstream`)
- `formal-improvements` is our working branch
- One upstream PR per tier

## Workflow

For each theorem:
1. Write `lean-proofs/inputs/TN-name.txt`
2. `aristotle formalize lean-proofs/inputs/TN-name.txt` → UUID
3. `aristotle list` → wait for COMPLETE (~30-60 min)
4. `aristotle result <UUID> --destination lean-proofs/results/TN-name.tar.gz`
5. Extract, verify 0 sorry
6. Implement algorithmic change in Python
7. Run benchmark locally or via submission
8. Commit, update ROADMAP.md

## Dependencies

- Python 3.12+, PyTorch, transformers, peft (from upstream SFP)
- Aristotle CLI (`/Users/diego/.local/bin/aristotle`)
- GPU for benchmarks (SmolLM2-135M runs on CPU, Qwen2.5-1.5B needs GPU)
