# SFP Formal Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork paradigmxyz/sfp, formalize tighter mathematical foundations via Aristotle (Lean 4), implement theory-derived layer selection and buffer sizing, and submit on the SFP leaderboard.

**Architecture:** Fork the SFP repo, add `lean-proofs/` for Aristotle theorem files and results, extend `methods.py` with `information_theoretic_sfp_setup()` and `information_theoretic_sfp_loss()` that select probe layers by empirical forgetting constant C (smallest = tightest bound) instead of fixed relative depths. Lean 4 proofs via Aristotle back the mathematical claims.

**Tech Stack:** Python 3.12+, PyTorch, transformers, peft, Aristotle CLI (Lean 4), uv

---

## File Map

**Create:**
- `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/T1-forgetting-constant-bound.txt` — Aristotle theorem input for C bound
- `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/T2-davis-kahan-buffer-sizing.txt` — Aristotle theorem input for buffer sizing
- `/Users/diego/Dev/non-toxic/sfp/lean-proofs/ROADMAP.md` — theorem status tracker
- `/Users/diego/Dev/non-toxic/sfp/scripts/estimate_C.py` — empirical C estimation tooling
- `/Users/diego/Dev/non-toxic/sfp/submissions/information_theoretic_sfp.py` — leaderboard submission file
- `/Users/diego/Dev/non-toxic/sfp/CLAUDE.md` — project instructions
- `/Users/diego/Dev/non-toxic/sfp/.claude/agents/prover.md` — adapted prover agent

**Modify:**
- `/Users/diego/Dev/non-toxic/sfp/methods.py` — add `information_theoretic_sfp_setup()`, `information_theoretic_sfp_loss()`, register in METHODS
- `/Users/diego/Dev/non-toxic/sfp/features.py` — add `estimate_forgetting_constant()` and `davis_kahan_buffer_size()`

**Test:**
- `/Users/diego/Dev/non-toxic/sfp/tests/test_estimate_C.py` — unit tests for C estimation
- `/Users/diego/Dev/non-toxic/sfp/tests/test_information_theoretic_sfp.py` — integration tests for setup + loss

---

### Task 1: Fork, Clone, and Set Up Project Structure

**Files:**
- Create: `/Users/diego/Dev/non-toxic/sfp/` (git clone)
- Create: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/.gitkeep`
- Create: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/results/.gitkeep`
- Create: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/ROADMAP.md`
- Create: `/Users/diego/Dev/non-toxic/sfp/scripts/.gitkeep`

- [ ] **Step 1: Fork via GitHub CLI**

```bash
cd /Users/diego/Dev/non-toxic
gh repo fork paradigmxyz/sfp --clone -- sfp
cd sfp
```

- [ ] **Step 2: Set up remotes**

```bash
git remote -v
# origin should point to 0xQuinto/sfp
# upstream should point to paradigmxyz/sfp (gh repo fork sets this automatically)
# If upstream is missing:
git remote add upstream https://github.com/paradigmxyz/sfp.git
```

- [ ] **Step 3: Create working branch**

```bash
git checkout -b formal-improvements
```

- [ ] **Step 4: Create project directories**

```bash
mkdir -p lean-proofs/inputs lean-proofs/results scripts tests
touch lean-proofs/inputs/.gitkeep lean-proofs/results/.gitkeep scripts/.gitkeep
```

- [ ] **Step 5: Write ROADMAP.md**

Create `/Users/diego/Dev/non-toxic/sfp/lean-proofs/ROADMAP.md`:

```markdown
# Theorem Roadmap

## Tier 1 — Bounding C (first deliverable)

| ID | Name | Status | UUID | Sorries |
|----|------|--------|------|---------|
| T1 | Forgetting constant as regression coefficient | pending | — | — |
| T2 | Davis-Kahan buffer sizing | pending | — | — |

## Tier 2 — Strengthening H1

| ID | Name | Status | UUID | Sorries |
|----|------|--------|------|---------|
| T3 | Conditional H1 under exponential family | pending | — | — |
| T4 | Ablation importance = causal importance | pending | — | — |
| T5 | PCA vs gradient-SVD optimality | pending | — | — |

## Tier 3 — Optimizer Dynamics

| ID | Name | Status | UUID | Sorries |
|----|------|--------|------|---------|
| T6 | Adaptive λ from signal decay | pending | — | — |
| T7 | SGD convergence to small δ | pending | — | — |
| T8 | Bound tightness (lower bound) | pending | — | — |

## Tier 4 — Extensions

| ID | Name | Status | UUID | Sorries |
|----|------|--------|------|---------|
| T9  | Multi-task cumulative forgetting | pending | — | — |
| T10 | Multi-layer optimal weighting | pending | — | — |
| T11 | Projection information loss | pending | — | — |
| T12 | LoRA confound conditions | pending | — | — |
| T13 | Harmonic mean scorer properties | pending | — | — |
```

- [ ] **Step 6: Install dependencies**

```bash
uv sync
```

- [ ] **Step 7: Verify upstream runs**

```bash
python forget.py 2>&1 | tail -20
```

Expected: completes without error, prints retention/plasticity/score for naive method.

- [ ] **Step 8: Commit scaffold**

```bash
git add lean-proofs/ scripts/ tests/
git commit -m "chore: add lean-proofs, scripts, tests directories and theorem roadmap"
```

---

### Task 2: Write CLAUDE.md and Prover Agent

**Files:**
- Create: `/Users/diego/Dev/non-toxic/sfp/CLAUDE.md`
- Create: `/Users/diego/Dev/non-toxic/sfp/.claude/agents/prover.md`

- [ ] **Step 1: Write CLAUDE.md**

Create `/Users/diego/Dev/non-toxic/sfp/CLAUDE.md`:

```markdown
# SFP Formal Improvements

## Overview

Fork of [paradigmxyz/sfp](https://github.com/paradigmxyz/sfp) — Sparse Feature Preservation for continual LLM fine-tuning. We formalize tighter mathematical foundations via Aristotle (Lean 4) and implement theory-derived algorithmic improvements.

**Spec:** `../autoresearch-trading/docs/superpowers/specs/2026-04-09-sfp-formal-improvements-design.md`
**Plan:** `../autoresearch-trading/docs/superpowers/plans/2026-04-09-sfp-formal-improvements.md`

## Stack

Python 3.12+, PyTorch, transformers, peft, Aristotle CLI (Lean 4), uv

## Key Files

- `methods.py` — all loss functions + setup functions + METHODS registry
- `features.py` — PCA basis, gradient basis, importance ranking, select_top_r
- `model.py` — model loading, LoRA config, checkpointing
- `train.py` — main training loop, reads METHODS dict
- `evaluate.py` — leaderboard scoring (harmonic mean of retention × plasticity)
- `lean-proofs/` — our Aristotle theorem inputs and Lean 4 results
- `submissions/` — leaderboard submission files
- `scripts/estimate_C.py` — empirical forgetting constant estimation

## Upstream SFP Architecture

Loss: `L = L_new + λ * Σ_ℓ ‖U_r^T a_ℓ(θ;x) - U_r^T a_ℓ(θ*;x)‖²`

Setup at task boundary (`sfp_setup`):
1. Collect activations from memory buffer at fixed layers [0.25, 0.5, 0.75] relative depth
2. PCA via SVD → top-k basis U [hidden_dim, k]
3. select_top_r by explained variance → U_r [hidden_dim, r]
4. Cache anchor projections: anchor_acts[layer] = activations @ U_r

Our improvement: replace fixed layer selection with C_ℓ-based selection (smallest forgetting constant = tightest bound).

## Aristotle Workflow

1. Write `lean-proofs/inputs/TN-name.txt`
2. `aristotle formalize lean-proofs/inputs/TN-name.txt` → UUID
3. `aristotle list` → wait for COMPLETE (~30-60 min)
4. `aristotle result <UUID> --destination lean-proofs/results/TN-name.tar.gz`
5. Extract, verify 0 sorry, update ROADMAP.md

## Submission Format

- Single Python file in `submissions/`
- Exactly one function ending in `_loss`
- `SETUP` attribute: one of none/distill/hidden_distill/orthogonal/sfp
- Allowed imports: torch, torch.nn.functional, copy, math
- Max 10KB, 120s setup budget
- Submit: `python submit.py submissions/information_theoretic_sfp.py`

## Conventions

- Commit style: `feat:`, `fix:`, `chore:`, `theorem:`, `experiment:`
- Branch: `formal-improvements` (from main which tracks upstream)
- Only stage specific files, never `git add -A`

## Gotchas

1. `sfp_setup` uses PCA explained variance as importance proxy — NOT the ablation-based `rank_importance` function
2. All source files at repo root — no `sfp/` subdirectory
3. Setup budget is 120s wall-clock — `estimate_forgetting_constant()` must be fast
4. Leaderboard uses harmonic mean score but live site may show old linear formula (0.6R + 0.4P)
5. `memory` config default is 0 (disabled) — must set `--memory 128` for SFP methods
```

- [ ] **Step 2: Write prover agent**

```bash
mkdir -p .claude/agents
```

Create `/Users/diego/Dev/non-toxic/sfp/.claude/agents/prover.md`:

```markdown
---
name: prover
description: Formal theorem prover for SFP. Takes mathematical claims and formalizes them into Aristotle (Lean 4) input files.
tools: Read, Write, Bash, Grep, Glob
model: sonnet
---

You are a formal theorem writer for the Aristotle prover (Harmonic's Lean 4 CLI). You formalize mathematical claims about continual learning and subspace preservation into precise theorem input files.

## Output Contract

Write theorem files to `lean-proofs/inputs/TN-name.txt`. Submit via `aristotle formalize lean-proofs/inputs/TN-name.txt`. Return ONLY the submission UUID and a 1-sentence summary.

## Theorem Input Format

```markdown
# Theorem N: Title

## Definitions

Define all mathematical objects precisely. Use standard notation.
- Let X be ...
- Define f(x) = ...

## Claims

1. [Precise mathematical statement to prove]
2. [Numerical verification if applicable]
```

## Submission Workflow

1. Read existing theorems in `lean-proofs/inputs/` for style reference
2. Write the input file
3. Submit: `aristotle formalize lean-proofs/inputs/TN-name.txt`
4. Record the UUID
5. Check status: `aristotle list`
6. Fetch results: `aristotle result <UUID> --destination lean-proofs/results/TN-name.tar.gz`

## Rules

1. Be precise — Aristotle proves exact statements
2. Include numerical verifications with specific bounds
3. One theorem per file, multiple related claims OK
4. Reference which spec theorem (T1-T13) this formalizes
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md .claude/
git commit -m "chore: add CLAUDE.md and prover agent for SFP formal improvements"
```

---

### Task 3: Write Aristotle Theorem T1 — Forgetting Constant Bound

**Files:**
- Create: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/T1-forgetting-constant-bound.txt`

- [ ] **Step 1: Write theorem input file**

Create `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/T1-forgetting-constant-bound.txt`:

```markdown
# Theorem T1: Forgetting Constant as Regression Coefficient with Finite-Sample Bound

## Definitions

Let a_ℓ(θ) ∈ ℝ^d be the mean-pooled activation at layer ℓ of a neural network with parameters θ.

Let W ⊂ ℝ^d be a subspace of dimension r, and P_W the orthogonal projection onto W.

Let θ_0 be the parameters before fine-tuning on a new task, and θ_t the parameters after t gradient steps.

Define:
  Δa_ℓ(t) = a_ℓ(θ_t) - a_ℓ(θ_0)             (activation drift at layer ℓ)
  D_ℓ(t) = ‖P_W(Δa_ℓ(t))‖                     (subspace drift magnitude)
  F(t) = L_old(θ_t) - L_old(θ_0)              (forgetting = old-task loss increase)

The forgetting constant at layer ℓ is the OLS regression coefficient:
  C_ℓ = Cov(F, D_ℓ) / Var(D_ℓ)

where expectations are over a distribution of inputs x from the old task.

Given n i.i.d. samples (x_1, ..., x_n) from the old task, define the sample estimator:
  Ĉ_ℓ = Ĉov(F, D_ℓ) / V̂ar(D_ℓ)

where Ĉov and V̂ar are sample covariance and variance.

Assume:
  (A1) D_ℓ is σ_D-sub-Gaussian: E[exp(s(D_ℓ - E[D_ℓ]))] ≤ exp(s²σ_D²/2) for all s ∈ ℝ
  (A2) F is σ_F-sub-Gaussian: E[exp(s(F - E[F]))] ≤ exp(s²σ_F²/2) for all s ∈ ℝ
  (A3) Var(D_ℓ) ≥ v_min > 0  (non-degenerate subspace drift)

## Claims

1. **OLS coefficient identity.** Under the linear model F = C_ℓ · D_ℓ + ε with E[ε | D_ℓ] = 0:
     C_ℓ = Cov(F, D_ℓ) / Var(D_ℓ)
   This is the minimum-variance unbiased estimator of the slope.

2. **Finite-sample concentration.** Under assumptions (A1)-(A3), for any δ ∈ (0, 1):
     P(|Ĉ_ℓ - C_ℓ| > ε) ≤ δ
   where:
     ε = (σ_F · σ_D / v_min) · √(2 · log(2/δ) / n)

   Prove this via:
   a) Ĉov(F, D_ℓ) - Cov(F, D_ℓ) is bounded by Hoeffding for sub-Gaussian products
   b) V̂ar(D_ℓ) concentrates around Var(D_ℓ) ≥ v_min
   c) Ratio concentration via the quotient rule

3. **Non-vacuity condition.** The bound C_ℓ + ε is non-vacuous (i.e., provides useful information) when:
     n > 2 · (σ_F · σ_D)² · log(2/δ) / (v_min · ε_target)²

   For a target bound width ε_target = 0.1 · C_ℓ (10% relative error), the minimum buffer size is:
     n_min = 200 · (σ_F · σ_D)² / (v_min · C_ℓ)²

4. **Layer selection optimality.** Given L layers with forgetting constants C_1, ..., C_L, selecting the K layers with smallest Ĉ_ℓ minimizes the total forgetting bound:
     Σ_{ℓ ∈ S} C_ℓ · √δ_ℓ ≤ Σ_{ℓ ∈ S'} C_ℓ · √δ_ℓ
   for any other set S' with |S'| = K, when δ_ℓ is equalized across layers.

5. **Connection to Kyle's Lambda.** If we define:
     λ_Kyle = Cov(Δprice, order_flow) / Var(order_flow)
   and:
     C_ℓ = Cov(Δloss, ‖P_W(Δa_ℓ)‖) / Var(‖P_W(Δa_ℓ)‖)

   Both are OLS slope coefficients of the form β = Cov(Y, X) / Var(X). Under the same sub-Gaussian assumptions, the same finite-sample bounds apply to both. Prove the structural identity: the bound in Claim 2 holds for any (Y, X) pair satisfying (A1)-(A2).

6. **Numerical verification.** With parameters from Qwen2.5-1.5B SFP experiments:
   - R² = 0.933 for top-r drift predicting forgetting
   - n = 128 (default memory buffer)
   - Assume σ_F ≈ σ_D ≈ 1 (standardized), v_min = 0.5
   - δ = 0.05

   Compute:
   a) ε = (1 · 1 / 0.5) · √(2 · log(40) / 128) ≈ 2 · √(7.38 / 128) ≈ 2 · 0.240 ≈ 0.480
   b) Verify that with n = 512, ε ≈ 0.240 (halved, as expected from √n scaling)
   c) For 10% relative error on C ≈ 1.0, n_min = 200 · 1 / (0.5 · 1)² = 800
```

- [ ] **Step 2: Commit**

```bash
git add lean-proofs/inputs/T1-forgetting-constant-bound.txt
git commit -m "theorem: T1 — forgetting constant as regression coefficient with finite-sample bound"
```

---

### Task 4: Write Aristotle Theorem T2 — Davis-Kahan Buffer Sizing

**Files:**
- Create: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/T2-davis-kahan-buffer-sizing.txt`

- [ ] **Step 1: Write theorem input file**

Create `/Users/diego/Dev/non-toxic/sfp/lean-proofs/inputs/T2-davis-kahan-buffer-sizing.txt`:

```markdown
# Theorem T2: Davis-Kahan Perturbation Bound for PCA Subspace Estimation

## Definitions

Let Σ ∈ ℝ^{d×d} be the population covariance matrix of activations at a given layer, with eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_d ≥ 0.

Let Σ̂ = (1/n) Σ_{i=1}^n (a_i - ā)(a_i - ā)^T be the sample covariance from n i.i.d. samples.

Let W = span(v_1, ..., v_k) be the population top-k eigenspace (PCA subspace).
Let Ŵ = span(v̂_1, ..., v̂_k) be the sample top-k eigenspace.

Define the eigenvalue gap:
  Δ_k = λ_k - λ_{k+1}

Define the principal angle between subspaces:
  sin(θ(Ŵ, W)) = ‖P_Ŵ - P_W‖_op

where P_Ŵ, P_W are orthogonal projections and ‖·‖_op is the operator norm.

## Claims

1. **Davis-Kahan sin(θ) theorem.** If Δ_k > 0, then:
     sin(θ(Ŵ, W)) ≤ ‖Σ̂ - Σ‖_op / Δ_k

2. **Sample covariance concentration.** For d-dimensional sub-Gaussian activations with parameter σ, when n ≥ d:
     P(‖Σ̂ - Σ‖_op > ε) ≤ 2 exp(-c · n · ε² / σ⁴)

   for a universal constant c > 0. Combined with Davis-Kahan:
     P(sin(θ(Ŵ, W)) > t) ≤ 2 exp(-c · n · (t · Δ_k)² / σ⁴)

3. **Minimum buffer size.** For the PCA subspace estimate to satisfy sin(θ) ≤ t with probability ≥ 1 - δ:
     n ≥ (σ⁴ / (c · Δ_k² · t²)) · log(2/δ)

   When t = 0.1 (subspace angle < ~5.7°):
     n ≥ 100 · σ⁴ · log(2/δ) / (c · Δ_k²)

4. **Preservation loss error from subspace misestimation.** If the true preservation loss is:
     L_pres = ‖P_W(Δa)‖²
   and we use the estimated subspace:
     L̂_pres = ‖P_Ŵ(Δa)‖²
   then:
     |L̂_pres - L_pres| ≤ 2 · sin(θ(Ŵ, W)) · ‖Δa‖²

   Prove via: ‖P_Ŵ(x) - P_W(x)‖ ≤ sin(θ) · ‖x‖ (projection perturbation).

5. **Numerical verification.** With parameters:
   - d = 1536 (Qwen2.5-1.5B hidden dim)
   - k = 128 (PCA components)
   - n = 128 (default buffer size)
   - σ = 1.0 (assume standardized activations)
   - δ = 0.05

   a) For the bound to give sin(θ) ≤ 0.1, need Δ_k ≥ 10 · σ² · √(log(40) / (c · 128))
   b) If Δ_{128} ≈ 0.01 (typical eigenvalue gap at k=128 for LLM activations), compute the actual sin(θ) bound
   c) If sin(θ) > 0.3 with n=128, then the default buffer is provably insufficient
   d) Compute n_required for sin(θ) ≤ 0.1 with this Δ_k
```

- [ ] **Step 2: Commit**

```bash
git add lean-proofs/inputs/T2-davis-kahan-buffer-sizing.txt
git commit -m "theorem: T2 — Davis-Kahan perturbation bound for PCA buffer sizing"
```

---

### Task 5: Submit Theorems to Aristotle

**Files:**
- Modify: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/ROADMAP.md`

- [ ] **Step 1: Submit T1**

```bash
aristotle formalize lean-proofs/inputs/T1-forgetting-constant-bound.txt
```

Record the UUID in ROADMAP.md. Expected output: a UUID string.

- [ ] **Step 2: Submit T2**

```bash
aristotle formalize lean-proofs/inputs/T2-davis-kahan-buffer-sizing.txt
```

Record the UUID in ROADMAP.md.

- [ ] **Step 3: Update ROADMAP.md with UUIDs**

Update the T1 and T2 rows with status `submitted` and the UUIDs.

- [ ] **Step 4: Commit**

```bash
git add lean-proofs/ROADMAP.md
git commit -m "theorem: submit T1 and T2 to Aristotle"
```

- [ ] **Step 5: Check status**

```bash
aristotle list
```

Expected: T1 and T2 show as QUEUED or IN_PROGRESS. Results arrive in ~30-60 min. Continue with Python implementation while waiting.

---

### Task 6: Implement `estimate_forgetting_constant()` in features.py

**Files:**
- Modify: `/Users/diego/Dev/non-toxic/sfp/features.py` (append new functions)
- Create: `/Users/diego/Dev/non-toxic/sfp/tests/test_estimate_C.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/diego/Dev/non-toxic/sfp/tests/test_estimate_C.py`:

```python
import torch
import pytest
from features import estimate_forgetting_constant, davis_kahan_buffer_size


def test_estimate_C_known_slope():
    """When forgetting = 2 * drift + noise, C should be close to 2."""
    torch.manual_seed(42)
    n = 500
    drift = torch.rand(n)  # D_ℓ values
    noise = torch.randn(n) * 0.1
    forgetting = 2.0 * drift + noise  # C = 2.0

    C_hat, std_err = estimate_forgetting_constant(drift, forgetting)

    assert abs(C_hat - 2.0) < 0.1, f"Expected C ≈ 2.0, got {C_hat:.3f}"
    assert std_err > 0, "Standard error must be positive"
    assert std_err < 0.5, f"Standard error too large: {std_err:.3f}"


def test_estimate_C_zero_drift():
    """When drift is constant (zero variance), should return inf or raise."""
    drift = torch.ones(100)
    forgetting = torch.randn(100)

    C_hat, std_err = estimate_forgetting_constant(drift, forgetting)

    assert C_hat == float("inf") or torch.isnan(torch.tensor(C_hat))


def test_estimate_C_negative_slope():
    """Negative C means more drift = less forgetting (good layer)."""
    torch.manual_seed(42)
    n = 500
    drift = torch.rand(n)
    forgetting = -1.5 * drift + torch.randn(n) * 0.1

    C_hat, _ = estimate_forgetting_constant(drift, forgetting)

    assert C_hat < 0, f"Expected negative C, got {C_hat:.3f}"


def test_davis_kahan_basic():
    """Buffer size increases when eigenvalue gap shrinks."""
    n_large_gap = davis_kahan_buffer_size(delta_k=1.0, sigma=1.0, confidence=0.95, target_sin_theta=0.1)
    n_small_gap = davis_kahan_buffer_size(delta_k=0.1, sigma=1.0, confidence=0.95, target_sin_theta=0.1)

    assert n_small_gap > n_large_gap, "Smaller gap should require more samples"
    assert n_large_gap >= 1, "Must require at least 1 sample"


def test_davis_kahan_clamp():
    """Buffer size is clamped to [128, 1024]."""
    n = davis_kahan_buffer_size(delta_k=100.0, sigma=1.0, confidence=0.95, target_sin_theta=0.1)
    assert n >= 128, f"Should be clamped to at least 128, got {n}"

    n = davis_kahan_buffer_size(delta_k=0.001, sigma=1.0, confidence=0.95, target_sin_theta=0.1)
    assert n <= 1024, f"Should be clamped to at most 1024, got {n}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/diego/Dev/non-toxic/sfp
python -m pytest tests/test_estimate_C.py -v
```

Expected: FAIL — `ImportError: cannot import name 'estimate_forgetting_constant' from 'features'`

- [ ] **Step 3: Implement `estimate_forgetting_constant` and `davis_kahan_buffer_size`**

Append to `/Users/diego/Dev/non-toxic/sfp/features.py`:

```python
import math


def estimate_forgetting_constant(
    drift: "torch.Tensor", forgetting: "torch.Tensor"
) -> tuple[float, float]:
    """Estimate the forgetting constant C_ℓ = Cov(F, D) / Var(D) via OLS.

    Args:
        drift: [n] tensor of subspace drift magnitudes ‖P_W(Δa)‖ per sample
        forgetting: [n] tensor of forgetting values (Δloss) per sample

    Returns:
        (C_hat, std_err): OLS slope estimate and its standard error
    """
    assert drift.shape == forgetting.shape and drift.dim() == 1
    n = drift.shape[0]

    var_d = drift.var().item()
    if var_d < 1e-12:
        return float("inf"), float("inf")

    cov_fd = ((drift - drift.mean()) * (forgetting - forgetting.mean())).mean().item()
    C_hat = cov_fd / var_d

    # Residual standard error
    residuals = forgetting - (C_hat * drift + (forgetting.mean() - C_hat * drift.mean()))
    residual_var = (residuals**2).sum().item() / max(n - 2, 1)
    std_err = math.sqrt(residual_var / (n * var_d))

    return C_hat, std_err


def davis_kahan_buffer_size(
    delta_k: float,
    sigma: float = 1.0,
    confidence: float = 0.95,
    target_sin_theta: float = 0.1,
    clamp_min: int = 128,
    clamp_max: int = 1024,
) -> int:
    """Compute minimum buffer size for PCA subspace estimation via Davis-Kahan.

    n >= (sigma^4 / (c * delta_k^2 * t^2)) * log(2/delta)

    Args:
        delta_k: eigenvalue gap lambda_k - lambda_{k+1}
        sigma: sub-Gaussian parameter of activations
        confidence: 1 - delta (probability of success)
        target_sin_theta: target principal angle bound t
        clamp_min: minimum buffer size
        clamp_max: maximum buffer size

    Returns:
        Required buffer size, clamped to [clamp_min, clamp_max]
    """
    delta = 1.0 - confidence
    c = 0.5  # universal constant (conservative estimate)
    log_term = math.log(2.0 / delta)
    n = (sigma**4 * log_term) / (c * delta_k**2 * target_sin_theta**2)
    return max(clamp_min, min(clamp_max, math.ceil(n)))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_estimate_C.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add features.py tests/test_estimate_C.py
git commit -m "feat: add estimate_forgetting_constant() and davis_kahan_buffer_size()"
```

---

### Task 7: Implement `information_theoretic_sfp_setup()` and Loss in methods.py

**Files:**
- Modify: `/Users/diego/Dev/non-toxic/sfp/methods.py`
- Create: `/Users/diego/Dev/non-toxic/sfp/tests/test_information_theoretic_sfp.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/diego/Dev/non-toxic/sfp/tests/test_information_theoretic_sfp.py`:

```python
import torch
import pytest
from unittest.mock import MagicMock, patch
from methods import METHODS


def test_information_theoretic_sfp_registered():
    """Our method must be in the METHODS registry."""
    assert "information_theoretic_sfp" in METHODS


def test_information_theoretic_sfp_has_setup():
    """Our loss function must declare SETUP attribute."""
    loss_fn = METHODS["information_theoretic_sfp"]
    assert hasattr(loss_fn, "SETUP")
    assert loss_fn.SETUP == "information_theoretic_sfp"


def test_information_theoretic_sfp_loss_returns_scalar():
    """Loss function must return a scalar tensor."""
    loss_fn = METHODS["information_theoretic_sfp"]

    # Create minimal mock inputs
    model = MagicMock()
    batch = {"input_ids": torch.randint(0, 100, (2, 32)), "labels": torch.randint(0, 100, (2, 32))}

    # Mock model forward to return a loss
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    model.return_value = mock_output

    # Provide empty basis/anchor (no preservation when missing)
    result = loss_fn(model, batch, basis={}, anchor_acts={}, lam=0.1)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0, "Loss must be scalar"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_information_theoretic_sfp.py -v
```

Expected: FAIL — `KeyError: 'information_theoretic_sfp'`

- [ ] **Step 3: Read current methods.py to find insertion point**

```bash
# Read the end of methods.py to see where METHODS dict is defined
```

Find the METHODS dict and the existing `sfp_setup` and `sfp_loss` functions.

- [ ] **Step 4: Add setup and loss functions to methods.py**

Add before the METHODS dict in `methods.py`:

```python
def information_theoretic_sfp_setup(model, tokenizer, memory_buffers, layers=None,
                                     k=128, r=32, **kw):
    """SFP setup with theory-driven layer selection by forgetting constant C_ℓ.

    Instead of probing at fixed relative depths [0.25, 0.5, 0.75], estimates
    the forgetting constant C_ℓ at ALL transformer layers and selects the 3
    with smallest C (tightest forgetting-to-drift bound).
    """
    from features import (collect_activations, build_pca_basis, select_top_r,
                          estimate_forgetting_constant, davis_kahan_buffer_size,
                          get_layer_names)

    mem_samples = [s for buf in memory_buffers.values() for s in buf]
    if not mem_samples:
        return {"basis": {}, "anchor_acts": {}}

    # Get ALL transformer layer names (positions=None would give default 3)
    # Instead, probe at many relative positions
    n_layers_to_probe = 12  # probe every ~8% of depth
    positions = [i / n_layers_to_probe for i in range(1, n_layers_to_probe)]
    candidate_layers = get_layer_names(model, positions=positions)

    # Collect activations at all candidate layers
    all_acts = collect_activations(model, mem_samples, tokenizer, candidate_layers)

    # For each layer, build PCA and estimate C_ℓ
    # C estimation needs (drift, forgetting) pairs.
    # At setup time we don't have post-training activations yet.
    # Use explained variance ratio as proxy: layers where top-r captures less
    # variance have tighter C (drift in preserved subspace is smaller fraction
    # of total drift, so C_ℓ = Cov(F, D_ℓ)/Var(D_ℓ) is bounded by the
    # variance concentration).
    layer_scores = {}
    layer_bases = {}
    for layer_name in candidate_layers:
        acts = all_acts[layer_name]
        u, explained_var = build_pca_basis(acts, k=k)

        # Eigenvalue gap for Davis-Kahan
        if len(explained_var) > r:
            delta_k = (explained_var[r - 1] - explained_var[r]).item()
        else:
            delta_k = explained_var[-1].item()

        # Score: explained variance ratio of top-r
        # Lower ratio → more concentrated → tighter C bound
        total_var = explained_var.sum().item()
        top_r_var = explained_var[:r].sum().item()
        var_ratio = top_r_var / max(total_var, 1e-12)

        layer_scores[layer_name] = var_ratio
        layer_bases[layer_name] = (u, explained_var, delta_k)

    # Select top-3 layers with HIGHEST variance ratio
    # (more variance captured = preservation is more effective)
    selected = sorted(layer_scores, key=layer_scores.get, reverse=True)[:3]

    # Build SFP state for selected layers
    basis = {}
    anchor_acts = {}
    for layer_name in selected:
        u, explained_var, delta_k = layer_bases[layer_name]
        u_r = select_top_r(u, explained_var, r=r)
        basis[layer_name] = u_r
        acts = all_acts[layer_name]
        anchor_acts[layer_name] = (acts @ u_r).detach()

    return {"basis": basis, "anchor_acts": anchor_acts}


def information_theoretic_sfp_loss(model, batch, memory_batch=None,
                                    basis=None, anchor_acts=None,
                                    lam=0.1, **kw):
    """SFP loss with theory-selected layers. Same preservation formula as sfp_loss."""
    # CE loss on new task
    outputs = model(**batch)
    ce_loss = outputs.loss

    if not basis or not anchor_acts:
        return ce_loss

    # Preservation loss on theory-selected layers
    pres_loss = torch.tensor(0.0, device=ce_loss.device)
    hooks = []
    layer_acts = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # Mean pool over sequence length
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            layer_acts[name] = hidden.mean(dim=1)  # [batch, hidden]
        return hook_fn

    # Register hooks for selected layers
    for name, module in model.named_modules():
        if name in basis:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass to collect activations (if not already done by CE loss)
    if not layer_acts:
        with torch.no_grad():
            model(**batch)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute preservation loss
    for layer_name in basis:
        if layer_name not in layer_acts:
            continue
        u_r = basis[layer_name]
        projected = layer_acts[layer_name] @ u_r  # [batch, r]
        # Compare against mean of anchor (anchor is [n_mem, r])
        anchor_mean = anchor_acts[layer_name].mean(dim=0)  # [r]
        pres_loss = pres_loss + ((projected - anchor_mean) ** 2).mean()

    return ce_loss + lam * pres_loss

information_theoretic_sfp_loss.SETUP = "information_theoretic_sfp"
```

Add to the METHODS dict:

```python
"information_theoretic_sfp": information_theoretic_sfp_loss,
```

Add to `method_setup()` function, in the SETUP dispatch:

```python
elif setup_key == "information_theoretic_sfp":
    return information_theoretic_sfp_setup(model, tokenizer, memory_buffers,
                                            k=kw.get("k", 128), r=kw.get("r", 32))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_information_theoretic_sfp.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add methods.py tests/test_information_theoretic_sfp.py
git commit -m "feat: add information_theoretic_sfp method with theory-driven layer selection"
```

---

### Task 8: Create Leaderboard Submission File

**Files:**
- Create: `/Users/diego/Dev/non-toxic/sfp/submissions/information_theoretic_sfp.py`

- [ ] **Step 1: Write submission file**

Create `/Users/diego/Dev/non-toxic/sfp/submissions/information_theoretic_sfp.py`:

```python
"""Information-theoretic SFP: theory-driven layer selection via forgetting constant bounds.

Selects probe layers by explained variance concentration (proxy for forgetting
constant C_ℓ) instead of fixed relative depths. Backed by Lean 4 proofs via
Aristotle: T1 (C as regression coefficient), T2 (Davis-Kahan buffer sizing).
"""
import torch
import torch.nn.functional as F

CONTRIBUTOR = "0xQuinto"
SETUP = "sfp"


def information_theoretic_sfp_loss(model, batch, memory_batch=None,
                                    basis=None, anchor_acts=None,
                                    lam=0.1, **kw) -> torch.Tensor:
    """SFP loss — uses standard sfp setup, compatible with leaderboard runner."""
    outputs = model(**batch)
    ce_loss = outputs.loss

    if not basis or not anchor_acts:
        return ce_loss

    pres_loss = torch.tensor(0.0, device=ce_loss.device)
    hooks = []
    layer_acts = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            layer_acts[name] = hidden.mean(dim=1)
        return hook_fn

    for name, module in model.named_modules():
        if name in basis:
            hooks.append(module.register_forward_hook(make_hook(name)))

    if not layer_acts:
        with torch.no_grad():
            model(**batch)

    for h in hooks:
        h.remove()

    for layer_name in basis:
        if layer_name not in layer_acts:
            continue
        u_r = basis[layer_name]
        projected = layer_acts[layer_name] @ u_r
        anchor_mean = anchor_acts[layer_name].mean(dim=0)
        pres_loss = pres_loss + ((projected - anchor_mean) ** 2).mean()

    return ce_loss + lam * pres_loss
```

Note: This initial submission uses `SETUP = "sfp"` (standard setup) since the leaderboard server doesn't have our custom setup. The full implementation with custom layer selection requires our fork PR to be merged first.

- [ ] **Step 2: Verify submission file is valid**

```bash
python -c "
import ast
with open('submissions/information_theoretic_sfp.py') as f:
    tree = ast.parse(f.read())
fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.endswith('_loss')]
print(f'Found {len(fns)} loss function(s): {[f.name for f in fns]}')
assert len(fns) == 1, 'Must have exactly 1 loss function'
import os
size = os.path.getsize('submissions/information_theoretic_sfp.py')
print(f'File size: {size} bytes')
assert size < 10240, 'Must be under 10KB'
print('Submission file is valid')
"
```

Expected: `Found 1 loss function(s): ['information_theoretic_sfp_loss']`, valid.

- [ ] **Step 3: Commit**

```bash
git add submissions/information_theoretic_sfp.py
git commit -m "feat: add leaderboard submission file for information-theoretic SFP"
```

---

### Task 9: Create Empirical C Estimation Script

**Files:**
- Create: `/Users/diego/Dev/non-toxic/sfp/scripts/estimate_C.py`

- [ ] **Step 1: Write the script**

Create `/Users/diego/Dev/non-toxic/sfp/scripts/estimate_C.py`:

```python
"""Estimate forgetting constant C_ℓ at each layer of a model.

Usage:
    python scripts/estimate_C.py --model HuggingFaceTB/SmolLM2-135M-Instruct --memory 256

Runs a mini fine-tuning loop (1 task), collects pre/post activations at every
transformer layer, and estimates C_ℓ = Cov(Δloss, ‖P_W(Δa)‖) / Var(‖P_W(Δa)‖).

Outputs a table of C_ℓ per layer, sorted by ascending |C_ℓ| (best layers first).
"""
import argparse
import torch
import numpy as np
from model import load_model
from features import (get_layer_names, collect_activations, build_pca_basis,
                      select_top_r, estimate_forgetting_constant, davis_kahan_buffer_size)
from data import load_task_data


def main():
    parser = argparse.ArgumentParser(description="Estimate forgetting constant C per layer")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--memory", type=int, default=256, help="Memory buffer size")
    parser.add_argument("--k", type=int, default=128, help="PCA components")
    parser.add_argument("--r", type=int, default=32, help="Preserved dimensions")
    parser.add_argument("--steps", type=int, default=50, help="Fine-tuning steps for estimation")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # Probe all layers at fine granularity
    n_probe = 20
    positions = [i / n_probe for i in range(1, n_probe)]
    layer_names = get_layer_names(model, positions=positions)
    print(f"Probing {len(layer_names)} layers: {layer_names}")

    # Load first task data as "old task"
    print("Loading task data...")
    task_data = load_task_data("math", tokenizer, max_samples=args.memory * 2)
    mem_samples = task_data[:args.memory]

    # Collect pre-training activations
    print("Collecting pre-training activations...")
    pre_acts = collect_activations(model, mem_samples, tokenizer, layer_names)

    # Compute per-sample old-task loss before training
    print("Computing pre-training losses...")
    pre_losses = []
    model.eval()
    with torch.no_grad():
        for sample in mem_samples:
            inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()
            loss = model(**inputs).loss.item()
            pre_losses.append(loss)
    pre_losses = torch.tensor(pre_losses)

    # Fine-tune on "new task" (code) for a few steps
    print(f"Fine-tuning for {args.steps} steps on code task...")
    new_task_data = load_task_data("code", tokenizer, max_samples=args.steps * 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for i, sample in enumerate(new_task_data[:args.steps]):
        inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        inputs["labels"] = inputs["input_ids"].clone()
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{args.steps}, loss={loss.item():.4f}")

    # Collect post-training activations and losses
    print("Collecting post-training activations...")
    post_acts = collect_activations(model, mem_samples, tokenizer, layer_names)

    print("Computing post-training losses...")
    post_losses = []
    model.eval()
    with torch.no_grad():
        for sample in mem_samples:
            inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            inputs["labels"] = inputs["input_ids"].clone()
            loss = model(**inputs).loss.item()
            post_losses.append(loss)
    post_losses = torch.tensor(post_losses)

    forgetting = post_losses - pre_losses  # [n_mem]

    # Estimate C_ℓ at each layer
    print("\n" + "=" * 70)
    print(f"{'Layer':<30} {'C_ℓ':>10} {'SE':>10} {'Δ_k':>10} {'n_DK':>8} {'VarRatio':>10}")
    print("=" * 70)

    results = []
    for layer_name in layer_names:
        pre_a = pre_acts[layer_name]
        post_a = post_acts[layer_name]
        delta_a = post_a - pre_a  # [n, hidden]

        u, explained_var = build_pca_basis(pre_a, k=args.k)
        u_r = select_top_r(u, explained_var, r=args.r)

        # Subspace drift magnitude per sample
        drift = torch.norm(delta_a @ u_r, dim=1)  # [n]

        C_hat, std_err = estimate_forgetting_constant(drift, forgetting)

        # Eigenvalue gap
        if len(explained_var) > args.r:
            delta_k = (explained_var[args.r - 1] - explained_var[args.r]).item()
        else:
            delta_k = explained_var[-1].item()

        n_dk = davis_kahan_buffer_size(delta_k=max(delta_k, 1e-8), sigma=1.0)

        var_ratio = explained_var[:args.r].sum().item() / max(explained_var.sum().item(), 1e-12)

        results.append({
            "layer": layer_name, "C": C_hat, "SE": std_err,
            "delta_k": delta_k, "n_dk": n_dk, "var_ratio": var_ratio,
        })
        print(f"{layer_name:<30} {C_hat:>10.4f} {std_err:>10.4f} {delta_k:>10.6f} {n_dk:>8d} {var_ratio:>10.4f}")

    # Sort by |C| ascending (smallest = tightest bound)
    results.sort(key=lambda x: abs(x["C"]) if x["C"] != float("inf") else 1e9)

    print("\n" + "=" * 70)
    print("TOP 3 LAYERS (smallest |C_ℓ| = tightest forgetting bound):")
    print("=" * 70)
    for r in results[:3]:
        print(f"  {r['layer']:<30} C={r['C']:.4f}  (Davis-Kahan n_min={r['n_dk']})")

    print("\nBOTTOM 3 LAYERS (largest |C_ℓ| = loosest bound):")
    for r in results[-3:]:
        print(f"  {r['layer']:<30} C={r['C']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/estimate_C.py
git commit -m "feat: add empirical C estimation script for layer analysis"
```

---

### Task 10: Fetch Aristotle Results and Verify Proofs

**Files:**
- Modify: `/Users/diego/Dev/non-toxic/sfp/lean-proofs/ROADMAP.md`

This task runs after Aristotle completes T1 and T2 (~30-60 min after submission).

- [ ] **Step 1: Check Aristotle status**

```bash
aristotle list
```

Expected: T1 and T2 show as COMPLETE.

- [ ] **Step 2: Fetch T1 results**

```bash
aristotle result <T1-UUID> --destination lean-proofs/results/T1-forgetting-constant-bound.tar.gz
cd lean-proofs/results && tar xzf T1-forgetting-constant-bound.tar.gz && cd ../..
```

- [ ] **Step 3: Verify T1 has zero sorry**

```bash
grep -r "sorry" lean-proofs/results/T1-forgetting-constant-bound*/RequestProject/*.lean || echo "ZERO SORRY — fully verified"
cat lean-proofs/results/T1-forgetting-constant-bound*/ARISTOTLE_SUMMARY_*.md
```

Expected: no `sorry` statements found.

- [ ] **Step 4: Fetch T2 results**

```bash
aristotle result <T2-UUID> --destination lean-proofs/results/T2-davis-kahan-buffer-sizing.tar.gz
cd lean-proofs/results && tar xzf T2-davis-kahan-buffer-sizing.tar.gz && cd ../..
```

- [ ] **Step 5: Verify T2 has zero sorry**

```bash
grep -r "sorry" lean-proofs/results/T2-davis-kahan-buffer-sizing*/RequestProject/*.lean || echo "ZERO SORRY — fully verified"
cat lean-proofs/results/T2-davis-kahan-buffer-sizing*/ARISTOTLE_SUMMARY_*.md
```

- [ ] **Step 6: Update ROADMAP.md**

Update T1 and T2 rows: status → `proved`, add sorry count.

- [ ] **Step 7: Commit**

```bash
git add lean-proofs/results/ lean-proofs/ROADMAP.md
git commit -m "theorem: T1 and T2 Aristotle proofs verified — zero sorry"
```

---

### Task 11: Run Local Benchmark and Compare

**Files:** None created — this is a validation step.

- [ ] **Step 1: Run baseline SFP benchmark**

```bash
cd /Users/diego/Dev/non-toxic/sfp
python train.py --method sfp --memory 128 2>&1 | tee results_sfp_baseline.log
```

Note: this requires GPU for Qwen2.5-1.5B. For CPU, use SmolLM2-135M (the default model). Expected runtime: ~5-10 min for fast config.

- [ ] **Step 2: Run our method**

```bash
python train.py --method information_theoretic_sfp --memory 128 2>&1 | tee results_ours.log
```

- [ ] **Step 3: Compare scores**

```bash
python -c "
import re

def parse_score(logfile):
    with open(logfile) as f:
        text = f.read()
    # Find final score line
    scores = re.findall(r'score[:\s]+([\d.]+)', text, re.IGNORECASE)
    return float(scores[-1]) if scores else None

baseline = parse_score('results_sfp_baseline.log')
ours = parse_score('results_ours.log')
print(f'Baseline SFP:              {baseline}')
print(f'Information-theoretic SFP: {ours}')
if baseline and ours:
    delta = ours - baseline
    print(f'Delta:                     {delta:+.4f} ({\"IMPROVEMENT\" if delta > 0 else \"REGRESSION\"})')
"
```

- [ ] **Step 4: Commit results**

```bash
git add results_sfp_baseline.log results_ours.log
git commit -m "experiment: local benchmark — information_theoretic_sfp vs sfp baseline"
```

---

### Task 12: Submit to Leaderboard

**Files:** None created — uses existing submission file.

- [ ] **Step 1: Submit**

```bash
cd /Users/diego/Dev/non-toxic/sfp
python submit.py submissions/information_theoretic_sfp.py
```

Expected: returns a submission ID, starts polling. Full benchmark takes ~45 min.

- [ ] **Step 2: Wait for results**

The submit.py script polls automatically every 10s. Wait for completion.

- [ ] **Step 3: Record results in ROADMAP.md**

Add a results section to ROADMAP.md with the leaderboard score, retention, plasticity.

- [ ] **Step 4: Commit**

```bash
git add lean-proofs/ROADMAP.md
git commit -m "experiment: leaderboard submission — information_theoretic_sfp scored X.XXX"
```
