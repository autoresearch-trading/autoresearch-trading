# Research: Aristotle CLI Setup & Representation Learning Theorem Candidates

## Question
1. How to set up Aristotle (Harmonic's Lean 4 CLI) from scratch
2. What representation learning claims are worth formalizing for this branch

## Sources
1. Aristotle Terms of Use, March 2026: `https://aristotle.harmonic.fun/terms`
2. Aristotle auth portal: `https://auth.aristotle.harmonic.fun/`
3. Sonoda et al. (2025). "Lean Formalization of Generalization Error Bound by Rademacher Complexity and Dudley's Entropy Integral." arXiv:2503.19605. Code: `github.com/auto-res/lean-rademacher`
4. HaoChen et al. (NeurIPS 2021). "Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss." arXiv:2106.04156
5. Tosh et al. (2021). arXiv:2008.01064 — pen-and-paper SSL generalization bounds
6. Project batch scripts: `docs/archive/proofs/fetch_results_v2.sh`

## Aristotle CLI Setup

**Binary.** Already at `/Users/diego/.local/bin/aristotle`. No public installer — distributed directly by Harmonic after signup at `auth.aristotle.harmonic.fun`. Contact `aristotle@harmonic.fun` to reacquire.

**Authentication.** `ARISTOTLE_API_KEY` env var, format `arstl_<token>`. Auth0-based. Free tier — no fees per Terms Section 9.

**Core commands:**
```bash
aristotle formalize <path>.txt                     # submit → UUID
aristotle list                                      # QUEUED → IN_PROGRESS → COMPLETE
aristotle result <UUID> --destination <path>.tar.gz # download
```

**Output structure (extracted tarball):**
- `ARISTOTLE_SUMMARY_<UUID>.md` — what proved, what `sorry` (unproved gaps)
- `RequestProject/*.lean` — machine-verified Lean 4

**Gotchas:**
- "Zero sorry" = fully proved. Always check summary.
- Aristotle generates counterexamples for false claims (catches threshold bugs).
- Typical turnaround: 20-60 min.
- ToS 3c prohibits using outputs as ML training data.

## SSL Formal Verification: State of the Art

**No published Lean 4 formalization of InfoNCE, SimCLR, NT-Xent, or masked prediction exists.** SSL theory literature is entirely pen-and-paper or empirical.

Closest Lean foundation: Sonoda et al. (2025) mechanized Rademacher complexity + Dudley entropy integral on Mathlib. Applicable to linear probe guarantees but not yet applied to SSL.

## Theorem Candidates (T48-T52)

| # | Claim | Effort | Value |
|---|-------|--------|-------|
| **T48** | Gate 1 threshold: under binomial null, tau=51.4% is the min per-symbol accuracy for 15/25 symbols to be significant at p<0.05 | Low | High |
| **T49** | Receptive field arithmetic: dilated CNN with kernel=5, dilations [1,2,4,8,16,32] → RF=253. Block masking B=5 leaves ≥248 context positions | Trivial | High |
| **T50** | Walk-forward embargo: if is_open has half-life 20 (rho≈0.966), embargo K<20 leaves train-test correlation >0.5 | Low | Moderate |
| **T51** | Symbol probe: balanced 25-class linear classifier on isotropic embeddings can't exceed 1/25 + ε by VC bounds. >20% = symbol-specific encoding | Moderate | Moderate |
| **T52** | Equal-symbol sampling: without it, gradient contribution ratio N_BTC/N_min ≥ 40. Equal sampling reduces per-symbol gradient variance by 25× | Moderate | Moderate |

## Verdict

**Selectively valuable, narrower scope than supervised branch.**

**Submit first (combinatorial/arithmetic, <30min each):**
- T48 — Gate threshold justification
- T49 — Receptive field
- T50 — Embargo length

These provide machine-verified justification for the three most-questioned design decisions in the spec.

**Skip:** InfoNCE / NT-Xent information-theoretic bounds. Building the mutual information scaffolding in Lean 4 from scratch is a multi-year library contribution. Already proved informally in van den Oord et al. 2018 — use as reference, don't re-prove.

**Cannot be formalized:** Representation quality itself. Requires empirical evaluation gates, not formal proofs.
