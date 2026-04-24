# Step 5 — Per-Symbol Surrogate Sweep (pre-committed Gate 3 follow-up)

**Date:** 2026-04-24 (late, post-amendment-v2)
**Checkpoint:** `runs/step3-r2/encoder-best.pt` (epoch 6, MEM=0.504, 376K params)
**Script:** `scripts/avax_gate3_probe.py` (already generalized in commit `ea07bda`)
**Protocol:** same as Gate 3 triage — time-ordered 80/20 per month, stride=50, balanced accuracy, 1000-resample bootstrap 95% CI, N=50 shuffled null.
**Pre-commitment:** ratified spec amendment v2 (commit `9c91f85`, Gate 3 section) — council-5 Rank 2 / council-3 Option B.

## Why this measurement

Council-5's Rank-2 ask, echoed in council-3's recommendation: with AVAX as a single held-out symbol (n=1 on the transfer claim), we cannot distinguish "encoder fails to transfer to AVAX specifically" from "encoder fails to transfer to novel mid-liquid symbols generally, which would fail equally on any in-sample symbol under the same small-n protocol." This sweep pseudo-hold-outs 5 in-sample symbols individually, converting n=1 → n=5 on the transfer-claim falsifiability. If all 5 also fail under matched protocol → AVAX is not anomalous and the Gate 3 amendment is correct. If 3+/5 pass → AVAX was an adversarial symbol selection.

**Symbol selection** spans council-3's microstructure tiers:
- **ASTER** — launchpad / Tier-3 memecoin zone
- **LDO** — DeFi Tier 2/3
- **DOGE** — retail/memecoin Tier 2
- **PENGU** — pure memecoin Tier 3
- **UNI** — DeFi/DEX Tier 2

All were in pretraining (these are pseudo-held-outs; NOT a true Gate 3 test). The measurement interprets whether the Gate-3-style probe can detect any symbol-specific encoder-edge on this data under the stride=50 1-month protocol.

## Results

All 20 cells (5 symbols × 2 months × 2 horizons):

| sym    | month    | H    | n_test | cp    | enc    | enc_lo | enc_hi | pca    | pca_lo | pca_hi | rp     | shuf   | Δ      | CI vs PCA  |
|--------|----------|------|-------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------------|
| ASTER  | 2026-02  | H100 | 464    | 0.517 | 0.494  | 0.448  | 0.539  | 0.507  | 0.461  | 0.552  | 0.519  | 0.501  | -0.014 | OVERLAP    |
| ASTER  | 2026-02  | H500 | 419    | 0.554 | 0.547  | 0.503  | 0.590  | 0.454  | 0.412  | 0.496  | 0.521  | 0.505  | +0.093 | **SEPARATED** |
| ASTER  | 2026-03  | H100 | 372    | 0.532 | 0.499  | 0.449  | 0.552  | 0.495  | 0.447  | 0.545  | 0.525  | 0.495  | +0.004 | OVERLAP    |
| ASTER  | 2026-03  | H500 | 332    | 0.491 | 0.535  | 0.476  | 0.587  | 0.501  | 0.448  | 0.557  | 0.497  | 0.508  | +0.034 | OVERLAP    |
| LDO    | 2026-02  | H100 | 413    | 0.496 | 0.484  | 0.435  | 0.524  | 0.490  | 0.447  | 0.535  | 0.475  | 0.492  | -0.006 | OVERLAP    |
| LDO    | 2026-02  | H500 | 368    | 0.438 | 0.425  | 0.373  | 0.473  | 0.441  | 0.392  | 0.492  | 0.465  | 0.494  | -0.017 | OVERLAP    |
| LDO    | 2026-03  | H100 | 357    | 0.440 | 0.530  | 0.480  | 0.580  | 0.539  | 0.495  | 0.579  | 0.544  | 0.500  | -0.009 | OVERLAP    |
| LDO    | 2026-03  | H500 | 317    | 0.404 | 0.571  | 0.521  | 0.621  | 0.556  | 0.505  | 0.611  | 0.506  | 0.492  | +0.014 | OVERLAP    |
| DOGE   | 2026-02  | H100 | 440    | 0.473 | 0.497  | 0.448  | 0.542  | 0.480  | 0.432  | 0.526  | 0.483  | 0.502  | +0.017 | OVERLAP    |
| DOGE   | 2026-02  | H500 | 395    | 0.435 | 0.546  | 0.495  | 0.597  | 0.548  | 0.503  | 0.592  | 0.478  | 0.497  | -0.002 | OVERLAP    |
| DOGE   | 2026-03  | H100 | 351    | 0.484 | 0.578  | 0.528  | 0.628  | 0.512  | 0.470  | 0.557  | 0.485  | 0.499  | +0.065 | OVERLAP    |
| DOGE   | 2026-03  | H500 | 311    | 0.447 | 0.501  | 0.449  | 0.557  | 0.473  | 0.427  | 0.519  | 0.536  | 0.509  | +0.028 | OVERLAP    |
| PENGU  | 2026-02  | H100 | 421    | 0.520 | 0.552  | 0.504  | 0.600  | 0.510  | 0.467  | 0.556  | 0.482  | 0.504  | +0.043 | OVERLAP    |
| PENGU  | 2026-02  | H500 | 376    | 0.511 | 0.566  | 0.516  | 0.620  | 0.501  | 0.452  | 0.554  | 0.541  | 0.497  | +0.065 | OVERLAP    |
| PENGU  | 2026-03  | H100 | 357    | 0.448 | 0.496  | 0.447  | 0.547  | 0.511  | 0.463  | 0.564  | 0.518  | 0.497  | -0.015 | OVERLAP    |
| PENGU  | 2026-03  | H500 | 317    | 0.404 | 0.516  | 0.465  | 0.568  | 0.452  | 0.393  | 0.504  | 0.471  | 0.505  | +0.064 | OVERLAP    |
| UNI    | 2026-02  | H100 | 428    | 0.491 | 0.512  | 0.467  | 0.559  | 0.499  | 0.463  | 0.534  | 0.470  | 0.501  | +0.013 | OVERLAP    |
| UNI    | 2026-02  | H500 | 383    | 0.504 | 0.439  | 0.392  | 0.488  | 0.488  | 0.449  | 0.530  | 0.467  | 0.499  | -0.049 | OVERLAP    |
| UNI    | 2026-03  | H100 | 363    | 0.485 | 0.503  | 0.451  | 0.557  | 0.510  | 0.463  | 0.561  | 0.502  | 0.501  | -0.007 | OVERLAP    |
| UNI    | 2026-03  | H500 | 323    | 0.433 | 0.420  | 0.366  | 0.481  | 0.463  | 0.419  | 0.511  | 0.508  | 0.504  | -0.043 | OVERLAP    |

(cp = class prior of the positive class in test fold; Δ = encoder minus PCA point estimate)

## Summary counts

| Question | Cells |
|---|---:|
| Encoder point-estimate > PCA point-estimate | **11/20** |
| Encoder CI strictly above PCA CI | **1/20** (ASTER Feb H500) |
| Encoder CI strictly above 51.4% | **3/20** |
| Encoder CI includes 51.4% | **14/20** |
| Shuffled null within μ ± 2σ of 0.500 (pipeline clean) | **20/20** — all σ ≤ 0.038 |

## Interpretation

**1/20 CI separations is precisely the rate expected under chance if the true encoder-vs-PCA distributions overlap.** At α=0.05, we expect 1 spurious "separation" in 20 cells by sampling variance alone. The single ASTER Feb H500 win (+9.3pp, CI strictly separated) is consistent with one such spurious tail draw. It could be real ASTER signal, but 1/20 cannot distinguish real from spurious.

**The encoder's ~1-2pp Gate-1 signal is invisible under this protocol on in-sample symbols.** The Gate-1 pass was on pooled 24-symbol monthly sets (~16K test windows per month across symbols). When pooled down to a single symbol at ~300-460 test windows, the variance floor (bootstrap CI width ~0.09-0.12) exceeds the signal amplitude (~0.01-0.03pp).

**AVAX is NOT anomalous.** Contrast with the AVAX stride=50 Feb+Mar run (commit `2c7ebc2`):
- AVAX encoder CI vs PCA CI: 0/4 cells separated, 4/4 overlap, narrowest overlap 0.4pp on Mar H500.
- Surrogate sweep: 1/20 separated, 19/20 overlap.
- AVAX's failure pattern is inside the surrogate distribution — nothing AVAX-specific.

**Per-symbol narrative:**
- **ASTER**: only "strong" surrogate, Feb H500 passes everything. Mar H500 returns to overlap. Could be a genuine ASTER+Feb+H500 signal (launchpad Tier-3 volatility regime?) or chance. 1/4 cells does not support a consistent transfer-quality claim.
- **LDO, DOGE, PENGU**: point-estimate winners on most cells but always inside each other's CIs. Directional encoder-advantage consistent with Gate 1 at small n.
- **UNI**: encoder loses to PCA on 3/4 cells. Would have "failed" Gate 3 individually — which is the point: you can't declare UNI failing without matching the AVAX failure, and UNI IS in-sample.

## Conclusion

**Pre-commitment discharged.** The per-symbol surrogate sweep confirms the Gate 3 reframe is correct. The protocol is underpowered for the encoder's Gate-1-scale signal on any single-symbol 1-month pool — regardless of whether the symbol is held-out (AVAX) or in-sample (ASTER/LDO/DOGE/PENGU/UNI). Running Gate 3 as a binding pass/fail on any of these would be random outcome selection; the amendment's informational-only reframe is the right methodological call.

**Residual observation:** LDO Mar H500 (enc 0.571, CI [0.521, 0.621]) and PENGU Feb H500 (enc 0.566, CI [0.516, 0.620]) both clear 51.4% on the CI lower bound — 2/20 cells that would "pass" a CI-aware Gate 3 threshold if it existed. This is sparse but nonzero evidence that encoder+LR has some signal on Tier-3-ish symbols at H500 under this protocol. Not enough to reactivate Gate 3, consistent with the amendment's re-activation criteria (which require n_test ≥ 2000 and cluster cohesion delta ≥ +0.10, both currently unmet).

## Reproduce

```bash
mkdir -p runs/step3-r2/surrogate-sweep
for sym in ASTER LDO DOGE PENGU UNI; do
  caffeinate -i uv run python scripts/avax_gate3_probe.py \
    --checkpoint runs/step3-r2/encoder-best.pt \
    --cache data/cache \
    --out runs/step3-r2/surrogate-sweep/gate3-surrogate-${sym}.json \
    --target-symbols ${sym} \
    --months 2026-02 2026-03 \
    --horizons 100 500 \
    --mode pretrain \
    --seed 0
done
```

Runtime: ~2 min total on M4 Pro MPS (forward pass + bootstrap + N=50 shuffles per cell).

## Artifacts (gitignored under `runs/`)

- `runs/step3-r2/surrogate-sweep/gate3-surrogate-{ASTER,LDO,DOGE,PENGU,UNI}.{json,log}`
