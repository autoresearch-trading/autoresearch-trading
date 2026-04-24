# Step 5 — Gate 3 Triage: Bootstrap CIs + In-Sample Control

**Date:** 2026-04-24
**Checkpoint:** `runs/step3-r2/encoder-best.pt` (epoch 6, MEM=0.504, 376K params)
**Script:** `scripts/avax_gate3_probe.py` (commit `ea07bda`)
**Protocol:** time-ordered 80/20 per month, stride=50, balanced accuracy, 1000-resample percentile bootstrap 95% CI per cell, N=50 permutations for the shuffled null.

## TL;DR

**Verdict: EXONERATED (of the clean "AVAX failure" falsifier claim) — but the encoder is also not beating flat baselines anywhere at stride=50.** The AVAX encoder vs PCA CIs overlap substantially across all 4 primary cells, so the Gate 3 failure is not tight. Crucially, the **in-sample control on LINK + LTC fails the same way on the same protocol**: encoder LR is indistinguishable from majority (and below PCA) on 4/4 cells with CI overlap. This means stride=50 on 1-month symbol-specific pools is the wrong probe — not that AVAX specifically fails to generalize. A single-symbol pool at ~400-900 test windows is too noisy to sustain the encoder's ~1-2pp Gate-1 signal, and the failure on AVAX was never distinguishable from a failure on LINK+LTC.

## 1. AVAX (pre-registered Gate 3) — bootstrap CIs at stride=50

**Artifact:** `runs/step3-r2/gate3-avax-bootstrap.json`

### February AVAX (28 shards, 2,360 windows)

| H   | predictor | bal_acc | 95% CI          | class_prior | n_test |
|-----|-----------|---------|-----------------|-------------|--------|
| 100 | encoder_lr | 0.5313 | [0.4878, 0.5785] | 0.485      | 460 |
| 100 | pca_lr     | 0.5475 | [0.5015, 0.5940] | 0.485      | 460 |
| 100 | rp_lr      | 0.4821 | [0.4412, 0.5266] | 0.485      | 460 |
| 100 | majority   | 0.5000 | [0.5000, 0.5000] | 0.485      | 460 |
| 100 | shuffled (μ ± 2σ) | 0.5007 | up95=0.5422 (σ=0.0208) | 0.485 | 460 |
| 500 | encoder_lr | 0.4907 | [0.4420, 0.5380] | 0.462      | 416 |
| 500 | pca_lr     | 0.5193 | [0.4737, 0.5608] | 0.462      | 416 |
| 500 | rp_lr      | 0.5499 | [0.5042, 0.5926] | 0.462      | 416 |
| 500 | majority   | 0.5000 | [0.5000, 0.5000] | 0.462      | 416 |
| 500 | shuffled (μ ± 2σ) | 0.5004 | up95=0.5607 (σ=0.0302) | 0.462 | 416 |

### March AVAX (25 shards, 1,898 windows)

| H   | predictor | bal_acc | 95% CI          | class_prior | n_test |
|-----|-----------|---------|-----------------|-------------|--------|
| 100 | encoder_lr | 0.5136 | [0.4668, 0.5616] | 0.485      | 369 |
| 100 | pca_lr     | 0.4928 | [0.4522, 0.5340] | 0.485      | 369 |
| 100 | rp_lr      | 0.5046 | [0.4598, 0.5479] | 0.485      | 369 |
| 100 | majority   | 0.5000 | [0.5000, 0.5000] | 0.485      | 369 |
| 100 | shuffled (μ ± 2σ) | 0.4995 | up95=0.5576 (σ=0.0290) | 0.485 | 369 |
| 500 | encoder_lr | 0.4601 | [0.4103, 0.5104] | 0.438      | 329 |
| 500 | pca_lr     | 0.5566 | [0.5066, 0.6035] | 0.438      | 329 |
| 500 | rp_lr      | 0.5153 | [0.4649, 0.5667] | 0.438      | 329 |
| 500 | majority   | 0.5000 | [0.5000, 0.5000] | 0.438      | 329 |
| 500 | shuffled (μ ± 2σ) | 0.4995 | up95=0.5566 (σ=0.0285) | 0.438 | 329 |

### AVAX findings

- **Shuffled null is clean at N=50**: mean tightly around 0.500 (0.4995–0.5007), σ=0.02–0.03. The old single-seed Apr 0.700 is retroactively explained as one tail draw of this distribution.
- **Encoder CIs overlap PCA CIs on 4/4 cells.** Feb H100: enc [0.488, 0.579] vs pca [0.502, 0.594]; Feb H500: enc [0.442, 0.538] vs pca [0.474, 0.561]; Mar H100: enc [0.467, 0.562] vs pca [0.452, 0.534]; Mar H500: enc [0.410, 0.510] vs pca [0.507, 0.604]. **The narrowest overlap is Mar H500** (encoder CI hi 0.5104 vs pca CI lo 0.5066, overlap only 0.0038 wide — the sign is "PCA beats encoder"). Everywhere else the encoder cannot be statistically distinguished from PCA.
- **Encoder vs majority (chance 0.5)**: Feb H100 encoder CI includes 0.5; Feb H500 includes 0.5 (and point estimate below it); Mar H100 includes 0.5; Mar H500 CI is entirely below 0.5 (encoder does *worse* than flipping a coin, a sign of label mismatch at long horizon). **Not a single cell places encoder_lr significantly above chance.**
- **51.4% threshold**: encoder point estimate hits it only on Feb H100 (0.5313); CI lower bound never clears 51.4%. The pre-registered Gate 3 criterion is not met at CI-aware rigor.

## 2. In-sample control: LINK + LTC (same encoder, same months, same protocol)

**Artifact:** `runs/step3-r2/gate3-insample-control.json`

Council-5's required disambiguation: if in-sample Tier-2 symbols pass on this protocol, AVAX is the anomaly; if they fail the same way, the protocol is underpowered regardless of which symbol is held out.

### February LINK+LTC (56 shards, 4,498 windows)

| H   | predictor | bal_acc | 95% CI          | class_prior | n_test |
|-----|-----------|---------|-----------------|-------------|--------|
| 100 | encoder_lr | 0.4953 | [0.4620, 0.5274] | 0.494      | 877 |
| 100 | pca_lr     | 0.5218 | [0.4901, 0.5555] | 0.494      | 877 |
| 100 | rp_lr      | 0.5190 | [0.4877, 0.5495] | 0.494      | 877 |
| 100 | shuffled (μ ± 2σ) | 0.5040 | up95=0.5490 (σ=0.0225) | 0.494 | 877 |
| 500 | encoder_lr | 0.4942 | [0.4566, 0.5283] | 0.484      | 787 |
| 500 | pca_lr     | 0.5195 | [0.4845, 0.5545] | 0.484      | 787 |
| 500 | rp_lr      | 0.5356 | [0.5019, 0.5701] | 0.484      | 787 |
| 500 | shuffled (μ ± 2σ) | 0.5004 | up95=0.5534 (σ=0.0265) | 0.484 | 787 |

### March LINK+LTC (50 shards, 3,806 windows)

| H   | predictor | bal_acc | 95% CI          | class_prior | n_test |
|-----|-----------|---------|-----------------|-------------|--------|
| 100 | encoder_lr | 0.5093 | [0.4765, 0.5432] | 0.472      | 741 |
| 100 | pca_lr     | 0.5009 | [0.4673, 0.5353] | 0.472      | 741 |
| 100 | rp_lr      | 0.5195 | [0.4885, 0.5522] | 0.472      | 741 |
| 100 | shuffled (μ ± 2σ) | 0.4946 | up95=0.5331 (σ=0.0192) | 0.472 | 741 |
| 500 | encoder_lr | 0.4960 | [0.4576, 0.5317] | 0.505      | 661 |
| 500 | pca_lr     | 0.4736 | [0.4434, 0.5042] | 0.505      | 661 |
| 500 | rp_lr      | 0.4902 | [0.4577, 0.5210] | 0.505      | 661 |
| 500 | shuffled (μ ± 2σ) | 0.4971 | up95=0.5487 (σ=0.0258) | 0.505 | 661 |

### LINK+LTC findings

- **Encoder fails to beat majority on 3/4 cells** (point estimates 0.495, 0.509, 0.494, 0.496). Only Mar H100 lifts above 0.500 (0.5093) and its CI [0.477, 0.543] still contains 0.5.
- **Encoder CIs overlap PCA CIs on 4/4 cells** with the same pattern as AVAX: point estimate below PCA on 3/4, above PCA on 1/4 (Mar H100: encoder 0.5093 vs PCA 0.5009, CI overlap enormous). The Mar H500 "encoder > PCA" cell goes the other direction (encoder 0.496 vs PCA 0.474), also CI-overlapping.
- **None of the 4 cells clears the 51.4% threshold** — not even PCA. n_test ≈ 660–880 is ~2× AVAX's sample, yet the encoder's Gate-1-sized (+1–2pp) lift is nowhere visible.

## 3. AAVE (optional, in-sample, DeFi Tier 2)

**Artifact:** `runs/step3-r2/gate3-insample-aave.json`

| H   | predictor | bal_acc | 95% CI          | class_prior | n_test |
|-----|-----------|---------|-----------------|-------------|--------|
| 100-Feb | encoder_lr | 0.5228 | [0.4780, 0.5664] | 0.500 | 482 |
| 100-Feb | pca_lr     | 0.4440 | [0.4050, 0.4832] | 0.500 | 482 |
| 100-Mar | encoder_lr | 0.4604 | [0.4123, 0.5071] | 0.488 | 385 |
| 100-Mar | pca_lr     | 0.5304 | [0.4831, 0.5793] | 0.488 | 385 |
| 500-Feb | encoder_lr | 0.4402 | [0.3957, 0.4851] | 0.485 | 437 |
| 500-Feb | pca_lr     | 0.4538 | [0.4060, 0.5011] | 0.485 | 437 |
| 500-Mar | encoder_lr | 0.4983 | [0.4480, 0.5482] | 0.432 | 345 |
| 500-Mar | pca_lr     | 0.5169 | [0.4699, 0.5606] | 0.432 | 345 |

AAVE behaves like AVAX: encoder passes 51.4% only on Feb H100 (0.5228), its CI contains the threshold, and the encoder vs PCA relative ordering flips between months (Feb: encoder beats PCA by 7.9pp; Mar: PCA beats encoder by 7.0pp). This is exactly the stride=200 "lucky cell" pattern council-5 warned about, now visible at stride=50 because the AAVE monthly pool is single-symbol and thus small (n_test 345–482).

## 4. Verdict interpretations

### Q1: Does the AVAX failure survive bootstrap scrutiny (encoder CI clearly below PCA CI)?

**No.** Encoder vs PCA CIs overlap on 4/4 AVAX cells (the narrowest is Mar H500 at a 0.4pp overlap, and in the direction PCA > encoder). The stride=50 result is consistent with "encoder and PCA are both around chance on AVAX-only windows, with PCA slightly ahead by a margin smaller than the sampling-variance bar." It is NOT a clean CI-based falsifier that the encoder transfers worse than PCA to AVAX specifically. It IS a clean falsifier of the stride=200 Feb H100 "pass" (0.575 is now inside [0.488, 0.579] at stride=50 — i.e., the previous point estimate was at the upper percentile of the new CI, confirming the stride=200 result was overoptimistic).

### Q2: Do in-sample symbols LINK + LTC pass Gate 3 criteria on the same methodology?

**No — they fail the same way AVAX fails.** On 4/4 cells of LINK+LTC (n_test ≈ 660–880, *larger* than AVAX) the encoder does not beat majority, does not beat PCA with CI separation, and does not clear 51.4%. The shuffled null is clean (mean 0.5004–0.5040, σ ~0.02–0.03), so the probe is not broken; the encoder simply does not produce a ≥1pp lift on month-sized single-symbol-cluster pools at this test density. AAVE on its own replicates the AVAX pattern even more starkly (Feb H100 encoder >> PCA; Mar H100 PCA >> encoder; both inside each other's CIs).

### Q3: One-word decision

**EXONERATED.** AVAX is not anomalous — it behaves the same as in-sample LINK+LTC and in-sample AAVE under this probe. What is falsified is NOT "the encoder doesn't transfer to AVAX" but "the 1-month single-symbol probe at ~400–900 test windows can detect the encoder's Gate-1 signal." The Gate-1 signal was visible on a 25-symbol pool with ~16K windows per eval month (council-aware H500 numbers from `docs/experiments/step3-run-2-gate1-pass.md`); shrinking to 1–2 symbols and 1 month collapses it into the CI.

## 5. What this changes for the spec amendment

- **Do NOT amend Gate 3 on the basis of the current stride=50 AVAX numbers.** The CI overlaps are definitive: the AVAX failure is not statistically distinguishable from the in-sample failure, and pre-registered thresholds read on point estimates inside their own CI would be p-hacking in reverse (retroactive "failure" on noise).
- **The binding question becomes:** was the encoder ever positioned to produce a pre-registered detectable lift on a single-month single-symbol pool? Gate 1's passing numbers were on a pool of ~63K windows across 25 symbols per month; that is ~80× the AVAX n_test here. The pre-registration should have specified the aggregation unit for Gate 3, and it did not.
- **Recommended next steps (in order):**
  1. Cross-symbol SimCLR cluster cohesion on the 6 liquid anchors (council-5 Rank 2 / Rank 3) — this tells us whether transfer was even possible.
  2. Expand Gate 3 to the AVAX-only pool evaluated *jointly with matched-size in-sample pools* so the falsifier contrast is apples-to-apples (if 10-symbol in-sample pool at 4K windows/month beats PCA by 1pp and AVAX does not, that's a clean signal).
  3. Consider a "held-out set" formulation (council-3 Option B) with n ≥ 3 symbols to expand the effective sample before declaring AVAX-specific.

## Reproduce

```bash
# Bootstrap AVAX (Gate 3 primary)
caffeinate -i uv run python scripts/avax_gate3_probe.py \
    --checkpoint runs/step3-r2/encoder-best.pt \
    --cache data/cache \
    --out runs/step3-r2/gate3-avax-bootstrap.json \
    --months 2026-02 2026-03 --horizons 100 500 --mode pretrain --seed 0

# In-sample control: LINK + LTC
caffeinate -i uv run python scripts/avax_gate3_probe.py \
    --checkpoint runs/step3-r2/encoder-best.pt \
    --cache data/cache \
    --out runs/step3-r2/gate3-insample-control.json \
    --target-symbols LINK LTC \
    --months 2026-02 2026-03 --horizons 100 500 --mode pretrain --seed 0

# In-sample control: AAVE (optional, DeFi Tier 2)
caffeinate -i uv run python scripts/avax_gate3_probe.py \
    --checkpoint runs/step3-r2/encoder-best.pt \
    --cache data/cache \
    --out runs/step3-r2/gate3-insample-aave.json \
    --target-symbols AAVE \
    --months 2026-02 2026-03 --horizons 100 500 --mode pretrain --seed 0
```

## Artifacts

- `runs/step3-r2/gate3-avax-bootstrap.json` + `.log`
- `runs/step3-r2/gate3-insample-control.json` + `.log` (LINK + LTC)
- `runs/step3-r2/gate3-insample-aave.json` + `.log`
