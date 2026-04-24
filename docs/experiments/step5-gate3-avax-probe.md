# Step 5 — Gate 3 AVAX Held-Out Symbol Probe

**Date:** 2026-04-24
**Checkpoint:** `runs/step3-r2/encoder-best.pt` (epoch 6, MEM=0.504, 376K params)
**Script:** `scripts/avax_gate3_probe.py`
**Commits:** `5acde01` (probe script); this writeup

## TL;DR

**Gate 3 FAILS.** On the pre-designated held-out symbol AVAX, the SSL
encoder does not beat flat baselines on any month × horizon combination at
stride=50 (the reliable-sample-size setting). The stride=200 Feb H100 result
that looked like a pass (encoder 0.575, +7.9pp over PCA) does not survive
re-evaluation at higher data density: at stride=50 it becomes 0.531
(PCA 0.548, −1.6pp). The flip is consistent with small-sample variance at
n≈120 test, where the binomial SE on balanced accuracy is ±0.065 per class.

| Run | Feb H100 enc | Feb H100 PCA | Mar H500 enc | Mar H500 PCA | Verdict |
|---|---|---|---|---|---|
| stride=200 (n~120 test) | 0.575 | 0.496 | 0.402 | 0.539 | encoder ahead on 1/6 cells |
| stride=50 (n~480 test) | 0.531 | 0.548 | 0.460 | 0.557 | encoder ahead on 0/6 cells |

**This is a real finding, not a measurement artifact.** Unlike Gate 1's
run-0 where the failure was three probe bugs, the Gate 3 probe tooling is
clean: identical architecture to `temporal_stability.py` which produced the
passing Gate 1 result. The difference is the data, not the probe.

## Protocol

- Script: `scripts/avax_gate3_probe.py` (commit `5acde01`)
- Checkpoint: `runs/step3-r2/encoder-best.pt` (unchanged, same as Gate 1 run-2)
- Data: AVAX-only shards, 2026-02 (28 shards), 2026-03 (25 shards), 2026-04 (7 shards, informational)
- Predictors: encoder_lr, pca_lr, rp_lr, majority, shuffled_pca_lr
- Horizons: H100 (spec), H500 (Gate 1 empirical)
- Split: time-ordered 80/20 per month, min_valid=200
- Two stride settings:
  - stride=200 (eval): no window overlap, small sample
  - stride=50 (pretrain): 4× density, overlapping windows (effective n ≈ sqrt(4)× = 2× independent samples)

## Full results

### Stride=200 (eval, reference protocol)

**February AVAX** (600 windows, ~120 test)
| predictor | H100 | H500 |
|---|---|---|
| encoder_lr | **0.575** | 0.443 |
| pca_lr | 0.496 | 0.491 |
| rp_lr | 0.487 | 0.576 |
| majority | 0.500 | 0.500 |
| shuffled_pca_lr | 0.488 | 0.566 |

**March AVAX** (484 windows, ~97 test)
| predictor | H100 | H500 |
|---|---|---|
| encoder_lr | 0.492 | 0.402 |
| pca_lr | 0.495 | 0.539 |
| rp_lr | 0.477 | 0.535 |
| majority | 0.500 | 0.500 |
| shuffled_pca_lr | 0.538 | 0.540 |

**April AVAX** (78 windows) — below min_valid=200, all N/A.

### Stride=50 (pretrain, 4× density)

**February AVAX** (2,360 windows, ~472 test)
| predictor | H100 | H500 |
|---|---|---|
| encoder_lr | 0.531 | 0.491 |
| pca_lr | **0.548** | 0.519 |
| rp_lr | 0.482 | 0.550 |
| majority | 0.500 | 0.500 |
| shuffled_pca_lr | 0.504 | 0.526 |

**March AVAX** (1,898 windows, ~380 test)
| predictor | H100 | H500 |
|---|---|---|
| encoder_lr | 0.514 | 0.460 |
| pca_lr | 0.493 | **0.557** |
| rp_lr | 0.505 | 0.515 |
| majority | 0.500 | 0.500 |
| shuffled_pca_lr | 0.493 | 0.484 |

**April AVAX** (303 windows, ~60 test) — still underpowered; shuffled
hits 0.700 H500, confirming noise dominates.

## Analysis

### 1. The Feb H100 "pass" at stride=200 does not replicate

At stride=200, Feb H100 encoder = 0.575 looked like +7.9pp over PCA.
At stride=50, the same encoder checkpoint evaluated on the same data gets
0.531 — only +1.7pp over majority, actually **-1.7pp below PCA**. The
stride=200 sample (n=120 test) is too small; binomial SE per class is
sqrt(0.5·0.5/60) ≈ 0.065, giving 95% CI ≈ ±0.13 on balanced accuracy.
Every stride=200 number is inside 1-2σ of chance.

### 2. Encoder does NOT transfer to AVAX at stride=50

The reliable-sample-size result: encoder_lr loses to PCA on AVAX at every
tested month × horizon. This is the opposite of Gate 1's finding on the
pretrained universe, where encoder beat PCA by +1.9-2.3pp across 15-17
symbols on matched-density held-out months.

### 3. Why Gate 3 may fail while Gate 1 passes

Candidate mechanisms (not yet distinguished):

**H1: Symbol-specific features, not universal tape structure.**
The encoder learned to recognize tape regimes in a symbol-conditional way.
The 256-dim global embedding encodes "BTC-like tape state" vs "SOL-like
tape state" rather than "absorption" vs "climax" in a symbol-invariant
way. Cross-symbol SimCLR was only applied to 6 liquid anchors
(BTC/ETH/SOL/BNB/LINK/LTC) per gotcha #25; AVAX's structure may be too
different for forced invariance to have kicked in even if we had tried.

**H2: Feature-distribution mismatch.**
AVAX's 17-feature distribution differs from pretrained symbols in ways
that mis-calibrate the encoder's BatchNorm running stats and early-layer
filters. The encoder sees AVAX through the wrong statistical lens.

**H3: Label structure mismatch.**
AVAX direction labels at the 100/500-event horizons have different class
balance or different conditional distribution than the training symbols,
making even a universal representation useless when fitted with LR on
AVAX-only labels.

### 4. What this does NOT yet rule out

- **Other held-out symbols might pass.** AVAX is n=1. We can't generalize
  from a single held-out symbol to "the representations don't transfer."
- **Other checkpoints might be better.** Gate 1 used best-MEM epoch 6.
  Final-epoch or later-epoch checkpoints might have learned more
  symbol-invariant features (council-6 would argue this is unlikely — MEM
  minimum should be the maximum representation purity).
- **Cross-symbol SimCLR expansion might fix it.** Currently only 6 liquid
  anchors use cross-symbol positive pairs. Widening to all 24 pretraining
  symbols could force more symbol-invariance — but that's a retraining
  decision, not a Gate 3 remediation.

## Implications for the research program

1. **Gate 3 as currently written is not going to pass on this checkpoint.**
   Gate 3's spec language ("held-out symbol accuracy > 51.4% at H100") is
   a binary test; encoder 0.514 on Mar stride=50 is right at the threshold
   and likely noise.

2. **The "representation learning" claim needs qualification.** On the
   pretrained universe, the encoder beats PCA (Gate 1 pass). On a truly
   held-out symbol, it does not. This is compatible with "the encoder
   learned useful features that are symbol-specific, not universal" — a
   weaker but still interesting claim, and one the spec should acknowledge.

3. **Step 4 (Gate 2 fine-tuning) decision changes.** Fine-tuning unfreezes
   the encoder and trains direction heads. On the pretrained universe
   this will likely work (Gate 1 signal is there). On AVAX it will likely
   not transfer. Step 4 can still proceed with in-universe symbols but
   we need to be explicit that fine-tuning does not address the Gate 3
   finding.

4. **Spec amendment scope expands.** Not just Gate 1's horizon+held-out
   language — we also need a principled decision on Gate 3:
   - Option A: Drop Gate 3 (AVAX sample too small to be statistically
     meaningful at matched-density; defer to multi-symbol held-out in
     Step 6 interpretation).
   - Option B: Broaden Gate 3 (designate N held-out symbols, pass if
     encoder > flat on ≥K/N).
   - Option C: Reframe Gate 3 (acknowledge non-transfer as a finding;
     pass criterion is "representations capture meaningful tape features
     on the pretrained universe," not "representations are universal
     across symbols").

## Next actions (dispatched)

- Council-5 (falsifiability skeptic): is the stride=50 AVAX failure a
  clean falsifier, or is the small-n caveat too strong to draw a
  conclusion?
- Council-3 (microstructure theory): is AVAX structurally different
  enough from the pretrained universe that cross-symbol generalization
  would not be expected even from a good encoder?
- After council comes back: decide on Gate 3 spec amendment direction
  (A / B / C above) and update the spec in the same pass as Gate 1's
  H500/matched-density language.

## Reproduce

```bash
# stride=200 (eval, matches Gate 1 protocol)
caffeinate -i uv run python scripts/avax_gate3_probe.py \
    --checkpoint runs/step3-r2/encoder-best.pt \
    --cache data/cache \
    --out runs/step3-r2/avax-gate3-probe.json \
    --months 2026-02 2026-03 2026-04 --horizons 100 500 \
    --mode eval --seed 0

# stride=50 (pretrain, 4x density)
caffeinate -i uv run python scripts/avax_gate3_probe.py \
    --checkpoint runs/step3-r2/encoder-best.pt \
    --cache data/cache \
    --out runs/step3-r2/avax-gate3-probe-stride50.json \
    --months 2026-02 2026-03 2026-04 --horizons 100 500 \
    --mode pretrain --seed 0
```

Artifacts:
- `runs/step3-r2/avax-gate3-probe.json` (stride=200)
- `runs/step3-r2/avax-gate3-probe.log`
- `runs/step3-r2/avax-gate3-probe-stride50.json` (stride=50)
- `runs/step3-r2/avax-gate3-probe-stride50.log`
