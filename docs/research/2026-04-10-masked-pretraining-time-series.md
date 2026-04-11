# Research: Masked Pretraining for Time Series

## Question
What block sizes, masking ratios, and augmentation strategies work for masked/contrastive pretraining on time series? Does our council's recommendation of 20-event blocks at 20% masking hold up?

## Sources
1. SimMTM (NeurIPS 2023, arXiv:2302.00861) — masked time series modeling via series-level reconstruction
2. TimeMAE (arXiv:2303.00320) — masked autoencoders for time series with sub-series segmentation
3. PatchTST (ICLR 2023, arXiv:2211.14730) — channel-independent patched Transformer for TS
4. Ti-MAE (arXiv:2301.08871) — time series MAE with 60% masking
5. Nature Scientific Reports (2025, s41598-025-33636-w) — ablation study on block size × masking ratio
6. TS2Vec (AAAI 2022) — hierarchical contrastive for TS via temporal cropping
7. TS-TCC (TPAMI 2023) — temporal contrastive with weak+strong augmentation asymmetry
8. TNC (ICLR 2021) — temporal neighborhood coding
9. Ruan et al. (2023, arXiv:2306.05987v2) — contrastive on CAC40 order flow (triplet loss, no augmentation)
10. Finding Order in Chaos (ETH, arXiv:2309.13439) — time reversal harms causal TS representations

## Key Findings

### Masking Strategy

| Paper | Architecture | Mask Ratio | Block Size | Block/Seq Ratio |
|-------|-------------|------------|------------|-----------------|
| BERT | Transformer | 15% | 1 token | point |
| Image MAE | ViT | 75% | 14×14 patch | ~2% spatial |
| Ti-MAE | Transformer | 60% | 1 timestep | point |
| TimeMAE | Transformer | 40-60% | sub-series (W steps) | ~5-15% |
| PatchTST | Transformer | ~40% | patch 16 steps | ~3-5% |
| Nature ablation | CNN-like | **50% optimal** | medium block | ~4-5% |

- **SimMTM key insight:** Point-level masking is trivially solvable by local interpolation — the exact problem our council identified with 5-event blocks and RF=253.
- **TimeMAE:** Segments into non-overlapping sub-series before masking. Point-level has "low semantic density."
- **Nature ablation:** 50% masking optimal, medium blocks best, random masking > grid masking.
- **Architecture dependence:** Transformers tolerate 40-75% masking. For CNNs with full-coverage RF, block SIZE matters more than ratio.

### Augmentation Strategy

**Safe augmentations:**
- Jitter (Gaussian noise): σ=0.05-0.1 in normalized space (parametric augmentation paper: σ=0.5 too aggressive for financial)
- Amplitude scaling: [0.9, 1.1] (preserves sign of returns)
- Temporal crop: drop 20-30 events from one end (TS2Vec's core mechanism)
- Latent masking: TS2Vec masks latent vectors, not raw inputs

**Harmful augmentations (empirical evidence):**
- Time reversal: ETH paper shows degraded representations for causal TS
- Segment permutation as positive pairs: destroys causal order
- Point-level masking > 30% without block structure: SimMTM shows "ruins vital temporal variations"

**Financial-specific:**
- Ruan et al. (CAC40 order flow): pure temporal proximity without augmentation is sufficient
- Parametric Augmentation (ICLR 2024): recommends σ=0.05-0.1 for financial applications (not 0.5)

**TS-TCC finding:** Augmentation asymmetry (weak view + strong view) is critical — two weak augmentations can fail to prevent collapse (up to 20pp difference).

## Relevance to Our Project

1. **20-event blocks confirmed.** Literature consistently uses 5-15% of sequence length as block size. Our 20/200 = 10% is in the sweet spot.
2. **20% masking is conservative.** Literature uses 40-60% for Transformers. For CNN first run, 20% is safe — can increase to 30-40% if reconstruction is too easy.
3. **Augmentation noise validated.** Council-6's σ=0.05 (trade features) and σ=0.15 (OB features) aligns with literature's σ=0.05-0.1 for financial data. σ=0.02 (original spec) was too weak.
4. **Add temporal crop augmentation.** TS2Vec's approach (overlapping subseries) maps to our ±10 event jitter but could be extended to ±20-30 events for the "strong" view.
5. **Asymmetric augmentation.** TS-TCC evidence: pair a weak view (jitter only) with a strong view (crop + larger noise) for better contrastive learning.

## Recommendation

- Keep 20-event blocks, 20% masking for run 1
- Consider increasing masking to 30-40% if MEM loss converges too fast (reconstruction too easy)
- Implement asymmetric augmentation: View 1 (weak) = jitter(σ=0.05) + scale([0.9,1.1]); View 2 (strong) = temporal crop(±20 events) + jitter(σ=0.1)
- Exclude time reversal and segment permutation (empirical evidence of harm)
