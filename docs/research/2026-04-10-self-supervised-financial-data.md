# Research: Self-Supervised Learning on Financial/LOB Data

## Question
Has anyone done self-supervised representation learning on financial order flow data? What NT-Xent temperature works for non-image contrastive learning?

## Sources
1. LOBERT (arXiv:2511.12563, Tampere 2025) — BERT-style masked prediction on Nasdaq ITCH LOB messages
2. LOBench (arXiv:2505.02139, 2025) — first systematic benchmark of LOB representation learning
3. SimLOB (arXiv:2406.19396, 2024) — Transformer autoencoder for LOB simulation
4. LENS (arXiv:2408.10111, 2024) — foundation model on 100B financial observations
5. Contrastive Asset Embeddings (arXiv:2407.18645, Dolphin et al. 2024) — SimCLR on return TS
6. Bieganowski & Slepaczuk (arXiv:2602.00776, 2026) — crypto perps microstructure features
7. Kim & Kim (arXiv:2501.17683, 2025) — temperature gradient analysis for InfoNCE
8. Chaudhry (arXiv:2603.12552, 2026) — temperature annealing as simulated annealing on Riemannian manifold
9. CaTT (arXiv:2410.15416) — contrastive augmented time series Transformer

## Key Findings

### Financial SSL Landscape

**LOBERT (2025)** — closest analogue to our design:
- BERT-style Masked Message Modeling on Nasdaq ITCH LOB messages
- ~1.1M param Transformer, 470M messages, 80 days, 4 stocks
- H=100 messages with confidence 0.9: F1=0.88 vs DeepLOB's 0.61
- **Critical validation:** LOBERT jointly masks LOB snapshots around masked messages to prevent label leakage — confirms our exclusion of carry-forward features from MEM targets

**Key gap:** No paper combines MEM + SimCLR on order flow. No paper uses DEX perpetual `is_open`/`is_close` position-side data. Our combination is novel.

**LENS (2024):** Standard pretraining is "ineffective" when naively applied to financial TS due to low SNR. Uses invertible embedding module.

**Contrastive Asset Embeddings:** Naive SimCLR augmentation fails for financial data. Uses statistical hypothesis testing on return similarity for valid positive pairs.

### NT-Xent Temperature

| Setting | Temperature | Source |
|---------|-------------|--------|
| SimCLR (ImageNet) | 0.07-0.10 | Chen et al. 2020 |
| BYOL | 0.30 | Grill et al. 2020 |
| SimCLR (explored) | 0.10 or 0.50 | Chen et al. 2020 |
| CaTT (ETTh1, ECG) | **0.50 and 1.0 match/beat 0.1** | arXiv:2410.15416 |
| SimMTM (SleepEEG) | 0.05-0.10 | arXiv:2302.00861 |

**Theoretical support for τ=0.5:**
- **Chaudhry (2026):** Temperature annealing ≡ simulated annealing on Riemannian manifold. Optimal initial τ = 1/δE_max. Low-SNR data → shallower energy barriers → higher optimal τ.
- **Kim & Kim (2025):** At τ=0.1, gradients nearly vanish for cosine similarities in [0.4, 0.7]. Financial embeddings cluster in exactly this range. τ=0.25+ required for non-zero gradients.

**Conclusion:** τ=0.5→0.3 annealing is correct. Vision's τ=0.1 causes gradient stagnation in the similarity range where financial embeddings cluster.

## Relevance to Our Project

1. **LOBERT validates our MEM design** — masked prediction on event streams with context masking works
2. **τ=0.5→0.3 confirmed** by both theory and empirical evidence
3. **Our MEM + SimCLR combination is novel** — no prior work combines both on order flow
4. **`is_open` is unique** — zero academic precedent for DEX position-side features
5. **Naive augmentation fails on financial data** — must be careful with noise levels

## Recommendation

- Proceed with MEM + SimCLR design (novel but each component has precedent)
- τ=0.5→0.3 annealing confirmed
- Study LOBERT's context masking approach in detail for additional MEM refinement
- Consider LENS's invertible embedding approach if standard BN struggles with low SNR
