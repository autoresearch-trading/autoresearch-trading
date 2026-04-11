# Research Synthesis: Pre-Implementation Literature Review

**Date:** 2026-04-10
**Researchers:** R1 (TS masking/augmentation), R2 (financial SSL/temperature), R3 (collapse prevention)

## Executive Summary

The literature validates the council's Round 5 recommendations and adds three new findings that should be incorporated:

### Confirmed by Literature

| Council Decision | Evidence |
|---|---|
| 20-event blocks (not 5) | SimMTM, TimeMAE, Nature ablation: point/small masking trivially solvable |
| 20% masking rate | Conservative vs literature's 40-60% but safe for first CNN run |
| τ=0.5→0.3 NT-Xent temperature | Kim & Kim (gradient vanishing at 0.1), CaTT (0.5 matches/beats 0.1 on TS), Chaudhry (low SNR → higher τ) |
| Augmentation noise σ=0.05/0.15 | Parametric Aug (ICLR 2024): σ=0.05-0.1 for financial |
| Exclude time reversal | ETH paper: empirically harmful for causal TS |
| Exclude carry-forward features from MEM | LOBERT: jointly masks LOB context around masked messages |
| MEM + contrastive combination | Each component has strong precedent; combination is novel |

### New Findings to Incorporate

1. **Asymmetric augmentation** (TS-TCC): Use weak view + strong view, not two identical augmentations. Up to 20pp improvement. Proposed: View 1 (weak) = jitter(σ=0.05) + scale([0.9,1.1]); View 2 (strong) = crop(±20 events) + jitter(σ=0.1).

2. **VICReg variance term** (ICLR 2022): Add `loss_var = F.relu(1.0 - std_per_dim).mean()` with weight μ=2.0 to the training loss. Zero-cost collapse insurance with provable guarantees. Our setting has elevated collapse risk due to weak augmentation diversity.

3. **Effective rank target > 60** (not > 30): Council thresholds were kill switches. Healthy training on 128-dim projector should target > 60. Monitor trajectory — flat/declining rank is a warning even above threshold.

### Key Reference Papers

- **LOBERT** (arXiv:2511.12563): Closest analogue — BERT-masked LOB messages, validates our MEM design
- **RankMe** (arXiv:2210.02885): Gold standard collapse metric
- **TS-TCC** (TPAMI 2023): Augmentation asymmetry critical for TS SSL
- **Kim & Kim** (arXiv:2501.17683): τ=0.1 gradient vanishing proof for [0.4, 0.7] similarity range

### Our Novelty

No paper combines MEM + SimCLR on order flow data. No paper uses DEX perpetual `is_open`/`is_close` features. Our approach has strong theoretical backing from individual components but the combination is untested.
