# Research: Embedding Collapse Prevention

## Question
What are the best metrics and methods for detecting/preventing embedding collapse in self-supervised learning? Do our council's thresholds hold up?

## Sources
1. VICReg (Bardes et al., ICLR 2022, arXiv:2105.04906) — variance-invariance-covariance regularization
2. DirectCLR / Dimensional Collapse (Jing et al., ICLR 2022, arXiv:2110.09348) — analysis of collapse in SimCLR
3. Barlow Twins (Zbontar et al., ICML 2021, arXiv:2103.03230) — redundancy reduction
4. RankMe (Garrido et al., ICML 2023, arXiv:2210.02885) — effective rank metric predicts downstream accuracy
5. VCReg Theory (Mialon et al., 2024, arXiv:2209.14905) — provable pairwise independence
6. WERank (2024, arXiv:2402.09586) — weight-space regularization for weak augmentation domains
7. TS-TCC (Eldele et al., TPAMI 2023) — time series contrastive with augmentation asymmetry

## Key Findings

### Collapse Types
1. **Complete collapse:** All embeddings → constant vector. std → 0, loss → 0. Easy to detect.
2. **Dimensional collapse:** Embeddings span low-dimensional subspace. Loss looks normal. Much more dangerous. **Jing et al. showed this occurs even in SimCLR.**

### Detection Metrics

**Primary: RankMe effective rank** (ICML 2023)
```python
def effective_rank(embeddings):
    z = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-9]
    return torch.exp(-(p * p.log()).sum()).item()
```
Predicts downstream accuracy across hundreds of SSL runs. Label-free, no hyperparameters.

**Secondary metrics:**
- Per-dimension std (VICReg): any dim with std < 0.1 is dead
- Off-diagonal covariance norm
- Uniformity (Wang & Isola): log mean pairwise Gaussian similarity

**Critical: NT-Xent loss values alone are NOT reliable collapse indicators** (Jing et al.)

### Council Threshold Assessment

Council: effective rank > 20 at epoch 5, > 30 at epoch 10 (on 128-dim projector).

**These are kill switches, not health targets:**
- < 10: likely collapse, stop training
- 20-30: "not dead" — investigate
- **> 60: healthy training (the real target)**
- > 90: excellent
- **Trajectory matters more than absolute value** — flat/declining rank at 25 is worse than rank 40 and rising

### Prevention Methods

1. **VICReg variance term** (zero-cost add-on to NT-Xent):
```python
std_z = torch.sqrt(proj_embeddings.var(dim=0) + 1e-4)
loss_var = F.relu(1.0 - std_z).mean()  # weight: 1.0-5.0
```

2. **Nonlinear projection head** (already in our spec): buffers encoder from contrastive loss collapse
3. **Augmentation asymmetry** (TS-TCC): weak + strong views, up to 20pp difference vs symmetric
4. **WERank**: weight-space regularization for domains with weak augmentation diversity

### Elevated Risk in Our Setting

- Weak augmentation diversity (financial TS << image crops + color jitter) → higher dimensional collapse risk
- Symbol identity shortcut → encoder may collapse to symbol clusters
- Adjacent windows as false negatives → NT-Xent treats similar windows as negatives

## Relevance to Our Project

1. **Raise effective rank target to > 60**, not > 30
2. **Add VICReg variance term** (weight 1.0-5.0) alongside NT-Xent as collapse insurance
3. **Run symbol probe from epoch 2**, not epoch end
4. **Implement asymmetric augmentation** (weak + strong views)
5. **Monitor per-dim std every 100 batches** (zero cost, early warning)
6. **Do NOT rely on NT-Xent loss as health indicator**

## Recommendation

Add VICReg variance term with weight μ=2.0 to the training loss. This is one line of code and provides provable collapse prevention (Mialon et al. 2024) with zero downside. Combined with effective rank monitoring at target > 60, this gives both prevention and detection.
