# 2D Multi-Head Attention Trading Model — Design Spec

## Problem

Our current flat MLP model achieves 18/25 passing symbols on full-test Sortino evaluation. It flattens a (50, 31) observation window into a 1550-dim vector, destroying temporal ordering. The model can't efficiently learn sequential motifs like "volatility spiked then order flow reversed." Every temporal architecture we tried failed because of compute constraints (5-min CPU budget).

## Solution

Replace the flat MLP with a 2D multi-head attention model inspired by the Percepta "Can LLMs Be Computers?" paper. Key insight: head dimension 2 is both Turing complete and enables log-time lookups. Many tiny heads (32 × 2D) replace few large heads, keeping compute tractable while extracting temporal patterns.

Run on RunPod H100 80GB with Flash Attention, removing the compute constraint that blocked all previous temporal experiments.

## Architecture

```
Input: (batch=256, window=2000, features=31)
       2000 steps = ~3 hours of market history

  → Linear projection: 31 → 64 (d_model)
  → RoPE positional encoding (relative positions)

  → Attention Layer 1: 32 heads × 2D, Flash Attention, causal mask
    → Gated FFN: 64 → 128 → 64
    → Residual connection + LayerNorm
    Learns: raw temporal motifs ("where did vol spike?", "where did flow reverse?")

  → Attention Layer 2: 32 heads × 2D, Flash Attention, causal mask
    → Gated FFN: 64 → 128 → 64
    → Residual connection + LayerNorm
    Composes: "vol spike THEN flow reverse", "spread widen WHILE depth drop"

  → Attention Layer 3: 32 heads × 2D, Flash Attention, causal mask
    → Gated FFN: 64 → 128 → 64
    → Residual connection + LayerNorm
    Detects: regime-level patterns ("trending shifting to choppy")

  → Multi-Pool Aggregation:
    mean(64) + max(64) + last(64) + attn-weighted(64) = 256 dims

  → MLP Head: 256 → 512 → 512 → 3 (flat/long/short)
    ReLU, orthogonal init

  Ensemble: 10 seeds, logit sum
```

### Why Each Component

| Component | Rationale |
|-----------|-----------|
| **Window=2000** | See full funding cycles, session patterns, multi-hour trends. Matches the 800-step decision horizon with 2.5x lookback. |
| **Linear 31→64** | Project raw features into attention space. 64 = 32 heads × 2D. |
| **RoPE** | Relative positional encoding — "5 steps ago" matters more than "step #1847". Works with variable sequence lengths. |
| **32 heads × 2D** | Percepta's key insight: many tiny heads, each finds one pattern. 2D enables log-time lookup via convex hull queries. Sufficient for Turing completeness. |
| **Flash Attention** | Eliminates O(T^2) memory for attention matrix. Required for window=2000 to fit in 80GB. |
| **Causal mask** | Model only sees past, not future. Matches real-time trading. |
| **Gated FFN** | Same as Percepta paper: `gate, val = linear(x).chunk(2); out = relu(gate) * val`. More expressive than standard FFN. |
| **Residual + LayerNorm** | Standard transformer practice. Stabilizes training for deeper models. |
| **3 layers** | Layer 1: raw motifs. Layer 2: composed motifs. Layer 3: regime detection. |
| **Multi-pool** | mean=overall pattern, max=strongest signal, last=most recent state, attn-weighted=learned importance. Aggregates 2000 steps into 256 dims. |
| **10-seed ensemble** | More diversity with full training per seed. Budget allows it. |
| **No flat vector** | The bold bet — attention layers replace brute-force flattening. If temporal extraction works, flat is unnecessary overhead. |

### Model Size

- Attention params per layer: 4 × 64 × 64 = 16K
- FFN params per layer: 2 × 64 × 128 = 16K
- 3 layers: ~96K attention/FFN params
- Input projection: 31 × 64 = 2K
- Pooling attention: 64 params
- MLP head: 256×512 + 512×512 + 512×3 = ~395K
- **Total: ~0.5M parameters** (tiny model, big data)

## Training

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 200 | Enough to converge with larger model. Previously limited to 25-30. |
| Batch size | 256 | Smooth gradients. H100 has headroom. |
| Optimizer | AdamW | Proven. |
| Learning rate | 1e-3 | Proven baseline. May need tuning. |
| LR schedule | Cosine annealing | Failed at 30 epochs (too aggressive). At 200 epochs, gentle. |
| Weight decay | 5e-4 | Current v5 best. |
| Loss | Recency-weighted focal (gamma=1.0, decay=1.0) | Proven on v5. |
| Grad clipping | 1.0 | Proven. |
| Samples/symbol | 20,000 | More data — enough epochs to digest. Failed before at 25 epochs, fine at 200. |
| Mixed precision | FP16 autocast | 15x faster on H100 vs FP32. |
| Gradient checkpointing | Yes | Trades compute for memory. Required for window=2000. |
| Horizon | 800 | Matches min_hold. Proven alignment. |
| Min hold | 800 | Proven on v5. |
| Fee mult | 1.5 | Proven. |

## Evaluation

- Full test set (28K steps per symbol), no truncation
- Sortino ratio (downside vol only)
- 25 symbols, guardrails: >=10 trades, <=20% drawdown
- Correct annualization: steps_per_day = total_steps / 20 test days

## Infrastructure

| Resource | Spec |
|----------|------|
| GPU | RunPod H100 80GB |
| Runtime | PyTorch 2.x with Flash Attention 2 |
| Precision | FP16 mixed precision via `torch.cuda.amp` |
| Memory mgmt | Gradient checkpointing (`torch.utils.checkpoint`) |
| Cost | ~$3-4/hr, ~10 min per run = ~$0.60/run |
| Data transfer | rsync cached .npz files to RunPod (~2GB) |

### RunPod Setup

1. Spin up H100 80GB pod with PyTorch template
2. Install deps: `pip install torch numpy pandas gymnasium xgboost`
3. Transfer code: `rsync -avz ./ runpod:/workspace/autoresearch-trading/`
4. Transfer caches: `rsync -avz .cache/ runpod:/workspace/autoresearch-trading/.cache/`
5. Run: `python train.py`

## Success Criteria

| Metric | Current Best (flat MLP) | Target |
|--------|------------------------|--------|
| Symbols passing | 18/25 | 20+/25 |
| Sortino | 0.230 | >0.230 |
| Max drawdown | 0.367 | <0.30 |

If the 2D attention model fails to beat 18/25 after thorough tuning (epoch sweep, LR sweep, window size sweep), we fall back to the safe bet: add 2D attention alongside the flat MLP on `autoresearch/v5-features`.

## Fallback Plan

If the bold bet fails:
1. Return to `autoresearch/v5-features` branch (18/25, Sortino=0.230)
2. Add 2D attention output as extra features alongside flat+mean+std
3. This is zero-risk — the MLP can learn to ignore useless attention features
4. Run locally on M4 (no RunPod needed for the safe version)

## Files to Create/Modify

- `train.py` — replace DirectionClassifier with attention model, update training loop for FP16/checkpointing
- `prepare.py` — no changes (features and eval already correct)
- `requirements-gpu.txt` — new file for RunPod deps (flash-attn, etc.)
- `scripts/run_runpod.sh` — setup and launch script for RunPod

## What This Does NOT Include

- No changes to features (v5 is set)
- No changes to evaluation (Sortino + full test is set)
- No changes to labeling (horizon=800, fee_mult=1.5 is set)
- No cross-asset features (blocked by eval architecture)
- No online learning or adaptation (future work)
