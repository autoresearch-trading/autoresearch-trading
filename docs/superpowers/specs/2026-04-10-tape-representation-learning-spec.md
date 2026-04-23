# Tape Representation Learning: Specification

## Goal

Train a self-supervised model on 40GB of raw DEX perpetual futures trade data (160 days, 25 symbols, millions of order events) to learn meaningful representations of tape states — the way a human tape reader develops intuition from watching the flow. The model observes millions of order events and learns to distinguish accumulation from distribution, absorption from breakout, climax from drift.

Direction prediction is a **downstream probing task**, not the primary objective. The model first learns to see; then we test whether what it sees is useful.

## Why This Is Different

**Previous approach (supervised):**
```
Raw trades → order events → 17 features → supervised CNN → predict direction at 52%
Constraint: noisy binary labels, marginal signal, multiple testing kills significance
```

**Representation learning approach:**
```
Raw trades → order events → 17 features → self-supervised CNN → learned tape representations
                                                                → probe: can it predict direction?
                                                                → probe: can it detect Wyckoff states?
                                                                → probe: are representations universal?
```

The model decides what matters. 40GB of raw trades is massive for representation learning — more market observation than any human tape reader could process in a lifetime. The self-supervised objective extracts signal from every event, not just from noisy direction labels.

## Raw Trade Schema

Each trade from Pacifica:
```
ts_ms:       1775044557458        (timestamp in ms)
price:       68686.0              (execution price)
qty:         0.0019               (size)
side:        open_long            (open_long/close_long/open_short/close_short)
cause:       normal               (normal/market_liquidation/backstop_liquidation — from Apr 1)
event_type:  fulfill_taker        (fulfill_taker/fulfill_maker — from Apr 1)
```

## Order Event Grouping

**Trades with the same timestamp are fragments of one order being filled across price levels.** Group them before feeding to the model.

**Dedup required before grouping:**
- **Pre-April data:** `df.drop_duplicates(subset=['ts_ms', 'qty', 'price'], keep='first')` — WITHOUT `side`
- **April+ data:** Filter to `event_type == 'fulfill_taker'`

**Validation required (Step 0):** Verify same-timestamp = same-order assumption. Step 0 measured **3-16% mixed-side events** across 25 symbols after correct no-side dedup (median ~3%; liquid symbols BTC/ETH/SOL toward upper end). Raw data before dedup shows ~99% cross-side pairs, confirming the API records both counterparties per fill — no-side dedup collapses these correctly. The earlier "59%" figure in drafts was an artifact of measuring on undeduped or dedup-with-side data.

## Input Representation (17 features per order event)

**From trade events (9 features):**
```
1.  log_return:         log(vwap / prev_event_vwap)
2.  log_total_qty:      log(total_qty / rolling_median_event_qty)  — rolling 1000-event median, causal
3.  is_open:            fraction of fills that are opens [0,1]
4.  time_delta:         log(ts - prev_event_ts + 1)
5.  num_fills:          log(fill count)
6.  book_walk:          abs(last_fill - first_fill) / max(spread, 1e-8 * mid)
7.  effort_vs_result:   clip(log_total_qty - log(|return| + 1e-6), -5, 5)
8.  climax_score:       clip(min(qty_zscore, return_zscore), 0, 5)  — rolling 1000-event σ
9.  prev_seq_time_span: log(last_ts - first_ts + 1) for PREVIOUS 200-event window
```

**From orderbook (8 features, aligned by nearest prior snapshot, ~24s cadence, 10 levels):**
```
10. log_spread:          log((best_ask - best_bid) / mid + 1e-10)
11. imbalance_L1:        (bid_notional - ask_notional) / total_notional at L1
12. imbalance_L5:        inverse-level-weighted notional imbalance L1:5
13. depth_ratio:         log(max(bid_notional, 1e-6) / max(ask_notional, 1e-6))
14. trade_vs_mid:        clip((vwap - mid) / max(spread, 1e-8 * mid), -5, 5)
15. delta_imbalance_L1:  change since previous snapshot, carry-forward
16. kyle_lambda:         per-SNAPSHOT Cov/Var over 50 snapshots (~20 min), forward-filled
17. cum_ofi_5:           piecewise Cont 2014 OFI over 5 snapshots (~120s), normalized
```

**Three load-bearing features for tape reading (Wyckoff):**
- `effort_vs_result` — the master signal: absorption (high) vs ease-of-movement (low)
- `climax_score` — phase transition markers (buying/selling climax)
- `is_open` — the DEX-specific Composite Operator footprint (no equivalent in traditional markets)

## Tape States the Model Should Learn

### Wyckoff Patterns (Council-4)

| State | Key Feature Signature | Duration (events) | Within 200-event window? |
|-------|----------------------|-------------------|--------------------------|
| **Absorption** | effort_vs_result > 1.5 sustained, flat log_return, high volume | 50-300 | Yes |
| **Buying Climax** | climax_score > 2.5, positive log_return spike, high is_open | 1-20 | Yes |
| **Selling Climax** | climax_score > 2.5, negative spike, high effort_vs_result | 1-20 | Yes |
| **Markup** | Low effort_vs_result, positive log_return, expanding is_open | 50-500+ | Partial |
| **Markdown** | Mirror of markup, negative log_return | 50-500+ | Partial |
| **Spring + Test** | Negative spike + high effort_vs_result + is_open spike + recovery + declining volume retest | 30-200 | Yes |
| **Upthrust** | Mirror of spring, positive spike + is_open short | 30-200 | Yes |
| **Shakeout** | Rapid negative spike, moderate volume, immediate recovery | 5-30 | Yes |
| **Accumulation** | Sustained absorption + oscillating range + elevated is_open + low kyle_lambda | 2,000-50,000 | No — local signature only |
| **Distribution** | Mirror of accumulation, is_open short-side | 2,000-50,000 | No — local signature only |

### Microstructure Regimes (Council-2 Cont)

| Regime | Feature Signature |
|--------|-------------------|
| **Informed flow** | kyle_lambda elevated, cum_ofi_5 persistent, is_open elevated, effort_vs_result low |
| **Noise flow** | kyle_lambda ≈ 0, cum_ofi_5 oscillating, effort_vs_result high |
| **Liquidity withdrawal** | delta_imbalance_L1 directional, log_spread widening, depth_ratio extreme |
| **Momentum** | log_return consistent sign, cum_ofi_5 aligned, effort_vs_result low |
| **Stress** | log_spread extreme, depth_ratio extreme, climax_score high, time_delta low |

### Self-Labels from Features (no human annotation)

Computable Wyckoff labels for evaluation and contrastive pair construction:

```python
# Absorption — sustained high effort, narrow price range, high volume
is_absorption = (
    mean(effort_vs_result[-100:]) > 1.5
    and std(log_return[-100:]) < 0.5 * rolling_std
    and mean(log_total_qty[-100:]) > 0.5
)

# Buying Climax — extreme climax + positive spike + prior uptrend
is_buying_climax = (
    max(climax_score[-10:]) > 2.5
    and log_return_at_climax > 2 * rolling_std
    and mean(log_return[-50:-10]) > 0
)

# Spring — downside probe + absorption at low + is_open spike + recovery
is_spring = (
    min(log_return[-50:]) < -2 * rolling_std
    and effort_vs_result_at_min > 1.0
    and is_open_at_min > 0.5
    and mean(log_return[-10:]) > 0
)

# Informed Flow — elevated lambda + persistent directional OFI
is_informed = (
    kyle_lambda > rolling_75th_pct
    and abs(cum_ofi_5) > rolling_50th_pct
    and cum_ofi_5_sign_consistent_over_3_snapshots
)

# Stress — extreme spread + extreme book asymmetry
is_stressed = (
    log_spread > rolling_90th_pct
    and abs(depth_ratio) > rolling_90th_pct
)
```

All thresholds are rolling per-symbol (causal, no lookahead).

## Sequence Length

**200 order events** (not raw trades).

Measured window duration at stride-50 event rate (Step 0, pre-April data):
- BTC: ~5 min (median inter-event gap 1.5s; earlier "~10 min" spec figure was off by 2×)
- ETH: ~6 min
- SOL/HYPE: ~10 min
- Illiquid alts (2Z, CRV, LDO, UNI): 56–68 min per 200-event window

Sufficient for local tape patterns (springs, climaxes, absorption episodes). NOT sufficient for full Wyckoff cycles (accumulation → markup requires 10K+ events). Phase-level inference is future work via a hierarchical architecture (sequence of local embeddings).

## Architecture

### Self-Supervised Encoder (~400K params)

```
Input: (batch, 200, 17)

BatchNorm1d(17)                              — normalize input features
Conv1d(17 → 64, kernel=5, dilation=1)        — local patterns (RF=5)
LayerNorm + ReLU + Dropout(0.1)
Conv1d(64 → 128, kernel=5, dilation=2)       — RF=13
LayerNorm + ReLU + Dropout(0.1)
Conv1d(128 → 128, kernel=5, dilation=4)      — RF=29        + residual
LayerNorm + ReLU
Conv1d(128 → 128, kernel=5, dilation=8)      — RF=61        + residual
LayerNorm + ReLU
Conv1d(128 → 128, kernel=5, dilation=16)     — RF=125       + residual
LayerNorm + ReLU
Conv1d(128 → 128, kernel=5, dilation=32)     — RF=253       + residual
LayerNorm + ReLU

Per-position output: (batch, 200, 128)
Global embedding: concat[GlobalAvgPool(128), last_position(128)] → (batch, 256)

~400K parameters
```

### Pretraining Heads (discarded after pretraining)

```
MEM Decoder:    Linear(128 → 17) at masked positions → MSE loss (14 of 17 features)
Projection:     Linear(256 → 256) + ReLU + Linear(256 → 128) → L2-norm → NT-Xent loss
```

### Fine-tuning Heads (added after pretraining)

```
Linear(256 → 64) + ReLU → [Linear(64 → 1)] × 4, sigmoid  — per-horizon direction
```

### Why 400K (up from 91K)

The constraint that capped the supervised model at 91K was overfitting to noisy binary labels. Self-supervised pretraining on 40GB does not have this constraint — the reconstruction and contrastive objectives are self-consistent. Measured total windows (stride=50, pre-April, 25 symbols, 161 days): **627K** — not the ~3.5M figure earlier in this spec, which assumed ~28K events/day across all symbols and did not account for much lower event rates on illiquid alts. BTC alone contributes ~133K windows; all other symbols average ~2K/symbol. At 627K windows and 400K params, the data-to-params ratio is **~1:1.6** — still workable for self-supervised learning but tight enough that model size is a live question for the Step 3 pretraining plan (council-6 review pending).

**Hard cap: do not exceed 500K params** without clearing all evaluation gates first.

## Training Strategy

### Pretraining Objective

**Masked Event Modeling (MEM) — weight annealed 0.90→0.60 over 20 epochs:**
- Mask **20% of events** using **block masking** (consecutive **20-event blocks**; 4 blocks per 200-event window) — per `docs/knowledge/decisions/mem-block-size-20.md` (2026-04-10 council round 5). 5-event blocks were bridgeable at layer 3 (dilation=4) and taught no tape structure.
- **Mask-first-then-encode flow (mandatory):** BatchNorm the full input → zero-fill masked positions in BN-normalized space (= training mean) → encode the MASKED input → decode at masked positions → MSE against BN-normalized original. Encoding the unmasked input defeats MEM entirely (the decoder trivially copies the ground-truth features the encoder already saw).
- Reconstruct in **BatchNorm-normalized** feature space (not raw values) — BN runs BEFORE masking to keep running statistics clean (BN-after-masking contaminates stats with 20% artificial zeros).
- **Exclude 3 carry-forward features** from reconstruction targets: `delta_imbalance_L1` (90% zero), `kyle_lambda` (forward-filled), `cum_ofi_5` (forward-filled) — trivially predictable from adjacent events via copy
- Reconstruct only 14 features: log_return, log_total_qty, is_open, time_delta, num_fills, book_walk, effort_vs_result, climax_score, prev_seq_time_span, log_spread, imbalance_L1, imbalance_L5, depth_ratio, trade_vs_mid

**SimCLR Contrastive — weight annealed 0.10→0.40 over 20 epochs:**
- Generate 2 augmented views of each window
- NT-Xent loss on L2-normalized global embeddings
- **Temperature τ=0.5 anneal to τ=0.3 by epoch 10** (then hold constant) — per `docs/knowledge/decisions/ntxent-temperature.md`. ImageNet default τ=0.1 is explicitly rejected: too cold, pushes genuinely similar market states apart using spurious features (symbol identity, time-of-day). Schedule: `tau = max(0.3, 0.5 - epoch * 0.02)` for epochs 1–10, then constant 0.3.
- Augmentations that preserve market meaning:
  - Window start jitter: **±25 events** (strengthened 2026-04-15 per council-6 — crosses BTC session micro-boundaries; shifts illiquid-alt window centers by ~10 min)
  - Additive Gaussian noise: σ = 0.02 × feature_std (continuous features only)
  - **Timing-feature noise**: σ = 0.10 Gaussian noise on `time_delta` and `prev_seq_time_span` during view generation (5× baseline). Forces encoder to rely on relative rhythms, not absolute session-indicative magnitudes.
  - Feature dropout: p=0.05 per feature per event (zero to BatchNorm mean)
  - Time scale dilation: multiply time_delta by factor in [0.8, 1.2]
- Augmentations that DESTROY meaning (do NOT use):
  - Time reversal (breaks causality)
  - Event shuffling (destroys sequence order)
  - Large noise > 0.1 std
- **Cross-symbol positive pairs:** same-date, same-hour windows from different liquid symbols (BTC, ETH, SOL, BNB, LINK, LTC) as soft positives (weight 0.5). **AVAX is the held-out symbol for Gate 3 and MUST NOT appear in contrastive pairs during pretraining** — its inclusion in an earlier draft was a spec bug that would have contaminated Gate 3. LTC substitutes for AVAX in the 6-symbol anchor set (liquid, non-held-out, distinct regime from memecoins).
- Batch: 256 windows → 512 augmented views → 256 positive pairs, 65K negative pairs

**Direction labels: NOT used during pretraining.** Pure self-supervised.

### Data Loading

- All pre-April data for pretraining (Oct 2025 – Mar 2026)
- **Stride = 50** during pretraining (4× data vs. stride=200, no leakage with walk-forward embargo)
- **Stride = 200** for evaluation probes
- April 1-13 reserved for probe evaluation only
- April 14+ untouched (final evaluation)
- Do NOT construct windows crossing day boundaries
- Shuffle across symbols and dates each epoch
- **Equal-symbol sampling** per epoch to prevent BTC dominance

### Pretraining Hyperparameters

- Optimizer: AdamW(weight_decay=1e-4)
- Learning rate: OneCycleLR(max_lr=1e-3, pct_start=0.2) — higher than supervised (smoother loss surface)
- Batch size: 256 (512 views for contrastive)
- Epochs: 20-40 (monitor probe accuracy every 5 epochs)
- Stopping: if MEM loss improves < 1% over last 20% of epochs, stop
- **Gradient clipping `max_norm=1.0`** — primary anti-collapse mechanism for the projection head (bf16 + high-τ early phase can spike grads 10–100×).
- **Mixed precision: bf16** via `torch.autocast(dtype=torch.bfloat16)` — ~1.8× throughput on H100 with no accuracy cost at this model size.
- **`torch.compile(encoder, mode="reduce-overhead")`** — dilated CNN with static shapes (B=256, T=200, F=17) gets significant kernel-fusion gains. Apply to encoder only, not MEM decoder (shapes vary with masked-position count).
- **Compute cap: 1 H100-day (24 GPU-hours) before evaluation gates must be run**

### Monitoring During Pretraining

- MEM reconstruction MSE (should decrease); per-feature MSE breakouts for `log_return` and `effort_vs_result` (highest-variance features — flag underfitting if > 0.9× baseline variance at epoch 2)
- NT-Xent contrastive loss (should decrease, watch for collapse)
- **Embedding collapse detector:** flag if per-batch embedding std < **0.05** (NOT 1e-4 — at 256 dims std 1e-3 is already functionally collapsed for a 128-class probe). Monitor every step.
- **Effective rank of the 256×B embedding matrix** (count singular values above 1% of max): flag < 20 at epoch 5, < 30 at epoch 10 as collapse early warning.
- Direction probe accuracy on April 1-13 every 5 epochs (early stopping signal)
- **Hour-of-day probe** (24-class LR on frozen embeddings) every 5 epochs. If accuracy exceeds 10% or stratified cross-session variance exceeds 1.5pp, flag as session-of-day shortcut — early warning before Gate 1.

### Fine-Tuning (after Gate 1 passes)

- Add 4 direction heads: `Linear(256→64) + ReLU → [Linear(64→1)] × 4`
- **Freeze encoder for 5 epochs** (linear probe only)
- **Unfreeze at lr = 5e-5** (10× lower than pretraining lr)
- Label smoothing: ε = 0.10/0.08/0.05/0.05 for 10/50/100/500 event horizons
- Loss weights: 0.10/0.20/0.50/0.20 (100-event primary, 500-event de-weighted per council recommendation)
- Walk-forward validation with 600-event embargo

## Evaluation Protocol

### Pre-Registration (before any data is touched)

- **Pretraining objective:** MEM (block masking) + SimCLR contrastive
- **Probing task:** logistic regression at 100-event horizon, April 1-13
- **Success threshold:** > 51.4% linear probe accuracy on 15+/25 symbols
- **Held-out symbol:** AVAX (pre-designated, irrevocable)
- **Model size cap:** 500K params
- **Compute budget:** 1 H100-day before gates

### Gate 0: Flat-Feature Baseline Grid (before pretraining)

Compute four baselines over the same walk-forward folds (3-fold, 600-event embargo, min_train=2000, min_test=500) at H10/H50/H100/H500:

1. **PCA(n=20) + LogisticRegression** on 85-dim flat features (mean/std/skew/kurt/last per channel).
2. **Random Projection (85→20, frozen) + LogisticRegression** — adaptive-structure control.
3. **Majority-class predictor** (training-fold majority) — the true noise floor.
4. **Shuffled-labels PCA+LR** — null-hypothesis pipeline check; must stay within ±0.005 of 0.500.

**Metric: balanced accuracy at ALL horizons** (council-1 + council-5 findings, 2026-04-15 — raw accuracy is gameable via per-fold label imbalance; up to 10pp inflation observed on illiquid symbols at H10).

**Gate 0 passes** iff (a) shuffled-labels control ≈ 0.500, (b) per-symbol per-horizon tables for all four baselines published. Gate 0 itself is NOT a threshold-gate — it establishes the noise floor against which Gate 1 is measured.

See `docs/experiments/gate0-summary.md` for the 2026-04-15 baseline run.

### Session-of-Day Confound Check (pre-pretraining)

Before launching pretraining, run an LR probe on a single hour-of-day feature (4-hour bins, one-hot) against the same walk-forward folds and labels as Gate 0. If this single-feature model exceeds PCA+LR on the 85-dim flat features by > 0.5pp balanced accuracy on ≥ 5 symbols, the `_last` statistic block in `tape/flat_features.py` (particularly `time_delta_last`, `prev_seq_time_span_last`) is leaking session-of-day — prune it before training. Cost: < 5 minutes.

### Gate 1: Linear Probe on Frozen Embeddings (after pretraining)

Train logistic regression (C ∈ {0.001, 0.01, 0.1}) on frozen 256-dim pretrained embeddings. Evaluate on **April 1–13 held-out period** at H100, balanced accuracy per symbol.

**All four conditions MUST hold (binding stop-gates, revised 2026-04-15 per council-1/5/6):**

1. Balanced accuracy ≥ 51.4% on 15+/25 symbols (absolute sanity floor).
2. Balanced accuracy > Majority-class baseline + **1.0pp** on 15+/25 symbols (strengthened from +0.5pp over PCA).
3. Balanced accuracy > Random-Projection control + **1.0pp** on 15+/25 symbols (adaptive-structure test).
4. Hour-of-day 24-class probe on the same frozen embeddings < **10%** accuracy AND stratified accuracy variance < **1.5pp** across UTC sessions (Asia 0–8 / Europe 8–16 / US 16–24). Catches session-of-day shortcuts.

Plus existing diagnostics:
- Symbol-identity probe < 20%
- CKA > 0.7 between seed-varied runs
- Per-fold balanced-accuracy standard deviation reported alongside means

**STOP if any of 1–4 fail, or if any representation diagnostic fails.** The encoder has not learned tape microstructure — likely learned session-of-day or class imbalance.

### Gate 2: Fine-Tuned vs Supervised Baseline (after fine-tuning)

Fine-tuned CNN (pretrained encoder + direction heads) must exceed logistic regression on flat (200×17=3400) features by ≥ 0.5pp at primary horizon on 15+ symbols.

**STOP if fine-tuning does not beat the linear baseline.** Pretraining added nothing.

### Gate 3: Cross-Symbol Transfer (universality)

AVAX excluded entirely from pretraining. After fine-tuning, evaluate on AVAX.

**STOP if AVAX accuracy < 51.4% at 100-event horizon.** Model learned symbol-specific patterns, not universal microstructure.

### Gate 4: Temporal Stability

Evaluate probe accuracy on training months 1-4 vs months 5-6 separately.

**STOP if accuracy drops > 3pp between periods on > 10/25 symbols.** Representations memorized regime-specific noise.

**Metric: balanced accuracy at ALL horizons** (H10, H50, H100, H500). Revised 2026-04-15 per council-1 / council-5 Gate 0 review — raw accuracy is gameable via per-fold label imbalance at every horizon, not just H500. Measured Gate 0 H10 inflation up to 9.9pp on illiquid symbols (2Z: raw 0.621 → balanced 0.522). Symmetric use of balanced accuracy eliminates the ambiguity.

### Representation Quality Metrics (not go/no-go, but diagnostic)

**Symbol identity probe:** Linear classifier on frozen embeddings predicting symbol (1 of 25). Target: < 20% accuracy. If symbol is decodable, representations are not universal.

**CKA stability:** Train two models with different random seeds. Centered Kernel Alignment between representation spaces must be > 0.7. If < 0.7, representations are noise-fitted.

**Wyckoff label probes:** Linear probes on frozen embeddings for self-labeled tape states:
- Absorption detection (binary)
- Climax detection (binary)
- Informed flow detection (binary)
- Stress detection (binary)

**Cluster analysis:** k-means (k=8-16) on embeddings. Color by Wyckoff labels, symbol, time-of-day. Expect: clusters correspond to market states, NOT to symbols or clock time.

**Embedding trajectory:** For consecutive windows from same symbol-day, plot embedding distance over time. Expect slow drift during stable regimes, sharp jumps at genuine state transitions.

### Phase 2: Trading Performance (conditional on Gates 1-4)

Only after representation quality is validated:
- Add fee model and position sizing
- Evaluate Sortino across ALL symbols
- Compare to prior baseline (v11: Sortino=0.353 on 9/23 symbols)
- Compute Deflated Sharpe Ratio

## Implementation Plan

### Step 0: Label Validation + Data Validation (local, 15 min)
Same as prior spec — validate base rate, same-timestamp assumption, dedup rates, OB cadence.
Additionally: compute Wyckoff self-labels for all training data.

### Step 1: Data Pipeline (local CPU)
Build PyTorch Dataset: raw trades → dedup → group → align OB → 17 features → cache .npz.
Same pipeline as prior spec. Add stride=50 option for pretraining.

### Step 2: Baselines — Gate 0 (local, 30 min)
- PCA (n=50) + logistic regression on flat features
- Random encoder + linear probe
- Record reference numbers

### Step 3: Pretraining (RunPod H100, ~12 hours)
- MEM + contrastive on all pre-April data
- ~400K param encoder
- Monitor MEM loss, contrastive loss, embedding collapse
- Probe every 5 epochs on April 1-13

### Step 4: Evaluation — Gates 1-4 (local, 2 hours)
- Linear probe on frozen embeddings
- Symbol probe, CKA, cluster analysis
- Wyckoff label probes
- Go/no-go decision

### Step 5: Fine-Tuning (conditional, RunPod H100, ~4 hours)
- Freeze → unfreeze protocol
- Walk-forward evaluation
- Per-symbol accuracy, temporal stability

### Step 6: Interpretation (conditional on Gates 1-4)
- Feature attribution per Wyckoff state
- Embedding trajectory during known events
- Discrete codebook: k-means (k=128) → "market state vocabulary"
- Cross-symbol regime correspondence

## Future Work

### Hierarchical Architecture (Level 2)

200 events covers ~10 minutes — sufficient for local patterns but not full Wyckoff cycles. The natural evolution:

```
Level 1: 200-event CNN → 256-dim local embedding (this spec)
Level 2: Sequence of 50 Level-1 embeddings → Transformer/LSTM → phase-level embedding
         50 windows × 200 events = 10,000 events ≈ 8.5 hours
```

Level 2 would enable: accumulation/distribution phase detection, multi-hour regime tracking, position-level trade management. Build only after Level 1 representations prove useful (Gates 1-4 pass).

### Liquidation Cascade Detection

April+ data has `cause` field (market_liquidation, backstop_liquidation). A specialized fine-tuning head for cascade detection could exploit this labeled subset. Requires: accumulate more April data.

## Compute Requirements

| Step | Hardware | Time |
|------|----------|------|
| Label + data validation | Local CPU | ~15 min |
| Data pipeline | Local CPU | ~3-5 hours |
| Baselines (Gate 0) | Local CPU | ~30 min |
| Pretraining | RunPod H100 | ~12 hours |
| Evaluation (Gates 1-4) | Local CPU | ~2 hours |
| Fine-tuning | RunPod H100 | ~4 hours |
| Interpretation | Local CPU | ~2 hours |

## Data

- **25 symbols**: 2Z, AAVE, ASTER, AVAX, BNB, BTC, CRV, DOGE, ENA, ETH, FARTCOIN, HYPE, KBONK, KPEPE, LDO, LINK, LTC, PENGU, PUMP, SOL, SUI, UNI, WLFI, XPL, XRP
- **Date range**: 2025-10-16 to 2026-03-25 (~160 days)
- **April hold-out**: April 14+ untouched. April 1-13 for probe evaluation.
- **Held-out symbol**: AVAX (pre-designated, excluded from pretraining)
- **Sync**: `rclone sync r2:pacifica-trading-data ./data/ --transfers 32 --checkers 64 --size-only`

## Gotchas

1. **R2 fake timestamps**: Use `--size-only` with rclone
2. **OB alignment**: `np.searchsorted(ob_ts, trade_ts, side="right") - 1` — vectorized
3. **Order event grouping**: dedup FIRST, then group. Pre-April: `(ts_ms, qty, price)` without `side`
4. **Rolling medians**: never global statistics (lookahead). Rolling 1000-event, causal.
5. **effort_vs_result epsilon**: 1e-6 (not 1e-4). Clip [-5, 5]. Uses median-normalized log_total_qty.
6. **climax_score σ**: rolling 1000-event σ, not global.
7. **MEM reconstruction targets**: exclude delta_imbalance_L1, kyle_lambda, cum_ofi_5 (trivial copy from neighbors)
8. **MEM reconstruction space**: compute loss in BatchNorm-normalized space, not raw feature space
8a. **MEM flow order (CRITICAL)**: BN full input → zero-fill masked positions in BN-normalized space → encode MASKED input → decode → MSE vs BN-normalized original. Encoding the UNMASKED input and applying the mask only to the loss is a silent bug — the decoder trivially copies the ground truth (knowledge/concepts/mem-pretraining.md §"Critical: Mask Token Replacement Order").
8b. **MEM block size = 20, masking rate = 20%**: 5-event blocks are bridged by layer-1 convolutions (k=5, d=1) from p-1 and p+5 — no tape structure learned (knowledge/decisions/mem-block-size-20.md).
8c. **NT-Xent temperature τ=0.5→0.3 by epoch 10, NOT 0.1**: ImageNet default τ=0.1 is too cold for financial data, drives collapse via spurious-feature learning (knowledge/decisions/ntxent-temperature.md).
9. **Embedding collapse**: flag per-batch embedding std < **0.05** (not 1e-4). Also monitor **effective rank** of 256×B embedding matrix — flag < 20 at epoch 5 or < 30 at epoch 10 (knowledge/concepts/contrastive-learning.md).
10. **Cross-symbol contrastive pairs**: only for liquid symbols (BTC, ETH, SOL, BNB, LINK, LTC). **AVAX excluded — it is the Gate 3 held-out symbol.** Do NOT force invariance with memecoins.
11. **Day boundaries**: do not construct windows crossing day boundaries
12. **Symbol sampling**: equal-symbol sampling per epoch to prevent BTC dominance
13. **April hold-out**: April 14+ untouched — do not view, even informally
14. **BatchNorm at inference**: `model.eval()` for entire evaluation
15. **Dedup key**: `(ts_ms, qty, price)` without `side`
16. **OB cadence**: ~24s, not ~3s. 10 levels, not 25.
17. **depth_ratio log(0)**: epsilon guard required for one-sided books
18. **trade_vs_mid div-by-zero**: guard with max(spread, 1e-8*mid), clip [-5, 5]
19. **kyle_lambda**: per-SNAPSHOT, not per-event. Uses Δmid, not Δvwap.
20. **cum_ofi_5**: piecewise Cont 2014 formula (naive delta has wrong sign during trends)
21. **Stride**: 50 for pretraining, 200 for evaluation probes
22. **Pre-warm rolling windows**: from prior day end. Mask first 1000 events of first calendar day only.
