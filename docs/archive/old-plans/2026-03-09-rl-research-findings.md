# RL for Trading — Deep Research Findings (March 2026)

## Key Takeaway

No single algorithm replaced PPO/SAC. Instead, the field advanced via **risk-aware variants**, **offline sequence modeling (Decision Transformers)**, and **adversarial robustness training**. All are practical on Apple Silicon.

## 1. Best Algorithms (2025-2026)

- **Risk-Aware PPO**: PPO with drawdown/volatility penalties in reward. Most robust for options/derivatives.
- **TD3/TD3+BC**: Continuous-action portfolio control with multi-objective rewards. Good for position sizing.
- **SAPPO**: Sentiment/context-weighted advantage updates — feed regime as multiplier into PPO advantage.
- **TimesNet + Actor-Critic**: Regime predictor feeding latent state to policy. Handles non-stationarity.
- **Bayesian Adversarial Training**: Train on GAN-generated synthetic market data for robustness. [Code](https://github.com/XiaHaochong98/Bayesian-Robust-Financial-Trading-with-Adversarial-Synthetic-Market-Data)

**Selection rule**: Offline-first (Decision Transformer / CQL / IQL) for extracting policies from historical data. Risk-aware PPO for autoresearch iteration loop.

## 2. Decision Transformers for Trading

- Reframe offline RL as conditional sequence modeling (predict action conditioned on desired returns)
- **Critic-Guided DT**: Use learned Q-value to reweight DT outputs. [Code](https://github.com/sharkwyf/critic-guided-decision-transformer)
- **LoRA fine-tuning**: Adapt pretrained sequence models with lightweight adapters for regime robustness
- DTs handle non-stationarity via return-to-go conditioning + explicit regime labels
- Repos: [kzl/decision-transformer](https://github.com/kzl/decision-transformer), [TroddenSpade DT](https://github.com/TroddenSpade/Decision-Transformer-on-Offline-Reinforcement-Learning)

## 3. State Representation (Orderbook/Microstructure)

**Recommended: CNN + Multi-Head Attention (DeepLOB style)**

```python
class OrderbookEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.fc = nn.Linear(64, hidden_dim)
```

- Multi-channel inputs: bid_price, ask_price, bid_vol, ask_vol, trade_sign, CVD, TFI, OFI, funding_rate
- Per-symbol z-score normalization with rolling mean/std
- Temporal encoding: relative time deltas + Fourier features for funding periodicity
- Ref: [DeepLOB guide](https://arxiv.org/html/2403.09267v4)

## 4. Reward Shaping

Best practical formula:
```
Reward = realized_PnL - λ_vol * rolling_std(PnL) - λ_draw * drawdown - tx_costs
```

- Use EMA for mean/variance (differentiable, less noisy)
- Tune λ weights via walk-forward validation
- Consider CVaR/tail-loss penalties for crypto's heavy tails
- [RF-Agent](https://arxiv.org/html/2602.23876v1) for automated reward function discovery

## 5. Overfitting Prevention

- **Walk-forward validation**: Rolling train/val/test windows, retrain each fold
- **Regime-aware splits**: Ensure different regimes in train vs test
- **Adversarial synthetic data**: GAN-generated extreme scenarios. [Code](https://github.com/XiaHaochong98/Bayesian-Robust-Financial-Trading-with-Adversarial-Synthetic-Market-Data)
- **Conservative offline RL** (CQL/IQL): Penalizes over-estimation in unseen states
- **Policy ensembles**: Average across multiple trained policies
- **Risk metric gating**: Only keep models passing Sharpe + drawdown thresholds

## 6. Multi-Asset RL

- **Single-agent portfolio RL**: Output portfolio weights via softmax, continuous actions (DDPG/TD3)
- **Multi-agent**: One agent per asset + portfolio manager aggregator
- **Shared encoder**: Share LOB/microstructure encoder across assets, per-asset embeddings to centralized actor
- Proportional sampling by liquidity (so BTC doesn't dominate)

## 7. Apple Silicon Compatibility

- **PyTorch MPS**: Works but incomplete operator coverage. Some ops fail on MPS.
- **Practical approach**: CPU for environment loop, MPS for batched network forward/backward
- **CleanRL** recommended over SB3 for MPS — simpler, easier to add CPU fallbacks
- **MLX**: Great for inference, not yet integrated with RL libraries
- For heavy training: consider cloud GPU, use Mac for rapid iteration

## 8. Key Repos to Clone

| Repo | Purpose |
|------|---------|
| [kzl/decision-transformer](https://github.com/kzl/decision-transformer) | Offline RL baseline |
| [critic-guided-decision-transformer](https://github.com/sharkwyf/critic-guided-decision-transformer) | DT with Q-value rescoring |
| [Bayesian-Robust-Trading](https://github.com/XiaHaochong98/Bayesian-Robust-Financial-Trading-with-Adversarial-Synthetic-Market-Data) | Adversarial robustness |
| [RLTrader](https://github.com/notadamking/RLTrader) | Crypto RL environment |
| [FinRL Contest 2025](https://github.com/Open-Finance-Lab/FinRL_Contest_2025) | Walk-forward templates |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Minimal RL loops (MPS-friendly) |
