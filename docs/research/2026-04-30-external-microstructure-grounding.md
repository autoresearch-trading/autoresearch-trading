# External grounding for the full-fidelity Pacifica plan

Date: 2026-04-30

## Honest answer

The first brainstorm was mostly grounded in this repo's evidence and in market-microstructure reasoning, not in a fresh external state-of-the-art literature scan. It was directionally aligned with the literature, but it should not have been presented as externally grounded.

This note adds the external grounding and changes the priority order where the literature argues for it.

## What the external literature supports

### 1. Order-flow imbalance and queue imbalance are core baselines

External support:

- Cont, Kukanov, Stoikov, "The Price Impact of Order Book Events"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822
- Lipton, Pesavento, Sotiropoulos, "Trade Arrival Dynamics and Quote Imbalance in a Limit Order Book"  
  https://arxiv.org/abs/1312.0514
- Kolm, Turiel, Westray, "Deep Order Flow Imbalance"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141
- Stoikov, "The Micro-Price"  
  https://ssrn.com/abstract=2970694

Implication for this project:

The full-fidelity pipeline should build OFI, queue imbalance, microprice, signed trade imbalance, spread/depth state, and short-horizon price-impact features before any deep model. These should be treated as baseline features and ablation controls.

The plan remains aligned, but it needs a stronger "rank 0" requirement: build execution/adverse-selection measurement and OFI/microprice features before testing strategy ideas.

### 2. Deep LOB models are useful, but not proof of tradable alpha

External support:

- Zhang, Zohren, Roberts, "DeepLOB"  
  https://arxiv.org/abs/1808.03668
- Sirignano and Cont, "Universal Features of Price Formation in Financial Markets"  
  https://www.pnas.org/doi/10.1073/pnas.1807626116
- Tsantekidis et al., "Forecasting Stock Prices from the Limit Order Book Using Convolutional Neural Networks"  
  https://ieeexplore.ieee.org/document/8010701
- Kolm, Turiel, Westray, "Deep Order Flow Imbalance"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141

Implication for this project:

The prior decision to avoid another generic encoder/RL/direction model is externally supported. LOB ML papers often report predictive/classification improvements, not audited post-fee, queue-aware PnL. For this repo, deep models should only come after simple OFI/queue/microprice baselines show post-cost value.

### 3. Market making is fundamentally adverse-selection constrained

External support:

- Glosten and Milgrom, "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders"  
  https://www.sciencedirect.com/science/article/abs/pii/0304405X85900444
- Kyle, "Continuous Auctions and Insider Trading"  
  https://www.jstor.org/stable/1913210
- Avellaneda and Stoikov, "High-frequency Trading in a Limit Order Book"  
  https://www.researchgate.net/publication/24086205_High-frequency_trading_in_a_limit_order_book
- Cartea, Jaimungal, Penalva, "Algorithmic and High-Frequency Trading"  
  https://www.cambridge.org/core/books/algorithmic-and-highfrequency-trading/7A0A5BD6EACCF3B6BA1C99B5E69ED70E

Implication for this project:

The repo's observed maker adverse selection is exactly what theory predicts. Queue/fill-quality measurement should be promoted from "maybe" to core infrastructure. Any passive execution idea must report fill-conditioned markouts, realized spread, fill probability, queue-position assumptions, and post-fee PnL.

### 4. Toxicity is real, but VPIN-style shortcuts are controversial

External support:

- Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High-frequency World"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
- Easley, Lopez de Prado, O'Hara, "The Microstructure of the Flash Crash"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695041
- Andersen and Bondarenko, "VPIN and the Flash Crash"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731
- Andersen and Bondarenko, "Assessing Measures of Order Flow Toxicity and Early Warning Signals for Market Turbulence"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2449004

Implication for this project:

The toxic-regime/no-quote overlay remains the most defensible first trading-adjacent direction, but it should not rely on one magic toxicity metric. Use an ensemble of OFI, signed flow, spread/depth collapse, BBO turnover, nonce gaps, cancel/depth shocks, liquidation flow, volatility, and mark/oracle dislocation. Evaluate by adverse-selection reduction and realized-spread improvement, not AUC.

### 5. Crypto/perp markets have exploitable frictions, but execution realism dominates

External support:

- Makarov and Schoar, "Trading and Arbitrage in Cryptocurrency Markets"  
  https://academic.oup.com/jfe/article/135/2/293/5728511
- Alexander, Choi, Park, Sohn, "BitMEX Bitcoin Derivatives: Price Discovery, Informational Efficiency, and Hedging Effectiveness"  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3353583
- Akyildirim, Corbet, Katsiampa, Kellard, Sensoy, "The Development of Bitcoin Futures"  
  https://www.sciencedirect.com/science/article/pii/S1544612319308700
- Paradigm, "A Guide to Designing Effective Perpetual Contracts"  
  https://www.paradigm.xyz/2021/08/a-guide-to-designing-effective-perpetual-contracts

Implication for this project:

Funding, OI, mark/oracle basis, liquidations, and venue-specific mechanics are valid research variables. But apparent basis/funding edges must survive fees, slippage, queue position, liquidation risk, funding payment timing, and mark/index construction details.

### 6. DeFi/derivatives liquidation and oracle mechanics matter

External support:

- Gudgeon, Perez, Harz, Livshits, Gervais, "The Decentralized Financial Crisis"  
  https://arxiv.org/abs/2002.08099
- Qin, Zhou, Livshits, Gervais, "Attacking the DeFi Ecosystem with Flash Loans for Fun and Profit"  
  https://arxiv.org/abs/2003.03810
- Schar, "Decentralized Finance: On Blockchain- and Smart Contract-Based Financial Markets"  
  https://research.stlouisfed.org/publications/review/2021/02/05/decentralized-finance-on-blockchain-and-smart-contract-based-financial-markets

Implication for this project:

Liquidation events and mark/oracle dislocations should be modeled as venue-specific stress/forced-flow states. The plan's shift from pre-cascade direction to post-event absorption/stabilization is externally reasonable.

## Changed ranking after external grounding

Original plan ranking:

1. Toxic-regime / no-quote overlay
2. Mark/oracle/mid dislocation reversion
3. Stale quote / oracle cross
4. Post-liquidation absorption
5. Queue/fill-quality model
6. Funding/OI carry
7. Nonce/order-count toxicity features

Revised ranking:

0. Full-fidelity execution and adverse-selection measurement layer
   - event joiner;
   - OFI/queue/microprice baseline features;
   - fill/no-fill model;
   - fill-conditioned markouts;
   - realized spread;
   - latency/queue assumptions;
   - post-fee PnL.

1. Toxic-regime / no-quote overlay
   - still the best first trading-adjacent direction;
   - use toxicity to avoid adverse selection, not to predict direction.

2. Queue/fill-quality selective maker model
   - promoted because literature and repo results both say adverse selection is central;
   - only useful if measured with conservative fill assumptions.

3. Stale quote / oracle / cross-venue dislocation
   - keep high-upside, but only if latency/fill validation is strict.

4. Post-liquidation absorption / re-entry
   - keep, but frame as exhaustion/stabilization after forced flow is observed.

5. Funding/OI carry with toxicity overlay
   - separate slower-horizon sleeve;
   - useful as regime/crowding context even if not standalone alpha.

6. Mark/oracle/mid basis reversion
   - downgrade unless early data shows enough truly executable dislocations;
   - basis can be risk premium or stress, not automatically mispricing.

## Practical gates added by external grounding

Every future probe should report:

- net PnL after maker/taker fees;
- spread and measured slippage;
- fill-conditioned markout;
- realized spread;
- queue/fill assumptions;
- signal decay curve under realistic latency;
- day-blocked or purged walk-forward validation;
- regime split: calm, high-vol, liquidation, funding window, spread-widening, low-liquidity;
- ablation versus simple OFI, queue imbalance, signed flow, spread/depth, volatility, and recent returns.

## Bottom line

The external literature mostly supports the direction of the brainstorm, but it changes the emphasis:

- less confidence in mark/oracle basis as an early alpha;
- more emphasis on OFI/microprice/queue imbalance baselines;
- much more emphasis on fill-conditioned adverse-selection measurement;
- strong support for toxic-regime/no-quote as the first practical path;
- strong skepticism toward DeepLOB/RL/AUC chasing unless simple post-cost baselines already work.
