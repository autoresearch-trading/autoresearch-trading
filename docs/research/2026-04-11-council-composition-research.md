# Research: Council Composition for Tape Representation Learning

## Question
Are our 6 council agents the right advisory team for self-supervised representation learning on DEX perpetual futures order flow data?

## Sources
1. Zhong et al. (2025). "LOBench: Benchmarking Limit Order Book Representation Learning." arXiv:2505.02139
2. Cont et al. (2021). "Cross-Impact of Order Flow Imbalance in Equity Markets." arXiv:2112.13213
3. Kolm & Westray (2023). "Deep Order Flow Imbalance." Mathematical Finance 33:4
4. Yue et al. (2022). "TS2Vec: Towards Universal Representation of Time Series." AAAI 2022, arXiv:2106.10466
5. Nagy (2023). "Generative Pre-Trained Transformer for Limit Order Book." ACM ICAIF, arXiv:2309.00638

## Key Findings

**Cont vs Kyle (council-2 vs council-3) — not redundant.** They operate at complementary timescales. Cont's OFI explains 60-80% of contemporaneous price variance at 1-minute frequency (instantaneous flow pressure). Kyle's lambda captures permanent information impact over ~20-minute rolling windows. Kolm & Westray (2023) explicitly combines both and finds the alpha decay structure requires both. In our feature set: `cum_ofi_5` is Cont, `kyle_lambda` is Kyle.

**Wyckoff (council-4) — reframe needed.** Zero peer-reviewed papers validate Wyckoff's phase labels statistically. However, the derived features map to real microstructure: effort_vs_result ≈ inverse Kyle lambda at trade level, climax_score ≈ LOB liquidity crises, is_open ≈ direct informed trader participation measure. Recommendation: reframe from "Wyckoff tape reading" to "volume-price microstructure phenomenology."

**Missing: regime non-stationarity.** LOBench (2025) identifies this as the primary failure mode for financial SSL. Generic SSL (TS2Vec) underperforms domain-specific approaches because it ignores cross-feature constraints and autocorrelation structure. Gate 4 tests temporal stability, but no council member explicitly owns this concern. Recommendation: expand council-5's charter.

**Data engineering on council — no.** Nagy (2023) finds preprocessing decisions are load-bearing, but the gotchas in CLAUDE.md already encode these. Elevating to advisory creates churn on settled decisions.

## Verdict
All 6 seats justified and non-redundant. Two changes implemented:
1. Council-4 reframed to "microstructure phenomenologist" (academic grounding, not narrative)
2. Council-5 expanded with regime non-stationarity charter and council-4 falsifiability mandate
