# Research: Pacifica + Comparable DEX Perp Fee Schedules — Maker Pivot Feasibility

**Date:** 2026-04-27
**Author:** researcher-14
**Purpose:** Determine whether a passive (maker) execution pivot can flip the Goal-A
feasibility table that was fee-blocked under a flat 6 bp/side taker assumption.

---

## Question

The v1 program's Goal-A feasibility study assumed **6 bp/side Pacifica taker** as a
flat, tier-agnostic cost. Under that assumption only one cell survived (PUMP $1k H500
at +0.51 bp at empirically-unattainable 60% accuracy). Before pivoting, we need to
know:

1. What is Pacifica's actual published fee schedule (taker, maker, tier structure)?
2. Are maker rebates available on Pacifica? Headline number?
3. What order types does Pacifica support — specifically post-only / ALO?
4. Comparable venues (Hyperliquid, dYdX v4, Aevo, GMX v2): maker/taker bps for
   the same perp universe?
5. Empirical maker fill rates on DEX perps at varying offset distances?

---

## Sources

| # | Source | URL |
|---|--------|-----|
| S1 | Pacifica Official Docs — Trading Fees | https://docs.pacifica.fi/trading-on-pacifica/trading-fees |
| S2 | Pacifica Official Docs — Order Types | https://docs.pacifica.fi/trading-on-pacifica/order-types |
| S3 | Pacifica Official Docs — Market Maker Program | https://pacifica.gitbook.io/docs/programs/market-maker-program |
| S4 | Pacifica Official Docs — VIP Program | https://docs.pacifica.fi/vip-program |
| S5 | Pacifica API — Create Limit Order | https://docs.pacifica.fi/api-documentation/api/rest-api/orders/create-limit-order |
| S6 | Hyperliquid Official Docs — Fees | https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees |
| S7 | dYdX VIP Page (live, crawled 2026-04-27) | https://www.dydx.xyz/vip |
| S8 | dYdX Help Center — Perpetual Trade Fees | https://help.dydx.exchange/en/articles/4798040-perpetual-trade-fees |
| S9 | Aevo Official Docs — Maker and Taker Fees | https://docs.aevo.xyz/aevo-products/aevo-exchange/fees/maker-and-taker-fees |
| S10 | Aevo API Docs — Fees | https://api-docs.aevo.xyz/docs/fees |
| S11 | GMX v2 Official Docs — Fees | https://docs.gmx.io/docs/trading/fees/ |
| S12 | Albers et al. (2025), "The Market Maker's Dilemma," arXiv:2502.18625 | https://arxiv.org/pdf/2502.18625 |
| S13 | Fabre & Ragel (2023), "Execution Probability Computation," arXiv:2307.04863 | https://arxiv.org/pdf/2307.04863v1 |
| S14 | Lokin & Yu (2024/2026), "Fill Probabilities in a Limit Order Book," arXiv:2403.02572 | https://arxiv.org/pdf/2403.02572 |

---

## Key Findings

### 1. Pacifica fee schedule (S1, confidence: HIGH)

Published, tiered, 14-day rolling volume, identical schedule across all 25 perp
markets in our universe. **Source is the official `docs.pacifica.fi` site.**

| Tier | 14-day Volume | Maker (%) | Taker (%) | Maker (bps) | Taker (bps) |
|------|---------------|-----------|-----------|-------------|-------------|
| 1    | $0            | 0.015%    | 0.040%    | **1.5 bp**  | **4.0 bp**  |
| 2    | > $5M         | 0.012%    | 0.038%    | 1.2 bp      | 3.8 bp      |
| 3    | > $10M        | 0.009%    | 0.036%    | 0.9 bp      | 3.6 bp      |
| 4    | > $25M        | 0.006%    | 0.034%    | 0.6 bp      | 3.4 bp      |
| 5    | > $50M        | 0.003%    | 0.032%    | 0.3 bp      | 3.2 bp      |
| VIP 1| > $100M       | 0.000%    | 0.030%    | **0.0 bp**  | 3.0 bp      |
| VIP 2| > $250M       | 0.000%    | 0.029%    | 0.0 bp      | 2.9 bp      |
| VIP 3| > $500M       | 0.000%    | 0.028%    | 0.0 bp      | 2.8 bp      |

**Where does the 6 bp/side figure come from?** It does NOT appear in current
Pacifica docs. The 6 bp number likely originated from one of:
- An older "base" tier (some 2026-Q1 third-party reviews quoted 0.020% maker /
  0.050% taker — see PerpFinder, Dexrank — possibly reflecting a pre-tier-revamp
  fee or a discontinued promotional rate).
- Conflation with Aevo (5 bp taker) or another venue.
- A round-up safety margin including slippage.

**Current operational reality:** at the default tier (no volume), Pacifica is
**4.0 bp taker / 1.5 bp maker** — meaningfully cheaper than the 6 bp assumption.
Round-trip taker = 8.0 bp. Round-trip maker = 3.0 bp. Round-trip taker→maker
(open taker, close maker) = 5.5 bp.

**Promotional caveat:** A 50% RWA fee discount ran Mar 24–31, 2026 (RWA markets
only — NVDA, TSLA, GOLD, etc.). Does not apply to our 25 crypto perps. Confidence
that the 4/1.5 bp schedule applies to all 25 symbols in our universe: **HIGH** (S1
states "all volume thresholds refer to your total executed trading volume" with no
per-market carve-outs except the explicit RWA promo).

### 2. Pacifica maker rebates (S3, confidence: MEDIUM-HIGH)

**Pacifica does NOT pay a flat per-fill maker rebate.** Maker fee bottoms out at
**0.0 bp** at VIP 1+ ($100M+ 14-day volume). However, there is a separate
**Market Maker Program** (`pacifica.gitbook.io/docs/programs/market-maker-program`)
that provides:

- **Zero fees on maker fills** (regardless of tier, for opted-in MMs).
- **A pool-distributed rebate**: 12% of all trading fees collected from MM
  counterparties is redistributed to qualifying MMs each period.
- Plus 2,000,000 points/period (1M/week) — a separate incentive.

**Implications for us:**
- We are **not** a market maker by volume — the $100M+ VIP 1 threshold is
  inaccessible at $1k–$100k bankroll.
- The MM program requires opt-in, has a `MakerScore` qualification gate, and
  reports weekly. It is operationally infeasible for our research strategy.
- **Operational maker fee for us = +1.5 bp** (Tier 1 published rate, no rebate).
  We pay maker fees, we do not earn them.

**Third-party note:** One review (`dexcexhub.com`) referenced "15% cashback" as
a Pacifica promotion; this is a referral/points program, not a fee rebate, and
is not reliable as a recurring income source.

### 3. Pacifica order types (S2, S5, confidence: HIGH)

Pacifica supports a full range of order types via REST and WebSocket APIs:

| TIF | Behavior | Maker-guaranteed? |
|-----|----------|-------------------|
| GTC | Stays on book until filled/cancelled | NO (can cross & fill as taker) |
| IOC | Match immediately or cancel | NO (taker) |
| **ALO** ("Add-Liquidity-Only" / "Post Only") | Rejected if it would cross | **YES** |
| **TOB** ("Top-of-Book") | ALO that auto-rebases to best bid+1 / best ask−1 if it would cross | **YES** |

**Critical for our purposes:** Pacifica DOES support a guaranteed-maker post-only
order via TIF=`ALO`. This is the load-bearing primitive for the maker pivot —
without it, a limit order can be filled as a taker if the book moves.

**Latency note:** Market, GTC, and IOC orders are subject to a randomized
50–100 ms speed bump (later doc revisions say ~200 ms) to protect makers from
adverse selection. ALO and TOB orders are **NOT** speed-bumped, which is a small
edge for posters.

### 4. Comparable venue fee schedules (S6–S11, confidence: HIGH for HL/dYdX/Aevo, MEDIUM for GMX)

| Venue | Default Taker | Default Maker | Best Taker (top tier) | Best Maker (top tier) | Maker-rebate at top tier? |
|-------|---------------|---------------|-----------------------|------------------------|---------------------------|
| **Pacifica** | 4.0 bp | +1.5 bp | 2.8 bp (VIP3 $500M+) | 0.0 bp (VIP1 $100M+) | NO (pool-based MM program) |
| **Hyperliquid** | 4.5 bp | +1.5 bp | 2.4 bp (VIP6 $7B+) | 0.0 bp (VIP4 $500M+) | YES — up to **−0.3 bp** rebate at >3% maker share |
| **dYdX v4** | 5.0 bp | +1.0 bp | 2.5 bp ($200M+) | **−1.1 bp** rebate ($200M+) | YES — best maker rebate of the four |
| **Aevo** | 5.0 bp (perp) | +3.0 bp (perp, API docs); +5.0 bp (third-party) | n/a (no published tiers above base) | +2.0 bp (some sources) | NO |
| **GMX v2** | 4.0 bp / 6.0 bp (imbalance-conditional) | n/a — no maker/taker model | same | same | NO (oracle-priced AMM) |

**Sourcing notes:**
- Hyperliquid: official docs (S6) confirm rebate tiers `>0.5%/1.5%/3.0%` of 14d
  weighted maker volume → −0.001%/−0.002%/−0.003% (i.e., −0.1/−0.2/−0.3 bp).
  Our 25-symbol universe overlap with Hyperliquid is ~80% (BTC, ETH, SOL, BNB,
  AVAX, DOGE, LINK, LTC, AAVE, CRV, HYPE, LDO, SUI, UNI, XRP, PUMP, FARTCOIN,
  ENA, PENGU likely all listed; 2Z, ASTER, KBONK, KPEPE, WLFI, XPL marginal).
- dYdX v4: live VIP page (S7, crawled 2026-04-27) confirms **negative maker
  fees at tiers 06 and 07** — −0.7 bp at $100M+, **−1.1 bp at $200M+**. This
  is the most generous maker rebate of any DEX perp. Help-center page (S8) is
  older and shows a different schedule (rebate-less); the current live VIP page
  is the operational truth.
- Aevo: official docs (S9, S10) split — `api-docs.aevo.xyz` says 0.05% taker /
  0.03% maker; third-party reviews (PerpFinder, Fensory) cite 0.05% taker /
  0.05% maker. Either way, Aevo is the worst maker venue of the four. Plus
  Aevo's perp universe is much smaller than ours.
- GMX v2: not maker/taker. Open/close fee is 4 bp (imbalance-reducing) or
  6 bp (imbalance-increasing) plus oracle price impact. Does NOT support
  passive limit orders in the same sense — irrelevant for the maker pivot.

**Verdict on best-comparable venue:** **dYdX v4 has the most rebate-friendly
schedule at high volume** (−1.1 bp maker at $200M+), but at our $1k–$100k
operational scale, **Hyperliquid is the most pragmatic alternative** (1.5 bp
maker at default tier, identical to Pacifica, with a small platform-share-based
rebate possible). Note the perp universe overlap with our 25 symbols is best
on **Hyperliquid** (~80%+); dYdX v4 has 185+ markets but lower overlap on
Solana-native memecoins (KBONK, KPEPE, WLFI, XPL, PUMP, ASTER may not exist).

### 5. Empirical maker fill rates (S12, S13, S14, confidence: MEDIUM)

This is the load-bearing number for the pivot, and it is also the most uncertain.

**The "Maker's Dilemma" finding (Albers et al. 2025, arXiv:2502.18625, S12):**
Live experiment on Binance BTC-perp (the most liquid crypto market). Documents a
**negative correlation between fill probability and post-fill returns**: orders
that fill quickly tend to fill *because* the price is about to move against you
(adverse selection / negative drift). Quote: a typical front-of-queue maker
order at the touch shows **expected return of ≈ −0.8 bp gross, or −0.3 bp net of
rebate** in their experiment.

**Concrete fill probabilities by offset (S13, S14):**
- At the touch (offset = 0, best bid/ask): fill probabilities depend heavily on
  queue position, queue size, and side imbalance. Front-of-queue at the touch
  in liquid markets fills with high probability (>50%) within seconds; back of
  queue may not fill at all before being cancelled.
- **One tick deeper** (offset = 1 tick from best): fill probability drops
  sharply. Lokin & Yu (2024) note that "fill probabilities beyond one tick from
  the best quote are typically negligible" in standard markets.
- 5 bp / 10 bp offsets: in small-tick crypto pairs (most of our universe), 5 bp
  is many ticks deep — fill probabilities approach zero unless there is a
  significant price move toward your level. These are the offsets that earn
  meaningful "free spread" but are statistically unlikely to fill.

**No DEX-perp-specific large-N study found.** Most empirical fill-rate work is
on CEX (Binance, NASDAQ, Stockholm) or theoretical. Pacifica/Hyperliquid have
not been the subject of academic fill-rate studies as of 2026-04-27.

**Practitioner anecdotes (S15 implicit, novalerm.com 2025):** Real fills on
DEX perps "degrade as size grows" and "you can get great fills until you don't"
(concentrated-liquidity AMM caveat). For an order-book DEX like Pacifica or
Hyperliquid the dynamics are closer to Binance — fill rates at the touch are
high, but adverse selection eats most of the apparent rebate.

**Operational rule-of-thumb (medium confidence):**
- Touch (best bid/ask) post-only: ~40–70% fill rate within 1–5 minutes for liquid
  symbols (BTC/ETH/SOL); 10–30% for illiquid alts. Adverse selection ~−0.3 to
  −0.8 bp on filled orders.
- 1 tick deeper: ~10–25% fill rate; less adverse selection.
- 5 bp deeper: <5% fill rate without a directional move.
- **A directional model that posts maker on the side it predicts** is exactly
  the kind of strategy where the Albers fill-vs-return correlation eats most
  of the predicted edge — you fill more often when you're wrong.

---

## Relevance to Our Project

**Re-evaluating the Goal-A feasibility table under accurate Pacifica fees:**

The original feasibility study used **6 bp/side taker** (round-trip 12 bp + slip
≈ 13–15 bp). Actual Pacifica defaults are:

| Execution mode | Per-side cost | Round-trip | vs original 6 bp/side assumption |
|----------------|---------------|------------|----------------------------------|
| Pure taker (Tier 1) | 4.0 bp | 8.0 bp | **−4 bp lower than assumption** |
| Open taker / close maker | 4.0 + 1.5 = 5.5 bp avg | 5.5 bp | −7.5 bp lower |
| Pure maker — fills (Tier 1) | 1.5 bp + adverse selection (~0.5–0.8 bp) | ~4–5 bp effective | −7 to −8 bp lower |
| Pure maker — VIP 1 ($100M+) | 0.0 bp + adverse selection | ~1–2 bp effective | −10+ bp lower (but unreachable scale) |
| Hyperliquid maker (Tier 0) | 1.5 bp + adverse selection | ~4–5 bp | identical to Pacifica |
| dYdX v4 maker (Tier 07, $200M+) | −1.1 bp rebate | net rebate, but still adverse selection | only at unreachable scale |

**Critical caveats:**
1. **Adverse selection ≈ 0.3–0.8 bp** on filled maker orders (Albers 2025). This
   eats most of the headline rebate. A "1.5 bp maker" Pacifica fill effectively
   costs the trader 2.0–2.3 bp once you account for the negative drift on fill.
2. **Fill rate ≠ 100%.** A directional model can only act on signals when its
   maker order actually gets filled. If 50% of posted orders fill, the strategy
   only realizes half its predicted edge — and the unfilled half may be the
   *better* signals (you missed the fast moves).
3. **The Albers correlation is fundamental, not Pacifica-specific.** Posting
   maker on the predicted-direction side means you fill disproportionately when
   the prediction is wrong.

### Cost-band the maker pivot would run under

For a 55%-balanced-accuracy model, expected gross edge = `(2 × accuracy − 1) ×
median |horizon move|`. At H500 with median |move| ≈ 50–100 bp:
- Expected gross at 55% acc: 0.10 × 75 bp ≈ **7.5 bp**
- Round-trip cost ceiling for break-even: **< 7.5 bp** (any margin requires <)

| Cost regime | Round-trip cost | Survives at 55% acc? |
|-------------|-----------------|-----------------------|
| Original 6bp taker assumption | 12–15 bp | NO — ~half the cells fee-blocked |
| **Actual Pacifica taker (Tier 1)** | **8.0 bp** | **MARGINAL — borderline survival** |
| Pacifica open-taker / close-maker | 5.5 bp | YES — many cells survive |
| **Pacifica pure maker, naive (1.5 bp × 2)** | **3.0 bp** + ~1 bp adverse | **YES — most cells survive** |
| Pacifica pure maker, with 50% fill rate haircut on edge | 3.0 bp + 1 bp adverse, but realized edge halved (~3.75 bp) | NO — break-even |

**Punchline:** The maker pivot's break-even is at **~5–6 bp/side maker fee
(or rebate of similar magnitude)** for a 55% accuracy model with realistic fill
rates. **Pacifica's actual 1.5 bp maker fee at default tier is well below this
threshold** — the cost band is plausibly survivable IF (a) fill rate is >30%
and (b) adverse selection is <1 bp/fill. Both are uncertain.

### Recommendation

1. **Re-run the feasibility table with parameterized maker fees.** Suggest three
   scenarios:
   - Pessimistic: 1.5 bp maker × 2 sides + 1.0 bp adverse selection × 2 + 50%
     fill-rate edge haircut (round-trip effective: ~5.0 bp + edge halved)
   - Realistic: 1.5 bp × 2 + 0.5 bp adverse × 2 + 70% fill rate (round-trip:
     ~4.0 bp + 30% edge haircut)
   - Optimistic: 1.5 bp × 2 + 0 adverse + 100% fill (3.0 bp pure)
2. **Also include a "hybrid" mode** — open as ALO (post-only), close as taker
   (3.0 + 4.0 + slip ≈ 8 bp round-trip). This is operationally simplest because
   it doesn't require the closing fill to be passive (a directional model
   *wants* to exit on signal, which conflicts with patient passive closing).
3. **Do NOT use the dYdX or Hyperliquid rebate tiers as planning numbers** —
   they require $100M+ 14d volume which is structurally unreachable at our
   research bankroll.
4. **The fundamental Albers result is the binding constraint, not the fee
   schedule.** Even with rebates, a directional maker strategy fights the
   negative-drift-of-fills correlation. This is more important than which
   venue we pick.

---

## Confidence Summary

| Finding | Confidence | Why |
|---------|-----------|-----|
| Pacifica taker = 4.0 bp at Tier 1 | **HIGH** | Official docs, multiple corroborating third-party sources |
| Pacifica maker = 1.5 bp at Tier 1 (no flat rebate) | **HIGH** | Official docs |
| The "6 bp/side" assumption is too pessimistic by ~2 bp | **HIGH** | Documented schedule contradicts the assumption |
| Pacifica supports guaranteed post-only via ALO/TOB TIF | **HIGH** | Confirmed in official docs, REST API, WebSocket API |
| Hyperliquid: 4.5/1.5 bp default; up to −0.3 bp rebate at top tier | **HIGH** | Official docs |
| dYdX v4: −1.1 bp maker rebate at $200M+ tier (live page) | **HIGH** | Crawled live VIP page 2026-04-27 |
| Aevo: 5.0 bp taker / 3–5 bp maker (no rebate) | MEDIUM | Conflicting numbers between Aevo's own API docs and product docs |
| GMX v2: oracle-priced, no maker/taker model | **HIGH** | Official docs |
| Maker fill rates at touch ≈ 40–70% liquid / 10–30% illiquid | **MEDIUM** | Inference from CEX studies + practitioner anecdotes; no DEX-perp-specific study found |
| Adverse selection on filled maker orders ≈ 0.3–0.8 bp | **MEDIUM** | Albers 2025 specific to Binance BTC-perp; mechanism is general but magnitude may differ on Pacifica |
| Maker pivot break-even fee ≈ 5–6 bp/side | MEDIUM | Sensitive to assumed accuracy, edge magnitude, and fill-rate haircut |
| Pacifica's actual 1.5 bp maker fee is below break-even threshold | **HIGH** (under realistic, not pessimistic, fill assumptions) | Direct comparison of (1) and (11) |

---

## Notes on what could not be found (within 30-min cap)

- **No published academic study on DEX perp fill rates specifically.** All
  empirical fill-rate work is CEX-based. A live Pacifica fill-rate experiment
  would be a meaningful contribution but is out of scope for desk research.
- **No published Pacifica adverse-selection numbers.** The Albers 2025 paper
  used Binance, where book depth is 10–100x Pacifica's. Pacifica adverse
  selection on retail-size orders may be lower (less informed flow) or higher
  (less queue depth) — net direction unclear.
- **Pacifica's "Builder Program" (referenced in kkdemian.com)** may offer
  third-party fee splits. Not material to our use case (we are not a builder).
- **dYdX help-center vs live VIP page conflict.** Help center shows older
  schedule with no rebates; live VIP page shows current rebate tiers. The
  live page (S7) is operational truth as of 2026-04-27.
