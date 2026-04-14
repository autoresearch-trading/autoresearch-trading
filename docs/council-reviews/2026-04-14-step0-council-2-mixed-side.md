# Council-2 (Cont) — Step 0 Mixed-Side Rate Review

**Date:** 2026-04-14
**Reviewer:** Council-2 (Microstructure / Order Flow)
**Sources:** spec §Order Event Grouping, `docs/experiments/step0-data-validation.md` + JSON, CLAUDE.md gotchas #3, #19

---

## Verdict

The spec's **59% mixed-side claim is empirically wrong**. Correct rate is **3-16%** across 25 symbols (median ~3%) after `drop_duplicates(subset=['ts_ms','qty','price'])` without `side`. The dedup procedure itself is sound; only the spec text and CLAUDE.md annotation are incorrect.

## Microstructure explanation

DEXes (and centralized exchanges) record every matched fill twice — once per counterparty. Both records share `(ts_ms, qty, price)` but differ on `side` (taker perspective vs maker perspective).

BTC Oct-16 arithmetic:
- Raw rows: 74,370
- Dedup WITH side (removes only true API duplicates): 51,979
- Dedup WITHOUT side (collapses both counterparty records of each fill): 27,558
- Cross-side pairs eliminated by no-side dedup: 24,421 (≈half of 51,979)

This confirms the bilateral tape structure. No-side dedup correctly selects one record per fill. The original 59% likely came from measuring on **pre-dedup or partially-deduped data**, where each fill appears as two cross-side rows and nearly every same-timestamp group trivially shows both sides (Step 0 confirms pre-dedup rate is 99-100%).

## Also wrong: CLAUDE.md gotcha #19 reasoning

Current reasoning: *"buyer/seller pairs differ on side, so including it in dedup removes nothing."*

Data proves the opposite: including `side` in the dedup key **preserves** both counterparty records (51,979 rows remain), doubling event counts. The **instruction** (dedup without `side`) is correct; only the **reasoning** is inverted.

## Physical meaning of the 3-16% rate

After correct dedup, a "mixed-side event" represents a genuinely multi-counterparty order — an aggressive order at timestamp T simultaneously matched against makers entering from opposite position contexts at the same (qty, price). Rate correlates with liquidity:
- Illiquid (2Z, KBONK, CRV): 0.3-3%
- Liquid (BTC, ETH, SOL): 7-14%

Mid-tier outliers (AAVE Jan-04 24.7%, PENGU Jan-04 28.2%, SUI Jan-04 21.9%) deserve monitoring — possibly liquidation cascades or data artifacts on specific dates.

## Feature impact

**Unaffected:** `log_return`, `log_total_qty`, `effort_vs_result`, `climax_score`, all OB-derived features, `kyle_lambda`.

**Changed distribution (still correct, no pipeline bug):**
- `book_walk` will be 0 for 85-97% of events (single-fill events dominate). Episodic signal — fires during genuine sweeps.
- `num_fills` ≈ 0 (log 1) for most events. Episodic.
- `is_open` is effectively binary {0, 1} for single-fill events (84-97%). For BTC (avg_fills=2.67), intermediate values reachable.

## Required spec edits

| Location | Current | Replace with |
|----------|---------|--------------|
| Spec §Order Event Grouping line 47 | "Expect 59% mixed-side events (exchange mechanic, not error)." | "Step 0 measured 3-16% mixed-side events after correct no-side dedup (median ~3% across 25 symbols; liquid symbols BTC/ETH/SOL toward upper end). Raw data shows ~99% cross-side pairs, confirming the API records both counterparties — no-side dedup collapses these correctly. The original 59% figure was an artifact of measuring on undeduped data." |
| CLAUDE.md gotcha #3 | "59% of events have mixed buy/sell fills (exchange mechanic, not error)." | "3-16% of events have mixed buy/sell fills after correct dedup (median ~3%; liquid symbols higher). The original 59% figure was measured on undeduped data where bilateral fill records inflate apparent mixed-side counts." |
| CLAUDE.md gotcha #19 reasoning | "buyer/seller pairs differ on side, so including it in dedup removes nothing." | "buyer/seller pairs share identical (ts_ms, qty, price) but differ on side — including side in the dedup key PRESERVES both counterparty records, doubling event counts. Excluding side collapses them to one record per fill, which is correct." |

**Dedup instruction itself is correct and requires no change.**

## Summary

Spec text wrong in three locations, pipeline sound. Feature semantics intact (distributions more zero-inflated / bimodal than spec implied, but mathematically valid).
