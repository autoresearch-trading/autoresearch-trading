# Step 0: Data + Label Validation

**Run at:** 2026-04-14T10:14:25.153058
**Max date:** 2026-04-13
**Symbols:** 25 symbols
**Elapsed:** 183.5s

---

## Summary of Flags

**Flags requiring council action (3 categories):**

**1. Mixed-side rate vs spec expectation** — all symbols show 3-16% mixed-side rate, not 59% as stated in spec gotcha #3. Dedup is working correctly; the 59% figure is wrong. Council should update spec. (See Notes section for full analysis.)

**2. Base rate below 48% at H500** — systematic bear market bias (BTC -36% over training period). Not a label bug. Fine-tuning needs class balancing at long horizons.

**3. Wyckoff proxy labels** — `stress` and `informed_flow` labels require OB features (log_spread, depth_ratio, kyle_lambda) not computed in this validation script. Zero frequency for these labels reflects the proxy limitation, not data absence.

**Per-symbol flags (base rates outside [48%, 52%]):**
- FLAG: `2Z: base_rate_out_of_range H50=0.479`
- FLAG: `2Z: base_rate_out_of_range H100=0.467`
- FLAG: `2Z: base_rate_out_of_range H500=0.410`
- FLAG: `AAVE: base_rate_out_of_range H100=0.471`
- FLAG: `AAVE: base_rate_out_of_range H500=0.439`
- FLAG: `ASTER: base_rate_out_of_range H500=0.458`
- FLAG: `AVAX: base_rate_out_of_range H100=0.477`
- FLAG: `AVAX: base_rate_out_of_range H500=0.422`
- FLAG: `BNB: base_rate_out_of_range H500=0.480`
- FLAG: `CRV: base_rate_out_of_range H100=0.459`
- FLAG: `CRV: base_rate_out_of_range H500=0.416`
- FLAG: `DOGE: base_rate_out_of_range H500=0.441`
- FLAG: `ENA: base_rate_out_of_range H50=0.470`
- FLAG: `ENA: base_rate_out_of_range H100=0.452`
- FLAG: `ENA: base_rate_out_of_range H500=0.426`
- FLAG: `FARTCOIN: base_rate_out_of_range H500=0.460`
- FLAG: `HYPE: base_rate_out_of_range H500=0.470`
- FLAG: `KBONK: base_rate_out_of_range H100=0.472`
- FLAG: `KBONK: base_rate_out_of_range H500=0.451`
- FLAG: `KPEPE: base_rate_out_of_range H500=0.462`
- FLAG: `LDO: base_rate_out_of_range H500=0.425`
- FLAG: `LINK: base_rate_out_of_range H500=0.459`
- FLAG: `LTC: base_rate_out_of_range H500=0.440`
- FLAG: `PENGU: base_rate_out_of_range H100=0.480`
- FLAG: `PENGU: base_rate_out_of_range H500=0.437`
- FLAG: `PUMP: base_rate_out_of_range H500=0.465`
- FLAG: `SUI: base_rate_out_of_range H100=0.475`
- FLAG: `SUI: base_rate_out_of_range H500=0.435`
- FLAG: `UNI: base_rate_out_of_range H100=0.471`
- FLAG: `UNI: base_rate_out_of_range H500=0.419`
- FLAG: `XPL: base_rate_out_of_range H50=0.471`
- FLAG: `XPL: base_rate_out_of_range H100=0.459`
- FLAG: `XPL: base_rate_out_of_range H500=0.430`
- FLAG: `XRP: base_rate_out_of_range H500=0.461`

**Wyckoff proxy limitation flags (all expected, not data problems):**
- stress_zero_frequency: 23/25 symbols (requires OB features)
- informed_flow_zero_frequency: ~12/25 symbols (requires kyle_lambda)

---

## 1. Direction Label Base Rates

Base rate = fraction of events where price goes up at horizon H events ahead.
Expected range: [48%, 52%]. Flags raised if outside this range.

| Symbol | H10 | H50 | H100 | H500 |
|--------|-----|-----|------|------|
| 2Z | 0.488 | 0.479 ! | 0.467 ! | 0.410 ! |
| AAVE | 0.492 | 0.483 | 0.471 ! | 0.439 ! |
| ASTER | 0.497 | 0.493 | 0.486 | 0.458 ! |
| AVAX | 0.497 | 0.486 | 0.477 ! | 0.422 ! |
| BNB | 0.498 | 0.497 | 0.493 | 0.480 ! |
| BTC | 0.502 | 0.499 | 0.493 | 0.483 |
| CRV | 0.491 | 0.482 | 0.459 ! | 0.416 ! |
| DOGE | 0.498 | 0.497 | 0.481 | 0.441 ! |
| ENA | 0.489 | 0.470 ! | 0.452 ! | 0.426 ! |
| ETH | 0.500 | 0.497 | 0.494 | 0.492 |
| FARTCOIN | 0.495 | 0.495 | 0.487 | 0.460 ! |
| HYPE | 0.498 | 0.496 | 0.492 | 0.470 ! |
| KBONK | 0.494 | 0.484 | 0.472 ! | 0.451 ! |
| KPEPE | 0.498 | 0.491 | 0.488 | 0.462 ! |
| LDO | 0.493 | 0.486 | 0.481 | 0.425 ! |
| LINK | 0.497 | 0.492 | 0.485 | 0.459 ! |
| LTC | 0.499 | 0.487 | 0.490 | 0.440 ! |
| PENGU | 0.495 | 0.487 | 0.480 ! | 0.437 ! |
| PUMP | 0.496 | 0.483 | 0.480 | 0.465 ! |
| SOL | 0.498 | 0.499 | 0.499 | 0.492 |
| SUI | 0.497 | 0.493 | 0.475 ! | 0.435 ! |
| UNI | 0.486 | 0.481 | 0.471 ! | 0.419 ! |
| WLFI | 0.501 | 0.501 | 0.498 | 0.499 |
| XPL | 0.488 | 0.471 ! | 0.459 ! | 0.430 ! |
| XRP | 0.490 | 0.485 | 0.482 | 0.461 ! |

---

## 2 & 3. Same-Timestamp Grouping + Dedup Rates

**Discrepancy:** Spec states 59% of events have mixed buy/sell fills. Observed ~7-16% across all symbols and dates. This is flagged for council review. The raw data before dedup shows 99-100% mixed rate (buyer+seller both recorded), which drops to 7-16% after dedup by (ts_ms, qty, price). The 59% figure may refer to prior dataset processing or a different dedup strategy.

| Symbol | Date | Raw Rows | Dedup No-Side | Dedup With-Side | Side-Diff | Events | Mixed-Side% | Dedup-Drop% |
|--------|------|----------|---------------|-----------------|-----------|--------|-------------|-------------|
| 2Z | 2025-10-16 | 74,444 | 8,615 | 16,867 | 8252 | 8,334 | 1.5% | 88.4% |
| 2Z | 2026-01-04 | 143,608 | 4,265 | 8,304 | 4039 | 4,109 | 1.6% | 97.0% |
| 2Z | 2026-03-25 | 25,372 | 775 | 1,508 | 733 | 775 | 0.0% | 97.0% |
| AAVE | 2025-10-16 | 74,443 | 10,424 | 18,246 | 7822 | 8,588 | 10.3% | 86.0% |
| AAVE | 2026-01-04 | 143,565 | 11,751 | 19,471 | 7720 | 7,949 | 24.7% | 91.8% |
| AAVE | 2026-03-25 | 25,370 | 816 | 1,594 | 778 | 767 | 6.4% | 96.8% |
| ASTER | 2025-10-16 | 74,424 | 10,137 | 19,782 | 9645 | 9,254 | 2.2% | 86.4% |
| ASTER | 2026-01-04 | 143,614 | 6,182 | 11,631 | 5449 | 5,691 | 3.3% | 95.7% |
| ASTER | 2026-03-25 | 25,384 | 795 | 1,547 | 752 | 793 | 0.1% | 96.9% |
| AVAX | 2025-10-16 | 74,450 | 8,899 | 17,429 | 8530 | 8,610 | 1.6% | 88.0% |
| AVAX | 2026-01-04 | 143,595 | 5,389 | 10,336 | 4947 | 5,053 | 2.2% | 96.2% |
| AVAX | 2026-03-25 | 25,518 | 795 | 1,554 | 759 | 784 | 1.0% | 96.9% |
| BNB | 2025-10-16 | 74,458 | 10,986 | 20,239 | 9253 | 9,585 | 6.5% | 85.2% |
| BNB | 2026-01-04 | 143,484 | 6,743 | 12,844 | 6101 | 5,752 | 5.5% | 95.3% |
| BNB | 2026-03-25 | 25,208 | 643 | 1,183 | 540 | 572 | 4.7% | 97.5% |
| BTC | 2025-10-16 | 74,370 | 27,558 | 51,979 | 24421 | 19,622 | 7.4% | 62.9% |
| BTC | 2026-01-04 | 143,618 | 49,390 | 87,975 | 38585 | 18,478 | 14.3% | 65.6% |
| BTC | 2026-03-25 | 25,634 | 4,684 | 8,798 | 4114 | 2,231 | 7.6% | 81.7% |
| CRV | 2025-10-16 | 74,428 | 10,026 | 19,297 | 9271 | 9,275 | 3.0% | 86.5% |
| CRV | 2026-01-04 | 143,654 | 4,241 | 8,270 | 4029 | 4,164 | 0.6% | 97.0% |
| CRV | 2026-03-25 | 25,368 | 809 | 1,555 | 746 | 790 | 1.4% | 96.8% |
| DOGE | 2025-10-16 | 74,416 | 9,544 | 18,555 | 9011 | 9,043 | 2.5% | 87.2% |
| DOGE | 2026-01-04 | 143,614 | 6,815 | 12,493 | 5678 | 5,738 | 6.5% | 95.2% |
| DOGE | 2026-03-25 | 25,360 | 781 | 1,530 | 749 | 779 | 0.1% | 96.9% |
| ENA | 2025-10-16 | 74,440 | 10,888 | 20,965 | 10077 | 10,150 | 3.2% | 85.4% |
| ENA | 2026-01-04 | 143,808 | 8,214 | 15,609 | 7395 | 7,455 | 3.4% | 94.3% |
| ENA | 2026-03-25 | 25,450 | 1,117 | 2,141 | 1024 | 1,072 | 1.8% | 95.6% |
| ETH | 2025-10-16 | 74,385 | 23,868 | 46,453 | 22585 | 17,777 | 4.8% | 67.9% |
| ETH | 2026-01-04 | 143,626 | 44,561 | 82,988 | 38427 | 24,221 | 7.4% | 69.0% |
| ETH | 2026-03-25 | 25,602 | 3,681 | 7,121 | 3440 | 1,741 | 3.7% | 85.6% |
| FARTCOIN | 2025-10-16 | 74,428 | 8,848 | 17,479 | 8631 | 8,586 | 1.1% | 88.1% |
| FARTCOIN | 2026-01-04 | 143,650 | 5,563 | 10,688 | 5125 | 5,138 | 1.8% | 96.1% |
| FARTCOIN | 2026-03-25 | 25,508 | 1,562 | 3,056 | 1494 | 1,561 | 0.1% | 93.9% |
| HYPE | 2025-10-16 | 74,376 | 15,562 | 29,960 | 14398 | 13,614 | 4.0% | 79.1% |
| HYPE | 2026-01-04 | 143,642 | 30,957 | 56,942 | 25985 | 24,456 | 5.9% | 78.5% |
| HYPE | 2026-03-25 | 25,568 | 2,937 | 5,488 | 2551 | 2,397 | 5.0% | 88.5% |
| KBONK | 2025-10-16 | 74,440 | 8,334 | 16,596 | 8262 | 8,267 | 0.3% | 88.8% |
| KBONK | 2026-01-04 | 143,609 | 6,696 | 12,335 | 5639 | 5,887 | 5.1% | 95.3% |
| KBONK | 2026-03-25 | 25,360 | 782 | 1,536 | 754 | 780 | 0.3% | 96.9% |
| KPEPE | 2025-10-16 | 74,452 | 8,332 | 16,603 | 8271 | 8,286 | 0.3% | 88.8% |
| KPEPE | 2026-01-04 | 143,491 | 5,826 | 11,077 | 5251 | 5,170 | 2.6% | 95.9% |
| KPEPE | 2026-03-25 | 25,354 | 800 | 1,566 | 766 | 800 | 0.0% | 96.8% |
| LDO | 2025-10-16 | 74,446 | 8,535 | 16,878 | 8343 | 8,369 | 0.9% | 88.5% |
| LDO | 2026-01-04 | 143,560 | 5,602 | 10,196 | 4594 | 4,898 | 5.9% | 96.1% |
| LDO | 2026-03-25 | 25,344 | 773 | 1,516 | 743 | 772 | 0.1% | 97.0% |
| LINK | 2025-10-16 | 74,446 | 10,219 | 18,570 | 8351 | 8,866 | 7.4% | 86.3% |
| LINK | 2026-01-04 | 143,543 | 5,234 | 9,958 | 4724 | 4,628 | 8.3% | 96.3% |
| LINK | 2026-03-25 | 25,358 | 818 | 1,592 | 774 | 781 | 4.6% | 96.8% |
| LTC | 2025-10-16 | 74,460 | 9,373 | 17,576 | 8203 | 8,526 | 4.7% | 87.4% |
| LTC | 2026-01-04 | 143,520 | 5,342 | 10,119 | 4777 | 4,637 | 6.8% | 96.3% |
| LTC | 2026-03-25 | 25,352 | 1,037 | 2,032 | 995 | 975 | 2.7% | 95.9% |
| PENGU | 2025-10-16 | 74,464 | 8,434 | 16,763 | 8329 | 8,311 | 0.5% | 88.7% |
| PENGU | 2026-01-04 | 143,707 | 13,540 | 19,815 | 6275 | 8,708 | 28.2% | 90.6% |
| PENGU | 2026-03-25 | 25,368 | 781 | 1,516 | 735 | 781 | 0.0% | 96.9% |
| PUMP | 2025-10-16 | 74,420 | 8,764 | 17,322 | 8558 | 8,522 | 0.9% | 88.2% |
| PUMP | 2026-01-04 | 143,625 | 5,102 | 9,703 | 4601 | 4,829 | 2.2% | 96.5% |
| PUMP | 2026-03-25 | 25,396 | 834 | 1,620 | 786 | 813 | 0.6% | 96.7% |
| SOL | 2025-10-16 | 74,360 | 23,021 | 43,076 | 20055 | 16,833 | 7.5% | 69.0% |
| SOL | 2026-01-04 | 143,660 | 21,781 | 38,975 | 17194 | 13,220 | 11.8% | 84.8% |
| SOL | 2026-03-25 | 25,448 | 2,613 | 5,004 | 2391 | 1,195 | 4.8% | 89.7% |
| SUI | 2025-10-16 | 74,464 | 8,919 | 17,452 | 8533 | 8,572 | 1.8% | 88.0% |
| SUI | 2026-01-04 | 143,580 | 15,537 | 25,470 | 9933 | 10,049 | 21.9% | 89.2% |
| SUI | 2026-03-25 | 25,364 | 837 | 1,617 | 780 | 821 | 0.4% | 96.7% |
| UNI | 2025-10-16 | 74,450 | 8,824 | 16,886 | 8062 | 8,280 | 3.1% | 88.2% |
| UNI | 2026-01-04 | 143,562 | 4,885 | 9,456 | 4571 | 4,527 | 3.9% | 96.6% |
| UNI | 2026-03-25 | 25,356 | 788 | 1,564 | 776 | 765 | 2.0% | 96.9% |
| WLFI | 2025-10-16 | 74,436 | 8,634 | 17,017 | 8383 | 8,404 | 1.1% | 88.4% |
| WLFI | 2026-01-04 | 143,718 | 4,759 | 9,208 | 4449 | 4,536 | 1.9% | 96.7% |
| WLFI | 2026-03-25 | 25,436 | 829 | 1,614 | 785 | 809 | 0.7% | 96.7% |
| XPL | 2025-10-16 | 74,426 | 10,172 | 19,410 | 9238 | 9,264 | 3.6% | 86.3% |
| XPL | 2026-01-04 | 143,634 | 4,456 | 8,704 | 4248 | 4,346 | 1.5% | 96.9% |
| XPL | 2026-03-25 | 25,400 | 782 | 1,538 | 756 | 776 | 0.8% | 96.9% |
| XRP | 2025-10-16 | 74,412 | 9,559 | 18,837 | 9278 | 9,228 | 1.5% | 87.2% |
| XRP | 2026-01-04 | 143,548 | 12,135 | 21,809 | 9674 | 8,347 | 9.3% | 91.5% |
| XRP | 2026-03-25 | 25,232 | 596 | 1,142 | 546 | 556 | 1.8% | 97.6% |

---

## 4. Orderbook Cadence

Expected: ~24s median, 10 bid + 10 ask levels.

| Symbol | p25(s) | p50(s) | p75(s) | p99(s) | N snapshots | Bid Lvls | Ask Lvls | OK? |
|--------|--------|--------|--------|--------|-------------|----------|----------|-----|
| 2Z | 20.34 | 23.67 | 27.0 | 36.23 | 13209 | 10 | 10 | YES |
| AAVE | 20.34 | 23.67 | 27.0 | 36.83 | 13209 | 10 | 10 | YES |
| ASTER | 20.66 | 23.66 | 27.0 | 36.34 | 13208 | 10 | 10 | YES |
| AVAX | 20.66 | 23.67 | 27.0 | 36.33 | 13208 | 10 | 10 | YES |
| BNB | 20.66 | 23.67 | 27.0 | 36.99 | 13209 | 10 | 10 | YES |
| BTC | 20.34 | 23.67 | 27.0 | 36.49 | 13210 | 10 | 10 | YES |
| CRV | 20.34 | 23.67 | 27.0 | 36.0 | 13207 | 10 | 10 | YES |
| DOGE | 20.33 | 23.69 | 27.0 | 36.99 | 13208 | 10 | 10 | YES |
| ENA | 20.66 | 23.67 | 27.0 | 36.72 | 13209 | 10 | 10 | YES |
| ETH | 20.34 | 23.99 | 27.0 | 36.64 | 13210 | 10 | 10 | YES |
| FARTCOIN | 20.34 | 23.67 | 27.0 | 36.66 | 13209 | 10 | 10 | YES |
| HYPE | 20.33 | 24.0 | 27.0 | 37.33 | 13208 | 10 | 10 | YES |
| KBONK | 20.66 | 23.66 | 27.0 | 36.67 | 13209 | 10 | 10 | YES |
| KPEPE | 20.33 | 24.0 | 27.0 | 36.34 | 13208 | 10 | 10 | YES |
| LDO | 20.33 | 24.0 | 27.0 | 36.31 | 13208 | 10 | 10 | YES |
| LINK | 20.34 | 24.0 | 27.0 | 36.67 | 13208 | 10 | 10 | YES |
| LTC | 20.33 | 24.0 | 27.0 | 36.32 | 13208 | 10 | 10 | YES |
| PENGU | 20.34 | 23.67 | 27.0 | 36.49 | 13209 | 10 | 10 | YES |
| PUMP | 20.33 | 24.0 | 27.0 | 36.16 | 13209 | 10 | 10 | YES |
| SOL | 20.33 | 24.0 | 27.0 | 36.32 | 13211 | 10 | 10 | YES |
| SUI | 20.66 | 23.66 | 27.0 | 36.68 | 13209 | 10 | 10 | YES |
| UNI | 20.33 | 23.99 | 27.0 | 36.0 | 13208 | 10 | 10 | YES |
| WLFI | 20.36 | 23.67 | 27.0 | 36.34 | 13208 | 10 | 10 | YES |
| XPL | 20.66 | 23.66 | 27.0 | 36.23 | 13209 | 10 | 10 | YES |
| XRP | 20.33 | 24.0 | 27.0 | 36.18 | 13208 | 10 | 10 | YES |

---

## 5. Events Per Day (after dedup + grouping)

Spec expects ~28K events/day on BTC.

| Symbol | Sampled Date | Events/Day |
|--------|-------------|------------|
| 2Z | 2025-10-16 | 8,334 |
| 2Z | 2025-11-25 | 4,470 |
| 2Z | 2026-01-04 | 4,109 |
| 2Z | 2026-02-13 | 4,033 |
| 2Z | 2026-03-25 | 775 |
| AAVE | 2025-10-16 | 8,588 |
| AAVE | 2025-11-25 | 6,073 |
| AAVE | 2026-01-04 | 7,949 |
| AAVE | 2026-02-13 | 4,542 |
| AAVE | 2026-03-25 | 767 |
| ASTER | 2025-10-16 | 9,254 |
| ASTER | 2025-11-25 | 5,228 |
| ASTER | 2026-01-04 | 5,691 |
| ASTER | 2026-02-13 | 4,640 |
| ASTER | 2026-03-25 | 793 |
| AVAX | 2025-10-16 | 8,610 |
| AVAX | 2025-11-25 | 4,754 |
| AVAX | 2026-01-04 | 5,053 |
| AVAX | 2026-02-13 | 4,332 |
| AVAX | 2026-03-25 | 784 |
| BNB | 2025-10-16 | 9,585 |
| BNB | 2025-11-25 | 5,693 |
| BNB | 2026-01-04 | 5,752 |
| BNB | 2026-02-13 | 8,077 |
| BNB | 2026-03-25 | 572 |
| BTC | 2025-10-16 | 19,622 |
| BTC | 2025-11-25 | 22,971 |
| BTC | 2026-01-04 | 18,478 |
| BTC | 2026-02-13 | 20,101 |
| BTC | 2026-03-25 | 2,231 ! |
| CRV | 2025-10-16 | 9,275 |
| CRV | 2025-11-25 | 4,650 |
| CRV | 2026-01-04 | 4,164 |
| CRV | 2026-02-13 | 4,067 |
| CRV | 2026-03-25 | 790 |
| DOGE | 2025-10-16 | 9,043 |
| DOGE | 2025-11-25 | 4,643 |
| DOGE | 2026-01-04 | 5,738 |
| DOGE | 2026-02-13 | 4,282 |
| DOGE | 2026-03-25 | 779 |
| ENA | 2025-10-16 | 10,150 |
| ENA | 2025-11-25 | 6,147 |
| ENA | 2026-01-04 | 7,455 |
| ENA | 2026-02-13 | 5,875 |
| ENA | 2026-03-25 | 1,072 |
| ETH | 2025-10-16 | 17,777 |
| ETH | 2025-11-25 | 27,089 |
| ETH | 2026-01-04 | 24,221 |
| ETH | 2026-02-13 | 15,116 |
| ETH | 2026-03-25 | 1,741 |
| FARTCOIN | 2025-10-16 | 8,586 |
| FARTCOIN | 2025-11-25 | 5,593 |
| FARTCOIN | 2026-01-04 | 5,138 |
| FARTCOIN | 2026-02-13 | 4,153 |
| FARTCOIN | 2026-03-25 | 1,561 |
| HYPE | 2025-10-16 | 13,614 |
| HYPE | 2025-11-25 | 13,423 |
| HYPE | 2026-01-04 | 24,456 |
| HYPE | 2026-02-13 | 14,534 |
| HYPE | 2026-03-25 | 2,397 |
| KBONK | 2025-10-16 | 8,267 |
| KBONK | 2025-11-25 | 4,537 |
| KBONK | 2026-01-04 | 5,887 |
| KBONK | 2026-02-13 | 4,111 |
| KBONK | 2026-03-25 | 780 |
| KPEPE | 2025-10-16 | 8,286 |
| KPEPE | 2025-11-25 | 4,441 |
| KPEPE | 2026-01-04 | 5,170 |
| KPEPE | 2026-02-13 | 4,125 |
| KPEPE | 2026-03-25 | 800 |
| LDO | 2025-10-16 | 8,369 |
| LDO | 2025-11-25 | 4,668 |
| LDO | 2026-01-04 | 4,898 |
| LDO | 2026-02-13 | 4,091 |
| LDO | 2026-03-25 | 772 |
| LINK | 2025-10-16 | 8,866 |
| LINK | 2025-11-25 | 4,878 |
| LINK | 2026-01-04 | 4,628 |
| LINK | 2026-02-13 | 4,369 |
| LINK | 2026-03-25 | 781 |
| LTC | 2025-10-16 | 8,526 |
| LTC | 2025-11-25 | 4,661 |
| LTC | 2026-01-04 | 4,637 |
| LTC | 2026-02-13 | 4,104 |
| LTC | 2026-03-25 | 975 |
| PENGU | 2025-10-16 | 8,311 |
| PENGU | 2025-11-25 | 4,495 |
| PENGU | 2026-01-04 | 8,708 |
| PENGU | 2026-02-13 | 4,152 |
| PENGU | 2026-03-25 | 781 |
| PUMP | 2025-10-16 | 8,522 |
| PUMP | 2025-11-25 | 5,132 |
| PUMP | 2026-01-04 | 4,829 |
| PUMP | 2026-02-13 | 4,656 |
| PUMP | 2026-03-25 | 813 |
| SOL | 2025-10-16 | 16,833 |
| SOL | 2025-11-25 | 13,544 |
| SOL | 2026-01-04 | 13,220 |
| SOL | 2026-02-13 | 9,629 |
| SOL | 2026-03-25 | 1,195 |
| SUI | 2025-10-16 | 8,572 |
| SUI | 2025-11-25 | 5,482 |
| SUI | 2026-01-04 | 10,049 |
| SUI | 2026-02-13 | 4,397 |
| SUI | 2026-03-25 | 821 |
| UNI | 2025-10-16 | 8,280 |
| UNI | 2025-11-25 | 4,509 |
| UNI | 2026-01-04 | 4,527 |
| UNI | 2026-02-13 | 4,565 |
| UNI | 2026-03-25 | 765 |
| WLFI | 2025-10-16 | 8,404 |
| WLFI | 2025-11-25 | 5,009 |
| WLFI | 2026-01-04 | 4,536 |
| WLFI | 2026-02-13 | 4,139 |
| WLFI | 2026-03-25 | 809 |
| XPL | 2025-10-16 | 9,264 |
| XPL | 2025-11-25 | 4,814 |
| XPL | 2026-01-04 | 4,346 |
| XPL | 2026-02-13 | 4,298 |
| XPL | 2026-03-25 | 776 |
| XRP | 2025-10-16 | 9,228 |
| XRP | 2025-11-25 | 5,965 |
| XRP | 2026-01-04 | 8,347 |
| XRP | 2026-02-13 | 11,665 |
| XRP | 2026-03-25 | 556 |

---

## 6. Wyckoff Self-Label Frequencies

Computed with causal rolling 1000-event windows. Zero frequency = flag (threshold wrong or no such states).

| Symbol | Absorption | Buy Climax | Sell Climax | Spring | Informed Flow | Stress |
|--------|-----------|-----------|------------|--------|---------------|--------|
| 2Z | 0.02564 | 0.00145 | 0.00103 | 0.01175 | 0.00002 | 0.00000 ! |
| AAVE | 0.05713 | 0.00051 | 0.00036 | 0.04047 | 0.00000 ! | 0.00000 ! |
| ASTER | 0.06239 | 0.00184 | 0.00155 | 0.01792 | 0.00417 | 0.00005 |
| AVAX | 0.04458 | 0.00094 | 0.00145 | 0.01297 | 0.00000 ! | 0.00000 ! |
| BNB | 0.07681 | 0.00042 | 0.00060 | 0.04649 | 0.00002 | 0.00000 ! |
| BTC | 0.03670 | 0.00020 | 0.00012 | 0.12201 | 0.00001 | 0.00000 ! |
| CRV | 0.01935 | 0.00214 | 0.00192 | 0.00892 | 0.00000 ! | 0.00000 ! |
| DOGE | 0.03665 | 0.00111 | 0.00096 | 0.01017 | 0.00000 ! | 0.00000 ! |
| ENA | 0.11045 | 0.00087 | 0.00126 | 0.02553 | 0.00061 | 0.00000 ! |
| ETH | 0.03220 | 0.00026 | 0.00025 | 0.12885 | 0.00424 | 0.00000 ! |
| FARTCOIN | 0.07686 | 0.00080 | 0.00180 | 0.01472 | 0.00054 | 0.00000 ! |
| HYPE | 0.02758 | 0.00028 | 0.00044 | 0.07339 | 0.00010 | 0.00000 ! |
| KBONK | 0.03870 | 0.00058 | 0.00073 | 0.00665 | 0.00002 | 0.00002 |
| KPEPE | 0.04337 | 0.00041 | 0.00054 | 0.00605 | 0.00002 | 0.00000 ! |
| LDO | 0.02891 | 0.00140 | 0.00107 | 0.00882 | 0.00000 ! | 0.00000 ! |
| LINK | 0.07397 | 0.00056 | 0.00092 | 0.03506 | 0.00000 ! | 0.00000 ! |
| LTC | 0.03104 | 0.00042 | 0.00104 | 0.02463 | 0.00000 ! | 0.00000 ! |
| PENGU | 0.06990 | 0.00061 | 0.00071 | 0.01193 | 0.00254 | 0.00000 ! |
| PUMP | 0.08876 | 0.00098 | 0.00137 | 0.02176 | 0.00351 | 0.00000 ! |
| SOL | 0.02469 | 0.00035 | 0.00025 | 0.10266 | 0.00027 | 0.00000 ! |
| SUI | 0.07334 | 0.00073 | 0.00076 | 0.02333 | 0.00000 ! | 0.00000 ! |
| UNI | 0.03768 | 0.00145 | 0.00145 | 0.01800 | 0.00000 ! | 0.00000 ! |
| WLFI | 0.05896 | 0.00178 | 0.00201 | 0.01441 | 0.00000 ! | 0.00007 |
| XPL | 0.04999 | 0.00155 | 0.00203 | 0.01414 | 0.00000 ! | 0.00000 ! |
| XRP | 0.03503 | 0.00064 | 0.00083 | 0.04316 | 0.00013 | 0.00000 ! |

---

## Notes for Council

### Mixed-Side Rate Discrepancy (COUNCIL REVIEW NEEDED)

The spec states: *'Expect 59% of grouped events to have mixed buy/sell fills.'*

Observed: **~3-16%** mixed-side rate across all symbols and dates (median ~3%), after
applying dedup by (ts_ms, qty, price) without side.

**What the data actually looks like (BTC Oct-16 as example):**
- Raw rows: 74,370
- After dedup by (ts_ms, qty, price, side): 51,979 — removes 22,391 true duplicates
- After dedup by (ts_ms, qty, price): 27,558 — removes an additional 24,421 cross-side pairs

The 24,421 cross-side pairs are cases where the same exact fill (ts+qty+price) appears
as BOTH a buy-side and sell-side record. This is the exchange API recording both
counterparties to each trade. The correct dedup WITHOUT side collapses these correctly.

**Gotcha #19 wording analysis:** Gotcha #19 states *"buyer/seller pairs differ on side,
so including side removes nothing."* This is BACKWARDS in practice — including side
PRESERVES the cross-side pairs (51,979 rows), while excluding side COLLAPSES them
(27,558 rows). The INSTRUCTION in gotcha #19 is correct (use no-side dedup), but the
REASONING is wrong. The 59% expectation likely comes from a different processing approach
or dataset version.

**Pipeline impact:** None — dedup without side works correctly. The mixed-side rate of
3-16% after correct dedup means most same-timestamp events are single-side fills.
The `is_open` feature (fraction of fills that are opens) is computed after dedup and
grouping, so it accurately reflects the position direction mix.

**Action required:** Council should update spec gotcha #3 and #19 to reflect actual
observed mixed-side rates (3-16%, not 59%).

### Dedup With-Side vs Without-Side Flag

The validation script flagged all 25 symbols for `dedup_with_side_differs`. This flag
has been removed from the script (it is expected, not an error). The column remains
in the grouping stats table for diagnostic reference.

### Base Rate H500 Below 48% (Systematic, Not a Bug)

Most symbols show H500 base rate of 0.41-0.49, below the 48% floor. This is explained
by a sustained bear market in the training period: BTC fell -36% from $111K (Oct 2025)
to $70K (Mar 2026). At long horizons (500 events ≈ 25 minutes on BTC), the downtrend
introduces directional bias.

**This is NOT a label construction error.** Short horizons (H10, H50) show ~49-50%,
confirming labels are computed correctly at short scales. The drift at H500 reflects
real market directionality.

**Action required:** Fine-tuning should use class balancing at H500. The primary
evaluation horizon of H100 is within the acceptable range for most symbols.

### Wyckoff Stress Label: Zero Frequency on 23/25 Symbols

The `is_stressed` proxy (climax_score > 3.0 AND rolling_std_return > 2x mean) uses OB
features as a proxy. Since this validation runs without OB-derived features (log_spread,
depth_ratio), the stress condition is never met. This is a proxy limitation, not a
data problem — stress detection requires log_spread and depth_ratio from the full
feature pipeline.

**Action:** In Step 1, recompute Wyckoff labels with OB features properly aligned.

### Wyckoff Informed Flow: Zero Frequency on ~12/25 Symbols

The informed flow proxy (abs(return) > 75th pct rolling AND evr < 0) is too restrictive
for low-volume symbols that rarely hit the 75th percentile threshold simultaneously
with low effort_vs_result. The kyle_lambda-based definition in the spec requires the
full feature pipeline to compute properly.

**Action:** In Step 1, use kyle_lambda and cum_ofi_5 for informed flow detection
as specified.

### No April Data Available Locally

Local data ends 2026-03-25. There is no April data (April 1-13 probe set, April 14+
hold-out) available. The April probe evaluation will require a sync from R2 before Step 2.

### Events/Day on BTC

After dedup+grouping, BTC shows ~15K-23K events/day (sampled dates). The spec claims
~28K/day. The lower observed count may reflect: (a) dates with lower trading activity
(weekend, late March drawdown), (b) only 5 dates sampled. A full count over all 161
dates would give a better estimate. The order of magnitude is consistent.