# Candidate Silver Duplicate-Key Diagnostics

Generated: 2026-05-14T17:35Z

Context: local-snapshot refresh `20260514T135053Z` completed manifest, candidate silver, and candidate regime builds. This diagnostic captured the original `candidate_silver_duplicate_keys` failure from the old generic-key verifier.

Superseded status: on 2026-05-14T19:09Z, `scripts/verify_pacifica_side_by_side_refresh.py` was fixed to use channel-specific semantic duplicate keys plus exact-payload duplicate metrics via DuckDB/parquet-side aggregation, and the same candidate reran green (`ok=True`, `failures=[]`) under this directory.

## Verification verdict

Current verifier verdict is green for candidate `20260514T135053Z` (`ok=True`, `failures=[]`). Promotion still requires Diego's explicit approval; do not advance `data/ops/pacifica-source-manifest/source_manifest_previous.csv`, overwrite canonical silver/regime, or refresh canonical eligibility from this candidate until that approval is given.

## Build outputs

- Manifest: `data/ops/pacifica-source-manifest/source_manifest_20260514T135053Z.csv`
- Incremental plan: `data/ops/pacifica-source-manifest/incremental_plan_20260514T135053Z.csv`
- Candidate silver: `data/pacifica_silver_partitioned_candidate_20260514T135053Z`
- Candidate regime delta: `data/ops/pacifica-regime-candidate-20260514T135053Z_delta`
- Candidate regime full: `data/ops/pacifica-regime-candidate-20260514T135053Z`
- Verification report: `docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification/README.md`

## High-level counts

Manifest:

```text
manifest_rows=139375
sealed=139299
unsealed_gzip_unreadable=76
plan_rows=139299
manifest_dates=2026-04-30..2026-05-14
```

Candidate silver row counts vs canonical:

```text
prices: canonical=1,039,338 candidate=5,252,223 delta=4,212,885
trades: canonical=91,069 candidate=699,037 delta=607,968
bbo: canonical=12,974,473 candidate=23,669,943 delta=10,695,470
book: canonical=14,137,732 candidate=60,389,166 delta=46,251,434
candle: canonical=1,915,776 candidate=8,152,538 delta=6,236,762
mark_price_candle: canonical=22,446,290 candidate=158,241,147 delta=135,794,857
```

Candidate regime passed duplicate/null key checks:

```text
regime_state candidate rows=1,110,785
regime duplicate_keys_candidate=0
regime key_nulls_candidate=0
```

## Duplicate-key failure details

The old verifier used a generic silver key of `symbol,event_ts_ms,recv_ms` when those columns existed. Under that old key, candidate duplicate counts exceeded canonical on several channels:

```text
prices: candidate=102 canonical=40,875         # lower than canonical; not the blocker by itself
trades: candidate=72,059 canonical=10,816      # increased
bbo: candidate=1,690,847 canonical=1,463,012   # increased
book: candidate=9,885 canonical=1,408,469      # lower than canonical; not the blocker by itself
candle: candidate=1,373,120 canonical=620,408  # increased
mark_price_candle: candidate=30,311,501 canonical=7,291,825  # increased
```

Observed sample patterns suggest this is a mixed issue:

1. Some duplicates are real cross-run duplicate rows that should likely be de-duplicated or handled by source-object overlap rules.
   - Example: `prices/EURUSD` at the same `event_ts_ms` and `recv_ms` appeared identically in both `run-20260430T212120Z` and `run-20260501T034250Z`.
2. Some duplicates are semantically distinct rows that the verifier's generic key is too coarse to judge.
   - `trades`: multiple distinct `history_id` values can share the same `symbol,event_ts_ms,recv_ms`.
   - `bbo`: multiple distinct `order_id` / `last_order_id` updates can share the same `symbol,event_ts_ms,recv_ms`.
   - `candle` and likely `mark_price_candle`: multiple intervals (`1m`, `3m`, `5m`, `15m`, `1h`, etc.) can share the same `symbol,event_ts_ms,recv_ms`.

## Sample evidence

`prices` duplicate sample:

```text
symbol=EURUSD event_ts_ms=1777607031308 recv_ms=1777607031408 n=2
source paths:
  channel=prices/symbol=EURUSD/date=2026-05-01/run-20260430T212120Z.jsonl.gz
  channel=prices/symbol=EURUSD/date=2026-05-01/run-20260501T034250Z.jsonl.gz
values were identical in the sampled rows.
```

`trades` duplicate sample:

```text
symbol=SOL event_ts_ms=1777905531834 recv_ms=1777905832495 n=31
same source object, but rows have distinct history_id values such as 180412339, 180412341, ..., 180412399 and different price/qty values.
```

`bbo` duplicate sample:

```text
symbol=ETH event_ts_ms=1777646154495 recv_ms=1777646162359 n=10
same source object, but rows have distinct order_id / last_order_id and changing ask_qty values.
```

`candle` duplicate sample:

```text
symbol=ASTER event_ts_ms=1777593600000 recv_ms=1777593681888 n=11
same source object, but rows are distinct intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d.
```

## Resolution note

The next safe action listed below has been completed: the verifier now uses channel-specific duplicate-key semantics and exact-payload duplicate metrics, and the existing candidate reran green. Keep the sample evidence as historical root-cause context only. Promotion remains a separate, approval-gated step.

## Historical next safe action

The safe action at the time of this diagnostic was:

- do not promote this candidate until the verifier/key-semantics fix is tested and the side-by-side verifier is rerun green;
- channel-specific duplicate keys:
  - `trades`: include `history_id` where present, else `nonce` plus price/qty/direction fallback.
  - `bbo`: include `order_id` / `last_order_id` when present.
  - `candle` and `mark_price_candle`: include `interval,start_ts_ms,end_ts_ms`.
  - `prices`: keep strict row duplicate detection; investigate/de-dupe cross-run identical snapshots.
  - `book`: keep current checks but verify source-level overlap if needed.
- add a separate exact-row duplicate metric for candidate silver so true duplicate payload overlap is not hidden by wider semantic keys.
- rerun the verifier against the existing candidate artifacts before any canonical promotion.
