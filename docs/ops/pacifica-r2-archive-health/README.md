# Pacifica R2 Raw Archive Health

No R2 writes or deletes were executed. This is a read-only diagnostic report from an object inventory snapshot plus optional read-only sample downloads.

Generated at: `2026-05-13T15:19:05.678772+00:00`
Inventory CSV: `data/ops/pacifica-r2-archive-health/bounded_live_inventory_20260513T151734Z.csv`
Raw prefix: `raw/pacifica/full_fidelity/`
Remote base for sample reads: `r2:pacifica-trading-data`

Inventory scope for this committed run: bounded live sample, not a full-bucket proof. The CSV was built from line-oriented `rclone lsf --recursive --files-only --format pst --separator ';'` listings for six high-signal prefixes over today/yesterday: `bbo/BTC`, `book/ETH`, `trades/BTC`, `mark_price_candle/ICP`, `prices/BTC`, and `candle/BTC`. A full recursive live listing was attempted with stdout streamed to `data/ops/pacifica-r2-archive-health/r2_raw_inventory_20260513T145212Z.lsf`, exceeded the 600s foreground cap, and was marked `.partial_timeout` without being used. A second background `--fast-list` attempt was killed after remaining too slow and was marked `.partial_killed`; do not treat either partial file as evidence.

Current interpretation: freshness is green in this bounded sample, but the newest sampled `run-20260513T122511Z` payloads exposed a sidecar lag. Missing sidecars are reported for sampled prefixes, and remote gzip verification fails for sampled payloads whose sibling `.sha256` object is not yet present. This should be rechecked after the fresh sidecar-repair lane runs; it is not a deletion/retention problem.

## Summary

- OK: False
- Failures: ['R2_GZIP_SAMPLE_VERIFICATION_FAILED', 'R2_SIDECAR_MISSING']
- Raw-prefix objects: 548
- Payload objects: 276
- Sidecar objects: 272
- Payload bytes: 248623683
- Latest payload mod time: 2026-05-13T13:03:10+00:00
- Latest payload age minutes: 135.73
- Stale threshold minutes: 180.0
- Latest payload freshness OK: True
- Missing payload sidecars: 4
- Orphan sidecars: 0
- Active current-hour payload objects: 0
- Distinct channels: 6
- Distinct dates: 2
- Distinct symbols: 3
- Remote gzip sample verification: 5 OK / 8 sampled
- Remote gzip sample failures: 3
- Gzip audit local root: `not requested`
- Gzip-readable local payloads: 0 / 0
- Bad gzip local payloads: 0
- Missing local payloads for gzip audit: 0

## Prefix summary

| channel | date | payload_objects | payload_bytes | symbols | latest_mod_time |
| --- | --- | --- | --- | --- | --- |
| bbo | 2026-05-13 | 14 | 52391496 | 1 | 2026-05-13T13:03:10+0000 |
| book | 2026-05-13 | 14 | 17057505 | 1 | 2026-05-13T13:03:10+0000 |
| candle | 2026-05-13 | 13 | 3197982 | 1 | 2026-05-13T12:23:17+0000 |
| mark_price_candle | 2026-05-13 | 13 | 27816307 | 1 | 2026-05-13T12:24:51+0000 |
| prices | 2026-05-13 | 14 | 1096520 | 1 | 2026-05-13T13:03:09+0000 |
| trades | 2026-05-13 | 14 | 338284 | 1 | 2026-05-13T12:58:06+0000 |
| bbo | 2026-05-12 | 27 | 63298129 | 1 | 2026-05-12T23:59:55+0000 |
| book | 2026-05-12 | 27 | 27683776 | 1 | 2026-05-12T23:59:55+0000 |
| candle | 2026-05-12 | 46 | 5366038 | 1 | 2026-05-13T00:00:23+0000 |
| mark_price_candle | 2026-05-12 | 40 | 48128226 | 1 | 2026-05-12T23:59:55+0000 |
| prices | 2026-05-12 | 27 | 1742846 | 1 | 2026-05-12T23:59:46+0000 |
| trades | 2026-05-12 | 27 | 506574 | 1 | 2026-05-12T23:59:11+0000 |

## Channel coverage

| channel | payload_objects | payload_bytes | dates | symbols | latest_mod_time |
| --- | --- | --- | --- | --- | --- |
| candle | 59 | 8564020 | 2 | 1 | 2026-05-13T12:23:17+0000 |
| mark_price_candle | 53 | 75944533 | 2 | 1 | 2026-05-13T12:24:51+0000 |
| bbo | 41 | 115689625 | 2 | 1 | 2026-05-13T13:03:10+0000 |
| book | 41 | 44741281 | 2 | 1 | 2026-05-13T13:03:10+0000 |
| prices | 41 | 2839366 | 2 | 1 | 2026-05-13T13:03:09+0000 |
| trades | 41 | 844858 | 2 | 1 | 2026-05-13T12:58:06+0000 |

## Date coverage

| date | payload_objects | payload_bytes | channels | symbols | latest_mod_time |
| --- | --- | --- | --- | --- | --- |
| 2026-05-13 | 82 | 101898094 | 6 | 3 | 2026-05-13T13:03:10+0000 |
| 2026-05-12 | 194 | 146725589 | 6 | 3 | 2026-05-13T00:00:23+0000 |

## Channel/date/symbol coverage

Full coverage is written to `channel_date_symbol_coverage.csv`; the table below shows the newest 25 rows.

| channel | date | symbol | payload_objects | payload_bytes | hours | latest_mod_time |
| --- | --- | --- | --- | --- | --- | --- |
| bbo | 2026-05-13 | BTC | 14 | 52391496 | 13 | 2026-05-13T13:03:10+0000 |
| book | 2026-05-13 | ETH | 14 | 17057505 | 13 | 2026-05-13T13:03:10+0000 |
| candle | 2026-05-13 | BTC | 13 | 3197982 | 13 | 2026-05-13T12:23:17+0000 |
| mark_price_candle | 2026-05-13 | ICP | 13 | 27816307 | 13 | 2026-05-13T12:24:51+0000 |
| prices | 2026-05-13 | BTC | 14 | 1096520 | 13 | 2026-05-13T13:03:09+0000 |
| trades | 2026-05-13 | BTC | 14 | 338284 | 13 | 2026-05-13T12:58:06+0000 |
| bbo | 2026-05-12 | BTC | 27 | 63298129 | 24 | 2026-05-12T23:59:55+0000 |
| book | 2026-05-12 | ETH | 27 | 27683776 | 24 | 2026-05-12T23:59:55+0000 |
| candle | 2026-05-12 | BTC | 46 | 5366038 | 24 | 2026-05-13T00:00:23+0000 |
| mark_price_candle | 2026-05-12 | ICP | 40 | 48128226 | 24 | 2026-05-12T23:59:55+0000 |
| prices | 2026-05-12 | BTC | 27 | 1742846 | 24 | 2026-05-12T23:59:46+0000 |
| trades | 2026-05-12 | BTC | 27 | 506574 | 24 | 2026-05-12T23:59:11+0000 |

## Latest payload objects

| key | size_bytes | mod_time | channel | symbol | date | hour |
| --- | --- | --- | --- | --- | --- | --- |
| raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | 1032244 | 2026-05-13T13:03:10+00:00 | book | ETH | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | 2383252 | 2026-05-13T13:03:10+00:00 | bbo | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=prices/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | 64516 | 2026-05-13T13:03:09+00:00 | prices | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | 42047 | 2026-05-13T12:58:06+00:00 | trades | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | 488141 | 2026-05-13T12:24:52+00:00 | book | ETH | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | 430504 | 2026-05-13T12:24:52+00:00 | bbo | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | 697464 | 2026-05-13T12:24:51+00:00 | mark_price_candle | ICP | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=00/run-20260513T000011Z.jsonl.gz | 9621797 | 2026-05-13T12:24:51+00:00 | mark_price_candle | ICP | 2026-05-13 | 00 |
| raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=08/run-20260513T000011Z.jsonl.gz | 3521141 | 2026-05-13T12:24:51+00:00 | mark_price_candle | ICP | 2026-05-13 | 08 |
| raw/pacifica/full_fidelity/channel=prices/symbol=BTC/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | 29745 | 2026-05-13T12:24:44+00:00 | prices | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=candle/symbol=BTC/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | 67007 | 2026-05-13T12:23:17+00:00 | candle | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=candle/symbol=BTC/date=2026-05-13/hour=08/run-20260513T000011Z.jsonl.gz | 522639 | 2026-05-13T12:23:17+00:00 | candle | BTC | 2026-05-13 | 08 |
| raw/pacifica/full_fidelity/channel=candle/symbol=BTC/date=2026-05-13/hour=00/run-20260513T000011Z.jsonl.gz | 1062511 | 2026-05-13T12:23:17+00:00 | candle | BTC | 2026-05-13 | 00 |
| raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | 7862 | 2026-05-13T12:23:17+00:00 | trades | BTC | 2026-05-13 | 12 |
| raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-13/hour=11/run-20260513T000011Z.jsonl.gz | 975934 | 2026-05-13T12:09:24+00:00 | book | ETH | 2026-05-13 | 11 |
| raw/pacifica/full_fidelity/channel=prices/symbol=BTC/date=2026-05-13/hour=11/run-20260513T000011Z.jsonl.gz | 58885 | 2026-05-13T12:09:23+00:00 | prices | BTC | 2026-05-13 | 11 |
| raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=11/run-20260513T000011Z.jsonl.gz | 932005 | 2026-05-13T12:09:18+00:00 | mark_price_candle | ICP | 2026-05-13 | 11 |
| raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=10/run-20260513T000011Z.jsonl.gz | 1449716 | 2026-05-13T12:09:18+00:00 | mark_price_candle | ICP | 2026-05-13 | 10 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=11/run-20260513T000011Z.jsonl.gz | 2546672 | 2026-05-13T12:09:13+00:00 | bbo | BTC | 2026-05-13 | 11 |
| raw/pacifica/full_fidelity/channel=candle/symbol=BTC/date=2026-05-13/hour=11/run-20260513T000011Z.jsonl.gz | 84801 | 2026-05-13T12:04:27+00:00 | candle | BTC | 2026-05-13 | 11 |

## Remote gzip sample verification

This optional sample uses read-only `rclone cat` calls for payloads and sibling `.sha256` sidecars, then verifies SHA-256 and gzip decompression locally. It does not write or delete R2 objects.

| key | status | rows_read | size_bytes | sha256 | sidecar_sha256 | sha256_matches_sidecar | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | sidecar_missing_or_invalid | 2179 | 1032244 | 7bd59d1410cd6e767b38974b154b53978f5a93c7cc8babaaf6228efb2162dc52 |  | False |  |
| raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | sidecar_missing_or_invalid | 8272 | 2383252 | b5902346fb383dba2774c2851d6d9c8369bc141751e69885de35d96bb79d3cf1 |  | False |  |
| raw/pacifica/full_fidelity/channel=prices/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | sidecar_missing_or_invalid | 184 | 64516 | 88abc1b0e2f2e576702cf61de2b72c3f51f1587970ae8ec0dc77550f5c380449 |  | False |  |
| raw/pacifica/full_fidelity/channel=trades/symbol=BTC/date=2026-05-13/hour=12/run-20260513T122511Z.jsonl.gz | ok | 201 | 42047 | d436c2d9fd94561bca97e4bf743117643cd979fb0ac8c20b0db57a102ae6a8c7 | d436c2d9fd94561bca97e4bf743117643cd979fb0ac8c20b0db57a102ae6a8c7 | True |  |
| raw/pacifica/full_fidelity/channel=mark_price_candle/symbol=ICP/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | ok | 2305 | 697464 | 6267b405eddef90edb80ffdde301788aa60c7313a55041016ff8f571c50cde22 | 6267b405eddef90edb80ffdde301788aa60c7313a55041016ff8f571c50cde22 | True |  |
| raw/pacifica/full_fidelity/channel=candle/symbol=BTC/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | ok | 229 | 67007 | b91ba99f3d17504e5dfb5129bf4b8d1b3ed6d89a4fd8dc5b2a731407287a277c | b91ba99f3d17504e5dfb5129bf4b8d1b3ed6d89a4fd8dc5b2a731407287a277c | True |  |
| raw/pacifica/full_fidelity/channel=book/symbol=ETH/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | ok | 1027 | 488141 | a0c4005c7bdcdaf184ef9be12e42d51fdf82263d48c6ef1c4376ff4ffe4b8428 | a0c4005c7bdcdaf184ef9be12e42d51fdf82263d48c6ef1c4376ff4ffe4b8428 | True |  |
| raw/pacifica/full_fidelity/channel=bbo/symbol=BTC/date=2026-05-13/hour=12/run-20260513T000011Z.jsonl.gz | ok | 1492 | 430504 | 0ab6c0174913bf6db0b4075bd935ae72d44b6b66a367a0aef6d6bd59cf45065c | 0ab6c0174913bf6db0b4075bd935ae72d44b6b66a367a0aef6d6bd59cf45065c | True |  |

## Local Gzip integrity audit

This optional audit is local-only and only checks rehydrated payloads under `--local-raw-root`; it does not read, write, or delete remote R2 objects.

_No rows._

## Output files

- `prefix_summary.csv` — payload counts/bytes by channel/date.
- `channel_coverage.csv` — payload counts/bytes/dates/symbols by channel.
- `date_coverage.csv` — payload counts/bytes/channels/symbols by date.
- `channel_date_symbol_coverage.csv` — payload counts/bytes/hours by channel/date/symbol.
- `missing_sidecars.csv` — payloads without matching `.sha256` sidecars.
- `orphan_sidecars.csv` — `.sha256` sidecars without matching payloads.
- `active_hour_objects.csv` — current UTC hour payloads; these should normally be absent for sealed-chunk uploads.
- `latest_remote_objects.csv` — newest payload objects in this inventory snapshot.
- `remote_gzip_sample_verification.csv` — optional read-only remote payload+sidecar SHA/gzip sample verification.
- `gzip_integrity_audit.csv` — optional local gzip decompression status for rehydrated payloads.
