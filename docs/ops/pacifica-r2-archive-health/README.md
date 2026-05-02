# Pacifica R2 Raw Archive Health

No R2 writes or deletes were executed. This is a read-only diagnostic report from an object inventory snapshot.

Generated at: `2026-05-02T18:07:20.586728+00:00`
Raw prefix: `raw/pacifica/full_fidelity/`

## Summary

- Raw-prefix objects: 20
- Payload objects: 10
- Sidecar objects: 10
- Payload bytes: 3639082
- Latest payload mod time: 2026-05-01T23:56:02.221000+00:00
- Missing payload sidecars: 0
- Orphan sidecars: 0
- Active current-hour payload objects: 0

## Prefix summary

| channel | date | payload_objects | payload_bytes | symbols | latest_mod_time |
| --- | --- | --- | --- | --- | --- |
| bbo | 2026-05-01 | 9 | 2853956 | 1 | 2026-05-01T23:56:02+0000 |
| bbo | 2026-04-30 | 1 | 785126 | 1 | 2026-05-01T18:32:37+0000 |

## Latest payload objects

| key | size_bytes | mod_time | channel | symbol | date | hour |
| --- | --- | --- | --- | --- | --- | --- |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/hour=23/run-20260501T220740Z.jsonl.gz | 94693 | 2026-05-01T23:56:02.221Z | bbo | 2Z | 2026-05-01 | 23 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/hour=22/run-20260501T220740Z.jsonl.gz | 91067 | 2026-05-01T22:59:16.681Z | bbo | 2Z | 2026-05-01 | 22 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/hour=22/run-20260501T215139Z.jsonl.gz | 287 | 2026-05-01T22:07:53.123Z | bbo | 2Z | 2026-05-01 | 22 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/hour=21/run-20260501T215139Z.jsonl.gz | 34700 | 2026-05-01T22:07:52.315Z | bbo | 2Z | 2026-05-01 | 21 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/hour=21/run-20260501T214645Z.jsonl.gz | 13551 | 2026-05-01T21:51:44.664Z | bbo | 2Z | 2026-05-01 | 21 |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/run-20260501T101607Z.jsonl.gz | 543 | 2026-05-01T18:32:44.152Z | bbo | 2Z | 2026-05-01 |  |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/run-20260501T101433Z.jsonl.gz | 21428 | 2026-05-01T18:32:42.476Z | bbo | 2Z | 2026-05-01 |  |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/run-20260501T034250Z.jsonl.gz | 17788 | 2026-05-01T18:32:40.973Z | bbo | 2Z | 2026-05-01 |  |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-05-01/run-20260430T212120Z.jsonl.gz | 2579899 | 2026-05-01T18:32:39.432Z | bbo | 2Z | 2026-05-01 |  |
| raw/pacifica/full_fidelity/channel=bbo/symbol=2Z/date=2026-04-30/run-20260430T212120Z.jsonl.gz | 785126 | 2026-05-01T18:32:37.240Z | bbo | 2Z | 2026-04-30 |  |

## Output files

- `prefix_summary.csv` — payload counts/bytes by channel/date.
- `missing_sidecars.csv` — payloads without matching `.sha256` sidecars.
- `orphan_sidecars.csv` — `.sha256` sidecars without matching payloads.
- `active_hour_objects.csv` — current UTC hour payloads; these should normally be absent for sealed-chunk uploads.
- `latest_remote_objects.csv` — newest payload objects in this inventory snapshot.
