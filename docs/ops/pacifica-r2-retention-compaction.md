# Pacifica R2 Retention and Cold-Compaction Policy

Updated: 2026-05-02

## Current stance

R2 is the durable raw archive for the Pacifica full-fidelity collector. Fly `/data` is only a bounded spool.

Do not enable remote deletion just because objects are old. Raw R2 expiry is allowed only after a verified compacted/cold archive exists and a separate destructive apply step is explicitly approved.

## What is already bounded

Fly is bounded by:

- compact payload mode: `PACIFICA_FULL_FIDELITY_RAW_PAYLOAD_MODE=compact`;
- gzip raw chunks: `.jsonl.gz`;
- free disk guard: `PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB=50` on Fly;
- lifecycle loop: `PACIFICA_FULL_FIDELITY_LIFECYCLE_INTERVAL_S=1800`;
- local spool retention: `PACIFICA_FULL_FIDELITY_RETENTION_DAYS=1`;
- Fly local prune execute: `PACIFICA_R2_PRUNE_EXECUTE=1`;
- prune rule: only local files marked R2-verified by byte size plus `.sha256` sidecar can be pruned.

## What this policy bounds

This policy bounds future R2 growth by adding explicit gates before raw object expiry.

Default policy:

| Age | R2 raw action |
| --- | --- |
| 0-60 days | Keep raw full-fidelity `.jsonl.gz` and `.sha256` sidecars. |
| 60+ days | Build/verify compacted cold archive before considering raw expiry. |
| 90+ days | Raw objects may become eligible for expiry review only if compacted archive and manifest checks pass. |

These defaults are intentionally conservative while the archive is young.

## Required gates before any R2 raw expiry

A raw R2 object can be considered for deletion only if all gates pass:

1. Same channel/symbol/date/hour coverage exists in a compacted/cold archive.
2. The compacted archive has a manifest with row counts, byte totals, and checksums.
3. The raw object has a `.sha256` sidecar and was previously verified by the lifecycle DB.
4. A restore sample from the compacted archive has rebuilt silver successfully.
5. Regime-state and diagnostic reports can be rebuilt from the retained/compacted data.
6. The deletion candidate list is reviewed as a CSV/artifact.
7. Diego explicitly approves a separate destructive apply step.

## Non-destructive planner

On Fly, the ops watchdog runs this daily by default:

1. `rclone lsjson r2:pacifica-trading-data/raw/pacifica/full_fidelity --recursive --files-only`
2. `scripts/pacifica_r2_inventory.py` to convert the R2 listing into `/data/ops/r2_inventory.csv`
3. `scripts/plan_pacifica_r2_retention.py` to write `/data/ops/pacifica-r2-retention/`
4. `rclone copy /data/ops r2:pacifica-trading-data/ops/pacifica/full_fidelity/watchdogs/latest` when `PACIFICA_OPS_UPLOAD_REPORTS=1`

This gives the R2 compression/retention policy its own always-on operational path on the archival appliance. It is still non-destructive: the planner produces review artifacts only and never deletes raw objects.

Manual usage remains:

```bash
python scripts/plan_pacifica_r2_retention.py \
  --inventory-csv path/to/r2_inventory.csv \
  --out-dir docs/ops/pacifica-r2-retention
```

The planner writes:

- `README.md`
- `r2_retention_plan.csv`
- `r2_retention_summary.csv`
- `r2_retention_policy.csv`

It does not run `rclone delete`, does not write delete commands, and does not mutate R2.

## Current live R2 size smoke check

Latest non-destructive size check:

```text
rclone size r2:pacifica-trading-data/raw/pacifica/full_fidelity --json
count=12688
bytes=19203724868
```

Interpretation: R2 is currently around 19.2 GB for this prefix. This is acceptable for the young archive; no remote expiry should run yet.

## Read-only R2 archive health checker

Initial Cloudflare/R2-backed read-only health tooling now exists:

```bash
python scripts/check_pacifica_r2_archive_health.py \
  --inventory-csv path/to/r2_inventory.csv \
  --out-dir docs/ops/pacifica-r2-archive-health

# Optional local-only gzip readability audit over rehydrated payloads from the same inventory.
python scripts/check_pacifica_r2_archive_health.py \
  --inventory-csv path/to/r2_inventory.csv \
  --out-dir docs/ops/pacifica-r2-archive-health \
  --local-raw-root path/to/rehydrated/raw/pacifica_full_fidelity
```

The checker writes:

- `README.md` — read-only summary with payload/sidecar counts, latest payload time, and active-hour warnings;
- `prefix_summary.csv` — payload counts/bytes by channel/date;
- `missing_sidecars.csv` — payloads without matching `.sha256` sidecars;
- `orphan_sidecars.csv` — `.sha256` sidecars without matching payloads;
- `active_hour_objects.csv` — current UTC hour payloads, which should normally be absent after the active-hour upload fix;
- `latest_remote_objects.csv` — newest payload objects in the sampled inventory;
- `gzip_integrity_audit.csv` — optional local decompression/readability status for rehydrated payloads when `--local-raw-root` is provided.

A bounded Cloudflare MCP spot check on 2026-05-02 listed 20 objects from `pacifica-trading-data/raw/pacifica/full_fidelity/` and wrote `docs/ops/pacifica-r2-archive-health/`. That sample found 10 payload objects, 10 sidecars, 0 missing sidecars, 0 orphan sidecars, and 0 active current-hour payload objects. This was only a bounded prefix sample, not a full-bucket proof.

## Cold archive builder / verifier

Initial non-destructive tooling now exists:

```bash
python scripts/build_pacifica_cold_archive.py build \
  --raw-root path/to/rehydrated/raw/pacifica_full_fidelity \
  --out-dir docs/ops/pacifica-cold-archive

python scripts/build_pacifica_cold_archive.py verify \
  --manifest docs/ops/pacifica-cold-archive/manifest.csv \
  --raw-root path/to/rehydrated/raw/pacifica_full_fidelity

python scripts/build_pacifica_cold_archive.py restore-sample \
  --manifest docs/ops/pacifica-cold-archive/manifest.csv \
  --raw-root path/to/rehydrated/raw/pacifica_full_fidelity \
  --out-dir docs/ops/pacifica-cold-archive/restore-sample \
  --max-sources 5

python scripts/build_pacifica_cold_archive.py restore-raw-cache \
  --manifest docs/ops/pacifica-cold-archive/manifest.csv \
  --out-raw-root data/pacifica_cold_restored_raw_sample \
  --original-raw-root path/to/rehydrated/raw/pacifica_full_fidelity
```

The builder writes:

- `archive_part-00000.parquet` — lossless rows with `source_key`, `line_number`, partition columns, and original `raw_json` line text;
- `manifest.csv` — source file size, SHA-256, row count, archive file, archive size, archive SHA-256, and verification status;
- `README.md` — local verification summary.

The restore sampler writes:

- `restore-sample/restore_sample_report.csv` — per sampled source, expected/restored/raw row counts and exact line-sequence match status;
- `restore-sample/README.md` — local restore-sampling summary.

The raw-cache restorer writes:

- partitioned `.jsonl.gz` files under the requested output raw root, preserving `channel=/symbol=/date=/hour=` paths from `source_key`;
- `restore_raw_cache_report.csv` — per source expected/restored/raw row counts, row-match status, line-match status, and restored path;
- `README.md` — local cold-to-raw restore summary.

A bounded R2 rehydration sample on 2026-05-02 copied 3 non-current-hour BBO chunks from `pacifica-trading-data/raw/pacifica/full_fidelity/` into gitignored `data/pacifica_r2_rehydrate_sample/`, built `docs/ops/pacifica-cold-archive-sample/`, and verified 3/3 sources, 655/655 rows. Restore sampling matched all 3 sampled sources exactly. `restore-raw-cache` then recreated `data/pacifica_cold_restored_raw_sample/`, and the silver builder read that restored cache into `data/pacifica_silver_from_cold_sample/` with 655 BBO rows for symbol `2Z`. This remains a small proof-of-path, not full archive coverage.

A bounded multi-channel R2 rehydration sample on 2026-05-02 selected 12 non-current-hour chunks, 2 each for `bbo`, `book`, `candle`, `mark_price_candle`, `prices`, and `trades`, using `docs/ops/pacifica-r2-archive-health/multichannel_selected_objects.csv`. It copied payloads and `.sha256` sidecars into gitignored `data/pacifica_r2_rehydrate_multichannel_sample/`, verified 12/12 payload SHA sidecars, and counted 12,250 raw rows. The cold archive at `docs/ops/pacifica-cold-archive-multichannel-sample/` verified 12/12 sources and 12,250/12,250 rows; restore sampling matched all 12 sampled sources; `restore-raw-cache` recreated `data/pacifica_cold_restored_raw_multichannel_sample/` with zero row or line mismatches; silver smoke from that restored cache wrote `data/pacifica_silver_from_cold_multichannel_sample/` with rows: bbo=2,957, book=1,687, candle=4,776, mark_price_candle=1,617, prices=50, trades=1,163. This is still bounded to symbol `2Z`, so it is path validation, not representative archive coverage.

A broader bounded R2 rehydration sample on 2026-05-02 selected 27 non-current-hour chunks across symbols `2Z`, `BTC`, `ETH`, and `SOL`, dates `2026-04-30` and `2026-05-01`, and all six channels. It used `docs/ops/pacifica-r2-archive-health/broader_selected_objects.csv` and copied payloads plus `.sha256` sidecars into gitignored `data/pacifica_r2_rehydrate_broader_sample/`. Local SHA plus gzip decompression checks passed after replacing one sidecar-valid but gzip-CRC-invalid object (`channel=prices/symbol=2Z/date=2026-05-01/run-20260501T101607Z.jsonl.gz`) with a decompressible sibling chunk. This proves `.sha256` sidecar matching alone is not sufficient; retention-grade manifests should include decompression/readability checks. The cold archive at `docs/ops/pacifica-cold-archive-broader-sample/` verified 27/27 sources and 8,313/8,313 rows; restore sampling matched all 27 sampled sources; `restore-raw-cache` recreated `data/pacifica_cold_restored_raw_broader_sample/` with zero row or line mismatches; silver smoke from that restored cache wrote `data/pacifica_silver_from_cold_broader_sample/` with rows: bbo=2,927, book=71, candle=4,794, mark_price_candle=241, prices=23, trades=257.

This remains a local/rehydrated-cache workflow. It does not upload compacted archives, enrich the R2 retention inventory, or delete remote raw objects by itself.

## Future implementation path

1. Keep raw full-fidelity in R2 while the archive matures.
2. Rehydrate bounded R2 partitions locally and run `scripts/check_pacifica_r2_archive_health.py --local-raw-root ...` for sampled/inventory-selected gzip decompression/readability audits; `.sha256` sidecar matching alone can still preserve a gzip-invalid object.
3. Run `scripts/build_pacifica_cold_archive.py` to produce compacted parquet + manifest verification artifacts from gzip-readable sealed raw chunks.
4. Expand restore/silver diagnostics to larger representative samples before treating cold archive manifests as retention-grade.
5. Add R2 upload/copy of verified cold archive artifacts under a separate cold prefix.
6. Enrich `scripts/plan_pacifica_r2_retention.py` inventory rows with `compacted_verified` and `manifest_verified` flags from the verified cold manifest.
7. Review candidate objects.
8. Only then add a separate destructive apply tool, protected by explicit flags and a reviewed input file.

## Do not do

- Do not configure R2 bucket lifecycle expiry directly on `raw/pacifica/full_fidelity` until compaction/manifest gates exist.
- Do not delete raw R2 objects by age alone.
- Do not use destructive `rclone sync` from Fly or local machines.
- Do not prune local laptop data unless explicitly enabled separately.
