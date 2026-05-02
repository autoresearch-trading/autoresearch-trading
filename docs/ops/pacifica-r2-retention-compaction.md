# Pacifica R2 Retention and Cold-Compaction Policy

Updated: 2026-05-02

## Current stance

R2 is the durable raw archive for the Pacifica full-fidelity collector. Fly `/data` is only a bounded spool.

Do not enable remote deletion just because objects are old. Raw R2 expiry is allowed only after a verified compacted/cold archive exists and a separate destructive apply step is explicitly approved.

## What is already bounded

Fly is bounded by:

- compact payload mode: `PACIFICA_FULL_FIDELITY_RAW_PAYLOAD_MODE=compact`;
- gzip raw chunks: `.jsonl.gz`;
- free disk guard: `PACIFICA_FULL_FIDELITY_MIN_FREE_DISK_GB=10` on Fly;
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

Use:

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

## Future implementation path

1. Keep raw full-fidelity in R2 while the archive matures.
2. Add a compacted cold archive builder once there is enough data to justify compaction.
3. Produce a compaction manifest and verification report.
4. Run `scripts/plan_pacifica_r2_retention.py` against an R2 inventory enriched with `compacted_verified` and `manifest_verified` flags.
5. Review candidate objects.
6. Only then add a separate destructive apply tool, protected by explicit flags and a reviewed input file.

## Do not do

- Do not configure R2 bucket lifecycle expiry directly on `raw/pacifica/full_fidelity` until compaction/manifest gates exist.
- Do not delete raw R2 objects by age alone.
- Do not use destructive `rclone sync` from Fly or local machines.
- Do not prune local laptop data unless explicitly enabled separately.
