# Pacifica Source-Object Manifest and Incremental Side-by-Side Refresh

Updated: 2026-05-13T17:58:50Z

## Purpose

The raw archive is append-only and partitioned as sealed source objects. Rebuilding all silver/regime artifacts from scratch is increasingly expensive and risks confusing active/live chunks with sealed historical data.

This layer adds a manifest keyed by the raw source-object identity:

```text
channel / symbol / date / hour / run
```

A source object is considered processable only when its `.jsonl.gz.sha256` sidecar exists and is valid. Missing sidecars, invalid sidecars, checksum mismatches, or unreadable gzip chunks are treated as unsealed and are excluded from incremental plans.

## New scripts

```text
scripts/build_pacifica_source_manifest.py
  Builds `source_manifest.csv` from local raw JSONL.GZ chunks.
  Diffing a current manifest against a previous manifest yields a plan of only new/changed sealed chunks.
  Incremental silver requires manifests built with `--verify-sha --count-rows` so planned chunks have verified sidecar checksums and readable gzip rows.

scripts/build_pacifica_full_fidelity_silver.py --layout incremental
  Writes deterministic per-source-object silver parquet chunks under:
  channel=<channel>/symbol=<symbol>/date=<date>/hour=<hour>/run=<run>/part.parquet
  Existing canonical silver is not touched. Use `--out-dir` for a side-by-side candidate.

scripts/build_non_hft_regime_state.py --source-plan <incremental_plan.csv>
  Writes `regime_state_delta.parquet` for affected symbol/date partitions only.
  It does not overwrite canonical `regime_state.parquet`.

scripts/verify_pacifica_side_by_side_refresh.py
  Compares canonical vs candidate silver/regime outputs and writes verification CSVs/README.
  Checks row counts, symbol/date/channel coverage, missing/null keys, duplicate-key regressions, and report diffs.
  Missing required key columns in non-empty candidate silver/regime tables fail closed.
  Candidate duplicate keys fail when they exceed canonical baseline duplicate counts; this keeps canonical self-checks useful while still blocking duplicate regressions.
```

## Safe side-by-side flow

Do not point these commands at canonical output directories unless the final verification has already passed and Diego has explicitly approved promotion.

```bash
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
MANIFEST_DIR=data/ops/pacifica-source-manifest
CANDIDATE_SILVER=data/pacifica_silver_partitioned_candidate_${RUN_TS}
CANDIDATE_REGIME=data/ops/pacifica-regime-candidate-${RUN_TS}
VERIFY_DIR=docs/ops/pacifica-incremental-refresh/latest-side-by-side-verification
mkdir -p "$MANIFEST_DIR"

# 1. Build current source-object manifest from local raw cache.
uv run python scripts/build_pacifica_source_manifest.py \
  --raw-dir data/pacifica_full_fidelity \
  --out "$MANIFEST_DIR/source_manifest_${RUN_TS}.csv" \
  --previous "$MANIFEST_DIR/source_manifest_previous.csv" \
  --verify-sha \
  --count-rows \
  --plan-out "$MANIFEST_DIR/incremental_plan_${RUN_TS}.csv"

# 2. Build a side-by-side silver candidate only from new/changed sealed chunks.
#    Optional seed is allowed only when the base silver directory is already
#    partitioned/source-object layout. Flat v1 files like trades.parquet are
#    refused because affected rows cannot be safely removed before adding
#    source-object chunks.
uv run python scripts/build_pacifica_full_fidelity_silver.py \
  --layout incremental \
  --raw-dir data/pacifica_full_fidelity \
  --out-dir "$CANDIDATE_SILVER" \
  --source-manifest "$MANIFEST_DIR/source_manifest_${RUN_TS}.csv" \
  --previous-source-manifest "$MANIFEST_DIR/source_manifest_previous.csv"

# If and only if the canonical/base silver directory has no selected flat
# <channel>.parquet files, add:
#   --base-silver-dir data/pacifica_silver_partitioned

# 3. Build an incremental regime delta for changed symbol/date partitions.
uv run python scripts/build_non_hft_regime_state.py \
  --silver-dir "$CANDIDATE_SILVER" \
  --out-dir "${CANDIDATE_REGIME}_delta" \
  --source-plan "$CANDIDATE_SILVER/incremental_plan.csv" \
  --bucket 1min

# 4. Build a full candidate regime snapshot for side-by-side verification.
#    This still writes only to the candidate output path.
uv run python scripts/build_non_hft_regime_state.py \
  --silver-dir "$CANDIDATE_SILVER" \
  --out-dir "$CANDIDATE_REGIME" \
  --bucket 1min

# 5. Verify candidate vs canonical before any promotion.
uv run python scripts/verify_pacifica_side_by_side_refresh.py \
  --canonical-silver-dir data/pacifica_silver_partitioned \
  --candidate-silver-dir "$CANDIDATE_SILVER" \
  --canonical-regime-dir docs/experiments/non-hft-regime-state \
  --candidate-regime-dir "$CANDIDATE_REGIME" \
  --out-dir "$VERIFY_DIR"
```

Only after the verification report is green and manually reviewed should `source_manifest_previous.csv` be advanced to the new manifest or canonical silver/regime paths be promoted. Candidate data under `data/` remains gitignored and must not be committed.

## Verification artifacts

The side-by-side verifier writes:

```text
summary.csv
silver_row_counts.csv
silver_coverage.csv
silver_duplicates_nulls.csv
regime_row_counts.csv
regime_coverage.csv
regime_duplicates_nulls.csv
report_diff.patch
README.md
```

Failure labels include:

```text
candidate_silver_row_count_regression
candidate_silver_coverage_regression
candidate_silver_key_nulls
candidate_silver_missing_key_columns
candidate_silver_duplicate_keys
candidate_regime_row_count_regression
candidate_regime_symbol_coverage_regression
candidate_regime_key_nulls
candidate_regime_missing_key_columns
candidate_regime_duplicate_keys
```

## TDD / smoke verification

Focused tests:

```text
uv run pytest \
  tests/scripts/test_build_pacifica_source_manifest.py \
  tests/scripts/test_build_pacifica_full_fidelity_silver.py \
  tests/scripts/test_build_non_hft_regime_state.py \
  tests/scripts/test_verify_pacifica_side_by_side_refresh.py -q
```

Latest local focused result after fail-closed hardening: `59 passed` for source-manifest, silver, regime-state, side-by-side verifier, and regime-governor tests.

Side-by-side smoke fixture result at 2026-05-13T16:17:04Z:

```text
processed_source_objects=1
planned_source_objects=1
delta_rows=1
verification_ok=True
verification_failures=[]
```

Read-only canonical self-check at 2026-05-13T18:11:33Z:

```text
uv run python scripts/verify_pacifica_side_by_side_refresh.py \
  --canonical-silver-dir data/pacifica_silver_partitioned \
  --candidate-silver-dir data/pacifica_silver_partitioned \
  --canonical-regime-dir docs/experiments/non-hft-regime-state \
  --candidate-regime-dir docs/experiments/non-hft-regime-state \
  --out-dir data/ops/pacifica-incremental-refresh-selfcheck-20260513T181133Z \
  --channels prices,trades,bbo,book,candle,mark_price_candle
ok=True
failures=[]
```

The self-check output is under gitignored `data/ops/`; it proves the verifier CLI runs against the current local canonical artifacts, not that a new candidate is safe to promote.

## Safety boundaries

- Canonical `data/pacifica_silver_partitioned/` is not overwritten by incremental mode unless explicitly used as `--out-dir`; do not do that before verification.
- Canonical `docs/experiments/non-hft-regime-state/regime_state.parquet` is not overwritten by `--source-plan`; incremental mode writes `regime_state_delta.parquet` instead.
- Raw archives and `.sha256` sidecars are read only.
- Generated candidate data and manifests under `data/` are gitignored and should not be committed.
- Report CSVs/README under `docs/ops/pacifica-incremental-refresh/` are small documentation artifacts and can be committed when intentionally refreshed.
