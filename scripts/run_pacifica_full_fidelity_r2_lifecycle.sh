#!/bin/bash
# Scan/upload/verify Pacifica full-fidelity raw chunks to R2.
#
# This script uses copy/upload semantics only. It never deletes remote objects.
# Local pruning is dry-run by default; set PACIFICA_R2_PRUNE_EXECUTE=1 only
# after verifying the lifecycle state and retention policy.

set -euo pipefail

PYTHON_CMD=(uv run python)
if [ "${PACIFICA_USE_SYSTEM_PYTHON:-0}" = "1" ]; then
  PYTHON_CMD=(python)
fi

ROOT="${PACIFICA_FULL_FIDELITY_ROOT:-data/pacifica_full_fidelity}"
STATE_DB="${PACIFICA_FULL_FIDELITY_STATE_DB:-data/pacifica_full_fidelity_storage.sqlite}"
R2_PREFIX="${PACIFICA_FULL_FIDELITY_R2_PREFIX:-raw/pacifica/full_fidelity}"
REMOTE_BASE="${PACIFICA_FULL_FIDELITY_REMOTE_BASE:-r2:pacifica-trading-data}"
RETENTION_DAYS="${PACIFICA_FULL_FIDELITY_RETENTION_DAYS:-3}"
MIN_UPLOAD_AGE_SECONDS="${PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS:-7200}"
UPLOAD_ORDER="${PACIFICA_FULL_FIDELITY_UPLOAD_ORDER:-newest-first}"
LIFECYCLE_STATE_DIR="${PACIFICA_FULL_FIDELITY_LIFECYCLE_STATE_DIR:-$(dirname "$STATE_DB")/.lifecycle}"

# Fast lane: discover only recently closed partitions and upload newest eligible chunks.
RECENT_SCAN_HOURS="${PACIFICA_FULL_FIDELITY_RECENT_SCAN_HOURS:-12}"
FRESH_UPLOAD_LIMIT="${PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_LIMIT:-${PACIFICA_FULL_FIDELITY_UPLOAD_LIMIT:-${PACIFICA_FULL_FIDELITY_BATCH_LIMIT:-2000}}}"
FRESH_UPLOAD_TRANSFERS="${PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_TRANSFERS:-16}"
FRESH_UPLOAD_CHECKERS="${PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_CHECKERS:-32}"
FRESH_UPLOAD_SIDECAR_WORK_DIR="${PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_SIDECAR_WORK_DIR:-$LIFECYCLE_STATE_DIR/fresh-upload-sidecars}"

# Safety lane: slower verification/pruning and bounded historical backlog progress.
BACKLOG_UPLOAD_LIMIT="${PACIFICA_FULL_FIDELITY_BACKLOG_UPLOAD_LIMIT:-250}"
VERIFY_LIMIT="${PACIFICA_FULL_FIDELITY_VERIFY_LIMIT:-${PACIFICA_FULL_FIDELITY_BATCH_LIMIT:-500}}"
FULL_SCAN_INTERVAL_S="${PACIFICA_FULL_FIDELITY_FULL_SCAN_INTERVAL_S:-21600}"
FULL_SCAN_MARKER="$LIFECYCLE_STATE_DIR/full_scan_last_run_epoch.txt"

run_storage() {
  "${PYTHON_CMD[@]}" scripts/pacifica_full_fidelity_storage.py "$@"
}

positive_int() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    0) return 1 ;;
    *) return 0 ;;
  esac
}

full_scan_due() {
  if ! positive_int "$FULL_SCAN_INTERVAL_S"; then
    return 1
  fi
  if [ ! -s "$FULL_SCAN_MARKER" ]; then
    return 0
  fi
  local now last elapsed
  now="$(date +%s)"
  last="$(cat "$FULL_SCAN_MARKER" 2>/dev/null || echo 0)"
  case "$last" in
    ''|*[!0-9]*) return 0 ;;
  esac
  elapsed=$((now - last))
  [ "$elapsed" -ge "$FULL_SCAN_INTERVAL_S" ]
}

mark_full_scan_run() {
  mkdir -p "$LIFECYCLE_STATE_DIR"
  date +%s > "$FULL_SCAN_MARKER"
}

if positive_int "$RECENT_SCAN_HOURS"; then
  run_storage \
    --root "$ROOT" \
    --state-db "$STATE_DB" \
    --r2-prefix "$R2_PREFIX" \
    scan --skip-current-hour --recent-hours "$RECENT_SCAN_HOURS"
else
  echo "{\"recent_scan_skipped\":true,\"reason\":\"PACIFICA_FULL_FIDELITY_RECENT_SCAN_HOURS<=0\"}"
fi

if positive_int "$FRESH_UPLOAD_LIMIT"; then
  run_storage \
    --root "$ROOT" \
    --state-db "$STATE_DB" \
    --r2-prefix "$R2_PREFIX" \
    --remote-base "$REMOTE_BASE" \
    --min-upload-age-seconds "$MIN_UPLOAD_AGE_SECONDS" \
    --upload-order "$UPLOAD_ORDER" \
    --upload-limit "$FRESH_UPLOAD_LIMIT" \
    --transfers "$FRESH_UPLOAD_TRANSFERS" \
    --checkers "$FRESH_UPLOAD_CHECKERS" \
    --sidecar-work-dir "$FRESH_UPLOAD_SIDECAR_WORK_DIR" \
    upload-batch
else
  echo "{\"fresh_upload_skipped\":true,\"reason\":\"PACIFICA_FULL_FIDELITY_FRESH_UPLOAD_LIMIT<=0\"}"
fi

if full_scan_due; then
  run_storage \
    --root "$ROOT" \
    --state-db "$STATE_DB" \
    --r2-prefix "$R2_PREFIX" \
    scan --skip-current-hour
  mark_full_scan_run
else
  echo "{\"full_scan_skipped\":true,\"interval_s\":$FULL_SCAN_INTERVAL_S}"
fi

run_storage \
  --state-db "$STATE_DB" \
  --min-upload-age-seconds "$MIN_UPLOAD_AGE_SECONDS" \
  reset-mismatch-errors --execute

run_storage \
  --state-db "$STATE_DB" \
  --remote-base "$REMOTE_BASE" \
  --min-upload-age-seconds "$MIN_UPLOAD_AGE_SECONDS" \
  --upload-order "oldest-first" \
  --upload-limit "$BACKLOG_UPLOAD_LIMIT" \
  --verify-limit "$VERIFY_LIMIT" \
  upload-verify

if [ "${PACIFICA_R2_PRUNE_EXECUTE:-0}" = "1" ]; then
  run_storage \
    --state-db "$STATE_DB" \
    prune --retention-days "$RETENTION_DAYS" --execute
else
  run_storage \
    --state-db "$STATE_DB" \
    prune --retention-days "$RETENTION_DAYS"
fi
