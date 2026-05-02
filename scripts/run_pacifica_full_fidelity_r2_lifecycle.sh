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
LIMIT_ARG=()
if [ -n "${PACIFICA_FULL_FIDELITY_BATCH_LIMIT:-}" ]; then
  LIMIT_ARG=(--limit "${PACIFICA_FULL_FIDELITY_BATCH_LIMIT}")
fi
MIN_UPLOAD_AGE_SECONDS="${PACIFICA_FULL_FIDELITY_MIN_UPLOAD_AGE_SECONDS:-7200}"

"${PYTHON_CMD[@]}" scripts/pacifica_full_fidelity_storage.py \
  --root "$ROOT" \
  --state-db "$STATE_DB" \
  --r2-prefix "$R2_PREFIX" \
  scan --skip-current-hour

"${PYTHON_CMD[@]}" scripts/pacifica_full_fidelity_storage.py \
  --state-db "$STATE_DB" \
  --remote-base "$REMOTE_BASE" \
  --min-upload-age-seconds "$MIN_UPLOAD_AGE_SECONDS" \
  "${LIMIT_ARG[@]}" \
  upload-verify

if [ "${PACIFICA_R2_PRUNE_EXECUTE:-0}" = "1" ]; then
  "${PYTHON_CMD[@]}" scripts/pacifica_full_fidelity_storage.py \
    --state-db "$STATE_DB" \
    prune --retention-days "$RETENTION_DAYS" --execute
else
  "${PYTHON_CMD[@]}" scripts/pacifica_full_fidelity_storage.py \
    --state-db "$STATE_DB" \
    prune --retention-days "$RETENTION_DAYS"
fi
