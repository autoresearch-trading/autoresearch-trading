#!/usr/bin/env bash
# runpod/launch.sh — one-shot Step 3 pretraining sequence on an H100 pod.
#
# Required env vars (set by runpodctl / pod template):
#   R2_CACHE_PREFIX      e.g. r2:pacifica-cache/v1   (source of .npz shards)
#   OUT_PREFIX           e.g. r2:pacifica-models/step3 (artifact destination)
#
# Optional env vars (all have sane defaults):
#   EPOCHS                      default 30
#   BATCH_SIZE                  default 256
#   CHANNEL_MULT                default 1.0
#   MAX_HOURS                   default 23.0  (1h headroom under spec 24h cap)
#                               MAX_H100_HOURS accepted as deprecated alias
#   SEED                        default 0
#   MEM_WEIGHT_START            default unset → PretrainConfig default (0.90)
#   MEM_WEIGHT_END              default unset → PretrainConfig default (0.60)
#   CONTRASTIVE_WEIGHT_START    default unset → PretrainConfig default (0.10)
#   CONTRASTIVE_WEIGHT_END      default unset → PretrainConfig default (0.40)
#   ANNEAL_EPOCHS               default unset → PretrainConfig default (20)
#
# Exit codes: 0 = success, non-zero = failure (RunPod will surface the exit code).

set -euo pipefail

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-256}
CHANNEL_MULT=${CHANNEL_MULT:-1.0}
MAX_HOURS=${MAX_HOURS:-${MAX_H100_HOURS:-23.0}}
SEED=${SEED:-0}

mkdir -p /workspace/cache /workspace/runs

echo "[launch] === Step 3 pretraining start ==="
echo "[launch] EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  CHANNEL_MULT=$CHANNEL_MULT"
echo "[launch] MAX_HOURS=$MAX_HOURS  SEED=$SEED"
echo "[launch] R2_CACHE_PREFIX=${R2_CACHE_PREFIX:-<not set>}"
echo "[launch] OUT_PREFIX=${OUT_PREFIX:-<not set>}"

# 1. Pull cache shards from R2
echo "[launch] --- syncing cache from R2 ---"
rclone sync "${R2_CACHE_PREFIX}" /workspace/cache \
    --transfers 32 --checkers 64 --size-only \
    --progress

# 2. Build the annealed-weight flags (only forward if explicitly set)
EXTRA_FLAGS=()
if [[ -n "${MEM_WEIGHT_START:-}" ]]; then
    EXTRA_FLAGS+=("--mem-weight-start" "$MEM_WEIGHT_START")
fi
if [[ -n "${MEM_WEIGHT_END:-}" ]]; then
    EXTRA_FLAGS+=("--mem-weight-end" "$MEM_WEIGHT_END")
fi
if [[ -n "${CONTRASTIVE_WEIGHT_START:-}" ]]; then
    EXTRA_FLAGS+=("--contrastive-weight-start" "$CONTRASTIVE_WEIGHT_START")
fi
if [[ -n "${CONTRASTIVE_WEIGHT_END:-}" ]]; then
    EXTRA_FLAGS+=("--contrastive-weight-end" "$CONTRASTIVE_WEIGHT_END")
fi
if [[ -n "${ANNEAL_EPOCHS:-}" ]]; then
    EXTRA_FLAGS+=("--anneal-epochs" "$ANNEAL_EPOCHS")
fi

# 3. Run pretraining
echo "[launch] --- starting pretraining ---"
python scripts/run_pretrain.py \
    --cache /workspace/cache \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --channel-mult "$CHANNEL_MULT" \
    --out-dir /workspace/runs/run \
    --max-hours "$MAX_HOURS" \
    --seed "$SEED" \
    "${EXTRA_FLAGS[@]}"

# 4. Export Gate 1-ready checkpoint (strips optimizer state, stamps provenance)
echo "[launch] --- exporting checkpoint ---"
python scripts/export_checkpoint.py \
    --src /workspace/runs/run/encoder.pt \
    --dst /workspace/runs/run/encoder-gate1.pt

# 5. Run full April 1–13 probe report
echo "[launch] --- running probe report ---"
python scripts/run_pretrain_probes.py \
    --checkpoint /workspace/runs/run/encoder-gate1.pt \
    --cache /workspace/cache \
    --out /workspace/runs/run/april-probe-report

# 6. Push all artifacts back to R2
echo "[launch] --- pushing artifacts to R2 ---"
rclone copy /workspace/runs/run "${OUT_PREFIX}" \
    --transfers 16 \
    --progress

echo "[launch] === done ==="
