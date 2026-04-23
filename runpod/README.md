# Step 3 RunPod Execution

Self-supervised pretraining (MEM + SimCLR) on a single H100 pod.
Compute cap: 1 H100-day (23h default with 1h headroom for sync overhead).

## Image build (local)

Bake the current commit SHA so `export_checkpoint.py` can stamp provenance
without a live `.git/` directory inside the container:

```bash
docker build \
  --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -f runpod/Dockerfile \
  -t pacifica-step3:0.1 \
  .
```

## Push to RunPod registry

Use the `flash` skill to push the image to the RunPod container registry.
Reference: invoke the `flash` skill from Claude Code for the full command form.

## Launch a single H100 pod

Use the `runpodctl` skill from Claude Code.
Minimum required env vars: `R2_CACHE_PREFIX`, `OUT_PREFIX`.

Example (override annealed schedule endpoints for an ablation):

```
R2_CACHE_PREFIX=r2:pacifica-cache/v1
OUT_PREFIX=r2:pacifica-models/step3-seed0
EPOCHS=30
BATCH_SIZE=256
CHANNEL_MULT=1.0
MAX_HOURS=23.0
SEED=0
MEM_WEIGHT_START=0.90
MEM_WEIGHT_END=0.60
CONTRASTIVE_WEIGHT_START=0.10
CONTRASTIVE_WEIGHT_END=0.40
ANNEAL_EPOCHS=20
```

## Required env vars

| Var | Required | Default | Description |
|-----|----------|---------|-------------|
| `R2_CACHE_PREFIX` | yes | — | R2 path to .npz shard cache |
| `OUT_PREFIX` | yes | — | R2 destination for artifacts |
| `EPOCHS` | no | 30 | Training epochs |
| `BATCH_SIZE` | no | 256 | Per-device batch size |
| `CHANNEL_MULT` | no | 1.0 | Channel width multiplier |
| `MAX_HOURS` | no | 23.0 | Hard wall-clock cap (hardware-agnostic). `MAX_H100_HOURS` accepted as deprecated alias. |
| `SEED` | no | 0 | RNG seed |
| `MEM_WEIGHT_START` | no | (0.90) | MEM loss weight at epoch 1 |
| `MEM_WEIGHT_END` | no | (0.60) | MEM loss weight at anneal end |
| `CONTRASTIVE_WEIGHT_START` | no | (0.10) | Contrastive weight at epoch 1 |
| `CONTRASTIVE_WEIGHT_END` | no | (0.40) | Contrastive weight at anneal end |
| `ANNEAL_EPOCHS` | no | (20) | Epochs over which to anneal |

## Compute cap discipline

`MAX_HOURS` defaults to **23.0**, leaving 1h headroom under the spec's
1 H100-day cap (spec §Training). The remaining hour covers:
- rclone cache sync (~20 min for ~2GB)
- checkpoint export + probe report (~15 min)
- rclone artifact push (~5 min)

Never set `MAX_HOURS` above 23.0 without council sign-off.

## Artifact layout (pushed to OUT_PREFIX)

```
encoder.pt            raw training checkpoint (with epoch/loss history)
encoder-gate1.pt      stripped + provenance-stamped Gate 1 checkpoint
april-probe-report/   JSON + MD probe results (direction, symbol, hour-of-day)
training-log.jsonl    per-epoch loss/metric log
```
