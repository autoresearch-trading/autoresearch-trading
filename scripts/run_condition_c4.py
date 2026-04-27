# scripts/run_condition_c4.py
"""Condition 4 — Embedding trajectory test (post-Gate-2 pre-registration).

Per the ratified pre-registration (commit `c28bc17`,
`docs/superpowers/specs/2026-04-26-post-gate2-pre-registration.md`):

    Condition 4 — Embedding trajectory test.
    - Manually identify ≥10 climax events on held-out Feb+Mar data using
      `climax_score > 3.0` as the seed criterion (then human-validate as
      actual phase transitions).
    - For each event window at time t, compute encoder distance
      ‖embed(t) − embed(t-50)‖ and compare to the within-symbol-day
      distance distribution.
    - PASS: ≥ 7/10 events show distance jump > 2σ above the within-symbol-day
      mean.

Mechanical operationalization
-----------------------------
The pre-reg says "manually identify" with "human-validate." Without a
human in the loop at this step, we operationalize the seed criterion
deterministically: rank all candidate windows by `max(climax_score) over
the LAST 50 events` and take the top-K windows that satisfy
`climax_seed_score > 3.0`. We require 10 distinct (symbol, date) pairs
to avoid sampling 10 windows from a single symbol-day's tightly packed
climax cluster — which would not be 10 INDEPENDENT events. This is the
spec's "≥10 climax events" interpretation that makes the test
falsifiable without manual curation.

Within-symbol-day distance distribution
---------------------------------------
For each (symbol, date) shard, we build a stride=50 dataset, embed every
window, and compute the consecutive-window distance series:

    d_i = ‖emb_i − emb_{i-1}‖    for i = 1, …, N_shard − 1

Each `d_i` corresponds to "embedding distance between window ending at
event t and window ending at event t − stride (=50)" — exactly the
‖embed(t) − embed(t-50)‖ quantity in the pre-reg.

The within-symbol-day reference distribution is built from ALL
consecutive-window distances within that shard.

Per-event test
--------------
For a candidate climax event at window position k inside its shard:
  - jump = d_k (distance between window k and window k-1)
  - reference mean μ, std σ from the shard's full distance series (excluding d_k)
  - pass_event = jump > μ + 2 * σ

Aggregate test: pass_count = sum(pass_event), C4 PASS iff pass_count >= 7
out of the K=10 selected events.

Underpowered guards (skipped, not failed):
  - shards with <20 stride=50 windows (insufficient distance distribution)
  - candidate windows at position 0 (no predecessor for d_k)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tape.constants import HELD_OUT_SYMBOL, PRETRAINING_SYMBOLS
from tape.dataset import TapeDataset
from tape.probe_utils import (
    forward_embeddings,
    load_encoder,
    pick_device,
    shards_for_sym_months,
)
from tape.wyckoff_labels import climax_seed_score

# Pre-registered constants (binding via commit c28bc17).
TEST_FEB_MAR: tuple[str, str] = ("2026-02", "2026-03")
CLIMAX_SEED_THRESHOLD: float = 3.0
N_EVENTS_REQUIRED: int = 10
PASS_COUNT_REQUIRED: int = 7
SIGMA_THRESHOLD: float = 2.0
STRIDE_C4: int = 50  # close to (t-50) granularity for the trajectory test
MIN_SHARD_WINDOWS: int = 20  # underpowered guard


def _candidate_climax_events_in_shard(
    enc,
    device: torch.device,
    shard_path: Path,
    batch_size: int,
) -> dict | None:
    """Embed all stride=50 windows in one shard, return candidates + distances.

    Returns dict:
      - emb            : (N, 256)
      - distances      : (N-1,) consecutive-window L2 distances
      - climax_scores  : (N,) per-window climax_seed_score
      - shard_path     : Path
      - sym, date      : str
      - n_windows      : int

    Returns None if the shard is underpowered (<MIN_SHARD_WINDOWS).
    """
    ds = TapeDataset([shard_path], stride=STRIDE_C4, mode="pretrain")
    if len(ds) < MIN_SHARD_WINDOWS:
        return None

    bundle = forward_embeddings(
        enc, ds, device, batch_size=batch_size, return_features=True
    )
    emb = bundle["emb"]
    feats = bundle["features"]
    sym = bundle["sym"][0] if bundle["sym"] else ""
    date = bundle["date"][0] if bundle["date"] else ""

    # Per-window climax seed score.
    climax_scores = np.fromiter(
        (climax_seed_score(feats[i]) for i in range(len(feats))),
        dtype=np.float32,
    )

    # Consecutive-window distances.
    distances = np.linalg.norm(emb[1:] - emb[:-1], axis=1).astype(np.float32)

    return {
        "emb": emb,
        "distances": distances,
        "climax_scores": climax_scores,
        "shard_path": shard_path,
        "sym": str(sym),
        "date": str(date),
        "n_windows": int(len(ds)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--n-events",
        type=int,
        default=N_EVENTS_REQUIRED,
        help="Number of distinct (symbol, date) climax events to select",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[c4] device={device}")
    enc = load_encoder(args.checkpoint, device)
    print(f"[c4] loaded encoder from {args.checkpoint}")
    print(
        f"[c4] climax_seed_threshold={CLIMAX_SEED_THRESHOLD} "
        f"sigma_threshold={SIGMA_THRESHOLD}"
    )
    print(
        f"[c4] N_EVENTS_REQUIRED={args.n_events} "
        f"PASS_COUNT_REQUIRED={PASS_COUNT_REQUIRED}/{args.n_events}"
    )

    target_symbols = [s for s in PRETRAINING_SYMBOLS if s != HELD_OUT_SYMBOL]

    # ---- 1. Discover all shards across 24 symbols × Feb+Mar. ----
    all_shards: list[Path] = []
    for sym in target_symbols:
        all_shards.extend(shards_for_sym_months(args.cache, sym, TEST_FEB_MAR))
    print(f"[c4] discovered {len(all_shards)} shards across Feb+Mar held-out")

    # ---- 2. Embed each shard at stride=50 + collect distances + climax scores. ----
    started = time.time()
    shard_records: list[dict] = []
    for i, shard in enumerate(all_shards, 1):
        rec = _candidate_climax_events_in_shard(
            enc=enc,
            device=device,
            shard_path=shard,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - started
        if rec is None:
            print(
                f"[c4] [{i:3d}/{len(all_shards)}] {shard.name}: "
                f"underpowered, skip elapsed={elapsed:.0f}s"
            )
            continue
        shard_records.append(rec)
        if i % 50 == 0:
            print(
                f"[c4] [{i:3d}/{len(all_shards)}] {shard.name}: "
                f"n={rec['n_windows']} max_climax={rec['climax_scores'].max():.2f} "
                f"elapsed={elapsed:.0f}s"
            )

    print(f"[c4] {len(shard_records)} usable shards (out of {len(all_shards)})")

    # ---- 3. Find best climax candidate per shard (top-1 per shard). ----
    candidates: list[dict] = []
    for rec in shard_records:
        cs = rec["climax_scores"]
        # Restrict to windows at position >= 1 (need predecessor for d_k).
        valid_positions = np.flatnonzero(cs > CLIMAX_SEED_THRESHOLD)
        valid_positions = valid_positions[valid_positions >= 1]
        if len(valid_positions) == 0:
            continue
        best_pos = int(valid_positions[int(np.argmax(cs[valid_positions]))])
        candidates.append(
            {
                "sym": rec["sym"],
                "date": rec["date"],
                "shard_path": rec["shard_path"],
                "position": best_pos,
                "climax_score": float(cs[best_pos]),
                "distances": rec["distances"],
                "n_windows": rec["n_windows"],
            }
        )

    print(
        f"[c4] {len(candidates)} (sym,date) shards have >=1 candidate "
        f"with climax_score > {CLIMAX_SEED_THRESHOLD}"
    )

    if len(candidates) < args.n_events:
        print(
            f"[c4] WARNING: only {len(candidates)} candidates available "
            f"(need {args.n_events}); proceeding with available"
        )

    # Sort by climax_score descending; take top-N.
    candidates.sort(key=lambda c: c["climax_score"], reverse=True)
    selected = candidates[: args.n_events]
    print(f"[c4] selected top-{len(selected)} climax events")

    # ---- 4. Test each selected event for >2σ jump. ----
    per_event: list[dict] = []
    for ev in selected:
        distances = ev["distances"]
        k = ev["position"]
        d_k = float(distances[k - 1])  # d_k = ‖emb_k - emb_{k-1}‖, stored at idx k-1

        # Reference distribution = all distances in shard EXCEPT d_k itself.
        ref = np.delete(distances, k - 1)
        ref_mean = float(np.mean(ref))
        ref_std = float(np.std(ref)) + 1e-12
        z = (d_k - ref_mean) / ref_std
        pass_event = bool(z > SIGMA_THRESHOLD)
        per_event.append(
            {
                "sym": ev["sym"],
                "date": ev["date"],
                "position": int(ev["position"]),
                "climax_score": float(ev["climax_score"]),
                "n_windows_in_shard": int(ev["n_windows"]),
                "distance_at_event": d_k,
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "z_score": float(z),
                "pass_event": pass_event,
            }
        )

    pass_count = sum(1 for ev in per_event if ev["pass_event"])
    c4_pass = pass_count >= PASS_COUNT_REQUIRED and len(selected) >= args.n_events

    summary = {
        "condition": "C4_embedding_trajectory_climax",
        "test_window": list(TEST_FEB_MAR),
        "stride": STRIDE_C4,
        "climax_seed_threshold": CLIMAX_SEED_THRESHOLD,
        "sigma_threshold": SIGMA_THRESHOLD,
        "n_events_required": args.n_events,
        "pass_count_required": PASS_COUNT_REQUIRED,
        "n_events_selected": len(selected),
        "pass_count": pass_count,
        "c4_pass": c4_pass,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "per_event": per_event,
    }

    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== CONDITION 4 VERDICT ===")
    print(
        f"  events_selected = {len(selected)}/{args.n_events} "
        f"(climax_score > {CLIMAX_SEED_THRESHOLD})"
    )
    print(
        f"  pass_count = {pass_count}/{len(selected)} "
        f"(threshold {PASS_COUNT_REQUIRED}/{args.n_events})"
    )
    for ev in per_event:
        flag = "PASS" if ev["pass_event"] else "fail"
        print(
            f"    [{flag}] {ev['sym']:8s} {ev['date']} "
            f"climax={ev['climax_score']:.2f} "
            f"d={ev['distance_at_event']:.3f} "
            f"(μ={ev['ref_mean']:.3f}, σ={ev['ref_std']:.3f}) "
            f"z={ev['z_score']:+.2f}"
        )
    print(f"  CONDITION 4 = {'PASS' if c4_pass else 'FAIL'}")
    print(f"  Report: {args.out}")
    return 0 if c4_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
