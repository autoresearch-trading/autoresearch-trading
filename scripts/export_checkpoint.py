# scripts/export_checkpoint.py
"""Strip optimizer state, stamp provenance, write Gate 1-ready checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from tape.model import EncoderConfig, TapeEncoder


def _git_sha(repo: Path = Path(".")) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    )


def _spec_sha() -> str:
    spec = Path(
        "docs/superpowers/specs/2026-04-10-tape-representation-learning-spec.md"
    )
    return hashlib.sha256(spec.read_bytes()).hexdigest()[:16]


def export_for_gate1(
    src: Path,
    dst: Path,
    *,
    git_sha: str | None = None,
    spec_sha: str | None = None,
) -> dict:
    payload = torch.load(src, map_location="cpu")
    enc_cfg = EncoderConfig(**payload["encoder_config"])
    enc = TapeEncoder(enc_cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    n_params = sum(p.numel() for p in enc.parameters())

    out = {
        "encoder_state_dict": payload["encoder_state_dict"],
        "encoder_config": payload["encoder_config"],
        "n_epochs_run": payload.get("n_epochs_run"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "seed": payload.get("seed"),
        "git_sha": git_sha or _git_sha(),
        "spec_sha": spec_sha or _spec_sha(),
        "n_params": n_params,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    info = export_for_gate1(args.src, args.dst)
    print(
        json.dumps({k: info[k] for k in ("n_params", "git_sha", "spec_sha")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
