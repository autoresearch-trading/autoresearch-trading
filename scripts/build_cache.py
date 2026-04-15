#!/usr/bin/env python
# scripts/build_cache.py
"""CLI: build per-symbol-day .npz shards for the tape pipeline.

Usage:
    uv run python scripts/build_cache.py \\
        --symbols BTC ETH SOL \\
        --start-date 2025-10-16 --end-date 2026-03-31 \\
        --out data/cache/v1/

    # Smoke-test on one symbol-day:
    uv run python scripts/build_cache.py \\
        --symbols AAVE \\
        --start-date 2025-10-16 --end-date 2025-10-16 \\
        --out /tmp/tape_cache_smoke/

Safety:
    - Hard-gated: end-date >= 2026-04-14 is rejected (April hold-out, gotcha #17).
    - Existing shards are skipped unless --force is set.
    - Per-shard failures are logged to stderr and do NOT abort the run.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

from tape.cache import build_symbol_day, save_shard
from tape.constants import APRIL_HELDOUT_START, PREAPRIL_END, PREAPRIL_START, SYMBOLS


def _daterange(start: str, end: str):
    """Yield ISO date strings from start to end (inclusive)."""
    s = _date.fromisoformat(start)
    e = _date.fromisoformat(end)
    while s <= e:
        yield s.isoformat()
        s += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build per-symbol-day .npz feature cache shards."
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=list(SYMBOLS),
        help="Symbols to process. Default: all 25 pretraining symbols.",
    )
    p.add_argument(
        "--start-date",
        default=PREAPRIL_START,
        help=f"Inclusive start date (ISO). Default: {PREAPRIL_START}",
    )
    p.add_argument(
        "--end-date",
        default=PREAPRIL_END,
        help=f"Inclusive end date (ISO). Default: {PREAPRIL_END}",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory for .npz shards.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing shards (default: skip).",
    )
    args = p.parse_args()

    # Safety: never touch the April hold-out (gotcha #17)
    if args.end_date >= APRIL_HELDOUT_START:
        print(
            f"ERROR: end-date {args.end_date!r} is in or past the held-out range "
            f"(>= {APRIL_HELDOUT_START}). Aborting.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out)
    total = 0
    built = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for sym in args.symbols:
        for d in _daterange(args.start_date, args.end_date):
            total += 1
            shard_path = out_dir / f"{sym}__{d}.npz"
            if shard_path.exists() and not args.force:
                skipped += 1
                continue
            try:
                shard = build_symbol_day(sym, d)
            except Exception as exc:
                print(f"[{sym} {d}] FAILED: {exc}", file=sys.stderr)
                failed += 1
                continue
            if shard is None:
                # Missing data or too few events — not an error
                continue
            save_shard(shard, out_dir)
            built += 1
            if built % 25 == 0:
                elapsed = time.time() - t0
                rate = built / elapsed if elapsed > 0 else 0.0
                print(
                    f"  built={built}  skipped={skipped}  failed={failed}"
                    f"  elapsed={elapsed:.1f}s  rate={rate:.1f} shards/s"
                )

    elapsed = time.time() - t0
    print(
        f"done. total={total}  built={built}  skipped={skipped}"
        f"  failed={failed}  elapsed={elapsed:.1f}s"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
