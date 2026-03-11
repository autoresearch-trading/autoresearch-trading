"""Pre-build v3 feature cache for all symbols × splits (parallel)."""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare import DEFAULT_SYMBOLS, make_env

SPLITS = ["train", "val", "test"]


def build_one(args):
    sym, split = args
    import warnings

    warnings.filterwarnings("ignore")
    t0 = time.time()
    env = make_env(sym, split, window_size=50, trade_batch=100)
    return sym, split, env.num_steps, time.time() - t0


if __name__ == "__main__":
    tasks = [(sym, split) for sym in DEFAULT_SYMBOLS for split in SPLITS]
    total = len(tasks)
    cpus = min(os.cpu_count() or 4, 8)
    print(f"Building {total} cache files with {cpus} workers...")
    t_start = time.time()
    done = 0
    failed = []

    with ProcessPoolExecutor(max_workers=cpus) as ex:
        futs = {ex.submit(build_one, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            try:
                sym, split, steps, elapsed = fut.result()
                print(
                    f"  [{done}/{total}] {sym}/{split}: {steps} steps ({elapsed:.1f}s)"
                )
            except Exception as e:
                sym, split = futs[fut]
                failed.append(f"{sym}/{split}")
                print(f"  [{done}/{total}] {sym}/{split}: FAILED ({e})")

    print(f"\nDone: {done - len(failed)}/{total} OK, {len(failed)} failed")
    print(f"Total time: {time.time() - t_start:.1f}s")
    if failed:
        print(f"Failed: {failed}")
